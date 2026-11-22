"""Shared download/parallelism helpers for the raster ``lcz_grid_*`` family
(crop+mask onto an LCZ map's grid — see ``_lcz_grid_raster_base.py`` for the
rest of that shared machinery).

Everything here is stdlib + ``httpx`` (a core dependency). Nothing in this
module imports an optional package at import time, so importing it never
fails because ``cdsapi``/``contextily``/``dask``/… is missing.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

#: Bounded default for every thread pool in this package (I/O-bound work).
DEFAULT_MAX_WORKERS = min(8, os.cpu_count() or 4)

#: Streaming chunk size for downloads.
CHUNK_SIZE = 1 << 20

DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF = 1.5

#: HTTP statuses worth retrying (the rest of 4xx are permanent failures).
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


# ── Parallelism helpers ───────────────────────────────────────────────────────

def default_max_workers() -> int:
    """Bounded default worker count: ``min(8, os.cpu_count() or 4)``."""
    return DEFAULT_MAX_WORKERS


def resolve_max_workers(max_workers: Optional[int], n_items: int) -> int:
    """Clamp a user-supplied ``max_workers`` to something usable.

    Returns 1 (i.e. run inline) for empty/single-item workloads so tiny
    requests don't pay thread-pool startup costs.
    """
    if n_items <= 1:
        return 1
    if max_workers is None:
        return max(1, min(DEFAULT_MAX_WORKERS, n_items))
    try:
        workers = int(max_workers)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"max_workers must be an integer or None, got {max_workers!r}") from exc
    if workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {workers}")
    return min(workers, n_items)


def parallel_ordered(
    func: Callable[[Any], Any],
    items: Sequence[Any],
    max_workers: Optional[int] = None,
    default: Any = None,
    label: str = "task",
) -> list[Any]:
    """Apply ``func`` to every item in parallel, preserving input order.

    Failures are logged and replaced by ``default`` instead of aborting the
    whole batch (a single bad month should not kill a 20-year request). The
    caller is responsible for turning a run of ``default`` values into a
    clear "no data" error.
    """
    items = list(items)
    if not items:
        return []

    workers = resolve_max_workers(max_workers, len(items))
    results: list[Any] = [default] * len(items)

    if workers <= 1:
        for i, item in enumerate(items):
            try:
                results[i] = func(item)
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                logger.warning("%s failed for %r: %s", label, item, exc)
        return results

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=str(label)) as pool:
        futures = {pool.submit(func, item): i for i, item in enumerate(items)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                logger.warning("%s failed for %r: %s", label, items[idx], exc)

    return results


# ── Download helpers ──────────────────────────────────────────────────────────

def _is_retryable(exc: BaseException) -> bool:
    """True when retrying the same request may plausibly succeed."""
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx is a core dependency
        return False

    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status in _RETRYABLE_STATUS
    # Connection resets, timeouts, DNS hiccups, truncated bodies.
    return isinstance(exc, (httpx.TransportError, httpx.TimeoutException, OSError))


def _stream_to_file(
    client: Any,
    url: str,
    tmp_path: str,
    *,
    headers: Optional[Mapping[str, str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> int:
    """Stream ``url`` into ``tmp_path``; returns bytes written.

    Raises on HTTP errors, truncated bodies and short reads so the caller can
    retry. The destination is only ever a ``.part`` file, so a crash leaves
    the real cache path untouched.
    """
    written = 0
    with client.stream("GET", url, headers=dict(headers or {}), timeout=timeout) as resp:
        resp.raise_for_status()
        with open(tmp_path, "wb") as fh:
            for chunk in resp.iter_bytes(chunk_size=CHUNK_SIZE):
                fh.write(chunk)
                written += len(chunk)

    content_length = resp.headers.get("content-length")
    # Skip the check when the body was content-encoded (httpx decodes it, so
    # content-length would describe the compressed payload).
    if content_length and not resp.headers.get("content-encoding"):
        try:
            expected = int(content_length)
        except ValueError:
            expected = 0
        if expected > 0 and written != expected:
            raise OSError(
                f"truncated download for {url}: {written} of {expected} bytes"
            )
    return written


def download_file(
    url: str,
    cache_path: str,
    use_cache: bool = True,
    verbose: bool = True,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
    headers: Optional[Mapping[str, str]] = None,
    auth: Any = None,
    client: Any = None,
    min_bytes: int = 1,
) -> bool:
    """Download ``url`` to ``cache_path`` if not already cached.

    Atomic (writes ``cache_path.part`` then renames), retries transient
    failures with exponential backoff, and treats an existing non-empty file
    as a cache hit so offline reruns keep working.

    Parameters
    ----------
    url : str
        Source URL.
    cache_path : str
        Destination file. Parent directories are created on demand.
    use_cache : bool
        Reuse ``cache_path`` when it already exists and is non-empty.
    verbose : bool
        Log cache hits and completed downloads.
    timeout : float
        Per-request timeout in seconds.
    max_retries : int
        Total attempts (>= 1).
    backoff : float
        Base delay in seconds between attempts (doubles each retry).
    headers : mapping, optional
        Extra request headers.
    auth : httpx.Auth or tuple, optional
        Per-request authentication (e.g. ``(user, password)``).
    client : httpx.Client, optional
        Shared client (safe to reuse across threads). One is created and
        closed internally when omitted.
    min_bytes : int
        Minimum size for an existing file to count as cached.

    Returns
    -------
    bool
        True on success (or cache hit), False when every attempt failed.
    """
    cache_path = os.path.expanduser(str(cache_path))

    if use_cache and os.path.exists(cache_path) and os.path.getsize(cache_path) >= max(1, min_bytes):
        if verbose:
            logger.info("Cache hit: %s", os.path.basename(cache_path))
        return True

    parent = os.path.dirname(os.path.abspath(cache_path))
    os.makedirs(parent, exist_ok=True)
    tmp_path = cache_path + ".part"
    attempts = max(1, int(max_retries))

    own_client = client is None
    if own_client:
        import httpx

        client = httpx.Client(
            follow_redirects=True,
            timeout=timeout,
            auth=auth,
            headers=dict(headers or {}),
            trust_env=True,
        )

    try:
        for attempt in range(1, attempts + 1):
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                _stream_to_file(
                    client,
                    url,
                    tmp_path,
                    # headers/auth already live on an owned client; pass them
                    # again for externally supplied clients.
                    headers=headers if not own_client else None,
                    timeout=timeout,
                )
                os.replace(tmp_path, cache_path)
                if verbose:
                    logger.info("Downloaded: %s", os.path.basename(cache_path))
                return True
            except Exception as exc:  # noqa: BLE001 - classified below
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:  # pragma: no cover - best effort
                        pass
                retryable = _is_retryable(exc)
                if not retryable or attempt >= attempts:
                    logger.warning(
                        "Failed to download %s (attempt %d/%d): %s",
                        url, attempt, attempts, exc,
                    )
                    return False
                delay = backoff * (2 ** (attempt - 1))
                logger.debug(
                    "Retrying %s in %.1fs after: %s", url, delay, exc,
                )
                time.sleep(delay)
        return False  # pragma: no cover - loop always returns
    finally:
        if own_client:
            try:
                client.close()
            except Exception:  # pragma: no cover - best effort
                pass


def download_many(
    entries: Mapping[str, tuple[str, str]],
    *,
    use_cache: bool = True,
    verbose: bool = False,
    max_workers: Optional[int] = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
    headers: Optional[Mapping[str, str]] = None,
    auth: Any = None,
    client: Any = None,
    progress_desc: Optional[str] = None,
) -> dict[str, bool]:
    """Download many ``{key: (url, cache_path)}`` pairs in parallel.

    Returns
    -------
    dict[str, bool]
        ``key -> success``, in the same order as ``entries``, so callers can
        iterate deterministically regardless of completion order.
    """
    items = list(entries.items())
    if not items:
        return {}

    workers = resolve_max_workers(max_workers, len(items))
    results: dict[str, bool] = {}
    if workers <= 1:
        for key, (url, path) in items:
            results[key] = download_file(
                url, path, use_cache, verbose,
                timeout=timeout, max_retries=max_retries, backoff=backoff,
                headers=headers, auth=auth, client=client,
            )
        return results

    own_client = client is None
    if own_client:
        import httpx

        client = httpx.Client(
            follow_redirects=True,
            timeout=timeout,
            auth=auth,
            headers=dict(headers or {}),
            trust_env=True,
        )

    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lcz-dl")
    try:
        futures = {}
        for key, (url, path) in items:
            futures[pool.submit(
                download_file,
                url, path, use_cache, False,
                timeout=timeout, max_retries=max_retries, backoff=backoff,
                headers=headers if not own_client else None,
                client=client,
            )] = key

        iterator = as_completed(futures)
        if verbose and progress_desc and len(items) > 1:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, total=len(futures), desc=progress_desc, leave=False)
            except Exception:  # pragma: no cover - tqdm is optional at runtime
                pass

        for future in iterator:
            key = futures[future]
            try:
                results[key] = bool(future.result())
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                logger.warning("Download failed for %s: %s", key, exc)
                results[key] = False
    finally:
        pool.shutdown(wait=True)
        if own_client:
            try:
                client.close()
            except Exception:  # pragma: no cover - best effort
                pass

    # Restore input ordering (dict insertion order drives caching/writing).
    ordered = {key: results.get(key, False) for key, _ in items}
    if verbose:
        ok = sum(1 for v in ordered.values() if v)
        if ok != len(ordered):
            logger.warning(
                "%d of %d download(s) unavailable; continuing with the rest.",
                len(ordered) - ok, len(ordered),
            )
    return ordered


def normalize_int_list(
    value: Any,
    name: str,
    default: Optional[Iterable[int]] = None,
) -> list[int]:
    """Normalize ``years``/``months``-style arguments to a list of ints.

    Accepts ``None`` (→ ``default``), a bare int, a string digit, or any
    iterable of int-like values. Raises ``TypeError`` with the offending
    argument named, instead of a cryptic ``'int' object is not iterable``.
    """
    if value is None:
        return [int(v) for v in default] if default is not None else []
    if isinstance(value, (str, bytes)):
        values: list[Any] = [value]
    elif hasattr(value, "__iter__"):
        values = list(value)
    else:
        values = [value]
    try:
        return [int(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be an integer or an iterable of integers, got {value!r}"
        ) from exc

