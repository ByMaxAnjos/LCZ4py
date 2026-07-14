"""Diagnóstico standalone para o erro de abertura /vsicurl/ do Zenodo.

Rode com:  python diagnose_zenodo.py
Cole a saída completa para eu analisar.
"""
import os
import sys
import time

URL = "https://zenodo.org/records/8419340/files/lcz_filter_v3.tif?download=1"

print("=== 1. Versões ===")
import rasterio
print("rasterio:", rasterio.__version__)
print("GDAL:", rasterio.gdal_version())
print("python:", sys.version)

print("\n=== 2. Alcance de rede simples (requests) ===")
try:
    import requests
    t0 = time.time()
    r = requests.head(URL, allow_redirects=True, timeout=20)
    print("status:", r.status_code, " tempo:", round(time.time() - t0, 2), "s")
    for h in ("x-ratelimit-remaining", "retry-after", "content-length", "accept-ranges"):
        print(f"  {h}: {r.headers.get(h)}")
except Exception as e:
    print("FALHOU:", type(e).__name__, e)

print("\n=== 3. Range request real (isso é o que o GDAL faz) ===")
try:
    r = requests.get(URL, headers={"Range": "bytes=0-1023"}, timeout=20)
    print("status:", r.status_code, " bytes recebidos:", len(r.content))
except Exception as e:
    print("FALHOU:", type(e).__name__, e)

print("\n=== 4. Abertura via GDAL /vsicurl/ com log verboso ===")
os.environ["CPL_CURL_VERBOSE"] = "YES"
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "300")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "YES")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.ovr,.zip")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")

vsi_path = f"/vsicurl/{URL}"
t0 = time.time()
try:
    with rasterio.open(vsi_path) as src:
        print("ABRIU OK:", src.profile)
except Exception as e:
    print("FALHOU:", type(e).__name__, e)
print("tempo total:", round(time.time() - t0, 2), "s")

print("\n=== 5. Variáveis de ambiente de proxy (se houver, podem ser a causa) ===")
for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY"):
    if k in os.environ:
        print(f"  {k} = {os.environ[k]}")
if not any(k in os.environ for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")):
    print("  (nenhuma configurada)")
