"""
LCZ4r multilingual message system — Python equivalent of the R ``.lcz_messages``
helper.

Usage
-----
>>> from LCZ4py._internal.i18n_messages import lcz_msg
>>> print(lcz_msg("no_map_input", "en"))
>>> print(lcz_msg("city_not_found", "zh", city="Berlin"))
>>> print(lcz_msg("save_output_path", "pt", path="/tmp/lcz.tif"))

Adding a new key
----------------
1. Add an entry to ``LCZ_MESSAGES`` with ``en``, ``pt``, ``es`` and ``zh`` strings
2. Call ``lcz_msg("your_key", lang, param=value)`` in the function
3. Use ``{varname}`` placeholders; rich markup like ``{val}``, ``{arg}``,
   ``{fn}``, ``{path}``, ``{url}`` is preserved untouched so the caller
   can pass the result through its preferred rich-print helper.
"""

from __future__ import annotations

import re
from typing import Any

# ── Message lookup table ──────────────────────────────────────────────────────
#
# Each value is a dict ``{"en": str|list[str], "pt": ..., ...}``.
# A list-shaped value mirrors R's ``cli_abort(c("msg", "i" = "hint"))`` form;
# ``lcz_msg`` returns only the first line for plain-text callers.

LCZ_MESSAGES: dict[str, dict[str, Any]] = {

    # ── Input validation ─────────────────────────────────────────────────────

    "no_map_input": {
        "en": "Provide either {arg city} (a place name) or {arg roi} (a GeoDataFrame polygon).",
        "pt": "Forneça {arg city} (nome de local) ou {arg roi} (polígono GeoDataFrame).",
        "es": "Proporcione {arg city} (nombre del lugar) o {arg roi} (polígono GeoDataFrame).",
        "zh": "请提供 {arg city}（地名）或 {arg roi}（GeoDataFrame 多边形）。",
    },
    "city_not_found": {
        "en": "No polygon boundary found for {val {city}}. Check spelling at {url https://nominatim.openstreetmap.org}.",
        "pt": "Nenhum limite poligonal encontrado para {val {city}}. Verifique em {url https://nominatim.openstreetmap.org}.",
        "es": "No se encontró límite poligonal para {val {city}}. Compruebe en {url https://nominatim.openstreetmap.org}.",
        "zh": "未找到 {val {city}} 的多边形边界，请在 {url https://nominatim.openstreetmap.org} 检查拼写。",
    },
    "large_raster_error": {
        "en": [
            "Raster crop returned None — region may be outside the map extent.",
            "i=Crop a smaller area first, or use lcz_get_map2 with a pre-clipped file.",
        ],
        "pt": [
            "O recorte do raster retornou None — a região pode estar fora da extensão do mapa.",
            "i=Recorte uma área menor primeiro, ou use lcz_get_map2 com arquivo pré-recortado.",
        ],
        "es": [
            "El recorte del raster devolvió None — la región puede estar fuera de la extensión del mapa.",
            "i=Recorte primero un área más pequeña, o use lcz_get_map2 con archivo pre-recortado.",
        ],
        "zh": [
            "栅格裁剪返回 None，该区域可能超出地图范围。",
            "i=请先裁剪较小的区域，或使用 lcz_get_map2 加载预裁剪文件。",
        ],
    },
    "map_download_failed": {
        "en": [
            "Failed to download the LCZ global map from Zenodo.",
            "i=Download manually from {url {url}} then load with lcz_get_map2.",
        ],
        "pt": [
            "Falha ao baixar o mapa global LCZ do Zenodo.",
            "i=Baixe manualmente de {url {url}} e use lcz_get_map2.",
        ],
        "es": [
            "Error al descargar el mapa global LCZ desde Zenodo.",
            "i=Descargue manualmente de {url {url}} y use lcz_get_map2.",
        ],
        "zh": [
            "从 Zenodo 下载全球 LCZ 地图失败。",
            "i=请从 {url {url}} 手动下载，然后使用 lcz_get_map2 加载。",
        ],
    },
    "invalid_raster": {
        "en": "{arg {arg}} must be a SpatRaster. Use lcz_get_map.",
        "pt": "{arg {arg}} deve ser um objeto SpatRaster. Use lcz_get_map.",
        "es": "{arg {arg}} debe ser un objeto SpatRaster. Use lcz_get_map.",
        "zh": "{arg {arg}} 必须是 SpatRaster 对象，请使用 lcz_get_map。",
    },

    # ── Download / cache progress ─────────────────────────────────────────────

    "map_connecting": {
        "en": "Connecting to Zenodo LCZ global map (COG streaming — only your region will be fetched)...",
        "pt": "Conectando ao mapa global LCZ do Zenodo (streaming COG — apenas a sua região será baixada)...",
        "es": "Conectando al mapa global LCZ de Zenodo (streaming COG — solo se descargará su región)...",
        "zh": "正在连接 Zenodo 全球 LCZ 地图（COG 流式传输——仅获取您的区域）...",
    },
    "clipped_cache_hit": {
        "en": "Clipped LCZ map loaded from cache ({path {path}}).",
        "pt": "Mapa LCZ recortado carregado do cache ({path {path}}).",
        "es": "Mapa LCZ recortado cargado desde la caché ({path {path}}).",
        "zh": "已从缓存加载裁剪后的 LCZ 地图（{path {path}}）。",
    },
    "clipped_cached": {
        "en": "Clipped LCZ map saved to cache ({path {path}}).",
        "pt": "Mapa LCZ recortado salvo no cache ({path {path}}).",
        "es": "Mapa LCZ recortado guardado en caché ({path {path}}).",
        "zh": "裁剪后的 LCZ 地图已保存至缓存（{path {path}}）。",
    },
    "geocoding": {
        "en": "Querying OpenStreetMap/Nominatim for {val {city}}...",
        "pt": "Consultando OpenStreetMap/Nominatim para {val {city}}...",
        "es": "Consultando OpenStreetMap/Nominatim para {val {city}}...",
        "zh": "正在向 OpenStreetMap/Nominatim 查询 {val {city}}...",
    },
    "geocode_cache_hit": {
        "en": "Study area for {val {city}} loaded from cache ({path {path}}).",
        "pt": "Área de estudo de {val {city}} carregada do cache ({path {path}}).",
        "es": "Área de estudio de {val {city}} cargada desde la caché ({path {path}}).",
        "zh": "已从缓存加载 {val {city}} 的研究区域（{path {path}}）。",
    },
    "geocode_cached": {
        "en": "Study area boundary cached to {path {path}}.",
        "pt": "Limite da área de estudo armazenado em {path {path}}.",
        "es": "Límite del área de estudio almacenado en {path {path}}.",
        "zh": "研究区域边界已缓存至 {path {path}}。",
    },

    # ── Crop / mask progress ──────────────────────────────────────────────────

    "large_area_simplify": {
        "en": "Large area ({area} km²): simplifying boundary for faster masking.",
        "pt": "Área grande ({area} km²): simplificando contorno para mask mais rápido.",
        "es": "Área grande ({area} km²): simplificando contorno para enmascaramiento más rápido.",
        "zh": "检测到大面积区域（{area} km²），正在简化边界以加速掩膜处理。",
    },
    "large_area_tiling": {
        "en": "Very large area ({area} M km²): switching to tiled parallel processing.",
        "pt": "Área muito grande ({area} M km²): usando processamento paralelo por quadrantes.",
        "es": "Área muy grande ({area} M km²): usando procesamiento paralelo por cuadrantes.",
        "zh": "超大面积区域（{area} 百万 km²），切换为并行分块处理模式。",
    },
    "tiling_start": {
        "en": "Processing {n} tile(s) on {cores} core(s)...",
        "pt": "Processando {n} quadrante(s) em {cores} núcleo(s)...",
        "es": "Procesando {n} cuadrante(s) en {cores} núcleo(s)...",
        "zh": "正在使用 {cores} 个核心处理 {n} 个分块...",
    },
    "tiling_mosaic": {
        "en": "Merging {n} non-empty tile(s) into final map...",
        "pt": "Mesclando {n} quadrante(s) não-vazios no mapa final...",
        "es": "Fusionando {n} cuadrante(s) no vacíos en el mapa final...",
        "zh": "正在将 {n} 个非空分块合并为最终地图...",
    },

    # ── Save / done ───────────────────────────────────────────────────────────

    "save_output_path": {
        "en": "Output saved to {path {path}}",
        "pt": "Saída salva em {path {path}}",
        "es": "Salida guardada en {path {path}}",
        "zh": "输出已保存至 {path {path}}",
    },
    "analysis_done": {
        "en": "Done. {val {n}} rows processed.",
        "pt": "Concluído. {val {n}} linhas processadas.",
        "es": "Listo. {val {n}} filas procesadas.",
        "zh": "完成，共处理 {val {n}} 行。",
    },

    # ── UI labels (map/chart titles, legend titles, axis labels) ──────────────

    "lcz_map_title": {
        "en": "Local Climate Zone Map",
        "pt": "Mapa de Zonas Climáticas Locais",
        "es": "Mapa de Zonas Climáticas Locales",
        "zh": "局部气候区地图",
    },
    "lcz_class_legend": {
        "en": "LCZ Class",
        "pt": "Classe LCZ",
        "es": "Clase SCL",
        "zh": "LCZ 类型",
    },
    "lcz_area_title": {
        "en": "LCZ Area Distribution",
        "pt": "Distribuição de Área LCZ",
        "es": "Distribución de Área SCL",
        "zh": "LCZ 面积分布",
    },
    "lcz_code_label": {
        "en": "LCZ code",
        "pt": "Código LCZ",
        "es": "Código SCL",
        "zh": "LCZ 代码",
    },
    "area_km2_label": {
        "en": "Area [km²]",
        "pt": "Área [km²]",
        "es": "Área [km²]",
        "zh": "面积 [km²]",
    },

    # ── Time-series / anomaly / UHI / interpolation labels ───────────────────

    "lcz_ts_title": {
        "en": "LCZ Time Series",
        "pt": "Séries Temporais LCZ",
        "es": "Series Temporales SCL",
        "zh": "LCZ 时间序列",
    },
    "lcz_anomaly_title": {
        "en": "LCZ Temperature Anomaly",
        "pt": "Anomalia de Temperatura LCZ",
        "es": "Anomalía de Temperatura SCL",
        "zh": "LCZ 温度异常",
    },
    "lcz_uhi_title": {
        "en": "Urban Heat Island Intensity",
        "pt": "Intensidade da Ilha de Calor Urbana",
        "es": "Intensidad de la Isla de Calor Urbana",
        "zh": "城市热岛强度",
    },
    "lcz_interp_map_title": {
        "en": "LCZ Interpolated Map",
        "pt": "Mapa Interpolado LCZ",
        "es": "Mapa Interpolado SCL",
        "zh": "LCZ 插值地图",
    },
    "lcz_interp_eval_title": {
        "en": "Interpolation Cross-Validation",
        "pt": "Validação Cruzada da Interpolação",
        "es": "Validación Cruzada de Interpolación",
        "zh": "插值交叉验证",
    },
    "anomaly_label": {
        "en": "Anomaly [°C]",
        "pt": "Anomalia [°C]",
        "es": "Anomalía [°C]",
        "zh": "异常值 [°C]",
    },
    "uhi_label": {
        "en": "UHI [°C]",
        "pt": "ICU [°C]",
        "es": "ICU [°C]",
        "zh": "城市热岛 [°C]",
    },
    "temperature_label": {
        "en": "Air temperature [°C]",
        "pt": "Temperatura do ar [°C]",
        "es": "Temperatura del aire [°C]",
        "zh": "气温 [°C]",
    },
    "time_label": {
        "en": "Time",
        "pt": "Tempo",
        "es": "Tiempo",
        "zh": "时间",
    },
    "no_urban_rural": {
        "en": [
            "Could not find urban (LCZ 1–10) and rural (LCZ 11–16) stations.",
            "i=Check your data or supply station lists via T_urban/T_rural.",
        ],
        "pt": [
            "Estações urbanas (LCZ 1–10) e rurais (LCZ 11–16) não encontradas.",
            "i=Verifique os dados ou use T_urban/T_rural.",
        ],
        "es": [
            "No se encontraron estaciones urbanas (SCL 1–10) y rurales (SCL 11–16).",
            "i=Revise los datos o use T_urban/T_rural.",
        ],
        "zh": [
            "未找到城市（LCZ 1–10）和乡村（LCZ 11–16）站点。",
            "i=请检查数据或使用 T_urban/T_rural 指定站点。",
        ],
    },
    "lcz_ignored_non_krige": {
        "en": "LCZ classes available but method={arg {method}} ignores them — use method='krige' for External Drift Kriging.",
        "pt": "Classes LCZ disponíveis, mas method={arg {method}} as ignora — use method='krige' para Krigagem com Deriva Externa.",
        "es": "Clases LCZ disponibles, pero method={arg {method}} las ignora — use method='krige' para Kriging con Deriva Externa.",
        "zh": "存在 LCZ 类别，但 method={arg {method}} 未使用它们 — 请使用 method='krige' 以启用外部漂移克里金法。",
    },

    # ── ML interpolation (lcz_interp_map_plus) ──────────────────────────────────

    "interp_plus_download": {
        "en": "Downloading urban parameters ({n} workers)...",
        "pt": "Baixando parâmetros urbanos ({n} workers)...",
        "es": "Descargando parámetros urbanos ({n} workers)...",
        "zh": "正在下载城市参数（{n} 个工作进程）...",
    },
    "interp_plus_no_data": {
        "en": [
            "No urban parameters available — cannot train ML model.",
            "i=Check network connection or specify ucp_variables manually.",
        ],
        "pt": [
            "Nenhum parâmetro urbano disponível — impossível treinar modelo ML.",
            "i=Verifique a conexão ou especifique ucp_variables manualmente.",
        ],
        "es": [
            "No hay parámetros urbanos disponibles — no se puede entrenar el modelo ML.",
            "i=Verifique la conexión o especifique ucp_variables manualmente.",
        ],
        "zh": [
            "无可用城市参数——无法训练 ML 模型。",
            "i=请检查网络连接或手动指定 ucp_variables。",
        ],
    },
    "interp_plus_saved": {
        "en": "ML interpolation map saved to {path {path}}",
        "pt": "Mapa de interpolação ML salvo em {path {path}}",
        "es": "Mapa de interpolación ML guardado en {path {path}}",
        "zh": "ML 插值地图已保存至 {path {path}}",
    },
    "interp_plus_complete": {
        "en": "ML interpolation complete: {n} band(s), method={method}",
        "pt": "Interpolação ML concluída: {n} banda(s), method={method}",
        "es": "Interpolación ML completa: {n} banda(s), method={method}",
        "zh": "ML 插值完成：{n} 个波段，method={method}",
    },
    "interp_plus_few_stations": {
        "en": "Group '{group}': only {n} station(s) — model may be unstable.",
        "pt": "Grupo '{group}': apenas {n} estação(ões) — modelo pode ser instável.",
        "es": "Grupo '{group}': solo {n} estación(es) — el modelo puede ser inestable.",
        "zh": "分组 '{group}'：仅 {n} 个站点——模型可能不稳定。",
    },

    # ── Spatial autocorrelation (lcz_spatial_stats) ────────────────────────────

    "moran_computing": {
        "en": "Computing Global Moran's I statistic...",
        "pt": "Calculando estatística de Moran's I Global...",
        "es": "Calculando estadística de Moran's I Global...",
        "zh": "正在计算全局 Moran's I 统计量...",
    },
    "moran_too_few_stations": {
        "en": "Need at least 3 valid stations for Moran's I, got {val {n}}.",
        "pt": "Necessário pelo menos 3 estações válidas para Moran's I, obtido {val {n}}.",
        "es": "Se necesitan al menos 3 estaciones válidas para Moran's I, se obtuvieron {val {n}}.",
        "zh": "Moran's I 至少需要 3 个有效站点，当前仅有 {val {n}} 个。",
    },
    "moran_title": {
        "en": "Global Moran's I — Permutation Test",
        "pt": "Moran's I Global — Teste de Permutação",
        "es": "Moran's I Global — Teste de Permutación",
        "zh": "全局 Moran's I — 置换检验",
    },
    "moran_simulated": {
        "en": "Simulated I",
        "pt": "I simulado",
        "es": "I simulado",
        "zh": "模拟 I 值",
    },
    "moran_i_label": {
        "en": "Moran's I",
        "pt": "Moran's I",
        "es": "Moran's I",
        "zh": "Moran's I",
    },
    "moran_freq_label": {
        "en": "Frequency",
        "pt": "Frequência",
        "es": "Frecuencia",
        "zh": "频次",
    },
    "lisa_computing": {
        "en": "Computing Local Indicators of Spatial Association (LISA)...",
        "pt": "Calculando Indicadores Locais de Associação Espacial (LISA)...",
        "es": "Calculando Indicadores Locales de Asociación Espacial (LISA)...",
        "zh": "正在计算局部空间关联指标 (LISA)...",
    },
    "lisa_too_few_stations": {
        "en": "Need at least 3 valid stations for LISA, got {val {n}}.",
        "pt": "Necessário pelo menos 3 estações válidas para LISA, obtido {val {n}}.",
        "es": "Se necesitan al menos 3 estaciones válidas para LISA, se obtuvieron {val {n}}.",
        "zh": "LISA 至少需要 3 个有效站点，当前仅有 {val {n}} 个。",
    },
    "lisa_title": {
        "en": "Local Indicators of Spatial Association (LISA)",
        "pt": "Indicadores Locais de Associação Espacial (LISA)",
        "es": "Indicadores Locales de Asociación Espacial (LISA)",
        "zh": "局部空间关联指标 (LISA)",
    },
    "getis_computing": {
        "en": "Computing Getis-Ord Gi* hotspot statistic...",
        "pt": "Calculando estatística de hotspot Getis-Ord Gi*...",
        "es": "Calculando estadística de hotspots Getis-Ord Gi*...",
        "zh": "正在计算 Getis-Ord Gi* 热点统计量...",
    },
    "getis_too_few_stations": {
        "en": "Need at least 3 valid stations for Getis-Ord Gi*, got {val {n}}.",
        "pt": "Necessário pelo menos 3 estações válidas para Getis-Ord Gi*, obtido {val {n}}.",
        "es": "Se necesitan al menos 3 estaciones válidas para Getis-Ord Gi*, se obtuvieron {val {n}}.",
        "zh": "Getis-Ord Gi* 至少需要 3 个有效站点，当前仅有 {val {n}} 个。",
    },
    "getis_title": {
        "en": "Getis-Ord Gi* Hotspot Analysis",
        "pt": "Análise de Hotspot Getis-Ord Gi*",
        "es": "Análisis de Hotspots Getis-Ord Gi*",
        "zh": "Getis-Ord Gi* 热点分析",
    },

    # ── Thermal comfort / UTCI (lcz_thermal) ──────────────────────────────────

    "utci_map_title": {
        "en": "Universal Thermal Climate Index (UTCI)",
        "pt": "Índice Universal de Clima Térmico (UTCI)",
        "es": "Índice Universal de Clima Térmico (UTCI)",
        "zh": "通用热气候指数 (UTCI)",
    },
    "utci_hist_title": {
        "en": "UTCI Distribution",
        "pt": "Distribuição UTCI",
        "es": "Distribución UTCI",
        "zh": "UTCI 分布",
    },
    "qf_map_title": {
        "en": "Anthropogenic Heat Flux (Q_F)",
        "pt": "Fluxo de Calor Antropogênico (Q_F)",
        "es": "Flujo de Calor Antropogénico (Q_F)",
        "zh": "人为热通量 (Q_F)",
    },
    "qf_bar_title": {
        "en": "Anthropogenic Heat by LCZ Class",
        "pt": "Calor Antropogênico por Classe LCZ",
        "es": "Calor Antropogénico por Clase SCL",
        "zh": "按 LCZ 类别的人为热",
    },

    # ── Land Surface Temperature (lcz_get_lst) ───────────────────────────────

    "lst_no_scenes_found": {
        "en": [
            "No {arg {source}} LST scenes found for {val {start_date}}–{val {end_date}}.",
            "i=Check the date range and that the area is covered by this source.",
        ],
        "pt": [
            "Nenhuma cena de LST {arg {source}} encontrada para {val {start_date}}–{val {end_date}}.",
            "i=Verifique o intervalo de datas e se a área é coberta por essa fonte.",
        ],
        "es": [
            "No se encontraron escenas de LST {arg {source}} para {val {start_date}}–{val {end_date}}.",
            "i=Revise el rango de fechas y si el área está cubierta por esta fuente.",
        ],
        "zh": [
            "在 {val {start_date}}–{val {end_date}} 期间未找到 {arg {source}} LST 场景。",
            "i=请检查日期范围以及该区域是否被此数据源覆盖。",
        ],
    },
    "lst_day_skipped": {
        "en": "Skipped {val {date}}: no usable LST scene.",
        "pt": "Ignorado {val {date}}: nenhuma cena de LST utilizável.",
        "es": "Omitido {val {date}}: ninguna escena de LST utilizable.",
        "zh": "已跳过 {val {date}}：没有可用的 LST 场景。",
    },
    "lst_source_out_of_bounds": {
        "en": [
            "The LCZ map's area falls outside the GOES full-disk view.",
            "i=Use source='sentinel3' for areas outside the Americas.",
        ],
        "pt": [
            "A área do mapa LCZ está fora da visão de disco completo do GOES.",
            "i=Use source='sentinel3' para áreas fora das Américas.",
        ],
        "es": [
            "El área del mapa SCL está fuera de la vista de disco completo de GOES.",
            "i=Use source='sentinel3' para áreas fuera de América.",
        ],
        "zh": [
            "LCZ 地图区域超出 GOES 全圆盘视野范围。",
            "i=对于美洲以外的区域，请使用 source='sentinel3'。",
        ],
    },

    # ── Surface UHI (lcz_uhi_surface) ────────────────────────────────────────

    "uhi_shape_mismatch": {
        "en": [
            "LCZ map shape {val {lcz_shape}} does not match LST stack shape {val {lst_shape}}.",
            "i=Both must come from the same LCZ map grid — re-run lcz_get_lst on this lcz_map.",
        ],
        "pt": [
            "O formato do mapa LCZ {val {lcz_shape}} não corresponde ao formato da pilha de LST {val {lst_shape}}.",
            "i=Ambos devem vir da mesma grade do mapa LCZ — execute lcz_get_lst novamente sobre este lcz_map.",
        ],
        "es": [
            "La forma del mapa SCL {val {lcz_shape}} no coincide con la forma de la pila de LST {val {lst_shape}}.",
            "i=Ambos deben provenir de la misma grilla del mapa SCL — vuelva a ejecutar lcz_get_lst sobre este lcz_map.",
        ],
        "zh": [
            "LCZ 地图形状 {val {lcz_shape}} 与 LST 栈形状 {val {lst_shape}} 不匹配。",
            "i=两者必须来自同一 LCZ 地图网格 — 请对此 lcz_map 重新运行 lcz_get_lst。",
        ],
    },
    "uhi_surface_title": {
        "en": "Surface Urban Heat Island Intensity (Urban − Rural)",
        "pt": "Intensidade da Ilha de Calor de Superfície (Urbano − Rural)",
        "es": "Intensidad de la Isla de Calor de Superficie (Urbano − Rural)",
        "zh": "地表城市热岛强度（城市 − 乡村）",
    },
    "uhi_surface_percentile_title": {
        "en": "Surface UHI Intensity (Urban − Rural P{val {p}})",
        "pt": "Intensidade da IChS (Urbano − Rural P{val {p}})",
        "es": "Intensidad de la ICS (Urbano − Rural P{val {p}})",
        "zh": "地表热岛强度（城市 − 乡村第 {val {p}} 百分位）",
    },
    "uhi_delta_label": {
        "en": "ΔT [°C or K]", "pt": "ΔT [°C ou K]", "es": "ΔT [°C o K]", "zh": "ΔT [°C 或 K]",
    },
    "uhi_lcz_title": {
        "en": "SUHI by LCZ Class (Δ vs LCZ {val {ref}})",
        "pt": "IChS por Classe LCZ (Δ vs LCZ {val {ref}})",
        "es": "ICS por Clase SCL (Δ vs SCL {val {ref}})",
        "zh": "各 LCZ 类别的地表热岛强度（相对于 LCZ {val {ref}}）",
    },
    "uhi_utfvi_title": {
        "en": "Urban Thermal Field Variance Index — {val {date}}",
        "pt": "Índice de Variância do Campo Térmico Urbano — {val {date}}",
        "es": "Índice de Varianza del Campo Térmico Urbano — {val {date}}",
        "zh": "城市热场变异指数 — {val {date}}",
    },
    "uhi_utfvi_category_title": {
        "en": "UTFVI Ecological Evaluation Categories",
        "pt": "Categorias de Avaliação Ecológica do UTFVI",
        "es": "Categorías de Evaluación Ecológica del UTFVI",
        "zh": "UTFVI 生态评价类别",
    },
    "uhi_utfvi_category_label": {
        "en": "Category", "pt": "Categoria", "es": "Categoría", "zh": "类别",
    },
    "uhi_percent_pixels_label": {
        "en": "% of valid pixels", "pt": "% de pixels válidos",
        "es": "% de píxeles válidos", "zh": "有效像元占比 (%)",
    },
    "uhi_hotspot_title": {
        "en": "LST Hotspot Analysis (Getis-Ord Gi*) — {val {date}}",
        "pt": "Análise de Hotspot de LST (Getis-Ord Gi*) — {val {date}}",
        "es": "Análisis de Hotspot de LST (Getis-Ord Gi*) — {val {date}}",
        "zh": "LST 热点分析（Getis-Ord Gi*）— {val {date}}",
    },
    "uhi_transect_title": {
        "en": "Urban-Rural Gradient Profile",
        "pt": "Perfil de Gradiente Urbano-Rural",
        "es": "Perfil de Gradiente Urbano-Rural",
        "zh": "城乡梯度剖面",
    },
    "uhi_distance_label": {
        "en": "Distance from urban core [km]",
        "pt": "Distância do núcleo urbano [km]",
        "es": "Distancia del núcleo urbano [km]",
        "zh": "距城市核心的距离 [km]",
    },
    "uhi_persistence_title": {
        "en": "SUHI Persistence (% of dates ΔT > {val {thr}})",
        "pt": "Persistência da IChS (% de datas com ΔT > {val {thr}})",
        "es": "Persistencia de la ICS (% de fechas con ΔT > {val {thr}})",
        "zh": "地表热岛持续性（ΔT 超过 {val {thr}} 的日期占比）",
    },
    "uhi_persistence_label": {
        "en": "Persistence [%]", "pt": "Persistência [%]",
        "es": "Persistencia [%]", "zh": "持续性 [%]",
    },
    "uhi_persistence_by_class_title": {
        "en": "SUHI Persistence by LCZ Class",
        "pt": "Persistência da IChS por Classe LCZ",
        "es": "Persistencia de la ICS por Clase SCL",
        "zh": "各 LCZ 类别的地表热岛持续性",
    },

    # ── Planetary Computer (lcz_get_planetary_computer) ──────────────────────

    "pc_no_items_found": {
        "en": [
            "No {arg {collection}} items found for this area{date_range}.",
            "i=Check the collection id, date range, and that the area is covered.",
        ],
        "pt": [
            "Nenhum item de {arg {collection}} encontrado para esta área{date_range}.",
            "i=Verifique o id da coleção, o intervalo de datas e a cobertura da área.",
        ],
        "es": [
            "No se encontraron elementos de {arg {collection}} para esta área{date_range}.",
            "i=Revise el id de la colección, el rango de fechas y la cobertura del área.",
        ],
        "zh": [
            "在此区域{date_range}未找到 {arg {collection}} 的条目。",
            "i=请检查集合 ID、日期范围以及该区域是否被覆盖。",
        ],
    },
    "pc_asset_missing": {
        "en": [
            "Asset {arg {asset}} not found in {arg {collection}} items.",
            "i=Available assets on the matched item: {val {available}}.",
        ],
        "pt": [
            "Asset {arg {asset}} não encontrado nos itens de {arg {collection}}.",
            "i=Assets disponíveis no item encontrado: {val {available}}.",
        ],
        "es": [
            "No se encontró el asset {arg {asset}} en los elementos de {arg {collection}}.",
            "i=Assets disponibles en el elemento encontrado: {val {available}}.",
        ],
        "zh": [
            "在 {arg {collection}} 的条目中未找到资产 {arg {asset}}。",
            "i=匹配条目中可用的资产：{val {available}}。",
        ],
    },
    "pc_unknown_collection": {
        "en": [
            "Unknown collection shortcut {arg {collection}} — pass 'assets' explicitly.",
            "i=Use a raw Planetary Computer collection id, or one of: {val {known}}.",
        ],
        "pt": [
            "Atalho de coleção desconhecido {arg {collection}} — informe 'assets' explicitamente.",
            "i=Use um id de coleção do Planetary Computer, ou um destes: {val {known}}.",
        ],
        "es": [
            "Atajo de colección desconocido {arg {collection}} — indique 'assets' explícitamente.",
            "i=Use un id de colección de Planetary Computer, o uno de: {val {known}}.",
        ],
        "zh": [
            "未知的集合快捷方式 {arg {collection}} —请显式传入 'assets'。",
            "i=请使用 Planetary Computer 的原始集合 ID，或以下之一：{val {known}}。",
        ],
    },
    "pc_asset_partial": {
        "en": "Asset {val {asset}}: only {val {n_items}} item(s) merged, coverage may be incomplete.",
        "pt": "Asset {val {asset}}: apenas {val {n_items}} item(ns) mesclado(s), a cobertura pode estar incompleta.",
        "es": "Asset {val {asset}}: solo se combinaron {val {n_items}} elemento(s), la cobertura puede estar incompleta.",
        "zh": "资产 {val {asset}}：仅合并了 {val {n_items}} 个条目，覆盖可能不完整。",
    },

    # ── Spectral indices (lcz_get_indices / lcz_cal_indices) ─────────────────

    "indices_no_bands": {
        "en": [
            "No index could be computed — no recognized bands in {val {bands}}.",
            "i=Expected Sentinel-2 (B02/B03/B04/B08/B11/B12) or Landsat (blue/green/red/nir08/swir16/swir22) band names.",
        ],
        "pt": [
            "Nenhum índice pôde ser calculado — nenhuma banda reconhecida em {val {bands}}.",
            "i=Esperado nomes de banda Sentinel-2 (B02/B03/B04/B08/B11/B12) ou Landsat (blue/green/red/nir08/swir16/swir22).",
        ],
        "es": [
            "No se pudo calcular ningún índice — ninguna banda reconocida en {val {bands}}.",
            "i=Se esperaban nombres de banda Sentinel-2 (B02/B03/B04/B08/B11/B12) o Landsat (blue/green/red/nir08/swir16/swir22).",
        ],
        "zh": [
            "无法计算任何指数 — {val {bands}} 中没有可识别的波段。",
            "i=应为 Sentinel-2 (B02/B03/B04/B08/B11/B12) 或 Landsat (blue/green/red/nir08/swir16/swir22) 波段名称。",
        ],
    },
    "indices_unknown": {
        "en": [
            "Unknown index/indices: {val {unknown}}.",
            "i=Known indices: {val {known}}.",
        ],
        "pt": [
            "Índice(s) desconhecido(s): {val {unknown}}.",
            "i=Índices conhecidos: {val {known}}.",
        ],
        "es": [
            "Índice(s) desconocido(s): {val {unknown}}.",
            "i=Índices conocidos: {val {known}}.",
        ],
        "zh": [
            "未知指数：{val {unknown}}。",
            "i=已知指数：{val {known}}。",
        ],
    },
    "indices_missing_bands": {
        "en": [
            "Cannot compute {val {indices}} — missing band(s): {val {bands}}.",
            "i=Re-run lcz_get_planetary_computer with assets= including these bands.",
        ],
        "pt": [
            "Não é possível calcular {val {indices}} — banda(s) faltante(s): {val {bands}}.",
            "i=Execute novamente lcz_get_planetary_computer com assets= incluindo essas bandas.",
        ],
        "es": [
            "No se puede calcular {val {indices}} — banda(s) faltante(s): {val {bands}}.",
            "i=Vuelva a ejecutar lcz_get_planetary_computer con assets= incluyendo estas bandas.",
        ],
        "zh": [
            "无法计算 {val {indices}} — 缺少波段：{val {bands}}。",
            "i=请使用包含这些波段的 assets= 重新运行 lcz_get_planetary_computer。",
        ],
    },
    "indices_computed": {
        "en": "Computed {val {name}}.",
        "pt": "{val {name}} calculado.",
        "es": "{val {name}} calculado.",
        "zh": "已计算 {val {name}}。",
    },
    "indices_stats_title": {
        "en": "Spectral Index by LCZ Class",
        "pt": "Índice Espectral por Classe LCZ",
        "es": "Índice Espectral por Clase SCL",
        "zh": "按 LCZ 类别的光谱指数",
    },
    "indices_shape_mismatch": {
        "en": [
            "LCZ map shape {val {lcz_shape}} does not match indices stack shape {val {idx_shape}}.",
            "i=Both must come from the same LCZ map grid — re-run lcz_get_planetary_computer on this lcz_map.",
        ],
        "pt": [
            "O formato do mapa LCZ {val {lcz_shape}} não corresponde ao formato da pilha de índices {val {idx_shape}}.",
            "i=Ambos devem vir da mesma grade do mapa LCZ — execute lcz_get_planetary_computer novamente sobre este lcz_map.",
        ],
        "es": [
            "La forma del mapa SCL {val {lcz_shape}} no coincide con la forma de la pila de índices {val {idx_shape}}.",
            "i=Ambos deben provenir de la misma grilla del mapa SCL — vuelva a ejecutar lcz_get_planetary_computer sobre este lcz_map.",
        ],
        "zh": [
            "LCZ 地图形状 {val {lcz_shape}} 与指数栈形状 {val {idx_shape}} 不匹配。",
            "i=两者必须来自同一 LCZ 地图网格 — 请对此 lcz_map 重新运行 lcz_get_planetary_computer。",
        ],
    },
    "indices_effect_size_title": {
        "en": "Effect Size (Cohen's d): Built-up vs Natural",
        "pt": "Tamanho do Efeito (Cohen's d): Urbano vs Natural",
        "es": "Tamaño del Efecto (Cohen's d): Urbano vs Natural",
        "zh": "效应量 (Cohen's d)：建成区与自然区",
    },
    "indices_scatter_title": {
        "en": "{val {x}} vs {val {y}} by LCZ Class",
        "pt": "{val {x}} vs {val {y}} por Classe LCZ",
        "es": "{val {x}} vs {val {y}} por Clase SCL",
        "zh": "{val {x}} 对比 {val {y}}（按 LCZ 类别）",
    },
    "indices_radar_title": {
        "en": "LCZ Class Profile Across Indices",
        "pt": "Perfil da Classe LCZ Através dos Índices",
        "es": "Perfil de la Clase SCL a Través de los Índices",
        "zh": "LCZ 类别的多指数轮廓",
    },
    "indices_correlation_title": {
        "en": "Correlation Between Spectral Indices",
        "pt": "Correlação Entre Índices Espectrais",
        "es": "Correlación Entre Índices Espectrales",
        "zh": "光谱指数间的相关性",
    },
    "indices_type_urban": {
        "en": "Built-up", "pt": "Urbano", "es": "Urbano", "zh": "建成区",
    },
    "indices_type_natural": {
        "en": "Natural", "pt": "Natural", "es": "Natural", "zh": "自然区",
    },
    "indices_magnitude_negligible": {
        "en": "Negligible", "pt": "Insignificante", "es": "Insignificante", "zh": "微不足道",
    },
    "indices_magnitude_small": {
        "en": "Small", "pt": "Pequeno", "es": "Pequeño", "zh": "小",
    },
    "indices_magnitude_medium": {
        "en": "Medium", "pt": "Médio", "es": "Medio", "zh": "中",
    },
    "indices_magnitude_large": {
        "en": "Large", "pt": "Grande", "es": "Grande", "zh": "大",
    },

    # ── lcz_cal_indexes ──────────────────────────────────────────────────────

    "cal_indexes_title": {
        "en": "{variable} by LCZ class",
        "pt": "{variable} por classe LCZ",
        "es": "{variable} por clase LCZ",
        "zh": "按 LCZ 类别划分的 {variable}",
    },
    "cal_indexes_no_overlap": {
        "en": "No overlapping valid pixels between the LCZ map and the grid data.",
        "pt": "Nenhum pixel válido sobreposto entre o mapa LCZ e os dados de grade.",
        "es": "No hay píxeles válidos superpuestos entre el mapa LCZ y los datos de la cuadrícula.",
        "zh": "LCZ 地图与网格数据之间没有重叠的有效像元。",
    },

    # ── shared: raster lcz_grid_* (crop/mask onto an LCZ map's grid) ────────

    "grid_raster_invalid_resampling": {
        "en": "Invalid resampling method. Use: {valid}.",
        "pt": "Método de reamostragem inválido. Use: {valid}.",
        "es": "Método de remuestreo inválido. Use: {valid}.",
        "zh": "无效的重采样方法。请使用：{valid}。",
    },
    "grid_raster_cache_hit": {
        "en": "Using cached {name} stack.",
        "pt": "Usando pilha {name} em cache.",
        "es": "Usando pila {name} en caché.",
        "zh": "使用缓存的 {name} 数据栈。",
    },
    "grid_raster_done": {
        "en": "Complete: {name} stack with {n_bands} band(s), cropped to the LCZ map.",
        "pt": "Concluído: pilha {name} com {n_bands} banda(s), recortada ao mapa LCZ.",
        "es": "Completo: pila {name} con {n_bands} banda(s), recortada al mapa LCZ.",
        "zh": "完成：{name} 数据栈共 {n_bands} 个波段，已裁剪至 LCZ 地图范围。",
    },

    # ── lcz_grid_chirps ──────────────────────────────────────────────────────

    "grid_chirps_invalid_resolution": {
        "en": "Invalid resolution: {bad}. Use: {valid}.",
        "pt": "Resolução inválida: {bad}. Use: {valid}.",
        "es": "Resolución inválida: {bad}. Use: {valid}.",
        "zh": "无效的分辨率：{bad}。请使用：{valid}。",
    },
    "grid_chirps_invalid_agg": {
        "en": "Invalid agg_fun. Options: {valid}.",
        "pt": "agg_fun inválido. Opções: {valid}.",
        "es": "agg_fun inválido. Opciones: {valid}.",
        "zh": "无效的 agg_fun。可选：{valid}。",
    },
    "grid_chirps_default_years": {
        "en": "Using years {years} (default: last 2 complete years).",
        "pt": "Usando anos {years} (padrão: últimos 2 anos completos).",
        "es": "Usando años {years} (por defecto: últimos 2 años completos).",
        "zh": "使用年份 {years}（默认：最近两个完整年份）。",
    },
    "grid_chirps_invalid_years_range": {
        "en": "Invalid year(s): {bad}. Must be between 1981 and {max_year}.",
        "pt": "Ano(s) inválido(s): {bad}. Deve estar entre 1981 e {max_year}.",
        "es": "Año(s) inválido(s): {bad}. Debe estar entre 1981 y {max_year}.",
        "zh": "无效年份：{bad}。必须在 1981 到 {max_year} 之间。",
    },
    "grid_chirps_invalid_months": {
        "en": "months must be integers between 1 and 12.",
        "pt": "months deve ser inteiro entre 1 e 12.",
        "es": "months debe ser entero entre 1 y 12.",
        "zh": "months 必须是 1 到 12 之间的整数。",
    },
    "grid_chirps_muni_not_sf": {
        "en": "municipalities must be a GeoDataFrame.",
        "pt": "municipalities deve ser um GeoDataFrame.",
        "es": "municipalities debe ser un GeoDataFrame.",
        "zh": "municipalities 必须是 GeoDataFrame。",
    },
    "grid_chirps_no_data_params": {
        "en": "No data available for the provided parameters.",
        "pt": "Nenhum dado disponível para os parâmetros fornecidos.",
        "es": "No hay datos disponibles para los parámetros indicados.",
        "zh": "没有符合所给参数的数据。",
    },
    "grid_chirps_download_start": {
        "en": "Downloading {n_files} CHIRPS file(s) from CHC/UCSB...",
        "pt": "Baixando {n_files} arquivo(s) CHIRPS do CHC/UCSB...",
        "es": "Descargando {n_files} archivo(s) CHIRPS de CHC/UCSB...",
        "zh": "正在从 CHC/UCSB 下载 {n_files} 个 CHIRPS 文件...",
    },
    "grid_chirps_parquet_cache_hit": {
        "en": "All files found in Parquet cache. Loading...",
        "pt": "Todos os arquivos no cache Parquet. Carregando...",
        "es": "Todos los archivos en caché Parquet. Cargando...",
        "zh": "所有文件均命中 Parquet 缓存，正在加载...",
    },
    "grid_chirps_skip_missing": {
        "en": "File not available: {filename}",
        "pt": "Arquivo não disponível: {filename}",
        "es": "Archivo no disponible: {filename}",
        "zh": "文件不可用：{filename}",
    },
    "grid_chirps_extract_warn": {
        "en": "Could not process {filename}.",
        "pt": "Não foi possível processar {filename}.",
        "es": "No se pudo procesar {filename}.",
        "zh": "无法处理 {filename}。",
    },
    "grid_chirps_no_data": {
        "en": "No data was successfully extracted.",
        "pt": "Nenhum dado foi extraído com sucesso.",
        "es": "No se extrajo ningún dato correctamente.",
        "zh": "没有成功提取任何数据。",
    },
    "grid_chirps_agg_start": {
        "en": "Aggregating to {n_mun} municipality/ies...",
        "pt": "Agregando para {n_mun} município(s)...",
        "es": "Agregando a {n_mun} municipio(s)...",
        "zh": "正在聚合到 {n_mun} 个市镇...",
    },
    "grid_chirps_agg_done": {
        "en": "Complete: {n_rows} observations ({n_mun} municipalities).",
        "pt": "Concluído: {n_rows} observações ({n_mun} municípios).",
        "es": "Completo: {n_rows} observaciones ({n_mun} municipios).",
        "zh": "完成：{n_rows} 条观测（{n_mun} 个市镇）。",
    },

    # ── lcz_grid_pdsi ────────────────────────────────────────────────────────

    "grid_pdsi_invalid_source": {
        "en": "Invalid source. Options: {valid}.",
        "pt": "source inválido. Opções: {valid}.",
        "es": "source inválido. Opciones: {valid}.",
        "zh": "无效的 source。可选：{valid}。",
    },
    "grid_pdsi_invalid_agg": {
        "en": "Invalid agg_fun. Options: {valid}.",
        "pt": "agg_fun inválido. Opções: {valid}.",
        "es": "agg_fun inválido. Opciones: {valid}.",
        "zh": "无效的 agg_fun。可选：{valid}。",
    },
    "grid_pdsi_default_years": {
        "en": "Using years {years} (default: last 2 complete years).",
        "pt": "Usando anos {years} (padrão: últimos 2 anos completos).",
        "es": "Usando años {years} (por defecto: últimos 2 años completos).",
        "zh": "使用年份 {years}（默认：最近两个完整年份）。",
    },
    "grid_pdsi_invalid_years_range": {
        "en": "Invalid year(s) for this source: {bad}.",
        "pt": "Ano(s) inválido(s) para esta fonte: {bad}.",
        "es": "Año(s) inválido(s) para esta fuente: {bad}.",
        "zh": "该数据源不支持以下年份：{bad}。",
    },
    "grid_pdsi_invalid_months": {
        "en": "months must be integers between 1 and 12.",
        "pt": "months deve ser inteiro entre 1 e 12.",
        "es": "months debe ser entero entre 1 y 12.",
        "zh": "months 必须是 1 到 12 之间的整数。",
    },
    "grid_pdsi_muni_not_sf": {
        "en": "municipalities must be a GeoDataFrame.",
        "pt": "municipalities deve ser um GeoDataFrame.",
        "es": "municipalities debe ser un GeoDataFrame.",
        "zh": "municipalities 必须是 GeoDataFrame。",
    },
    "grid_pdsi_download_start": {
        "en": "Downloading {n_files} PDSI file(s) ({source}), no authentication...",
        "pt": "Baixando {n_files} arquivo(s) PDSI ({source}), sem autenticação...",
        "es": "Descargando {n_files} archivo(s) PDSI ({source}), sin autenticación...",
        "zh": "正在下载 {n_files} 个 PDSI 文件（{source}），无需身份验证...",
    },
    "grid_pdsi_parquet_cache_hit": {
        "en": "All data found in Parquet cache. Loading...",
        "pt": "Todos os dados no cache Parquet. Carregando...",
        "es": "Todos los datos en caché Parquet. Cargando...",
        "zh": "所有数据均命中 Parquet 缓存，正在加载...",
    },
    "grid_pdsi_skip_missing": {
        "en": "File not available: {filename}",
        "pt": "Arquivo não disponível: {filename}",
        "es": "Archivo no disponible: {filename}",
        "zh": "文件不可用：{filename}",
    },
    "grid_pdsi_extract_warn": {
        "en": "Could not process {filename}.",
        "pt": "Não foi possível processar {filename}.",
        "es": "No se pudo procesar {filename}.",
        "zh": "无法处理 {filename}。",
    },
    "grid_pdsi_no_data": {
        "en": "No data was successfully extracted.",
        "pt": "Nenhum dado foi extraído com sucesso.",
        "es": "No se extrajo ningún dato correctamente.",
        "zh": "没有成功提取任何数据。",
    },
    "grid_pdsi_agg_start": {
        "en": "Aggregating to {n_mun} municipality/ies...",
        "pt": "Agregando para {n_mun} município(s)...",
        "es": "Agregando a {n_mun} municipio(s)...",
        "zh": "正在聚合到 {n_mun} 个市镇...",
    },
    "grid_pdsi_agg_done": {
        "en": "Complete: {n_rows} observations ({n_mun} municipalities).",
        "pt": "Concluído: {n_rows} observações ({n_mun} municípios).",
        "es": "Completo: {n_rows} observaciones ({n_mun} municipios).",
        "zh": "完成：{n_rows} 条观测（{n_mun} 个市镇）。",
    },

    # ── lcz_grid_era5 ────────────────────────────────────────────────────────

    "grid_era5_missing_years": {
        "en": "years is required.",
        "pt": "years é obrigatório.",
        "es": "years es obligatorio.",
        "zh": "years 是必需的。",
    },
    "grid_era5_out_of_coverage": {
        "en": "This LCZ map is outside the Latin America coverage of lcz_grid_era5. Use lcz_grid_era5_global instead (requires a free Copernicus CDS API key).",
        "pt": "Este mapa LCZ está fora da cobertura da América Latina de lcz_grid_era5. Use lcz_grid_era5_global (requer uma chave de API do CDS Copernicus gratuita).",
        "es": "Este mapa LCZ está fuera de la cobertura de América Latina de lcz_grid_era5. Use lcz_grid_era5_global (requiere una clave de API de CDS Copernicus gratuita).",
        "zh": "此 LCZ 地图超出 lcz_grid_era5 的拉丁美洲覆盖范围。请改用 lcz_grid_era5_global（需要免费的 Copernicus CDS API 密钥）。",
    },

    # ── lcz_grid_era5_global ─────────────────────────────────────────────────

    "grid_era5_global_no_cdsapi": {
        "en": "The 'cdsapi' package is required for lcz_grid_era5_global. Install it with: pip install cdsapi",
        "pt": "O pacote 'cdsapi' é necessário para lcz_grid_era5_global. Instale com: pip install cdsapi",
        "es": "El paquete 'cdsapi' es necesario para lcz_grid_era5_global. Instale con: pip install cdsapi",
        "zh": "lcz_grid_era5_global 需要 'cdsapi' 软件包。请使用以下命令安装：pip install cdsapi",
    },
    "grid_era5_global_no_auth": {
        "en": "Copernicus CDS credentials not found. Set CDSAPI_KEY (and optionally CDSAPI_URL), pass cds_key/cds_url, or create ~/.cdsapirc. Get a free key at https://cds.climate.copernicus.eu",
        "pt": "Credenciais do Copernicus CDS não encontradas. Defina CDSAPI_KEY (e opcionalmente CDSAPI_URL), passe cds_key/cds_url, ou crie ~/.cdsapirc. Obtenha uma chave gratuita em https://cds.climate.copernicus.eu",
        "es": "No se encontraron credenciales de Copernicus CDS. Defina CDSAPI_KEY (y opcionalmente CDSAPI_URL), pase cds_key/cds_url, o cree ~/.cdsapirc. Obtenga una clave gratuita en https://cds.climate.copernicus.eu",
        "zh": "未找到 Copernicus CDS 凭据。请设置 CDSAPI_KEY（可选 CDSAPI_URL）、传入 cds_key/cds_url，或创建 ~/.cdsapirc。可在 https://cds.climate.copernicus.eu 免费获取密钥。",
    },
    "grid_era5_global_download_start": {
        "en": "Requesting {n_files} ERA5-Land monthly file(s) from Copernicus CDS...",
        "pt": "Solicitando {n_files} arquivo(s) mensal(is) ERA5-Land ao Copernicus CDS...",
        "es": "Solicitando {n_files} archivo(s) mensual(es) ERA5-Land al Copernicus CDS...",
        "zh": "正在向 Copernicus CDS 请求 {n_files} 个 ERA5-Land 月度文件...",
    },
    "grid_era5_global_download_warn": {
        "en": "CDS request failed for {year}-{month:02d}",
        "pt": "Falha na solicitação ao CDS para {year}-{month:02d}",
        "es": "Falló la solicitud al CDS para {year}-{month:02d}",
        "zh": "{year}-{month:02d} 的 CDS 请求失败",
    },
    "grid_era5_global_skip_missing": {
        "en": "Skipping {year}-{month:02d}: file not downloaded",
        "pt": "Ignorando {year}-{month:02d}: arquivo não baixado",
        "es": "Omitiendo {year}-{month:02d}: archivo no descargado",
        "zh": "跳过 {year}-{month:02d}：文件未下载",
    },
    "grid_era5_global_extract_warn": {
        "en": "Could not crop/reproject {year}-{month:02d}",
        "pt": "Não foi possível recortar/reprojetar {year}-{month:02d}",
        "es": "No se pudo recortar/reproyectar {year}-{month:02d}",
        "zh": "无法裁剪/重投影 {year}-{month:02d}",
    },
    "grid_era5_invalid_years_range": {
        "en": "Invalid year(s): {bad}. Must be between 1950 and 2025.",
        "pt": "Ano(s) inválido(s): {bad}. Deve estar entre 1950 e 2025.",
        "es": "Año(s) inválido(s): {bad}. Debe estar entre 1950 y 2025.",
        "zh": "无效年份：{bad}。必须在 1950 到 2025 之间。",
    },
    "grid_era5_invalid_months": {
        "en": "months must be integers between 1 and 12.",
        "pt": "months deve ser inteiro entre 1 e 12.",
        "es": "months debe ser entero entre 1 y 12.",
        "zh": "months 必须是 1 到 12 之间的整数。",
    },
    "grid_era5_invalid_vars": {
        "en": "vars contains invalid alias(es): {bad}. Use: {valid}.",
        "pt": "vars contém alias inválido(s): {bad}. Use: {valid}.",
        "es": "vars contiene alias inválido(s): {bad}. Use: {valid}.",
        "zh": "vars 包含无效别名：{bad}。请使用：{valid}。",
    },
    "grid_era5_invalid_agg": {
        "en": "Invalid agg_fun. Options: {valid}.",
        "pt": "agg_fun inválido. Opções: {valid}.",
        "es": "agg_fun inválido. Opciones: {valid}.",
        "zh": "无效的 agg_fun。可选：{valid}。",
    },
    "grid_era5_muni_not_sf": {
        "en": "municipalities must be a GeoDataFrame.",
        "pt": "municipalities deve ser um GeoDataFrame.",
        "es": "municipalities debe ser un GeoDataFrame.",
        "zh": "municipalities 必须是 GeoDataFrame。",
    },
    "grid_era5_download_start": {
        "en": "Downloading {n_files} ERA5-Land file(s) from Zenodo...",
        "pt": "Baixando {n_files} arquivo(s) ERA5-Land do Zenodo...",
        "es": "Descargando {n_files} archivo(s) ERA5-Land de Zenodo...",
        "zh": "正在从 Zenodo 下载 {n_files} 个 ERA5-Land 文件...",
    },
    "grid_era5_agg_start": {
        "en": "Aggregating to {n_mun} municipality/ies...",
        "pt": "Agregando para {n_mun} município(s)...",
        "es": "Agregando a {n_mun} municipio(s)...",
        "zh": "正在聚合到 {n_mun} 个市镇...",
    },
    "grid_era5_skip_missing": {
        "en": "File not available: {filename}",
        "pt": "Arquivo não disponível: {filename}",
        "es": "Archivo no disponible: {filename}",
        "zh": "文件不可用：{filename}",
    },
    "grid_era5_extract_warn": {
        "en": "Could not process {file}.",
        "pt": "Não foi possível processar {file}.",
        "es": "No se pudo procesar {file}.",
        "zh": "无法处理 {file}。",
    },
    "grid_era5_no_data": {
        "en": "No data was successfully extracted.",
        "pt": "Nenhum dado foi extraído com sucesso.",
        "es": "No se extrajo ningún dato correctamente.",
        "zh": "没有成功提取任何数据。",
    },
    "grid_era5_agg_done": {
        "en": "Complete: {n_rows} observations ({n_mun} municipalities).",
        "pt": "Concluído: {n_rows} observações ({n_mun} municípios).",
        "es": "Completo: {n_rows} observaciones ({n_mun} municipios).",
        "zh": "完成：{n_rows} 条观测（{n_mun} 个市镇）。",
    },

    # ── lcz_grid_pollution_merra2 ────────────────────────────────────────────

    "grid_merra2_invalid_pollutants": {
        "en": "Invalid pollutant(s): {bad}. Use: pm25, aod, so2.",
        "pt": "Poluente(s) inválido(s): {bad}. Use: pm25, aod, so2.",
        "es": "Contaminante(s) inválido(s): {bad}. Use: pm25, aod, so2.",
        "zh": "无效污染物：{bad}。请使用：pm25, aod, so2。",
    },
    "grid_merra2_invalid_resolution": {
        "en": "resolution must be 'monthly' or 'daily'.",
        "pt": "resolution deve ser 'monthly' ou 'daily'.",
        "es": "resolution debe ser 'monthly' o 'daily'.",
        "zh": "resolution 必须是 'monthly' 或 'daily'。",
    },
    "grid_merra2_so2_monthly_only": {
        "en": "so2 is only available at monthly resolution; dropping it.",
        "pt": "so2 só está disponível na resolução mensal; removendo.",
        "es": "so2 solo está disponible en resolución mensual; eliminando.",
        "zh": "so2 仅支持月度分辨率，已将其移除。",
    },
    "grid_merra2_no_pollutants": {
        "en": "No pollutants left after removing so2 (daily not supported).",
        "pt": "Nenhum poluente restante após remover so2 (diário não suportado).",
        "es": "No quedan contaminantes tras eliminar so2 (diario no soportado).",
        "zh": "移除 so2（不支持日度）后没有剩余污染物。",
    },
    "grid_merra2_invalid_years_range": {
        "en": "Invalid year(s): {bad}. Must be between 1980 and the current year.",
        "pt": "Ano(s) inválido(s): {bad}. Deve estar entre 1980 e o ano atual.",
        "es": "Año(s) inválido(s): {bad}. Debe estar entre 1980 y el año actual.",
        "zh": "无效年份：{bad}。必须在 1980 到当前年份之间。",
    },
    "grid_merra2_invalid_months": {
        "en": "months must be integers between 1 and 12.",
        "pt": "months deve ser inteiro entre 1 e 12.",
        "es": "months debe ser entero entre 1 y 12.",
        "zh": "months 必须是 1 到 12 之间的整数。",
    },
    "grid_merra2_invalid_agg": {
        "en": "Invalid agg_fun.",
        "pt": "agg_fun inválido.",
        "es": "agg_fun inválido.",
        "zh": "无效的 agg_fun。",
    },
    "grid_merra2_no_auth": {
        "en": "NASA Earthdata credentials not found. Set EARTHDATA_USER/EARTHDATA_PASSWORD or pass netrc_path.",
        "pt": "Credenciais NASA Earthdata não encontradas. Defina EARTHDATA_USER/EARTHDATA_PASSWORD ou use netrc_path.",
        "es": "Credenciales NASA Earthdata no encontradas. Defina EARTHDATA_USER/EARTHDATA_PASSWORD o use netrc_path.",
        "zh": "未找到 NASA Earthdata 凭据。请设置 EARTHDATA_USER/EARTHDATA_PASSWORD 或传入 netrc_path。",
    },
    "grid_merra2_muni_not_sf": {
        "en": "municipalities must be a GeoDataFrame.",
        "pt": "municipalities deve ser um GeoDataFrame.",
        "es": "municipalities debe ser un GeoDataFrame.",
        "zh": "municipalities 必须是 GeoDataFrame。",
    },
    "grid_merra2_download_start": {
        "en": "Downloading {n_files} MERRA-2 file(s) from GES DISC...",
        "pt": "Baixando {n_files} arquivo(s) MERRA-2 do GES DISC...",
        "es": "Descargando {n_files} archivo(s) MERRA-2 de GES DISC...",
        "zh": "正在从 GES DISC 下载 {n_files} 个 MERRA-2 文件...",
    },
    "grid_merra2_parquet_cache_hit": {
        "en": "All data found in Parquet cache. Loading...",
        "pt": "Todos os dados no cache Parquet. Carregando...",
        "es": "Todos los datos en caché Parquet. Cargando...",
        "zh": "所有数据均命中 Parquet 缓存，正在加载...",
    },
    "grid_merra2_agg_start": {
        "en": "Aggregating to {n_mun} municipality/ies...",
        "pt": "Agregando para {n_mun} município(s)...",
        "es": "Agregando a {n_mun} municipio(s)...",
        "zh": "正在聚合到 {n_mun} 个市镇...",
    },
    "grid_merra2_skip_missing": {
        "en": "File not available: {filename}",
        "pt": "Arquivo não disponível: {filename}",
        "es": "Archivo no disponible: {filename}",
        "zh": "文件不可用：{filename}",
    },
    "grid_merra2_extract_warn": {
        "en": "Could not process {filename}.",
        "pt": "Não foi possível processar {filename}.",
        "es": "No se pudo procesar {filename}.",
        "zh": "无法处理 {filename}。",
    },
    "grid_merra2_no_data": {
        "en": "No data was successfully extracted.",
        "pt": "Nenhum dado foi extraído com sucesso.",
        "es": "No se extrajo ningún dato correctamente.",
        "zh": "没有成功提取任何数据。",
    },
    "grid_merra2_agg_done": {
        "en": "Complete: {n_rows} observations.",
        "pt": "Concluído: {n_rows} observações.",
        "es": "Completo: {n_rows} observaciones.",
        "zh": "完成：{n_rows} 条观测。",
    },

    # ── lcz_grid_pollution_ghap ──────────────────────────────────────────────

    "grid_ghap_no2_unavailable": {
        "en": "NO2 is not yet publicly available for GHAP; skipping.",
        "pt": "NO2 ainda não está disponível publicamente no GHAP; ignorando.",
        "es": "NO2 aún no está disponible públicamente en GHAP; omitiendo.",
        "zh": "GHAP 尚未公开发布 NO2 数据，已跳过。",
    },
    "grid_ghap_invalid_pollutants": {
        "en": "Invalid pollutant(s): {bad}. Use: pm25, o3, co.",
        "pt": "Poluente(s) inválido(s): {bad}. Use: pm25, o3, co.",
        "es": "Contaminante(s) inválido(s): {bad}. Use: pm25, o3, co.",
        "zh": "无效污染物：{bad}。请使用：pm25, o3, co。",
    },
    "grid_ghap_no_pollutants": {
        "en": "No pollutants left after filtering.",
        "pt": "Nenhum poluente restante após filtragem.",
        "es": "No quedan contaminantes tras el filtrado.",
        "zh": "过滤后没有剩余污染物。",
    },
    "grid_ghap_invalid_resolution": {
        "en": "Invalid resolution. Options: daily, monthly, annual.",
        "pt": "resolution inválido. Opções: daily, monthly, annual.",
        "es": "resolution inválido. Opciones: daily, monthly, annual.",
        "zh": "无效的 resolution。可选：daily, monthly, annual。",
    },
    "grid_ghap_invalid_months": {
        "en": "months must be integers between 1 and 12.",
        "pt": "months deve ser inteiro entre 1 e 12.",
        "es": "months debe ser entero entre 1 y 12.",
        "zh": "months 必须是 1 到 12 之间的整数。",
    },
    "grid_ghap_invalid_agg": {
        "en": "Invalid agg_fun.",
        "pt": "agg_fun inválido.",
        "es": "agg_fun inválido.",
        "zh": "无效的 agg_fun。",
    },
    "grid_ghap_muni_not_sf": {
        "en": "municipalities must be a GeoDataFrame.",
        "pt": "municipalities deve ser um GeoDataFrame.",
        "es": "municipalities debe ser un GeoDataFrame.",
        "zh": "municipalities 必须是 GeoDataFrame。",
    },
    "grid_ghap_fallback_annual": {
        "en": "{pollutant} is annual-only; falling back to resolution='annual'.",
        "pt": "{pollutant} só está disponível anualmente; usando resolution='annual'.",
        "es": "{pollutant} solo está disponible anualmente; usando resolution='annual'.",
        "zh": "{pollutant} 仅提供年度数据，回退为 resolution='annual'。",
    },
    "grid_ghap_years_unavail": {
        "en": "{pollutant}: year(s) {bad_years} unavailable (range: {avail}).",
        "pt": "{pollutant}: ano(s) {bad_years} indisponível(is) (intervalo: {avail}).",
        "es": "{pollutant}: año(s) {bad_years} no disponible(s) (rango: {avail}).",
        "zh": "{pollutant}：年份 {bad_years} 不可用（范围：{avail}）。",
    },
    "grid_ghap_no_data_to_download": {
        "en": "No data available for the provided parameters.",
        "pt": "Nenhum dado disponível para os parâmetros fornecidos.",
        "es": "No hay datos disponibles para los parámetros indicados.",
        "zh": "没有符合所给参数的数据。",
    },
    "grid_ghap_download_start": {
        "en": "Downloading {n_files} GHAP file(s) from Zenodo...",
        "pt": "Baixando {n_files} arquivo(s) GHAP do Zenodo...",
        "es": "Descargando {n_files} archivo(s) GHAP de Zenodo...",
        "zh": "正在从 Zenodo 下载 {n_files} 个 GHAP 文件...",
    },
    "grid_ghap_parquet_cache_hit": {
        "en": "All data found in Parquet cache. Loading...",
        "pt": "Todos os dados no cache Parquet. Carregando...",
        "es": "Todos los datos en caché Parquet. Cargando...",
        "zh": "所有数据均命中 Parquet 缓存，正在加载...",
    },
    "grid_ghap_agg_start": {
        "en": "Aggregating to {n_mun} municipality/ies...",
        "pt": "Agregando para {n_mun} município(s)...",
        "es": "Agregando a {n_mun} municipio(s)...",
        "zh": "正在聚合到 {n_mun} 个市镇...",
    },
    "grid_ghap_skip_missing": {
        "en": "File not available: {filename}",
        "pt": "Arquivo não disponível: {filename}",
        "es": "Archivo no disponible: {filename}",
        "zh": "文件不可用：{filename}",
    },
    "grid_ghap_extract_warn": {
        "en": "Could not process {filename}.",
        "pt": "Não foi possível processar {filename}.",
        "es": "No se pudo procesar {filename}.",
        "zh": "无法处理 {filename}。",
    },
    "grid_ghap_no_data": {
        "en": "No data was successfully extracted.",
        "pt": "Nenhum dado foi extraído com sucesso.",
        "es": "No se extrajo ningún dato correctamente.",
        "zh": "没有成功提取任何数据。",
    },
    "grid_ghap_agg_done": {
        "en": "Complete: {n_rows} observations.",
        "pt": "Concluído: {n_rows} observações.",
        "es": "Completo: {n_rows} observaciones.",
        "zh": "完成：{n_rows} 条观测。",
    },
    "grid_ghap_zip_no_nc": {
        "en": "No .nc files found inside {filename}.",
        "pt": "Nenhum arquivo .nc encontrado dentro de {filename}.",
        "es": "No se encontraron archivos .nc dentro de {filename}.",
        "zh": "{filename} 内未找到 .nc 文件。",
    },

    # ── plot_lcz_relationship ────────────────────────────────────────────────

    "plot_lcz_rel_no_overlap": {
        "en": "No overlapping valid pixels between the LCZ map and the variable raster.",
        "pt": "Nenhum pixel válido em comum entre o mapa LCZ e o raster da variável.",
        "es": "No hay píxeles válidos en común entre el mapa LCZ y el ráster de la variable.",
        "zh": "LCZ 地图与变量栅格之间没有重叠的有效像元。",
    },
    "plot_lcz_rel_invalid_type": {
        "en": "Invalid plot_type. Options: {valid}.",
        "pt": "plot_type inválido. Opções: {valid}.",
        "es": "plot_type inválido. Opciones: {valid}.",
        "zh": "无效的 plot_type。可选：{valid}。",
    },

    # ── plot_grid_only ───────────────────────────────────────────────────────

    "plot_grid_bad_color_by": {
        "en": "Column '{col}' not found. Available: {avail}.",
        "pt": "Coluna '{col}' não encontrada. Disponíveis: {avail}.",
        "es": "Columna '{col}' no encontrada. Disponibles: {avail}.",
        "zh": "未找到 '{col}' 列。可用列：{avail}。",
    },
    "plot_grid_no_contextily": {
        "en": "contextily not installed; skipping basemap. Install with pip install contextily.",
        "pt": "contextily não instalado; ignorando mapa base. Instale com pip install contextily.",
        "es": "contextily no instalado; omitiendo mapa base. Instale con pip install contextily.",
        "zh": "未安装 contextily，跳过底图。请使用 pip install contextily 安装。",
    },

    # ── lcz_climate_compute_spi ──────────────────────────────────────────────

    "spi_var_not_found": {
        "en": "Column '{var}' not found. Available: {cols}.",
        "pt": "Coluna '{var}' não encontrada. Disponíveis: {cols}.",
        "es": "Columna '{var}' no encontrada. Disponibles: {cols}.",
        "zh": "未找到 '{var}' 列。可用列：{cols}。",
    },
    "spi_missing_col": {
        "en": "Required column '{col}' not found.",
        "pt": "Coluna obrigatória '{col}' não encontrada.",
        "es": "Columna obligatoria '{col}' no encontrada.",
        "zh": "未找到必需的列 '{col}'。",
    },
    "spi_invalid_scales": {
        "en": "scales must be a non-empty list of integers >= 1.",
        "pt": "scales deve ser uma lista não vazia de inteiros >= 1.",
        "es": "scales debe ser una lista no vacía de enteros >= 1.",
        "zh": "scales 必须是非空的、值 >= 1 的整数列表。",
    },
    "spi_invalid_ref_period": {
        "en": "ref_start must be before ref_end.",
        "pt": "ref_start deve ser anterior a ref_end.",
        "es": "ref_start debe ser anterior a ref_end.",
        "zh": "ref_start 必须早于 ref_end。",
    },
    "spi_title": {
        "en": "Computing SPI (Standardized Precipitation Index)",
        "pt": "Calculando SPI (Standardized Precipitation Index)",
        "es": "Calculando SPI (Standardized Precipitation Index)",
        "zh": "正在计算 SPI（标准化降水指数）",
    },
    "spi_computing_scale": {
        "en": "Computing {s}-month scale -> {col}...",
        "pt": "Calculando escala de {s} mês(es) -> {col}...",
        "es": "Calculando escala de {s} mes(es) -> {col}...",
        "zh": "正在计算 {s} 个月尺度 -> {col}...",
    },
    "spi_done": {
        "en": "Done: {n_rows} rows, {n_na} NA(s) in '{col1}'.",
        "pt": "Concluído: {n_rows} linhas, {n_na} NA(s) em '{col1}'.",
        "es": "Completo: {n_rows} filas, {n_na} NA(s) en '{col1}'.",
        "zh": "完成：{n_rows} 行，'{col1}' 中有 {n_na} 个缺失值。",
    },

    # ── lcz_climate_compute_spei ─────────────────────────────────────────────

    "spei_invalid_pet_method": {
        "en": "pet_method must be 'column' or 'thornthwaite'.",
        "pt": "pet_method deve ser 'column' ou 'thornthwaite'.",
        "es": "pet_method debe ser 'column' o 'thornthwaite'.",
        "zh": "pet_method 必须是 'column' 或 'thornthwaite'。",
    },
    "spei_missing_col": {
        "en": "Required column '{col}' not found.",
        "pt": "Coluna obrigatória '{col}' não encontrada.",
        "es": "Columna obligatoria '{col}' no encontrada.",
        "zh": "未找到必需的列 '{col}'。",
    },
    "spei_pet_col_missing": {
        "en": "PET column '{pet_var}' not found. Use pet_method='thornthwaite' or provide the column.",
        "pt": "Coluna de PET '{pet_var}' não encontrada. Use pet_method='thornthwaite' ou forneça a coluna.",
        "es": "Columna de PET '{pet_var}' no encontrada. Use pet_method='thornthwaite' o proporcione la columna.",
        "zh": "未找到 PET 列 '{pet_var}'。请使用 pet_method='thornthwaite' 或提供该列。",
    },
    "spei_temp_col_missing": {
        "en": "Temperature column '{temp_var}' required for pet_method='thornthwaite'.",
        "pt": "Coluna de temperatura '{temp_var}' necessária para pet_method='thornthwaite'.",
        "es": "Columna de temperatura '{temp_var}' necesaria para pet_method='thornthwaite'.",
        "zh": "pet_method='thornthwaite' 需要温度列 '{temp_var}'。",
    },
    "spei_title": {
        "en": "Computing SPEI (Standardized Precipitation-Evapotranspiration Index)",
        "pt": "Calculando SPEI (Standardized Precipitation-Evapotranspiration Index)",
        "es": "Calculando SPEI (Standardized Precipitation-Evapotranspiration Index)",
        "zh": "正在计算 SPEI（标准化降水蒸散指数）",
    },
    "spei_computing_scale": {
        "en": "Computing {s}-month scale -> {col}...",
        "pt": "Calculando escala de {s} mês(es) -> {col}...",
        "es": "Calculando escala de {s} mes(es) -> {col}...",
        "zh": "正在计算 {s} 个月尺度 -> {col}...",
    },
    "spei_done": {
        "en": "Done: {n_rows} rows, {n_na} NA(s) in '{col1}'.",
        "pt": "Concluído: {n_rows} linhas, {n_na} NA(s) em '{col1}'.",
        "es": "Completo: {n_rows} filas, {n_na} NA(s) en '{col1}'.",
        "zh": "完成：{n_rows} 行，'{col1}' 中有 {n_na} 个缺失值。",
    },
}


# ── Message helper ─────────────────────────────────────────────────────────────

# Compile placeholder regex once (10–100× faster than re.sub in a hot loop).
_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def lcz_msg(key: str, lang: str = "en", **kwargs: Any) -> str:
    """Retrieve a localized message string.

    Parameters
    ----------
    key : str
        Key in :data:`LCZ_MESSAGES`.
    lang : str, optional
        One of ``"en"``, ``"pt"``, ``"es"``, ``"zh"``. Unknown values
        silently fall back to ``"en"``.
    **kwargs
        Named values used to substitute ``{varname}`` placeholders in the
        template. Values are coerced to ``str`` via :func:`format`.

    Returns
    -------
    str
        The rendered (or fallback) message. Markup such as ``{val ...}``,
        ``{arg ...}``, ``{fn ...}``, ``{path ...}``, ``{url ...}`` is left
        untouched so the caller can render it via its preferred rich
        printer.

    Notes
    -----
    List-shaped templates (e.g. ``[main, "i=hint"]``) mirror R's
    ``cli_abort(c("msg", "i" = "hint"))`` form. ``lcz_msg`` returns only
    the first line for plain-text callers; rich callers should introspect
    ``LCZ_MESSAGES[key][lang]`` directly to access hints.
    """
    if lang not in ("en", "pt", "es", "zh"):
        lang = "en"

    msgs = LCZ_MESSAGES.get(key)
    if msgs is None:
        return f"<missing i18n key: {key}>"

    template = msgs.get(lang, msgs["en"])
    if isinstance(template, list):
        # Render only the first line for plain-text callers; rich callers
        # should introspect ``LCZ_MESSAGES[key][lang]`` directly.
        template = template[0]

    if not kwargs:
        return template

    # Single-pass regex substitution (faster than chained gsub / str.replace).
    return _PLACEHOLDER_RE.sub(
        lambda m: str(kwargs.get(m.group(1), m.group(0))), template
    )


__all__ = ["LCZ_MESSAGES", "lcz_msg"]