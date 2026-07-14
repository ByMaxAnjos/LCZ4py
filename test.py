# Advanced usage with all new features
import asyncio
from LCZ4py.general.lcz_get_map import lcz_get_map
from LCZ4py.general.lcz_plot_map import lcz_plot_map
from LCZ4py.general.lcz_plot_parameters import lcz_plot_parameters
from LCZ4py.general.lcz_cal_area import lcz_cal_area
from LCZ4py.general.lcz_get_parameters import lcz_get_parameters


# 1. Download with COG streaming (only downloads needed pixels)
map_path = lcz_get_map(city="Berlin", isave_map=True)


# 2. Interactive map with WebGL
result = lcz_plot_map(map_path, use_webgl=True, use_duckdb=True)
result.fig.show()  # Opens interactive HTML in browser

# 3. Calculate areas with DuckDB
area_result = lcz_cal_area(
    map_path, 
    plot_type="sunburst",  # New hierarchical chart!
    use_duckdb=True
)
area_result.fig.show()

# 4. Get parameters with vectorized lookup
params = lcz_get_parameters(map_path, istack=True, isave=True)


# 5. Plot parameters with Datashader for massive rasters
figures = lcz_plot_parameters(
    params,
    iselect=["SVFmean", "AHmean", "BSFmean"],
    use_datashader=True,
    isave=True,
    save_extension="html"
)
for fig in figures:
    fig.show()