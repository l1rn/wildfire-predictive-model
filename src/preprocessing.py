from src.data_loader import *
from src.config import DATA_DIR
import numpy as np

KELVIN = 273.15
def calculate_vpd(t2m_k, d2m_k) -> float:
    t_c = t2m_k - KELVIN
    d_c = d2m_k - KELVIN

    es = 610.78 * np.exp((17.2694 * t_c) / (t_c + 237.3))
    ea = 610.78 * np.exp((17.2694 * d_c) / (d_c + 237.3))
    return es - ea

def process_data():
    """ Data Integration """
    dem = load_static_raster(f"{DATA_DIR}/khmao_topography.tif")
    lc = load_static_raster(f"{DATA_DIR}/khmao_lc_90m.tif")
    human_mod = load_static_raster(f"{DATA_DIR}/khmao_human_mod_90m.tif")
        
    ds = load_meterological(f"{DATA_DIR}/khmao_era5.nc")
    firms = load_firms(f"{DATA_DIR}/khmao_fire_archive.csv")
    
    t2m_monthly = ds["t2m"].resample(valid_time="1ME").mean()
    d2p_monthly = ds["d2m"].resample(valid_time="1ME").mean()
    t2m_monthly = t2m_monthly.rio.write_crs("EPSG:4326")
    d2p_monthly = d2p_monthly.rio.write_crs("EPSG:4326")
    
    # lc_matched = lc.rio.reproject_match(temp_monthly)
    # human_mod_matched = human_mod.rio.reproject_match(temp_monthly)
    # dem_matched = dem.rio.reproject_match(temp_monthly)
    vpd_monthly = calculate_vpd(t2m_monthly, d2p_monthly)
    print(vpd_monthly.min().values)
    print(vpd_monthly.max().values)
