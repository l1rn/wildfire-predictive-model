from src.data_loader import *
from src.config import DATA_DIR

def process_data():
    """ Data Integration """
    dem = load_static_raster(f"{DATA_DIR}/khmao_topography.tif")
    lc = load_static_raster(f"{DATA_DIR}/khmao_lc_90m.tif")
    human_mod = load_static_raster(f"{DATA_DIR}/khmao_human_mod_90m.tif")
        
    ds = load_meterological(f"{DATA_DIR}/khmao_era5.nc")
    firms = load_firms(f"{DATA_DIR}/khmao_fire_archive.csv")
    
    temp = ds["t2m"]
    temp_month = temp.mean(dim="valid_time")
    temp_month = temp_month.rio.write_crs("EPSG:4326")
    
    lc_matched = lc.rio.reproject_match(temp_month)
    human_mod_matched = human_mod.rio.reproject_match(temp_month)
    dem_matched = dem.rio.reproject_match(temp_month)
    
    print("HMD resolution:", human_mod.rio.resolution())
    print("LandCover resolution:", lc.rio.resolution())
    print("DEM resolution:", dem.rio.resolution())
    print("ERA resolution:", temp.rio.resolution())
    
    print("Matched LandCover resolution:", lc_matched.rio.resolution())
    print("Matched HMD resolution:", human_mod_matched.rio.resolution())
    print("Matched DEM resolution:", dem_matched.rio.resolution())
    
    print("temp_month Shape: ", temp_month.shape)