#!/usr/bin/env python3
"""
Extracts the WGS84 Bounding Box from the airborne hyperspectral data.
"""

import rasterio
from rasterio.warp import transform_bounds
from pathlib import Path
import sys

def get_airborne_bbox():
    # Define paths
    base_dir = Path(__file__).parent / "data" / "you-shall-not-pass" / "Obrazy lotnicze"
    bsq_path = base_dir / "221000_Odra_HS_Blok_A_008_VS_join_atm.bsq"
    
    if not bsq_path.exists():
        print(f"Error: Data file not found at {bsq_path}")
        return

    try:
        with rasterio.open(bsq_path) as src:
            # Get bounds in the local projected CRS (Poland CS2000)
            local_bounds = src.bounds
            
            # Transform to WGS84 (Longitude, Latitude)
            wgs84_bounds = transform_bounds(src.crs, 'EPSG:4326', *local_bounds)
            
            print(f"File: {bsq_path.name}")
            print(f"Local CRS: {src.crs}")
            print("-" * 30)
            print(f"Bounding Box (WGS84):")
            print(f"  Min Lon (Left):   {wgs84_bounds[0]:.6f}")
            print(f"  Min Lat (Bottom): {wgs84_bounds[1]:.6f}")
            print(f"  Max Lon (Right):  {wgs84_bounds[2]:.6f}")
            print(f"  Max Lat (Top):    {wgs84_bounds[3]:.6f}")
            print("-" * 30)
            print(f"STAC/Leaflet Format [min_lon, min_lat, max_lon, max_lat]:")
            print(list(wgs84_bounds))

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    get_airborne_bbox()
