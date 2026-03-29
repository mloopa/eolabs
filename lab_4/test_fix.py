#!/usr/bin/env python3
"""Test the fixed get_lat_lon_grids function."""

import sys
import os
from pathlib import Path

# Apply the PROJ fix first
def find_working_proj_db():
    """Find a working PROJ database on the system."""
    possible_paths = [
        # Conda environments
        os.path.join(sys.prefix, "Library", "share", "proj"),
        os.path.join(sys.prefix, "share", "proj"),
        # GDAL bundled
        os.path.join(os.path.dirname(os.__file__), "site-packages", "osgeo", "data", "proj"),
        # System paths
        r"C:\Program Files\GDAL\proj",
        r"C:\OSGeo4W\share\proj",
    ]
    
    for path in possible_paths:
        if os.path.exists(os.path.join(path, "proj.db")):
            return path
    
    # Fallback: try to use GDAL's internal database
    try:
        from osgeo import gdal
        gdal_data = gdal.GetConfigOption("GDAL_DATA")
        if gdal_data and os.path.exists(os.path.join(gdal_data, "proj.db")):
            return gdal_data
    except:
        pass
    
    return None

if sys.platform == 'win32':
    proj_path = find_working_proj_db()
    if proj_path:
        os.environ['PROJ_DATA'] = proj_path
        os.environ['PROJ_LIB'] = proj_path
        print(f"Set PROJ_DATA to: {proj_path}")
    else:
        print("Warning: Could not find a working PROJ database")

# Now test the imports and function
try:
    import numpy as np
    from osgeo import gdal, osr
    print("✅ GDAL/OGR imported successfully")
    
    # Test the new get_lat_lon_grids function
    def get_lat_lon_grids(file_path, pixel_center=True):
        """
        Compute per-pixel longitude and latitude grids in WGS84 (EPSG:4326).

        This version uses GDAL's AutoCreateWarpedVRT to avoid PROJ database issues.
        """
        # Open the source dataset
        src_ds = gdal.Open(file_path)
        if src_ds is None:
            raise ValueError(f"Could not open: {file_path}")
        
        # Get source geotransform and dimensions
        src_gt = src_ds.GetGeoTransform()
        width, height = src_ds.RasterXSize, src_ds.RasterYSize
        
        # Create a virtual dataset warped to WGS84
        # Use GDAL's built-in WGS84 definition to avoid PROJ database issues
        wgs84_wkt = """GEOGCS["WGS 84",
            DATUM["WGS_1984",
                SPHEROID["WGS 84",6378137,298.257223563,
                    AUTHORITY["EPSG","7030"]],
                AUTHORITY["EPSG","6326"]],
            PRIMEM["Greenwich",0,
                AUTHORITY["EPSG","8901"]],
            UNIT["degree",0.01745329251994328,
                AUTHORITY["EPSG","9122"]]]"""
        
        # Create warped VRT to WGS84
        vrt_ds = gdal.AutoCreateWarpedVRT(src_ds, None, wgs84_wkt, gdal.GRA_Bilinear)
        
        if vrt_ds is None:
            # Fallback: try using the original projection if it's already geographic
            src_srs = osr.SpatialReference()
            src_srs.ImportFromWkt(src_ds.GetProjection())
            if src_srs.IsGeographic():
                # Source is already in geographic coordinates
                vrt_ds = src_ds
            else:
                raise RuntimeError("Could not create warped VRT and source is not geographic")
        
        # Get the warped geotransform
        vrt_gt = vrt_ds.GetGeoTransform()
        
        # Calculate pixel coordinates
        ox, pw, xr, oy, yr, ph = vrt_gt
        if pixel_center:
            ox += pw / 2.0
            oy += ph / 2.0
        
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        
        # Transform pixel coordinates to geographic coordinates
        lons = ox + cols * pw + rows * xr
        lats = oy + cols * yr + rows * ph
        
        # Clean up
        if vrt_ds != src_ds:
            vrt_ds = None
        src_ds = None
        
        return lons, lats
    
    print("✅ get_lat_lon_grids function defined successfully")
    
    # Look for a test file
    data_dir = Path("../data")
    hrfi_files = list(data_dir.glob("MTG_HRFI_POL_*.zip"))
    if hrfi_files:
        import zipfile
        # Extract first band to test
        with zipfile.ZipFile(hrfi_files[0], "r") as zf:
            for member in zf.namelist():
                if member.endswith(".tif"):
                    zf.extract(member, "temp_test")
                    test_file = f"temp_test/{os.path.basename(member)}"
                    break
        
        if os.path.exists(test_file):
            print(f"Testing with file: {test_file}")
            lons, lats = get_lat_lon_grids(test_file)
            print(f"✅ Success! Generated grids of shape: {lons.shape}")
            print(f"   Longitude range: {lons.min():.2f} to {lons.max():.2f}")
            print(f"   Latitude range: {lats.min():.2f} to {lats.max():.2f}")
            
            # Clean up
            os.remove(test_file)
            os.rmdir("temp_test")
        else:
            print("❌ No test file found")
    else:
        print("❌ No HRFI files found for testing")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
