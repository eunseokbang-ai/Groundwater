"""KNGeoid18 (NGII text grid: lat lon N per line) -> GeoTIFF geoid for Agisoft Metashape, plus the compound
coordinate system that uses it (Korea 2000 / Central Belt 2010 + KNGeoid18 height) as a .prj.

Laid out like Metashape's own geoids/egm96-15.tif, but in Korea 2000 geographic (EPSG:4737: Metashape ties a
geoid to the datum of the compound CS's horizontal part -- in EPSG:4326 it reported 'Vertical datum missing: KGD2002
(EPSG::4737) to KNGeoid18 height'): Float32 undulation N in metres, one pixel per grid
node (pixel centres on the nodes, AREA_OR_POINT=Area), nodata -32767. KNGeoid18 refers to GRS80 (Korea 2000);
checked with Metashape 2.3.2: WGS84 height 100 m at Iksan -> 75.474 m (N = 24.526 m, as heights.geoid_n).

    python scripts/kngeoid_to_geotiff.py [KNGeoid18.dat] [out folder]

Writes KNGeoid18.tif and Korea2000_CentralBelt2010_KNGeoid18.prj (default: next to the .dat).
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
DAT = ROOT / "Filed_Data" / "Geoid_Model" / "KNGeoid18.dat"
NODATA = -32767.0

# the compound CS: EPSG:5186 + a vertical CS whose datum names the geoid file (Metashape looks it up in its geoids
# folder, or among the files registered with CoordinateSystem.addGeoid)
WKT = ('COMPD_CS["Korea 2000 / Central Belt 2010 + KNGeoid18 height",'
       'PROJCS["Korea 2000 / Central Belt 2010",GEOGCS["Korea 2000",DATUM["Geocentric_datum_of_Korea",'
       'SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],'
       'AUTHORITY["EPSG","6737"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
       'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9102"]],AUTHORITY["EPSG","4737"]],'
       'PROJECTION["Transverse_Mercator",AUTHORITY["EPSG","9807"]],PARAMETER["latitude_of_origin",38],'
       'PARAMETER["central_meridian",127],PARAMETER["scale_factor",1],PARAMETER["false_easting",200000],'
       'PARAMETER["false_northing",600000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AUTHORITY["EPSG","5186"]],'
       'VERT_CS["KNGeoid18 height",VERT_DATUM["KNGeoid18",2005,EXTENSION["PROJ4_GRIDS","KNGeoid18.tif"]],'
       'UNIT["metre",1,AUTHORITY["EPSG","9001"]]]]')


def read_grid(path: Path):
    d = np.loadtxt(path)
    lats, lons = np.unique(d[:, 0]), np.unique(d[:, 1])
    N = np.full((len(lats), len(lons)), np.nan)
    N[np.searchsorted(lats, d[:, 0]), np.searchsorted(lons, d[:, 1])] = d[:, 2]
    return lats, lons, N


def main(dat: Path = DAT, out: Path | None = None) -> tuple[Path, Path]:
    out = Path(out or dat.parent)
    lats, lons, N = read_grid(dat)
    dlat, dlon = float(np.median(np.diff(lats))), float(np.median(np.diff(lons)))
    tif = out / "KNGeoid18.tif"
    img = np.where(np.isfinite(N), N, NODATA)[::-1].astype(np.float32)       # north up
    with rasterio.open(tif, "w", driver="GTiff", width=img.shape[1], height=img.shape[0], count=1, dtype="float32",
                       crs="EPSG:4737", transform=from_origin(lons[0] - dlon / 2, lats[-1] + dlat / 2, dlon, dlat),
                       nodata=NODATA, compress="deflate") as ds:
        ds.write(img, 1)
        ds.update_tags(TIFFTAG_IMAGEDESCRIPTION='VERT_DATUM["KNGeoid18",2005]', AREA_OR_POINT="Area")
    prj = out / "Korea2000_CentralBelt2010_KNGeoid18.prj"
    prj.write_text(WKT, encoding="ascii")
    print(f"{tif}  {img.shape[1]} x {img.shape[0]} nodes, {dlon * 3600:.1f}\" grid, "
          f"lon {lons[0]:.3f}-{lons[-1]:.3f}, lat {lats[0]:.3f}-{lats[-1]:.3f}, N {np.nanmin(N):.2f}-{np.nanmax(N):.2f} m")
    print(prj)
    return tif, prj


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:3]))
