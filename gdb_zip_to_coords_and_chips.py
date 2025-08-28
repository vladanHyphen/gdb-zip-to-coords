# gdb_zip_to_coords_and_chips.py
# 1) Upload ZIP with .gdb → choose layer → extract lon/lat to Excel/CSV/GeoJSON
# 2) Optional: export original shapes as GeoJSON/Shapefile
# 3) Optional: generate satellite chips + chips.csv (Esri World Imagery XYZ)

import io, os, math, time, zipfile, tempfile
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, mapping

# ---------- Engines (Fiona or Pyogrio) ----------
ENGINE = None
_HAS_FIONA = False
_HAS_PYOGRIO = False
try:
    import fiona  # type: ignore
    _HAS_FIONA = True
    ENGINE = "fiona"
except Exception:
    pass
try:
    import pyogrio  # type: ignore
    _HAS_PYOGRIO = True
    if ENGINE is None:
        ENGINE = "pyogrio"
except Exception:
    pass

def require_engine():
    if ENGINE is None:
        st.error("Neither Fiona nor Pyogrio is installed. Install one of them:\n\n"
                 "  pip install fiona\n"
                 "  # or\n"
                 "  pip install pyogrio")
        st.stop()

def list_layers_any(path: str):
    if _HAS_FIONA:
        from fiona import listlayers
        return listlayers(path)
    return pyogrio.list_layers(path)

def read_file_any(path: str, layer: str):
    engines = []
    if ENGINE == "fiona":
        engines = ["fiona"] + (["pyogrio"] if _HAS_PYOGRIO else [])
    else:
        engines = ["pyogrio"] + (["fiona"] if _HAS_FIONA else [])
    last_err = None
    for eng in engines:
        try:
            return gpd.read_file(path, layer=layer, engine=eng)
        except Exception as e:
            last_err = e
    raise last_err

def to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        st.error("Layer has no CRS. Please define a CRS in your GIS first.")
        return gdf
    try:
        if str(gdf.crs).lower() in ("epsg:4326", "wgs84", "wgs 84"):
            return gdf
    except Exception:
        pass
    return gdf.to_crs(4326)

def sanitize_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.select_dtypes(include=["datetimetz"]).columns:
        out[c] = out[c].dt.tz_convert("UTC").dt.tz_localize(None)
    def clean_val(v):
        if isinstance(v, (int, float, str, type(None), pd.Timestamp)):
            return v
        try:
            return str(v)
        except Exception:
            return repr(v)
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].map(clean_val)
    return out

# ---------- Optional shapefile exporter ----------
def shapefile_zip_from_gdf(gdf: gpd.GeoDataFrame, layer_name="layer") -> io.BytesIO:
    import fiona
    from fiona.crs import CRS
    from tempfile import TemporaryDirectory
    schema_geom = "Polygon"
    # detect geometry
    geom_types = set(gdf.geom_type.dropna().unique())
    if geom_types <= {"Point"}:
        schema_geom = "Point"
    elif geom_types <= {"LineString", "MultiLineString"}:
        schema_geom = "LineString"
    else:
        schema_geom = "Polygon"

    # make attribute schema (simple types only)
    schema_props = {}
    for c in gdf.columns:
        if c == "geometry": continue
        dt = gdf[c].dtype
        if pd.api.types.is_integer_dtype(dt):   schema_props[c] = "int"
        elif pd.api.types.is_float_dtype(dt):   schema_props[c] = "float"
        else:                                   schema_props[c] = "str"

    out_zip = io.BytesIO()
    with TemporaryDirectory() as tmpd:
        shp_path = Path(tmpd) / f"{layer_name}.shp"
        with fiona.open(shp_path, "w", driver="ESRI Shapefile",
                        crs=CRS.from_epsg(4326).to_wkt(),
                        schema={"geometry": schema_geom, "properties": schema_props},
                        encoding="utf-8") as dst:
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty: continue
                props = {k: (None if pd.isna(row[k]) else row[k]) for k in gdf.columns if k != "geometry"}
                dst.write({"type":"Feature","geometry":mapping(geom),"properties":props})
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
            for ext in (".shp",".shx",".dbf",".prj",".cpg"):
                fp = Path(tmpd) / f"{layer_name}{ext}"
                if fp.exists(): z.write(fp, arcname=f"{layer_name}{ext}")
    out_zip.seek(0)
    return out_zip

# ---------- Satellite tile chipper (no ArcGIS SDK required) ----------
import requests
from PIL import Image

SATELLITE_SOURCES = {
    "Esri (REST)":   "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "Esri (WMTS)":   "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/WMTS/tile/1.0.0/World_Imagery/default/GoogleMapsCompatible/{z}/{y}/{x}.jpg",
    "Esri (Server)": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "Carto Streets": "https://tile.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
}
TILE_SIZE = 256
R = 6378137.0
MAX_LAT = 85.05112878

def lonlat_to_merc(lon, lat):
    lat = max(min(lat, MAX_LAT), -MAX_LAT)
    x = R * math.radians(lon)
    y = R * math.log(math.tan(math.pi/4.0 + math.radians(lat)/2.0))
    return x, y

def merc_to_lonlat(x, y):
    lon = math.degrees(x / R)
    lat = math.degrees(2.0 * math.atan(math.exp(y / R)) - math.pi/2.0)
    return lon, lat

def chip_bbox_wgs84(lon, lat, chip_size_m):
    x, y = lonlat_to_merc(lon, lat)
    h = chip_size_m / 2.0
    return merc_to_lonlat(x - h, y - h) + merc_to_lonlat(x + h, y + h)  # lon_min, lat_min, lon_max, lat_max

def world_size(z): return TILE_SIZE * (2 ** z)

def lonlat_to_world_px(lon, lat, z):
    s = float(world_size(z))
    x = (lon + 180.0) / 360.0 * s
    siny = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)) * s
    return x, y

def pick_zoom_for_target_res(lat_deg, target_m_per_px, z_min=10, z_max=19):
    denom = TILE_SIZE * target_m_per_px
    if denom <= 0: return 19
    raw = math.log2((math.cos(math.radians(lat_deg)) * 2 * math.pi * R) / denom)
    z = int(math.floor(raw))
    return max(z_min, min(z, z_max))

def fetch_tile(session: requests.Session, url_template: str, z, x, y, retries=3, timeout=20):
    url = url_template.format(z=z, x=x, y=y)
    last_e = None
    for _ in range(retries):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 200:
                return Image.open(io.BytesIO(r.content)).convert("RGB")
            else:
                last_e = RuntimeError(f"HTTP {r.status_code}")
        except Exception as e:
            last_e = e
        time.sleep(0.5)
    raise last_e or RuntimeError("tile fetch failed")

def mosaic_and_crop(lon_min, lat_min, lon_max, lat_max, z, source_url) -> Image.Image:
    x0, y1 = lonlat_to_world_px(lon_min, lat_min, z)
    x1, y0 = lonlat_to_world_px(lon_max, lat_max, z)
    left, right = min(x0, x1), max(x0, x1)
    top, bottom = min(y0, y1), max(y0, y1)
    tx0 = int(math.floor(left / TILE_SIZE))
    tx1 = int(math.floor((right - 1e-6) / TILE_SIZE))
    ty0 = int(math.floor(top / TILE_SIZE))
    ty1 = int(math.floor((bottom - 1e-6) / TILE_SIZE))
    width_px = int(math.ceil(right - left))
    height_px = int(math.ceil(bottom - top))
    canvas = Image.new("RGB", ((tx1 - tx0 + 1) * TILE_SIZE, (ty1 - ty0 + 1) * TILE_SIZE), (0,0,0))
    with requests.Session() as s:
        for ty in range(ty0, ty1+1):
            for tx in range(tx0, tx1+1):
                try:
                    tile = fetch_tile(s, source_url, z, tx, ty)
                    canvas.paste(tile, ((tx - tx0) * TILE_SIZE, (ty - ty0) * TILE_SIZE))
                except Exception:
                    pass
    crop_left = int(round(left - tx0 * TILE_SIZE))
    crop_top = int(round(top - ty0 * TILE_SIZE))
    crop = canvas.crop((crop_left, crop_top, crop_left + width_px, crop_top + height_px))
    return crop

# ---------- UI ----------
st.set_page_config(page_title="📦 GDB → Coordinates → Chips", layout="wide")
st.title("📦 GDB → Coordinates → Chips")

require_engine()

uploaded_zip = st.file_uploader("Upload a ZIP that contains your File Geodatabase (.gdb)", type="zip")
if not uploaded_zip:
    st.info("Upload a ZIP with a `.gdb` folder inside (File Geodatabase).")
    st.stop()

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    zip_path = tmpdir / "uploaded.zip"
    zip_path.write_bytes(uploaded_zip.read())

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(tmpdir)

    gdbs = [p for p in tmpdir.rglob("*.gdb") if p.is_dir()]
    if not gdbs:
        st.error("No `.gdb` folder found in the ZIP.")
        st.stop()

    # If multiple GDBs, let user pick
    if len(gdbs) > 1:
        gdb_choice = st.selectbox("Select a Geodatabase", [str(p) for p in gdbs])
        gdb_path = Path(gdb_choice)
    else:
        gdb_path = gdbs[0]
        st.success(f"Found geodatabase: {gdb_path.name} (engine: {ENGINE})")

    # List layers
    try:
        layers = list_layers_any(str(gdb_path))
    except Exception as e:
        st.error(f"Failed to list layers: {e}")
        st.stop()

    layer = st.selectbox("Select layer", layers)

    mode = st.radio("Coordinate mode for non-point geometries",
                    ["centroid", "bbox center"], horizontal=True)

    # Optional: show raw count/fields
    if st.checkbox("Show layer field schema"):
        try:
            sample = read_file_any(str(gdb_path), layer=layer).head(1)
            st.write(sample.dtypes)
        except Exception as e:
            st.info(f"Schema preview skipped: {e}")

    # Process button
    if st.button("Extract coordinates"):
        try:
            gdf = read_file_any(str(gdb_path), layer=layer)
        except Exception as e:
            st.error(f"Failed to read layer: {e}")
            st.stop()

        if gdf.empty:
            st.warning("Layer has no features.")
            st.stop()

        # Represent as points if not already points
        if not (gdf.geom_type == "Point").all():
            try:
                gdfp = gdf.to_crs(3857)
            except Exception:
                gdfp = gdf
            if mode == "centroid":
                rep = gdfp.geometry.centroid
            else:
                b = gdfp.geometry.bounds
                rep = gpd.points_from_xy((b.minx + b.maxx)/2.0, (b.miny + b.maxy)/2.0, crs=gdfp.crs)
            gdf = gpd.GeoDataFrame(gdf.drop(columns="geometry"), geometry=rep, crs=gdfp.crs)

        # Reproject to WGS84
        gdf = to_wgs84(gdf)
        gdf["longitude"] = gdf.geometry.x
        gdf["latitude"]  = gdf.geometry.y

        st.subheader("Preview (first 25)")
        st.dataframe(gdf.drop(columns="geometry").head(25))

        # Downloads
        tbl = sanitize_for_excel(gdf.drop(columns="geometry").copy())
        csv_bytes = tbl.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv_bytes, "coordinates.csv", "text/csv")

        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            tbl.to_excel(writer, index=False, sheet_name="coords")
        st.download_button("⬇️ Download Excel", data=excel_buf.getvalue(),
                           file_name="coordinates.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # GeoJSON (points)
        try:
            pts_geojson = gdf.to_json()
            st.download_button("⬇️ Download Points GeoJSON",
                               data=pts_geojson.encode("utf-8"),
                               file_name="coordinates.geojson",
                               mime="application/geo+json")
        except Exception as e:
            st.info(f"GeoJSON (points) export skipped: {e}")

        st.divider()
        st.subheader("Optional: Export original shapes (not just points)")

        # Reload original layer in WGS84 for shape export
        try:
            full_gdf = to_wgs84(read_file_any(str(gdb_path), layer=layer))
        except Exception as e:
            full_gdf = None
            st.info(f"Original layer reload skipped: {e}")

        colg1, colg2 = st.columns(2)
        if full_gdf is not None:
            try:
                # GeoJSON (full shapes)
                full_geojson = full_gdf.to_json()
                colg1.download_button("⬇️ Full Shapes GeoJSON",
                                      data=full_geojson.encode("utf-8"),
                                      file_name="layer_full.geojson",
                                      mime="application/geo+json")
            except Exception as e:
                colg1.info(f"GeoJSON export skipped: {e}")
            try:
                shp_zip = shapefile_zip_from_gdf(full_gdf, layer_name="layer")
                colg2.download_button("⬇️ Full Shapes Shapefile (.zip)",
                                      data=shp_zip.getvalue(),
                                      file_name="layer_shapefile.zip",
                                      mime="application/zip")
            except Exception as e:
                colg2.info(f"Shapefile export skipped: {e}")

        st.divider()
        st.subheader("Optional: Generate satellite chips + chips.csv")

        with st.form("chips_form"):
            chip_size_m = st.number_input("Chip size (m)", min_value=16, value=64, step=8)
            pixels      = st.number_input("Output size (pixels)", min_value=64, value=256, step=64)
            zoom        = st.number_input("Force zoom (10..19, blank for auto)", min_value=10, max_value=19, value=19, step=1)
            source_name = st.selectbox("Tile source",
                                       list(SATELLITE_SOURCES.keys()),
                                       index=2)  # Esri (Server)
            out_dir     = st.text_input("Output base folder", r"C:\GIF\roof_dataset")
            overwrite   = st.checkbox("Overwrite existing images", value=True)
            clear_split = st.checkbox("Clear existing data/val split", value=True)
            go = st.form_submit_button("Generate chips")

        if go:
            if not out_dir:
                st.error("Please provide an output base folder path.")
                st.stop()

            # Build coords array from table
            coord_rows = tbl[["latitude","longitude"]].dropna().to_numpy()
            if coord_rows.size == 0:
                st.error("No valid latitude/longitude rows.")
                st.stop()

            out_dir_p = Path(out_dir)
            img_dir = out_dir_p / "data" / "train" / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            if clear_split:
                val_dir = out_dir_p / "data" / "val"
                if val_dir.exists():
                    import shutil
                    shutil.rmtree(val_dir, ignore_errors=True)

            # Auto or forced zoom
            mean_lat = float(np.mean(coord_rows[:,0]))
            target_res = float(chip_size_m) / float(pixels)
            z = int(zoom) if zoom else pick_zoom_for_target_res(mean_lat, target_res)
            z = max(10, min(19, z))

            st.info(f"Using zoom z={z}  (target ~{target_res:.2f} m/px at lat {mean_lat:.3f}) — source={source_name}")
            src_url = SATELLITE_SOURCES[source_name]

            chips_meta: List[dict] = []
            prog = st.progress(0.0, text="Fetching tiles…")
            total = len(coord_rows)
            for i, (lat, lon) in enumerate(coord_rows, start=1):
                lon_min, lat_min, lon_max, lat_max = chip_bbox_wgs84(float(lon), float(lat), chip_size_m)
                try:
                    crop = mosaic_and_crop(lon_min, lat_min, lon_max, lat_max, z, src_url)
                except Exception as e:
                    st.warning(f"[skip] ({lat:.6f},{lon:.6f}) tile fetch failed: {e}")
                    prog.progress(i/total, text=f"{i}/{total}")
                    continue
                chip_img = crop.resize((int(pixels), int(pixels)), Image.BILINEAR)
                fname = f"{float(lat):.6f}_{float(lon):.6f}.png"
                out_path = img_dir / fname
                if (not out_path.exists()) or overwrite:
                    chip_img.save(out_path)
                chips_meta.append({
                    "file": str(out_path),
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "chip_size_m": float(chip_size_m),
                    "pixels": int(pixels),
                    "zoom": int(z),
                    "source": source_name
                })
                prog.progress(i/total, text=f"{i}/{total}")

            if not chips_meta:
                st.error("No chips were written. Check connectivity or parameters.")
                st.stop()

            chips_csv = out_dir_p / "chips.csv"
            pd.DataFrame(chips_meta).to_csv(chips_csv, index=False)
            st.success(f"✅ Wrote {len(chips_meta)} chips to: {img_dir}")
            st.write(f"CSV: `{chips_csv}`")

            # Small sample preview grid
            try:
                import base64
                from PIL import Image
                thumbs = []
                for rec in chips_meta[:12]:
                    im = Image.open(rec["file"]).resize((96,96))
                    buf = io.BytesIO(); im.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                    thumbs.append(f'<img src="data:image/png;base64,{b64}" style="margin:4px;border:1px solid #ddd">')
                st.markdown("**Sample chips:**<br>" + "".join(thumbs), unsafe_allow_html=True)
            except Exception:
                pass

st.caption("Tip: after chips are generated, you can run your inference script on `chips.csv`.")
