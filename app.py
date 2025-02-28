import streamlit as st
from streamlit_folium import st_folium
import folium
import requests
import rasterio
from rasterio.transform import from_origin
import numpy as np
from PIL import Image
import io
from pyproj import Transformer, CRS
import mercantile
import math
import os
import torch
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import geopandas as gpd
from rasterio.plot import show
from rasterio.features import shapes
import ssl
import tempfile
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Segmentation App by Fahad Karim", layout="wide")

# Create tabs
tab1, tab2 = st.tabs(["🌍 GeoTIFF Downloader", "🔍 Segmentation"])

# --- HELPER FUNCTIONS ---
def get_satellite_tile(x, y, z):
    """Fetch satellite tile from Google Satellite"""
    url = f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    return None

def get_mercator_resolution(zoom):
    """Get the resolution (meters per pixel) for a given zoom level in Web Mercator"""
    return 156543.03392 * math.cos(0) / (2 ** zoom)

def tile_bounds_to_mercator(tile, resolution):
    """Convert tile coordinates to Web Mercator bounds"""
    tile_size = 256
    origin_x = -20037508.342789244
    origin_y = 20037508.342789244
    
    min_x = origin_x + tile.x * tile_size * resolution
    max_y = origin_y - tile.y * tile_size * resolution
    max_x = min_x + tile_size * resolution
    min_y = max_y - tile_size * resolution
    
    return min_x, min_y, max_x, max_y

def create_geotiff(tiles, images_dict, output_path, zoom):
    """Create a GeoTIFF from tiles and their images"""
    resolution = get_mercator_resolution(zoom)
    
    y_values = sorted(set(tile.y for tile in tiles))
    x_ranges = {y: (min(t.x for t in tiles if t.y == y), max(t.x for t in tiles if t.y == y)) for y in y_values}
    
    width = sum(x_ranges[y][1] - x_ranges[y][0] + 1 for y in y_values) * 256 // len(y_values)
    height = len(y_values) * 256
    
    min_tile_x = min(x_ranges[y][0] for y in y_values)
    min_tile_y = min(y_values)
    max_tile_x = max(x_ranges[y][1] for y in y_values)
    max_tile_y = max(y_values)
    
    min_x, _, _, max_y = tile_bounds_to_mercator(mercantile.Tile(min_tile_x, min_tile_y, zoom), resolution)
    _, min_y, max_x, _ = tile_bounds_to_mercator(mercantile.Tile(max_tile_x, max_tile_y, zoom), resolution)
    
    transform = from_origin(min_x, max_y, resolution, resolution)
    
    full_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3,
        dtype=np.uint8,
        crs=CRS.from_epsg(3857),
        transform=transform,
    ) as dst:
        for i, y in enumerate(y_values):
            row_images = []
            for x in range(x_ranges[y][0], x_ranges[y][1] + 1):
                tile = mercantile.Tile(x, y, zoom)
                if tile in images_dict:
                    row_images.append(np.array(images_dict[tile]))
            
            if row_images:
                row = np.hstack(row_images)
                y_start = i * 256
                y_end = (i + 1) * 256
                x_end = min(width, row.shape[1])
                full_image[y_start:y_end, 0:x_end] = row[:, 0:x_end]
        
        for i in range(3):
            dst.write(full_image[:, :, i], i+1)

def download_sam_checkpoint(model_type):
    """Download SAM model checkpoint"""
    checkpoint_urls = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    }
    
    models_dir = "sam_models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = f"{models_dir}/sam_{model_type}.pth"
    
    if not os.path.exists(model_path):
        try:
            response = requests.get(checkpoint_urls[model_type], stream=True, verify=False)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            progress_bar = st.progress(0)
            
            with open(model_path, 'wb') as f:
                downloaded = 0
                for data in response.iter_content(block_size):
                    f.write(data)
                    downloaded += len(data)
                    if total_size > 0:
                        progress = int(downloaded / total_size * 100)
                        progress_bar.progress(min(progress/100, 1.0))
            return model_path
        except Exception as e:
            st.error(f"Download failed: {str(e)}")
            return None
    return model_path

def generate_masks_using_automatic_mask_generator(source, model_type, checkpoint_path, **kwargs):
    """Generate segmentation masks using SAM"""
    with rasterio.open(source) as src:
        image = src.read()
        transform = src.transform
        crs = src.crs
        
        if image.shape[0] >= 3:
            rgb_image = np.stack([image[i] for i in range(3)], axis=-1)
        else:
            rgb_image = np.stack([image[0]]*3, axis=-1)
        
        rgb_image = ((rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image)) * 255).astype(np.uint8)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)
        mask_generator = SamAutomaticMaskGenerator(sam, **kwargs)
        masks = mask_generator.generate(rgb_image)
        
        mask_image = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
        for i, mask_data in enumerate(masks):
            mask = mask_data["segmentation"]
            mask_image[mask] = i + 1
        
        with rasterio.open(
            "segmentation_masks.tif",
            'w',
            driver='GTiff',
            height=mask_image.shape[0],
            width=mask_image.shape[1],
            count=1,
            dtype=mask_image.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(mask_image, 1)
        
        return mask_image, "segmentation_masks.tif", masks

def sam_masks_to_geojson(masks, transform, crs, output_path):
    """Convert SAM masks directly to GeoJSON"""
    if not masks:
        st.warning("No masks to convert to GeoJSON")
        return None
    
    features = []
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        mask_shapes = list(shapes(np.array(mask, dtype=np.int16), mask=mask, transform=transform))
        
        for shape, value in mask_shapes:
            features.append({
                'type': 'Feature',
                'properties': {
                    'id': i + 1,
                    'area': float(mask_data.get('area', 0)),
                    'score': float(mask_data.get('predicted_iou', 0)),
                    'stability': float(mask_data.get('stability_score', 0)),
                },
                'geometry': shape
            })
    
    geojson_data = {'type': 'FeatureCollection', 'features': features}
    gdf = gpd.GeoDataFrame.from_features(geojson_data['features'], crs=crs)
    gdf.to_file(output_path, driver="GeoJSON")
    return gdf

# --- TAB 1: GEOIFF DOWNLOADER ---
with tab1:
    st.title("Satellite Imagery Downloader")
    st.warning("")
    
    m = folium.Map(
        location=[0, 0],
        zoom_start=2,
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite'
    )
    
    draw = folium.plugins.Draw(
        export=True,
        position='topleft',
        draw_options={
            'rectangle': True,
            'polygon': False,
            'polyline': False,
            'circle': False,
            'marker': False,
            'circlemarker': False
        }
    )
    m.add_child(draw)
    
    map_data = st_folium(m, width=800, height=600)
    
    if 'drawn_features' not in st.session_state:
        st.session_state.drawn_features = None
    if 'confirmed_large_download' not in st.session_state:
        st.session_state.confirmed_large_download = False
    
    zoom = st.slider("Select zoom level", 10, 21, 17)
    
    if map_data and 'all_drawings' in map_data and map_data['all_drawings']:
        st.session_state.drawn_features = map_data['all_drawings'][-1]
    
    if st.session_state.get('drawn_features'):
        geometry = st.session_state.drawn_features['geometry']
        coordinates = geometry['coordinates'][0]
        bounds = [
            min(c[0] for c in coordinates),
            min(c[1] for c in coordinates),
            max(c[0] for c in coordinates),
            max(c[1] for c in coordinates)
        ]
        
        tiles = list(mercantile.tiles(bounds[0], bounds[1], bounds[2], bounds[3], zoom))
        st.write(f"Selected area requires {len(tiles)} tiles")
        
        if len(tiles) > 500:
            st.warning("Large area selected! This may take time.")
            if st.button("Confirm Download"):
                st.session_state.confirmed_large_download = True
        else:
            st.session_state.confirmed_large_download = True
        
        if st.session_state.get('confirmed_large_download', False):
            if st.button("Download GeoTIFF"):
                with st.spinner("Downloading..."):
                    images = {}
                    for tile in tiles:
                        images[tile] = get_satellite_tile(tile.x, tile.y, tile.z)
                    
                    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_file:
                        create_geotiff(tiles, images, tmp_file.name, zoom)
                        st.session_state.downloaded_image = tmp_file.name
                        st.success(f"Download complete! File saved to: {tmp_file.name}")
    
    if 'downloaded_image' in st.session_state:
        with rasterio.open(st.session_state.downloaded_image) as src:
            st.write("Preview of downloaded GeoTIFF:")
            fig, ax = plt.subplots()
            show(src, ax=ax)
            ax.set_title("Downloaded GeoTIFF Preview")
            st.pyplot(fig)
            
            with open(st.session_state.downloaded_image, "rb") as f:
                st.download_button(
                    "Download GeoTIFF File",
                    data=f,
                    file_name="downloaded_image.tif",
                    mime="image/tiff"
                )

# --- TAB 2: SEGMENTATION ---
# --- TAB 2: SEGMENTATION ---
with tab2:
    st.title("Geospatial Segmentation by Fahad Karim")
    
    # Model configuration section
    with st.sidebar.expander("Model Configuration", expanded=True):
        model_type = st.selectbox("Model", ["vit_h", "vit_l", "vit_b"], index=1, key="model_type")
        models_dir = "sam_models"
        checkpoint_path = f"{models_dir}/sam_{model_type}.pth"
        
        # Model management tabs
        tab_download, tab_upload, tab_local = st.tabs(["Download", "Upload", "Local Path"])
        
        with tab_download:
            if st.button("Download Model", key="download_btn"):
                ssl._create_default_https_context = ssl._create_unverified_context
                downloaded_path = download_sam_checkpoint(model_type)
                if downloaded_path:
                    checkpoint_path = downloaded_path
                    st.success(f"Model downloaded to: {checkpoint_path}")
        
        with tab_upload:
            uploaded_model = st.file_uploader("Upload .pth file", type="pth")
            if uploaded_model:
                try:
                    os.makedirs(models_dir, exist_ok=True)
                    with open(checkpoint_path, "wb") as f:
                        f.write(uploaded_model.getbuffer())
                    st.success(f"Model saved to: {checkpoint_path}")
                    # Force refresh to update model status
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Upload failed: {str(e)}")
        
        with tab_local:
            custom_path = st.text_input("Custom model path", checkpoint_path)
            if os.path.exists(custom_path):
                checkpoint_path = custom_path
                st.success(f"Using model: {checkpoint_path}")
            elif custom_path:
                st.error("File not found")

    # Check model status
    model_exists = os.path.exists(checkpoint_path)
    if model_exists:
        model_size = round(os.path.getsize(checkpoint_path) / (1024*1024), 2)
        st.sidebar.success(f"Model ready ({model_size} MB)")
    else:
        st.sidebar.warning("Model not available")

    # Segmentation parameters
    with st.sidebar.expander("Segmentation Parameters"):
        col1, col2 = st.columns(2)
        with col1:
            points_per_side = st.slider("Points Per Side", 4, 64, 32)
            pred_iou_thresh = st.slider("IOU Threshold", 0.0, 1.0, 0.75)
        with col2:
            stability_score_thresh = st.slider("Stability Threshold", 0.0, 1.0, 0.8)
            min_mask_region_area = st.slider("Min Mask Area", 0, 1000, 50)

    # File upload and processing
    uploaded_file = st.file_uploader("Upload GeoTIFF", type=["tif", "tiff"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            image_path = tmp_file.name
        
        if st.button("Run Segmentation"):
            if not model_exists:
                st.error("Model not available. Please download/upload a valid model first.")
            else:
                with st.spinner("Processing..."):
                    mask_image, mask_path, masks = generate_masks_using_automatic_mask_generator(
                        image_path,
                        model_type,
                        checkpoint_path,
                        points_per_side=points_per_side,
                        pred_iou_thresh=pred_iou_thresh,
                        stability_score_thresh=stability_score_thresh,
                        min_mask_region_area=min_mask_region_area
                    )
                    
                    # Results display
                    st.header("Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Mask Preview")
                        fig, ax = plt.subplots()
                        ax.imshow(mask_image)
                        ax.axis('off')
                        st.pyplot(fig)
                        
                        with open(mask_path, "rb") as f:
                            st.download_button(
                                "Download Mask",
                                data=f,
                                file_name="segmentation_mask.tif",
                                mime="image/tiff"
                            )
                    
                    with col2:
                        st.subheader("GeoJSON Output")
                        output_geojson = "segmentation.geojson"
                        gdf = sam_masks_to_geojson(masks, rasterio.open(image_path).transform, "EPSG:3857", output_geojson)
                        
                        if gdf is not None:
                            st.dataframe(gdf.head())
                            
                            with open(output_geojson, "rb") as f:
                                st.download_button(
                                    "Download GeoJSON",
                                    data=f,
                                    file_name=output_geojson,
                                    mime="application/json"
                                )