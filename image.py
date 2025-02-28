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
from datetime import datetime

# Set page config
st.set_page_config(page_title="Google Satellite Imagery Downloader", layout="wide")

# Google Maps Satellite tile URL
GOOGLE_SATELLITE_URL = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"

def get_satellite_tile(x, y, z):
    """Fetch satellite tile from Google Satellite"""
    url = GOOGLE_SATELLITE_URL.format(x=x, y=y, z=z)
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    return None

def get_mercator_resolution(zoom):
    """Get the resolution (meters per pixel) for a given zoom level in Web Mercator"""
    return 156543.03392 * math.cos(0) / (2 ** zoom)

def tile_bounds_to_mercator(tile, resolution):
    """Convert tile coordinates to Web Mercator bounds"""
    # Get the mercator coordinates for the tile
    tile_size = 256  # Standard tile size
    origin_x = -20037508.342789244  # Web Mercator origin X
    origin_y = 20037508.342789244   # Web Mercator origin Y
    
    # Calculate mercator coordinates
    min_x = origin_x + tile.x * tile_size * resolution
    max_y = origin_y - tile.y * tile_size * resolution
    max_x = min_x + tile_size * resolution
    min_y = max_y - tile_size * resolution
    
    return min_x, min_y, max_x, max_y

def create_geotiff(tiles, images_dict, output_path, zoom):
    """
    Create a GeoTIFF from tiles and their images
    """
    # Calculate resolution at this zoom level
    resolution = get_mercator_resolution(zoom)
    
    # Get unique y values and x ranges
    y_values = sorted(set(tile.y for tile in tiles))
    x_ranges = {}
    for y in y_values:
        x_tiles = [tile for tile in tiles if tile.y == y]
        x_ranges[y] = (min(tile.x for tile in x_tiles), max(tile.x for tile in x_tiles))
    
    # Calculate overall dimensions
    width = sum(x_ranges[y][1] - x_ranges[y][0] + 1 for y in y_values) * 256 // len(y_values)
    height = len(y_values) * 256
    
    # Calculate overall bounds in Web Mercator
    min_tile_x = min(x_ranges[y][0] for y in y_values)
    min_tile_y = min(y_values)
    max_tile_x = max(x_ranges[y][1] for y in y_values)
    max_tile_y = max(y_values)
    
    # Get mercator bounds
    min_x, _, _, max_y = tile_bounds_to_mercator(mercantile.Tile(min_tile_x, min_tile_y, zoom), resolution)
    _, min_y, max_x, _ = tile_bounds_to_mercator(mercantile.Tile(max_tile_x, max_tile_y, zoom), resolution)
    
    # Create combined image
    full_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Transform using the top-left origin (Web Mercator uses top-left origin for tiles)
    transform = from_origin(min_x, max_y, resolution, resolution)
    
    # Create GeoTIFF with the exact tile resolution
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3,
        dtype=np.uint8,
        crs=CRS.from_epsg(3857),  # Web Mercator
        transform=transform,
    ) as dst:
        # Fill with image data
        for i, y in enumerate(y_values):
            row_images = []
            for x in range(x_ranges[y][0], x_ranges[y][1] + 1):
                tile = mercantile.Tile(x, y, zoom)
                if tile in images_dict:
                    row_images.append(images_dict[tile])
            
            if row_images:
                row = np.hstack(row_images)
                # Extract the row slice from the full image
                y_start = i * 256
                y_end = (i + 1) * 256
                x_end = min(width, row.shape[1])
                # Insert the row into the full image
                full_image[y_start:y_end, 0:x_end] = row[:, 0:x_end]
        
        # Write the channels
        for i in range(3):
            dst.write(full_image[:, :, i], i+1)

def main():
    st.title("Google Satellite Imagery Downloader")
    st.caption("Download high-resolution satellite imagery from Google Maps")
    
    # Add disclaimer
    st.warning("Please ensure you comply with Google's Terms of Service when using this tool. This tool is for educational purposes only.")
    
    # Initialize session state
    if 'drawn_features' not in st.session_state:
        st.session_state.drawn_features = None
    if 'confirmed_large_download' not in st.session_state:
        st.session_state.confirmed_large_download = False
    
    # Create map with Google Satellite basemap
    m = folium.Map(
        location=[0, 0],
        zoom_start=2,
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite'
    )
    
    # Add drawing controls
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
    
    # Display map
    map_data = st_folium(m, width=800, height=600)
    
    # Update session state with drawn features
    if map_data is not None and 'all_drawings' in map_data:
        if map_data['all_drawings']:
            st.session_state.drawn_features = map_data['all_drawings'][-1]
    
    # Zoom level selector
    zoom = st.slider("Select zoom level (higher = more detailed)", 
                    min_value=10, 
                    max_value=21, 
                    value=17,
                    help="Google Satellite imagery typically supports up to zoom level 21 in most areas")
    
    # Add information about maximum size
    st.info("For best results, select a small area with a high zoom level. Larger areas may take longer to process.")
    
    # First check if a rectangle has been drawn
    if st.session_state.drawn_features is None:
        st.warning("Please draw a rectangle on the map before downloading")
        download_disabled = True
    else:
        download_disabled = False
        
    # Extract bounds and calculate tiles (if a rectangle is drawn)
    tiles = []
    if not download_disabled:
        try:
            # Extract bounds from drawn features
            geometry = st.session_state.drawn_features['geometry']
            coordinates = geometry['coordinates'][0]
            bounds = [
                min(coord[0] for coord in coordinates),  # min longitude
                min(coord[1] for coord in coordinates),  # min latitude
                max(coord[0] for coord in coordinates),  # max longitude
                max(coord[1] for coord in coordinates)   # max latitude
            ]
            
            # Calculate required tiles
            tiles = list(mercantile.tiles(
                bounds[0], bounds[1], bounds[2], bounds[3], zoom
            ))
            
            # Show tile count information
            st.write(f"Area selected: {len(tiles)} tiles at zoom level {zoom}")
            
            # Check for large downloads
            if len(tiles) > 500:
                st.warning(f"You've selected {len(tiles)} tiles, which is a large area. This may take a while to download. Consider selecting a smaller area or lower zoom level.")
                
                # Use columns to place buttons side by side
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Proceed with download"):
                        st.session_state.confirmed_large_download = True
                with col2:
                    if st.button("Cancel"):
                        st.session_state.confirmed_large_download = False
                        st.rerun()
        except Exception as e:
            st.error(f"Error calculating tiles: {str(e)}")
            download_disabled = True
    
    # Main download button - only show if we haven't already confirmed a large download
    if not st.session_state.confirmed_large_download and not download_disabled:
        if len(tiles) <= 500:  # Only show the download button directly for small areas
            if st.button("Download Selected Area"):
                start_download(tiles, zoom)
    
    # If user confirmed a large download, start the process
    if st.session_state.confirmed_large_download and tiles:
        start_download(tiles, zoom)
        # Reset the confirmation flag after starting the download
        st.session_state.confirmed_large_download = False

def start_download(tiles, zoom):
    """Handle the download process"""
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Download tiles and store in dictionary
        images_dict = {}
        total_tiles = len(tiles)
        
        for idx, tile in enumerate(tiles):
            status_text.text(f"Downloading tile {idx+1} of {total_tiles}")
            progress_bar.progress((idx + 1) / total_tiles)
            
            img = get_satellite_tile(tile.x, tile.y, zoom)
            if img is None:
                st.warning(f"Failed to download tile at {tile.x}, {tile.y}, {zoom}")
                continue
            
            # Convert to numpy array
            img_array = np.array(img)
            images_dict[tile] = img_array
        
        if not images_dict:
            st.error("Failed to download any tiles")
            return
        
        # Create GeoTIFF with proper alignment
        status_text.text("Creating GeoTIFF...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"google_satellite_{timestamp}.tiff"
        create_geotiff(tiles, images_dict, output_path, zoom)
        
        # Offer download
        status_text.text("Ready for download!")
        with open(output_path, 'rb') as f:
            st.download_button(
                label="Download GeoTIFF",
                data=f,
                file_name=output_path,
                mime="image/tiff"
            )
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()