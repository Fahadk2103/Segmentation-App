import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import geopandas as gpd
import rasterio
from rasterio.plot import show
from rasterio.features import shapes
import streamlit as st
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import warnings
import ssl
import requests
from io import BytesIO
warnings.filterwarnings('ignore')

def download_sam_checkpoint(model_type):
    """Function to download the SAM checkpoint with SSL certificate handling"""
    checkpoint_urls = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    }
    
    models_dir = "sam_models"
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = f"{models_dir}/sam_{model_type}.pth"
    
    if not os.path.exists(model_path):
        st.info(f"Downloading {model_type} model... This might take a while.")
        
        try:
            # Using requests instead of urllib to handle SSL issues more gracefully
            response = requests.get(checkpoint_urls[model_type], stream=True, verify=False)
            response.raise_for_status()
            
            # Get total file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            
            progress_bar = st.progress(0)
            
            with open(model_path, 'wb') as f:
                downloaded = 0
                for data in response.iter_content(block_size):
                    f.write(data)
                    downloaded += len(data)
                    if total_size > 0:  # Only update if content length was received
                        progress = int(downloaded / total_size * 100)
                        progress_bar.progress(min(progress/100, 1.0))
            
            st.success(f"Downloaded {model_type} model to {model_path}")
        except Exception as e:
            st.error(f"Download failed: {str(e)}")
            st.info("Alternative: You can manually download the model from the link below and place it in the 'sam_models' folder:")
            st.code(checkpoint_urls[model_type])
            return None
    
    return model_path

def generate_masks_using_automatic_mask_generator(source, model_type, checkpoint_path, 
                                                points_per_side=32, 
                                                pred_iou_thresh=0.88, 
                                                stability_score_thresh=0.95, 
                                                min_mask_region_area=100):
    """Generate segmentation masks from a GeoTIFF using SAM AutomaticMaskGenerator"""
    # Load the image with rasterio to preserve geospatial info
    with rasterio.open(source) as src:
        # Read image and get metadata
        image = src.read()
        transform = src.transform
        crs = src.crs
        
        # Print debug information
        st.write(f"Original image shape: {image.shape}, dtype: {image.dtype}")
        
        # CRITICAL FIX: Properly handle image dimensions for SAM
        # SAM expects 3-channel RGB images in (H, W, 3) format
        
        # Get original shape
        if len(image.shape) == 3:
            num_bands, height, width = image.shape
        else:
            # If somehow the image has unexpected dimensions
            st.error(f"Unexpected image dimensions: {image.shape}")
            # Try to recover - assume it's a single band
            num_bands = 1
            if len(image.shape) == 2:
                height, width = image.shape
                image = image.reshape(1, height, width)
            else:
                raise ValueError(f"Cannot process image with shape {image.shape}")
        
        st.write(f"Detected {num_bands} bands, height={height}, width={width}")
        
        # Create a 3-channel RGB image regardless of input
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Fill the RGB channels based on available bands
        if num_bands >= 3:
            # Use the first three bands as RGB
            for i in range(3):
                channel = image[i]
                # Normalize to 0-255 range
                if channel.dtype != np.uint8:
                    min_val = np.min(channel)
                    max_val = np.max(channel)
                    if max_val > min_val:
                        channel = ((channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                    else:
                        channel = np.zeros_like(channel, dtype=np.uint8)
                rgb_image[:, :, i] = channel
        else:
            # If fewer than 3 bands, duplicate the first band
            channel = image[0]
            # Normalize to 0-255 range
            if channel.dtype != np.uint8:
                min_val = np.min(channel)
                max_val = np.max(channel)
                if max_val > min_val:
                    channel = ((channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    channel = np.zeros_like(channel, dtype=np.uint8)
            
            # Duplicate across all three channels
            rgb_image[:, :, 0] = channel
            rgb_image[:, :, 1] = channel
            rgb_image[:, :, 2] = channel
        
        # Ensure we have an RGB uint8 image
        rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
        
        st.write(f"Converted image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")
        
        # Initialize SAM
        st.write("Initializing SAM model...")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.write(f"Using device: {device}")
        sam.to(device=device)
        
        # Initialize the automatic mask generator
        st.write("Setting up mask generator...")
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
            crop_n_layers=0,  # Don't crop image - process as a whole
            crop_n_points_downscale_factor=1
        )
        
        # Generate masks
        st.write("Generating masks... This may take a while.")
        masks = mask_generator.generate(rgb_image)
        
        st.write(f"Generated {len(masks)} masks")
        
        # Check if any masks were generated
        if len(masks) == 0:
            st.warning("No masks were generated. Try adjusting the parameters.")
            # Create an empty mask image that we can save
            mask_image = np.zeros((height, width), dtype=np.uint8)
            
            # Save as GeoTIFF even if empty
            with rasterio.open(
                "segmentation_masks.tif",
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=mask_image.dtype,
                crs=crs,
                transform=transform,
            ) as dst:
                dst.write(mask_image, 1)
            
            return mask_image, "segmentation_masks.tif", []
        
        # Create a mask array the size of the image
        mask_image = np.zeros((height, width), dtype=np.uint8)
        
        # Assign unique values to each mask
        for i, mask_data in enumerate(masks):
            mask = mask_data["segmentation"]  # This is a boolean array
            mask_area = np.sum(mask)
            if mask_area > min_mask_region_area:
                mask_image[mask] = i + 1  # Start from 1
        
        # Save as GeoTIFF
        with rasterio.open(
            "segmentation_masks.tif",
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=mask_image.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(mask_image, 1)
        
        # Create a colored visualization of the masks
        color_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create random colors for visualization
        np.random.seed(42)  # For reproducibility
        mask_colors = np.random.randint(0, 255, size=(len(masks) + 1, 3), dtype=np.uint8)
        mask_colors[0] = [0, 0, 0]  # Background is black
        
        # Apply colors to the mask image
        for i in range(1, len(masks) + 1):
            color_mask[mask_image == i] = mask_colors[i]
        
        # Save colored mask as PNG
        color_mask_path = "colored_masks.png"
        Image.fromarray(color_mask).save(color_mask_path)
        
        return mask_image, "segmentation_masks.tif", masks

def raster_to_geojson(raster_path, output_path, transform, crs):
    """Convert a raster mask to GeoJSON with proper projection"""
    with rasterio.open(raster_path) as src:
        image = src.read(1)  # Read the first band
        mask = image > 0  # Create a binary mask
        
        # Check if mask contains any valid data
        if not np.any(mask):
            st.warning("No valid mask regions found. Try adjusting the filtering parameters.")
            return None
        
        # Get the shapes of all features
        results = (
            {'properties': {'value': v}, 'geometry': s}
            for i, (s, v) in enumerate(
                shapes(image, mask=mask, transform=src.transform))
        )
        
        # Convert results to a list to ensure we have features
        results_list = list(results)
        
        if not results_list:
            st.warning("No valid geometries were extracted from the mask.")
            return None
        
        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features(results_list, crs=crs)
        
        # Save to GeoJSON
        gdf.to_file(output_path, driver="GeoJSON")
        
        return gdf

def sam_masks_to_geojson(masks, transform, crs, output_path):
    """Convert SAM masks directly to GeoJSON without using raster intermediary"""
    if not masks:
        st.warning("No masks to convert to GeoJSON")
        return None
    
    features = []
    
    for i, mask_data in enumerate(masks):
        # Get the mask segmentation
        mask = mask_data["segmentation"]
        
        # Create a temporary binary image for this mask
        h, w = mask.shape
        mask_image = np.zeros((h, w), dtype=np.uint8)
        mask_image[mask] = 1
        
        # Extract shapes with the appropriate transform
        mask_shapes = list(shapes(mask_image, mask=mask_image > 0, transform=transform))
        
        # Add each shape as a feature
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
    
    # Create a GeoJSON feature collection
    geojson_data = {
        'type': 'FeatureCollection',
        'features': features
    }
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(geojson_data['features'], crs=crs)
    
    # Save to file
    gdf.to_file(output_path, driver="GeoJSON")
    
    return gdf

def main():
    # Disable SSL verification globally (only if needed)
    ssl._create_default_https_context = ssl._create_unverified_context
    
    st.set_page_config(page_title="Segment Anything GeoTIFF App", layout="wide")
    st.title("Geospatial Segmentation with SAM")
    
    # Sidebar for model configuration
    st.sidebar.header("Model Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select SAM Model",
        ["vit_h", "vit_l", "vit_b"],
        index=1  # Default to vit_l as it's a good balance between accuracy and speed
    )
    
    # Model parameters - adjusted default values for better results
    points_per_side = st.sidebar.slider("Points Per Side", min_value=4, max_value=64, value=32, step=4)
    pred_iou_thresh = st.sidebar.slider("Prediction IOU Threshold", min_value=0.0, max_value=1.0, value=0.75, step=0.01)
    stability_score_thresh = st.sidebar.slider("Stability Score Threshold", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
    min_mask_region_area = st.sidebar.slider("Minimum Mask Region Area", min_value=0, max_value=1000, value=50, step=10)
    
    # Set default checkpoint path
    models_dir = "sam_models"
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_path = f"{models_dir}/sam_{model_type}.pth"
    
    # Model download/upload options
    st.sidebar.subheader("Model Download/Upload")
    download_tab, upload_tab, local_tab = st.sidebar.tabs(["Download", "Upload", "Local Path"])
    
    with download_tab:
        st.caption("Disable SSL verification if you encounter certificate errors")
        disable_ssl = st.checkbox("Disable SSL Verification", value=True)
        
        if st.button("Download/Verify Model"):
            if disable_ssl:
                ssl._create_default_https_context = ssl._create_unverified_context
            downloaded_path = download_sam_checkpoint(model_type)
            if downloaded_path:
                checkpoint_path = downloaded_path
                st.sidebar.success(f"Model ready at {checkpoint_path}")
    
    with upload_tab:
        st.caption("Upload model file if you have it locally")
        st.warning("Note: Streamlit has a 200MB file upload limit. For larger models, download directly or use the local path option.")
        uploaded_file = st.file_uploader("Upload .pth file", type=["pth"])
        if uploaded_file is not None:
            with open(checkpoint_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Model uploaded and saved to {checkpoint_path}")
    
    with local_tab:
        st.caption("Specify path to model file already on your system")
        local_path = st.text_input("Full path to model file (.pth)")
        if local_path and os.path.exists(local_path):
            checkpoint_path = local_path
            st.success(f"Will use model at: {checkpoint_path}")
        elif local_path:
            st.error(f"No file found at {local_path}")
    
    # Check if model exists
    model_exists = os.path.exists(checkpoint_path)
    if not model_exists:
        st.warning(f"Model not found at {checkpoint_path}. Please download or upload the model first.")
    else:
        model_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # Size in MB
        st.success(f"Model found at {checkpoint_path} ({model_size:.2f} MB)")
    
    # Upload GeoTIFF file
    st.header("Upload GeoTIFF")
    uploaded_file = st.file_uploader("Choose a GeoTIFF file", type=["tif", "tiff"])
    
    # Add debug mode option
    debug_mode = st.checkbox("Enable debug mode", value=True)
    
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        temp_tif_path = "temp_upload.tif"
        with open(temp_tif_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the uploaded GeoTIFF
        try:
            with rasterio.open(temp_tif_path) as src:
                if debug_mode:
                    st.write(f"GeoTIFF info: {src.count} bands, shape: {src.shape}, dtype: {src.dtypes}")
                
                # Store transform and CRS for later use
                raster_transform = src.transform
                raster_crs = src.crs
                
                fig, ax = plt.subplots(figsize=(10, 10))
                show(src, ax=ax)
                st.pyplot(fig)
                
                # Segment the image when the user clicks the "Segment" button
                if st.button("Segment Image"):
                    if not model_exists:
                        st.error(f"Model checkpoint not found at {checkpoint_path}. Please download or upload the model first.")
                    else:
                        with st.spinner("Segmenting image... This may take a while."):
                            try:
                                # Generate masks using AutomaticMaskGenerator
                                mask_image, mask_path, sam_masks = generate_masks_using_automatic_mask_generator(
                                    temp_tif_path, 
                                    model_type, 
                                    checkpoint_path,
                                    points_per_side,
                                    pred_iou_thresh,
                                    stability_score_thresh,
                                    min_mask_region_area
                                )
                                
                                if mask_image is not None and mask_path is not None:
                                    # Display results header
                                    st.header("Segmentation Results")
                                    
                                    # Create columns for mask and GeoJSON outputs
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.subheader("Mask Output")
                                        
                                        # Provide download button for the mask raster
                                        with open(mask_path, 'rb') as f:
                                            st.download_button(
                                                label="Download Mask Raster (GeoTIFF)",
                                                data=f,
                                                file_name=mask_path,
                                                mime="image/tiff"
                                            )
                                        
                                        # Show mask visualization
                                        if os.path.exists("colored_masks.png"):
                                            st.image("colored_masks.png", caption="Segmentation Masks")
                                            
                                            # Provide download for the colored visualization
                                            with open("colored_masks.png", 'rb') as f:
                                                st.download_button(
                                                    label="Download Visualization (PNG)",
                                                    data=f,
                                                    file_name="colored_masks.png",
                                                    mime="image/png"
                                                )
                                    
                                    with col2:
                                        st.subheader("GeoJSON Output")
                                        
                                        # Convert SAM masks directly to GeoJSON
                                        output_geojson = "segmentation_results.geojson"
                                        
                                        # Convert using SAM mask data (preferred method for better attributes)
                                        if sam_masks:
                                            gdf = sam_masks_to_geojson(sam_masks, raster_transform, raster_crs, output_geojson)
                                        else:
                                            # Fallback to raster conversion
                                            gdf = raster_to_geojson(mask_path, output_geojson, raster_transform, raster_crs)
                                        
                                        # Only proceed with visualization if we have valid geometries
                                        if gdf is not None:
                                            # Show a preview of the GeoJSON data
                                            st.dataframe(gdf.head())
                                            
                                            # Provide download button for the GeoJSON file
                                            with open(output_geojson, 'rb') as f:
                                                st.download_button(
                                                    label="Download GeoJSON",
                                                    data=f,
                                                    file_name=output_geojson,
                                                    mime="application/json"
                                                )
                                        else:
                                            st.warning("No valid GeoJSON conversion was possible.")
                                    
                                    # Visualize the overlay results
                                    st.subheader("Segmentation Overlay")
                                    
                                    if gdf is not None:
                                        # Show the original image with segmentation overlay
                                        fig, ax = plt.subplots(figsize=(12, 12))
                                        
                                        # Display the background image
                                        with rasterio.open(temp_tif_path) as src:
                                            show(src, ax=ax)
                                        
                                        # Plot the masks with random colors
                                        gdf.plot(ax=ax, alpha=0.5, cmap='tab20')
                                        
                                        # Show the plot
                                        st.pyplot(fig)
                                    else:
                                        st.warning("No valid segmentation overlay could be created.")
                                else:
                                    st.error("Segmentation failed to produce valid results.")
                            
                            except Exception as e:
                                st.error(f"Error during segmentation: {str(e)}")
                                if debug_mode:
                                    import traceback
                                    st.code(traceback.format_exc())
                                st.info("Check that the model file is valid and matches the selected model type.")
        except Exception as e:
            st.error(f"Error opening GeoTIFF: {str(e)}")
            if debug_mode:
                import traceback
                st.code(traceback.format_exc())
    
    st.sidebar.header("About")
    st.sidebar.info("""
    This app uses the Segment Anything Model (SAM) from Meta AI to segment GeoTIFF imagery.
    
    Upload a GeoTIFF file, configure the model parameters, and get segmentation results in both raster mask and GeoJSON formats.
    """)

if __name__ == "__main__":
    main()