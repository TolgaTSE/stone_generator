import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
import gc
from PIL import TiffImagePlugin

# Increase PIL image size limit and TIFF parameters
Image.MAX_IMAGE_PIXELS = None
TiffImagePlugin.LOAD_TRUNCATED_IMAGES = True

def load_large_image(uploaded_file):
    """Handle large TIFF files using PIL directly"""
    try:
        # Read the file into a bytes buffer
        image_bytes = uploaded_file.getvalue()
        image_buffer = io.BytesIO(image_bytes)
        
        # Open with PIL
        image = Image.open(image_buffer)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
            
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        st.error("File details:")
        st.error(f"File size: {len(image_bytes) / (1024*1024):.2f} MB")
        st.error(f"File type: {uploaded_file.type}")
        return None

def detect_and_move_flakes(image, redistribution_intensity, flake_size_range, color_sensitivity):
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Ensure image is in RGB format
        if len(img_array.shape) != 3:
            st.error("Please upload a color image")
            return None
        
        # Get dimensions
        height, width = img_array.shape[:2]
        
        # Create new image
        new_image = img_array.copy()
        
        # Calculate parameters
        min_flake_size = int(20 * flake_size_range)
        max_flake_size = int(200 * flake_size_range)
        step_size = int(max_flake_size // (redistribution_intensity + 1))
        
        # Create progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        # Process image
        total_steps = ((height - max_flake_size) // step_size) + 1
        current_step = 0
        
        for y in range(0, height - max_flake_size, step_size):
            for x in range(0, width - max_flake_size, step_size):
                # Get region of interest
                roi = img_array[y:y+max_flake_size, x:x+max_flake_size]
                
                # Calculate color variance
                variance = np.var(roi, axis=(0,1))
                
                # Adjust threshold based on color sensitivity
                threshold = 500 * (1 - color_sensitivity)
                
                if np.sum(variance) > threshold:
                    # Get the flake
                    flake = roi.copy()
                    
                    # Calculate movement range
                    move_range = int(min(height, width) * redistribution_intensity)
                    
                    # Find new random position
                    new_y = np.random.randint(
                        max(0, y-move_range), 
                        min(height-max_flake_size, y+move_range)
                    )
                    new_x = np.random.randint(
                        max(0, x-move_range), 
                        min(width-max_flake_size, x+move_range)
                    )
                    
                    # Place flake in new position
                    new_image[new_y:new_y+max_flake_size, new_x:new_x+max_flake_size] = flake
            
            # Update progress
            current_step += 1
            progress = current_step / total_steps
            progress_bar.progress(progress)
            progress_text.text(f"Processing... {int(progress * 100)}%")
            
            # Clear memory periodically
            if current_step % 10 == 0:
                gc.collect()
        
        progress_text.text("Processing complete!")
        return Image.fromarray(new_image)
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def save_large_image(image, filename):
    """Save image with appropriate format"""
    try:
        image.save(filename, "PNG", dpi=(300, 300))
    except Exception as e:
        st.error(f"Error saving image: {str(e)}")

def main():
    st.title("Stone Pattern Generator")
    
    # Add file size warning
    st.write("Note: Large files may take longer to process")
    
    uploaded_file = st.file_uploader("Choose an image...", 
                                   type=["jpg", "jpeg", "png", "tif", "tiff"])
    
    if uploaded_file is not None:
        try:
            # Show file details
            file_size_mb = uploaded_file.size / (1024*1024)
            st.write(f"File size: {file_size_mb:.2f} MB")
            
            # Display original image
            with st.spinner('Loading image...'):
                image = load_large_image(uploaded_file)
            
            if image is not None:
                st.image(image, caption="Original Image", use_column_width=True)
                
                # Control parameters
                st.sidebar.header("Pattern Controls")
                
                redistribution_intensity = st.sidebar.slider(
                    "Redistribution Intensity",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.5,
                    help="Controls how far flakes can move from their original position"
                )
                
                flake_size_range = st.sidebar.slider(
                    "Flake Size Range",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.0,
                    help="Adjusts the size range of detected flakes"
                )
                
                color_sensitivity = st.sidebar.slider(
                    "Color Sensitivity",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.5,
                    help="Controls how sensitive the detection is to color variations"
                )
                
                if st.button("Generate New Design"):
                    with st.spinner('Generating new design...'):
                        # Generate variation
                        variation = detect_and_move_flakes(
                            image,
                            redistribution_intensity,
                            flake_size_range,
                            color_sensitivity
                        )
                        
                        if variation is not None:
                            # Create directory if it doesn't exist
                            if not os.path.exists("generated_images"):
                                os.makedirs("generated_images")
                            
                            # Save image
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"generated_images/variation_{timestamp}.png"
                            save_large_image(variation, filename)
                            
                            # Display variation
                            st.image(variation, caption="New Design", use_column_width=True)
                            
                            # Download button
                            with open(filename, 'rb') as file:
                                st.download_button(
                                    label="Download New Design",
                                    data=file,
                                    file_name=f"new_design_{timestamp}.png",
                                    mime="image/png"
                                )
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please try with a different image or contact support")

if __name__ == "__main__":
    main()
