import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
import gc

# Increase PIL image size limit
Image.MAX_IMAGE_PIXELS = None

def load_large_image(file):
    """Handle different image formats"""
    image = Image.open(file)
    # Convert to RGB mode if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def process_in_chunks(image, chunk_size=2000):
    """Process large images in chunks"""
    width, height = image.size
    x_chunks = (width + chunk_size - 1) // chunk_size
    y_chunks = (height + chunk_size - 1) // chunk_size
    return width, height, x_chunks, y_chunks, chunk_size

def detect_and_move_flakes(image, redistribution_intensity, flake_size_range, color_sensitivity):
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Ensure image is in RGB format
    if len(img_array.shape) != 3:
        st.error("Please upload a color image")
        return None
    
    # Get dimensions and chunk information
    width, height = image.size
    
    # Create new image
    new_image = img_array.copy()
    
    # Calculate parameters
    min_flake_size = int(20 * flake_size_range)
    max_flake_size = int(200 * flake_size_range)
    step_size = int(max_flake_size // (redistribution_intensity + 1))
    
    # Process image
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
        if y % 100 == 0:
            progress = y / (height - max_flake_size)
            st.progress(progress)
    
    return Image.fromarray(new_image)

def save_large_image(image, filename):
    """Save image with appropriate format"""
    image.save(filename, "PNG", dpi=(300, 300))

def main():
    st.title("Stone Pattern Generator")
    
    uploaded_file = st.file_uploader("Choose an image...", 
                                   type=["jpg", "jpeg", "png", "tif", "tiff"])
    
    if uploaded_file is not None:
        try:
            # Display original image
            image = load_large_image(uploaded_file)
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
                progress_text = st.empty()
                progress_text.text("Generating new design...")
                
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
                    
                    progress_text.text("New design generated successfully!")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
