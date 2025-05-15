import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
import tifffile
import gc  # Garbage collector for memory management

# Increase PIL image size limit
Image.MAX_IMAGE_PIXELS = None

def load_large_image(file):
    """Handle both TIFF and regular image files"""
    file_extension = file.name.lower().split('.')[-1]
    
    if file_extension == 'tif' or file_extension == 'tiff':
        # Save uploaded file temporarily
        with open('temp.tiff', 'wb') as f:
            f.write(file.getvalue())
        # Read with tifffile
        img_array = tifffile.imread('temp.tiff')
        # Remove temporary file
        os.remove('temp.tiff')
        # Convert to PIL Image
        return Image.fromarray(img_array)
    else:
        return Image.open(file)

def process_in_chunks(image, chunk_size=2000):
    """Process large images in chunks"""
    width, height = image.size
    
    # Calculate number of chunks
    x_chunks = (width + chunk_size - 1) // chunk_size
    y_chunks = (height + chunk_size - 1) // chunk_size
    
    return width, height, x_chunks, y_chunks, chunk_size

def detect_and_move_flakes(image, redistribution_intensity, flake_size_range, color_sensitivity):
    # Get image dimensions and chunk information
    width, height, x_chunks, y_chunks, chunk_size = process_in_chunks(image)
    
    # Create new image (as numpy array)
    new_image = np.array(image)
    
    # Process each chunk
    for y_chunk in range(y_chunks):
        for x_chunk in range(x_chunks):
            # Calculate chunk coordinates
            x_start = x_chunk * chunk_size
            y_start = y_chunk * chunk_size
            x_end = min((x_chunk + 1) * chunk_size, width)
            y_end = min((y_chunk + 1) * chunk_size, height)
            
            # Extract chunk
            chunk = new_image[y_start:y_end, x_start:x_end]
            
            # Convert chunk to LAB color space
            chunk_lab = cv2.cvtColor(chunk, cv2.COLOR_RGB2LAB)
            
            # Calculate flake sizes based on slider
            min_flake_size = int(20 * flake_size_range)
            max_flake_size = int(200 * flake_size_range)
            
            # Calculate step size based on redistribution intensity
            step_size = int(max_flake_size // (redistribution_intensity + 1))
            
            # Process flakes in chunk
            for y in range(0, chunk.shape[0]-max_flake_size, step_size):
                for x in range(0, chunk.shape[1]-max_flake_size, step_size):
                    # Get region of interest
                    roi = chunk_lab[y:y+max_flake_size, x:x+max_flake_size]
                    
                    # Calculate color variance
                    variance = np.var(roi, axis=(0,1))
                    
                    # Adjust threshold based on color sensitivity
                    threshold = 500 * (1 - color_sensitivity)
                    
                    if np.sum(variance) > threshold:
                        # Get the flake
                        flake = chunk[y:y+max_flake_size, x:x+max_flake_size].copy()
                        
                        # Calculate movement range
                        move_range = int(min(height, width) * redistribution_intensity)
                        
                        # Calculate global coordinates
                        global_y = y_start + y
                        global_x = x_start + x
                        
                        # Find new random position
                        new_y = np.random.randint(
                            max(0, global_y-move_range), 
                            min(height-max_flake_size, global_y+move_range)
                        )
                        new_x = np.random.randint(
                            max(0, global_x-move_range), 
                            min(width-max_flake_size, global_x+move_range)
                        )
                        
                        # Place flake in new position
                        new_image[new_y:new_y+max_flake_size, 
                                new_x:new_x+max_flake_size] = flake
            
            # Clear memory
            gc.collect()
    
    return Image.fromarray(new_image)

def save_large_image(image, filename):
    """Save large image with appropriate format"""
    if filename.lower().endswith(('.tif', '.tiff')):
        # Save as TIFF
        tifffile.imwrite(filename, np.array(image))
    else:
        # Save as PNG
        image.save(filename, "PNG", dpi=(300, 300))

def main():
    st.title("Terrazzo Pattern Generator")
    
    # File uploader with TIFF support
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
            
            # Output format selection
            output_format = st.sidebar.selectbox(
                "Output Format",
                ["PNG", "TIFF"],
                help="Select the output file format"
            )
            
            if st.button("Generate New Design"):
                progress_text = st.empty()
                progress_bar = st.progress(0)
                progress_text.text("Generating new design...")
                
                # Generate variation with parameters
                variation = detect_and_move_flakes(
                    image,
                    redistribution_intensity,
                    flake_size_range,
                    color_sensitivity
                )
                
                # Create directory if it doesn't exist
                if not os.path.exists("generated_images"):
                    os.makedirs("generated_images")
                
                # Save with selected format
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                extension = ".tiff" if output_format == "TIFF" else ".png"
                filename = f"generated_images/variation_{timestamp}{extension}"
                
                save_large_image(variation, filename)
                
                # Display variation
                st.image(variation, caption="New Design", use_column_width=True)
                
                # Download button
                with open(filename, 'rb') as file:
                    st.download_button(
                        label=f"Download New Design ({output_format})",
                        data=file,
                        file_name=f"new_design_{timestamp}{extension}",
                        mime=f"image/{output_format.lower()}"
                    )
                
                progress_text.text("New design generated successfully!")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
