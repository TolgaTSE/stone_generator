import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime

# Increase PIL image size limit
Image.MAX_IMAGE_PIXELS = None

def detect_and_move_flakes(image, redistribution_intensity, flake_size_range, color_sensitivity):
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    
    # Create a copy for the new image
    new_image = img_array.copy()
    
    # Get dimensions
    height, width = img_array.shape[:2]
    
    # Create a mask for tracking placed flakes
    placed_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Calculate flake sizes based on slider
    min_flake_size = int(20 * flake_size_range)
    max_flake_size = int(200 * flake_size_range)
    
    # Calculate step size based on redistribution intensity
    step_size = int(max_flake_size // (redistribution_intensity + 1))
    
    # Detect flakes using color differences
    for y in range(0, height-max_flake_size, step_size):
        for x in range(0, width-max_flake_size, step_size):
            # Get region of interest
            roi = lab[y:y+max_flake_size, x:x+max_flake_size]
            
            # Calculate color variance
            variance = np.var(roi, axis=(0,1))
            
            # Adjust threshold based on color sensitivity
            threshold = 500 * (1 - color_sensitivity)
            
            # If there's significant color variance (likely a flake)
            if np.sum(variance) > threshold:
                # Get the flake
                flake = img_array[y:y+max_flake_size, x:x+max_flake_size].copy()
                
                # Calculate movement range based on redistribution intensity
                move_range = int(min(height, width) * redistribution_intensity)
                
                # Find new random position
                new_y = np.random.randint(max(0, y-move_range), min(height-max_flake_size, y+move_range))
                new_x = np.random.randint(max(0, x-move_range), min(width-max_flake_size, x+move_range))
                
                # If position is not already used
                if np.sum(placed_mask[new_y:new_y+max_flake_size, 
                                    new_x:new_x+max_flake_size]) == 0:
                    # Place flake in new position
                    new_image[new_y:new_y+max_flake_size, 
                            new_x:new_x+max_flake_size] = flake
                    
                    # Update placed mask
                    placed_mask[new_y:new_y+max_flake_size, 
                              new_x:new_x+max_flake_size] = 1
    
    return Image.fromarray(new_image)

def main():
    st.title("Terrazzo Pattern Generator")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
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
            try:
                progress_text = st.empty()
                progress_text.text("Generating new design...")
                
                # Generate variation with parameters
                variation = detect_and_move_flakes(
                    image,
                    redistribution_intensity,
                    flake_size_range,
                    color_sensitivity
                )
                
                # Save image
                if not os.path.exists("generated_images"):
                    os.makedirs("generated_images")
                    
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_images/variation_{timestamp}.png"
                variation.save(filename, "PNG", dpi=(300, 300))
                
                # Display variation
                st.image(variation, caption="New Design", use_column_width=True)
                
                # Download button
                buf = io.BytesIO()
                variation.save(buf, format="PNG", dpi=(300, 300))
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Download New Design",
                    data=byte_im,
                    file_name=f"new_design_{timestamp}.png",
                    mime="image/png"
                )
                
                progress_text.text("New design generated successfully!")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
