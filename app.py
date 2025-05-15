import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime

# Increase PIL image size limit
Image.MAX_IMAGE_PIXELS = None

def detect_and_move_flakes(image):
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
    
    # Parameters for flake detection
    min_flake_size = 20
    max_flake_size = 200
    
    # Detect flakes using color differences
    for y in range(0, height-max_flake_size, max_flake_size//2):
        for x in range(0, width-max_flake_size, max_flake_size//2):
            # Get region of interest
            roi = lab[y:y+max_flake_size, x:x+max_flake_size]
            
            # Calculate color variance
            variance = np.var(roi, axis=(0,1))
            
            # If there's significant color variance (likely a flake)
            if np.sum(variance) > 500:  # Threshold for flake detection
                # Get the flake
                flake = img_array[y:y+max_flake_size, x:x+max_flake_size].copy()
                
                # Find new random position
                new_y = np.random.randint(0, height-max_flake_size)
                new_x = np.random.randint(0, width-max_flake_size)
                
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
        try:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
            
            if st.button("Generate New Design"):
                progress_text = st.empty()
                progress_text.text("Generating new design...")
                
                # Generate variation
                variation = detect_and_move_flakes(image)
                
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
