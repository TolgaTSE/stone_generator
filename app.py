import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime

def create_variation(image):
    # Convert PIL Image to OpenCV format
    img_array = np.array(image)
    
    # Convert to BGR format if needed
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Get image dimensions
    height, width = img_array.shape[:2]
    
    # Create mask for stone flakes detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours (stone flakes)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create new image with same background
    new_image = img_array.copy()
    
    # Randomly reposition stone flakes
    for contour in contours:
        # Get the stone flake
        x, y, w, h = cv2.boundingRect(contour)
        flake = img_array[y:y+h, x:x+w]
        
        # Generate random new position
        new_x = np.random.randint(0, width - w)
        new_y = np.random.randint(0, height - h)
        
        # Place flake in new position
        new_image[new_y:new_y+h, new_x:new_x+w] = flake
    
    # Convert back to RGB
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(new_image)

def main():
    st.title("Stone Texture Generator")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Create button
        if st.button("Generate Variations"):
            # Create directory for saving images if it doesn't exist
            if not os.path.exists("generated_images"):
                os.makedirs("generated_images")
            
            # Generate 8 variations
            variations = []
            for i in range(8):
                variation = create_variation(image)
                variations.append(variation)
                
                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_images/variation_{i+1}_{timestamp}.png"
                variation.save(filename, "PNG", dpi=(300, 300))
            
            # Display variations in 2x4 grid
            col1, col2, col3, col4 = st.columns(4)
            col5, col6, col7, col8 = st.columns(4)
            
            cols = [col1, col2, col3, col4, col5, col6, col7, col8]
            
            for idx, (variation, col) in enumerate(zip(variations, cols)):
                with col:
                    st.image(variation, caption=f"Variation {idx+1}", use_column_width=True)
                    
                    # Convert image to bytes for download
                    buf = io.BytesIO()
                    variation.save(buf, format="PNG", dpi=(300, 300))
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label=f"Download #{idx+1}",
                        data=byte_im,
                        file_name=f"variation_{idx+1}.png",
                        mime="image/png"
                    )

if __name__ == "__main__":
    main()
