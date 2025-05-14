import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
import random

# Increase PIL image size limit
Image.MAX_IMAGE_PIXELS = None

def resize_if_needed(image, max_size=3000):
    width, height = image.size
    if width > max_size or height > max_size:
        ratio = min(max_size/width, max_size/height)
        new_size = (int(width*ratio), int(height*ratio))
        return image.resize(new_size, Image.LANCZOS)
    return image

def extract_and_reposition_flakes(img_array):
    # Convert to grayscale for flake detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size
    height, width = img_array.shape[:2]
    min_area = 100
    max_area = (width * height) // 20
    flakes = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, (255), -1)
            flake_img = img_array.copy()
            flake_img[mask == 0] = 0
            flakes.append({
                'image': flake_img[y:y+h, x:x+w],
                'mask': mask[y:y+h, x:x+w],
                'width': w,
                'height': h
            })
    
    return flakes

def create_variation(image):
    # Resize image if needed
    image = resize_if_needed(image)
    
    # Convert PIL Image to OpenCV format
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Get dimensions
    height, width = img_array.shape[:2]
    
    # Extract flakes
    flakes = extract_and_reposition_flakes(img_array)
    
    # Create new image (exact copy of original)
    new_image = img_array.copy()
    
    # Create mask to track placed flakes
    placed_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Randomly reposition flakes
    random.shuffle(flakes)
    for flake in flakes:
        # Try multiple positions
        for _ in range(10):
            new_x = random.randint(0, width - flake['width'])
            new_y = random.randint(0, height - flake['height'])
            
            # Check if position is available
            roi = placed_mask[new_y:new_y+flake['height'], new_x:new_x+flake['width']]
            if np.sum(roi) == 0:
                # Place flake
                new_image[new_y:new_y+flake['height'], new_x:new_x+flake['width']] = flake['image']
                placed_mask[new_y:new_y+flake['height'], new_x:new_x+flake['width']] = flake['mask']
                break
    
    # Convert back to RGB
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(new_image)

def main():
    st.title("Stone Texture Generator")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
            
            if st.button("Generate Variations"):
                if not os.path.exists("generated_images"):
                    os.makedirs("generated_images")
                
                variations = []
                progress_bar = st.progress(0)
                
                for i in range(8):
                    variation = create_variation(image)
                    variations.append(variation)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"generated_images/variation_{i+1}_{timestamp}.png"
                    variation.save(filename, "PNG", dpi=(300, 300))
                    
                    progress_bar.progress((i + 1) / 8)
                
                # Display variations
                col1, col2, col3, col4 = st.columns(4)
                col5, col6, col7, col8 = st.columns(4)
                cols = [col1, col2, col3, col4, col5, col6, col7, col8]
                
                for idx, (variation, col) in enumerate(zip(variations, cols)):
                    with col:
                        st.image(variation, caption=f"Variation {idx+1}", use_column_width=True)
                        
                        buf = io.BytesIO()
                        variation.save(buf, format="PNG", dpi=(300, 300))
                        byte_im = buf.getvalue()
                        
                        st.download_button(
                            label=f"Download #{idx+1}",
                            data=byte_im,
                            file_name=f"variation_{idx+1}.png",
                            mime="image/png"
                        )
                
                st.success("All variations generated successfully!")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
