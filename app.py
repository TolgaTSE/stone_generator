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
    """Resize image if it's too large"""
    width, height = image.size
    if width > max_size or height > max_size:
        ratio = min(max_size/width, max_size/height)
        new_size = (int(width*ratio), int(height*ratio))
        return image.resize(new_size, Image.LANCZOS)
    return image

def detect_flakes(img_array):
    """Improved flake detection"""
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size
    min_area = 100  # Minimum area to consider as a flake
    max_area = img_array.shape[0] * img_array.shape[1] // 4  # Maximum area
    filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    
    return filtered_contours

def create_variation(image):
    # Resize image if it's too large
    image = resize_if_needed(image)
    
    # Convert PIL Image to OpenCV format
    img_array = np.array(image)
    
    # Convert to BGR format if needed
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Get image dimensions
    height, width = img_array.shape[:2]
    
    # Detect stone flakes
    contours = detect_flakes(img_array)
    
    # Create new image with same background
    new_image = img_array.copy()
    
    # Create mask for placed flakes
    placed_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Randomly reposition stone flakes
    random.shuffle(contours)  # Randomize flake order
    
    for contour in contours:
        # Get the stone flake
        x, y, w, h = cv2.boundingRect(contour)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, (255), -1)
        flake = cv2.bitwise_and(img_array, img_array, mask=mask)
        
        # Try to find a new position (max 10 attempts)
        for _ in range(10):
            new_x = np.random.randint(0, width - w)
            new_y = np.random.randint(0, height - h)
            
            # Check if position is already occupied
            roi = placed_mask[new_y:new_y+h, new_x:new_x+w]
            if np.sum(roi) == 0:  # If position is free
                # Update placed mask
                placed_mask[new_y:new_y+h, new_x:new_x+w] = mask[y:y+h, x:x+w]
                
                # Place flake in new position
                mask_area = mask[y:y+h, x:x+w] > 0
                new_image[new_y:new_y+h, new_x:new_x+w][mask_area] = \
                    img_array[y:y+h, x:x+w][mask_area]
                break
    
    # Apply slight color variation while preserving overall color scheme
    hsv = cv2.cvtColor(new_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Randomly adjust saturation and value
    hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(0.9, 1.1)  # Saturation
    hsv[:, :, 2] = hsv[:, :, 2] * random.uniform(0.9, 1.1)  # Value
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    new_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Convert back to RGB
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(new_image)

def main():
    st.title("Stone Texture Generator")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
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
                progress_bar = st.progress(0)
                
                for i in range(8):
                    variation = create_variation(image)
                    variations.append(variation)
                    
                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"generated_images/variation_{i+1}_{timestamp}.png"
                    variation.save(filename, "PNG", dpi=(300, 300))
                    
                    # Update progress bar
                    progress_bar.progress((i + 1) / 8)
                
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
                
                st.success("All variations generated successfully!")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
