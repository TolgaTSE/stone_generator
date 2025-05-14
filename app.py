import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime

# Increase PIL image size limit
Image.MAX_IMAGE_PIXELS = None

def segment_flakes(image):
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert to LAB color space for better segmentation
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Threshold to find flakes
    _, thresh = cv2.threshold(l, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Noise removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    return sure_fg, sure_bg

def create_variation(image):
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # Get flake masks
    flake_mask, bg_mask = segment_flakes(image)
    
    # Create new image starting with original
    new_image = img_array.copy()
    
    # Find connected components (flakes)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(flake_mask, connectivity=8)
    
    # Create list of flakes
    flakes = []
    for i in range(1, num_labels):  # Skip background (0)
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Filter out very small or very large components
        if 100 < area < (width * height) // 20:
            mask = (labels == i).astype(np.uint8)
            flake = img_array.copy()
            flake[mask == 0] = 0
            flakes.append({
                'image': flake[y:y+h, x:x+w],
                'mask': mask[y:y+h, x:x+w],
                'width': w,
                'height': h
            })
    
    # Create placement mask
    placement_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Random placement of flakes
    np.random.shuffle(flakes)
    for flake in flakes:
        for _ in range(20):  # Try 20 times to place each flake
            new_x = np.random.randint(0, width - flake['width'])
            new_y = np.random.randint(0, height - flake['height'])
            
            # Check if area is available
            roi = placement_mask[new_y:new_y+flake['height'], 
                               new_x:new_x+flake['width']]
            
            if np.sum(roi) == 0:  # If area is free
                # Place flake
                new_image[new_y:new_y+flake['height'], 
                         new_x:new_x+flake['width']][flake['mask'] > 0] = \
                    flake['image'][flake['mask'] > 0]
                
                # Update placement mask
                placement_mask[new_y:new_y+flake['height'], 
                             new_x:new_x+flake['width']] = flake['mask']
                break
    
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
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                for i in range(8):
                    progress_text.text(f"Generating variation {i+1}/8...")
                    variation = create_variation(image)
                    variations.append(variation)
                    
                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"generated_images/variation_{i+1}_{timestamp}.png"
                    variation.save(filename, "PNG", dpi=(300, 300))
                    
                    progress_bar.progress((i + 1) / 8)
                
                # Display variations in grid
                col1, col2, col3, col4 = st.columns(4)
                col5, col6, col7, col8 = st.columns(4)
                cols = [col1, col2, col3, col4, col5, col6, col7, col8]
                
                for idx, (variation, col) in enumerate(zip(variations, cols)):
                    with col:
                        st.image(variation, caption=f"Variation {idx+1}", 
                                use_column_width=True)
                        
                        buf = io.BytesIO()
                        variation.save(buf, format="PNG", dpi=(300, 300))
                        byte_im = buf.getvalue()
                        
                        st.download_button(
                            label=f"Download #{idx+1}",
                            data=byte_im,
                            file_name=f"variation_{idx+1}.png",
                            mime="image/png"
                        )
                
                progress_text.text("All variations generated successfully!")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
