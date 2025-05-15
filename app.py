import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime

# Increase PIL image size limit
Image.MAX_IMAGE_PIXELS = None

def resize_for_processing(image, target_size=2000):
    """Resize image for processing while maintaining aspect ratio"""
    width, height = image.size
    ratio = min(target_size/width, target_size/height)
    new_size = (int(width*ratio), int(height*ratio))
    return image.resize(new_size, Image.LANCZOS)

def segment_flakes(image, chunk_size=1000):
    """Process image in chunks to save memory"""
    # Convert to numpy array
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # Process in chunks
    result = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(0, height, chunk_size):
        for x in range(0, width, chunk_size):
            # Get chunk coordinates
            y2 = min(y + chunk_size, height)
            x2 = min(x + chunk_size, width)
            
            # Process chunk
            chunk = img_array[y:y2, x:x2]
            
            # Convert chunk to LAB
            chunk_lab = cv2.cvtColor(chunk, cv2.COLOR_RGB2LAB)
            
            # Get L channel
            l_channel = chunk_lab[:, :, 0]
            
            # Threshold
            _, thresh = cv2.threshold(l_channel, 127, 255, cv2.THRESH_OTSU)
            
            # Save result
            result[y:y2, x:x2] = thresh
    
    return result

def create_variation(original_image):
    # Resize for processing
    process_image = resize_for_processing(original_image)
    
    # Get flake mask
    flake_mask = segment_flakes(process_image)
    
    # Convert back to original size
    flake_mask = cv2.resize(flake_mask, original_image.size[::-1])
    
    # Convert original image to numpy array
    img_array = np.array(original_image)
    
    # Create new image
    new_image = img_array.copy()
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(flake_mask)
    
    # Create random order for relocation
    order = np.arange(1, num_labels)
    np.random.shuffle(order)
    
    # Relocate each component
    height, width = img_array.shape[:2]
    for label in order:
        # Get component mask
        mask = (labels == label)
        
        if np.sum(mask) > 100:  # Skip very small components
            # Get component bounds
            y, x = np.where(mask)
            top, bottom = np.min(y), np.max(y)
            left, right = np.min(x), np.max(x)
            
            # Get component
            component = img_array[top:bottom+1, left:right+1].copy()
            component_mask = mask[top:bottom+1, left:right+1]
            
            # Find new position
            h, w = bottom-top+1, right-left+1
            new_x = np.random.randint(0, max(1, width - w))
            new_y = np.random.randint(0, max(1, height - h))
            
            # Place component
            new_image[new_y:new_y+h, new_x:new_x+w][component_mask] = \
                component[component_mask]
    
    return Image.fromarray(new_image)

def main():
    st.title("Stone Texture Generator")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
            
            if st.button("Generate New Design"):
                progress_text = st.empty()
                progress_text.text("Generating new design...")
                
                # Generate single variation
                variation = create_variation(image)
                
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
