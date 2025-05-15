import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
import gc

# Configure Streamlit
st.set_page_config(
    page_title="Stone Pattern Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add loading message at the start
with st.spinner('Initializing application...'):
    # Increase PIL image size limit
    Image.MAX_IMAGE_PIXELS = None

def load_large_image(uploaded_file):
    """Handle CMYK TIFF files with proper color conversion"""
    try:
        st.info("Loading image... This may take a moment for large files.")
        
        # Try direct PIL opening from buffer
        image = Image.open(uploaded_file)
        if image.mode == 'CMYK':
            st.info("Converting CMYK to RGB...")
            image = image.convert('RGB')
        
        st.success("Image loaded successfully!")
        return image
            
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        st.error(f"File size: {uploaded_file.size / (1024*1024):.2f} MB")
        return None

def main():
    try:
        st.title("Stone Pattern Generator")
        st.write("Upload a TIFF file to generate new pattern variations.")
        
        # Add file uploader with clear instructions
        st.write("### Upload Image")
        st.write("Supported formats: TIFF (CMYK or RGB)")
        uploaded_file = st.file_uploader("Choose an image...", 
                                       type=["tif", "tiff"])
        
        if uploaded_file is not None:
            # Display file information
            st.write(f"File name: {uploaded_file.name}")
            st.write(f"File size: {uploaded_file.size / (1024*1024):.2f} MB")
            
            # Load and display image
            image = load_large_image(uploaded_file)
            
            if image is not None:
                # Display original image
                st.image(image, caption="Original Image", use_column_width=True)
                
                # Control parameters in sidebar
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
                
                # Generate button
                if st.button("Generate New Design"):
                    with st.spinner('Generating new design...'):
                        try:
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
                                variation.save(filename, "PNG", dpi=(300, 300))
                                
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
                                
                                st.success("New design generated successfully!")
                                
                        except Exception as e:
                            st.error(f"Error generating design: {str(e)}")
        
        # Add instructions at the bottom
        st.markdown("""
        ### Instructions
        1. Upload a TIFF image file
        2. Adjust the controls in the sidebar
        3. Click 'Generate New Design'
        4. Download the new design if you like it
        """)
                
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Please try refreshing the page")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Critical error: {str(e)}")
        st.error("Please restart the application")
