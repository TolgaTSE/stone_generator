import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
import gc
import tifffile

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
    """Handle large CMYK TIFF files"""
    try:
        st.info("Loading image... This may take a moment for large files.")
        
        # Save to temporary file
        temp_path = "temp.tif"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Try different methods to load the image
        try:
            # Method 1: Using PIL with CMYK handling
            with Image.open(temp_path) as img:
                st.write(f"Original image mode: {img.mode}")
                if img.mode == 'CMYK':
                    img = img.convert('RGB')
                image = img.copy()
        except:
            # Method 2: Using OpenCV
            img_array = cv2.imread(temp_path, cv2.IMREAD_UNCHANGED)
            if img_array is None:
                raise Exception("Failed to load with OpenCV")
                
            # Convert from BGR to RGB
            if len(img_array.shape) == 3:
                if img_array.shape[2] == 4:  # CMYK
                    # Convert CMYK to RGB
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_CMYK2BGR)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
            image = Image.fromarray(img_array)
        
        # Remove temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Print debug information
        st.write(f"Image size: {image.size}")
        st.write(f"Image mode: {image.mode}")
        
        return image
            
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        st.error(f"File size: {uploaded_file.size / (1024*1024):.2f} MB")
        
        # Try one last method with tifffile
        try:
            import tifffile
            with tifffile.TiffFile(temp_path) as tif:
                img_array = tif.asarray()
                
                # Convert to RGB if needed
                if len(img_array.shape) == 3 and img_array.shape[2] == 4:  # CMYK
                    # Convert to RGB using a simple conversion
                    c, m, y, k = cv2.split(img_array)
                    # CMYK to RGB conversion
                    r = 255 * (1 - c/255) * (1 - k/255)
                    g = 255 * (1 - m/255) * (1 - k/255)
                    b = 255 * (1 - y/255) * (1 - k/255)
                    
                    img_array = cv2.merge([r, g, b]).astype(np.uint8)
                
                image = Image.fromarray(img_array)
                return image
                
        except Exception as e2:
            st.error(f"All methods failed: {str(e2)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
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
