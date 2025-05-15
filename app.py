import streamlit as st
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError, ImageCms
import io
import os
import tempfile
from datetime import datetime
import gc
import tifffile
from io import BytesIO

# Streamlit page settings
st.set_page_config(
    page_title="Stone Pattern Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)
Image.MAX_IMAGE_PIXELS = None  # Disable PIL pixel limit

def cmyk_to_rgb(cmyk_arr):
    """Improved CMYK to RGB conversion"""
    cmyk = cmyk_arr.astype(np.float32) / 255.0
    c, m, y, k = cv2.split(cmyk)
    
    # Adobe's CMYK to RGB conversion formula
    r = 255 * (1.0 - c) * (1.0 - k)
    g = 255 * (1.0 - m) * (1.0 - k)
    b = 255 * (1.0 - y) * (1.0 - k)
    
    rgb = cv2.merge([r, g, b])
    return np.clip(rgb, 0, 255).astype(np.uint8)

def apply_color_profile(image):
    """Apply color profile conversion if available"""
    if 'icc_profile' in image.info:
        try:
            srgb_profile = ImageCms.createProfile('sRGB')
            img_profile = ImageCms.ImageCmsProfile(BytesIO(image.info['icc_profile']))
            image = ImageCms.profileToProfile(image, img_profile, srgb_profile)
            st.success("Color profile successfully applied")
        except Exception as e:
            st.warning(f"Color profile conversion failed: {e}")
    return image

def validate_color_space(image):
    """Validate and correct color space if needed"""
    if image.mode == 'CMYK':
        st.warning("CMYK image detected, converting to RGB...")
        return image.convert('RGB')
    elif image.mode == 'RGB':
        return image
    else:
        st.warning(f"Unexpected color mode: {image.mode}")
        return image.convert('RGB')

def debug_color_info(image):
    """Display debug information about the image color space"""
    st.write("Image Mode:", image.mode)
    st.write("Color Space Info:", image.info.get('icc_profile') is not None)
    arr = np.array(image)
    st.write("Array Shape:", arr.shape)
    st.write("Value Range:", arr.min(), "-", arr.max())

def load_large_image(uploaded_file):
    """Load TIFF with improved color handling"""
    tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    temp_path = tmp.name
    try:
        tmp.write(uploaded_file.getbuffer())
        tmp.flush()
    finally:
        tmp.close()

    try:
        # Try PIL first
        try:
            with Image.open(temp_path) as img:
                img.load()
                if 'icc_profile' in img.info:
                    rgb = apply_color_profile(img)
                else:
                    rgb = validate_color_space(img)
                debug_color_info(rgb)
                return rgb
        except (UnidentifiedImageError, Exception) as pil_err:
            st.write("PIL load failed:", pil_err)

        # Fallback to tifffile
        with tifffile.TiffFile(temp_path) as tif:
            arr = tif.pages[0].asarray()
            arr = np.squeeze(arr)
            
            if arr.ndim == 3 and arr.shape[0] == 4:
                arr = np.transpose(arr, (1, 2, 0))
                rgb_arr = cmyk_to_rgb(arr)
            else:
                rgb_arr = arr

            image = Image.fromarray(rgb_arr)
            return validate_color_space(image)

    except Exception as e:
        st.error(f"Image load error: {e}")
        return None

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

def detect_and_move_flakes(image, redistribution_intensity, flake_size_range, color_sensitivity):
    """Detect and randomly move flake regions to create a new pattern."""
    try:
        arr = np.array(image)
        h, w = arr.shape[:2]
        new_arr = arr.copy()

        max_size = int(200 * flake_size_range)
        step = max(1, max_size // (int(redistribution_intensity * 10) + 1))

        prog = st.progress(0.0)
        txt = st.empty()
        total = max(1, ((h - max_size) // step) + 1)
        count = 0

        for y in range(0, h - max_size, step):
            for x in range(0, w - max_size, step):
                roi = arr[y:y+max_size, x:x+max_size]
                var = np.var(roi, axis=(0,1))
                thresh = 500 * (1 - color_sensitivity)
                if np.sum(var) > thresh:
                    flake = roi.copy()
                    rng = int(min(h, w) * redistribution_intensity)
                    new_y = np.random.randint(max(0, y - rng), min(h - max_size, y + rng))
                    new_x = np.random.randint(max(0, x - rng), min(w - max_size, x + rng))
                    new_arr[new_y:new_y+max_size, new_x:new_x+max_size] = flake

            count += 1
            p = count / total
            prog.progress(p)
            txt.text(f"Processing... {int(p*100)}%")
            if count % 10 == 0:
                gc.collect()

        txt.text("Processing complete!")
        return Image.fromarray(new_arr)

    except Exception as e:
        st.error(f"Processing error: {e}")
        return None

def save_png(image, path):
    try:
        image.save(path, "PNG", dpi=(300,300))
    except Exception as e:
        st.error(f"PNG save error: {e}")

def save_tiff(image, path):
    try:
        image.save(path, "TIFF", dpi=(300,300))
    except Exception as e:
        st.error(f"TIFF save error: {e}")

def main():
    st.title("Stone Pattern Generator")
    uploaded_file = st.file_uploader("Select a TIFF image...", type=["tif","tiff"])
    
    if uploaded_file:
        st.write(f"File: {uploaded_file.name} ({uploaded_file.size/(1024*1024):.2f} MB)")
        with st.spinner("Loading image..."):
            image = load_large_image(uploaded_file)

        if image:
            st.image(image, caption="Original", use_column_width=True)
            st.sidebar.header("Controls")
            ri = st.sidebar.slider("Redistribution Intensity",0.1,1.0,0.5)
            fs = st.sidebar.slider("Flake Size Range",0.5,2.0,1.0)
            cs = st.sidebar.slider("Color Sensitivity",0.1,1.0,0.5)

            if st.button("Generate New Design"):
                with st.spinner("Generating..."):
                    var = detect_and_move_flakes(image, ri, fs, cs)
                    if var:
                        os.makedirs("generated_images",exist_ok=True)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        png_path = f"generated_images/design_{ts}.png"
                        tiff_path = f"generated_images/design_{ts}.tif"
                        save_png(var, png_path)
                        save_tiff(var, tiff_path)
                        st.image(var, caption="New Design", use_column_width=True)
                        with open(png_path,'rb') as f:
                            st.download_button("Download PNG", data=f, file_name=os.path.basename(png_path), mime="image/png")
                        with open(tiff_path,'rb') as f:
                            st.download_button("Download TIFF", data=f, file_name=os.path.basename(tiff_path), mime="image/tiff")

if __name__=="__main__":
    main()
