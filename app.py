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

# Increase PIL image size limit
Image.MAX_IMAGE_PIXELS = None

def load_large_image(uploaded_file):
    """Handle large TIFF files, convert CMYK (first 4 channels) to RGB if present."""
    temp_path = "temp.tif"
    try:
        st.info("Loading image... This may take a moment for large files.")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with tifffile.TiffFile(temp_path) as tif:
            st.write("Reading TIFF metadata...")
            for page in tif.pages:
                try:
                    bps = page.bitspersample
                except AttributeError:
                    bps = page.tags['BitsPerSample'].value
                st.write(f"Bits per sample: {bps}")
                st.write(f"Sample format: {page.sampleformat}")
                st.write(f"Color space (Photometric): {getattr(page,'photometric','Unknown')}")
                st.write(f"Samples per pixel: {getattr(page,'samplesperpixel','Unknown')}")

            img_array = tif.asarray()

        st.write(f"Original array shape: {img_array.shape}, dtype: {img_array.dtype}")
        img_array = np.squeeze(img_array)

        # Eğer planar (C, H, W) gelmişse (örneğin shape[0] in (3,4,8)), (H,W,C) formatına çevir
        if img_array.ndim == 3 and img_array.shape[0] in (3, 4, 8):
            img_array = np.transpose(img_array, (1, 2, 0))
        st.write(f"Processed array shape: {img_array.shape}")

        # 4 veya daha fazla kanallıysa CMYK dönüştürme: ilk 4 kanal C,M,Y,K kabul edilir
        if img_array.ndim == 3 and img_array.shape[2] >= 4:
            st.write(f"Image has {img_array.shape[2]} channels; treating first 4 as CMYK.")
            cmyk = img_array[:, :, :4].astype(float)
            # normalize 0–1
            if img_array.dtype != np.uint8:
                cmyk = (cmyk - cmyk.min()) / (cmyk.max() - cmyk.min())
            else:
                cmyk /= 255.0

            c, m, y, k = cv2.split(cmyk)
            r = (1 - c) * (1 - k)
            g = (1 - m) * (1 - k)
            b = (1 - y) * (1 - k)
            rgb = cv2.merge([r, g, b])
            rgb = np.clip((rgb * 255), 0, 255).astype(np.uint8)
            image = Image.fromarray(rgb)

        # Sadece 3 kanallıysa doğrudan RGB
        elif img_array.ndim == 3 and img_array.shape[2] == 3:
            image = Image.fromarray(img_array)

        # Tek kanallıysa gri ton olarak
        elif img_array.ndim == 2:
            image = Image.fromarray(img_array)

        else:
            st.error(f"Unexpected format after channel handling: {img_array.shape}")
            image = None

        os.remove(temp_path)
        if image:
            st.success("Image loaded successfully!")
        return image

    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.error(f"File size: {uploaded_file.size / (1024*1024):.2f} MB")
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Fallback: PIL ile yükleme
        try:
            st.write("Trying alternative loading method...")
            uploaded_file.seek(0)
            image = Image.open(io.BytesIO(uploaded_file.getbuffer()))
            if image.mode == 'CMYK':
                image = image.convert('RGB')
            st.success("Alternative method succeeded!")
            return image
        except Exception as e2:
            st.error(f"Alternative method failed: {e2}")
            return None

def detect_and_move_flakes(image, redistribution_intensity, flake_size_range, color_sensitivity):
    try:
        arr = np.array(image)
        h, w = arr.shape[:2]
        new = arr.copy()

        max_size = int(200 * flake_size_range)
        step = max(1, max_size // (int(redistribution_intensity*10) + 1))

        prog = st.progress(0.0)
        info = st.empty()
        total = max(1, ((h - max_size)//step)+1)
        i = 0

        for y in range(0, h-max_size, step):
            for x in range(0, w-max_size, step):
                roi = arr[y:y+max_size, x:x+max_size]
                var = np.var(roi, axis=(0,1))
                thresh = 500 * (1-color_sensitivity)
                if np.sum(var) > thresh:
                    flake = roi.copy()
                    rng = int(min(h,w)*redistribution_intensity)
                    ny = np.random.randint(max(0,y-rng), min(h-max_size, y+rng))
                    nx = np.random.randint(max(0,x-rng), min(w-max_size, x+rng))
                    new[ny:ny+max_size, nx:nx+max_size] = flake

            i += 1
            p = i/total
            prog.progress(p)
            info.text(f"Processing... {int(p*100)}%")
            if i%10==0:
                gc.collect()

        info.text("Processing complete!")
        return Image.fromarray(new)

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def save_large_image(image, filename):
    try:
        image.save(filename, "PNG", dpi=(300,300))
    except Exception as e:
        st.error(f"Error saving image: {e}")

def main():
    st.title("Stone Pattern Generator")
    file = st.file_uploader("Choose an image...", type=["tif","tiff"])
    if file:
        st.write(f"Loading file: {file.name}")
        st.write(f"Size: {file.size/(1024*1024):.2f} MB")
        with st.spinner("Loading image..."):
            img = load_large_image(file)
        if img:
            st.image(img, caption="Original", use_column_width=True)
            st.sidebar.header("Pattern Controls")
            ri = st.sidebar.slider("Redistribution Intensity",0.1,1.0,0.5)
            fs = st.sidebar.slider("Flake Size Range",0.5,2.0,1.0)
            cs = st.sidebar.slider("Color Sensitivity",0.1,1.0,0.5)
            if st.button("Generate New Design"):
                with st.spinner("Generating..."):
                    var = detect_and_move_flakes(img,ri,fs,cs)
                    if var:
                        os.makedirs("generated_images",exist_ok=True)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        fn = f"generated_images/variation_{ts}.png"
                        save_large_image(var, fn)
                        st.image(var, caption="New Design", use_column_width=True)
                        with open(fn,"rb") as f:
                            st.download_button("Download New Design",f, file_name=f"new_design_{ts}.png",
                                               mime="image/png")

if __name__=="__main__":
    main()
