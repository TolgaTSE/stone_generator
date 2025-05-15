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
    """Handle large TIFFs (CMYK + spot channels) by letting PIL composite to RGB."""
    temp_path = "temp.tif"
    try:
        st.info("Loading image... may take a moment for large files.")
        # 1) write to disk so Pillow can see the full file structure
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 2) Try PIL composite first (handles CMYK + spot plates automatically)
        try:
            with Image.open(temp_path) as pil_img:
                rgb = pil_img.convert("RGB")
                st.success("Loaded via Pillow composite!")
                os.remove(temp_path)
                return rgb
        except Exception as pil_e:
            st.write("Pillow composite failed, falling back to tifffile:", pil_e)

        # 3) If you really want to inspect channels, use tifffile
        with tifffile.TiffFile(temp_path) as tif:
            st.write("Reading TIFF metadata (via tifffile)...")
            for page in tif.pages:
                try:
                    bps = page.bitspersample
                except AttributeError:
                    bps = page.tags['BitsPerSample'].value
                st.write(f" • Bits per sample: {bps}")
                st.write(f" • Sample format: {page.sampleformat}")
                st.write(f" • Photometric: {getattr(page,'photometric','Unknown')}")
                st.write(f" • Samples per pixel: {getattr(page,'samplesperpixel','Unknown')}")

            arr = tif.asarray()  # full 8-channel array

        st.write(f"Raw array shape: {arr.shape}, dtype={arr.dtype}")

        # squeeze single dims and reorder planar if needed
        arr = np.squeeze(arr)
        if arr.ndim == 3 and arr.shape[0] in (3,4,8):
            arr = np.transpose(arr, (1,2,0))
        st.write(f"Processed array shape: {arr.shape}")

        # Fallback manual CMYK→RGB (but this will still lose spot plates!)
        if arr.ndim==3 and arr.shape[2]>=4:
            st.warning(f"{arr.shape[2]} channels: manually converting first 4 as CMYK → RGB")
            cmyk = arr[..., :4].astype(float)
            if arr.dtype==np.uint8:
                cmyk /= 255.0
            else:
                cmyk = (cmyk - cmyk.min())/(cmyk.max()-cmyk.min())
            c, m, y, k = cv2.split(cmyk)
            r = (1-c)*(1-k); g = (1-m)*(1-k); b = (1-y)*(1-k)
            rgb = np.clip((cv2.merge([r,g,b]) * 255),0,255).astype(np.uint8)
            image = Image.fromarray(rgb)
        elif arr.ndim==3 and arr.shape[2]==3:
            image = Image.fromarray(arr)
        elif arr.ndim==2:
            image = Image.fromarray(arr)
        else:
            st.error(f"Unhandled format: {arr.shape}")
            image = None

        os.remove(temp_path)
        return image

    except Exception as e:
        st.error(f"Error loading image: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
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
