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
    """Handle large TIFF files with arbitrary extra dimensions."""
    temp_path = "temp.tif"
    try:
        st.info("Loading image... This may take a moment for large files.")
        # 1) Geçici olarak diske yaz
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 2) tifffile ile oku
        with tifffile.TiffFile(temp_path) as tif:
            st.write("Reading TIFF metadata...")
            for page in tif.pages:
                # BitsPerSample
                try:
                    bps = page.bitspersample
                except AttributeError:
                    bps = page.tags['BitsPerSample'].value
                st.write(f"Bits per sample: {bps}")

                # SampleFormat
                sf = page.sampleformat
                st.write(f"Sample format: {sf}")

                # Photometric
                phot = getattr(page, 'photometric', 'Unknown')
                st.write(f"Color space: {phot}")

                # SamplesPerPixel
                spp = getattr(page, 'samplesperpixel', 'Unknown')
                st.write(f"Samples per pixel: {spp}")

            # Ham array'i al
            img_array = tif.asarray()
        
        st.write(f"Original array shape: {img_array.shape}, dtype: {img_array.dtype}")

        # 3) Fazla boyutları temizle (singleton dims)
        img_array = np.squeeze(img_array)

        # 4) Planar (C, H, W) gelenleri (H, W, C) yap
        if img_array.ndim == 3 and img_array.shape[0] in (3, 4, 8):
            # Burada 8 kanallı planar da (8, H, W) olabilir; transpose ile (H, W, 8)
            img_array = np.transpose(img_array, (1, 2, 0))
        st.write(f"Processed array shape: {img_array.shape}")

        # 5) Kanal sayısına göre uygun dönüşüm
        if img_array.ndim == 3 and img_array.shape[2] == 4:
            # CMYK -> RGB
            st.write("Converting CMYK to RGB...")
            if img_array.dtype != np.uint8:
                img_array = ((img_array - img_array.min()) * (255.0 / (img_array.max() - img_array.min()))).astype(np.uint8)
            cmyk = img_array.astype(float) / 255.0
            c, m, y, k = cv2.split(cmyk)
            r = (1.0 - c) * (1.0 - k)
            g = (1.0 - m) * (1.0 - k)
            b = (1.0 - y) * (1.0 - k)
            rgb = cv2.merge([r, g, b])
            rgb = (rgb * 255).astype(np.uint8)
            image = Image.fromarray(rgb)

        elif img_array.ndim == 3 and img_array.shape[2] > 4:
            # Örneğin 8 kanallı: fazla kanalları atıp ilk 3’ü RGB kabul et
            st.warning(f"Image has {img_array.shape[2]} channels; using first 3 as RGB.")
            rgb_arr = img_array[:, :, :3]
            image = Image.fromarray(rgb_arr)

        else:
            # 1 veya 3 kanallı (grayscale veya RGB) direkt
            image = Image.fromarray(img_array)

        os.remove(temp_path)
        st.success("Image loaded successfully!")
        return image

    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.error(f"File size: {uploaded_file.size / (1024*1024):.2f} MB")
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # 6) Fallback: PIL bytes yöntemi
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
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        new_image = img_array.copy()

        min_flake_size = int(20 * flake_size_range)
        max_flake_size = int(200 * flake_size_range)
        step_size = max(1, int(max_flake_size // (redistribution_intensity + 1)))

        progress_bar = st.progress(0.0)
        progress_text = st.empty()
        total_steps = max(1, ((height - max_flake_size) // step_size) + 1)
        current_step = 0

        for y in range(0, height - max_flake_size, step_size):
            for x in range(0, width - max_flake_size, step_size):
                roi = img_array[y:y+max_flake_size, x:x+max_flake_size]
                variance = np.var(roi, axis=(0,1))
                threshold = 500 * (1 - color_sensitivity)
                if np.sum(variance) > threshold:
                    flake = roi.copy()
                    move_range = int(min(height, width) * redistribution_intensity)
                    new_y = np.random.randint(max(0, y-move_range), min(height-max_flake_size, y+move_range))
                    new_x = np.random.randint(max(0, x-move_range), min(width-max_flake_size, x+move_range))
                    new_image[new_y:new_y+max_flake_size, new_x:new_x+max_flake_size] = flake

            current_step += 1
            progress = current_step / total_steps
            progress_bar.progress(progress)
            progress_text.text(f"Processing... {int(progress * 100)}%")

            if current_step % 10 == 0:
                gc.collect()

        progress_text.text("Processing complete!")
        return Image.fromarray(new_image)

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def save_large_image(image, filename):
    try:
        image.save(filename, "PNG", dpi=(300, 300))
    except Exception as e:
        st.error(f"Error saving image: {e}")

def main():
    st.title("Stone Pattern Generator")
    uploaded_file = st.file_uploader("Choose an image...", type=["tif", "tiff"])
    if uploaded_file is not None:
        st.write(f"Loading file: {uploaded_file.name}")
        st.write(f"File size: {uploaded_file.size/(1024*1024):.2f} MB")

        with st.spinner('Loading image...'):
            image = load_large_image(uploaded_file)

        if image:
            st.image(image, caption="Original Image", use_column_width=True)
            st.sidebar.header("Pattern Controls")
            redistribution_intensity = st.sidebar.slider(
                "Redistribution Intensity", 0.1, 1.0, 0.5,
                help="How far flakes can move from original"
            )
            flake_size_range = st.sidebar.slider(
                "Flake Size Range", 0.5, 2.0, 1.0,
                help="Size range of detected flakes"
            )
            color_sensitivity = st.sidebar.slider(
                "Color Sensitivity", 0.1, 1.0, 0.5,
                help="Sensitivity to color variance"
            )

            if st.button("Generate New Design"):
                with st.spinner('Generating new design...'):
                    variation = detect_and_move_flakes(
                        image, redistribution_intensity,
                        flake_size_range, color_sensitivity
                    )
                    if variation:
                        os.makedirs("generated_images", exist_ok=True)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        fn = f"generated_images/variation_{ts}.png"
                        save_large_image(variation, fn)
                        st.image(variation, caption="New Design", use_column_width=True)
                        with open(fn, 'rb') as f:
                            st.download_button(
                                label="Download New Design",
                                data=f, file_name=f"new_design_{ts}.png",
                                mime="image/png"
                            )

if __name__ == "__main__":
    main()
