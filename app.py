import streamlit as st
import cv2
import numpy as np
from PIL import Image
from wand.image import Image as WandImage
import io
import os
from datetime import datetime
import gc
import tifffile

# Streamlit page configuration
st.set_page_config(
    page_title="Stone Pattern Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)
Image.MAX_IMAGE_PIXELS = None  # Disable PIL max pixel limit

def load_large_image(uploaded_file):
    """Composite all channels (CMYK + spot plates) via ImageMagick (wand), fallback to PIL."""
    temp_path = "temp.tif"
    try:
        # Write to disk
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Try ImageMagick composite via Wand
        try:
            with WandImage(filename=temp_path) as wimg:
                # Remove alpha and composite on white
                wimg.background_color = 'white'
                wimg.alpha_channel = 'remove'
                blob = wimg.make_blob(format='png')
                image = Image.open(io.BytesIO(blob))
                st.success("Loaded via Wand/ImageMagick composite!")
                os.remove(temp_path)
                return image
        except Exception as wand_e:
            st.write("Wand composite failed:", wand_e)

        # Fallback to PIL composite
        try:
            with Image.open(temp_path) as pil_img:
                rgb = pil_img.convert("RGB")
                st.success("Loaded via PIL composite!")
                os.remove(temp_path)
                return rgb
        except Exception as pil_e:
            st.error(f"PIL fallback failed: {pil_e}")

        return None

    except Exception as e:
        st.error(f"Error loading image: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return None


def detect_and_move_flakes(image, redistribution_intensity, flake_size_range, color_sensitivity):
    """Detect flake regions and randomly move them to create a new stone pattern."""
    try:
        arr = np.array(image)
        h, w = arr.shape[:2]
        new_arr = arr.copy()

        max_size = int(200 * flake_size_range)
        step = max(1, max_size // (int(redistribution_intensity * 10) + 1))

        progress_bar = st.progress(0.0)
        progress_text = st.empty()
        total_steps = max(1, ((h - max_size) // step) + 1)
        step_count = 0

        for y in range(0, h - max_size, step):
            for x in range(0, w - max_size, step):
                roi = arr[y:y+max_size, x:x+max_size]
                var = np.var(roi, axis=(0,1))
                threshold = 500 * (1 - color_sensitivity)
                if np.sum(var) > threshold:
                    flake = roi.copy()
                    rng = int(min(h, w) * redistribution_intensity)
                    new_y = np.random.randint(max(0, y - rng), min(h - max_size, y + rng))
                    new_x = np.random.randint(max(0, x - rng), min(w - max_size, x + rng))
                    new_arr[new_y:new_y+max_size, new_x:new_x+max_size] = flake

            step_count += 1
            progress = step_count / total_steps
            progress_bar.progress(progress)
            progress_text.text(f"Processing... {int(progress*100)}%")
            if step_count % 10 == 0:
                gc.collect()

        progress_text.text("Processing complete!")
        return Image.fromarray(new_arr)

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None


def save_large_image(image, filename):
    """Save image as PNG with high DPI."""
    try:
        image.save(filename, "PNG", dpi=(300, 300))
    except Exception as e:
        st.error(f"Error saving image: {e}")


def main():
    st.title("Stone Pattern Generator")
    uploaded_file = st.file_uploader("Choose a TIFF image...", type=["tif", "tiff"])
    if uploaded_file is not None:
        st.write(f"File: {uploaded_file.name} • Size: {uploaded_file.size/(1024*1024):.2f} MB")
        with st.spinner("Loading image..."):
            image = load_large_image(uploaded_file)

        if image is not None:
            st.image(image, caption="Original Image", use_column_width=True)

            st.sidebar.header("Pattern Controls")
            redistribution_intensity = st.sidebar.slider(
                "Redistribution Intensity", 0.1, 1.0, 0.5,
                help="How far flakes can move from their original position"
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
                with st.spinner("Generating design..."):
                    variation = detect_and_move_flakes(
                        image,
                        redistribution_intensity,
                        flake_size_range,
                        color_sensitivity
                    )
                    if variation is not None:
                        os.makedirs("generated_images", exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"generated_images/variation_{timestamp}.png"
                        save_large_image(variation, filename)
                        st.image(variation, caption="New Design", use_column_width=True)
                        with open(filename, 'rb') as f:
                            st.download_button(
                                label="Download New Design",
                                data=f,
                                file_name=f"new_design_{timestamp}.png",
                                mime="image/png"
                            )

if __name__ == "__main__":
    main()
