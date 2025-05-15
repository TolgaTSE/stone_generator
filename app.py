import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageCms
import io
import os
from datetime import datetime
import gc
import tifffile

# Streamlit sayfa ayarları
st.set_page_config(
    page_title="Stone Pattern Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PIL için limitsiz piksel boyutu
Image.MAX_IMAGE_PIXELS = None

def load_large_image(uploaded_file):
    """TIFF içindeki gömülü ICC profiline göre CMYK→sRGB dönüşümü yapar,
       spot-kanalları atar ve doğru renk tonlarını elde eder."""
    temp_path = "temp.tif"
    try:
        # Geçici dosyaya yaz
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # tifffile ile oku
        with tifffile.TiffFile(temp_path) as tif:
            page = tif.pages[0]
            arr = page.asarray()  # (H, W, C) veya planar
            tags = page.tags

        # Fazla boyutları bırak
        arr = np.squeeze(arr)
        # Planar geldi ise (C, H, W) → (H, W, C)
        if arr.ndim == 3 and arr.shape[0] in (3, 4, 8):
            arr = np.transpose(arr, (1, 2, 0))

        # Gömülü ICC profili çek
        icc_profile = None
        if "ICCProfile" in tags:
            icc_profile = tags["ICCProfile"].value

        # CMYK veya CMYK+spot (ilk 4 kanal) → RGB
        if arr.ndim == 3 and arr.shape[2] >= 4:
            # Sadece temel 4 kanal
            cmyk_arr = arr[..., :4].astype(np.uint8)

            if icc_profile:
                # ICC profille renk yönetimi
                cmyk_img = Image.fromarray(cmyk_arr, mode="CMYK")
                in_prof = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
                out_prof = ImageCms.createProfile("sRGB")
                rgb_img = ImageCms.profileToProfile(
                    cmyk_img, in_prof, out_prof, outputMode="RGB"
                )
                image = rgb_img
            else:
                # Manuel CMYK→RGB
                cmyk = cmyk_arr.astype(float) / 255.0
                c, m, y, k = cv2.split(cmyk)
                r = (1 - c)*(1 - k)
                g = (1 - m)*(1 - k)
                b = (1 - y)*(1 - k)
                rgb = (np.clip((cv2.merge([r, g, b]) * 255), 0, 255)).astype(np.uint8)
                image = Image.fromarray(rgb)

        elif arr.ndim == 3 and arr.shape[2] == 3:
            # Zaten RGB
            image = Image.fromarray(arr)

        elif arr.ndim == 2:
            # Grayscale
            image = Image.fromarray(arr)

        else:
            st.error(f"Beklenmeyen dizi formatı: {arr.shape}")
            return None

        os.remove(temp_path)
        st.success("Görüntü başarıyla yüklendi!")
        return image

    except Exception as e:
        st.error(f"Görüntü yükleme hatası: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return None

def detect_and_move_flakes(image, redistribution_intensity, flake_size_range, color_sensitivity):
    """Pul benzeri bölgeleri algılayıp rastgele kaydırarak yeni taş deseni oluşturur."""
    try:
        arr = np.array(image)
        h, w = arr.shape[:2]
        new = arr.copy()

        max_size = int(200 * flake_size_range)
        step = max(1, max_size // (int(redistribution_intensity * 10) + 1))

        prog = st.progress(0.0)
        info = st.empty()
        total = max(1, ((h - max_size) // step) + 1)
        i = 0

        for y in range(0, h - max_size, step):
            for x in range(0, w - max_size, step):
                roi = arr[y:y+max_size, x:x+max_size]
                var = np.var(roi, axis=(0,1))
                thresh = 500 * (1 - color_sensitivity)
                if np.sum(var) > thresh:
                    flake = roi.copy()
                    rng = int(min(h, w) * redistribution_intensity)
                    ny = np.random.randint(max(0, y - rng), min(h - max_size, y + rng))
                    nx = np.random.randint(max(0, x - rng), min(w - max_size, x + rng))
                    new[ny:ny+max_size, nx:nx+max_size] = flake

            i += 1
            p = i / total
            prog.progress(p)
            info.text(f"İşleniyor... {int(p * 100)}%")
            if i % 10 == 0:
                gc.collect()

        info.text("İşlem tamamlandı!")
        return Image.fromarray(new)

    except Exception as e:
        st.error(f"İşleme hatası: {e}")
        return None

def save_large_image(image, filename):
    """PNG olarak kaydeder."""
    try:
        image.save(filename, "PNG", dpi=(300, 300))
    except Exception as e:
        st.error(f"Kaydetme hatası: {e}")

def main():
    st.title("Stone Pattern Generator")

    uploaded_file = st.file_uploader("Bir TIFF dosyası seçin...", type=["tif", "tiff"])
    if uploaded_file:
        st.write(f"Dosya: {uploaded_file.name}  •  Boyut: {uploaded_file.size/(1024*1024):.2f} MB")
        with st.spinner("Görüntü yükleniyor..."):
            image = load_large_image(uploaded_file)

        if image:
            st.image(image, caption="Orijinal Görüntü", use_column_width=True)

            st.sidebar.header("Desen Kontrolleri")
            redistribution_intensity = st.sidebar.slider(
                "Dağıtım Şiddeti", 0.1, 1.0, 0.5,
                help="Pulların ne kadar uzaklığa taşınacağını ayarlar"
            )
            flake_size_range = st.sidebar.slider(
                "Pul Boyutu Aralığı", 0.5, 2.0, 1.0,
                help="Algılanan pul bölgelerinin boyutunu ayarlar"
            )
            color_sensitivity = st.sidebar.slider(
                "Renk Duyarlılığı", 0.1, 1.0, 0.5,
                help="Renk varyansına ne kadar duyarlı olunacağını belirler"
            )

            if st.button("Yeni Tasarım Oluştur"):
                with st.spinner("Tasarım oluşturuluyor..."):
                    variation = detect_and_move_flakes(
                        image,
                        redistribution_intensity,
                        flake_size_range,
                        color_sensitivity
                    )
                    if variation:
                        os.makedirs("generated_images", exist_ok=True)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        fn = f"generated_images/variation_{ts}.png"
                        save_large_image(variation, fn)
                        st.image(variation, caption="Yeni Tasarım", use_column_width=True)
                        with open(fn, "rb") as f:
                            st.download_button(
                                label="Tasarımı İndir",
                                data=f,
                                file_name=f"new_design_{ts}.png",
                                mime="image/png"
                            )

if __name__ == "__main__":
    main()
