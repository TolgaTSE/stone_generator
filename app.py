def load_large_image(uploaded_file):
    """Handle CMYK TIFF files with proper color conversion"""
    try:
        # Save to temporary file
        temp_path = "temp.tif"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Open with PIL first to handle CMYK properly
        with Image.open(temp_path) as img:
            # Convert CMYK to RGB
            if img.mode == 'CMYK':
                img = img.convert('RGB')
            # Load the image
            img.load()
            # Make a copy
            image = img.copy()
        
        # Remove temporary file
        os.remove(temp_path)
        
        # Print image information
        st.write(f"Image size: {image.size}")
        st.write(f"Image mode: {image.mode}")
        
        return image
            
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        st.error(f"File size: {uploaded_file.size / (1024*1024):.2f} MB")
        
        # Try alternative method
        try:
            # Try direct PIL opening from buffer
            image = Image.open(io.BytesIO(uploaded_file.getvalue()))
            if image.mode == 'CMYK':
                image = image.convert('RGB')
            return image
            
        except Exception as e2:
            st.error(f"Alternative method also failed: {str(e2)}")
            return None
