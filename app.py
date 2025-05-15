import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
import gc
from libtiff import TIFF

def load_large_image(uploaded_file):
    """Handle large TIFF files using libtiff"""
    try:
        # Save uploaded file temporarily
        temp_path = "temp_image.tif"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Open TIFF file
        tiff = TIFF.open(temp_path, mode='r')
        
        # Read image
        image = tiff.read_image()
        
        # Close TIFF
        tiff.close()
        
        # Remove temporary file
        os.remove(temp_path)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:  # If grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] > 3:  # If RGBA or more channels
            image = image[:, :, :3]
            
        return Image.fromarray(image)
            
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

# Rest of your code stays the same...
