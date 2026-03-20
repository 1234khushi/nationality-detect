import streamlit as st
import tempfile
import os
import numpy as np
from PIL import Image

from utils.prediction import final_prediction

# ------------------ SESSION STATE ------------------
if "result" not in st.session_state:
    st.session_state.result = None

# ------------------ UI ------------------
st.title("Nationality Detection System")
st.write("Upload an image to detect nationality, emotion, age, and dress color.")

# ------------------ UPLOAD SECTION ------------------
st.header("Upload Image")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Show image preview
    image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image)

    st.image(image_rgb, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict", type="primary"):

        with st.spinner("Processing..."):
            try:
                result = final_prediction(image_rgb)

                st.session_state.result = result

            except Exception as e:
                st.error(f"Error during prediction: {e}")
                if "libGL.so.1" in str(e) or "headless OpenCV" in str(e):
                    st.info(
                        "Deployment fix: install 'opencv-python-headless' and remove 'opencv-python' "
                        "in Linux/server environments, or install the OS package that provides libGL."
                    )

# ------------------ RESULT DISPLAY ------------------
if st.session_state.result is not None:

    st.subheader("Prediction Result")

    result = st.session_state.result

    try:
        st.write(f"**Nationality:** {result.get('Nationality', 'N/A')}")
        st.write(f"**Emotion:** {result.get('Emotion', 'N/A')}")

        if "Age" in result:
            st.write(f"**Age:** {result['Age']}")

        if "Dress Color" in result:
            st.write(f"**Dress Color:** {result['Dress Color']}")

    except Exception as e:
        st.error(f"Error displaying result: {e}")

# ------------------ FOOTER ------------------
st.markdown("*Nationality Detection System | Image-based AI Pipeline*")
