import streamlit as st
from PIL import Image
from src.common import load_model_bundle, predict_image

st.set_page_config(page_title="Chest X-ray Classifier", page_icon="🩻")
st.title("🩻 Pneumonia Detection from Chest X-rays")
st.caption("Educational demonstration only — not for medical diagnosis.")

@st.cache_resource
def get_model():
    return load_model_bundle()

uploaded = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    if st.button("Analyze image", type="primary"):
        try:
            model = get_model()
            label, probability = predict_image(model, image)
            st.subheader(f"Model output: {label}")
            st.write(f"Model score for PNEUMONIA: {probability:.1%}")
            st.warning("This prediction can be wrong. Do not use it for healthcare decisions.")
        except Exception as exc:
            st.error(str(exc))
else:
    st.info("Train the model first, then upload an image to try the demo.")
