# app/main_streamlit.py
import streamlit as st
from src.inference import SkinClassifier
from PIL import Image
import io

st.set_page_config(page_title="Skin Disease Detector", layout='centered')
st.title('🩺 Skin Disease Detector')
st.write('Upload an image of the affected skin area (hand, arm, face, etc.)')

@st.cache_resource
def load_model():
    return SkinClassifier()

model = load_model()

uploaded = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
if uploaded is not None:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption='Uploaded image', use_column_width=True)
    with st.spinner('Analyzing...'):
        out = model.predict_from_pil(img)
    st.markdown('**Prediction:**')
    st.write(f"Class: {out['class_name']} — {out['disease_name']}")
    st.write(f"Confidence: {out['confidence']:.2f}")
    st.write('**Problem:**')
    st.write(out['problem'])
    st.write('**Suggested solution / next steps:**')
    st.write(out['solution'])
    st.info('This tool is for educational purposes only. Not a medical diagnosis. Consult a healthcare professional for medical advice.')
