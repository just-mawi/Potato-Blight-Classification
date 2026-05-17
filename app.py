import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image

IMG_SIZE      = 224
DISPLAY_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

@st.cache_resource
def load_session():
    return ort.InferenceSession('model.onnx')

def preprocess(pil_img):
    arr = np.array(pil_img.convert('RGB').resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
    arr = (arr / 127.5) - 1.0  # MobileNetV2 scale to [-1, 1]
    return np.expand_dims(arr, 0)

st.set_page_config(page_title='Potato Blight Classifier', page_icon='\U0001f954', layout='centered')
st.title('\U0001f954 Potato Blight Classifier')
st.markdown('Upload a potato leaf image to detect **Early Blight**, **Late Blight**, or a **Healthy** plant.')

session   = load_session()
input_name = session.get_inputs()[0].name

uploaded = st.file_uploader('Choose a leaf image (.jpg / .png)', type=['jpg', 'jpeg', 'png'])

if uploaded:
    img  = Image.open(uploaded)
    arr  = preprocess(img)

    probs      = session.run(None, {input_name: arr})[0][0]
    pred_idx   = int(np.argmax(probs))
    pred_label = DISPLAY_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    st.image(img, caption='Uploaded image', use_container_width=True)

    if pred_label == 'Healthy':
        st.success(f'**{pred_label}** — {confidence:.1%} confidence')
    else:
        st.error(f'**{pred_label}** — {confidence:.1%} confidence')

    st.subheader('Class Probabilities')
    for name, prob in zip(DISPLAY_NAMES, probs):
        st.write(f'**{name}** — {prob:.1%}')
        st.progress(float(prob))
