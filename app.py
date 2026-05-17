import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image
from matplotlib import cm as mpl_cm

IMG_SIZE      = 224
DISPLAY_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

@st.cache_resource
def load_models():
    pred_sess = ort.InferenceSession('model.onnx')
    feat_sess = ort.InferenceSession('model_features.onnx')
    w = np.load('head_weights.npy', allow_pickle=True).item()
    return pred_sess, feat_sess, w

def preprocess(pil_img):
    arr = np.array(pil_img.convert('RGB').resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
    return np.expand_dims((arr / 127.5) - 1.0, 0)

def head_forward(gap, w):
    """Forward pass through BN → Dense(256) → Dense(128) → Dense(3)."""
    x = (gap - w['bn_mean']) / np.sqrt(w['bn_var'] + w['bn_eps']) * w['bn_gamma'] + w['bn_beta']
    pre1 = x @ w['W1'] + w['b1']
    z1   = np.maximum(pre1, 0)
    pre2 = z1 @ w['W2'] + w['b2']
    z2   = np.maximum(pre2, 0)
    logits = z2 @ w['W3'] + w['b3']
    return logits, (pre1, z1, pre2, z2)

def gradcam_weights(pred_idx, w, acts):
    """Backprop gradient of logit[pred_idx] through head → GAP output."""
    pre1, z1, pre2, z2 = acts
    d = w['W3'][:, pred_idx]               # (128,)
    d = d * (pre2 > 0)                     # ReLU2 backward
    d = w['W2'] @ d                        # (256,)
    d = d * (pre1 > 0)                     # ReLU1 backward
    d = w['W1'] @ d                        # (1280,)
    d = d * w['bn_gamma'] / np.sqrt(w['bn_var'] + w['bn_eps'])  # BN backward
    return d                               # α_k for each MobileNetV2 channel

def gradcam_overlay(features, weights, pil_img, alpha=0.45):
    """Weighted feature map → heatmap → overlay."""
    heatmap = np.maximum(features @ weights, 0)          # (7, 7)
    heatmap /= heatmap.max() + 1e-8
    heatmap_up = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8))
              .resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    ) / 255.0
    colored = (mpl_cm.get_cmap('jet')(heatmap_up)[:, :, :3] * 255).astype(np.float32)
    orig    = np.array(pil_img.convert('RGB').resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
    return Image.fromarray((alpha * colored + (1 - alpha) * orig).astype(np.uint8))

st.set_page_config(page_title='Potato Blight Classifier', page_icon='\U0001f954', layout='centered')
st.title('\U0001f954 Potato Blight Classifier')
st.markdown('Upload a potato leaf image to detect **Early Blight**, **Late Blight**, or a **Healthy** plant.')

pred_sess, feat_sess, head_w = load_models()
pred_input = pred_sess.get_inputs()[0].name
feat_input = feat_sess.get_inputs()[0].name

uploaded = st.file_uploader('Choose a leaf image (.jpg / .png)', type=['jpg', 'jpeg', 'png'])

if uploaded:
    img = Image.open(uploaded)
    arr = preprocess(img)

    probs      = pred_sess.run(None, {pred_input: arr})[0][0]
    pred_idx   = int(np.argmax(probs))
    pred_label = DISPLAY_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    features   = feat_sess.run(None, {feat_input: arr})[0][0]  # (7, 7, 1280)
    gap        = features.mean(axis=(0, 1))                     # (1280,)
    _, acts    = head_forward(gap, head_w)
    alpha_k    = gradcam_weights(pred_idx, head_w, acts)
    cam_img    = gradcam_overlay(features, alpha_k, img)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img,     caption='Uploaded image', use_container_width=True)
    with col2:
        st.image(cam_img, caption='Grad-CAM',       use_container_width=True)

    if pred_label == 'Healthy':
        st.success(f'**{pred_label}** — {confidence:.1%} confidence')
    else:
        st.error(f'**{pred_label}** — {confidence:.1%} confidence')

    st.subheader('Class Probabilities')
    for name, prob in zip(DISPLAY_NAMES, probs):
        st.write(f'**{name}** — {prob:.1%}')
        st.progress(float(prob))
