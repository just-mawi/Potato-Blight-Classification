import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import cm as mpl_cm

keras = tf.keras
CHECKPOINT_PATH = 'best_potato_model.h5'
IMG_SIZE        = 224
DISPLAY_NAMES   = ['Early Blight', 'Late Blight', 'Healthy']

class MacroF1Score(keras.metrics.Metric):
    def __init__(self, num_classes, name='macro_f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(name='tp', shape=(num_classes,), initializer='zeros')
        self.fp = self.add_weight(name='fp', shape=(num_classes,), initializer='zeros')
        self.fn = self.add_weight(name='fn', shape=(num_classes,), initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_cls = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        y_true_i32 = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred_oh  = tf.one_hot(y_pred_cls, self.num_classes)
        y_true_oh  = tf.one_hot(y_true_i32, self.num_classes)
        self.tp.assign_add(tf.reduce_sum(y_pred_oh * y_true_oh,        axis=0))
        self.fp.assign_add(tf.reduce_sum(y_pred_oh * (1 - y_true_oh),  axis=0))
        self.fn.assign_add(tf.reduce_sum((1 - y_pred_oh) * y_true_oh,  axis=0))
    def result(self):
        eps = 1e-7
        precision = self.tp / (self.tp + self.fp + eps)
        recall    = self.tp / (self.tp + self.fn + eps)
        f1_per    = 2 * precision * recall / (precision + recall + eps)
        return tf.reduce_mean(f1_per)
    def reset_states(self):
        self.tp.assign(tf.zeros(self.num_classes))
        self.fp.assign(tf.zeros(self.num_classes))
        self.fn.assign(tf.zeros(self.num_classes))
    def get_config(self):
        cfg = super().get_config()
        cfg['num_classes'] = self.num_classes
        return cfg

@st.cache_resource
def load_models():
    model = keras.models.load_model(
        CHECKPOINT_PATH, custom_objects={'MacroF1Score': MacroF1Score}, compile=False)
    mnv2 = model.get_layer('mobilenetv2_1.00_224')
    gradcam_base = keras.Model(
        inputs=mnv2.input,
        outputs=[mnv2.get_layer('Conv_1').output, mnv2.output])
    head_input = keras.Input(shape=mnv2.output.shape[1:])
    hx = head_input
    for layer in model.layers:
        if not isinstance(layer, keras.Model):
            hx = layer(hx, training=False)
    head_model = keras.Model(inputs=head_input, outputs=hx)
    return model, gradcam_base, head_model

def preprocess(pil_img):
    arr = np.array(pil_img.convert('RGB').resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
    return keras.applications.mobilenet_v2.preprocess_input(arr)

def get_gradcam_overlay(arr, gradcam_base, head_model, pred_index, pil_img, alpha=0.45):
    img_batch = np.expand_dims(arr, 0)
    with tf.GradientTape() as tape:
        conv_outputs, base_out = gradcam_base(img_batch, training=False)
        tape.watch(conv_outputs)
        preds = head_model(base_out, training=False)
        class_channel = preds[:, pred_index]
    grads        = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap      = tf.squeeze(tf.nn.relu(conv_outputs[0] @ pooled_grads[..., tf.newaxis]))
    heatmap      = (heatmap / (tf.reduce_max(heatmap) + 1e-8)).numpy()
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    ) / 255.0
    colored = (mpl_cm.get_cmap('jet')(heatmap_resized)[:, :, :3] * 255).astype(np.float32)
    orig    = np.array(pil_img.convert('RGB').resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
    return Image.fromarray((alpha * colored + (1 - alpha) * orig).astype(np.uint8))

st.set_page_config(page_title='Potato Blight Classifier', page_icon='\U0001f954', layout='centered')
st.title('\U0001f954 Potato Blight Classifier')
st.markdown('Upload a potato leaf image to detect **Early Blight**, **Late Blight**, or a **Healthy** plant.')

model, gradcam_base, head_model = load_models()
uploaded = st.file_uploader('Choose a leaf image (.jpg / .png)', type=['jpg', 'jpeg', 'png'])

if uploaded:
    img  = Image.open(uploaded)
    arr  = preprocess(img)
    probs      = model(np.expand_dims(arr, 0), training=False).numpy()[0]
    pred_idx   = int(np.argmax(probs))
    pred_label = DISPLAY_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    cam_img = get_gradcam_overlay(arr, gradcam_base, head_model, pred_idx, img)
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
