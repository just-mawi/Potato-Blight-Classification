import onnx
import tf2onnx
import tensorflow as tf
import numpy as np

class MacroF1Score(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='macro_f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(name='tp', shape=(num_classes,), initializer='zeros')
        self.fp = self.add_weight(name='fp', shape=(num_classes,), initializer='zeros')
        self.fn = self.add_weight(name='fn', shape=(num_classes,), initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None): pass
    def result(self): return 0.0
    def reset_states(self): pass
    def get_config(self):
        cfg = super().get_config()
        cfg['num_classes'] = self.num_classes
        return cfg

model = tf.keras.models.load_model(
    'best_potato_model.h5',
    custom_objects={'MacroF1Score': MacroF1Score},
    compile=False
)

input_sig = [tf.TensorSpec([None, 224, 224, 3], tf.float32, name='input')]

# --- Full prediction model ---
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_sig, opset=13)
with open('model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())
print('Saved model.onnx')

# --- Feature extraction model ---
mnv2 = model.get_layer('mobilenetv2_1.00_224')
feat_model = tf.keras.Model(inputs=mnv2.input, outputs=mnv2.output)
onnx_feat, _ = tf2onnx.convert.from_keras(feat_model, input_signature=input_sig, opset=13)
with open('model_features.onnx', 'wb') as f:
    f.write(onnx_feat.SerializeToString())
print('Saved model_features.onnx')

# --- Head weights (BN + Dense layers) for numpy Grad-CAM backprop ---
bn_layer     = next(l for l in model.layers if isinstance(l, tf.keras.layers.BatchNormalization))
dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
assert len(dense_layers) == 3, f"Expected 3 Dense layers, got {len(dense_layers)}"

np.save('head_weights.npy', {
    'bn_gamma': bn_layer.gamma.numpy(),
    'bn_beta':  bn_layer.beta.numpy(),
    'bn_mean':  bn_layer.moving_mean.numpy(),
    'bn_var':   bn_layer.moving_variance.numpy(),
    'bn_eps':   bn_layer.epsilon,
    'W1': dense_layers[0].kernel.numpy(),  # (1280, 256)
    'b1': dense_layers[0].bias.numpy(),
    'W2': dense_layers[1].kernel.numpy(),  # (256, 128)
    'b2': dense_layers[1].bias.numpy(),
    'W3': dense_layers[2].kernel.numpy(),  # (128, 3)
    'b3': dense_layers[2].bias.numpy(),
})
print('Saved head_weights.npy')
print('  W1:', dense_layers[0].kernel.shape,
      ' W2:', dense_layers[1].kernel.shape,
      ' W3:', dense_layers[2].kernel.shape)
