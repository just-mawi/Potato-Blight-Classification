"""
Run this once locally to convert best_potato_model.h5 -> model.onnx
Requires: pip install tf2onnx tensorflow==2.3.1  (or whichever TF you trained with)
"""
import onnx  # must import before TF to avoid DLL conflict on Windows
import tf2onnx
import tensorflow as tf

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

input_signature = [tf.TensorSpec([None, 224, 224, 3], tf.float32, name='input')]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)

with open('model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

print('Done — model.onnx saved.')
