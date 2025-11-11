import joblib
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
import math
from datetime import timedelta

# === Attention 層 ===
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)
    
    def get_config(self):
        config = super().get_config()
        return config

# === 模型 lazy load ===
linear_model = joblib.load("../ridge_regression_baseline.pkl")
lstm_model = None
bilstm_model = None

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def _load_lstm():
    global lstm_model
    if lstm_model is None:
        lstm_model = tf.keras.models.load_model(
            "../lstm_model.keras",
            custom_objects={"Attention": Attention}
        )
    return lstm_model

def _load_bilstm():
    global bilstm_model
    if bilstm_model is None:
        bilstm_model = tf.keras.models.load_model("../bi_lstm_model.keras")
    return bilstm_model

# === 單步預測（可選保留）===
def predict_with_lstm(seq_inputs, static_inputs) -> float:
    model = _load_lstm()
    seq_array = np.array(seq_inputs, dtype=np.float32).reshape((1, 4, 2))
    static_array = np.array(static_inputs, dtype=np.float32).reshape((1, 6))
    return float(model.predict([seq_array, static_array])[0][0])

def predict_with_bilstm(seq_inputs, static_inputs) -> float:
    model = _load_bilstm()
    seq_array = np.array(seq_inputs, dtype=np.float32).reshape((1, 4, 2))
    static_array = np.array(static_inputs, dtype=np.float32).reshape((1, 6))
    return float(model.predict([seq_array, static_array])[0][0])

def predict_with_lr(flat_seq, log_predict_days, age_ratio) -> float:
    X = np.array([flat_seq + [log_predict_days, age_ratio]], dtype=np.float32)
    return float(linear_model.predict(X.reshape(1, -1))[0])

# === 多步遞迴預測 ===
def multi_step_recursive_predict(model, seq_data, base_static, birth_date, breed_name, breed_lifespan,
                                 n_steps=6, step_days=15, last_real_weight=0.0, last_mea_date=None):
    preds, target_dates = [], []
    seq_local = [[float(w), int(d)] for w, d in seq_data]
    last_days = seq_local[-1][1]
    # 先補一個「起點」
    preds.append(last_real_weight)
    target_dates.append(last_mea_date)

    for step in range(1, n_steps+1):
        next_days = last_days + step_days
        target_date = birth_date + timedelta(days=next_days)

        max_life_days_local = breed_lifespan.get(breed_name, 365*10)
        log_predict_days_local = math.log(next_days + 1)
        age_ratio_local = next_days / max_life_days_local
        static_local = [
            base_static[0], base_static[1], base_static[2],
            log_predict_days_local, age_ratio_local, base_static[5]
        ]

        seq_input = np.array([seq_local], dtype=np.float32)
        static_input = np.array([static_local], dtype=np.float32)

        pred = float(model.predict([seq_input, static_input], verbose=0)[0][0])
        pred = max(0.0, pred)

        preds.append(pred)
        target_dates.append(target_date)

        seq_local.append([pred, next_days])
        if len(seq_local) > 4:
            seq_local.pop(0)

        last_days = next_days

    return preds, target_dates

def multi_step_lr_predict(flat_seq, base_static, birth_date, breed_name, breed_lifespan,
                          n_steps=6, step_days=15, last_real_weight=0.0, last_mea_date=None):
    preds, target_dates = [], []
    seq_local = flat_seq.copy()
    last_days = 0
    # 先補一個「起點」
    preds.append(last_real_weight)
    target_dates.append(last_mea_date)

    seq_local = flat_seq.copy()
    seq_local = seq_local[1:] + [last_real_weight]   # 把最後真實 weight 放進序列

    last_days = (last_mea_date - birth_date).days    # 從最後量測的天數開始

    for step in range(1, n_steps+1):
        next_days = last_days + step_days
        max_life_days_local = breed_lifespan.get(breed_name, 365*10)
        log_predict_days_local = math.log(next_days + 1)
        age_ratio_local = next_days / max_life_days_local

        X_lr_local = np.array([seq_local + [log_predict_days_local, age_ratio_local]], dtype=np.float32)
        p = float(linear_model.predict(X_lr_local.reshape(1, -1))[0])
        preds.append(max(0.0, p))
        target_dates.append(birth_date + timedelta(days=next_days))

        seq_local = seq_local[1:] + [p]
        last_days = next_days

    return preds, target_dates
