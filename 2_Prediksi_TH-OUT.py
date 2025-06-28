import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import os

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Prediksi TH-OUT", layout="centered")
st.title("🚀 Prediksi Suhu TH-OUT")
st.markdown("Halaman ini memungkinkan Anda untuk memprediksi suhu TH-OUT menggunakan model jaringan Syaraf Tiruan yang sudah dilatih.")

# --- Lokasi Model dan Scaler ---
MODEL_SAVE_DIR = '.' # Asumsikan file berada di direktori root project
MODEL_WEIGHTS_PATH = os.path.join(MODEL_SAVE_DIR, 'ann_model.weights.h5')
SCALER_X_PATH = os.path.join(MODEL_SAVE_DIR, 'scaler_X.pkl')
SCALER_Y_PATH = os.path.join(MODEL_SAVE_DIR, 'scaler_y.pkl')

# --- 1. Definisi Arsitektur Model (HARUS SAMA dengan saat pelatihan) ---
def create_model(window_size):
    """
    Membangun arsitektur model ANN yang sama dengan yang dilatih.
    """
    model = Sequential([
        Flatten(input_shape=(window_size, 1)), # window_size timesteps, 1 feature (TH-IN)
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.1),
        Dense(1) # Output tunggal untuk regresi TH-OUT
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# --- 2. Memuat Model dan Scaler ---

# Tentukan window_size yang sama seperti saat pelatihan
window_size = 5

@st.cache_resource # Cache resources to load only once
def load_all_resources(model_dir, window_size):
    """
    Loads the trained model weights and the MinMaxScaler objects.
    """
    scaler_X = None
    scaler_y = None
    model = None

    # Load Scaler X
    try:
        with open(SCALER_X_PATH, 'rb') as f:
            scaler_X = pickle.load(f)
        st.sidebar.success(f"✅ Scaler X berhasil dimuat dari `{SCALER_X_PATH}`")
    except FileNotFoundError:
        st.sidebar.error(f"❌ Scaler X tidak ditemukan di `{SCALER_X_PATH}`. Harap jalankan 'EDA & Pelatihan Model' terlebih dahulu!")
        st.stop()
    except Exception as e:
        st.sidebar.error(f"❌ Error memuat Scaler X: {e}")
        st.stop()

    # Load Scaler Y
    try:
        with open(SCALER_Y_PATH, 'rb') as f:
            scaler_y = pickle.load(f)
        st.sidebar.success(f"✅ Scaler Y berhasil dimuat dari `{SCALER_Y_PATH}`")
    except FileNotFoundError:
        st.sidebar.error(f"❌ Scaler Y tidak ditemukan di `{SCALER_Y_PATH}`. Harap jalankan 'EDA & Pelatihan Model' terlebih dahulu!")
        st.stop()
    except Exception as e:
        st.sidebar.error(f"❌ Error memuat Scaler Y: {e}")
        st.stop()

    # Create model instance and load weights
    try:
        model = create_model(window_size)
        model.build(input_shape=(None, window_size, 1)) # Build model before loading weights
        model.load_weights(MODEL_WEIGHTS_PATH)
        st.sidebar.success(f"✅ Bobot model berhasil dimuat dari `{MODEL_WEIGHTS_PATH}`")
    except FileNotFoundError:
        st.sidebar.error(f"❌ Bobot model tidak ditemukan di `{MODEL_WEIGHTS_PATH}`. Harap jalankan 'EDA & Pelatihan Model' terlebih dahulu!")
        st.stop()
    except Exception as e:
        st.sidebar.error(f"❌ Error memuat bobot model: {e}")
        st.stop()

    return scaler_X, scaler_y, model

# Load resources (this will run only once)
scaler_X, scaler_y, model = load_all_resources(MODEL_SAVE_DIR, window_size)

# --- 3. Streamlit User Interface for Prediction ---

st.header(f"Masukkan {window_size} Nilai TH-IN Terakhir")
st.write(f"Harap masukkan {window_size} nilai `TH-IN` terbaru secara berurutan.")

# List to store TH-IN inputs
th_in_values = []

# Create input fields for window_size TH-IN values
for i in range(window_size):
    label_suffix = f" (t-{window_size - 1 - i})" if i < window_size - 1 else " (t-0, Terbaru)"
    value = st.number_input(
        f"TH-IN ke-{i+1}{label_suffix}:",
        min_value=0.0,
        max_value=150.0, # Adjust based on your training data's max TH-IN
        value=50.0 + i*0.5, # Default values
        step=0.1,
        format="%.2f",
        key=f"predict_th_in_input_{i}" # Unique key for each widget
    )
    th_in_values.append(value)

st.info("Nilai TH-IN yang dimasukkan: " + ", ".join([f"{v:.2f}°C" for v in th_in_values]))

# Prediction button
if st.button("🚀 Prediksi TH-OUT"):
    if all(val is not None for val in th_in_values):
        try:
            with st.spinner("Memproses prediksi..."):
                # Convert list of TH-IN values to NumPy array
                input_sequence_raw = np.array(th_in_values)

                # Reshape to (window_size, 1) as scaler expects 2D array for single feature
                input_sequence_for_scaler = input_sequence_raw.reshape(-1, 1)

                # Normalize input using the loaded scaler_X
                input_scaled = scaler_X.transform(input_sequence_for_scaler)

                # Reshape for model input: (1, window_size, 1)
                input_reshaped = input_scaled.reshape(1, window_size, 1)

                # Make the prediction
                prediction_scaled = model.predict(input_reshaped, verbose=0)[0][0]

                # Inverse transform the prediction to get the original scale
                prediction_th_out_original = scaler_y.inverse_transform([[prediction_scaled]])[0][0]

                st.success(f"**Prediksi Nilai TH-OUT:** **{prediction_th_out_original:.2f} °C**")
                st.balloons()

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
            st.warning("Pastikan input Anda valid dan model berfungsi dengan benar.")
    else:
        st.warning("Harap masukkan semua nilai TH-IN untuk melakukan prediksi.")

st.markdown("---")
st.caption("Aplikasi ini dibangun menggunakan Streamlit, TensorFlow, dan scikit-learn.")