import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns # Untuk EDA yang lebih baik
import pickle # Untuk menyimpan dan memuat scaler
import os # Untuk membuat direktori
import time # Untuk simulasi real-time
from collections import deque # Untuk menyimpan data historis dengan ukuran tetap
from io import StringIO # Untuk menangkap output summary model

# --- Konfigurasi Halaman Umum ---
st.set_page_config(
    page_title="Aplikasi Analisis Reaktor Nuklir",
    page_icon="⚛️",
    layout="wide"
)

# --- Lokasi File Data dan Model ---
DATA_PATH = 'Dataset_Nuklir(Coba).xlsx'
MODEL_SAVE_DIR = '.' # Simpan di direktori root project
MODEL_WEIGHTS_PATH = os.path.join(MODEL_SAVE_DIR, 'ann_model_weights.weights.h5')
SCALER_X_PATH = os.path.join(MODEL_SAVE_DIR, 'scaler_X.pkl')
SCALER_Y_PATH = os.path.join(MODEL_SAVE_DIR, 'scaler_y.pkl')

# --- Fungsi Utility Umum (dari kode pelatihan Anda) ---
@st.cache_data
def load_data(path):
    """Memuat data dari file Excel dan melakukan pembersihan awal."""
    try:
        data = pd.read_excel(path)
        data = data[['TH-IN', 'TH-OUT']]
        data_cleaned = data.dropna(subset=['TH-IN', 'TH-OUT'])
        return data_cleaned
    except FileNotFoundError:
        st.error(f"File data '{path}' tidak ditemukan. Pastikan sudah diunggah.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat atau membersihkan data: {e}")
        st.stop()

def create_dataset(X, y, window_size):
    """Membuat dataset dengan sliding window."""
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i + window_size])
        ys.append(y[i + window_size])
    return np.array(Xs), np.array(ys)

def create_ann_model(window_size):
    """Membangun arsitektur model ANN."""
    model = Sequential([
        Flatten(input_shape=(window_size, 1)),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.1),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

# --- 2. Memuat Model dan Scaler (Hanya sekali, di-cache) ---
window_size = 5 # Definisi window_size global

@st.cache_resource
def load_all_resources(model_dir, window_size):
    scaler_X, scaler_y, model = None, None, None
    try:
        with open(os.path.join(model_dir, 'scaler_X.pkl'), 'rb') as f:
            scaler_X = pickle.load(f)
        with open(os.path.join(model_dir, 'scaler_y.pkl'), 'rb') as f:
            scaler_y = pickle.load(f)
        st.sidebar.success("✅ Scalers berhasil dimuat.")
    except FileNotFoundError:
        st.sidebar.warning("Scalers tidak ditemukan. Harap latih model terlebih dahulu.")
    except Exception as e:
        st.sidebar.error(f"❌ Error memuat Scalers: {e}")

    try:
        model = create_ann_model(window_size)
        model.build(input_shape=(None, window_size, 1))
        model.load_weights(os.path.join(model_dir, 'ann_model_weights.weights.h5'))
        st.sidebar.success("✅ Bobot model berhasil dimuat.")
    except FileNotFoundError:
        st.sidebar.warning("Bobot model tidak ditemukan. Harap latih model terlebih dahulu.")
    except Exception as e:
        st.sidebar.error(f"❌ Error memuat bobot model: {e}")

    return scaler_X, scaler_y, model

# Load resources globally
scaler_X, scaler_y, model = load_all_resources(MODEL_SAVE_DIR, window_size)


# --- UI Navigasi ---
st.sidebar.title("Navigasi Aplikasi")
page_selection = st.sidebar.radio("Pilih Halaman:", ["🏠 Beranda", "📊 EDA & Pelatihan Model", "🚀 Prediksi TH-OUT"])

# --- Konten Halaman ---

if page_selection == "🏠 Beranda":
    st.title("Selamat Datang di Aplikasi Analisis Reaktor Nuklir")
    st.write("""
    Aplikasi ini menyediakan dua fungsi utama:
    1.  **Analisis Data Eksploratif (EDA) & Pelatihan Model**: Latih ulang model JST Anda dan lihat visualisasi data.
    2.  **Prediksi Suhu TH-OUT**: Gunakan model yang sudah dilatih untuk memprediksi suhu TH-OUT berdasarkan input TH-IN.
    """)
    st.info("Pilih halaman dari sidebar di sebelah kiri untuk memulai.")
    st.markdown("---")
    st.caption("Aplikasi ini dibangun menggunakan Streamlit, TensorFlow, dan scikit-learn.")

elif page_selection == "📊 EDA & Pelatihan Model":
    st.title("📊 Analisis Data Eksploratif (EDA) & Pelatihan Model JST")
    st.markdown("Halaman ini memungkinkan Anda untuk melihat statistik data, visualisasi, dan melatih ulang model JST.")

    st.header("1. Memuat dan Membersihkan Data")
    data_cleaned = load_data(DATA_PATH)
    st.write("### Data Mentah (Setelah Pembersihan Missing Values)")
    st.dataframe(data_cleaned.head())
    st.write(f"Jumlah baris setelah membersihkan missing values: **{len(data_cleaned)}**")

    # --- EDA Section ---
    st.header("2. Analisis Data Eksploratif (EDA)")

    st.subheader("Statistik Deskriptif")
    st.write(data_cleaned.describe())

    st.subheader("Visualisasi Data")

    # Plot 1: Fluktuasi Suhu TH-IN dan TH-OUT terhadap Waktu
    st.write("#### Fluktuasi Suhu TH-IN dan TH-OUT terhadap Waktu (Sebelum Normalisasi)")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(data_cleaned['TH-IN'], label='TH-IN', color='blue', alpha=0.7)
    ax1.plot(data_cleaned['TH-OUT'], label='TH-OUT', color='red', alpha=0.7)
    ax1.set_xlabel('Indeks Waktu')
    ax1.set_ylabel('Suhu (°C)')
    ax1.set_title('Fluktuasi Suhu TH-IN dan TH-OUT')
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    # Plot 2: Distribusi TH-IN dan TH-OUT
    st.write("#### Distribusi Suhu TH-IN dan TH-OUT")
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(data_cleaned['TH-IN'], kde=True, ax=axes[0], color='blue')
    axes[0].set_title('Distribusi TH-IN')
    axes[0].set_xlabel('Suhu TH-IN (°C)')
    axes[0].set_ylabel('Frekuensi')

    sns.histplot(data_cleaned['TH-OUT'], kde=True, ax=axes[1], color='red')
    axes[1].set_title('Distribusi TH-OUT')
    axes[1].set_xlabel('Suhu TH-OUT (°C)')
    axes[1].set_ylabel('Frekuensi')
    st.pyplot(fig2)

    # Plot 3: Scatter Plot TH-IN vs TH-OUT
    st.write("#### Hubungan TH-IN dan TH-OUT (Scatter Plot)")
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='TH-IN', y='TH-OUT', data=data_cleaned, alpha=0.5, ax=ax3, color='purple')
    ax3.set_xlabel('Suhu TH-IN (°C)')
    ax3.set_ylabel('Suhu TH-OUT (°C)')
    ax3.set_title('Scatter Plot TH-IN vs TH-OUT')
    ax3.grid(True)
    st.pyplot(fig3)

    # Korelasi
    st.subheader("Korelasi Antar Fitur")
    correlation_matrix = data_cleaned.corr()
    st.write(correlation_matrix)
    fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
    ax_corr.set_title('Matriks Korelasi')
    st.pyplot(fig_corr)


    # --- Normalisasi Data dan Proses Pelatihan ---
    st.header("3. Normalisasi Data dan Pelatihan Model Deep Learning")

    if st.button("Mulai Normalisasi & Pelatihan Model"):
        st.write("#### Normalisasi Data")
        X = data_cleaned[['TH-IN']]
        y = data_cleaned['TH-OUT']

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

        # Visualisasi Data Setelah Normalisasi
        data_normalized_df = pd.DataFrame(X_scaled, columns=['TH-IN'])
        data_normalized_df['TH-OUT'] = y_scaled

        fig_norm, ax_norm = plt.subplots(figsize=(12, 6))
        ax_norm.plot(data_normalized_df['TH-IN'], label='TH-IN (Normalized)', color='blue', alpha=0.7)
        ax_norm.plot(data_normalized_df['TH-OUT'], label='TH-OUT (Normalized)', color='red', alpha=0.7)
        ax_norm.set_xlabel('Indeks Waktu')
        ax_norm.set_ylabel('Suhu (Normalized)')
        ax_norm.set_title('Fluktuasi Suhu TH-IN dan TH-OUT setelah Normalisasi')
        ax_norm.legend()
        ax_norm.grid(True)
        st.pyplot(fig_norm)

        # Simpan scalers
        try:
            with open(SCALER_X_PATH, 'wb') as f:
                pickle.dump(scaler_X, f)
            with open(SCALER_Y_PATH, 'wb') as f:
                pickle.dump(scaler_y, f)
            st.success(f"✅ Scaler X dan Y berhasil disimpan di `{MODEL_SAVE_DIR}`.")
        except Exception as e:
            st.error(f"❌ Gagal menyimpan scalers: {e}")

        st.write("#### Membuat Dataset dengan Sliding Window")
        X_sliding, y_sliding = create_dataset(X_scaled, y_scaled.flatten(), window_size)
        st.write(f"Bentuk X_sliding: `{X_sliding.shape}` (samples, window_size, features)")
        st.write(f"Bentuk y_sliding: `{y_sliding.shape}` (samples,)")

        st.write("#### Pembagian Data (Training dan Testing)")
        split = int(len(X_sliding) * 0.9)
        X_train, X_test = X_sliding[:split], X_sliding[split:]
        y_train, y_test = y_sliding[:split], y_sliding[split:]
        st.write(f"Bentuk X_train: `{X_train.shape}`")
        st.write(f"Bentuk X_test: `{X_test.shape}`")

        st.write("#### Membangun dan Melatih Model Deep Learning")
        model = create_ann_model(window_size)
        st.text("Ringkasan Model:")
        string_model_summary = StringIO()
        model.summary(print_fn=lambda x: string_model_summary.write(x + '\n'))
        st.text(string_model_summary.getvalue())

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )

        with st.spinner("Model sedang dilatih... Ini mungkin memakan waktu beberapa menit."):
            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.1,
                callbacks=[early_stopping],
                verbose=0
            )
        st.success("🎉 Model selesai dilatih!")

        st.write("#### Visualisasi Loss Pelatihan")
        fig_loss, ax_loss = plt.subplots(figsize=(12, 6))
        ax_loss.plot(history.history['loss'], label='Training Loss')
        ax_loss.plot(history.history['val_loss'], label='Validation Loss')
        ax_loss.set_title('Training and Validation Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss (MSE)')
        ax_loss.legend()
        ax_loss.grid(True)
        st.pyplot(fig_loss)

        st.write("#### Evaluasi Model pada Data Test")
        loss, mae, mse = model.evaluate(X_test, y_test, verbose=0)
        st.write(f"**Test Loss (MSE):** `{loss:.4f}`")
        st.write(f"**Test MAE:** `{mae:.4f}`")
        st.write(f"**Test RMSE:** `{np.sqrt(mse):.4f}`")

        y_pred_scaled = model.predict(X_test, verbose=0).flatten()
        y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        mae_original = mean_absolute_error(y_test_original, y_pred_original)
        mse_original = mean_squared_error(y_test_original, y_pred_original)
        rmse_original = np.sqrt(mse_original)
        r2_original = r2_score(y_test_original, y_pred_original)

        st.write("#### Metrik Evaluasi pada Skala Asli:")
        st.write(f"**MAE (Original Scale):** `{mae_original:.4f}`")
        st.write(f"**MSE (Original Scale):** `{mse_original:.4f}`")
        st.write(f"**RMSE (Original Scale):** `{rmse_original:.4f}`")
        st.write(f"**R-squared (Original Scale):** `{r2_original:.4f}`")

        st.write("#### Perbandingan Aktual vs Prediksi TH-OUT (Skala Asli)")
        fig_pred, ax_pred = plt.subplots(figsize=(14, 7))
        ax_pred.plot(y_test_original, label='Aktual TH-OUT', color='blue', alpha=0.7)
        ax_pred.plot(y_pred_original, label='Prediksi TH-OUT', color='red', linestyle='--', alpha=0.7)
        ax_pred.set_title('Perbandingan Aktual vs Prediksi TH-OUT (Skala Asli)')
        ax_pred.set_xlabel('Indeks Sampel')
        ax_pred.set_ylabel('Suhu TH-OUT (°C)')
        ax_pred.legend()
        ax_pred.grid(True)
        st.pyplot(fig_pred)

        st.write("#### Scatter Plot Aktual vs Prediksi TH-OUT")
        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 8))
        ax_scatter.scatter(y_test_original, y_pred_original, alpha=0.5, color='green')
        ax_scatter.plot([min(y_test_original), max(y_test_original)],
                        [min(y_test_original), max(y_test_original)], 'r--', lw=2, label='Garis Ideal Prediksi')
        ax_scatter.set_xlabel('Nilai Aktual TH-OUT (°C)')
        ax_scatter.set_ylabel('Nilai Prediksi TH-OUT (°C)')
        ax_scatter.set_title('Scatter Plot Aktual vs Prediksi TH-OUT')
        ax_scatter.legend()
        ax_scatter.grid(True)
        st.pyplot(fig_scatter)

        # Simpan bobot model
        try:
            model.save_weights(MODEL_WEIGHTS_PATH)
            st.success(f"✅ Bobot model berhasil disimpan di: `{MODEL_WEIGHTS_PATH}`")
        except Exception as e:
            st.error(f"❌ Gagal menyimpan bobot model: {e}")

        st.info("Anda sekarang dapat beralih ke halaman 'Prediksi TH-OUT' untuk menggunakan model yang baru dilatih.")

elif page_selection == "🚀 Prediksi TH-OUT":
    st.title("🚀 Prediksi Suhu TH-OUT")
    st.markdown("Halaman ini memungkinkan Anda untuk memprediksi suhu TH-OUT menggunakan model JST yang sudah dilatih.")

    if scaler_X is None or scaler_y is None or model is None:
        st.warning("Model atau Scaler belum dimuat. Harap pastikan file model ada atau latih model di halaman 'EDA & Pelatihan Model'.")
    else:
        st.header(f"Masukkan {window_size} Nilai TH-IN Terakhir")
        st.write(f"Harap masukkan {window_size} nilai `TH-IN` terbaru secara berurutan.")

        th_in_values = []
        for i in range(window_size):
            label_suffix = f" (t-{window_size - 1 - i})" if i < window_size - 1 else " (t-0, Terbaru)"
            value = st.number_input(
                f"TH-IN ke-{i+1}{label_suffix}:",
                min_value=0.0,
                max_value=150.0,
                value=50.0 + i*0.5,
                step=0.1,
                format="%.2f",
                key=f"predict_th_in_input_{i}"
            )
            th_in_values.append(value)

        st.info("Nilai TH-IN yang dimasukkan: " + ", ".join([f"{v:.2f}°C" for v in th_in_values]))

        if st.button("🚀 Prediksi TH-OUT"):
            if all(val is not None for val in th_in_values):
                try:
                    with st.spinner("Memproses prediksi..."):
                        input_sequence_raw = np.array(th_in_values)
                        input_sequence_for_scaler = input_sequence_raw.reshape(-1, 1)
                        input_scaled = scaler_X.transform(input_sequence_for_scaler)
                        input_reshaped = input_scaled.reshape(1, window_size, 1)

                        prediction_scaled = model.predict(input_reshaped, verbose=0)[0][0]
                        prediction_th_out_original = scaler_y.inverse_transform([[prediction_scaled]])[0][0]

                        st.success(f"**Prediksi Nilai TH-OUT:** **{prediction_th_out_original:.2f} °C**")
                        st.balloons()

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
                    st.warning("Pastikan input Anda valid dan model berfungsi dengan benar.")
            else:
                st.warning("Harap masukkan semua nilai TH-IN untuk melakukan prediksi.")

        st.markdown("---")

        # --- Real-Time Monitoring Section ---
        st.header("📊 Real-Time Temperature Monitoring & Prediction")
        st.write("Tekan tombol 'Mulai Monitoring Real-Time' untuk memulai simulasi pemantauan suhu.")

        MAX_DATA_POINTS = 50
        actual_temps_history = deque(maxlen=MAX_DATA_POINTS)
        predicted_temps_history = deque(maxlen=MAX_DATA_POINTS)
        time_points_history = deque(maxlen=MAX_DATA_POINTS)

        time_placeholder = st.empty()
        actual_temp_placeholder = st.empty()
        predicted_temp_placeholder = st.empty()
        deviation_placeholder = st.empty()
        chart_placeholder = st.empty()

        FIXED_Y_MIN = 40
        FIXED_Y_MAX = 70

        if st.button("▶️ Mulai Monitoring Real-Time", key="start_realtime_monitor"):
            st.info("Monitoring Real-Time sedang berjalan. Tekan 'Stop' di sidebar (Esc) atau tutup tab browser untuk menghentikan.")

            current_time_index = 0
            last_th_in_values = deque(th_in_values, maxlen=window_size)
            
            while True:
                # Simulasi Data Masuk
                if not last_th_in_values:
                    current_th_in_actual = np.random.uniform(50, 70)
                else:
                    current_th_in_actual = last_th_in_values[-1] + np.random.uniform(-0.5, 0.5)
                    current_th_in_actual = max(0.0, min(150.0, current_th_in_actual))

                current_th_out_actual = current_th_in_actual + np.random.uniform(1.0, 3.0)

                last_th_in_values.append(current_th_in_actual)

                predicted_th_out_original = 0.0
                deviation = 0.0

                if len(last_th_in_values) == window_size:
                    input_sequence_for_scaler = np.array(list(last_th_in_values)).reshape(-1, 1)
                    input_scaled = scaler_X.transform(input_sequence_for_scaler)
                    input_reshaped = input_scaled.reshape(1, window_size, 1)

                    prediction_scaled = model.predict(input_reshaped, verbose=0)[0][0]
                    predicted_th_out_original = scaler_y.inverse_transform([[prediction_scaled]])[0][0]

                    deviation = abs(current_th_out_actual - predicted_th_out_original)
                else:
                    predicted_th_out_original = current_th_out_actual
                    deviation = 0.0

                time_placeholder.markdown(f"**Time:** `{time.strftime('%H:%M:%S')}`")
                actual_temp_placeholder.markdown(f"**Actual Temperature T-ISO1:** `{current_th_out_actual:.2f} °C`")
                predicted_temp_placeholder.markdown(f"**Predicted Temperature T-ISO1:** `{predicted_th_out_original:.2f} °C`")
                deviation_placeholder.markdown(f"**Deviation (Actual - Predicted):** `{deviation:.2f} °C`")

                actual_temps_history.append(current_th_out_actual)
                predicted_temps_history.append(predicted_th_out_original)
                time_points_history.append(current_time_index)

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(list(time_points_history), list(actual_temps_history), label='Actual', color='blue', linewidth=2)
                ax.plot(list(time_points_history), list(predicted_temps_history), label='Predicted', color='red', linestyle='--', linewidth=2)
                ax.set_xlabel('Time Index (s)')
                ax.set_ylabel('Temperature (°C)')
                ax.set_title('Predicted vs Actual Temperature (Real-Time Monitoring)')
                ax.legend()
                ax.grid(True)
                ax.set_ylim(FIXED_Y_MIN, FIXED_Y_MAX) # Set fixed Y-axis limits
                chart_placeholder.pyplot(fig, clear_figure=True)
                plt.close(fig)

                current_time_index += 1
                time.sleep(1)