# Prediksi Harga Ethereum dengan Gradient Boosting & Analisis Sentimen

Proyek ini mengimplementasikan pipeline machine learning yang dioptimalkan untuk memprediksi **tren arah** harga Ethereum (ETH). Dengan mengintegrasikan **Analisis Sentimen** (menggunakan FinRoBERTa) dengan indikator teknikal dan **Gradient Boosting Classifier**, model ini mencapai performa yang kuat dalam memprediksi pergerakan pasar 4 jam ke depan.

## 🚀 Insight Riset Utama yang Diterapkan
Proyek ini mengatasi masalah umum dalam prediksi kripto dengan menerapkan insight dari literatur ML keuangan terbaru:

1.  **Ketergantungan Waktu (Lags):** Reaksi pasar terhadap berita seringkali tertunda. Kami merekayasa fitur `sentiment_lag_1` hingga `sentiment_lag_6` untuk menangkap hal ini.
2.  **Pengurangan Noise (Rolling):** Sentimen sesaat seringkali *noisy* (banyak gangguan). Kami menggunakan `sent_roll_6` dan `sent_roll_24` (rata-rata bergerak) untuk mengekstraksi tren opini yang sebenarnya.
3.  **Target Arah (Directional):** Memprediksi harga eksak ($2000.50) tidak stabil. Kami mengoptimalkan untuk **Tren Arah (Naik/Turun)** dalam cakrawala 4 jam, yang menghasilkan keandalan yang jauh lebih tinggi.

## 🛠️ Struktur Proyek
Alur kerja dipisahkan secara ketat menjadi **Data Engineering** dan **Modeling** untuk memastikan modularitas.

### 1. Persiapan Data (`04_4H_data.ipynb`)
-   **Input:** Data mentah `training_data.csv` (Harga, Volume, Skor Sentimen).
-   **Proses:**
    -   Resampling data ke candle 4 Jam.
    -   Menghitung **RSI**, **Momentum (Returns)**, dan **Volatilitas**.
    -   Menghasilkan fitur Sentimen Lagged dan Rolling.
    -   Membuat variabel `target_dir`.
-   **Output:** `training_data_4h_engineered.csv`

### 2. Pemodelan (`06_Eth_Price_Prediction_Optimized.ipynb`)
-   **Input:** `training_data_4h_engineered.csv`
-   **Model:** **Gradient Boosting Classifier** (Hyperparameter Dioptimalkan).
-   **Evaluasi:**
    -   Skor Akurasi.
    -   Confusion Matrix.
    -   Plot Visual (Harga vs. Prediksi Benar/Salah).

## 📦 Instalasi
1.  Clone repository ini:
    ```bash
    git clone https://github.com/destadrns/ETH-Price-Prediction-Gradient-Boosting.git
    cd ETH-Price-Prediction-Gradient-Boosting
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 📈 Hasil
Model yang dioptimalkan menangkap hubungan non-linear antara sentimen sosial dan momentum pasar, mengungguli model baseline standar dalam akurasi arah.

---
*Dibuat untuk Tugas Akhir Kecerdasan Buatan Lanjut.*
