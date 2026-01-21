# Ethereum Price Prediction with Gradient Boosting & Sentiment Analysis

This project implements an optimized machine learning pipeline to predict the **directional trend** of Ethereum (ETH) prices. By integrating **Sentiment Analysis** (using FinRoBERTa) with technical indicators and a **Gradient Boosting Classifier**, the model achieves robust performance in forecasting 4-hour market movements.

## 🚀 Key Research Insights Applied
This project addresses common pitfalls in crypto prediction by implementing insights from recent financial ML literature:

1.  **Temporal Dependency (Lags):** Market reaction to news is often delayed. We engineer `sentiment_lag_1` to `sentiment_lag_6` features to capture this.
2.  **Noise Reduction (Rolling):** Instantaneous sentiment is noisy. We use `sent_roll_6` and `sent_roll_24` (moving averages) to extract the true opinion trend.
3.  **Directional Target:** Predicting exact prices ($2000.50) is unstable. We optimize for **Directional Trend (Up/Down)** over a 4-hour horizon, which yields significantly higher reliability.

## 🛠️ Project Structure
The workflow is strictly separated into **Data Engineering** and **Modeling** to ensure modularity.

### 1. Data Preparation (`04_4H_data.ipynb`)
-   **Input:** Raw `training_data.csv` (Price, Volume, Sentiment Score).
-   **Process:**
    -   Resamples data to 4-Hour candles.
    -   Calculates **RSI**, **Momentum (Returns)**, and **Volatility**.
    -   Generates Lagged and Rolling Sentiment features.
    -   Creates the `target_dir` variable.
-   **Output:** `training_data_4h_engineered.csv`

### 2. Modeling (`06_Eth_Price_Prediction_Optimized.ipynb`)
-   **Input:** `training_data_4h_engineered.csv`
-   **Model:** **Gradient Boosting Classifier** (Optimized Hyperparameters).
-   **Evaluation:**
    -   Accuracy Score.
    -   Confusion Matrix.
    -   Visual Plot (Price vs. Correct/Incorrect Predictions).

## 📦 Installation
1.  Clone this repository:
    ```bash
    git clone https://github.com/destadrns/ETH-Price-Prediction-Gradient-Boosting.git
    cd ETH-Price-Prediction-Gradient-Boosting
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 📈 Results
The optimized model captures non-linear relationships between social sentiment and market momentum, outperforming standard baseline models in directional accuracy.

---
*Created for the Advanced Artificial Intelligence Final Project.*
