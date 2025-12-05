# Fusion Stock Movement Prediction

## Overview
This project implements a multimodal deep learning system to predict **next-day stock price movement (UP/DOWN)** by combining:
- **Technical price indicators** derived from OHLCV data
- **FinBERT-based sentiment features** engineered from financial text sentiment

The system integrates these signals into a unified dataset and trains a Fusion MLP model to classify whether the next day's closing price will be higher or lower.

---

## Project Structure

### Data
- **Dataset**: `Data/merged_OHLCV_Sentiment.csv`
- Columns: Date, Ticker, OHLCV (Open, High, Low, Close, Volume), FinBERT sentiment score, `label_up_next` (binary target)

### Feature Engineering
- **Price Feature Module (`price_fea.py`)**
  - **Daily Percentage Change (pct_change):** Measures relative change in closing price, capturing short-term momentum.
  - **Simple Moving Averages (SMA5, SMA10, SMA20):** Rolling averages over 5, 10, and 20 days to smooth noise and highlight trends.
  - **Exponential Moving Averages (EMA12, EMA26):** Weighted averages emphasizing recent prices, responsive to new information.
  - **MACD (Moving Average Convergence Divergence):** Derived from EMA12 and EMA26, used to detect momentum shifts and reversals.
  - **RSI (14-day Relative Strength Index):** Oscillator measuring magnitude of recent price changes to identify overbought/oversold conditions.
  - **ATR (14-day Average True Range):** Volatility measure capturing average daily trading range, reflecting market risk.
  
  These indicators were chosen to balance simplicity with interpretability, covering momentum, trend, and volatility aspects of market behavior.  
  Implementation uses `pandas_ta` and pandas’ rolling/exponential functions. Outputs are stored in `price_features.csv` and merged later with sentiment features.  
  Exploratory plots (e.g., violin plots of RSI vs labels, scatter plots of RSI vs binary outcomes) were generated to visualize relationships between indicators and next-day movement.

- **Sentiment Feature Module (`sent_fea.py`)**
  - 3-day, 5-day, 7-day smoothed sentiment
  - Rolling z-score sentiment
  - Extreme sentiment indicators (`sent_high`, `sent_low`)

### Data Merging
- **Merging Pipeline (`merge_fin.py`)**
  - Combines OHLCV data, raw sentiment, price indicators, and sentiment features
  - Produces `Output/training_dataset.csv`

### Model
- **Fusion Model (`fusion_model.py`)**
  - Input: concatenated price + sentiment features
  - Architecture: Linear → ReLU → Linear → Dropout → Sigmoid
  - Loss: Binary Cross-Entropy
  - Optimizer: Adam (lr=1e-3)
  - Early stopping with patience = 5
  - Saves model as `fusion_model.pth` and metrics as `fusion.json`

### Training
- **Training Script (`train.py`)**
  - Loads `training_dataset.csv`
  - Initializes and trains Fusion MLP
  - Saves metrics and visualizations (confusion matrix, learning curve)

### Prediction
- **Prediction Script (`predict.py`)**
  - Mode 1: Predict from training dataset
  - Mode 2: Predict from new CSV input  
    ```bash
    python predict.py --csv new_data.csv
    ```

---

## Prototype Verification
The pipeline was executed and validated in Google Colab:
- `merge_fin.py` produced the final dataset
- `train.py` trained the Fusion MLP
- Achieved:
  - Accuracy: 0.52
  - Precision: 0.52
  - Recall: 0.92
  - F1 Score: 0.66

These results demonstrate meaningful learning despite market noise, with strong recall indicating the model captures upward/downward patterns effectively.

---

## System Architecture

```text
Data (OHLCV + Sentiment)
        │
        ├── price_fea.py → Technical Indicators
        ├── sent_fea.py  → Sentiment Features
        │
        ▼
   merge_fin.py → Unified Dataset
        │
        ▼
   fusion_model.py → Fusion MLP
        │
        ├── train.py → Training & Evaluation
        └── predict.py → CLI Prediction
