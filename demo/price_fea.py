import pandas as pd
import os
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns



def main(csvfile:str = "merged_OHLCV_Sentiment.csv", f:bool = True):
    # Check current working directory
    #print("Current working directory:", os.getcwd())

    # Check if the file exists before loading
    if not os.path.exists(csvfile):
        raise FileNotFoundError(csvfile + " not found in the current directory.")

    df = pd.read_csv(csvfile)
    
    # Step 2: Calculate technical indicators
    df['pct_change'] = df['Close'].pct_change()
    df['SMA5'] = ta.sma(df['Close'], length=5)
    df['SMA10'] = ta.sma(df['Close'], length=10)
    df['SMA20'] = ta.sma(df['Close'], length=20)
    df['EMA12'] = ta.ema(df['Close'], length=12)
    df['EMA26'] = ta.ema(df['Close'], length=26)

    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']

    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # Step 3: Drop rows with NaNs from indicator calculations
    df.dropna(inplace=True)

    # Step 4: Normalize the features
    features = ['pct_change', 'SMA5', 'SMA10', 'SMA20', 'EMA12', 'EMA26', 'MACD', 'RSI_14', 'ATR_14']
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Step 5: Save to CSV
    # Ensure output directory exists
    if f:
        os.makedirs("Output", exist_ok=True)
        df.to_csv("Output/price_features.csv", index=False)
    else:
        os.makedirs("Predict_file", exist_ok=True)
        
        df.to_csv("Predict_file/price_features.csv", index=False)

    # Step 5.5: Violin plot of raw RSI vs label
    
    if f:
        # Reload raw data to avoid scaled RSI
        df_raw = pd.read_csv("merged_OHLCV_Sentiment.csv")

        # Calculate raw RSI
        df_raw['RSI_14'] = ta.rsi(df_raw['Close'], length=14)

        # Drop NaNs from RSI or label
        df_raw.dropna(subset=['RSI_14', 'label_up_next'], inplace=True)

        # Create violin plot
        plt.figure(figsize=(6, 4))
        sns.violinplot(x='label_up_next', y='RSI_14', data=df_raw, inner='quartile')
        plt.xlabel("Next Day Up (1 = Yes, 0 = No)")
        plt.ylabel("RSI_14 (Raw)")
        plt.title("Distribution of RSI by Next-Day Movement")
        plt.grid(True)
        plt.tight_layout()

        # Save plot
        os.makedirs("Plot", exist_ok=True)
        plt.savefig("Plot/violin_rsi_vs_label.png")
        #plt.show()

        # Step 6: Plot correlation (RSI vs label)

        # Check if label column exists before plotting
        if 'label_up_next' in df.columns:
            plt.figure(figsize=(6, 4))
            plt.scatter(df['RSI_14'], df['label_up_next'], alpha=0.5)
            plt.xlabel("RSI_14")
            plt.ylabel("Next Day Up (1 = Yes, 0 = No)")
            plt.title("RSI vs Next-Day Movement")
            plt.grid(True)
            plt.tight_layout()

            # Ensure output directory exists
            os.makedirs("Plot", exist_ok=True)
            plt.savefig("Plot/rsi_vs_label.png")
            #plt.show()
        else:
            print("Warning: 'label_up_next' column not found — skipping correlation plot.")


if __name__ == "__main__":
    main()
