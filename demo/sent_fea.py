import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def calculate_zscore(df, input_col, window):
        """Calculates the rolling z-score for a given column and returns it."""
        rolling_mean = df.groupby('Ticker')[input_col].transform(
            lambda x: x.rolling(window=window, min_periods=window).mean()
        )
        rolling_std = df.groupby('Ticker')[input_col].transform(
            lambda x: x.rolling(window=window, min_periods=window).std()
        )
        epsilon = 1e-6
        z_score = (df[input_col] - rolling_mean) / (rolling_std + epsilon)
        # Fill initial NaNs (from min_periods) with 0.0 (neutral z-score)
        return z_score.fillna(0.0)

def main(csvfile:str = "merged_OHLCV_Sentiment.csv", f:bool = True):
    # --- 0. Create Sentiment Curation Directory ---
    if f:
        output_dir = "Output"
        os.makedirs(output_dir, exist_ok=True)
        #print(f"\nSaving features to '{output_dir}' directory...")

    # --- 1. Load and Prepare Base Data ---
    try:
        df_base = pd.read_csv(csvfile)
    except FileNotFoundError:
        print("Error: 'merged_OHLCV_Sentiment.csv' not found.")
        # Handle error
        exit()

    # Convert Date to datetime for sorting and plotting
    df_base["Date"] = pd.to_datetime(df_base["Date"])

    # CRITICAL: Sort by Ticker and Date
    df_base = df_base.sort_values(by=["Ticker", "Date"]).reset_index(drop=True)

    nan_count_sentiment = df_base['Sentiment'].isna().sum()

    #print(f"NaN values in 'Sentiment' column: {nan_count_sentiment}")

    # CRITICAL: Fill missing sentiment with 0.0 (Neutral) BEFORE calculations
    df_base['Sentiment'] = df_base['Sentiment'].fillna(0.0)

    #print("Base DataFrame loaded and prepared.")
    #print(f"Original shape: {df_base.shape}")
    #print(f"NaN values in 'Sentiment' column: {df_base['Sentiment'].isna().sum()}")

    # --- 2. Create Smoothed DataFrames ---

    # Create a copy for the 3-day window
    df_3d = df_base.copy()
    df_3d['sent_smooth'] = df_3d.groupby('Ticker')['Sentiment'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )

    # Create a copy for the 5-day window
    df_5d = df_base.copy()
    df_5d['sent_smooth'] = df_5d.groupby('Ticker')['Sentiment'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )

    # Create a copy for the 7-day window
    df_7d = df_base.copy()
    df_7d['sent_smooth'] = df_7d.groupby('Ticker')['Sentiment'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )

    '''print("\nSuccessfully created 3 separate smoothed DataFrames:")
    print(f"df_3d shape: {df_3d.shape}, head:\n{df_3d[['Ticker', 'Date', 'sent_smooth']].head(10)}")
    print(f"df_5d shape: {df_5d.shape}, head:\n{df_5d[['Ticker', 'Date', 'sent_smooth']].head(10)}")
    print(f"df_7d shape: {df_7d.shape}, head:\n{df_7d[['Ticker', 'Date', 'sent_smooth']].head(10)}")'''

    # --- 3. Plot Smoothing Comparison ---

    # Create a new temp table by combining the relevant columns
    df_plot_smooth = df_base[['Date', 'Ticker', 'Sentiment']].copy()
    df_plot_smooth['sent_smooth_3d'] = df_3d['sent_smooth']
    df_plot_smooth['sent_smooth_5d'] = df_5d['sent_smooth']
    df_plot_smooth['sent_smooth_7d'] = df_7d['sent_smooth']

    #print(f"\nCreated df_plot_smooth with shape {df_plot_smooth.shape}")

    # --- Plot the smoothing comparison ---

    # Filter for a readable slice (SPY, 6 months)
    df_spy_slice = df_plot_smooth[
        (df_plot_smooth['Ticker'] == 'SPY') & 
        (df_plot_smooth['Date'] >= '2025-02-01') & 
        (df_plot_smooth['Date'] <= '2025-08-30')
    ].copy()
    if f:
        plt.figure(figsize=(15, 8))
        plt.plot(df_spy_slice['Date'], df_spy_slice['Sentiment'], label='Original Raw Sentiment', color='gray', alpha=0.5, linestyle='--')
        plt.plot(df_spy_slice['Date'], df_spy_slice['sent_smooth_3d'], label='3-Day Smooth', linewidth=2)
        plt.plot(df_spy_slice['Date'], df_spy_slice['sent_smooth_5d'], label='5-Day Smooth', linewidth=2)
        plt.plot(df_spy_slice['Date'], df_spy_slice['sent_smooth_7d'], label='7-Day Smooth', linewidth=2)

        plt.title('Sentiment Smoothing Comparison (SPY: Feb-Aug 2025)', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Sentiment Score', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        os.makedirs("Plot", exist_ok=True)
        plt.savefig("Plot/sentiment_smoothing_comparison.png")

    '''
    print("Smoothing comparison plot saved as 'sentiment_smoothing_comparison.png'")
    print("- Original Raw Sentiment (Gray, Dashed): This line will be extremely spiky, jumping up and down every day. \n- 3-Day Smooth (Blue): This line will follow the raw sentiment fairly closely but will cut off the most extreme single-day spikes.\n- 5-Day Smooth (Orange): This line will be noticeably smoother, ignoring more of the 2-3 day chatter and showing the weekly trend. \n- 7-Day Smooth (Green): This will be the smoothest line, 
    representing the most stable, underlying sentiment trend. It will also lag the most behind the raw data.")
    '''

    # --- 4. Compute 20-Day Z-Score for Each Respective Table ---

    
    df_3d['sent_z_20d'] = calculate_zscore(df_3d, 'sent_smooth', 20)
    df_5d['sent_z_20d'] = calculate_zscore(df_5d, 'sent_smooth', 20)
    df_7d['sent_z_20d'] = calculate_zscore(df_7d, 'sent_smooth', 20)
    '''
    print("\nSuccessfully calculated and added 20-day z-score to each of the 3 tables.")
    print("df_3d 'sent_z_20d' head:")
    print(df_3d[['Date', 'sent_smooth', 'sent_z_20d']].head())
    print("\ndf_7d 'sent_z_20d' tail:")
    print(df_7d[['Date', 'sent_smooth', 'sent_z_20d']].tail())'''

    # --- 5. Create Temp Table and Plot Z-Score Comparison ---
    # Create a new temp table by combining the new z-score columns
    df_plot_zscore = df_base[['Date', 'Ticker']].copy()
    df_plot_zscore['z_from_3d_smooth'] = df_3d['sent_z_20d']
    df_plot_zscore['z_from_5d_smooth'] = df_5d['sent_z_20d']
    df_plot_zscore['z_from_7d_smooth'] = df_7d['sent_z_20d']

    #print(f"\nCreated df_plot_zscore with shape {df_plot_zscore.shape}")

    # --- Plot the z-score comparison ---

    # Filter for SPY and a recent 1-year slice for a readable plot
    df_spy_z_slice = df_plot_zscore[
        (df_plot_zscore['Ticker'] == 'SPY') & 
        (df_plot_zscore['Date'] >= '2025-02-01') & 
        (df_plot_zscore['Date'] <= '2025-08-30')
    ].copy()

    if f:

        plt.figure(figsize=(15, 8))

        # Plot the z-score lines
        plt.plot(df_spy_z_slice['Date'], df_spy_z_slice['z_from_3d_smooth'], label='Z-Score (from 3d-Smooth)', linewidth=2)
        plt.plot(df_spy_z_slice['Date'], df_spy_z_slice['z_from_5d_smooth'], label='Z-Score (from 5d-Smooth)', linewidth=2)
        plt.plot(df_spy_z_slice['Date'], df_spy_z_slice['z_from_7d_smooth'], label='Z-Score (from 7d-Smooth)', linewidth=2, color='black', alpha=0.8)

        # Add horizontal lines for "extreme" sentiment
        plt.axhline(y=1.5, color='r', linestyle='--', label='Extreme Positive (Z > 1.5)')
        plt.axhline(y=-1.5, color='b', linestyle='--', label='Extreme Negative (Z < -1.5)')
        plt.axhline(y=0.0, color='gray', linestyle=':', label='Neutral (Z = 0.0)')

        plt.title('Z-Score Comparison (SPY: Feb-Aug 2025)', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Sentiment Z-Score (Standard Deviations)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        plt.savefig("Plot/sentiment_zscore_comparison.png")
        #print("Z-Score comparison plot saved as 'sentiment_zscore_comparison.png'")

    # --- 6. Add Flags (Threshold 1.5) ---
    threshold = 1.5
    #print(f"Using new z-score threshold of: +/- {threshold}")

    # --- Add flags to the 3-day smooth table ---
    df_3d['sent_high'] = (df_3d['sent_z_20d'] > threshold).astype(int)
    df_3d['sent_low'] = (df_3d['sent_z_20d'] < -threshold).astype(int)

    # --- Add flags to the 5-day smooth table ---
    df_5d['sent_high'] = (df_5d['sent_z_20d'] > threshold).astype(int)
    df_5d['sent_low'] = (df_5d['sent_z_20d'] < -threshold).astype(int)

    # --- Add flags to the 7-day smooth table ---
    df_7d['sent_high'] = (df_7d['sent_z_20d'] > threshold).astype(int)
    df_7d['sent_low'] = (df_7d['sent_z_20d'] < -threshold).astype(int)

    # --- Verification ---
    '''
    print("\n--- Verification (df_3d, SPY, 2025) ---")
    print(df_3d[
        (df_3d['Ticker'] == 'SPY') &
        (df_3d['Date'] >= '2025-01-01') & 
        (df_3d['sent_low'] == 1)
    ][['Date', 'sent_z_20d', 'sent_high', 'sent_low']].head(10))

    print(df_3d[
        (df_3d['Ticker'] == 'SPY') &
        (df_3d['Date'] >= '2025-01-01') & 
        (df_3d['sent_high'] == 1)
    ][['Date', 'sent_z_20d', 'sent_high', 'sent_low']].head(10))

    print("\n--- Verification (df_5d, SPY, 2025) ---")
    print(df_5d[
        (df_5d['Ticker'] == 'SPY') &
        (df_5d['Date'] >= '2025-01-01') & 
        (df_5d['sent_low'] == 1)
    ][['Date', 'sent_z_20d', 'sent_high', 'sent_low']].head(10))

    print(df_5d[
        (df_5d['Ticker'] == 'SPY') &
        (df_5d['Date'] >= '2025-01-01') & 
        (df_5d['sent_high'] == 1)
    ][['Date', 'sent_z_20d', 'sent_high', 'sent_low']].head(10))

    print("\n--- Verification (df_7d, SPY, 2025) ---")
    print(df_7d[
        (df_7d['Ticker'] == 'SPY') &
        (df_7d['Date'] >= '2025-01-01') & 
        (df_7d['sent_low'] == 1)
    ][['Date', 'sent_z_20d', 'sent_high', 'sent_low']].head(10))

    print(df_7d[
        (df_7d['Ticker'] == 'SPY') &
        (df_7d['Date'] >= '2025-01-01') & 
        (df_7d['sent_high'] == 1)
    ][['Date', 'sent_z_20d', 'sent_high', 'sent_low']].head(10))'''

    # --- 1. Add 'Close_next' Column to Each DataFrame ---
    # This is the target variable for our plot

    #print("Adding 'Close_next' column to all 3 DataFrames...")
    try:
        df_3d['Close_next'] = df_3d.groupby('Ticker')['Close'].shift(-1)
        df_5d['Close_next'] = df_5d.groupby('Ticker')['Close'].shift(-1)
        df_7d['Close_next'] = df_7d.groupby('Ticker')['Close'].shift(-1)
    except NameError:
        print("Error: df_3d, df_5d, or df_7d not found.")
        print("Please re-run the previous feature generation steps first.")
        # As a fallback, try to load from the CSVs (but this is less ideal
        # as they don't have the 'Close' price. We will stop.)
        exit()
    except Exception as e:
        print(f"An error occurred: {e}")
        exit()

    if f :
        # --- 2. Setup the Comparison Plot ---
        # We'll create one figure with 3 subplots (1 row, 3 columns)
        # We use sharey=True so the y-axis (price) is consistent across all plots
        fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
        threshold = 1.5
        ticker_to_plot = 'SPY' # Plotting one ticker is cleaner

        #print(f"Generating plots for {ticker_to_plot}...")

        # --- 3. Plot 1: Z-Score (from 3d-Smooth) vs. Next Day Close ---
        # Filter for SPY and drop the last row (which has NaN for Close_next)
        spy_3d = df_3d[df_3d['Ticker'] == ticker_to_plot].dropna()
        axes[0].scatter(spy_3d['sent_z_20d'], spy_3d['Close_next'], alpha=0.1)
        axes[0].set_title('Z-Score (from 3d-Smooth) vs. Next Day Close', fontsize=14)
        axes[0].set_xlabel('Sentiment Z-Score', fontsize=12)
        axes[0].set_ylabel('Next Day Close Price ($)', fontsize=12)
        axes[0].axvline(x=threshold, color='r', linestyle='--', label=f'High Sent (Z > {threshold})')
        axes[0].axvline(x=-threshold, color='b', linestyle='--', label=f'Low Sent (Z < {-threshold})')
        axes[0].grid(True, linestyle='--', alpha=0.6)
        axes[0].legend()

        # --- 4. Plot 2: Z-Score (from 5d-Smooth) vs. Next Day Close ---
        spy_5d = df_5d[df_5d['Ticker'] == ticker_to_plot].dropna()
        axes[1].scatter(spy_5d['sent_z_20d'], spy_5d['Close_next'], alpha=0.1, color='orange')
        axes[1].set_title('Z-Score (from 5d-Smooth) vs. Next Day Close', fontsize=14)
        axes[1].set_xlabel('Sentiment Z-Score', fontsize=12)
        axes[1].axvline(x=threshold, color='r', linestyle='--')
        axes[1].axvline(x=-threshold, color='b', linestyle='--')
        axes[1].grid(True, linestyle='--', alpha=0.6)

        # --- 5. Plot 3: Z-Score (from 7d-Smooth) vs. Next Day Close ---
        spy_7d = df_7d[df_7d['Ticker'] == ticker_to_plot].dropna()
        axes[2].scatter(spy_7d['sent_z_20d'], spy_7d['Close_next'], alpha=0.1, color='green')
        axes[2].set_title('Z-Score (from 7d-Smooth) vs. Next Day Close', fontsize=14)
        axes[2].set_xlabel('Sentiment Z-Score', fontsize=12)
        axes[2].axvline(x=threshold, color='r', linestyle='--')
        axes[2].axvline(x=-threshold, color='b', linestyle='--')
        axes[2].grid(True, linestyle='--', alpha=0.6)

        # --- 6. Finalize and Save ---
        fig.suptitle(f'Sentiment Z-Score vs. Next Day Close ({ticker_to_plot})', fontsize=20, y=1.05)
        plt.tight_layout()
        plt.savefig("Plot/sentiment_vs_price_comparison.png")

        #print("Plot saved as 'sentiment_vs_price_comparison.png'")

    # --- 7. Save Final Feature Files ---

    # Define which columns to save
    columns_to_save = ['Ticker', 'Date', 'sent_smooth', 'sent_z_20d', 'sent_high', 'sent_low']
    if f:
        try:
            output_path_3d = os.path.join(output_dir, "sent_features_3d.csv")
            df_3d[columns_to_save].to_csv(output_path_3d, index=False)
            #print(f"Successfully saved 3-day features to: {output_path_3d}")

            output_path_5d = os.path.join(output_dir, "sent_features_5d.csv")
            df_5d[columns_to_save].to_csv(output_path_5d, index=False)
            #print(f"Successfully saved 5-day features to: {output_path_5d}")

            output_path_7d = os.path.join(output_dir, "sent_features_7d.csv")
            df_7d[columns_to_save].to_csv(output_path_7d, index=False)
            #print(f"Successfully saved 7-day features to: {output_path_7d}")

        except Exception as e:
            print(f"An error occurred during file saving: {e}")

        #print("\nSentiment feature generation complete.")
    else:
        os.makedirs("Predict_file", exist_ok=True)
        df_3d[columns_to_save].to_csv("Predict_file/sent_features.csv", index=False)

if __name__ == "__main__":
    main()