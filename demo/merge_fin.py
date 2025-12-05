import price_fea
import sent_fea
import pandas as pd
import numpy as np
import os

def ensure_label_up_next(csv_path: str, f:bool):
    if not f:
        df = pd.read_csv(csv_path)

        if "label_up_next" not in df.columns:
            # append the column labels to be 1
            df["label_up_next"] = 1.0     # 

            
            cols = [c for c in df.columns if c != "label_up_next"] + ["label_up_next"]
            df = df[cols]

            df.to_csv(csv_path, index=False)



def merge_final(csvf:str = "merged_OHLCV_Sentiment.csv", f:bool= True):
    merged = pd.read_csv(csvf)
    if f:
        price  = pd.read_csv("Output/price_features.csv")
        sent   = pd.read_csv("Output/sent_features_3d.csv")
    else:
        price  = pd.read_csv("Predict_file/price_features.csv")
        sent   = pd.read_csv("Predict_file/sent_features.csv")

        
    # columns that are NOT new features in price_features
    price_drop = ["Open","High","Low","Close","Volume","Sentiment","label_up_next"]

    price_new = price.copy()
    for col in price_drop:
        if col in price_new.columns:
            price_new = price_new.drop(columns=col)

    # price_new should now be: Ticker, Date, pct_change, SMA5, SMA10, SMA20, EMA12, EMA26, ...
    #print(price_new.columns)

    # sentiment: we keep Ticker, Date, and sentiment feature columns
    sent_new = sent.copy()
    # if there's any duplicate label or original Sentiment column in here, drop it too
    for col in ["Sentiment","label_up_next"]:
        if col in sent_new.columns:
            sent_new = sent_new.drop(columns=col)

    #print(sent_new.columns)

    df = merged.merge(price_new, on=["Ticker","Date"], how="inner") \
           .merge(sent_new,  on=["Ticker","Date"], how="inner")

    df = df.sort_values(["Date","Ticker"]).dropna().reset_index(drop=True)
    cols = [c for c in df.columns if c != 'label_up_next']  
    cols.append('label_up_next')                            
    df = df[cols]

    if f:
        os.makedirs("Output", exist_ok=True)
        df.to_csv("Output/training_dateset.csv", index=False)
    else:
        df.to_csv("Predict_file/predict_data.csv", index=False)
    

def main(csvf:str = "merged_OHLCV_Sentiment.csv", f:bool= True):
    ensure_label_up_next(csvf, f)
    sent_fea.main(csvf,f)
    price_fea.main(csvf,f)
    merge_final(csvf,f)


if __name__ == "__main__":
    main()


