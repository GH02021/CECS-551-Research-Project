import argparse, os
from fusion_model import FusionModel
import merge_fin


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FusionModel prediction for the given date and ticker."
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Date to predict, e.g. 2025-08-13",
    )
    parser.add_argument(
        "--ticker",
        required=True,
        help="Ticker symbol, e.g. SPY",
    )
    # new csv -optional
    parser.add_argument(
        "--csvf",
        default = None,
        help="Optional raw CSV file. If provided, merge_fin will run on this file.",
    )

    parser.add_argument(
        "--fmp",
        default = "Output/fusion_model.pth",
        help="Path to the trained fusion model checkpoint.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    
    fm = FusionModel()

    if args.csvf is None:
        label, prob = fm.predict(args.date, args.ticker)
    else:
        merge_fin.main(args.csvf,False)

        label, prob = fm.predict(date = args.date, ticker=args.ticker, csvfile="Predict_file/predict_data.csv", fmp="Output/fusion_model.pth")

    if label is not None:
        print(
            f"Prediction for {args.ticker} on {args.date}: "
            f"{label}, prob={prob:.6f}"
        )
    else:
        print(
            f"We need more historical data around {args.date} "
            f"for {args.ticker} before we can make a reliable prediction."
        )



if __name__ == "__main__":
    main()
