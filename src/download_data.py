import argparse, os
import pandas as pd

def fetch_openml_sms_spam() -> pd.DataFrame:
    from sklearn.datasets import fetch_openml
    df = fetch_openml('sms_spam', version=1, as_frame=True).frame
    # Harmonize expected column names if your pipeline assumes "label", "text":
    if 'class' in df.columns:
        df = df.rename(columns={'class': 'label'})
    if 'text' not in df.columns and 'sms' in df.columns:
        df = df.rename(columns={'sms': 'text'})
    # Common OpenML version already has 'text' and 'class', but we normalize anyway.
    return df[['label','text']]

def main(out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df = fetch_openml_sms_spam()
    df.to_csv(out_csv, index=False)
    print(f"Saved dataset to {out_csv} with shape {df.shape}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sms_spam.csv", help="Where to save csv")
    args = ap.parse_args()
    main(args.out)

