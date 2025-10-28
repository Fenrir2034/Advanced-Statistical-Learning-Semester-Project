from pathlib import Path
from typing import Tuple
import pandas as pd
from urllib.request import urlretrieve
from sklearn.model_selection import train_test_split

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

def load_or_download(path: str | None = None) -> pd.DataFrame:
    """Load UCI SMS Spam dataset or a provided CSV (columns: label,text)."""
    if path is None:
        data_dir = Path("data"); data_dir.mkdir(exist_ok=True, parents=True)
        zip_path = data_dir / "smsspamcollection.zip"
        if not zip_path.exists():
            urlretrieve(UCI_URL, zip_path.as_posix())
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open("SMSSpamCollection") as f:
                df = pd.read_csv(f, sep="\t", header=None, names=["label", "text"], encoding="utf-8")
    else:
        df = pd.read_csv(path)
        assert {"label","text"} <= set(df.columns), "CSV must have columns: label,text"
    # normalize labels
    if df["label"].dtype == object:
        df["label"] = df["label"].map({"ham":0, "spam":1}).astype(int)
    else:
        df["label"] = df["label"].astype(int)
    return df

def stratified_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 123) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    X = df["text"]
    y = df["label"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

