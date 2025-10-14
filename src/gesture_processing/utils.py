import os
import pandas as pd

def load_annotations(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")  # autodetect delimiter
    print(f"[INFO] Загружено {len(df)} аннотаций из {path}")
    if "attachment_id" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV должен содержать колонки 'attachment_id' и 'text'")
    return df

def ensure_dirs(dirs: list):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
