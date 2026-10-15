"""Download the original UCI CSV, validate it, and expose only allowed features."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

FEATURES = ["anchor_ratio", "trans_range", "node_density", "iterations"]
ALE_URL = ("https://archive.ics.uci.edu/static/public/844/"
           "average+localization+error+(ale)+in+sensor+node+localization+process+in+wsns.zip")
ALE_SHA256 = "5c8d3e608bcfef50d21c2f16e5313d23f749d8b3bced76b8238acca8c05c2be1"


def load_ale(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "ale.csv"
    if not path.exists():
        with urlopen(ALE_URL, timeout=60) as response:
            archive = response.read()
        with ZipFile(BytesIO(archive)) as z:
            content = z.read("mcs_ds_edited_iter_shuffled.csv")
        if sha256(content).hexdigest() != ALE_SHA256:
            raise ValueError("Downloaded ALE data do not match the recorded checksum.")
        path.write_bytes(content)
    if sha256(path.read_bytes()).hexdigest() != ALE_SHA256:
        raise ValueError("Local ALE data do not match the recorded checksum.")
    frame = pd.read_csv(path)
    if list(frame.columns) != FEATURES + ["ale", "sd_ale"]:
        raise ValueError("Unexpected ALE column schema.")
    if frame.shape != (107, 6) or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Expected 107 finite rows and six columns.")
    return frame[FEATURES].to_numpy(float), frame["ale"].to_numpy(float), frame
