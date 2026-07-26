"""
download_data.py
Downloads the NSL-KDD dataset (KDDTrain+.txt / KDDTest+.txt).

NSL-KDD is a public, widely-used benchmark dataset for intrusion detection
research (no license restrictions, safe for a public GitHub repo).

If the download fails (no internet, mirror down, corporate proxy, etc.),
this script falls back to generating a small SYNTHETIC dataset with the same
schema and statistical shape, so the rest of the pipeline (preprocess ->
train -> explain -> dashboard) can still be run and demoed end-to-end.

Usage:
    python data/download_data.py
"""

import os
import sys
import urllib.request

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import DATA_DIR, COLUMN_NAMES, ensure_dirs  # noqa: E402

# Known public mirrors of NSL-KDD (raw text, comma-separated, no header)
TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
TEST_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

TRAIN_PATH = os.path.join(DATA_DIR, "KDDTrain+.txt")
TEST_PATH = os.path.join(DATA_DIR, "KDDTest+.txt")


def _download(url: str, dest: str) -> bool:
    try:
        print(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
        return os.path.getsize(dest) > 0
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def _generate_synthetic(path: str, n_rows: int, seed: int):
    """Generate a synthetic NSL-KDD-shaped dataset as an offline fallback."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    protocols = ["tcp", "udp", "icmp"]
    services = ["http", "ftp", "smtp", "telnet", "domain_u", "private", "ssh"]
    flags = ["SF", "S0", "REJ", "RSTO", "RSTR"]
    labels_normal = ["normal"]
    labels_attack = ["neptune", "smurf", "satan", "ipsweep", "guess_passwd",
                      "buffer_overflow", "portsweep", "nmap", "back", "warezclient"]

    n_attack = n_rows // 3
    n_normal = n_rows - n_attack

    rows = []
    for _ in range(n_normal):
        rows.append(_synthetic_row(rng, protocols, services, flags, "normal", benign=True))
    for _ in range(n_attack):
        lbl = rng.choice(labels_attack)
        rows.append(_synthetic_row(rng, protocols, services, flags, lbl, benign=False))

    rng.shuffle(rows)
    df = pd.DataFrame(rows, columns=COLUMN_NAMES)
    df.to_csv(path, header=False, index=False)


def _synthetic_row(rng, protocols, services, flags, label, benign):
    import numpy as np
    duration = rng.exponential(2) if benign else rng.exponential(20)
    src_bytes = int(rng.lognormal(4, 1)) if benign else int(rng.lognormal(2, 3))
    dst_bytes = int(rng.lognormal(4, 1)) if benign else int(rng.lognormal(1, 2))
    count = rng.integers(1, 20) if benign else rng.integers(50, 511)
    serror_rate = round(rng.uniform(0, 0.1), 2) if benign else round(rng.uniform(0.5, 1.0), 2)

    row = {
        "duration": duration, "protocol_type": rng.choice(protocols),
        "service": rng.choice(services), "flag": rng.choice(flags),
        "src_bytes": src_bytes, "dst_bytes": dst_bytes, "land": 0,
        "wrong_fragment": 0, "urgent": 0, "hot": rng.integers(0, 3),
        "num_failed_logins": 0 if benign else rng.integers(0, 5),
        "logged_in": rng.integers(0, 2), "num_compromised": 0,
        "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0,
        "is_guest_login": rng.integers(0, 2) if not benign else 0,
        "count": count, "srv_count": rng.integers(1, count + 1),
        "serror_rate": serror_rate, "srv_serror_rate": serror_rate,
        "rerror_rate": round(rng.uniform(0, 0.2), 2),
        "srv_rerror_rate": round(rng.uniform(0, 0.2), 2),
        "same_srv_rate": round(rng.uniform(0.5, 1.0), 2) if benign else round(rng.uniform(0, 0.5), 2),
        "diff_srv_rate": round(rng.uniform(0, 0.1), 2),
        "srv_diff_host_rate": round(rng.uniform(0, 0.1), 2),
        "dst_host_count": rng.integers(1, 255),
        "dst_host_srv_count": rng.integers(1, 255),
        "dst_host_same_srv_rate": round(rng.uniform(0.5, 1.0), 2),
        "dst_host_diff_srv_rate": round(rng.uniform(0, 0.1), 2),
        "dst_host_same_src_port_rate": round(rng.uniform(0, 1.0), 2),
        "dst_host_srv_diff_host_rate": round(rng.uniform(0, 0.1), 2),
        "dst_host_serror_rate": serror_rate,
        "dst_host_srv_serror_rate": serror_rate,
        "dst_host_rerror_rate": round(rng.uniform(0, 0.2), 2),
        "dst_host_srv_rerror_rate": round(rng.uniform(0, 0.2), 2),
        "label": label, "difficulty": rng.integers(1, 21),
    }
    return row


def main():
    ensure_dirs()

    ok_train = _download(TRAIN_URL, TRAIN_PATH)
    ok_test = _download(TEST_URL, TEST_PATH)

    if not ok_train:
        print("Falling back to SYNTHETIC training data (offline demo mode).")
        _generate_synthetic(TRAIN_PATH, n_rows=15000, seed=42)
    if not ok_test:
        print("Falling back to SYNTHETIC test data (offline demo mode).")
        _generate_synthetic(TEST_PATH, n_rows=4000, seed=7)

    print("Done. Data available at:")
    print(f"  {TRAIN_PATH}")
    print(f"  {TEST_PATH}")


if __name__ == "__main__":
    main()
