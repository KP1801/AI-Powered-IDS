"""
utils.py
Shared constants and helper functions for the AI-Powered IDS project.

Dataset: NSL-KDD (an improved version of the classic KDD Cup 1999 dataset).
Each record describes a single network connection with 41 features plus a label.
"""

import os

# ---------------------------------------------------------------------------
# Column names for the NSL-KDD dataset (41 features + label + difficulty score)
# ---------------------------------------------------------------------------
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

# Categorical (non-numeric) columns that need encoding
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# ---------------------------------------------------------------------------
# NSL-KDD attack labels grouped into 5 top-level categories.
# This mapping is what makes the classifier's output human-readable and is
# also used to build the multi-class explanation report.
# ---------------------------------------------------------------------------
ATTACK_CATEGORY_MAP = {
    "normal": "Normal",

    # DoS - Denial of Service
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS",
    "apache2": "DoS", "processtable": "DoS", "udpstorm": "DoS",

    # Probe - surveillance / port scanning
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",

    # R2L - Remote to Local (unauthorized access from a remote machine)
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "xlock": "R2L", "xsnoop": "R2L",
    "snmpguess": "R2L", "snmpgetattack": "R2L", "httptunnel": "R2L",
    "sendmail": "R2L", "named": "R2L", "worm": "R2L",

    # U2R - User to Root (privilege escalation)
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}

CATEGORY_LIST = ["Normal", "DoS", "Probe", "R2L", "U2R"]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def map_attack_category(label: str) -> str:
    """Map a raw NSL-KDD label to one of the 5 top-level categories.
    Unknown/unseen labels (the test set has a few attacks absent from
    training) fall back to 'Unknown' rather than crashing.
    """
    label = label.strip().lower().replace(".", "")
    return ATTACK_CATEGORY_MAP.get(label, "Unknown")


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
