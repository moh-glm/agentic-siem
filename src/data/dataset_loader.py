
from pathlib import Path
import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw dataset directory
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "unsw_nb15"


# Official UNSW-NB15 feature names (49 columns)
COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state",
    "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss",
    "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts",
    "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz",
    "trans_depth", "res_bdy_len", "Sjit", "Djit", "Stime",
    "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack",
    "ackdat", "is_sm_ips_ports", "ct_state_ttl",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "attack_cat", "Label"
]


FILES = [
    "UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv",
]

def iter_dataset_chunks(chunksize: int = 100_000):
    """
    Read the complete UNSW-NB15 dataset in manageable chunks.

    Yields one DataFrame at a time to limit memory usage.
    """

    if chunksize <= 0:
        raise ValueError("chunksize must be greater than zero")

    for filename in FILES:
        file_path = DATA_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {file_path}"
            )

        print(f"Reading: {filename}")

        for chunk in pd.read_csv(
            file_path,
            header=None,
            names=COLUMNS,
            chunksize=chunksize,
            low_memory=False,
        ):
            yield chunk
