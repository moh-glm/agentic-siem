import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.agents.agent1.sampling import sample_dataset


class Agent1SamplingTests(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_dir = Path(self.temp_dir.name)

        # Create a small synthetic dataset.
        df = pd.DataFrame({
            "sport": range(100),
            "dsport": range(100),
            "proto": ["tcp"] * 100,
            "state": ["CON"] * 100,
            "dur": [0.1] * 100,
            "service": ["http"] * 100,
            "Label": [0] * 90 + [1] * 10,
        })

        for column in [
            "sbytes", "dbytes", "sttl", "dttl",
            "sloss", "dloss", "Sload", "Dload",
            "Spkts", "Dpkts", "swin", "dwin",
            "stcpb", "dtcpb", "smeansz", "dmeansz",
            "trans_depth", "res_bdy_len", "Sjit",
            "Djit", "Sintpkt", "Dintpkt", "tcprtt",
            "synack", "ackdat", "is_sm_ips_ports",
            "ct_state_ttl", "ct_flw_http_mthd",
            "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
            "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
            "ct_src_dport_ltm", "ct_dst_sport_ltm",
            "ct_dst_src_ltm",
        ]:
            df[column] = 0

        self.file_path = self.dataset_dir / "sample_processed.csv"
        df.to_csv(self.file_path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sample_size_and_class_balance(self):
        with patch(
            "src.agents.agent1.sampling.PROCESSED_DIR",
            self.dataset_dir,
        ):
            result = sample_dataset(
                sample_size=50,
                chunksize=25,
            )

        self.assertEqual(len(result), 50)
        self.assertEqual(result["Label"].sum(), 5)
        self.assertEqual(result["Label"].value_counts()[0], 45)

    def test_sampling_is_reproducible(self):
        with patch(
            "src.agents.agent1.sampling.PROCESSED_DIR",
            self.dataset_dir,
        ):
            first = sample_dataset(50, 25)
            second = sample_dataset(50, 25)

        pd.testing.assert_frame_equal(first, second)

    def test_sample_larger_than_dataset_is_rejected(self):
        with patch(
            "src.agents.agent1.sampling.PROCESSED_DIR",
            self.dataset_dir,
        ):
            with self.assertRaises(ValueError):
                sample_dataset(sample_size=200)

    def test_invalid_sample_size_is_rejected(self):
        with self.assertRaises(ValueError):
            sample_dataset(sample_size=0)


if __name__ == "__main__":
    unittest.main()
