import pandas as pd
import pyarrow.parquet as pq
import os


THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_FILE_DIR, "data/vsb-power-line-fault-detection")


def load_train() -> (pd.DataFrame, pd.DataFrame):
    train_data = pq.read_pandas(os.path.join(DATA_DIR, "train.parquet")).to_pandas()
    train_meta = pd.read_csv(
        os.path.join(DATA_DIR, "metadata_train.csv"), index_col="signal_id"
    )
    return train_data, train_meta


def load_test() -> (pd.DataFrame, pd.DataFrame):
    test_data = pq.read_pandas(os.path.join(DATA_DIR, "test.parquet")).to_pandas()
    test_meta = pd.read_csv(
        os.path.join(DATA_DIR, "metadata_test.csv"), index_col="signal_id"
    )
    return test_data, test_meta
