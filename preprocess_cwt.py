from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from scipy import signal
import load
import numpy as np
from tqdm import tqdm
import argparse


Fs = 40000000  # TODO do we even need this?
n = int(Fs * 20e-3)  # 20e-3 is the frequency of the signal
k = 24


def wavelet(s):
    downsampled = signal.resample(s, n // 2 // 400)
    widths = [2 ** (j / k) for j in range(1, 101)]
    z = signal.cwt(downsampled, signal.ricker, widths)
    return z


def preprocess(loader, output):
    signals, meta = loader()
    result = np.zeros((int(1e5), signals.shape[1]), dtype=np.float32)
    for i in tqdm(range(signals.shape[1])):
        z = wavelet(signals.iloc[:, i])
        z = np.ravel(z)
        result[:, i] = z.astype(np.float32)
    np.save(output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    args = parser.parse_args()
    if args.train:
        preprocess(load.load_train, "data/preprocessed/train.npy")
    if args.test:
        preprocess(load.load_test, "data/preprocessed/test.npy")
