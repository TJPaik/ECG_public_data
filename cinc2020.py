import numpy as np, os
import pandas as pd
import pickle
from scipy.io import loadmat
from pathlib import Path
from tqdm import tqdm


def cinc2020_loader(file_path: str, idx: int, meta_df: pd.DataFrame):
    fp2 = np.memmap(file_path, np.int, mode='r', shape=tuple(meta_df['shapes'][idx]),
                    offset=8 * meta_df['offsets'][idx])
    return np.copy(fp2)

# meta_df = pd.read_pickle('cinc2020_meta_info.pkl')
# for i in tqdm(range(len(ecg_data))):
#     A = cinc2020_loader('cinc2020.npy', i, meta_df)
#     assert np.all(A == ecg_data[i])
