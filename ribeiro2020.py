import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

def ribeiro2020_train_loader(file_path: str, idx: int, meta_df: pd.DataFrame):
    fp2 = np.memmap(file_path, np.float32, mode='r', shape=tuple(meta_df['shapes'][idx]),
                    offset=4 * meta_df['offsets'][idx])
    return np.copy(fp2)

# meta_df = pd.read_pickle('ribeiro2020_train_meta_info.pkl')
# for i, el in tqdm(enumerate(all_ecg_valid)):
#     A = ribeiro2020_train_loader('ribeiro2020_train.npy', i, meta_df)
#     assert np.all(A == el[:, S[i]:E[i] + 1])

import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm

def ribeiro2020_test_loader(file_path: str, idx: int, meta_df: pd.DataFrame):
    fp2 = np.memmap(file_path, np.float64, mode='r', shape=tuple(meta_df['shapes'][idx]),
                    offset=8 * meta_df['offsets'][idx])
    return np.copy(fp2)

# meta_df = pd.read_pickle('ribeiro2020_test_meta_info.pkl')
# for i in tqdm(range(len(ecg_data))):
#     A = ribeiro2020_test_loader('ribeiro2020_test.npy', i, meta_df)
#     assert np.all(A == ecg_data[i][:, S[i]:E[i] + 1])
