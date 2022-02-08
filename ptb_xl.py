# %%
import pandas as pd
import numpy as np
from itertools import chain
from tqdm import tqdm
import wfdb
import ast


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


path = 'ecg_data/ptb_xl/'

# load and convert annotation data
Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
# %%
X_100 = load_raw_data(Y, 100, path)
X_500 = load_raw_data(Y, 500, path)
X_100 = X_100.transpose(0, 2, 1) * 1000
X_500 = X_500.transpose(0, 2, 1) * 1000
# %%
Y_100 = Y.copy()
Y_500 = Y.copy()
# %%
Y_100['freq'] = 100
Y_500['freq'] = 500
# %%
Y_100['raw_wave_length'] = 1000
Y_500['raw_wave_length'] = 5000
# %%
Y_100['shapes'] = [(12, 1000) for _ in range(len(X_100))]
Y_500['shapes'] = [(12, 5000) for _ in range(len(X_500))]
# %%
Y_final = pd.concat([Y_100, Y_500])
Y_final['wave_second'] = 10
# %%
offsets = [12 * 1000 * i for i in range(len(X_100))]
offsets += [(offsets[-1] + 12 * 1000) + (12 * 5000 * i) for i in range(len(X_500))]
offsets = np.asarray(offsets)
# %%
Y_final['offsets'] = offsets
# %%
all_data_num = np.product(X_100.shape) + np.product(X_500.shape)
# %%
fp = np.memmap('ptb_xl.npy', np.float64, mode='w+', shape=(all_data_num,))
# %%
for el1, el2 in tqdm(zip(chain(X_100, X_500), offsets)):
    tmp_data = el1.flatten()
    fp[el2:  el2 + len(tmp_data)] = tmp_data
assert np.copy(fp[:2]).itemsize == 8
fp.flush()
Y_final = Y_final.reset_index()
# %%
Y_final.to_pickle('ptb_xl_meta_info.pkl')


# %%
####################################################################################################
def ptb_xl_loader(file_path: str, idx: int, meta_df: pd.DataFrame):
    fp2 = np.memmap(file_path, np.float64, mode='r', shape=tuple(meta_df['shapes'][idx]),
                    offset=8 * meta_df['offsets'][idx])
    return np.copy(fp2)

####################################################################################################
# %%
# meta_df = pd.read_pickle('ptb_xl_meta_info.pkl')
# for i, el in tqdm(enumerate(chain(X_100, X_500))):
#     A = ptb_xl_loader('ptb_xl.npy', i, meta_df)
#     assert np.all(A == el)
