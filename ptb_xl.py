import pandas as pd
import numpy as np
from itertools import chain
from tqdm import tqdm
import wfdb
import ast

def ptb_xl_loader(file_path: str, idx: int, meta_df: pd.DataFrame):
    fp2 = np.memmap(file_path, np.float64, mode='r', shape=tuple(meta_df['shapes'][idx]),
                    offset=8 * meta_df['offsets'][idx])
    return np.copy(fp2)

# meta_df = pd.read_pickle('ptb_xl_meta_info.pkl')
# for i, el in tqdm(enumerate(chain(X_100, X_500))):
#     A = ptb_xl_loader('ptb_xl.npy', i, meta_df)
#     assert np.all(A == el)
