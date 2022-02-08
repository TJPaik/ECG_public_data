import pandas as pd
from tqdm import tqdm
import numpy as np
from pathlib import Path


def zheng2020_loader(file_path: str, idx: int, meta_df: pd.DataFrame):
    fp2 = np.memmap(file_path, np.float64, mode='r', shape=tuple(meta_df['shapes'][idx]),
                    offset=8 * meta_df['offsets'][idx])
    return np.copy(fp2)

# meta_df = pd.read_pickle('zheng2020_meta_info.pkl')
# for i, el in tqdm(enumerate(all_data)):
#     A = zheng2020_loader('zheng2020.npy', i, meta_df)
#     assert np.all(A == el)
