from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

zheng_all_files = [el.as_posix() for el in Path('ecg_data/zheng2020').glob('**/*.*')]
zheng_all_files.sort()
assert len(zheng_all_files) == 10648

all_data = []
for el in tqdm(zheng_all_files):
    if el.endswith('.csv'):
        tmp_data = pd.read_csv(el)
        assert np.all(tmp_data.columns == ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])
        tmp_data = tmp_data.to_numpy()
        all_data.append(tmp_data)
for el in all_data:
    assert el.shape == (5000, 12)
all_data = np.asarray(all_data)
all_data = all_data.transpose(0, 2, 1)

meta_info = pd.read_excel('ecg_data/zheng2020/Diagnostics.xlsx')
meta_info['freq'] = 500
meta_info['shapes'] = [(12, 5000) for _ in range(len(all_data))]
meta_info['raw_wave_length'] = 5000
meta_info['wave_second'] = 10
meta_info['offsets'] = [i * 5000 * 12 for i in range(len(all_data))]

# save wave file
fp = np.memmap('zheng2020.npy', np.float64, mode='w+', shape=(np.product(all_data.shape),))

for el1, el2 in tqdm(zip(all_data, meta_info['offsets'])):
    tmp_data = el1.flatten()
    fp[el2:el2 + len(tmp_data)] = tmp_data
fp.flush()
assert np.copy(fp[:2]).itemsize == 8

all_beats = []
[all_beats.extend(el.split(' ')) for el in meta_info['Beat']]
all_beats = list(set(all_beats))
all_beats.sort()

# Extract features
df_dx = pd.DataFrame(columns=all_beats)
for el in tqdm(all_beats):
    col = [True if el in el2.split(' ') else False for el2 in meta_info['Beat']]
    df_dx[el] = col
df_dx['Rhythm_num'] = pd.Categorical(meta_info['Rhythm']).codes
final_df = pd.concat([meta_info, df_dx], axis=1)
final_df.to_pickle('zheng2020_meta_info.pkl')


####################################################################################################
def zheng2020_loader(file_path: str, idx: int, meta_df: pd.DataFrame):
    fp2 = np.memmap(file_path, np.float64, mode='r', shape=tuple(meta_df['shapes'][idx]),
                    offset=8 * meta_df['offsets'][idx])
    return np.copy(fp2)

####################################################################################################

# meta_df = pd.read_pickle('zheng2020_meta_info.pkl')
# for i, el in tqdm(enumerate(all_data)):
#     A = zheng2020_loader('zheng2020.npy', i, meta_df)
#     assert np.all(A == el)
