import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

NUM_THRESHOLD = 2000
path_to_files = [f'ecg_data/ribeiro2020_train/exams_part{i}.hdf5' for i in range(18)]
fs = [h5py.File(el, 'r') for el in tqdm(path_to_files)]

for el in fs:
    assert tuple([el2 for el2 in el.keys()]) == ('exam_id', 'tracings')

ecg_data = []
for el in tqdm(fs):
    ecg_data.append(
        np.array(el['tracings'])
    )

all_ecg_data = np.concatenate(ecg_data)
all_ecg_data = all_ecg_data.transpose(0, 2, 1) * 1000

S = []
E = []
for el in tqdm(all_ecg_data):
    tmp = np.where(~np.all(el == 0, axis=0))[0]
    if len(tmp) == 0:
        S.append(-1)
        E.append(-2)
    else:
        S.append(tmp[0])
        E.append(tmp[-1])
S = np.asarray(S)
E = np.asarray(E)
Q = E - S + 1

valid_index = (Q > NUM_THRESHOLD)
S = S[valid_index]
E = E[valid_index]
Q = Q[valid_index]
all_ecg_valid = all_ecg_data[valid_index]

all_data_num = sum(Q) * 12

# save wave file
fp = np.memmap('ribeiro2020_train.npy', np.float32, mode='w+', shape=(all_data_num,))
offsets = []
tmp_num = 0
for el1, el2, el3 in tqdm(zip(all_ecg_valid, S, E)):
    offsets.append(tmp_num)
    tmp_data = el1[:, el2: el3 + 1].flatten()
    fp[tmp_num: tmp_num + len(tmp_data)] = tmp_data
    tmp_num += len(tmp_data)
fp.flush()
assert np.copy(fp[:2]).itemsize == 4

ecg_ids = []
for el in tqdm(fs):
    ecg_ids.append(
        np.array(el['exam_id'])
    )
ecg_ids = np.concatenate(ecg_ids)
ecg_ids = ecg_ids[valid_index]
assert len(ecg_ids) == len(set(ecg_ids))

meta_info = pd.read_csv('ecg_data/ribeiro2020_train/exams.csv')

meta_info_final = pd.DataFrame()
meta_info_final['exam_id'] = ecg_ids
meta_info_final = meta_info_final.merge(meta_info, on='exam_id')
meta_info_final['offsets'] = np.asarray(offsets)
meta_info_final['shapes'] = [tuple([12, el]) for el in Q]
meta_info_final['freq'] = 400
meta_info_final['wave_second'] = Q / 400
meta_info_final['raw_wave_length'] = Q
meta_info_final.to_pickle('ribeiro2020_train_meta_info.pkl')


####################################################################################################
def ribeiro2020_train_loader(file_path: str, idx: int, meta_df: pd.DataFrame):
    fp2 = np.memmap(file_path, np.float32, mode='r', shape=tuple(meta_df['shapes'][idx]),
                    offset=4 * meta_df['offsets'][idx])
    return np.copy(fp2)


####################################################################################################

# meta_df = pd.read_pickle('ribeiro2020_train_meta_info.pkl')
# for i, el in tqdm(enumerate(all_ecg_valid)):
#     A = ribeiro2020_train_loader('ribeiro2020_train.npy', i, meta_df)
#     assert np.all(A == el[:, S[i]:E[i] + 1])

########################################################
########################################################
########################################################
########################################################

import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm

path_to_file = f'ecg_data/ribeiro2020_test/ecg_tracings.hdf5'
f = h5py.File(path_to_file, 'r')
ecg_data = np.array(f['tracings'])
ecg_data = ecg_data.transpose(0, 2, 1)
ecg_data = ecg_data.astype(np.float64) * 1000

S = []
E = []
for el in tqdm(ecg_data):
    tmp = np.where(~np.all(el == 0, axis=0))[0]
    if len(tmp) == 0:
        S.append(-1)
        E.append(-2)
    else:
        S.append(tmp[0])
        E.append(tmp[-1])
S = np.asarray(S)
E = np.asarray(E)
Q = E - S + 1

all_data_num = sum(Q) * 12

# save wave file
fp = np.memmap('ribeiro2020_test.npy', np.float64, mode='w+', shape=(all_data_num,))
offsets = []
tmp_num = 0
for el1, el2, el3 in tqdm(zip(ecg_data, S, E)):
    offsets.append(tmp_num)
    tmp_data = el1[:, el2: el3 + 1].flatten()
    fp[tmp_num: tmp_num + len(tmp_data)] = tmp_data
    tmp_num += len(tmp_data)
fp.flush()
assert np.copy(fp[:2]).itemsize == 8

meta_info1 = pd.read_csv('ecg_data/ribeiro2020_test/attributes.csv')
meta_info2 = pd.read_csv('ecg_data/ribeiro2020_test/annotations/gold_standard.csv')
meta_info = pd.concat([meta_info1, meta_info2], axis=1)
meta_info['offsets'] = np.asarray(offsets)
meta_info['shapes'] = [tuple([12, el]) for el in Q]
meta_info['freq'] = 400
meta_info['wave_second'] = Q / 400
meta_info['raw_wave_length'] = Q
meta_info.to_pickle('ribeiro2020_test_meta_info.pkl')


####################################################################################################
def ribeiro2020_test_loader(file_path: str, idx: int, meta_df: pd.DataFrame):
    fp2 = np.memmap(file_path, np.float64, mode='r', shape=tuple(meta_df['shapes'][idx]),
                    offset=8 * meta_df['offsets'][idx])
    return np.copy(fp2)

####################################################################################################

# meta_df = pd.read_pickle('ribeiro2020_test_meta_info.pkl')
# for i in tqdm(range(len(ecg_data))):
#     A = ribeiro2020_test_loader('ribeiro2020_test.npy', i, meta_df)
#     assert np.all(A == ecg_data[i][:, S[i]:E[i] + 1])
