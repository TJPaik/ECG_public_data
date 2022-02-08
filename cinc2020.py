import pickle
from pathlib import Path

import numpy as np
import os
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm


def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file, 'r') as f:
        header_data = f.readlines()

    return data, header_data


cinc_all_files = [el.as_posix() for el in Path('ecg_data/cinc2020').glob('**/*.*')]
cinc_all_files.sort()
assert len(cinc_all_files) == 86208

ecg_data, ecg_header, file_names = [], [], []
for el in cinc_all_files:
    if not el.endswith('hea') and not el.endswith('mat') and not el.endswith('gz'):
        raise NotImplementedError()
for el in tqdm(cinc_all_files):
    if el.endswith('.mat'):
        tmp = load_challenge_data(el)
        ecg_data.append(tmp[0])
        ecg_header.append(tmp[1])
        file_names.append(el)
lead_data = np.asarray([el2.split(' ')[-1].strip() for el2 in ecg_header[0][1:13]])
assert np.all(lead_data == np.asarray(['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']))
for el in ecg_header:
    assert np.all(lead_data == np.asarray([el2.split(' ')[-1].strip() for el2 in el[1:13]]))
    assert np.all(np.asarray([el2.split(' ')[2].lower() for el2 in el[1:13]]) == '1000/mv')

freq = []
for el in ecg_header:
    freq.append(
        int(el[0].split(' ')[2])
    )
assert set(freq) == {1000, 257, 500}
assert len(ecg_data) == 43101

wave_second = []
raw_wave_length = []
for el1, el2 in zip(ecg_data, freq):
    wave_length = el1.shape[1]
    second = wave_length / el2
    wave_second.append(second)
    raw_wave_length.append(wave_length)
wave_second = np.asarray(wave_second)
raw_wave_length = np.asarray(raw_wave_length)

for idx, el in tqdm(enumerate(ecg_data)):
    tmp = el.astype(np.int)
    assert np.all(tmp == el)
    ecg_data[idx] = tmp

all_data_num = 0
offsets = []
shapes = []
for el in tqdm(ecg_data):
    offsets.append(all_data_num)
    tmp_shape = el.shape
    shapes.append(tmp_shape)
    all_data_num += (tmp_shape[0] * tmp_shape[1])
shapes = np.asarray(shapes)
offsets = np.asarray(offsets)

# Save wave file
fp = np.memmap('cinc2020.npy', np.int, mode='w+', shape=(all_data_num,))

for el1, el2 in tqdm(zip(ecg_data, offsets)):
    tmp_data = el1.flatten()
    fp[el2:el2 + len(tmp_data)] = tmp_data
fp.flush()
assert np.copy(fp[:2]).itemsize == 8

raw_header = np.asarray([pickle.dumps(el) for el in tqdm(ecg_header)])


def extract_features(inputs):
    found = False
    age = None
    for el in inputs:
        if str(el).startswith('#Age'):
            found = True
            try:
                age = int(el[5:])
            except ValueError:
                print(el.strip())
                age = None
            break
    if not found:
        print('error0')

    found = False
    sex = None
    for el in inputs:
        if str(el).startswith('#Sex'):
            found = True
            sex = el[5:].strip()
            if sex in ['Male', 'M']:
                sex = 'M'
            elif sex in ['Female', 'F']:
                sex = 'F'
            else:
                sex = None
            break
    if not found:
        print('error1')

    found = False
    dx_row = ''
    for el in inputs:
        if str(el).startswith('#Dx'):
            found = True
            dx_row = el
            break
    if not found:
        print('error2')
    dx_row = dx_row[4:]
    dx_row_list = [int(el) for el in dx_row.split(',')]

    return dx_row_list, sex, age


# Extract features
df = pd.DataFrame()
df['offsets'] = offsets
df['shapes'] = [tuple(el) for el in shapes]
df['freq'] = freq
df['file_names'] = file_names
df['wave_second'] = wave_second
df['raw_wave_length'] = raw_wave_length
df['raw_header'] = raw_header

features = [extract_features(el) for el in ecg_header]
dx_map = pd.read_csv('Dx_map.csv')
dx_map_function = {k: v for k, v in zip(dx_map['SNOMED CT Code'], dx_map['Abbreviation'])}
dx = [tuple([dx_map_function[el2] for el2 in el[0]]) for el in features]
sex = [el[1] for el in features]
age = [el[2] for el in features]

df['sex'] = sex
df['age'] = age

dx_all = []
[dx_all.extend(el) for el in dx]
dx_all = list(set(dx_all))
dx_all.sort()
df_dx = pd.DataFrame(columns=dx_all)
for el in tqdm(dx_all):
    col = [True if el in el2 else False for el2 in dx]
    df_dx[el] = col

df_final = pd.concat([df, df_dx], axis=1)
df_final.to_pickle('cinc2020_meta_info.pkl')


####################################################################################################
def cinc2020_loader(file_path: str, idx: int, meta_df: pd.DataFrame):
    fp2 = np.memmap(file_path, np.int, mode='r', shape=tuple(meta_df['shapes'][idx]),
                    offset=8 * meta_df['offsets'][idx])
    return np.copy(fp2)

####################################################################################################
# meta_df = pd.read_pickle('cinc2020_meta_info.pkl')
# for i in tqdm(range(len(ecg_data))):
#     A = cinc2020_loader('cinc2020.npy', i, meta_df)
#     assert np.all(A == ecg_data[i])
