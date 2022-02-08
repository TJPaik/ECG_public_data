# %%
# To 500HZ
# %%
from tqdm import tqdm
from scipy.fft import dct, idct
import numpy as np
import pandas as pd


def normalize(wave: np.array, wanted_std: float):
    std = wave.std()
    if std == 0:
        return wave
    else:
        return (wave / wave.std()) * wanted_std


def padding(wave: np.array, wanted_pt_length: int):
    return np.concatenate([
        wave,
        np.zeros((len(wave), wanted_pt_length - wave.shape[1]))
    ], axis=1)


def cg_hz(wave: np.array, wanted_pt_length: int):
    if wave.shape[1] == wanted_pt_length:
        return wave.astype(np.float32)
    elif wave.shape[1] > wanted_pt_length:
        return normalize(idct(dct(wave, axis=1)[:, :wanted_pt_length], axis=1), wave.std())
    else:
        return normalize(idct(padding(dct(wave, axis=1), wanted_pt_length), axis=1), wave.std())


# %%
def cinc2020_loader(file_path: str, idx: int, meta_df: pd.DataFrame):
    fp2 = np.memmap(file_path, np.int, mode='r', shape=tuple(meta_df['shapes'][idx]),
                    offset=8 * meta_df['offsets'][idx])
    return np.copy(fp2)


meta_df = pd.read_pickle('cinc2020_meta_info.pkl')


# wanted_wave_length = ((500 / meta_df['freq']) * meta_df['raw_wave_length']).to_numpy()
# for el in wanted_wave_length:
#     if el != int(el):
#         print(el)
# wanted_wave_length = np.asarray([round(el) for el in wanted_wave_length])
# for el in wanted_wave_length:
#     if el != int(el):
#         print(el)
#
# all_num = sum(wanted_wave_length) * 12
# fp = np.memmap('cinc2020_500.npy', dtype=np.float32, mode='w+', shape=(all_num,))
# offsets = []
# offset = 0
#
# for i, el in tqdm(enumerate(wanted_wave_length)):
#     original_data = cinc2020_loader('cinc2020.npy', i, meta_df).astype(np.float32)
#     new_data = cg_hz(original_data, el)
#     tmp = new_data.flatten().astype(np.float32)
#     offsets.append(offset)
#     fp[offset:offset + len(tmp)] = tmp
#     offset += len(tmp)
# offsets = np.asarray(offsets)

# meta_df['new_raw_wave_length'] = wanted_wave_length
# meta_df['new_offsets'] = offsets
# meta_df.to_pickle('cinc2020_meta_info.pkl')

def cinc2020_500_loader(file_path: str, idx: int, meta_df: pd.DataFrame):
    fp2 = np.memmap(file_path, np.float32, mode='r', shape=(12, meta_df['new_raw_wave_length'][idx]),
                    offset=4 * meta_df['new_offsets'][idx])
    return np.copy(fp2)


# %%
def ptb_xl_loader(file_path: str, idx: int, meta_df: pd.DataFrame):
    fp2 = np.memmap(file_path, np.float64, mode='r', shape=tuple(meta_df['shapes'][idx]),
                    offset=8 * meta_df['offsets'][idx])
    return np.copy(fp2)


meta_df = pd.read_pickle('ptb_xl_meta_info.pkl')


# wanted_wave_length = ((500 / meta_df['freq']) * meta_df['raw_wave_length']).to_numpy()
# for el in wanted_wave_length:
#     if el != int(el):
#         print(el)
#
# wanted_wave_length = np.asarray([round(el) for el in wanted_wave_length])
# for el in wanted_wave_length:
#     if el != int(el):
#         print(el)
#
# all_num = sum(wanted_wave_length) * 12
# fp = np.memmap('ptb_xl_500.npy', dtype=np.float32, mode='w+', shape=(all_num,))
# offsets = []
# offset = 0
#
# for i, el in tqdm(enumerate(wanted_wave_length)):
#     original_data = ptb_xl_loader('ptb_xl.npy', i, meta_df).astype(np.float32)
#     new_data = cg_hz(original_data, el)
#     tmp = new_data.flatten().astype(np.float32)
#     offsets.append(offset)
#     fp[offset:offset + len(tmp)] = tmp
#     offset += len(tmp)
# offsets = np.asarray(offsets)
#
# meta_df['new_raw_wave_length'] = wanted_wave_length
# meta_df['new_offsets'] = offsets
# meta_df.to_pickle('ptb_xl_meta_info.pkl')


def ptb_xl_500_loader(file_path: str, idx: int, meta_df: pd.DataFrame):
    fp2 = np.memmap(file_path, np.float32, mode='r', shape=(12, meta_df['new_raw_wave_length'][idx]),
                    offset=4 * meta_df['new_offsets'][idx])
    return np.copy(fp2)


# %%

def ribeiro2020_train_loader(file_path: str, idx: int, meta_df: pd.DataFrame):
    fp2 = np.memmap(file_path, np.float32, mode='r', shape=tuple(meta_df['shapes'][idx]),
                    offset=4 * meta_df['offsets'][idx])
    return np.copy(fp2)


meta_df = pd.read_pickle('ribeiro2020_train_meta_info.pkl')


# wanted_wave_length = ((500 / meta_df['freq']) * meta_df['raw_wave_length']).to_numpy()
# for el in wanted_wave_length:
#     if el != int(el):
#         print(el)
#
# wanted_wave_length = np.asarray([round(el) for el in wanted_wave_length])
# for el in wanted_wave_length:
#     if el != int(el):
#         print(el)
#
# all_num = sum(wanted_wave_length) * 12
# fp = np.memmap('ribeiro2020_train_500.npy', dtype=np.float32, mode='w+', shape=(all_num,))
# offsets = []
# offset = 0
# #
# for i, el in tqdm(enumerate(wanted_wave_length)):
#     original_data = ribeiro2020_train_loader('ribeiro2020_train.npy', i, meta_df).astype(np.float32)
#     new_data = cg_hz(original_data, el)
#     tmp = new_data.flatten().astype(np.float32)
#     offsets.append(offset)
#     fp[offset:offset + len(tmp)] = tmp
#     offset += len(tmp)
# offsets = np.asarray(offsets)
#
# meta_df['new_raw_wave_length'] = wanted_wave_length
# meta_df['new_offsets'] = offsets
# meta_df.to_pickle('ribeiro2020_train_meta_info.pkl')


def ribeiro2020_train_500_loader(file_path: str, idx: int, meta_df: pd.DataFrame):
    fp2 = np.memmap(file_path, np.float32, mode='r', shape=(12, meta_df['new_raw_wave_length'][idx]),
                    offset=4 * meta_df['new_offsets'][idx])
    return np.copy(fp2)


# %%k

def ribeiro2020_test_loader(file_path: str, idx: int, meta_df: pd.DataFrame):
    fp2 = np.memmap(file_path, np.float64, mode='r', shape=tuple(meta_df['shapes'][idx]),
                    offset=8 * meta_df['offsets'][idx])
    return np.copy(fp2)


meta_df = pd.read_pickle('ribeiro2020_test_meta_info.pkl')


# wanted_wave_length = ((500 / meta_df['freq']) * meta_df['raw_wave_length']).to_numpy()
# for el in wanted_wave_length:
#     if el != int(el):
#         print(el)
#
# wanted_wave_length = np.asarray([round(el) for el in wanted_wave_length])
# for el in wanted_wave_length:
#     if el != int(el):
#         print(el)
#
# all_num = sum(wanted_wave_length) * 12
# fp = np.memmap('ribeiro2020_test_500.npy', dtype=np.float32, mode='w+', shape=(all_num,))
# offsets = []
# offset = 0
# #
# for i, el in tqdm(enumerate(wanted_wave_length)):
#     original_data = ribeiro2020_test_loader('ribeiro2020_test.npy', i, meta_df).astype(np.float32)
#     new_data = cg_hz(original_data, el)
#     tmp = new_data.flatten().astype(np.float32)
#     offsets.append(offset)
#     fp[offset:offset + len(tmp)] = tmp
#     offset += len(tmp)
# offsets = np.asarray(offsets)
#
# meta_df['new_raw_wave_length'] = wanted_wave_length
# meta_df['new_offsets'] = offsets
# meta_df.to_pickle('ribeiro2020_test_meta_info.pkl')


def ribeiro2020_test_500_loader(file_path: str, idx: int, meta_df: pd.DataFrame):
    fp2 = np.memmap(file_path, np.float32, mode='r', shape=(12, meta_df['new_raw_wave_length'][idx]),
                    offset=4 * meta_df['new_offsets'][idx])
    return np.copy(fp2)


# %%

def zheng2020_loader(file_path: str, idx: int, meta_df: pd.DataFrame):
    fp2 = np.memmap(file_path, np.float64, mode='r', shape=tuple(meta_df['shapes'][idx]),
                    offset=8 * meta_df['offsets'][idx])
    return np.copy(fp2)


meta_df = pd.read_pickle('zheng2020_meta_info.pkl')


# wanted_wave_length = ((500 / meta_df['freq']) * meta_df['raw_wave_length']).to_numpy()
# for el in wanted_wave_length:
#     if el != int(el):
#         print(el)
#
# wanted_wave_length = np.asarray([round(el) for el in wanted_wave_length])
# for el in wanted_wave_length:
#     if el != int(el):
#         print(el)
#
# all_num = sum(wanted_wave_length) * 12
# fp = np.memmap('zheng2020_500.npy', dtype=np.float32, mode='w+', shape=(all_num,))
# offsets = []
# offset = 0
# #
# for i, el in tqdm(enumerate(wanted_wave_length)):
#     original_data = zheng2020_loader('zheng2020.npy', i, meta_df).astype(np.float32)
#     new_data = cg_hz(original_data, el)
#     tmp = new_data.flatten().astype(np.float32)
#     offsets.append(offset)
#     fp[offset:offset + len(tmp)] = tmp
#     offset += len(tmp)
# offsets = np.asarray(offsets)
#
# meta_df['new_raw_wave_length'] = wanted_wave_length
# meta_df['new_offsets'] = offsets
# meta_df.to_pickle('zheng2020_meta_info.pkl')


def zheng2020_500_loader(file_path: str, idx: int, meta_df: pd.DataFrame):
    fp2 = np.memmap(file_path, np.float32, mode='r', shape=(12, meta_df['new_raw_wave_length'][idx]),
                    offset=4 * meta_df['new_offsets'][idx])
    return np.copy(fp2)
