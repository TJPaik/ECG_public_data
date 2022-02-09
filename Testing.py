import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def ribeiro2020_train_500_loader(file_path: str, idx: int, meta_df: pd.DataFrame):
    fp2 = np.memmap(file_path, np.float32, mode='r', shape=(12, meta_df['new_raw_wave_length'][idx]),
                    offset=4 * meta_df['new_offsets'][idx])
    return np.copy(fp2)


def ribeiro2020_test_500_loader(file_path: str, idx: int, meta_df: pd.DataFrame):
    fp2 = np.memmap(file_path, np.float32, mode='r', shape=(12, meta_df['new_raw_wave_length'][idx]),
                    offset=4 * meta_df['new_offsets'][idx])
    return np.copy(fp2)


def zheng2020_500_loader(file_path: str, idx: int, meta_df: pd.DataFrame):
    fp2 = np.memmap(file_path, np.float32, mode='r', shape=(12, meta_df['new_raw_wave_length'][idx]),
                    offset=4 * meta_df['new_offsets'][idx])
    return np.copy(fp2)


def ptb_xl_500_loader(file_path: str, idx: int, meta_df: pd.DataFrame):
    fp2 = np.memmap(file_path, np.float32, mode='r', shape=(12, meta_df['new_raw_wave_length'][idx]),
                    offset=4 * meta_df['new_offsets'][idx])
    return np.copy(fp2)


def cinc2020_500_loader(file_path: str, idx: int, meta_df: pd.DataFrame):
    fp2 = np.memmap(file_path, np.float32, mode='r', shape=(12, meta_df['new_raw_wave_length'][idx]),
                    offset=4 * meta_df['new_offsets'][idx])
    return np.copy(fp2)


meta_df1 = pd.read_pickle('ribeiro2020_train_meta_info.pkl')
meta_df2 = pd.read_pickle('ribeiro2020_test_meta_info.pkl')
meta_df3 = pd.read_pickle('zheng2020_meta_info.pkl')
meta_df4 = pd.read_pickle('ptb_xl_meta_info.pkl')
meta_df5 = pd.read_pickle('cinc2020_meta_info.pkl')
# %%
wave = ribeiro2020_train_500_loader('ribeiro2020_train_500.npy', 10, meta_df1)
plt.plot(wave[0])
plt.plot(wave[-1])
plt.show()

# %%
wave = ribeiro2020_test_500_loader('ribeiro2020_test_500.npy', 10, meta_df2)
plt.plot(wave[0])
plt.plot(wave[-1])
plt.show()
# %%
wave = zheng2020_500_loader('zheng2020_500.npy', 10, meta_df3)
plt.plot(wave[0])
plt.plot(wave[-1])
plt.show()
# %%
wave = ptb_xl_500_loader('ptb_xl_500.npy', 10, meta_df4)
plt.plot(wave[0])
plt.plot(wave[-1])
plt.show()
# %%
wave = cinc2020_500_loader('cinc2020_500.npy', 10, meta_df5)
plt.plot(wave[0])
plt.plot(wave[-1])
plt.show()
