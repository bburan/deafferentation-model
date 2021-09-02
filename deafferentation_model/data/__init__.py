from pathlib import Path

import numpy as np
import pandas as pd

#from .poles import compute_poles_audiogram


def _load_StElizabeth_files():
    base_path = Path('/media/nutshell/work/OHSU/data/CMAP_data/StElizabeth')
    base_path = base_path / "ABR data from Bramhall St. Elizabeth's study.csv"
    data = pd.read_csv(base_path)
    return data.dropna(axis='columns').rename(columns={'Unnamed: 8': 'sex'})


def load_poles_StElizabeth():
    data = _load_StElizabeth_files()
    audiograms = data.iloc[:, -6:]
    audiograms.columns = audiograms.columns.values.astype('f')
    return compute_poles_audiogram(audiograms)


def load_data_StElizabeth():
    data = _load_StElizabeth_files()
    return {
        'i_subject': data['Ear number'].values.astype('i') - 1,
        'i_sex': (data['sex'] == 'F').values.astype('i'),
        'i_stimuli': np.zeros(len(data)),
        'w1': data['Wave I amp peak to trough'].values,
    }


def load_subject_data_StElizabeth():
    data = _load_StElizabeth_files()
    data = data.iloc[:, :-6]
    data = data.rename(columns={
        'Wave I amp peak to trough': 'w1',
        'Age': 'age',
        'SL': 'SL',
        'Speech discrim': 'speech_discrim',
    })
    info = load_data_StElizabeth()
    for k, v in info.items():
        data[k] = v
    return data


def load_data(study):
    return globals()[f'load_data_{study}']()


def load_subject_data(study):
    return globals()[f'load_subject_data_{study}']()


def load_poles(study):
    return globals()[f'load_poles_{study}']()
