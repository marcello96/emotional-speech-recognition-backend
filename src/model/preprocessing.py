import os

import librosa
import numpy as np
import pandas as pd

# constants
NUMBER_OF_MFCCS = 25
RAVDESS_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
MODEL_LABELS = sorted(RAVDESS_LABELS)


def map_ravdess_filename_to_label(filename):
    label_no = int(filename[7:8]) - 1

    return RAVDESS_LABELS[label_no]


def extract_feature(row, mfccs_no=NUMBER_OF_MFCCS):
    file = row['file']
    try:
        series, sample_rate = librosa.load(file, res_type='kaiser_fast', offset=1.0, duration=2.5)
        mfcc = np.mean(librosa.feature.mfcc(y=series, sr=sample_rate, n_mfcc=mfccs_no).T, axis=0)
    except Exception:
        print("Error encountered while parsing file: ", file)
        return None, None

    feature = mfcc
    label = row['label']
    return [feature, label]


def load_files(directory_path, map_file_to_label=map_ravdess_filename_to_label):
    directories = os.listdir(directory_path)
    file_label_dict = dict()
    for actor_folder in directories:
        files = os.listdir(os.path.join(directory_path, actor_folder))
        for f in files:
            if f.endswith('.wav'):
                file_label_dict[os.path.join(directory_path, actor_folder, f)] = map_file_to_label(f)

    train = pd.DataFrame.from_dict(file_label_dict, orient='index')
    train = train.reset_index(drop=False)
    train = train.rename(columns={'index': 'file', 0: 'label'})
    train = train[['label', 'file']]

    return train


def prepare_learning_data(data, train_data_frac=0.80):
    train_files = data.sample(frac=train_data_frac, random_state=200)
    test_files = data.drop(train_files.index)

    train_data = train_files.apply(extract_feature, axis=1, result_type='broadcast')
    train_data.columns = ['feature', 'label']

    test_data = test_files.apply(extract_feature, axis=1, result_type='broadcast')
    test_data.columns = ['feature', 'label']

    x = np.array(train_data['feature'].tolist())
    y = np.array(train_data['label'].tolist())
    val_x = np.array(test_data['feature'].tolist())
    val_y = np.array(test_data['label'].tolist())

    return x, y, val_x, val_y
