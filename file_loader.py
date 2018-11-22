import pandas as pd  # data frame
import numpy as np  # matrix math
# from scipy.io import wavfile # reading the wavfile
import os  # interation with the OS
import matplotlib.pyplot as plt  # to view graphs
from scipy.io import wavfile
from tqdm import tqdm
import feature_extractor as fe
from keras.models import model_from_json
import librosa
import librosa.display
import network_model

PATH = "E:\Python\TESS"
RAVDESS = "E:\Python\Ravness"
TEST = ["03-01-08-02-01-01-13.wav", "happy.wav", "angry_test.wav", "happy_test.wav", "neutral_test.wav", "sad_test.wav"]


RAVDESS_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
LABELS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def load_files(path):
    # write the complete file loading function here, this will return
    # a dataframe having files and labels
    # loading the files
    train_labels = os.listdir(path)

    file_label_dict = dict()
    for label in train_labels:
        files = os.listdir(os.path.join(path, label))
        for f in files:
            file_label_dict[os.path.join(path, label, f)] = label

    train = pd.DataFrame.from_dict(file_label_dict, orient='index')
    train = train.reset_index(drop=False)
    train = train.rename(columns={'index': 'file', 0: 'label'})
    train = train[['label', 'file']]

    return train, train_labels


def load_ravness_files(path, labels):
    # loading the files
    directories = os.listdir(path)

    def map_filename_to_label(filename):
        label_no = int(filename[7:8]) - 1

        return labels[label_no]

    file_label_dict = dict()
    for actor_folder in directories:
        files = os.listdir(os.path.join(path, actor_folder))
        for f in files:
            file_label_dict[os.path.join(path, actor_folder, f)] = map_filename_to_label(f)

    train = pd.DataFrame.from_dict(file_label_dict, orient='index')
    train = train.reset_index(drop=False)
    train = train.rename(columns={'index': 'file', 0: 'label'})
    train = train[['label', 'file']]

    return train, labels


def audio_to_data(path):
    # we take a single path and convert it into data
    sample_rate, audio = wavfile.read(path)
    sec_duration = audio.shape[0] / sample_rate
    spectrogram = fe.log_spectrogram(audio, sample_rate, 10, 0)
    return spectrogram.T


def paths_to_data(paths, word2id):
    data = np.zeros(shape=(len(paths), 123, 153))
    labels = []
    indexes = []
    for i in tqdm(range(len(paths))):
        f = paths[i]
        audio = audio_to_data(paths[i])
        # if audio.shape != (81, 100):
        #     indexes.append(i)
        # else:
        data[i] = audio
        # print('Number of instances with inconsistent shape:', len(indexes))
        # mode, if unk is set we are doing it for unknown files
        labels.append(word2id[f.split(os.path.sep)[-2]])

    return data, labels, indexes


def parser(row):
    file = row['file']
    try:
        # here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file, res_type='kaiser_fast', offset=0.3, duration=2.5)
        # we extract mfcc feature from data
        mfcc = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=25).T, axis=0)

        # lpc = pysptk.sptk.lpc(X)
    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None, None

    feature = mfcc
    label = row['label']
    return [feature, label]



def open_and_lpc(path):
    X, sample_rate = librosa.load(path, res_type='kaiser_fast')
    # we extract mfcc feature from data
    mfcc = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=25).T, axis=0)

    return mfcc

def plot_history(acc, val_acc):
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('Skuteczność modelu z LPC')
    plt.ylabel('skuteczność')
    plt.xlabel('cykl')
    plt.legend(['treningowy', 'testowy'], loc='upper left')
    plt.show()


def plot_prediction(predictions, title, labels):
    # this is for plotting purpose
    index = np.arange(len(labels))
    plt.bar(index, predictions)
    plt.ylabel('prawdopodobieństwo', fontsize=9)
    plt.xticks(index, labels)
    plt.title(title)
    plt.show()


def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")


def read_model():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")

    return loaded_model


# data1, labels1 = load_files(PATH)
data, labels = load_ravness_files(RAVDESS, RAVDESS_LABELS)

# data, labels = load_files(PATH)
# split data into train and test set
#data = data.append(data1, ignore_index=True)
print(data)
train = data.sample(frac=0.80, random_state=200)
test = data.drop(train.index)
print(train)
temp = train.apply(parser, axis=1, result_type='broadcast')
temp.columns = ['feature', 'label']

tt = test.apply(parser, axis=1, result_type='broadcast')
tt.columns = ['feature', 'label']

# get train set distribution
# pp = train['label'].value_counts(normalize=True)

# word2id = dict((c, i+1) for i, c in enumerate(sorted(labels)))
# print(word2id)
#
# files = train['file'].values
# print("[!]For labled data...")
# data, l, i = paths_to_data(files[:1], word2id)
#
# plt.figure(figsize = (10, 10))
# plt.imshow(data[0])
# plt.title('Log Spectrogram')
# plt.show()

X = np.array(temp.feature.tolist())
y = np.array(temp.label.tolist())
val_x = np.array(tt.feature.tolist())
val_y2 = np.array(tt.label.tolist())
model, history, label_classes = network_model.cnn(X, y, val_x, val_y2, epochs=100)

#save_model(model)

plot_history(history.history['acc'], history.history['val_acc'])
#model = read_model()

for file in TEST:
    x_test = open_and_lpc(file).reshape(1, -1)
    prediction = model.predict(np.expand_dims(x_test, axis=2))
    print(prediction[0])
    plot_prediction(prediction[0], file, LABELS)
