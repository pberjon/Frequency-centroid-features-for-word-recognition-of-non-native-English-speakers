import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import shutil

##

#dir_src = "D:\\speech_commands_v0.01\\wav_files"
#dir_dest = "D:\\speech_commands_v0.01\\png_files"

def dataset(dir_src, dir_dest):
    """ take all the files of the dir_src directory and create spectrograms of them, then put these in the dir_dest directory. """
    for folder in os.listdir(dir_src):
        i=0
        for path in os.listdir(dir_src + "\\" + folder):
            try:
                X, sample_rate = librosa.load(dir_src + "\\" + folder + "\\" + path, res_type='kaiser_fast',duration=10, sr=22050*2, offset=0.5)
                melspec = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128)
                log_S = librosa.amplitude_to_db(melspec)

                plt.figure(figsize=(12,4))
                librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')
                plt.colorbar(format='%+02.0f dB')
                plt.tight_layout()
                plt.savefig(dir_dest + "\\" + path[:len(path)-4] + ".png")
                plt.close('all')
                i+=1
            except ValueError:
                pass
##

import os
import wave

import pylab
def graph_spectrogram(wav_file,name_dest):
    sound_info, frame_rate = get_wav_info(wav_file)
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.title('spectrogram of %r' % wav_file)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig(name_dest)
    pylab.close()
def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

##

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

dir_src = "D:\\PythonScripts\\new_study\\wav_files\\accents"
dir_dest = "D:\\PythonScripts\\new_study\\french_single_specs"
for accent in os.listdir(dir_src):
    for folder in os.listdir(dir_src + "\\" + accent):
        for path in os.listdir(dir_src+"\\"+accent+"\\"+folder):
            try:
                graph_spectrogram(dir_src+"\\"+accent+"\\"+folder+"\\"+path,dir_dest+"\\"+accent+"\\"+folder+"\\"+path[:len(path)-4]+".png")
            except ValueError:
                pass

def folders_creation(dir_dest):
    """ create a folder for each different spoken language"""

    files = os.listdir(dir_dest)
    for path in files:
        path = path[:len(path)-4]
        while path[len(path)-1].isdigit():
            path = path[:len(path)-1]
        try:
            os.mkdir(dir_dest + "\\" + path)
        except OSError:
            print ("Creation of the directory %s failed" % path)
        else:
            print ("Successfully created the directory %s " % path)

def png_copy(dir_dest):
    """ copy all the png files in the right folder """

    files = os.listdir(dir_dest)
    folders = []
    for name in files:
        if not(name.endswith(".png")):
            folders.append(name)
            files.remove(name)

    for file in files:
        for folder in folders:
            if folder in file:
                shutil.copy(dir_dest + "\\" + file, dir_dest + "\\" + folder)


def file_rename(dir_src):
    for folder in os.listdir(dir_src):
        i = 0
        for path in os.listdir(dir_src + "\\" + folder):
            os.rename(dir_src + "\\" + folder + "\\" + path, dir_src + "\\" + folder + "\\" + folder + "_" + str(i) + ".wav")
            i += 1

##
import numpy as np

import scipy
#DIR = 'D:\\PythonScripts\\new_study\\wav_files\\accents\\'
#DIR_mfccs = 'D:\\PythonScripts\\new_study\\mfccs_corpus\\'
#DIR_lpc = "D:\\PythonScripts\\new_study\\lpc_corpus\\"

def specs_dataset(dir_src,dir_dest):
    for accent in os.listdir(dir_src):
        accent_DIR = dir_src + accent + "\\"
        for word in os.listdir(accent_DIR):
            word_DIR = accent_DIR + word + "\\"
            for file in os.listdir(word_DIR):
                filepath = word_DIR + file + "\\"
                graph_spectrogram(filepath,dir_dest + accent + "\\" + word + "\\" + file + ".png")

def mfccs_dataset(dir_src,dir_mfccs):
    for accent in os.listdir(dir_src):
        accent_DIR = dir_src + accent + "\\"
        for word in os.listdir(accent_DIR):
            word_DIR = accent_DIR + word + "\\"
            for file in os.listdir(word_DIR):
                file_path = word_DIR + file + "\\"
                x, sr = librosa.load(file_path)
                mfccs = librosa.feature.mfcc(x, sr=sr)
                librosa.display.specshow(mfccs, sr=sr, x_axis='time')
                plt.show()
                plt.savefig(dir_mfccs + accent + "\\" + word + "\\" + file + ".png")
                plt.close('all')

def lpc_dataset(dir_src,dir_lpcs):
    for accent in os.listdir(dir_src):
        accent_DIR = dir_src + accent + "\\"
        for word in os.listdir(accent_DIR):
            word_DIR = accent_DIR + word + "\\"
            for file in os.listdir(word_DIR):
                file_path = word_DIR + file + "\\"
                y, sr = librosa.load(file_path, duration=0.020)
                a = librosa.lpc(y, 2)
                b = np.hstack([[0], -1 * a[1:]])
                y_hat = scipy.signal.lfilter(b, [1], y)
                fig, ax = plt.subplots()
                ax.plot(y)
                ax.plot(y_hat, linestyle='--')
                ax.legend(['y', 'y_hat'])
                ax.set_title('LP Model Forward Prediction')
                plt.show()
                plt.savefig(dir_lpcs + accent + "\\" + word + "\\" + file + ".png")
                plt.close('all')

#file_rename(dir_src)
#dataset(dir_src,dir_dest)
#folders_creation(dir_dest)
#png_copy(dir_dest)

#mfccs_dataset()
#lpc_dataset()
##
import librosa
import librosa.display
import matplotlib.pyplot as plt

##
from shutil import copyfile

for accent in os.listdir(DIR_lpc):
    accent_dir = DIR_lpc + accent + "\\"
    for word in os.listdir(accent_dir):
        word_dir = accent_dir + word + "\\"
        for file in os.listdir(word_dir):
            file_path = word_dir + file
            copyfile(file_path, 'D:\\PythonScripts\\new_study\\lpcs_accents\\' + accent + "\\" + file[:len(file)-4] + "_" + word + ".png")

##
import os
my_file = 'D:\\PythonScripts\\new_study\\wav_files\\accents\\arabic\\bring\\arabic1_bring_bring_bring.png'
base = os.path.splitext(my_file)[0]
os.rename(my_file, base + '.wav')

##
DIR = 'D:\\PythonScripts\\new_study\\wav_files\\accents\\'
for accent in os.listdir(DIR):
    DIR_ACCENT = DIR + accent + '\\'
    for word in os.listdir(DIR_ACCENT):
        DIR_WORD = DIR_ACCENT + word + "\\"
        for file in os.listdir(DIR_WORD):
            file_path = DIR_WORD + file
            new_path = '_'.join(file.split('_')[:2])
            os.rename(file_path, DIR_WORD + new_path + '.wav')

##
DIR = 'D:\\PythonScripts\\new_study\\wav_files\\accents\\arabic\\ask\\'
for i in range(len(os.listdir(DIR))):
    name_file = 'arabic' + str(i+1) + '_ask.wav'
    os.rename(DIR + os.listdir(DIR)[i], DIR + name_file)

##

import wave, os

dir = 'D:\\PythonScripts\\new_study\\wav_files\\accents\\original_samples\\'

for accent in os.listdir(dir):
    ACCENT_DIR = dir + accent + "\\"
    for word in os.listdir(ACCENT_DIR):
        WORD_DIR = ACCENT_DIR + word + "\\"
        for file in os.listdir(WORD_DIR):
            channels = 1
            swidth = 2
            multiplier = 0.8

            count = multiplier

            while count < 1:

                spf = wave.open(WORD_DIR + file, 'rb')
                fr=spf.getframerate() # frame rate
                signal = spf.readframes(-1)

                wf = wave.open('D:\\PythonScripts\\new_study\\wav_files\\accents\\speed_modified_samples\\' + accent + "\\" + word + "\\" + file[:len(file)-4] + '_slowed' + str(count) + '.wav', 'wb')
                wf.setnchannels(channels)
                wf.setsampwidth(swidth)
                wf.setframerate(fr*count)
                wf.writeframes(signal)
                wf.close()

                count += 0.01

        for file in os.listdir(WORD_DIR):
            channels = 1
            swidth = 2
            multiplier = 1.01

            count = multiplier

            while count < 1.2:

                spf = wave.open(WORD_DIR + file, 'rb')
                fr=spf.getframerate() # frame rate
                signal = spf.readframes(-1)

                wf = wave.open('D:\\PythonScripts\\new_study\\wav_files\\accents\\speed_modified_samples\\' + accent + "\\" + word + "\\" + file[:len(file)-4] + '_accelerated' + str(count) + '.wav', 'wb')
                wf.setnchannels(channels)
                wf.setsampwidth(swidth)
                wf.setframerate(fr*count)
                wf.writeframes(signal)
                wf.close()

                count += 0.01

##
import wave

def get_slowed(dir):

    for word in os.listdir(dir):
        WORD_DIR = dir + word + "\\"
        for file in os.listdir(WORD_DIR):
            channels = 1
            swidth = 2
            multiplier = 0.8

            count = multiplier

            while count < 1.0:

                spf = wave.open(WORD_DIR + file, 'rb')
                fr=spf.getframerate() # frame rate
                signal = spf.readframes(-1)

                wf = wave.open(dir + word + "\\" + file[:len(file)-4] + '_slowed' + str(count) + '.wav', 'wb')
                wf.setnchannels(channels)
                wf.setsampwidth(swidth)
                wf.setframerate(fr*count)
                wf.writeframes(signal)
                wf.close()

                count += 0.05






