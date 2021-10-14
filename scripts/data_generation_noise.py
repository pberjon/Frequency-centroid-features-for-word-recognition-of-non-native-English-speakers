from scipy.io import wavfile
import numpy as np
import os

def get_noisy(clean_path, noisy_path, desired_SNR):

    sr1, clean_data = wavfile.read(clean_path)
    sr2, noisy_data = wavfile.read(noisy_path)

    noisy_data = noisy_data[:len(clean_data)]

    N = len(clean_data)
    var_s = (1/N)*sum(clean_data**2)
    var_n = (1/N)*sum(noisy_data**2)

    new_var_n = (var_s)/(10**(desired_SNR/10))
    new_noisy_data = (new_var_n/var_n)*noisy_data

    return clean_data + new_noisy_data

def save_noisy(y, destinationpath):
    scaled = np.int16(y/np.max(np.abs(y)) * 32767)
    wavfile.write(destinationpath, 44100, scaled)

def noisy_dir(dir, noisy_path):
    list_snr = [-5,0,5,10]
    for word in os.listdir(dir):
        word_dir = os.path.join(dir, word)
        for file in os.listdir(word_dir):
            filepath = os.path.join(word_dir, file)
            for snr in list_snr:
                y = get_noisy(filepath, noisy_path, snr)
                save_noisy(y, os.path.join(dir, word, file[:len(file)-4] + "_" + str(snr) + "dB" + ".wav"))

def noisy_dataset(dir, noise_dir):
    babble_path = noise_dir + "babble.wav"
    hfchannel_path = noise_dir + "hfchannel.wav"
    white_path = noise_dir + "white.wav"

    for accent in os.listdir(dir):
        accent_dir = os.path.join(dir, accent, "noise_modified_samples")
        if accent == 'arabic':
            for snr in os.listdir(accent_dir):
                snr_dir = os.path.join(accent_dir, snr)
                if snr != "clean":
                    for noise in os.listdir(snr_dir):
                        noise_dir = os.path.join(snr_dir, noise)
                        feature_dir = os.path.join(noise_dir, "wavfiles")
                        type_dir = os.path.join(feature_dir, accent + "_testing_modified")
                        if noise == 'babble':
                            noisy_dir(type_dir, babble_path)
                        elif noise == 'hf_channel':
                            noisy_dir(type_dir, hfchannel_path)
                        else:
                            noisy_dir(type_dir, white_path)
