import os
import numpy as np
import scipy
from scipy.io import wavfile
import scipy.fftpack as fft
from scipy.signal import get_window
import IPython.display as ipd
import matplotlib.pyplot as plt

import emd
from scipy import ndimage
import random
import wave
import contextlib
import pylab

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def graph_spectrogram(wav_file,name_dest):

    def get_wav_info(wav_file):
        wav = wave.open(wav_file, 'r')
        frames = wav.readframes(-1)
        sound_info = pylab.fromstring(frames, 'int16')
        frame_rate = wav.getframerate()
        wav.close()
        return sound_info, frame_rate

    sound_info, frame_rate = get_wav_info(wav_file)
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig(name_dest)
    pylab.close()

def mfcc(filepath, destinationpath, mode=['coeffs','image']):

    sample_rate, audio = wavfile.read(filepath)

    def normalize_audio(audio):
        audio = audio / np.max(np.abs(audio))
        return audio

    audio = normalize_audio(audio)

    def frame_audio(audio, FFT_size=1024, hop_size=10, sample_rate=44100): # we use 1024 for FFT size and 44,1 kHz in order to have 1,024 samples = 0.022s of sound
        # hop_size in ms

        audio = np.pad(audio, int(FFT_size / 2), mode='reflect')
        frame_len = np.round(sample_rate * hop_size / 1000).astype(int)
        frame_num = int((len(audio) - FFT_size) / frame_len) + 1
        frames = np.zeros((frame_num,FFT_size))

        for n in range(frame_num):
            frames[n] = audio[n*frame_len:n*frame_len+FFT_size]

        return frames

    hop_size = 20 #ms
    FFT_size = 1024

    audio_framed = frame_audio(audio, FFT_size=FFT_size, hop_size=hop_size, sample_rate=sample_rate)
    window = get_window("hann", FFT_size, fftbins=True)
    audio_win = audio_framed * window
    audio_winT = np.transpose(audio_win)
    audio_fft = np.empty((int(1 + FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')
    for n in range(audio_fft.shape[1]):
        audio_fft[:, n] = fft.fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]
    audio_fft = np.transpose(audio_fft)
    audio_power = np.square(np.abs(audio_fft))
    freq_min = 0
    freq_high = sample_rate / 2
    mel_filter_num = 24

    def freq_to_mel(freq):
        return 2595.0 * np.log10(1.0 + freq / 700.0)

    def met_to_freq(mels):
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=44100):
        fmin_mel = freq_to_mel(fmin)
        fmax_mel = freq_to_mel(fmax)

        mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
        freqs = met_to_freq(mels)

        return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs

    filter_points, mel_freqs = get_filter_points(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate=44100)

    def get_filters(filter_points, FFT_size):
        filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1)))

        for n in range(len(filter_points)-2):
            filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
            filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])

        return filters

    filters = get_filters(filter_points, FFT_size)
    audio_filtered = np.dot(filters, np.transpose(audio_power))
    audio_log = 10.0 * np.log10(audio_filtered)

    def dct(dct_filter_num, filter_len):
        basis = np.empty((dct_filter_num,filter_len))
        basis[0, :] = 1.0 / np.sqrt(filter_len)

        samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

        for i in range(1, dct_filter_num):
            basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)

        return basis

    dct_filter_num = 24
    dct_filters = dct(dct_filter_num, mel_filter_num)
    cepstral_coefficients = np.dot(dct_filters, audio_log)

    if mode == 'coeffs':
        return cepstral_coefficients

    else:
        plt.figure(figsize=(15,5))
        plt.plot(np.linspace(0, len(audio) / sample_rate, num=len(audio)), audio)
        plt.imshow(cepstral_coefficients, aspect='auto', origin='lower')
        plt.savefig(destinationpath)
        plt.close('all')

def hilbert_spectrum(filepath,destination_path):
    sample_rate, audio = wavfile.read(filepath)
    duration = audio.shape[0]/sample_rate

    imf = emd.sift.sift(audio)
    IP, IF, IA = emd.spectra.frequency_transform(imf, sample_rate, 'nht')
    freq_edges, freq_centres = emd.spectra.define_hist_bins(1, 50, 8, 'log')

    freq_edges, freq_centres = emd.spectra.define_hist_bins(0, 100, 128, 'linear')

    # Amplitude weighted HHT
    spec_weighted = emd.spectra.hilberthuang_1d(IF, IA, freq_edges)

    # Unweighted HHT - we replace the instantaneous amplitude values with ones
    spec_unweighted = emd.spectra.hilberthuang_1d(IF, np.ones_like(IA), freq_edges)

    # Carrier frequency histogram definition
    freq_edges, freq_centres = emd.spectra.define_hist_bins(0, 5000, 256, 'linear')

    hht = emd.spectra.hilberthuang(IF[:, 3], IA[:, 3], freq_edges, mode='amplitude')
    time_centres = np.arange(2051)-.5

    # Apply smoothing
    hht = ndimage.gaussian_filter(hht, .7)

    ax = emd.plotting.plot_hilberthuang(hht, time_centres, freq_centres,
                                cmap='viridis', time_lims=(750, 1500),  log_y=True)
    plt.show()
    ax.figure.savefig(destination_path)
    plt.close('all')


def fourier_centroid(filepath):

    sample_rate, audio = wavfile.read(filepath)

    def normalize_audio(audio):
        audio = audio / np.max(np.abs(audio))
        return audio

    audio = normalize_audio(audio)

    def frame_audio(audio, FFT_size=1024, hop_size=10, sample_rate=44100): # we use 1024 for FFT size and 44,1 kHz in order to have 1,024 samples = 0.022s of sound
        # hop_size in ms

        audio = np.pad(audio, int(FFT_size / 2), mode='reflect')
        frame_len = np.round(sample_rate * hop_size / 1000).astype(int)
        frame_num = int((len(audio) - FFT_size) / frame_len) + 1
        frames = np.zeros((frame_num,FFT_size))

        for n in range(frame_num):
            frames[n] = audio[n*frame_len:n*frame_len+FFT_size]

        return frames

    hop_size = 20 #ms
    FFT_size = 1024

    audio_framed = frame_audio(audio, FFT_size=FFT_size, hop_size=hop_size, sample_rate=sample_rate)
    window = get_window("hann", FFT_size, fftbins=True)
    audio_win = audio_framed * window
    audio_winT = np.transpose(audio_win)
    audio_fft = np.empty((int(1 + FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')
    for n in range(audio_fft.shape[1]):
        audio_fft[:, n] = fft.fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]
    audio_fft = np.transpose(audio_fft)
    audio_power = np.square(np.abs(audio_fft))
    freq_min = 0
    freq_high = sample_rate / 2
    mel_filter_num = 24

    def freq_to_mel(freq):
        return 2595.0 * np.log10(1.0 + freq / 700.0)

    def met_to_freq(mels):
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=44100):
        fmin_mel = freq_to_mel(fmin)
        fmax_mel = freq_to_mel(fmax)

        mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
        freqs = met_to_freq(mels)

        return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs

    filter_points, mel_freqs = get_filter_points(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate=44100)

    def get_filters(filter_points, FFT_size):
        filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1)))

        for n in range(len(filter_points)-2):
            filters[n, filter_points[n] : filter_points[n + 1]] = [0] + [1] * (filter_points[n + 1] - filter_points[n] - 1)
            filters[n, filter_points[n + 1] : filter_points[n + 2]] = [1] * (filter_points[n + 2] - filter_points[n + 1] - 1) + [0]

        return filters

    filters = get_filters(filter_points, FFT_size)

    fourier_centroid = np.zeros(filters.shape)
    for i in range(filters.shape[0]):
        fourier_centroid[i] = [(filters[i][j]*j)/sum(filters[i]) for j in range(len(filters[i]))]

    return fourier_centroid

import numpy
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt

def instant_centroid(filepath):

    sample_rate, signal = scipy.io.wavfile.read(filepath)
    pre_emphasis=0.97
    emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    frame_size = 0.020
    frame_stride = 0.01

    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = numpy.zeros((pad_signal_length - signal_length))
    pad_signal = numpy.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]

    frames *= numpy.hamming(frame_length)

    NFFT = 1024

    mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    nfilt = 24

    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz

    from scipy import signal
    import numpy as np

    filters = np.zeros((len(hz_points)-2,int(NFFT/2+1)))


    for i in range(len(hz_points)-2):
        numtaps = filters.shape[1]-filters.shape[0]
        f1 = hz_points[i]+0.001
        f2 = hz_points[i+1]
        coeffs = [0] * i + list(signal.firwin(numtaps, [f1, f2], fs=sample_rate+1, pass_zero=False)) + [0] * (int(NFFT/2+1) - i - numtaps)
        filters[i] = coeffs

    pow_filtered = np.dot(filters, np.transpose(pow_frames))

    from scipy.signal import hilbert

    analytic_signal = hilbert(pow_filtered)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * sample_rate)

    inst_freq_centroid = np.zeros(instantaneous_frequency.shape)

    for i in range(len(inst_freq_centroid)):
        inst_freq_centroid[i] = [(amplitude_envelope[i,j]/sum(amplitude_envelope[i]))*instantaneous_frequency[i,j] for j in range(instantaneous_frequency.shape[1])]

    return inst_freq_centroid

def features(filepath, destination_file):

    from PIL import Image, ImageTk

    def get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    def get_concat_v(im1, im2):
        dst = Image.new('RGB', (im1.width, im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        return dst

    import cv2
    import numpy as np

    # Read images in one single 4D array; resize to [200, 200]

    mfccs = mfcc(filepath,'none','coeffs')
    fourier = fourier_centroid(filepath)
    instant = instant_centroid(filepath)

    img1 = plt.imshow(mfccs)
    image1 = Image.fromarray(np.uint8(img1.get_cmap()(img1.get_array())*255))
    image1 = image1.resize((200,200))

    img2 = plt.imshow(fourier)
    image2 = Image.fromarray(np.uint8(img2.get_cmap()(img2.get_array())*255))
    image2 = image2.resize((200,200))

    img3 = plt.imshow(instant)
    image3 = Image.fromarray(np.uint8(img3.get_cmap()(img3.get_array())*255))
    image3 = image3.resize((200,200))

    get_concat_h(get_concat_h(image1, image2), image3).save(destination_file)


def all_features(filepath,destination_file):
    mfccs = mfcc(filepath,'none','coeffs')
    fourier = fourier_centroid(filepath)
    instant = instant_centroid(filepath)

    list_shapes0 = [x.shape[0] for x in [mfccs,fourier,instant]]
    list_shapes1 = [x.shape[1] for x in [mfccs,fourier,instant]]

    max_shape0, max_shape1 = max(list_shapes0), max(list_shapes1)

    new_array_mfccs, new_array_fourier, new_array_instant = np.zeros((max_shape0,max_shape1)),np.zeros((max_shape0,max_shape1)),np.zeros((max_shape0,max_shape1))
    new_array_mfccs[:mfccs.shape[0],:mfccs.shape[1]] = mfccs
    new_array_fourier[:fourier.shape[0],:fourier.shape[1]] = fourier
    new_array_instant[:instant.shape[0],:instant.shape[1]] = instant

    image_mat = np.zeros((max_shape0,max_shape1,3))
    image_mat[:,:,0] = new_array_mfccs
    image_mat[:,:,1] = new_array_fourier
    image_mat[:,:,2] = new_array_instant

    x = image_mat[:,:,0]
    y = image_mat[:,:,1]
    z = image_mat[:,:,2]

    #ax = plt.axes(projection='3d')
   # ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5);
    #ax.figure.savefig(destination_file)
    plt.figure(figsize=(15,5))
    plt.imshow(image_mat, aspect='auto', origin='lower')
    plt.savefig(destination_file)
    plt.close('all')


def mfcc_fourier(filepath,destination_file):

    from PIL import Image, ImageTk

    def get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    def get_concat_v(im1, im2):
        dst = Image.new('RGB', (im1.width, im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        return dst

    import cv2
    import numpy as np

    # Read images in one single 4D array; resize to [200, 200]

    mfccs = mfcc(filepath,'none','coeffs')
    instant = instant_centroid(filepath)

    img1 = plt.imshow(mfccs)
    image1 = Image.fromarray(np.uint8(img1.get_cmap()(img1.get_array())*255))
    image1 = image1.resize((200,200))

    img2 = plt.imshow(instant)
    image2 = Image.fromarray(np.uint8(img2.get_cmap()(img2.get_array())*255))
    image2 = image2.resize((200,200))

    get_concat_h(image1, image2).save(destination_file)

def mfcc_instant(filepath,destination_file):

    from PIL import Image, ImageTk

    def get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    def get_concat_v(im1, im2):
        dst = Image.new('RGB', (im1.width, im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        return dst

    import cv2
    import numpy as np

    # Read images in one single 4D array; resize to [200, 200]

    mfccs = mfcc(filepath,'none','coeffs')
    fourier = fourier_centroid(filepath)

    img1 = plt.imshow(mfccs)
    image1 = Image.fromarray(np.uint8(img1.get_cmap()(img1.get_array())*255))
    image1 = image1.resize((200,200))

    img2 = plt.imshow(fourier)
    image2 = Image.fromarray(np.uint8(img2.get_cmap()(img2.get_array())*255))
    image2 = image2.resize((200,200))

    get_concat_h(image1, image2).save(destination_file)

def fourier_instant(filepath,destination_file):

    from PIL import Image, ImageTk

    def get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    def get_concat_v(im1, im2):
        dst = Image.new('RGB', (im1.width, im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        return dst

    import cv2
    import numpy as np

    # Read images in one single 4D array; resize to [200, 200]

    instant = instant_centroid(filepath)
    fourier = fourier_centroid(filepath)

    img1 = plt.imshow(instant)
    image1 = Image.fromarray(np.uint8(img1.get_cmap()(img1.get_array())*255))
    image1 = image1.resize((200,200))

    img2 = plt.imshow(fourier)
    image2 = Image.fromarray(np.uint8(img2.get_cmap()(img2.get_array())*255))
    image2 = image2.resize((200,200))

    get_concat_h(image1, image2).save(destination_file)

def spec_dataset(dir_src,dir_dest):
    for type in os.listdir(dir_src):
        for word in os.listdir(os.path.join(dir_src, type)):
            word_DIR = os.path.join(dir_src, type, word)
            for file in os.listdir(word_DIR):
                file_path = os.path.join(word_DIR, file)
                if not os.path.exists(os.path.join(dir_dest, type, word, file[:len(file)-4] + ".png")):
                    graph_spectrogram(file_path, os.path.join(dir_dest, type, word, file[:len(file)-4] + ".png"))
                    # print("Spec of : " + file)

def mfcc_dataset(dir_src,dir_dest):
    for type in os.listdir(dir_src):
        for word in os.listdir(os.path.join(dir_src, type)):
            word_DIR = os.path.join(dir_src, type, word)
            for file in os.listdir(word_DIR):
                file_path = os.path.join(word_DIR, file)
                if not os.path.exists(os.path.join(dir_dest, type, word, file[:len(file)-4] + ".png")):
                    mfcc(file_path, os.path.join(dir_dest, type, word, file[:len(file)-4] + ".png"))
                    # print("MFCC of : " + file)

def hilbert_dataset(dir_src,dir_dest):
    for type in os.listdir(dir_src):
        for word in os.listdir(os.path.join(dir_src, type)):
            word_DIR = os.path.join(dir_src, type, word)
            for file in os.listdir(word_DIR):
                file_path = os.path.join(word_DIR, file)
                if not os.path.exists(os.path.join(dir_dest, type, word, file[:len(file)-4] + ".png")):
                    hilbert_spectrum(file_path, os.path.join(dir_dest, type, word, file[:len(file)-4] + ".png"))
                    # print("Hilbert of : " + file)

def features_dataset(dir_src,dir_dest):
    for type in os.listdir(dir_src):
        for word in os.listdir(os.path.join(dir_src, type)):
            word_DIR = os.path.join(dir_src, type, word)
            for file in os.listdir(word_DIR):
                file_path = os.path.join(word_DIR, file)
                if not os.path.exists(os.path.join(dir_dest, type, word, file[:len(file)-4] + ".png")):
                    features(file_path, os.path.join(dir_dest, type, word, file[:len(file)-4] + ".png"))
                    # print("Features of : " + file)

def mfcc_fourier_dataset(dir_src,dir_dest):
    for type in os.listdir(dir_src):
        for word in os.listdir(os.path.join(dir_src, type)):
            word_DIR = os.path.join(dir_src, type, word)
            for file in os.listdir(word_DIR):
                file_path = os.path.join(word_DIR, file)
                if not os.path.exists(os.path.join(dir_dest, type, word, file[:len(file)-4] + ".png")):
                    mfcc_fourier(file_path, os.path.join(dir_dest, type, word, file[:len(file)-4] + ".png"))
                    # print("MFCC + Fourier of : " + file)

def mfcc_instant_dataset(dir_src,dir_dest):
    for type in os.listdir(dir_src):
        for word in os.listdir(os.path.join(dir_src, type)):
            word_DIR = os.path.join(dir_src, type, word)
            for file in os.listdir(word_DIR):
                file_path = os.path.join(word_DIR, file)
                if not os.path.exists(os.path.join(dir_dest, type, word, file[:len(file)-4] + ".png")):
                    mfcc_instant(file_path, os.path.join(dir_dest, type, word, file[:len(file)-4] + ".png"))
                    # print("MFCC + Instant of : " + file)

def fourier_instant_dataset(dir_src,dir_dest):
    for type in os.listdir(dir_src):
        for word in os.listdir(os.path.join(dir_src, type)):
            word_DIR = os.path.join(dir_src, type, word)
            for file in os.listdir(word_DIR):
                file_path = os.path.join(word_DIR, file)
                if not os.path.exists(os.path.join(dir_dest, type, word, file[:len(file)-4] + ".png")):
                    fourier_instant(file_path, os.path.join(dir_dest, type, word, file[:len(file)-4] + ".png"))
                    # print("MFCC + Instant of : " + file)

def all_datasets(dir_src,dir_dest):
    print("Features")
    features_dataset(dir_src, os.path.join(dir_dest, "mfcc+fourier+instant"))
    print("MFCC-Fourier")
    mfcc_fourier_dataset(dir_src, os.path.join(dir_dest, "mfcc+fourier"))
    print("MFCC-Instant")
    mfcc_instant_dataset(dir_src, os.path.join(dir_dest, "mfcc+instant"))
    print("Fourier-Instant")
    fourier_instant_dataset(dir_src, os.path.join(dir_dest, "fourier+instant"))
    print("Spectrograms")
    spec_dataset(dir_src, os.path.join(dir_dest, "spectrograms"))
    print("MFCC")
    mfcc_dataset(dir_src, os.path.join(dir_dest, "mfcc"))
    print("Hilbert")
    hilbert_dataset(dir_src, os.path.join(dir_dest, "hilbert"))

def accent_dataset(dir_src):
    list_noises = ['babble','hf_channel','white_noise']
    list_snr = ['clean','noise_10dB','noise_5dB','noise_0dB','noise_-5dB']
    for accent in os.listdir(dir_src):
        if accent == 'spanish':
            for snr in list_snr:
                if snr == 'clean':
                    print('-------------------------')
                    print("   SNR = ", snr)
                    print('-------------------------')
                    # for noise in list_noises:
                    #     print(noise.upper())
                    #     dir_noise = os.path.join(dir_src, accent, "noise_modified_samples", snr, noise)
                    dir_noise = os.path.join(dir_src, accent, snr)
                    all_datasets(os.path.join(dir_noise, "wavfiles"), dir_noise)
                else:
                    print('-------------------------')
                    print("   SNR = ", snr)
                    print('-------------------------')
                    for noise in list_noises:
                        print(noise.upper())
                        dir_noise = os.path.join(dir_src, accent, "noise_modified_samples", snr, noise)
                        all_datasets(os.path.join(dir_noise, "wavfiles"), dir_noise)


dir2 = "G:\\french\\mfcc+fourier\\"
dir1 = "G:\\Accent - Copie"
dir = "E:\\ACCENTS - Copie\\french\\noise_modified_samples\\noise_5dB\\white_noise\\wavfiles\\french_training\\bags\\french10_bags.wav"

# from PIL import Image
# import numpy as np
#
# w, h = 512, 512
# data = instant_centroid(dir)
# img = Image.fromarray(data, 'RGB')
# img.save('my.png')
# img.show()