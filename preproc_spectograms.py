
OVERLAP = 0.5
WINDOW_SIZE = 30

from scipy import signal
from scipy.fft import fftshift
import numpy as np

from scipy.signal import butter, sosfiltfilt, sosfreqz
rng = np.random.default_rng()

from mne.time_frequency import tfr_array_multitaper

from os import listdir, mkdir
from os.path import isfile, join, exists

"""
Plotting functions of YASA.
"""
import mne
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lspopt import spectrogram_lspopt
from matplotlib.colors import Normalize, ListedColormap
import json


def plot_spectrogram(
    data,
    sf,
    win_sec=30,
    fmin=0.5,
    fmax=25,
    trimperc=2.5,
    cmap="RdBu_r",
    vmin=None,
    vmax=None,
    train=False,
    arrays = False,
    **kwargs,
):
   
    # Increase font size while preserving original
    old_fontsize = plt.rcParams["font.size"]
    plt.rcParams.update({"font.size": 18})

    # Safety checks
    assert isinstance(data, np.ndarray), "Data must be a 1D NumPy array."
    assert isinstance(sf, (int, float)), "sf must be int or float."
    assert data.ndim == 1, "Data must be a 1D (single-channel) NumPy array."
    assert isinstance(win_sec, (int, float)), "win_sec must be int or float."
    assert isinstance(fmin, (int, float)), "fmin must be int or float."
    assert isinstance(fmax, (int, float)), "fmax must be int or float."
    assert fmin < fmax, "fmin must be strictly inferior to fmax."
    assert fmax < sf / 2, "fmax must be less than Nyquist (sf / 2)."
    assert isinstance(vmin, (int, float, type(None))), "vmin must be int, float, or None."
    assert isinstance(vmax, (int, float, type(None))), "vmax must be int, float, or None."
    if vmin is not None:
        assert isinstance(vmax, (int, float)), "vmax must be int or float if vmin is provided"
    if vmax is not None:
        assert isinstance(vmin, (int, float)), "vmin must be int or float if vmax is provided"


    # Calculate multi-taper spectrogram
    nperseg = int(win_sec * sf)
    assert data.size > 2 * nperseg, "Data length must be at least 2 * win_sec."
    
    
    noverlap = int(0.9 * nperseg)
    
    f, t, Sxx = spectrogram_lspopt(data, sf, nperseg=nperseg, noverlap=noverlap)
    Sxx = 10 * np.log10(Sxx)  # Convert uV^2 / Hz --> dB / Hz

    # Select only relevant frequencies (up to 30 Hz)
    good_freqs = np.logical_and(f >= fmin, f <= fmax)
    # if raw is fed, use the Sxx below
    Sxx = Sxx[good_freqs, :]
    f = f[good_freqs]
    #t /= 3600  # Convert t to hours

    # Normalization
    if vmin is None:
        vmin, vmax = np.percentile(Sxx, [0 + trimperc, 100 - trimperc])
    norm = Normalize(vmin=vmin, vmax=vmax)


    fig, ax1 = plt.subplots(nrows=1, figsize=(16, 8))


    # Draw Spectrogram
    im = ax1.pcolormesh(t, f, Sxx, norm=norm, cmap=cmap, antialiased=True, shading="auto")
    #ax1.set_xlim(0, t.max())
    ax1.set_ylabel("Frequency [Hz]")
    ax1.set_xlabel("Time [seconds]")


    # Add colorbar
    cbar = fig.colorbar(im, ax=ax1, shrink=0.95, fraction=0.1, aspect=25)
    cbar.ax.set_ylabel("Log Power (dB / Hz)", rotation=270, labelpad=20)

    # Revert font-size
    plt.rcParams.update({"font.size": old_fontsize})

    if(train and arrays):
        ax1.set_axis_off()
        cbar.remove()

        return fig,Sxx
    elif(arrays):
        return Sxx
    else:
        ...
        #ax1.set_xticks([0,1/12, 2/12, 3/12, 4/12, 5/12, 6/12], labels = np.arange(0, 35, 5))
        


    return fig

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfiltfilt(sos, data)
        return y

def overlapping_windows(sequence, labels,sampling_frequency, window_duration, overlap):
    window_len = sampling_frequency * window_duration
    step_size = (1-overlap) * window_len
    no_windows = int(sequence.shape[0]/step_size)

    sequence_windowed = []
    labels_windowed = []
    for i in range(0,no_windows):
        window_start = int((i)*step_size)
        sequence_windowed.append(sequence[window_start:(window_start+window_len)])
        current_window = []
        for j in range(0,len(labels)):
            # NEEDS TO BE CHANGED. If spindle more than 0.5s then keep
            if (labels[j][0] < (window_start+window_len)/sampling_frequency) and (labels[j][0] > window_start/sampling_frequency):
                current_window.append((labels[j][0]-(window_start/sampling_frequency), 0, labels[j][1]-(window_start/sampling_frequency), 1))
        labels_windowed.append(current_window)
    return sequence_windowed, labels_windowed


Dreams_path = '/home/marius/Documents/THESIS/data/DREAMS'

dir_files = [f for f in listdir(Dreams_path) if isfile(join(Dreams_path, f))]
excerpts = [f for f in dir_files if f[0:2] == 'ex']
ex_txts = [f for f in excerpts if f[-3:] == 'txt']
ex_txts = np.sort(ex_txts)

excerpt_data_list = []
fs_list = [100, 200, 50, 200, 200, 200, 200, 200]

for i, file_name in enumerate(ex_txts):
    excerpt = np.loadtxt(Dreams_path + '/' + file_name, skiprows= 1)
    excerpt = butter_bandpass_filter(excerpt, 0.3, 20, fs_list[i], 2)

    excerpt_data_list.append(excerpt)

dir_files = [f for f in listdir(Dreams_path) if isfile(join(Dreams_path, f))]
scoring = [f for f in dir_files if f[0:3] == 'Vis']
scoring_1 = [f for f in scoring if f[14] == '1']
scoring_1 = np.sort(scoring_1)

scoring_1_list = []

for i, file_name in enumerate(scoring_1):
    scoring = np.loadtxt(Dreams_path + '/' + file_name, skiprows= 1)
    scoring[:,1] = scoring[:,1] + scoring[:,0]
    scoring_1_list.append(scoring)

for i in range(0,len(excerpt_data_list)):
    sequence_windows, label_windows = overlapping_windows(excerpt_data_list[i], scoring_1_list[i], fs_list[i],WINDOW_SIZE, OVERLAP)
    counter = 0
    for j,window in enumerate(sequence_windows):
        if label_windows[j] == []:
            continue
        fig, Sxx = plot_spectrogram(window, fs_list[i], win_sec = 2, fmin = 0.3, fmax = 20,train=True, arrays=True)

        if not exists(Dreams_path + '/windowed'):
            mkdir(Dreams_path + '/windowed')

        if not exists(Dreams_path + '/windowed' + '/images'):
            mkdir(Dreams_path + '/windowed' + '/images')

        if not exists(Dreams_path + '/windowed' + '/images/' + str(i)):
            mkdir(Dreams_path + '/windowed' + '/images/' + str(i))

        if not exists(Dreams_path + '/windowed' + '/real/'):
            mkdir(Dreams_path + '/windowed' + '/real/')

        if not exists(Dreams_path + '/windowed' + '/real/' + str(i)):
            mkdir(Dreams_path + '/windowed' + '/real/' + str(i))

        if not exists(Dreams_path + '/windowed' + '/labels/'):
            mkdir(Dreams_path + '/windowed' + '/labels/')

        if not exists(Dreams_path + '/windowed' + '/labels/' + str(i)):
            mkdir(Dreams_path + '/windowed' + '/labels/' + str(i))


        fig.savefig(Dreams_path + '/windowed' + '/images/' + str(i) + "/" + str(counter) + '.png', bbox_inches='tight')
        print(Dreams_path + '/windowed' + '/real/' + str(i) + "/" + str(counter))
        np.save(Dreams_path + '/windowed' + '/real/' + str(i) + "/" + str(counter) + '.npy', Sxx)
        #np.save(Dreams_path + '/windowed' + '/labels/' + str(i) + "/" + str(j) + '.npy', label_windows)
        with open(Dreams_path + '/windowed' + '/labels/' + str(i) + "/" + str(counter) + '.json', 'w') as fp:
            json.dump({'boxes':label_windows[j], 'labels':[0]*len(label_windows[j])}, fp)

        counter += 1

        plt.close('all')

        