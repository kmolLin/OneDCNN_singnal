# -*- coding: utf-8 -*-
from pprint import pprint
from scipy import signal
from load import Features, DataProcess, normalizee, intter1d, inter11d

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy
import csv
import glob

data_path = "20200925_steel"
ax = ["ax1", "ay1", "az1", "ax2", "ay2", "az2"]

def draw_signal(features, dictt, slectmethod, show=False):

    for title, plot_name in slectmethod.items():
        if show is True:
            fig = plt.figure()
            plt.title(f"{title}")
        for i in range(len(plot_name)):
            data = np.array((features.unzip_data(f"{title}")[i][:70]))
            normm = np.sqrt(np.sum(data * data))
            dictt[f"{title}_{plot_name[i]}"] = normalizee(features.unzip_data(f"{title}")[i])
            if show is True:
                plt.plot(normalizee(features.unzip_data(f"{title}")[i]), label=f"{plot_name[i]}")
        if show is True:
            plt.legend()
    if show is True:
        plt.show()


def fft_data(data, sampling_time):
    fx = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data), d=sampling_time)
    return (freqs, fx)


def find_freq_cutter(freq, fx, high_freq, low_freq, twice=3):
    """let data inverse"""
    ax = np.zeros_like(fx)
    for i in range(twice):
        high = np.where((high_freq * i < freq) & (freq < low_freq * i))  # freqs == 46.0
        invers = np.where((freq > -1 * low_freq * i) & (freq < -1 * high_freq * i))
        # low = np.where(freq > low_freq)  # freqs == 46.0
        # invers_high = np.where(freq == -1 * high_freq)
        # inverse_low = np.where(freq == -1 * low_freq)
        for j in high:
            ax[j] = fx[j]
        for j in invers:
            ax[j] = fx[j]
    return np.fft.ifft(ax)


def find_freq_max(freq, fx, high_freq, low_freq):
    high = np.where(freq == high_freq)[0]  # freqs == 46.0
    low = np.where(freq == low_freq)[0]  # freqs == 46.0
    print(np.max(fx[high[0]:low[0]]))


if __name__ == '__main__':

    sample_time = 1.0 / 6400.0
    names = sorted(os.listdir(data_path))
    b, a = signal.butter(8, [0.03125], 'lowpass')
    b1, a1 = signal.butter(8, [0.00625], 'highpass')

    big_formate = []
    txx_all = []

    for im, t in enumerate(names):
        print(t)
        # if im < 10:
        #     continue
        # data_name = sorted(os.listdir(f"{data_path}/{t}"))
        all_files = sorted(glob.glob(f"{data_path}/{t}/*.csv"), key=os.path.getmtime)

        list_name_tmp = []
        for i in range(1, len(all_files) + 1):
            list_name_tmp.append(f"{data_path}/{t}/1_{i}.csv")

        tmp = []
        txx = []
        for i in range(len(ax)):
            txx.append([])
            tmp.append([])
        for name in list_name_tmp:
            df = pd.read_csv(f"{name}", header=None)

            for i in range(6):
                if len(tmp[i]) > 0:
                    tmp[i][0] = np.concatenate((tmp[i][0], df[i][1:].astype('float').to_numpy()))
                    # tmp[i][0] = signal.filtfilt(b, a, tmp[i][0])
                    # tmp[i][0] = signal.filtfilt(b1, a1, tmp[i][0])
                else:
                    # filtedData = signal.filtfilt(b, a, df[i][1:].to_numpy())
                    # filtedData = signal.filtfilt(b1, a1, filtedData)
                    tmp[i].append(df[i][1:].astype('float').to_numpy())

        # big_formate.append(tmp)
        # fig, axs = plt.subplots()  # (sharex=True, sharey=True)
        # fig.subplots_adjust(hspace=1.5, wspace=0.8)

        # with open(f"20200925_steel_merge/{t}.csv", 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     for i in range(len(tmp[0][0])):
        #         tss = [tmp[0][0][i], tmp[1][0][i], tmp[2][0][i], tmp[3][0][i], tmp[4][0][i], tmp[5][0][i]]
        #         writer.writerow(tss)
        #     #     print(tss)
        #     # writer = csv.writer(tss)

        for i in range(len(ax)):
            # rows = csv.reader()
            plt.subplot(6, 1, i + 1)  # , sharex=axs
            plt.title(f"{ax[i]}")
            freq, fx = fft_data(tmp[i][0], sample_time)
            filit_data = (find_freq_cutter(freq, fx, 45.0, 47.0, 3))
            plt.xlabel("Times (s)")
            plt.ylabel("Amplitude (V)")
            # freq, fx = fft_data(filit_data, sample_time)
            # txx[i].append(np.max(np.real(filit_data)) - np.min(np.real(filit_data)))
            # if i == 4:
            #     print(np.real(np.max(filit_data) - np.min(filit_data)))
            # find_freq_max(freq, abs(fx), 21.0, 24.0)
            # plt.plot(freq, abs(fx))
            # plt.xlim(0, 100)
            # plt.ylim(0, 1000)
            # plt.plot(tmp[i][0])

            plt.plot([c * (1 / 6400) for c in range(len(tmp[i][0]))], np.real(filit_data))  # np.real(filit_data))
        plt.show()

    exit()
    print(len(big_formate))

    acc = {"ax1": ["rms", "std", "fft_data"],
           "ay1": ["rms", "std", "fft_data"],
           "az1": ["rms", "std", "fft_data"],
           "ax2": ["rms", "std", "fft_data"],
           "ay2": ["rms", "std", "fft_data"],
           "az2": ["rms", "std", "fft_data"]
           }
    acc = {"ax1": ["fft_data"],
           "ay1": ["fft_data"],
           "az1": ["fft_data"],
           "ax2": ["fft_data"],
           "ay2": ["fft_data"],
           "az2": ["fft_data"]
           }
    load_data_formate = [v for v in acc.keys()]
    features = Features(load_data_formate, acc)

    dictt = {}

    for i, name in enumerate(big_formate):
        # if i == 50:
        # continue
        # print(type(name[0][0]))
        # exit()
        if i == 39:
            continue
        p = DataProcess(features, name, load_data_formate, specific_data=True)
        p.get_allfeature(117, acc)

    draw_signal(features, dictt, acc, True)

