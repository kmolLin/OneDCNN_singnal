
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

data_path = "20200925_steel_merge"
ax = ["ax1", "ay1", "az1", "ax2", "ay2", "az2"]

if __name__ == '__main__':

    sample_time = 1.0 / 6400.0
    names = sorted(os.listdir(data_path))
    all_files = sorted(glob.glob(f"{data_path}/*.csv"))
    for name in all_files:
        print(name)
        df = pd.read_csv(f"{name}", header=None)
        aa = df[0][1:].astype('float').to_numpy()

        for i in range(6):
            # rows = csv.reader()
            plt.subplot(6, 1, i + 1)  # , sharex=axs
            plt.title(f"{ax[i]}")
            # filit_data = (find_freq_cutter(freq, fx, 45.0, 47.0, 3))
            plt.xlabel("Times (s)")
            plt.ylabel("Amplitude (V)")

            plt.plot([c * (1 / 6400) for c in range(len(aa))], df[i][1:].astype('float').to_numpy())  # np.real(filit_data))
        plt.show()

