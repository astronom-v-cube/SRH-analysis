from astropy.io import fits
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
import os
import re
import matplotlib.patches as patches
import logging
from analise_utils import Monitoring, Extract, OsOperations, MplFunction
from config import directory

Extract = Extract()
MplFunction.set_mpl_rc()
Monitoring.start_log('logs')
logging.info(f'Start program show time profile on difference frequency')

# Каталог из которого будем брать файлы
folder_mode = 'folder_with_folders'    # 'one_folder' /one_time_moment/ or 'folder_with_folders' /many__time_moment/
### TO DO
mode = 'from_box'     # 'from_point' or 'from_box'
point = [395, 881]
stroka_1, stroka_2, stolbec_1, stolbec_2 = 360, 420, 870, 905

files, freqs = OsOperations.freq_sorted_1st_two_files_in_folders(directory)
colors = plt.cm.jet(np.linspace(0, 1, len(freqs)))

# logging.info(f'Path to files: {directory}')
# logging.info(f'Search {len(files)} files')
# logging.info(f'Files: \n {files}')

def multiple_crope_images_display(freqs):
    list_of_tb_value = []
    list_of_time = []

    for freq in tqdm(freqs):
        files_on_freq_folder = OsOperations.abс_sorted_files_in_folder(f'{directory}/{freq}')
        list_of_tb_value_on_freq = []
        list_of_time_obs_on_freq = []
        for index in range(0, len(files_on_freq_folder), 2):
            hdul1 = fits.open(f'{directory}/{freq}/{files_on_freq_folder[index]}', ignore_missing_simple=True)
            hdul2 = fits.open(f'{directory}/{freq}/{files_on_freq_folder[index+1]}', ignore_missing_simple=True)
            time_obs = hdul1[0].header['T-OBS']
            list_of_time_obs_on_freq.append(time_obs)
            data1 = hdul1[0].data
            data2 = hdul2[0].data
            I = data1 + data2
            if mode == 'from_box':
                image_slice = I[stroka_1:stroka_2, stolbec_1:stolbec_2]
                image_slice = image_slice.sum()
                list_of_tb_value_on_freq.append(image_slice)
            elif mode == 'from_point':
                list_of_tb_value_on_freq.append(I[point[0], point[1]])

        list_of_tb_value.append(list_of_tb_value_on_freq)
        list_of_time.append(list_of_time_obs_on_freq)

    list_of_time = [[datetime.strptime(time, '%H:%M:%S') for time in sublist] for sublist in list_of_time]

    fig_I = plt.figure(num="Intensity time profile", figsize=(10, 7))
    fig_V = plt.figure(num="Stocks V time profile", figsize=(10, 7))
    I_ax = fig_I.gca()
    V_ax = fig_V.gca()

    for index, time_profile in enumerate(list_of_tb_value):
        I_ax.plot(list_of_time[index], time_profile, label=f'{freqs[index]}', color=colors[index])
        I_ax.legend(bbox_to_anchor=(1.3, 1), loc="upper right")
    I_ax.set_yscale('log')
    fig_I.tight_layout()

    plt.show()


multiple_crope_images_display(freqs)