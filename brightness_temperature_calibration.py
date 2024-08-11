import logging
import re
import sys
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from astropy.io import fits
from tqdm import tqdm

from analise_utils import (ArrayOperations, Monitoring, MplFunction,
                            OsOperations, ZirinTb)
from config import *


def brightness_temperature_calibration(mode, folder_mode, postfix = 'calibrated_brightness', number_of_mode_values = 7, name_of_file = None):

    zirin = ZirinTb()
    MplFunction.set_mpl_rc()
    Monitoring.start_log('logs')
    logging.info(f'Start of the brightness temperature calibration program')
    logging.info(f'Path to files: {directory}')

    files, freqs = OsOperations.freq_sorted_files_in_folder(directory) if folder_mode == 'one_folder' else OsOperations.freq_sorted_1st_two_files_in_folders(directory)

    logging.info(f'Find {len(files)} files')
    logging.info(f'List files: \n {files}')

    OsOperations.create_place(directory, 'calibrated_brightness')
    if folder_mode == 'folder_with_folders':
        for freq in tqdm(freqs, desc='Создание папок для частот'):
            OsOperations.create_place(f'{directory}_{postfix}/{freq}')

    if mode == 'saved_settings':
        try:
            correction_factor_brightness_array = ArrayOperations.read_from_json(name_of_file)
            if len(files)/2 != len(correction_factor_brightness_array):
                Monitoring.logprint('Ошибка! Количество коррекционных коэффициентов не совпадает с количеством частот')
                sys.exit()
        except TypeError as err:
            Monitoring.logprint(f'\nОшибка! Выбран режим `saved_settings`, но не указан файл настроек в аргументах функции!\nПодробнее: {err}')
            sys.exit()
    elif mode == 'calculation':
        correction_factor_brightness_array = []
    else:
        Monitoring.logprint('Ошибка, неверный параметр `mode`')

    for index, image in enumerate(tqdm(files, desc='Общий прогресс выполнения')):

        if index % 2 == 0:

            data1 = fits.open(f'{directory}/{freqs[index//2] if folder_mode == "folder_with_folders" else ""}/{files[index]}', ignore_missing_simple=True)
            data2 = fits.open(f'{directory}/{freqs[index//2] if folder_mode == "folder_with_folders" else ""}/{files[index + 1]}', ignore_missing_simple=True)
            header1 = data1[0].header
            header2 = data2[0].header
            dateandtime = f'{header1['DATE-OBS']}' # T{header1['T-OBS']}
            img1 = data1[0].data
            img2 = data2[0].data
            data1.close()
            data2.close()

            if mode == 'calculation':

                current_frequency = int(re.search(r'(?<=[_.])\d{4,5}(?=[_.])', str(files[index])).group())
                Tb = zirin.getTbAtFrequency(current_frequency/1000)

                data_inside_circle1 = ArrayOperations.cut_sun_disk_data(img1)
                data_inside_circle2 = ArrayOperations.cut_sun_disk_data(img2)
                data_inside_circle1 = np.round(data_inside_circle1, -2)
                data_inside_circle2 = np.round(data_inside_circle2, -2)

                counter1 = Counter(data_inside_circle1)
                counter2 = Counter(data_inside_circle2)
                most_common_for_1 = counter1.most_common(number_of_mode_values)
                most_common_for_2 = counter2.most_common(number_of_mode_values)
                most_common_values_1 = [item[0] for item in most_common_for_1]
                most_common_values_2 = [item[0] for item in most_common_for_2]
                count_values_1 = [item[1] for item in most_common_for_1]
                count_values_2 = [item[1] for item in most_common_for_2]
                mode1 = np.mean(most_common_values_1)
                mode2 = np.mean(most_common_values_2)
                count_values_1 = np.sum(count_values_1)
                count_values_2 = np.sum(count_values_2)

                logging.info(f"For image {index+1} mode intensity: {str(mode1)} [standard:{int(Tb*1000)}], count: {count_values_1}")
                logging.info(f"For image {index+2} mode intensity: {str(mode2)} [standard:{int(Tb*1000)}], count: {count_values_2}")

                correction_factor_brightness = 1 / (((mode1 + mode2)/2) / (Tb * 1000))
                correction_factor_brightness_array.append(correction_factor_brightness)

                logging.info(f"For image {index + 1}, {index + 2} correction factor: {str(correction_factor_brightness)}")

            if folder_mode == 'one_folder':
                fits.writeto(f'{directory}_calibrated_brightness/{files[index][:-4]}_calibrated_brightness.fits', img1 * correction_factor_brightness_array[index//2], overwrite=True, header=header1)
                logging.info(f"Image {index+1}: {files[index]} - saved")
                fits.writeto(f'{directory}_calibrated_brightness/{files[index + 1][:-4]}_calibrated_brightness.fits', img2 * correction_factor_brightness_array[index//2], overwrite=True, header=header2)
                logging.info(f"Image {index+2}: {files[index + 1]} - saved")

            elif folder_mode == 'folder_with_folders':
                all_files_in_freq, freq = OsOperations.freq_sorted_files_in_folder(f'{directory}/{freqs[index//2]}')
                for file in tqdm(all_files_in_freq, desc=f'Обработка файлов частоты {freqs[index//2]}', leave=False):
                    hdul1 = fits.open(f'{directory}/{freqs[index//2]}/{file}', ignore_missing_simple=True)
                    img1 = hdul1[0].data
                    header1 = hdul1[0].header
                    hdul1.close()
                    fits.writeto(f'{directory}_{postfix}/{freqs[index//2]}/{file[:-4] if file[-1]== "t" else file[:-5]}_{postfix}.fits', img1 * correction_factor_brightness_array[index//2], overwrite=True, header=header1)

    ArrayOperations.save_on_json(correction_factor_brightness_array, f'BC_{dateandtime}')

    fig = plt.figure()
    ax = fig.gca()
    plt.plot(correction_factor_brightness_array)
    plt.grid()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.show()

if __name__ == "__main__":
    brightness_temperature_calibration(mode = mode, folder_mode = folder_mode, postfix = 'calibrated_brightness', number_of_mode_values = 7, name_of_file = 'BC_20220113.json')