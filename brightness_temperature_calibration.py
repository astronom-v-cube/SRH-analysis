import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from astropy.io import fits
import os, sys, re
import shutil
from tqdm import tqdm
import logging
from scipy import constants, stats
import scipy.optimize as opt
from collections import Counter
from analise_utils import ZirinTb, FindIntensity, MplFunction, Monitoring, Extract, ConvertingArrays, OsOperations

MplFunction.set_mpl_rc()
Monitoring.start_log('log')
logging.info(f'Start of the program to search for the brightness temperature of a calm sun')

##########
directory = "A:/14.05.24"
mode = 'calculation'     # 'saved_settings' or 'calculation'
folder_mode = 'folder_with_folders'    # 'one_folder' /one_time_moment/ or 'folder_with_folders' /many__time_moment/
postfix = 'calibrated_brightness'
number_of_mode_values = 7
##########

logging.info(f'Path to files: {directory}')

zirin = ZirinTb()

frequency = 2.4 # GHz
Tb = zirin.getTbAtFrequency(frequency) # получить яркостную температуру на частоте 2.4 GHz
Sfu = zirin.getSfuAtFrequency(frequency) # получить значение яркости на частоте 2.4 GHz в единицах SFU

correction_factor_brightness_array = []

files, freqs = OsOperations.freq_sorted_files_in_folder(directory) if folder_mode == 'one_folder' else OsOperations.freq_sorted_1st_two_files_in_folders(directory)

logging.info(f'Find {len(files)} files')
logging.info(f'List files: \n {files}')

OsOperations.create_place(directory, 'calibrated_brightness')
if folder_mode == 'folder_with_folders':
    for freq in tqdm(freqs, desc='Создание папок для частот'):
        OsOperations.create_place(f'{directory}_{postfix}/{freq}')

for index, image in enumerate(tqdm(files, desc='Общий прогресс выполнения')):

    if index % 2 == 0:

        # Считывание файлов
        data1 = fits.open(f'{directory}/{freqs[index//2] if folder_mode == "folder_with_folders" else ""}/{files[index]}', ignore_missing_simple=True)
        data2 = fits.open(f'{directory}/{freqs[index//2] if folder_mode == "folder_with_folders" else ""}/{files[index + 1]}', ignore_missing_simple=True)
        header1 = data1[0].header
        header2 = data2[0].header
        img1 = data1[0].data
        img2 = data2[0].data
        data1.close()
        data2.close()

        current_frequency = int(re.search(r'(?<=[_.])\d{4,5}(?=[_.])', str(files[index])).group())
        Tb = zirin.getTbAtFrequency(current_frequency/1000) # получить яркостную температуру на частоте

        if mode == 'saved_settings':

            correction_factor_brightness_array = np.load(f'correction_factor_{directory[-15:-7]}.npy')
            correction_factor_brightness = correction_factor_brightness_array[index//2]

            fits.writeto(f'{directory}_calibrated_brightness/{files[index][:-4]}_calibrated_brightness.fits', img1 * correction_factor_brightness, overwrite=True, header=header1)
            logging.info(f"Image {index+1}: {files[index]} - saved")
            fits.writeto(f'{directory}_calibrated_brightness/{files[index + 1][:-4]}_calibrated_brightness.fits', img2 * correction_factor_brightness, overwrite=True, header=header2)
            logging.info(f"Image {index+2}: {files[index + 1]} - saved")

        elif mode == 'calculation':

            # создаем массивы координат пикселей
            x1, y1 = np.indices(img1.shape)
            x2, y2 = np.indices(img2.shape)

            # определяем параметры окружности
            center_x, center_y = 512, 512
            radius = 100

            # вычисляем расстояние от каждой точки до центра окружности
            distance_from_center1 = np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)
            distance_from_center2 = np.sqrt((x2 - center_x)**2 + (y2 - center_y)**2)

            # создаем маску, где значения True соответствуют точкам, входящим в окружность
            mask1 = distance_from_center1 <= radius
            mask2 = distance_from_center2 <= radius

            # вырезаем данные внутри круга
            data_inside_circle1 = img1[mask1]
            data_inside_circle2 = img2[mask2]
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

            logging.info(f"For image {index+1} mode intensivity: {str(mode1)} [standard:{int(Tb*1000)}], count: {count_values_1}")
            logging.info(f"For image {index+2} mode intensivity: {str(mode2)} [standard:{int(Tb*1000)}], count: {count_values_2}")

            correction_factor_brightness = 1 / (((mode1 + mode2)/2) / (Tb * 1000))
            correction_factor_brightness_array.append(correction_factor_brightness)

            logging.info(f"For image {index + 1}, {index + 2} correction factor: {str(correction_factor_brightness)}")

            if folder_mode == 'one_folder':
                fits.writeto(f'{directory}_calibrated_brightness/{files[index][:-4]}_calibrated_brightness.fits', img1 * correction_factor_brightness, overwrite=True, header=header1)
                logging.info(f"Image {index+1}: {files[index]} - saved")
                fits.writeto(f'{directory}_calibrated_brightness/{files[index + 1][:-4]}_calibrated_brightness.fits', img2 * correction_factor_brightness, overwrite=True, header=header2)
                logging.info(f"Image {index+2}: {files[index + 1]} - saved")

            elif folder_mode == 'folder_with_folders':
                all_files_in_freq, freq = OsOperations.freq_sorted_files_in_folder(f'{directory}/{freqs[index//2]}')
                for file in tqdm(all_files_in_freq, desc=f'Обработка файлов частоты {freqs[index//2]}', leave=False):
                    hdul1 = fits.open(f'{directory}/{freqs[index//2]}/{file}', ignore_missing_simple=True)
                    img1 = hdul1[0].data
                    header1 = hdul1[0].header
                    hdul1.close()
                    fits.writeto(f'{directory}_{postfix}/{freqs[index//2]}/{file[:-4] if file[-1]== "t" else file[:-5]}_{postfix}.fits', img1 * correction_factor_brightness, overwrite=True, header=header1)

# np.save(f'correction_factor_{directory[-15:-7]}.npy', correction_factor_brightness_array)

fig = plt.figure()
ax = fig.gca()
plt.plot(correction_factor_brightness_array)
plt.grid()
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
plt.show()