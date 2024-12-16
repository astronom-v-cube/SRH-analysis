import logging
import numpy as np
from astropy.io import fits
from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm
from scipy.ndimage import shift

from analise_utils import ArrayOperations, Extract, Monitoring, OsOperations
from SRH_shift_maps.SRH_ShiftModule import find_min_deviation_with_correlation, find_min_deviation
from alignment_utils import Alignment
from config import *
from conri import create_coordinate_binding_img

Monitoring.start_log('log')
extract = Extract()
alignment = Alignment()
logging.info(f'Start program alignment of the solar disk')


# norm=TwoSlopeNorm(vmin=0, vcenter=vcenter) /// colors.Normalize(vmin=0, vcenter=vcenter))

##########
vcenter = 5000

setting_of_alignes = []
coordinates_of_control_point = []
coordinates_of_control_point = []

##########
logging.info(f'Path to files: {directory}')

files, freqs = OsOperations.freq_sorted_files_in_folder(directory) if folder_mode == 'one_folder' else OsOperations.freq_sorted_1st_two_files_in_folders(directory)
zero_map = create_coordinate_binding_img('temp_fits/srh_20240514T014017_9200_LCP.fit') # сюда по сути первый файл шоб дергать дату время поиска континуума

# Определяем место для сохранения выравненных файлов
OsOperations.create_place(directory, postfix)
if folder_mode == 'folder_with_folders':
    for freq in tqdm(freqs, desc='Создание папок для частот'):
        OsOperations.create_place(f'{directory}_{postfix}/{freq}')

iterable = range(0, len(files), 2)
intensity_maps_list = alignment.create_intensity_list()

logging.info(f'Find {len(files)} files')
logging.info(f'List files: \n {files}')

if aligned_mode == 'saved_settings':
    setting_of_alignes = np.load(f'{file_settings_name}.npy')

elif method == 'CS_aligned' or method == 'OWM_aligned':

    for i in tqdm(intensity_maps_list, leave=False, desc=f'process -> {method}'):

        if method == 'CS_aligned':
            best_delta, minimum = find_min_deviation_with_correlation(zero_map[y_limit[0]:y_limit[1], x_limit[0]:x_limit[1]], i[y_limit[0]:y_limit[1], x_limit[0]:x_limit[1]])

        elif method == 'OWM_aligned':
            best_delta, minimum = find_min_deviation(zero_map[y_limit[0]:y_limit[1], x_limit[0]:x_limit[1]], i[y_limit[0]:y_limit[1], x_limit[0]:x_limit[1]], [25, 25])

        setting_of_alignes.append([round(best_delta[0], 2), round(best_delta[1], 2)])
print('******')
print(setting_of_alignes)
print('******')

def alignment_sun_disk(files : list = files, method : str = 'search_max_in_area', area : int = None):

    # Загрузка первого файла и нахождение координат отличительного признака
    hdul1 = fits.open(f'{directory}/{freqs[0] if folder_mode == "folder_with_folders" else ""}/{files[0]}', ignore_missing_simple=True)
    hdul2 = fits.open(f'{directory}/{freqs[0] if folder_mode == "folder_with_folders" else ""}/{files[1]}', ignore_missing_simple=True)
    img1 = hdul1[0].data
    img2 = hdul2[0].data
    I = img1 + img2
    header1 = hdul1[0].header
    header2 = hdul2[0].header
    hdul1.close()
    hdul2.close()

    if aligned_mode == 'interactive':

        zero_reference_col, zero_reference_row, zero_max_value = ArrayOperations.calculate_weighted_centroid(I, tuple(control_point_1), 0.5, (x_limit, y_limit))
        zero_control_point = (zero_reference_row, zero_reference_col)
        print(zero_control_point)

        coordinates_of_control_point, square_psf = alignment.interactive_choise_AO()

        control_point_1 = (coordinates_of_control_point[0][0], coordinates_of_control_point[0][1])  # координаты признака на первом изображении
        print(coordinates_of_control_point)
        if method == 'WA_aligned':
            reference_col, reference_row, max_value = ArrayOperations.calculate_weighted_centroid(I, tuple(control_point_1), 0.5, (x_limit, y_limit))

        control_point_1 = (reference_row, reference_col)
        coordinates_of_control_point.append(control_point_1)
        logging.info(f"Max - {max_value} in {control_point_1}")

        # except Exception as err:
        #     Monitoring.logprint('The program is terminated due to lack of alignment data')
        #     Monitoring.logprint(err)

# больше не нужно в контексте того что юзается другая привязка
# """ if folder_mode == 'one_folder':
#         # пересохранение первой пары файлов в новую директорию
#         fits.writeto(f'{directory}_{postfix}/{files[0][:-4]}_{postfix}.fits', img1, overwrite=True, header=header1)
#         logging.info(f"Image {1}: {files[0]} - saved")
#         fits.writeto(f'{directory}_{postfix}/{files[1][:-4]}_{postfix}.fits', img2, overwrite=True, header=header2)
#         logging.info(f"Image {2}: {files[1]} - saved") """

    # if folder_mode == 'folder_with_folders':
    #     all_files_in_freq, freq = OsOperations.freq_sorted_files_in_folder(f'{directory}/{freqs[0]}')
    #     for file in tqdm(all_files_in_freq, desc='Копирование файлов привязочной частоты'):
    #         hdul1 = fits.open(f'{directory}/{freq[0]}/{file}', ignore_missing_simple=True)
    #         img1 = hdul1[0].data
    #         header1 = hdul1[0].header
    #         hdul1.close()
    #         fits.writeto(f'{directory}_{postfix}/{freq[0]}/{file[:-4] if file[-1]=="t" else file[:-5]}.fits', img1, overwrite=True, header=header1)
    #         logging.info(f"Image: {file} - saved")

    for i in tqdm(iterable, desc='Общий прогресс выполнения'):

        hdul1 = fits.open(f'{directory}/{freqs[i//2] if folder_mode == "folder_with_folders" else ""}/{files[i]}')
        hdul2 = fits.open(f'{directory}/{freqs[i//2] if folder_mode == "folder_with_folders" else ""}/{files[i + 1]}')
        img1 = hdul1[0].data
        img2 = hdul2[0].data
        header1 = hdul1[0].header
        header2 = hdul2[0].header
        hdul1.close()
        hdul2.close()

        if aligned_mode == 'interactive':

            I = img1 + img2

            try:
                control_point_2 = (coordinates_of_control_point[i][0], coordinates_of_control_point[i][1])  # координаты признака на текущем изображени
            except Exception as err:
                Monitoring.logprint('The program is terminated due to lack of alignment data')
                Monitoring.logprint(err)

            if method == 'WA_aligned':
                reference_col, reference_row, max_value = ArrayOperations.calculate_weighted_centroid(zero_map, tuple(control_point_2), 0.5, (x_limit, y_limit))
                control_point_2 = (reference_row, reference_col)
                coordinates_of_control_point.append(control_point_2)

                setting_of_alignes.append((control_point_1[0] - control_point_2[0], control_point_1[1] - control_point_2[1])) # dx, dy

                logging.info(f"Max - {max_value} in {control_point_2}")

            dx = control_point_1[0] - control_point_2[0]
            dy = control_point_1[1] - control_point_2[1]

            # Нахождение горизонтального и вертикального сдвигов между изображениями на опорной и расчетной частоте """ """

        # if aligned_mode == 'saved_settings':
        #     dx, dy = setting_of_alignes[i-2][0], setting_of_alignes[i-2][1]

        dx, dy = setting_of_alignes[i//2][0], setting_of_alignes[i//2][1]

        if folder_mode == 'one_folder':
            if interpolation:
                img1, img2 = alignment.interpolation_shift(img1, img2, dx, dy)
            else:
                img1, img2 = alignment.roll_shift(img1, img2, dx, dy)
            fits.writeto(f'{directory}_{postfix}/{files[i][:-5] if files[i][-1] == "s" else files[i][:-4]}_{postfix}.fits', img1, overwrite=True, header=header1)
            logging.info(f"Image {i+2}: {files[i]} - saved")
            fits.writeto(f'{directory}_{postfix}/{files[i+1][:-5] if files[i+1][-1] == "s" else files[i+1][:-4]}_{postfix}.fits', img2, overwrite=True, header=header2)
            logging.info(f"Image {i+2}: {files[i+1]} - saved")

        elif folder_mode == 'folder_with_folders':
            all_files_in_freq, freq = OsOperations.freq_sorted_files_in_folder(f'{directory}/{freqs[i//2]}')
            for file in tqdm(all_files_in_freq, desc=f'Обработка файлов частоты {freqs[i//2]}', leave=False):
                hdul1 = fits.open(f'{directory}/{freqs[i//2]}/{file}', ignore_missing_simple=True)
                img = hdul1[0].data
                header1 = hdul1[0].header
                hdul1.close()
                if interpolation:
                    img = alignment.interpolation_shift(img, None, dx, dy)
                else:
                    img = alignment.roll_shift(img, None, dx, dy)

                fits.writeto(f'{directory}_{postfix}/{freqs[i//2]}/{file[:-4] if file[-1]== "t" else file[:-5]}_{postfix}.fits', img, overwrite=True, header=header1)
                logging.info(f"Image: {file} - saved")

    Monitoring.logprint('Finish program alignment of the solar disk')
    print('For more details: read file "logs.log"')

    if saved_settings:

        ArrayOperations.save_on_json(setting_of_alignes, file_settings_name)
        print(setting_of_alignes)

        square_psf_list = sorted(square_psf, reverse=True)
        np.savez(f'psf_square in {file_settings_name}.npz', freqs = freqs.reverse(), psf_square = square_psf_list)

        print(f'Площади ДН: {square_psf_list}')

##############################################################
if __name__ == '__main__':
    alignment_sun_disk(method = method)