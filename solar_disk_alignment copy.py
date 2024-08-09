import logging
import os
import re
import shutil
import sys
from os import listdir
from os.path import isfile, join
from sys import platform

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import mplcursors
import numpy as np
from astropy.io import fits
from matplotlib.colors import TwoSlopeNorm
from matplotlib.widgets import Slider
from tqdm import tqdm


def logprint(msg):
    print(msg)
    logging.info(msg)
    
logging.basicConfig(filename = 'logs.log',  filemode='a', level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.info(f'Start program alignment of the solar disk')
# norm=TwoSlopeNorm(vmin=0, vcenter=vcenter)

##########
directory = '/run/media/dmitry/Дмитрий/datasets/16.07.23/temp_for_aligned'
flag = 'IV'
flag = 'RL'
working_mode = 'directory_with_folders' # 'folder' or 'directory_with_folders'
source_directory_path = '/run/media/dmitry/Дмитрий/datasets/16.07.23'
source_directory_path = "E:/datasets/16.07.23"
destination_folder_path = f'{source_directory_path}/temp_for_aligned'
vcenter = 25000
x_limit = (600, 775)
y_limit = (500, 675)
##########

if working_mode == 'directory_with_folders':

    def copy_first_two_files(source_folder, destination_folder):
        
        # Получаем список всех подпапок в исходной папке
        subfolders = [f.path for f in os.scandir(source_folder) if f.is_dir()]
        
        print('Making temp folder and copy files')
        # Проверяем, существует ли папка назначения, и создаем ее, если нет
        try:
            os.makedirs(destination_folder)
        except FileExistsError:
            shutil.rmtree(destination_folder) if platform == 'win32' else os.system(f'rm -rf {destination_folder}')
            os.makedirs(destination_folder)

        for subfolder in tqdm(subfolders):
            # Получаем список файлов в подпапке и сортируем их по имени
            files = sorted([f.path for f in os.scandir(subfolder) if f.is_file()], key=lambda x: x.lower())

            # Копируем первые два файла в новую папку
            for i in range(min(2, len(files))):
                file_to_copy = files[i]
                destination_file = os.path.join(destination_folder, os.path.basename(file_to_copy))
                shutil.copy(file_to_copy, destination_file)
                
    copy_first_two_files(source_directory_path, destination_folder_path)

logging.info(f'Path to files: {directory}')

pattern = re.compile(r'(?<=[_.])\d{4,5}(?=[_.])')

def extract_number(filename):
    match = pattern.search(filename)
    freqs.add(int(match.group()))
    return int(match.group())

freqs = set()

if working_mode == 'folfer':
    files = sorted(os.listdir(directory), key=extract_number)
    logging.info(f'Working mode - files in folder')
    
elif working_mode == 'directory_with_folders':
    files = sorted(os.listdir(destination_folder_path), key=extract_number)
    folders = sorted(os.listdir(source_directory_path))
    logging.info(f'Working mode - folders in directory')

logging.info(f'Find {len(files)} files')
logging.info(f'List files: \n {files}')

freqs = sorted(list(freqs))

# Функция для обработки событий клика мыши
def onclick(event):
    if event.dblclick:
        # Очистка списка координат кликов
        click_coords.clear()
        # Добавление координат клика в список
        click_coords.append((int(event.xdata), int(event.ydata)))
        plt.close()

# Обновление цветовой шкалы и изображения при изменении положения ползунка
def update(val):
    vmin = im.norm.vmin
    vmax = slider.val
    im.set_clim(vmin, vmax)
    fig.canvas.draw_idle()

def find_max_around_point(matrix : np.ndarray, point : tuple, size : int):
    """     
    The function searches for the maximum value in the matrix within a certain size around a given point. 
    The input arguments of the function are the matrix `matrix` (two-dimensional array), the specified point `point` (coordinates) and the size of the search area `size` (integer).
    The function returns a tuple of three values: the row `max_row` and column `max_col` of the maximum value, as well as the value `max_value` of the largest element in the specified area.
    """
    # Get the indices of the smaller region around the point
    row, col = point
    half_size = size // 2
    row_indices = range(max(0, row - half_size), min(matrix.shape[0], row + half_size + 1))
    col_indices = range(max(0, col - half_size), min(matrix.shape[1], col + half_size + 1))
    
    # Get the smaller region from the original matrix
    smaller_matrix = matrix[np.ix_(row_indices, col_indices)]
    
    # Find the maximum value in the smaller region
    max_value = np.max(smaller_matrix)
    
    # Get the coordinates of the maximum value in the original matrix
    max_indices = np.unravel_index(np.argmax(smaller_matrix), smaller_matrix.shape)
    max_row = row_indices[max_indices[0]]
    max_col = col_indices[max_indices[1]]
    
    return (max_row, max_col, max_value)

coordinates_of_control_point = []
coordinates_of_max_point_in_area = []

if flag == 'RL':
    iterable = range(0, len(files)) 
elif flag == 'IV':
    iterable = range(0, len(files), 2)
    
current_directory = directory if working_mode == 'folder' else destination_folder_path
    
for i in iterable:

    if flag == 'RL':
        data = fits.open(f'{current_directory}/{files[i]}', ignore_missing_simple=True)[0].data
        # Создание графика и отображение данных
        fig, ax = plt.subplots(figsize=(9, 9))
        im = ax.imshow(data, origin='lower', cmap='plasma', extent=[0, data.shape[1], 0, data.shape[0]], norm=TwoSlopeNorm(vmin=0, vcenter=vcenter))
        ax.set_xlim(x_limit) if len(x_limit) != 0 else logging.info(f'Limits for X not found')
        ax.set_ylim(y_limit) if len(y_limit) != 0 else logging.info(f'Limits for Y not found')
        mplcursors.cursor() # hover=True
        plt.title(f'{files[i]}')
        # fig.colorbar(im)
        # fig.tight_layout()

    elif flag == 'IV':
        # Считывание файлов
        data1 = fits.open(f'{current_directory}/{files[i]}', ignore_missing_simple=True)[0].data
        data2 = fits.open(f'{current_directory}/{files[i+1]}', ignore_missing_simple=True)[0].data

        # Создание регулярного выражения
        pattern = re.compile(r'(RCP|LCP|R|L)')
        # поиск совпадений в названии первого файла
        match1 = pattern.search(files[i])
        if match1.group() == 'RCP':
            RCP = data1
        elif match1.group() == 'LCP':
            LCP = data1

        # поиск совпадений в названии второго файла
        match2 = pattern.search(files[i+1])
        if match2.group() == 'RCP':
            RCP = data2
        elif match2.group() == 'LCP':
            LCP = data2

        I = RCP+LCP
        V = RCP-LCP

        # Создание графика и отображение данных
        fig, ax = plt.subplots(figsize=(9, 9))
        im = ax.imshow(-V, origin='lower', cmap='plasma', extent=[0, V.shape[1], 0, V.shape[0]], norm=colors.Normalize(vmin=0, vmax=350000))
        ax.set_xlim(x_limit) if len(x_limit) != 0 else logging.info(f'Limits for X not found')
        ax.set_ylim(y_limit) if len(y_limit) != 0 else logging.info(f'Limits for Y not found')
        mplcursors.cursor() # hover=True
        plt.title(f'{files[i]}')
        # fig.colorbar(im)
        # fig.tight_layout()

    # Создание слайдера для редактирования границ цветовой шкалы
    ax_slider = plt.axes([0.25, 0.03, 0.5, 0.01], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Threshold', im.norm.vmin, im.norm.vmax, valinit=im.norm.vmax)
    slider.on_changed(update)

    # Список для хранения координат кликов пользователя
    click_coords = []

    # Привязка обработчика событий клика мыши к графику
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Отображение графика на полный экран
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()

    # Вывод последней координаты двойного клика пользователя
    if len(click_coords) > 0:
        print(f"For image {i+1} last double click coordinates: {str(click_coords[-1])}")
        logging.info(f"For image {i+1} - {files[i]} - last double click coordinates: {str(click_coords[-1])}")
        if flag == 'RL':
            coordinates_of_control_point.append(click_coords[-1])
            plt.close()

        elif flag == 'IV':
            coordinates_of_control_point.append(click_coords[-1])
            coordinates_of_control_point.append(click_coords[-1])
    else:
        print("No double click coordinates recorded")
        logging.info(f"No double click coordinates recorded")
        sys.exit()
    
def alignment_sun_disk(files : list = files, method : str = 'search_max_in_area', area : int = None):
    
    # Загрузка первого файла и нахождение координат отличительного признака
    hdul1 = fits.open(f'{current_directory}/{files[0]}')
    img1 = hdul1[0].data  # данные первого изображения
    hdul1.close()

    try:
        if method == 'search_max_in_area':

            control_point_1 = (coordinates_of_control_point[0][0], coordinates_of_control_point[0][1])  # координаты признака на первом изображении
            if int(np.sqrt(img1.size)) == 1024:
                max_col, max_row, max_value = find_max_around_point(img1, reversed(control_point_1), 15)

            elif int(np.sqrt(img1.size)) == 512:
                max_col, max_row, max_value = find_max_around_point(img1, reversed(control_point_1), 13)
                
            control_point_1 = (max_row, max_col)
            coordinates_of_max_point_in_area.append(control_point_1)
            logging.info(f"Max - {max_value} in {control_point_1}")
                
            # else:
            #     max_col, max_row, max_value = find_max_around_point(img1, reversed(control_point_1), area)
            #     control_point_1 = (max_col, max_row)
            #     logging.info(f"Max - {max_value} in {control_point_1}")

        elif method == 'linear_image_shift':
            control_point_1 = (coordinates_of_control_point[0][0], coordinates_of_control_point[0][1])  # координаты признака на первом изображении

    except:
        logprint('The program is terminated due to lack of alignment data')

    # Определяем место для сохранения
    try:
        os.mkdir(f'{current_directory}_aligned')
    except:
        shutil.rmtree(f'{current_directory}_aligned') if platform == 'win32' else os.system(f'rm -rf {current_directory}_aligned')
        os.mkdir(f'{current_directory}_aligned')
        
    if working_mode == 'folder':
        
        # пересохранение первого файла в новую директорию
        hdul2 = fits.open(f'{current_directory}/{files[0]}')
        header = hdul2[0].header
        img2 = hdul2[0].data
        hdul2.close()

        fits.writeto(f'{current_directory}_aligned/{files[0][:-4]}_aligned.fits', img2, overwrite=True, header=header)
        logging.info(f"Image {1}: {files[0]} - saved")
        
        # Цикл для совмещения остальных файлов
        for i, file in enumerate(files[1:]):
            # Загрузка файла и нахождение координат отличительного признака
            hdul2 = fits.open(f'{current_directory}/{file}')
            header = hdul2[0].header
            img2 = hdul2[0].data
            hdul2.close()

            try:
                control_point_2 = (coordinates_of_control_point[i+1][0], coordinates_of_control_point[i+1][1])  # координаты признака на текущем изображени

            except:
                logprint('The program is terminated due to lack of alignment data')

            if method == 'search_max_in_area':

                if int(np.sqrt(img2.size)) == 1024:
                    max_col, max_row, max_value = find_max_around_point(img2, reversed(control_point_2), 15)

                elif int(np.sqrt(img2.size)) == 512:
                    max_col, max_row, max_value = find_max_around_point(img2, reversed(control_point_2), 8)
                    
                control_point_2 = (max_row, max_col)
                coordinates_of_max_point_in_area.append(control_point_2)
                logging.info(f"Max - {max_value} in {control_point_2}")

                # Нахождение горизонтального и вертикального сдвигов между изображениями
                dx = control_point_1[0] - control_point_2[0]
                dy = control_point_1[1] - control_point_2[1]

                # Сдвигаем изображение
                img2 = np.roll(img2, dx, axis=1)
                img2 = np.roll(img2, dy, axis=0)
                
                if working_mode == 'folder':
                    # Сохранение выравненного изображения
                    fits.writeto(f'{current_directory}_aligned/{file[:-4]}_aligned.fits', img2, overwrite=True, header=header)
                    logging.info(f"Image {i+2}: {file} - saved")     

            elif method == 'linear_image_shift':
                
                # Нахождение горизонтального и вертикального сдвигов между изображениями
                dx = control_point_1[0] - control_point_2[0]
                dy = control_point_1[1] - control_point_2[1]

                # Сдвигаем изображение
                img2 = np.roll(img2, dx, axis=1)
                img2 = np.roll(img2, dy, axis=0)

                if working_mode == 'folder':
                    # Сохранение выравненного изображения
                    fits.writeto(f'{current_directory}_aligned/{file[:-4]}_aligned.fits', img2, overwrite=True, header=header)
                    logging.info(f"Image {i+2}: {file} - saved")

    elif working_mode == 'directory_with_folders':
                
        # копируем первую папку, так как все привязки делаем по ней
        shutil.copytree(f'{source_directory_path}/{freqs[0]}', f'{source_directory_path}_aligned/{freqs[0]}', dirs_exist_ok=True)        
        logging.info(f"Image {freqs[0]} - saved")
        
        try:
            for fq in freqs[1:]:
                os.mkdir(f'{source_directory_path}_aligned/{fq}')
                logging.info(f'The folder has been created to save the results at the {fq} frequency')
        except Error as err: 
            logging.info(f'The folder for saving the results has not been created: {err}')
            
        for k, freq_folder in enumerate(freqs[1:]):
            
            files = [file for file in listdir(f'{source_directory_path}/{freq_folder}') if isfile(f'{source_directory_path}/{freq_folder}/{file}')]

            # Цикл для совмещения остальных файлов
            for i, file in enumerate(files):
                print(file)
                # Загрузка файла и нахождение координат отличительного признака
                hdul2 = fits.open(f'{source_directory_path}/{freq_folder}/{file}')
                header = hdul2[0].header
                img2 = hdul2[0].data
                hdul2.close()

                try:
                    control_point_2 = (coordinates_of_control_point[k+1][0], coordinates_of_control_point[k+1][1])  # координаты признака на текущем изображени

                except:
                    logprint('The program is terminated due to lack of alignment data')

                if method == 'search_max_in_area':

                    if int(np.sqrt(img2.size)) == 1024:
                        max_col, max_row, max_value = find_max_around_point(img2, reversed(control_point_2), 15)

                    elif int(np.sqrt(img2.size)) == 512:
                        max_col, max_row, max_value = find_max_around_point(img2, reversed(control_point_2), 8)
                        
                    control_point_2 = (max_row, max_col)
                    logging.info(f"Max - {max_value} in {control_point_2}")

                    # Нахождение горизонтального и вертикального сдвигов между изображениями
                    dx = control_point_1[0] - control_point_2[0]
                    dy = control_point_1[1] - control_point_2[1]

                    # Сдвигаем изображение
                    img2 = np.roll(img2, dx, axis=1)
                    img2 = np.roll(img2, dy, axis=0)
                    
                    # Сохранение выравненного изображения
                    fits.writeto(f'{source_directory_path}_aligned/{freq_folder}/{file[:-4]}_aligned.fits', img2, overwrite=True, header=header)
                    logging.info(f"Image {i+2}: {file} - saved")     

                elif method == 'linear_image_shift':
                    
                    # Нахождение горизонтального и вертикального сдвигов между изображениями
                    dx = control_point_1[0] - control_point_2[0]
                    dy = control_point_1[1] - control_point_2[1]

                    # Сдвигаем изображение
                    img2 = np.roll(img2, dx, axis=1)
                    img2 = np.roll(img2, dy, axis=0)

                    # Сохранение выравненного изображения
                    fits.writeto(f'{source_directory_path}_aligned/{freq_folder}/{file[:-4]}_aligned.fits', img2, overwrite=True, header=header)
                    logging.info(f"Image {i+2}: {file} - saved")
                    
            logprint('****************************')

    
    logprint('Finish program alignment of the solar disk')
    print('For more details: read file "logs.log"')
    print(coordinates_of_max_point_in_area)
    
    # setting_of_alignes = np.array([freqs, [coordinates_of_max_point_in_area[i:i+2] for i in range(0, len(coordinates_of_max_point_in_area), 2)]])
    
    # np.save(f'setting_of_alignes_{files[0][4:12]}.npy', setting_of_alignes)
    # # np.load('setting_of_alignes_20230716.npy', allow_pickle=True)
    # print(setting_of_alignes)

##############################################################
if __name__ == '__main__':
    
    alignment_sun_disk(method = 'search_max_in_area')