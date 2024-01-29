from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
import os
import re

import logging
# logging.basicConfig(filename = 'logs.log',  filemode='a', level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s', encoding = "UTF-8", datefmt='%d-%b-%y %H:%M:%S')
logging.basicConfig(filename = 'logs.log',  filemode='a', level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.info(f'Start program show of the solar on difference frequency')

# Каталог из которого будем брать файлы
directory = "D:/datasets/20.01.22/times/20220120T055530_calibrated_brightness_aligned"
vcenter = 200000

logging.info(f'Path to files: {directory}')

pattern = re.compile(r'(?<=[_.])\d{4,5}(?=[_.])')

# функция для извлечения цифр из названия файла
def extract_number(filename):
    match = pattern.search(filename)
    return int(match.group())

files = sorted(os.listdir(directory), key=extract_number)
logging.info(f'Search {len(files)} files')
logging.info(f'Files: \n {files}')

# отображение всех значений матрицы в консоль
np.set_printoptions(threshold=np.inf)

# Загрузка fits файла
# fits_image_file = fits.open('data/9_srh_V_2022-05-11T05_57_45_9400.fit', ignore_missing_simple=True)
# # Получение матрицы numpy из данных fits
# fits_data = fits_image_file[0].data

# получение границ кропа через срезы из строк и каждой строки
# stroka_1, stroka_2, stolbec_1, stolbec_2 = 615, 640, 315, 335
stroka_1, stroka_2, stolbec_1, stolbec_2 = 0, 1024, 0, 1024
    
def multiple_crope_images_display(input_matrix_list_files, NX=8, NY=4):
    # https://teletype.in/@pythontalk/matplotlib_subplot_tutorial
    fig, axs = plt.subplots(NY, NX, sharex=True, sharey=True, figsize=(NX*3,NY*3))

    for i in range(0, len(files)):
        vcenter = - 1500 * (i + 1) + 118000
        # print(input_matrix_list_files[i])
        fits_image_file = fits.open(f'{directory}/{input_matrix_list_files[i]}', ignore_missing_simple=True)[0].data
        ax = axs[i//NX, i%NX]
        freq = re.search(r'(?<=[_.])\d{4,5}(?=[_.])', str(input_matrix_list_files[i])).group()
        i_or_v = re.search(r'[I|V]', str(input_matrix_list_files[i]))
        r_or_l = re.search(r'(RCP|LCP|R|L)', str(input_matrix_list_files[i]))
        # print(f'рисую в клетке {i+1}')
        # ax.imshow((fits_image_file)[stroka_1:stroka_2, stolbec_1:stolbec_2], origin='lower', cmap='plasma', interpolation='gaussian')
        ax.imshow((fits_image_file)[stroka_1:stroka_2, stolbec_1:stolbec_2], origin='lower', cmap='plasma', norm=TwoSlopeNorm(vmin=5000, vcenter=vcenter), extent=[0, fits_image_file.shape[1], 0, fits_image_file.shape[0]])
        
        # ax.plot(653, 614, '+', markersize=25, color = 'k')
        # ax.plot(324, 628, 'x', markersize=15, color = 'k')
        
        ax.plot(890, 563, 'x', markersize=15, color = 'k')
        # circle = Circle((512, 512), 425, color='white', fill=False)
        # ax.add_artist(circle)
        ax.set_title(f'Freq {freq}, {r_or_l.group()}' if r_or_l else f'Freq {freq}, Stoks {i_or_v.group()}')
        ax.axis('off') 
    
    fig.tight_layout()
    # Получение менеджера окна
    manager = plt.get_current_fig_manager()
    # Открытие окна на полный экран
    manager.full_screen_toggle()
    plt.show()

multiple_crope_images_display(files)

logging.info(f'Stop')
