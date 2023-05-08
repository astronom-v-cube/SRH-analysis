from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
import re

import logging
# logging.basicConfig(filename = 'logs.log',  filemode='a', level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s', encoding = "UTF-8", datefmt='%d-%b-%y %H:%M:%S')
logging.basicConfig(filename = 'logs.log',  filemode='a', level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.info(f'Start program show of the solar on difference frequency')

# Каталог из которого будем брать файлы
directory = 'aligned'

logging.info(f'Путь к файлам: {directory}')
files = sorted(os.listdir(directory))
logging.info(f'Найдено {len(files)} файлов')
logging.info(f'Файлы: \n {files}')

# отображение всех значений матрицы в консоль
np.set_printoptions(threshold=np.inf)

# Загрузка fits файла
# fits_image_file = fits.open('data/9_srh_V_2022-05-11T05_57_45_9400.fit', ignore_missing_simple=True)
# # Получение матрицы numpy из данных fits
# fits_data = fits_image_file[0].data

# получение границ кропа через срезы из строк и каждой строки
# stroka_1, stroka_2, stolbec_1, stolbec_2 = 310, 360, 105, 165
stroka_1, stroka_2, stolbec_1, stolbec_2 = 0, 1024, 0, 1024
    
def multiple_crope_images_display(input_matrix_list_files, NX=4, NY=4):
    # https://teletype.in/@pythontalk/matplotlib_subplot_tutorial
    fig, axs = plt.subplots(NY, NX, sharex=True, sharey=True, figsize=(NX*3,NY*3))

    for i in range(0, len(files)):
        # print(input_matrix_list_files[i])
        fits_image_file = fits.open(f'{directory}/{input_matrix_list_files[i]}', ignore_missing_simple=True)[0].data
        ax = axs[i//NX, i%NX]
        freq = re.search(r'(?<=[_.])\d{4,5}(?=[_.])', str(input_matrix_list_files[i])).group()
        i_or_v = re.search(r'[I|V]', str(input_matrix_list_files[i]))
        r_or_l = re.search(r'(RCP|LCP|R|L)', str(input_matrix_list_files[i]))
        # print(f'рисую в клетке {i+1}')
        # ax.imshow((fits_image_file)[stroka_1:stroka_2, stolbec_1:stolbec_2], origin='lower', cmap='plasma', interpolation='gaussian')
        ax.imshow((fits_image_file)[stroka_1:stroka_2, stolbec_1:stolbec_2], origin='lower', cmap='plasma', extent=[0, fits_image_file.shape[1], 0, fits_image_file.shape[0]])
        # circle = Circle((512, 512), 425, color='white', fill=False)
        # ax.add_artist(circle)
        ax.set_title(f'Freq {freq}, {r_or_l.group()}' if r_or_l else f'Freq {freq}, Stoks {i_or_v.group()}')
        ax.axis('off') 
    
    fig.tight_layout() 
    plt.show()

multiple_crope_images_display(files)

logging.info(f'Stop')
