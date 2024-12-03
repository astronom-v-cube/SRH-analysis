import logging
import os
import re

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.patches import Circle
from tqdm import tqdm

from analise_utils import Extract, Monitoring

Extract = Extract()
Monitoring.start_log('logs')
logging.info(f'Start program show of the solar on difference frequency')

# Каталог из которого будем брать файлы
# directory = "D:/datasets/20.01.22/times/20220120T055800_calibrated_brightness_aligned"
# directory = "D:\datasets/20.01.22/times/20220120T055630_calibrated_brightness_COM_aligned"
directory = "/mnt/astro/14may_new_612_times/20240514T014200_OWM_aligned"
vcenter = 5000

logging.info(f'Path to files: {directory}')

files = sorted(os.listdir(directory), key=lambda x: Extract.extract_number(x))
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
# stroka_1, stroka_2, stolbec_1, stolbec_2 = 543, 583, 873, 913

def multiple_crope_images_display(input_matrix_list_files, NX=4, NY=4):
    # https://teletype.in/@pythontalk/matplotlib_subplot_tutorial
    fig, axs = plt.subplots(NY, NX, sharex=True, sharey=True, figsize=(NX*3,NY*3))

    for i in tqdm(range(0, len(files))):
        vcenter = - 1500 * (i + 1) + 200000
        # print(input_matrix_list_files[i])
        fits_image_file = fits.open(f'{directory}/{input_matrix_list_files[i]}', ignore_missing_simple=True)[0].data
        image_slice = (fits_image_file)[stroka_1:stroka_2, stolbec_1:stolbec_2]

        ax = axs[i//NX, i%NX]
        freq = re.search(r'(?<=[_.])\d{4,5}(?=[_.])', str(input_matrix_list_files[i])).group()
        i_or_v = re.search(r'[I|V]', str(input_matrix_list_files[i]))
        r_or_l = re.search(r'(RCP|LCP|R|L)', str(input_matrix_list_files[i]))
        # print(f'рисую в клетке {i+1}')
        # ax.imshow((fits_image_file)[stroka_1:stroka_2, stolbec_1:stolbec_2], origin='lower', cmap='plasma', interpolation='gaussian')
        ax.imshow((fits_image_file)[410:465, 262:312], origin='lower', cmap='plasma', norm=TwoSlopeNorm(vmin=0, vcenter=vcenter, vmax=300000), extent=[0, fits_image_file.shape[1], 0, fits_image_file.shape[0]])


        # Создание контуров только для определенной части изображения
        x, y  = range(stolbec_1, stolbec_2), range(stroka_1, stroka_2)
        contour = ax.contour(x, y, image_slice, colors='lime', levels=[image_slice.max()*0.2 - 1, image_slice.max()*0.5 - 1, image_slice.max()*0.7 - 1, image_slice.max()*0.9 - 1])
        # минус 1 т.к. для прямоугольника координаты это левый нижний угол а не центр
        rectangle1 = patches.Rectangle((885-1, 578-1), 3, 3, linewidth=1.5, edgecolor='k', facecolor='none', zorder = 5)
        rectangle2 = patches.Rectangle((890-1, 551-1), 3, 3, linewidth=1.5, edgecolor='k', facecolor='none', zorder = 5)
        rectangle3 = patches.Rectangle((880-1, 564-1), 3, 3, linewidth=1.5, edgecolor='k', facecolor='none', zorder = 5)
        rectangle4 = patches.Rectangle((893-1, 565-1), 3, 3, linewidth=1.5, edgecolor='k', facecolor='none', zorder = 5)
        ax.add_patch(rectangle1)
        ax.add_patch(rectangle2)
        ax.add_patch(rectangle3)
        ax.add_patch(rectangle4)
        for line in contour.collections:
            line.set_linewidth(1.2)  # Увеличение толщины контура
            line.set_antialiased(True)  # Сглаживание контуров

        # ax.plot(653, 614, '+', markersize=25, color = 'k')
        # ax.plot(324, 628, 'x', markersize=15, color = 'k')

        ax.plot(890, 563, 'x', markersize=15, color = 'k')
        # ax.plot(888, 572, '+', markersize=15, color = 'k')
        # ax.plot(891, 555, '+', markersize=15, color = 'k')
        # ax.plot(885, 563, '+', markersize=15, color = 'k')
        # ax.plot(893, 563, '+', markersize=15, color = 'k')
        # ax.grid(which='both', color='black', linestyle='-', linewidth=0.1)
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
