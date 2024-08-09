import logging
import os
import re
import sys

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.colors import TwoSlopeNorm
from matplotlib.widgets import Slider

from analise_utils import (ArrayOperations, ConvertingArrays, Extract,
                           FindIntensity, Monitoring, MplFunction,
                           OsOperations, ZirinTb)

Monitoring.start_log('log')
logging.info(f'Start program alignment of the solar disk')
extract = Extract()
# norm=TwoSlopeNorm(vmin=0, vcenter=vcenter)

##########
directory = '08-26-46'
flag = 'IV'
flag = 'RL'
vcenter = 25000
counter_level = 0.5
x_limit = (600, 750)
y_limit = (520, 700)

x_limit = (350, 450)
y_limit = (580, 680)
##########
logging.info(f'Path to files: {directory}')

freqs = set()
files = sorted(os.listdir(directory), key=lambda x: extract.extract_number(x, freqs))
freqs = sorted(list(freqs), reverse = True)

logging.info(f'Find {len(files)} files')
logging.info(f'List files: \n {files}')

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

coordinates_of_control_point = []
coordinates_of_reference_point_in_area = []
square_psf = list()

if flag == 'RL':
    iterable = range(0, len(files))
elif flag == 'IV':
    iterable = range(0, len(files), 2)

for i in iterable:

    if flag == 'RL':
        hdul = fits.open(f'{directory}/{files[i]}', ignore_missing_simple=True)
        data = hdul[0].data
        header = hdul[0].header

        psf_a = header['PSF_ELLA']
        psf_b = header['PSF_ELLB']
        square_psf.append(int(3.1415 * psf_a * psf_b))

        # Создание графика и отображение данных
        fig, ax = plt.subplots(figsize=(9, 9))
        vcenter = - 1500 * (i+1) + 118000
        im = ax.imshow(data, origin='lower', cmap='plasma', extent=[0, data.shape[1], 0, data.shape[0]], norm=TwoSlopeNorm(vmin=0, vcenter=vcenter))

        ax.set_xlim(x_limit) if len(x_limit) != 0 else logging.info(f'Limits for X not found')
        ax.set_ylim(y_limit) if len(y_limit) != 0 else logging.info(f'Limits for Y not found')
        # mplcursors.cursor() # hover=True
        plt.title(f'{files[i]}')
        # fig.colorbar(im)
        # fig.tight_layout()

    elif flag == 'IV':
        # Считывание файлов
        hdul1 = fits.open(f'{directory}/{files[i]}', ignore_missing_simple=True)
        data1 = hdul1[0].data
        header1 = hdul1[0].header

        psf_a = header1['PSF_ELLA']
        psf_b = header1['PSF_ELLB']
        square_psf.append(int(3.1415 * psf_a * psf_b))

        hdul2 = fits.open(f'{directory}/{files[i+1]}', ignore_missing_simple=True)
        data2 = hdul2[0].data

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
        im = ax.imshow(I, origin='lower', cmap='plasma', extent=[0, I.shape[1], 0, I.shape[0]], norm=colors.Normalize(vmin=0, vmax=350000))
        # mplcursors.cursor(hover=True)
        plt.title(f'{files[i]}')
        # fig.colorbar(im)
        # fig.tight_layout()

    # Создание слайдера для редактирования границ цветовой шкалы
    ax_slider = plt.axes([0.25, 0.03, 0.5, 0.01], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Threshold', im.norm.vmin, im.norm.vmax, valinit=im.norm.vmax/2)
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
        logging.info(f"For image {i+1} last double click coordinates: {str(click_coords[-1])}")
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
    hdul1 = fits.open(f'{directory}/{files[0]}')
    img1 = hdul1[0].data  # данные первого изображения

    try:
        if method == 'search_max_in_area':

            control_point_1 = (coordinates_of_control_point[0][0], coordinates_of_control_point[0][1])  # координаты признака на первом изображении
            if int(np.sqrt(img1.size)) == 1024:
                reference_col, reference_row, max_value = ArrayOperations.find_max_around_point(img1, reversed(control_point_1), 28)

            elif int(np.sqrt(img1.size)) == 512:
                reference_col, reference_row, max_value = ArrayOperations.find_max_around_point(img1, reversed(control_point_1), 14)

            control_point_1 = (reference_row, reference_col)
            coordinates_of_reference_point_in_area.append(control_point_1)
            logging.info(f"Reference - {max_value} in {control_point_1}")

            # else:
            #     max_col, max_row, max_value = find_max_around_point(img1, reversed(control_point_1), area)
            #     control_point_1 = (max_col, max_row)
            #     logging.info(f"Max - {max_value} in {control_point_1}")

        elif method == 'contour_weighted_average':

            control_point_1 = (coordinates_of_control_point[0][0], coordinates_of_control_point[0][1])  # координаты признака на первом изображении
            reference_col, reference_row = ArrayOperations.calculate_weighted_centroid(img1, control_point_1, 0.5)

            control_point_1 = (reference_col, reference_row)
            coordinates_of_reference_point_in_area.append(control_point_1)
            logging.info(f"Reference - {max_value} in {control_point_1}")

        elif method == 'linear_image_shift':
            control_point_1 = (coordinates_of_control_point[0][0], coordinates_of_control_point[0][1])  # координаты признака на первом изображении

    except:
        Monitoring.logprint('The program is terminated due to lack of alignment data')

    hdul1.close()

    # Определяем место для сохранения
    OsOperations.create_place(directory, 'aligned')

    # пересохранение первого файла в новую директорию
    hdul2 = fits.open(f'{directory}/{files[0]}')
    header = hdul2[0].header
    img2 = hdul2[0].data
    hdul2.close()
    fits.writeto(f'{directory}_aligned/{files[0][:-4]}_aligned.fits', img2, overwrite=True, header=header)
    logging.info(f"Image {1}: {files[0]} - saved")

    # Цикл для совмещения остальных файлов
    for i, file in enumerate(files[1:]):
        # Загрузка файла и нахождение координат отличительного признака
        hdul2 = fits.open(f'{directory}/{file}')
        header = hdul2[0].header
        img2 = hdul2[0].data
        hdul2.close()

        try:
            control_point_2 = (coordinates_of_control_point[i+1][0], coordinates_of_control_point[i+1][1])  # координаты признака на текущем изображени

        except:
            Monitoring.logprint('The program is terminated due to lack of alignment data')

        if method == 'search_max_in_area':

            if int(np.sqrt(img2.size)) == 1024:
                reference_col, reference_row, max_value = ArrayOperations.find_max_around_point(img2, reversed(control_point_2), 28)

            elif int(np.sqrt(img2.size)) == 512:
                reference_col, reference_row, max_value = ArrayOperations.find_max_around_point(img2, reversed(control_point_2), 14)

            control_point_2 = (reference_row, reference_col)
            coordinates_of_reference_point_in_area.append(control_point_2)
            logging.info(f"Reference - {max_value} in {control_point_2}")

        elif method == 'contour_weighted_average':

            reference_col, reference_row, max_value = ArrayOperations.calculate_weighted_centroid(img2, reversed(control_point_2), counter_level)
            control_point_2 = (reference_row, reference_col)
            coordinates_of_reference_point_in_area.append(control_point_2)
            logging.info(f"Reference - {max_value} in {control_point_2}")

        elif method == 'linear_image_shift':
            pass

        # Нахождение горизонтального и вертикального сдвигов между изображениями
        dx = control_point_1[0] - control_point_2[0]
        dy = control_point_1[1] - control_point_2[1]

        # Сдвигаем изображение
        img2 = np.roll(img2, dx, axis=1)
        img2 = np.roll(img2, dy, axis=0)

        # Сохранение выравненного изображения
        fits.writeto(f'{directory}_aligned/{file[:-4]}_aligned.fits', img2, overwrite=True, header=header)
        logging.info(f"Image {i+2}: {file} - saved")

    Monitoring.logprint('Finish program alignment of the solar disk')
    print('For more details: read file "logs.log"')

    np.save('setting_of_alignes.npy', coordinates_of_reference_point_in_area)
    print(coordinates_of_reference_point_in_area)

    square_psf_list = sorted(square_psf, reverse=True)
    np.savez('psf_square.npz', freqs = freqs, psf_square = square_psf_list)

    print(square_psf_list)

##############################################################
if __name__ == '__main__':
    alignment_sun_disk(method = 'search_max_in_area')