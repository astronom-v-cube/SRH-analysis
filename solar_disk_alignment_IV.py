import logging
import sys

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.colors import TwoSlopeNorm
from matplotlib.widgets import Slider
from tqdm import tqdm

from analise_utils import ArrayOperations, Extract, Monitoring, OsOperations

Monitoring.start_log('log')
extract = Extract()
logging.info(f'Start program alignment of the solar disk')
# norm=TwoSlopeNorm(vmin=0, vcenter=vcenter) /// colors.Normalize(vmin=0, vcenter=vcenter))

##########
# dateandtime = '20220120T060200copy'
# directory = f"D:/datasets/20.01.22/times/{dateandtime}"
directory = f"A:/flags"
file_settings_name = '14.05.24'
folder_mode = 'folder_with_folders'    # 'one_folder' /one_time_moment/ or 'folder_with_folders' /many__time_moment/
mode = 'interactive'          # 'saved_settings' or 'interactive'
method = 'WA'                # 'MAP' /max around point/ or 'COM' /center of mass/ or 'WA' /weighted average/
postfix = 'WA_aligned' if method == 'WA' else 'COM_aligned or MAP...'
vcenter = 5000

x_limit = (140, 220)
y_limit = (420, 490)

x_limit = (810, 950)
y_limit = (320, 460)

setting_of_alignes = []
##########
logging.info(f'Path to files: {directory}')

files, freqs = OsOperations.freq_sorted_files_in_folder(directory) if folder_mode == 'one_folder' else OsOperations.freq_sorted_1st_two_files_in_folders(directory)
iterable = range(0, len(files), 2)

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
    ax.figure.canvas.draw_idle()

if mode == 'saved_settings':
    setting_of_alignes = np.load(f'{file_settings_name}.npy')

elif mode == 'interactive':

    coordinates_of_control_point = []
    coordinates_of_max_point_in_area = []
    square_psf = list()

    for i in iterable:

        hdul1 = fits.open(f'{directory}/{freqs[i//2] if folder_mode == "folder_with_folders" else ""}/{files[i]}', ignore_missing_simple=True)
        hdul2 = fits.open(f'{directory}/{freqs[(i+1)//2] if folder_mode == "folder_with_folders" else ""}/{files[i+1]}', ignore_missing_simple=True)
        data1 = hdul1[0].data
        data2 = hdul2[0].data

        header1 = hdul1[0].header
        psf_a = header1['PSF_ELLA']
        psf_b = header1['PSF_ELLB']
        square_psf.append(int(3.1415 * psf_a * psf_b))

        I = data1 + data2

        fig, ax = plt.subplots(figsize=(12, 12))
        vcenter = -500 * (i + 1) + 80000
        vcenter = - 1500 * (i + 1) + 200000
        im = ax.imshow(I, origin='lower', cmap='plasma', extent=[0, I.shape[1], 0, I.shape[0]], norm=TwoSlopeNorm(vmin=0, vcenter=vcenter, vmax=200000))
        cropped_I = I[y_limit[0]:y_limit[1], x_limit[0]:x_limit[1]]
        # contour_levels = np.linspace(cropped_I.max() * 0.5, cropped_I.max(), 7)
        contour_levels = [cropped_I.max() * 0.5]
        ax.contour(cropped_I, levels=contour_levels, colors='k', extent=[x_limit[0], x_limit[1], y_limit[0], y_limit[1]])
        ax.set_xlim(x_limit) if len(x_limit) != 0 else logging.info(f'Limits for X not found')
        ax.set_ylim(y_limit) if len(y_limit) != 0 else logging.info(f'Limits for Y not found')
        # mplcursors.cursor(hover=True)
        plt.title(f'{files[i][:-4] + "  +  " + files[i+1][:-4]}')
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
        # Отображение на полный экран
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.show()

        # Вывод последней координаты двойного клика пользователя
        if len(click_coords) > 0:
            Monitoring.logprint(f"For image {i+1} last double click coordinates: {str(click_coords[-1])}")
            coordinates_of_control_point.append(click_coords[-1])
            coordinates_of_control_point.append(click_coords[-1])
            plt.close()

        else:
            Monitoring.logprint(f"No double click coordinates recorded")
            sys.exit()

def alignment_sun_disk(files : list = files, method : str = 'search_max_in_area', area : int = None):

    # Загрузка первого файла и нахождение координат отличительного признака
    hdul1 = fits.open(f'{directory}/{freqs[0] if folder_mode == "folder_with_folders" else ""}/{files[0]}', ignore_missing_simple=True)
    hdul2 = fits.open(f'{directory}/{freqs[0] if folder_mode == "folder_with_folders" else ""}/{files[1]}', ignore_missing_simple=True)
    img1 = hdul1[0].data
    img2 = hdul2[0].data
    header1 = hdul1[0].header
    header2 = hdul2[0].header
    hdul1.close()
    hdul2.close()

    if mode == 'interactive':

        I = img1 + img2

        # try:
        control_point_1 = (coordinates_of_control_point[0][0], coordinates_of_control_point[0][1])  # координаты признака на первом изображении

        # if method == 'search_max_in_area':

        #     if int(np.sqrt(img1.size)) == 1024:
        #         reference_col, reference_row, max_value = ArrayOperations.find_max_around_point(I, reversed(control_point_1), 14) if method == 'MAP' else ArrayOperations.find_center_of_mass(I, reversed(control_point_1), 14)

        #     elif int(np.sqrt(img1.size)) == 512:
        #         reference_col, reference_row, max_value = ArrayOperations.find_max_around_point(I, reversed(control_point_1), 28) if method == 'MAP' else ArrayOperations.find_center_of_mass(I, reversed(control_point_1), 28)
        print(coordinates_of_control_point)
        if method == 'WA':
            reference_col, reference_row, max_value = ArrayOperations.calculate_weighted_centroid(I, tuple(control_point_1), 0.5, (x_limit, y_limit))

        control_point_1 = (reference_row, reference_col)
        coordinates_of_max_point_in_area.append(control_point_1)
        logging.info(f"Max - {max_value} in {control_point_1}")

        # except Exception as err:
        #     Monitoring.logprint('The program is terminated due to lack of alignment data')
        #     Monitoring.logprint(err)

    # Определяем место для сохранения
    OsOperations.create_place(directory, postfix)
    if folder_mode == 'folder_with_folders':
        for freq in tqdm(freqs, desc='Создание папок для частот'):
            OsOperations.create_place(f'{directory}_{postfix}/{freq}')

    if folder_mode == 'one_folder':
        # пересохранение первой пары файлов в новую директорию
        fits.writeto(f'{directory}_{postfix}/{files[0][:-4]}_{postfix}.fits', img1, overwrite=True, header=header1)
        logging.info(f"Image {1}: {files[0]} - saved")
        fits.writeto(f'{directory}_{postfix}/{files[1][:-4]}_{postfix}.fits', img2, overwrite=True, header=header2)
        logging.info(f"Image {2}: {files[1]} - saved")

    elif folder_mode == 'folder_with_folders':
        all_files_in_freq, freq = OsOperations.freq_sorted_files_in_folder(f'{directory}/{freqs[0]}')
        for file in tqdm(all_files_in_freq, desc='Копирование файлов привязочной частоты'):
            hdul1 = fits.open(f'{directory}/{freq[0]}/{file}', ignore_missing_simple=True)
            img1 = hdul1[0].data
            header1 = hdul1[0].header
            hdul1.close()
            fits.writeto(f'{directory}_{postfix}/{freq[0]}/{file[:-4] if file[-1]=="t" else file[:-5]}.fits', img1, overwrite=True, header=header1)
            logging.info(f"Image: {file} - saved")

    # Цикл для совмещения остальных файлов
    for i in tqdm(iterable[1:], desc='Общий прогресс выполнения'):

        hdul1 = fits.open(f'{directory}/{freqs[i//2] if folder_mode == "folder_with_folders" else ""}/{files[i]}')
        hdul2 = fits.open(f'{directory}/{freqs[i//2] if folder_mode == "folder_with_folders" else ""}/{files[i + 1]}')
        img1 = hdul1[0].data  # данные первого изображения
        img2 = hdul2[0].data  # данные второго изображения
        header1 = hdul1[0].header
        header2 = hdul2[0].header
        hdul1.close()
        hdul2.close()

        if mode == 'interactive':

            I = img1 + img2

            try:
                control_point_2 = (coordinates_of_control_point[i][0], coordinates_of_control_point[i][1])  # координаты признака на текущем изображени
            except Exception as err:
                Monitoring.logprint('The program is terminated due to lack of alignment data')
                Monitoring.logprint(err)

            # if method == 'search_max_in_area':
            #     if int(np.sqrt(img2.size)) == 1024:
            #         reference_col, reference_row, max_value = ArrayOperations.find_max_around_point(I, reversed(control_point_2), 14)
            #     elif int(np.sqrt(img2.size)) == 512:
            #         reference_col, reference_row, max_value = ArrayOperations.find_max_around_point(I, reversed(control_point_2), 28)

            if method == 'WA':
                reference_col, reference_row, max_value = ArrayOperations.calculate_weighted_centroid(I, tuple(control_point_2), 0.5, (x_limit, y_limit))
                control_point_2 = (reference_row, reference_col)
                coordinates_of_max_point_in_area.append(control_point_2)
                logging.info(f"Max - {max_value} in {control_point_2}")

            # Нахождение горизонтального и вертикального сдвигов между изображениями на опорной и расчетной частоте
            dx = control_point_1[0] - control_point_2[0]
            dy = control_point_1[1] - control_point_2[1]
            setting_of_alignes.append((dx, dy))
            setting_of_alignes.append((dx, dy))

        elif mode == 'saved_settings':
            dx, dy = setting_of_alignes[i-2][0], setting_of_alignes[i-2][1]

        if folder_mode == 'one_folder':
            # Сдвигаем изображения
            img1 = np.roll(img1, dx, axis=1)
            img1 = np.roll(img1, dy, axis=0)
            img2 = np.roll(img2, dx, axis=1)
            img2 = np.roll(img2, dy, axis=0)
            # Сохранение выравненных изображений
            fits.writeto(f'{directory}_{postfix}/{files[i][:-5] if files[i][-1] == "s" else files[i][:-4]}_{postfix}.fits', img1, overwrite=True, header=header1)
            logging.info(f"Image {i+2}: {files[i]} - saved")
            fits.writeto(f'{directory}_{postfix}/{files[i+1][:-5] if files[i+1][-1] == "s" else files[i+1][:-4]}_{postfix}.fits', img2, overwrite=True, header=header2)
            logging.info(f"Image {i+2}: {files[i+1]} - saved")

        elif folder_mode == 'folder_with_folders':
            all_files_in_freq, freq = OsOperations.freq_sorted_files_in_folder(f'{directory}/{freqs[i//2]}')
            for file in tqdm(all_files_in_freq, desc=f'Обработка файлов частоты {freqs[i//2]}', leave=False):
                hdul1 = fits.open(f'{directory}/{freqs[i//2]}/{file}', ignore_missing_simple=True)
                img1 = hdul1[0].data
                header1 = hdul1[0].header
                hdul1.close()
                img1 = np.roll(img1, dx, axis=1)
                img1 = np.roll(img1, dy, axis=0)
                fits.writeto(f'{directory}_{postfix}/{freqs[i//2]}/{file[:-4] if file[-1]== "t" else file[:-5]}_{postfix}.fits', img1, overwrite=True, header=header1)
                logging.info(f"Image: {file} - saved")

    Monitoring.logprint('Finish program alignment of the solar disk')
    print('For more details: read file "logs.log"')

    if mode == 'interactive':
        np.save(f'{file_settings_name}.npy', setting_of_alignes)
        print(setting_of_alignes)

        square_psf_list = sorted(square_psf, reverse=True)
        np.savez(f'psf_square in {file_settings_name}.npz', freqs = freqs.reverse(), psf_square = square_psf_list)

        print(f'Площади ДН: {square_psf_list}')

##############################################################
if __name__ == '__main__':
    alignment_sun_disk(method = method)