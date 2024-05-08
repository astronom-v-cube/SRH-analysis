from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import mplcursors
import matplotlib.colors as colors
import numpy as np
from astropy.io import fits
import os, sys, re
import shutil
import logging
from tqdm import tqdm
from analise_utils import ArrayOperations, Extract, Monitoring, OsOperations

Monitoring.start_log('log')
extract = Extract()
logging.info(f'Start program alignment of the solar disk')
# norm=TwoSlopeNorm(vmin=0, vcenter=vcenter) /// colors.Normalize(vmin=0, vcenter=vcenter))

##########
a = '20220120T055630'
directory = f"D:/datasets/20.01.22/times/{a}_calibrated_brightness"
mode = 'saved_settings'     # 'saved_settings' or 'interactive'
method = 'MAP' # 'MAP' /max around point/ or 'COM' /center of mass/
postfix = 'MAP_aligned' if method == 'MAP' else 'COM_aligned'
vcenter = 5000 
# x_limit = (350, 450)
# y_limit = (580, 680)

y_limit = (360, 410)
x_limit = (720, 770)
setting_of_alignes = []
##########
logging.info(f'Path to files: {directory}')

freqs = set()
files = sorted(os.listdir(directory), key=lambda x: extract.extract_number(x, freqs))
freqs = sorted(list(freqs), reverse = True)
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
    setting_of_alignes = np.load('setting_of_alignes.npy')

elif mode == 'interactive':

    coordinates_of_control_point = []
    coordinates_of_max_point_in_area = []
    square_psf = list()

    for i in iterable:
        
        hdul1 = fits.open(f'{directory}/{files[i]}', ignore_missing_simple=True)
        data1 = hdul1[0].data
        hdul2 = fits.open(f'{directory}/{files[i+1]}', ignore_missing_simple=True)
        data2 = hdul2[0].data
        
        header1 = hdul1[0].header
        psf_a = header1['PSF_ELLA']
        psf_b = header1['PSF_ELLB']
        square_psf.append(int(3.1415 * psf_a * psf_b))

        I = data1 + data2

        # Создание графика и отображение данных
        fig, ax = plt.subplots(figsize=(12, 12))
        vcenter = - 1500 * (i + 1) + 80000
        im = ax.imshow(I, origin='lower', cmap='plasma', extent=[0, I.shape[1], 0, I.shape[0]], norm=TwoSlopeNorm(vmin=0, vcenter=vcenter))
        contour_levels = np.linspace(I.min(), I.max(), 150)
        ax.contour(I, levels=contour_levels, colors='k')    #, linestyles='solid'
        ax.set_xlim(x_limit) if len(x_limit) != 0 else logging.info(f'Limits for X not found')
        ax.set_ylim(y_limit) if len(y_limit) != 0 else logging.info(f'Limits for Y not found')
        # mplcursors.cursor(hover=True)
        plt.title(f'{files[i] + files[i+1]}')
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
            Monitoring.logprint(f"For image {i+1} last double click coordinates: {str(click_coords[-1])}")
            coordinates_of_control_point.append(click_coords[-1])
            coordinates_of_control_point.append(click_coords[-1])
            plt.close()
            
        else:
            Monitoring.logprint(f"No double click coordinates recorded")
            sys.exit()

def alignment_sun_disk(files : list = files, method : str = 'search_max_in_area', area : int = None):
    
    # Загрузка первого файла и нахождение координат отличительного признака
    hdul1 = fits.open(f'{directory}/{files[0]}')
    img1 = hdul1[0].data  # данные первого изображения
    header1 = hdul1[0].header
    hdul2 = fits.open(f'{directory}/{files[1]}')
    img2 = hdul2[0].data  # данные второго изображения
    header2 = hdul2[0].header
    
    if mode == 'interactive':
        
        I = img1 + img2

        try:
            control_point_1 = (coordinates_of_control_point[0][0], coordinates_of_control_point[0][1])  # координаты признака на первом изображении
            
            if method == 'search_max_in_area':

                if int(np.sqrt(img1.size)) == 1024:
                    max_col, max_row, max_value = ArrayOperations.find_max_around_point(I, reversed(control_point_1), 14) if method == 'MAP' else ArrayOperations.find_center_of_mass(I, reversed(control_point_1), 14)

                elif int(np.sqrt(img1.size)) == 512:
                    max_col, max_row, max_value = ArrayOperations.find_max_around_point(I, reversed(control_point_1), 28) if method == 'MAP' else ArrayOperations.find_center_of_mass(I, reversed(control_point_1), 28)
                    
                control_point_1 = (max_row, max_col)
                coordinates_of_max_point_in_area.append(control_point_1)
                logging.info(f"Max - {max_value} in {control_point_1}")
                
        except Exception as err:
            Monitoring.logprint('The program is terminated due to lack of alignment data')
            Monitoring.logprint('err')
        
    # Определяем место для сохранения
    
    OsOperations.create_place(directory, postfix)

    # пересохранение первой пары файлов в новую директорию
    fits.writeto(f'{directory}_{postfix}/{files[0][:-4]}_{postfix}.fits', img1, overwrite=True, header=header1)
    logging.info(f"Image {1}: {files[0]} - saved")
    fits.writeto(f'{directory}_{postfix}/{files[1][:-4]}_{postfix}.fits', img2, overwrite=True, header=header2)
    logging.info(f"Image {2}: {files[1]} - saved")
    
    hdul1.close()
    hdul2.close()

    # Цикл для совмещения остальных файлов
    for i in tqdm(iterable[1:]):
        
        hdul1 = fits.open(f'{directory}/{files[i]}')
        img1 = hdul1[0].data  # данные первого изображения
        header1 = hdul1[0].header
        hdul2 = fits.open(f'{directory}/{files[i + 1]}')
        img2 = hdul2[0].data  # данные второго изображения
        header2 = hdul2[0].header
        
        if mode == 'interactive':
        
            I = img1 + img2
            
            hdul1.close()
            hdul2.close()
            try:
                control_point_2 = (coordinates_of_control_point[i][0], coordinates_of_control_point[i][1])  # координаты признака на текущем изображени
            except Exception as err:
                Monitoring.logprint('The program is terminated due to lack of alignment data')
                Monitoring.logprint(err)

            if method == 'search_max_in_area':

                if int(np.sqrt(img2.size)) == 1024:
                    max_col, max_row, max_value = ArrayOperations.find_max_around_point(I, reversed(control_point_2), 14)

                elif int(np.sqrt(img2.size)) == 512:
                    max_col, max_row, max_value = ArrayOperations.find_max_around_point(I, reversed(control_point_2), 28)
                    
                control_point_2 = (max_row, max_col)
                coordinates_of_max_point_in_area.append(control_point_2)
                logging.info(f"Max - {max_value} in {control_point_2}")

            # Нахождение горизонтального и вертикального сдвигов между изображениями
            dx = control_point_1[0] - control_point_2[0]
            dy = control_point_1[1] - control_point_2[1]
            setting_of_alignes.append((dx, dy))
            setting_of_alignes.append((dx, dy))

        elif mode == 'saved_settings':
            dx, dy = setting_of_alignes[i-2][0], setting_of_alignes[i-2][1]

        # Сдвигаем изображения
        img1 = np.roll(img1, dx, axis=1)
        img1 = np.roll(img1, dy, axis=0)
        img2 = np.roll(img2, dx, axis=1)
        img2 = np.roll(img2, dy, axis=0)

        # Сохранение выравненных изображений
        fits.writeto(f'{directory}_{postfix}/{files[i][:-4]}_{postfix}.fits', img1, overwrite=True, header=header1)
        logging.info(f"Image {i+2}: {files[i]} - saved")   
        fits.writeto(f'{directory}_{postfix}/{files[i+1][:-4]}_{postfix}.fits', img2, overwrite=True, header=header2)
        logging.info(f"Image {i+2}: {files[i+1]} - saved")      
        
    Monitoring.logprint('Finish program alignment of the solar disk')
    print('For more details: read file "logs.log"')
    
    if mode == 'interactive':
        np.save('setting_of_alignes.npy', setting_of_alignes)
        print(setting_of_alignes)

        square_psf_list = sorted(square_psf, reverse=True)
        np.savez('psf_square.npz', freqs = freqs, psf_square = square_psf_list)
    
        print(square_psf_list)

##############################################################
if __name__ == '__main__':
    alignment_sun_disk(method = 'search_max_in_area')