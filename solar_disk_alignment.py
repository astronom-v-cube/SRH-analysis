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

def logprint(msg):
    print(msg)
    logging.info(msg)
    
logging.basicConfig(filename = 'logs.log',  filemode='a', level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.info(f'Start program alignment of the solar disk')
# norm=TwoSlopeNorm(vmin=0, vcenter=vcenter)

##########
directory = '08-26-46'
flag = 'IV'
flag = 'RL'
vcenter = 25000
x_limit = (600, 750)
y_limit = (520, 700)

x_limit = (350, 450)
y_limit = (580, 680)
##########
logging.info(f'Path to files: {directory}')

pattern = re.compile(r'(?<=[_.])\d{4,5}(?=[_.])')
freqs = set()

# функция для извлечения цифр из названия файла
def extract_number(filename):
    match = pattern.search(filename)
    freqs.add(int(match.group()))
    return int(match.group())

files = sorted(os.listdir(directory), key=extract_number)
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

    hdul1.close()

    # Определяем место для сохранения
    try:
        os.mkdir(f'{directory}_aligned')
    except:
        shutil.rmtree(f'{directory}_aligned')
        os.mkdir(f'{directory}_aligned')

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
            logprint('The program is terminated due to lack of alignment data')

        if method == 'search_max_in_area':

            if int(np.sqrt(img2.size)) == 1024:
                max_col, max_row, max_value = find_max_around_point(img2, reversed(control_point_2), 24)

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

            # Сохранение выравненного изображения
            fits.writeto(f'{directory}_aligned/{file[:-4]}_aligned.fits', img2, overwrite=True, header=header)
            logging.info(f"Image {i+2}: {file} - saved")          

        elif method == 'linear_image_shift':
            
            # Нахождение горизонтального и вертикального сдвигов между изображениями
            dx = control_point_1[0] - control_point_2[0]
            dy = control_point_1[1] - control_point_2[1]

            # Сдвигаем изображение
            img2 = np.roll(img2, dx, axis=1)
            img2 = np.roll(img2, dy, axis=0)

            # Сохранение выравненного изображения
            fits.writeto(f'{directory}_aligned/{file[:-4]}_aligned.fits', img2, overwrite=True, header=header)
            logging.info(f"Image {i+2}: {file} - saved")

    logprint('Finish program alignment of the solar disk')
    print('For more details: read file "logs.log"')
    
    np.save('setting_of_alignes.npy', coordinates_of_max_point_in_area)
    print(coordinates_of_max_point_in_area)

    square_psf_list = sorted(square_psf, reverse=True)
    np.savez('psf_square.npz', freqs = freqs, psf_square = square_psf_list)
    
    print(square_psf_list)

##############################################################
if __name__ == '__main__':
    alignment_sun_disk(method = 'search_max_in_area')