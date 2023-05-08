import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from astropy.io import fits
import os, sys
import shutil

import logging
logging.basicConfig(filename = 'logs.log',  filemode='a', level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.info(f'Start of the program to search for the brightness temperature of a calm sun')

##########
directory = 'test_dataset_2'
##########
logging.info(f'Path to files: {directory}')

files = sorted(os.listdir(directory))
logging.info(f'Find {len(files)} files')
logging.info(f'List files: \n {files}')

# Обновление цветовой шкалы и изображения при изменении положения ползунка
def update(val):
    vmin = im.norm.vmin
    vmax = slider.val
    im.set_clim(vmin, vmax)
    fig.canvas.draw_idle()

# Функция для обработки событий клика мыши
def onclick(event):
    if event.dblclick:
        # Добавление координат клика в список
        click_coords.append((int(event.xdata), int(event.ydata)))

def find_intensity_in_point(matrix : np.ndarray, point : tuple) -> float:
    """     
    The function returns the intensity value at a specific point. 
    The input arguments of the function are the matrix `matrix` (two-dimensional array) and the point `point` (coordinates).
    The function returns three values: x, y and the intensity value in them.
    """
    x, y = point
    return x, y, matrix[y][x]


for image in range(len(files)):
    # Считывание файлов
    data = fits.open(f'{directory}/{files[image]}', ignore_missing_simple=True)
    header = data[0].header
    img = data[0].data
    data.close()

    # Создание графика и отображение данных
    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(img, origin='lower', cmap='plasma', extent=[0, img.shape[1], 0, img.shape[0]])
    fig.colorbar(im)
    # fig.tight_layout()

    # Создание слайдера для редактирования границ цветовой шкалы
    ax_slider = plt.axes([0.25, 0.03, 0.5, 0.01], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Threshold', im.norm.vmin, im.norm.vmax, valinit=im.norm.vmax/2)
    slider.on_changed(update)

    # Список для хранения координат кликов пользователя
    click_coords = []

    # Привязка обработчика событий клика мыши к графику
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

    intensivity_list = []

    # Вывод последней координаты двойного клика пользователя
    if len(click_coords) > 0:
        logging.info(f"For image {image+1} double click coordinates: {str(click_coords)}")
        for i in click_coords:
            x, y, intensivity = find_intensity_in_point(img, i)
            intensivity_list.append(intensivity)
        logging.info(f"For image {image+1} double click intensivity: {str(intensivity_list)}")
        mean_intensivity = np.mean(intensivity_list)
        logging.info(f"For image {image+1} mean intensivity: {str(mean_intensivity)}")
        correction_factor_brightness = 1/(mean_intensivity/30000)
        logging.info(f"For image {image+1} correction factor: {str(correction_factor_brightness)}")

        # Определяем место для сохранения
        try:
            os.mkdir('calibrated_brightness')
        except:
            shutil.rmtree('calibrated_brightness')
            os.mkdir('calibrated_brightness')

        fits.writeto(f'calibrated_brightness/{files[image][:-4]}_calibrated_brightness.fits', img*correction_factor_brightness, overwrite=True, header=header)
        logging.info(f"Image {image+1}: {files[image]} - saved")

    else:
        print("No double click coordinates recorded")
        logging.info(f"No double click coordinates recorded")
        sys.exit()




##############################################################
if __name__ == '__main__':
    pass