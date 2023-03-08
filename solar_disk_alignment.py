import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from astropy.io import fits
import os, sys
import shutil

import logging
logging.basicConfig(filename = 'logs.log',  filemode='a', level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.info(f'Start program alignment of the solar disk')

##########
directory = 'test_dataset'
##########
logging.info(f'Path to files: {directory}')

files = sorted(os.listdir(directory))
logging.info(f'Find {len(files)} files')
logging.info(f'List files: \n {files}')

coordinates_of_distinguishing_feature = []

for i in range(0, len(files)):
    # Считывание файлов
    data = fits.open(f'{directory}/{files[i]}', ignore_missing_simple=True)[0].data

    # Создание графика и отображение данных
    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(data, origin='lower', cmap='plasma', extent=[0, data.shape[1], 0, data.shape[0]])
    fig.colorbar(im)
    # fig.tight_layout()
    # origin='lower'- расположение начал координат – точки [0,0]

    # Создание слайдера для редактирования границ цветовой шкалы
    ax_slider = plt.axes([0.25, 0.03, 0.5, 0.01], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Threshold', im.norm.vmin, im.norm.vmax, valinit=im.norm.vmax/2)

    # Обновление цветовой шкалы и изображения при изменении положения ползунка
    def update(val):
        vmin = im.norm.vmin
        vmax = slider.val
        im.set_clim(vmin, vmax)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Список для хранения координат кликов пользователя
    click_coords = []

    # Функция для обработки событий клика мыши
    def onclick(event):
        if event.dblclick:
            # Очистка списка координат кликов
            click_coords.clear()
            # Добавление координат клика в список
            click_coords.append((int(event.xdata), int(event.ydata)))

    # Привязка обработчика событий клика мыши к графику
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Отображение графика
    plt.show()

    # Вывод последней координаты двойного клика пользователя
    if len(click_coords) > 0:
        print(f"For image {i+1} last double click coordinates: {str(click_coords[-1])}")
        logging.info(f"For image {i+1} last double click coordinates: {str(click_coords[-1])}")
        coordinates_of_distinguishing_feature.append(click_coords[-1])
    else:
        print("No double click coordinates recorded")
        logging.info(f"No double click coordinates recorded")
        sys.exit()

# Загрузка первого файла и нахождение координат отличительного признака
hdul1 = fits.open(f'{directory}/{files[0]}')
img1 = hdul1[0].data  # данные первого изображения
try:
    kp1 = (coordinates_of_distinguishing_feature[0][0], coordinates_of_distinguishing_feature[0][1])  # координаты признака на первом изображении
except:
    print('The program is terminated due to lack of alignment data')
    logging.info('The program is terminated due to lack of alignment data')

hdul1.close()

# Создание нового массива пикселей
result = np.zeros_like(img1)

# Определяем место для сохранения
try:
    os.mkdir('aligned')
except:
    shutil.rmtree('aligned')
    os.mkdir('aligned')

# Цикл для совмещения остальных файлов
for i, file in enumerate(files[1:]):
    # Загрузка файла и нахождение координат отличительного признака
    hdul2 = fits.open(f'{directory}/{file}') 
    img2 = hdul2[0].data
    try:
        kp2 = (coordinates_of_distinguishing_feature[i+1][0], coordinates_of_distinguishing_feature[i+1][1])  # координаты признака на текущем изображени
    except:
        print('The program is terminated due to lack of alignment data')
        logging.info('The program is terminated due to lack of alignment data')

    hdul2.close()

    # Нахождение горизонтального и вертикального сдвигов между изображениями
    dx = kp1[0] - kp2[0]
    dy = kp1[1] - kp2[1]

    # Сдвигаем изображение
    img2 = np.roll(img2, dx, axis=1)
    img2 = np.roll(img2, dy, axis=0)

    # Сохранение выровненного изображения
    fits.writeto(f'aligned/{file[:-4]}_aligned.fits', img2, overwrite=True)
    logging.info(f"Image {i+1}: {file} - saved")

print('Finish program alignment of the solar disk')
print('For more details: read file "logs.log"')
logging.info(f'Finish program alignment of the solar disk')