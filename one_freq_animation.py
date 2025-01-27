import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from analise_utils import OsOperations


path = '/mnt/data/15may0306/2800'
fits_files = OsOperations.abс_sorted_files_in_folder(path)

# Функция для чтения данных из файла .fits
def read_fits(file_path):
    with fits.open(f'{path}/{file_path}') as hdul:
        data = hdul[0].data
    return data


# Чтение данных из всех файлов
data_list = [read_fits(f) for f in fits_files[1::2]]

# Настройка matplotlib
fig, ax = plt.subplots()

im = ax.imshow(data_list[0], norm = matplotlib.colors.TwoSlopeNorm(vmin=0, vcenter=1e5, vmax=2.5e5), cmap='plasma', origin='lower')

# Функция обновления для анимации
def update(frame):
    im.set_array(data_list[frame])
    return [im]

# Настройка анимации
fps = 30  # Укажите желаемый fps
ani = FuncAnimation(fig, update, frames=len(data_list), blit=True)

# Отображение окна с анимацией
plt.show()

# Сохранение в формате MP4
ani.save('animation.mp4', writer='ffmpeg', fps=fps)

# Или сохранение в формате GIF
ani.save('animation.gif', writer='pillow', fps=fps)