from cProfile import label
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from tqdm import tqdm
import logging, os
from scipy.ndimage import zoom
import concurrent.futures
matplotlib.use('Agg')

from analise_utils import Extract, Monitoring, OsOperations, ArrayOperations, ConvertingArrays

Monitoring.start_log('dreif')
logging.info(f'Start')

# freq_list = [12200,14480,15240,16000,16760,17520,18280,19040,19800,20560,21320,22080,23400]
# freq_list = [6000,6400,6800,7200,7600,8000,8400,8800,9200,9600,10000,10400,10800,11200,11600,12000]
freq_list = [11200]
directory = "D:/14may_new_612"
speeds_x, speeds_y = [], []

def calculate_freif(freq_number, speeds_x, speeds_y):

    logging.info(f'Частота {freq_number}')

    xxx, yyy = [], []


    files_on_freq_folder = OsOperations.abс_sorted_files_in_folder(f'{directory}/{freq_number}')

    for index in tqdm(range(0, len(files_on_freq_folder), 2), desc='Анализ файлов', leave=False):
        hdul1 = fits.open(f'{directory}/{freq_number}/{files_on_freq_folder[index]}', ignore_missing_simple=True) # LCP
        hdul2 = fits.open(f'{directory}/{freq_number}/{files_on_freq_folder[index+1]}', ignore_missing_simple=True) # RCP

        data1 = hdul1[0].data
        data2 = hdul2[0].data

        I = data1 + data2
        scale_factor = 1
        if scale_factor>1:
            I = zoom(I, scale_factor, order=3)

        max_coord = np.unravel_index(np.argmax(I), I.shape)
        try:
            max_index = ArrayOperations.calculate_weighted_centroid(I, (max_coord[1], max_coord[0]), 0.15, ((700*scale_factor, 1000*scale_factor), (300*scale_factor, 500*scale_factor)))
        except:
            print(f'Что-то не так с данными {index}')
            max_index = (0, 0)
        plt.imshow(I,  cmap='plasma', origin='lower')
        plt.contour(I, levels=[I.max() * 0.15])
        plt.ylim(300*scale_factor, 500*scale_factor)
        plt.xlim(700*scale_factor, 1000*scale_factor)
        plt.scatter(max_coord[1], max_coord[0], marker='D', s=30, c='g', zorder=6)
        plt.scatter(max_coord[1], max_coord[0], marker='+', s=30, c='k', zorder=7)
        plt.tight_layout()
        try:
            os.mkdir(f'dreif_2/{freq_number}')
        except:
            pass
        plt.savefig(f'dreif_2/{freq_number}/{index}.png', dpi=300)
        plt.close()

        # max_index = np.unravel_index(np.argmax(I), I.shape)
        xxx.append(max_index[0])
        yyy.append(max_index[1])

    xxx = ConvertingArrays.variable_running_mean(np.array(xxx))
    np.savetxt(f'dreif_2/xxx_{freq_number}.txt', np.array(xxx))
    yyy = ConvertingArrays.variable_running_mean(np.array(yyy))
    np.savetxt(f'dreif_2/yyy_{freq_number}.txt', np.array(yyy))

    # xxx = np.loadtxt('xxx.txt')
    # yyy = np.loadtxt('yyy.txt')

    xxx = (xxx - xxx[0])/scale_factor
    yyy = (yyy - yyy[0])/scale_factor

    p_x = np.polyfit(range(len(xxx)), xxx, 1)				# создание полинома первой степени - апроксимация
    ya_x = np.polyval(p_x, range(len(xxx)))
    k = p_x[0]  # Коэффициент наклона
    b = p_x[1]  # Свободный член
    logging.info(f'X: k={k} b={b}')
    logging.info(f'Скорость движения {((ya_x[-1]-ya_x[0])/len(ya_x))} пикселей на кадр (в 3.5 секунды)')
    speeds_x.append(((ya_x[-1]-ya_x[0])/len(ya_x)))

    p_y = np.polyfit(range(len(yyy)), yyy, 1)				# создание полинома первой степени - апроксимация
    ya_y = np.polyval(p_y, range(len(yyy)))
    k = p_y[0]  # Коэффициент наклона
    b = p_y[1]  # Свободный член
    logging.info(f'Y: k={k} b={b}')
    logging.info(f'Скорость движения {((ya_y[-1]-ya_y[0])/len(ya_y))} пикселей на кадр (в 3.5 секунды)')
    speeds_y.append(((ya_y[-1]-ya_y[0])/len(ya_y)))

    # matplotlib.use('Qt5Agg')

    plt.title(f'{freq_number}')
    plt.plot(xxx, c='b', linewidth=3)
    plt.plot(ya_x, c='b', linewidth=3, linestyle='--', label='X')
    # plt.ylabel('Ось Х (синяя)')
    # plt.twinx()
    plt.plot(yyy, c='r', linewidth=3)
    plt.plot(ya_y, c='r', linewidth=3, linestyle='--', label='Y')
    # plt.ylabel('Ось Y (красная)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'dreif_2/{freq_number}.png', dpi=300)

for f in freq_list:
    calculate_freif(str(f), speeds_x, speeds_y)

np.savetxt(f'dreif_2/speeds_x.txt', speeds_x)
np.savetxt(f'dreif_2/speeds_y.txt', speeds_y)
plt.close()
plt.plot(speeds_x)
plt.plot(speeds_y)
plt.grid(True)
plt.savefig(f'dreif_2/speeds')
plt.show()