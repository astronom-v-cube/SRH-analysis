import logging
import os, sys
import re
from scipy import constants
import scipy.optimize as opt
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

SMALL_SIZE = 10
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=6)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

logging.basicConfig(filename = 'logs.log',  filemode='a', level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.info(f'Start of the program to search intensity spectrum of a sun')

##########
directory36 = 'test_dataset_36'
directory612 = 'test_dataset_612'
##########

logging.info(f'Path to files 3-6: {directory36}')
logging.info(f'Path to files 6-12: {directory612}')

pattern = re.compile(r'(?<=[_.])\d{4,5}(?=[_.])')
freqs = set()

# функция для извлечения цифр из названия файла
def extract_number(filename):
    match = pattern.search(filename)
    freqs.add(int(match.group()))
    return int(match.group())

files36 = sorted(os.listdir(directory36), key=extract_number)
files612 = sorted(os.listdir(directory612), key=extract_number)
logging.info(f'Find {len(files36)} files 3-6')
logging.info(f'Find {len(files612)} files 6-12')
logging.info(f'List 3-6 files: \n {files36}')
logging.info(f'List 6-12 files: \n {files612}')
freqs = sorted(list(freqs))
logging.info(f'Working with freqs: {freqs}')

class ZirinTb():
    
    def fitFunc(self, f, A, B, C):
        return A + B*f + C*f**-1.8
    
    def __init__(self):
        self.frequency = np.array([1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2, 3.6, 4.2, 5.0, 5.8, 7.0, 8.2, 9.4, 10.6, 11.8, 13.2, 14.8, 16.4, 18.0]) # frequency [GHz]
        self.Tb = np.array([70.5, 63.8, 52.2, 42.9, 32.8, 27.1, 24.2, 21.7, 19.4, 17.6, 15.9, 14.1, 12.9, 12.2, 11.3, 11.0, 10.8, 10.8, 10.7, 10.3]) # brightness temperature [1e3K]
        self.guess = [1, 1, 1]
        self.fitTbParams, _ = opt.curve_fit(self.fitFunc, self.frequency, self.Tb, p0=self.guess)
        self.solarDiskRadius = np.deg2rad(900/3600)
        
    def getTbAtFrequency(self, f):
        return self.fitFunc(f, self.fitTbParams[0],self.fitTbParams[1],self.fitTbParams[2])
    
    def getSfuAtFrequency(self, f):
        return 2*constants.k*self.getTbAtFrequency(f)*1e3/(constants.c/(f*1e9))**2 * np.pi*self.solarDiskRadius**2 / 1e-22
    
zirin = ZirinTb()

def find_intensity_in_point(matrix : np.ndarray, point : tuple) -> float:
    """     
    The function returns the intensity value at a specific point. 
    The input arguments of the function are the matrix `matrix` (two-dimensional array) and the point `point` (coordinates).
    The function returns the intensity value in point.
    """
    x, y = point
    return matrix[y][x]

def find_intensity_in_four_point(matrix: np.ndarray, point: tuple) -> float:
    """
    The function returns the intensity value at a specific point and three neighboring points on the right and top.
    The input arguments of the function are the matrix `matrix` (two-dimensional array) and the point `point` (coordinates).
    The function returns the average of the intensity values at the four points.
    """
    x, y = point
    intensity_sum = matrix[y][x] + matrix[y][x+1] + matrix[y+1][x] + matrix[y+1][x+1]
    intensity_avg = intensity_sum / 4
    # почему тут делим а ниже нет???????????????????????????????????????????
    return intensity_avg

def find_intensity_in_nine_point(matrix: np.ndarray, point: tuple) -> float:
    """
    The function returns the intensity value at a specific point and three neighboring points on the right and top.
    The input arguments of the function are the matrix `matrix` (two-dimensional array) and the point `point` (coordinates).
    The function returns the average of the intensity values at the nine points.
    """
    x, y = point
    intensity_sum = matrix[y][x] + matrix[y+1][x-1] + matrix[y+1][x] + matrix[y+1][x+1] + matrix[y][x-1] + matrix[y][x+1] + matrix[y-1][x-1] + matrix[y-1][x] + matrix[y-1][x+1]
    intensity_avg = intensity_sum
    return intensity_avg

coordinates = (600, 750)

intensivity_list_R = []
intensivity_list_L = []

for list_of_files in [files36, files612]:
    
    for image_index in range(len(list_of_files)):

        try:
            r_or_l = re.search(r'(RCP|LCP|R|L)', str(list_of_files[image_index])).group()
            
        except AttributeError:
            logging.warning(f'{list_of_files[image_index]} - where is the polarization in the name? interrupting work...')
            sys.exit()
            
        # Считывание файлов
        if list_of_files == files36:
            directory = directory36
        elif list_of_files == files612:
            directory = directory612
        data = fits.open(f'{directory}/{list_of_files[image_index]}', ignore_missing_simple=True)
        img = data[0].data
        data.close()

        intensivity = find_intensity_in_nine_point(img, coordinates)
        logging.info(f'{list_of_files[image_index]} - {intensivity}')
        
        if r_or_l == 'RCP' or r_or_l == 'R':
            intensivity_list_R.append(intensivity)
        elif r_or_l == 'LCP' or r_or_l == 'L':
            intensivity_list_L.append(intensivity)

intensivity_list_L, intensivity_list_R = np.array(intensivity_list_L), np.array(intensivity_list_R)

# Конвертация яркостной температуры в плотность потока
flux_density_left = ((2 * 1.38*1e-16 * (np.array(freqs) * 1e6) ** 2) / (3e10)**2) * intensivity_list_L * ((2.4/3600*0.01745)**2) * 1e19
flux_density_right = ((2 * 1.38*1e-16 * (np.array(freqs) * 1e6) ** 2) / (3e10)**2) * intensivity_list_R * ((2.4/3600*0.01745)**2) * 1e19

logging.info(f'Start flux in s.f.u for LCP: {flux_density_left}')
logging.info(f'Start flux in s.f.u for RCP: {flux_density_right}')

flags = []

# flux_density_left_new, flux_density_right_ne, {flux_density_right}w, freqs_new = [], [], []

# # вычет подложки
# for idx, freq in enumerate(freqs):
#     if freq not in flags:
#         freqs_new.append(freq)
#         flux_density_left_new.append(flux_density_left[idx] - zirin.getSfuAtFrequency(freq/1000))
#         flux_density_right_new.append(flux_density_right[idx]- zirin.getSfuAtFrequency(freq/1000))

# flux_density_left = np.array(flux_density_left_new)
# flux_density_right = np.array(flux_density_right_new)
# freqs = np.array(freqs_new)

flux_density_left = np.array(flux_density_left)
flux_density_right = np.array(flux_density_right)
freqs = np.array(freqs)

flux_density = np.concatenate((flux_density_left, flux_density_right), axis=0)

p_left = np.polyfit(freqs, flux_density_left, 3)        # создание полинома - апроксимация
ya_left = np.polyval(p_left, freqs)        # координаты y для полинома

p_right = np.polyfit(freqs, flux_density_right, 3)        # создание полинома - апроксимация
ya_right = np.polyval(p_right, freqs)        # координаты y для полинома

# def running_mean(data, window_size):
#     window = np.ones(window_size) / window_size
#     smoothed_data = np.convolve(data, window, mode='same')
#     return smoothed_data

# def running_mean(data, window_size):
#     smoothed_data = []
#     for i in range(len(data)):
#         if i == 0:
#             smoothed_data.append((data[i] + data[i+1]) / 2)
#         elif i == len(data) - 1:
#             smoothed_data.append((data[i-1] + data[i]) / 2)
#         else:
#             smoothed_data.append((data[i-1] + data[i] + data[i+1]) / 3)
#     return smoothed_data

# new_y_1 = running_mean(flux_density_left, window_size=3)
# new_y_2 = running_mean(flux_density_right, window_size=3)

logging.info(f'Flux in s.f.u for LCP minus disk: {flux_density_left}')
logging.info(f'Flux in s.f.u for RCP minus disk: {flux_density_right}')
logging.info(f'Finish flux in s.f.u for LCP - polifit: {ya_left}')
logging.info(f'Finish flux in s.f.u for RCP - polifit: {ya_right}')

# создаем subplots
fig, axs = plt.subplots(1, 2)

plot_freqs = freqs/1000

# первый график для четных чисел
axs[0].plot(plot_freqs, flux_density_left)
axs[0].plot(plot_freqs, ya_left, linestyle = '--')
axs[0].set_title('LCP')
axs[0].grid(True)
axs[0].set_xlim(np.min(plot_freqs), np.max(plot_freqs))
# axs[0].set_ylim(3.5, 20)
axs[0].set_ylim(np.min(flux_density) - np.min(flux_density) * 0.1, np.max(flux_density) + np.max(flux_density) * 0.1)
axs[0].set_xlabel('Index')
axs[0].set_ylabel('Value')
axs[0].set_yscale('log')
# axs[0].set_xscale('log')

# второй график для нечетных чисел
axs[1].plot(plot_freqs, flux_density_right)
axs[1].plot(plot_freqs, ya_right, linestyle = '--')
# axs[1].scatter(plot_freqs, ya_right, s = 40)
axs[1].set_title('RCP')
axs[1].grid(True)
axs[1].set_xlim(np.min(plot_freqs), np.max(plot_freqs))
# axs[1].set_ylim(3.5, 20)
axs[1].set_ylim(np.min(flux_density) - np.min(flux_density) * 0.1, np.max(flux_density) + np.max(flux_density) * 0.1)
axs[1].set_xlabel('Index')
axs[1].set_ylabel('Value')
axs[1].set_yscale('log')
# axs[1].set_xscale('log')

axs[0].xaxis.set_ticks(plot_freqs)
axs[1].xaxis.set_ticks(plot_freqs)

plt.show()