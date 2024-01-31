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
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

logging.basicConfig(filename = '230122_logs.log',  filemode='a', level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.info(f'Start of the program to search intensity spectrum of a sun')

##########
directory = "D:/datasets/20.01.22/times/20220120T055430_calibrated_brightness_aligned"
psf_calibration = True
##########

psf_npz_file = np.load('psf_square.npz')
freqs_npz = sorted(psf_npz_file['freqs'])
psf_square = psf_npz_file['psf_square']
# psf_square = np.array([11.1460177 ,  9.72418879,  8.5339233 ,  7.57227139,  6.74631268, 6.04867257,  5.46902655,  4.96902655,  4.49115044,  4.12831858, 3.5280236 ,  3.23156342,  3.11651917,  2.99852507,  2.78466077, 2.76548673,  2.59734513,  2.47640118,  2.22713864,  2.01917404, 1.83185841,  1.70943953,  1.55899705,  1.42625369,  1.32300885, 1.21976401,  1.13126844,  1.0560472, 1]) # 16.07.23
psf_square = np.array([1.75, 1.7, 1.65, 1.6, 1.55, 1.5, 1.45, 1.4, 1.35, 1.3, 1.2, 1.15, 1.1, 1.05, 1])  # 20.01.22 (9800 = 1.25, 10200 = 1.2)

logging.info(f'Path to files: {directory}')

pattern = re.compile(r'(?<=[_.])\d{4,5}(?=[_.])')
freqs = set()

# функция для извлечения цифр из названия файла
def extract_number(filename):
    match = pattern.search(filename)
    freqs.add(int(match.group()))
    return int(match.group())

files = sorted(os.listdir(directory), key=extract_number)
logging.info(f'Find {len(files)} files')
logging.info(f'List files: \n {files}')
freqs = np.array(sorted(list(freqs)))
# if (freqs_npz != freqs).any():
#     logging.warning(f"Freqs in npz is not freqs in directory")
#     sys.exit()
        
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
    x, y = point
    return matrix[y][x]

def find_intensity_in_four_point(matrix: np.ndarray, point: tuple) -> float:
    x, y = point
    intensity_sum = matrix[y][x] + matrix[y][x+1] + matrix[y+1][x] + matrix[y+1][x+1]
    intensity_avg = intensity_sum
    return intensity_avg

def find_intensity_in_nine_point(matrix: np.ndarray, point: tuple) -> float:
    x, y = point
    intensity_sum = matrix[y+1][x-1] + matrix[y+1][x] + matrix[y+1][x+1] + matrix[y][x-1] +  matrix[y][x] + matrix[y][x+1] + matrix[y-1][x-1] + matrix[y-1][x] + matrix[y-1][x+1]
    intensity_avg = intensity_sum
    return intensity_avg

def find_intensity_in_alotof_point(matrix: np.ndarray, point: tuple) -> float:
    x, y = point
    data_array = matrix[y-80:y+80, x-80:x+80]
    #fig = plt.figure()
    #plt.imshow(data_array)
    #plt.plot()
    intensity_avg = np.sum(data_array)
    return intensity_avg

def find_intensity_in_sun_disk(matrix: np.ndarray, point: tuple) -> float:
    x1, y1 = np.indices(matrix.shape)
    center_x, center_y = 512, 512
    radius = 350

    # вычисляем расстояние от каждой точки до центра окружности
    distance_from_center = np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)

    # создаем маску, где значения True соответствуют точкам, входящим в окружность
    mask = distance_from_center <= radius

    # вырезаем данные внутри круга
    data_inside_circle = img[mask]
    print(len(data_inside_circle))
    intensity_avg = np.sum(data_inside_circle)
    return intensity_avg

# coordinates = (324, 628)
coordinates = (890, 563)
# coordinates = (650, 512)

intensivity_list_R = []
intensivity_list_L = []

for image_index, image_name in enumerate(files):

    try:
        r_or_l = re.search(r'(RCP|LCP|R|L)', str(files[image_index])).group()
        
    except AttributeError:
        logging.warning(f'{files[image_index]} - where is the polarization in the name? interrupting work...')
        sys.exit()
        
    # Считывание файлов
    data = fits.open(f'{directory}/{files[image_index]}', ignore_missing_simple=True)
    img = data[0].data
    data.close()

    intensivity = find_intensity_in_nine_point(img, coordinates)
    logging.info(f'{files[image_index]} - {intensivity}')
    
    if r_or_l == 'RCP' or r_or_l == 'R':
        intensivity_list_R.append(intensivity)
    elif r_or_l == 'LCP' or r_or_l == 'L':
        intensivity_list_L.append(intensivity)

intensivity_list_L, intensivity_list_R = np.array(intensivity_list_L), np.array(intensivity_list_R)

# flux_density_left, flux_density_right = [], []

# # вычет подложки
# for index, freq in enumerate(freqs):
#     flux_density_left.append(intensivity_list_L[index] - zirin.getTbAtFrequency(freq/1000))
#     flux_density_right.append(intensivity_list_R[index]- zirin.getTbAtFrequency(freq/1000))

# Конвертация яркостной температуры в плотность потока
flux_density_left = ((2 * 1.38*1e-16 * (np.array(freqs) * 1e6) ** 2) / (3e10)**2) * intensivity_list_L * ((2.4/3600*0.01745)**2) * 1e19
flux_density_right = ((2 * 1.38*1e-16 * (np.array(freqs) * 1e6) ** 2) / (3e10)**2) * intensivity_list_R * ((2.4/3600*0.01745)**2) * 1e19

logging.info(f'Start flux in s.f.u for LCP: [{", ".join(flux_density_left.astype(str))}]')
logging.info(f'Start flux in s.f.u for RCP: [{", ".join(flux_density_right.astype(str))}]')

correction_psf_left = flux_density_left * psf_square
correction_psf_right = flux_density_right * psf_square

logging.info(f'Correction from psf flux in s.f.u for LCP: [{", ".join(correction_psf_left.astype(str))}]')
logging.info(f'Correction from psf flux in s.f.u for RCP: [{", ".join(correction_psf_right.astype(str))}]')

flux_for_graph_RL = np.concatenate((flux_density_left, flux_density_right, correction_psf_left, correction_psf_right), axis=0)
flux_for_graph_I = np.concatenate((flux_density_left + flux_density_right, correction_psf_left + correction_psf_right), axis=0)

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

# new_y_1 = running_mean(flux_density_left, window_size=5)
# new_y_2 = running_mean(flux_density_right, window_size=5)

correction_quate_sun_left = [1.03838439, 0.95999654, 1.09428362, 0.97976959, 0.9899581, 0.87707643, 0.9681845, 0.98712561, 1.03482719, 1.06164441, 1.06819119, 1.02623596, 0.97603417, 1.00623033, 0.95329644]

correction_quate_sun_right = [1.03798513, 1.01688483, 1.04706266, 0.94311813, 0.94967881, 0.9540217, 0.94383052, 1.02094715, 0.99510278, 1.05711966, 1.10484654, 0.96657375, 1.07459668, 0.96531071, 0.94302758]

p_left = np.polyfit(freqs, np.log(flux_density_left * correction_quate_sun_left), 2)
ya_left = np.exp(np.polyval(p_left, freqs))
p_right = np.polyfit(freqs, np.log(flux_density_right * correction_quate_sun_right), 2)
ya_right = np.exp(np.polyval(p_right, freqs))

correction_psf_p_left = np.polyfit(freqs, np.log(correction_psf_left), 2)
correction_psf_ya_left = np.exp(np.polyval(correction_psf_p_left, freqs))
correction_psf_p_right = np.polyfit(freqs, np.log(correction_psf_right), 2)
correction_psf_ya_right = np.exp(np.polyval(correction_psf_p_right, freqs))

# logging.info(f'Flux in s.f.u for LCP minus disk: {flux_density_left}')
# logging.info(f'Flux in s.f.u for RCP minus disk: {flux_density_right}')

logging.info(f'Finish flux in s.f.u for LCP - polifit: [{", ".join(ya_left.astype(str))}]')
logging.info(f'Finish flux in s.f.u for RCP - polifit: [{", ".join(ya_right.astype(str))}]')
logging.info(f'Finish flux with correction from psf in s.f.u for LCP - polifit: [{", ".join(correction_psf_ya_left.astype(str))}]')
logging.info(f'Finish flux with correction from psf in s.f.u for RCP - polifit: [{", ".join(correction_psf_ya_right.astype(str))}]')

# создаем subplots
fig, axs = plt.subplots(1, 2, num="L and R polarization")

plot_freqs = freqs/1000
fig.suptitle(f'{directory[9:24]}')

# первый график для четных чисел
# axs[0].plot(plot_freqs, flux_density_left)
axs[0].plot(plot_freqs, flux_density_left, 'D', label = f"Наблюдаемый спектр", linewidth = 8, color = 'darkblue', markersize=10, zorder = 2)
axs[0].plot(plot_freqs, ya_left, linestyle = '--', linewidth = 4, zorder = 1)
if psf_calibration == True:
    axs[0].plot(plot_freqs, correction_psf_ya_left, linestyle = '--', linewidth = 4, zorder = 1)
    axs[0].plot(plot_freqs, correction_psf_left, 'o', label = f"Наблюдаемый спектр\nс поправкой на ширину\nдиаграммы направленности", linewidth = 8, color = 'firebrick', markersize=10, zorder = 2)
axs[0].set_title('LCP', fontsize = 16)
axs[0].grid(True)
axs[0].set_xlim(np.min(plot_freqs) - np.min(plot_freqs) * 0.09, np.max(plot_freqs) + np.min(plot_freqs) * 0.09)
axs[0].set_ylim(np.min(flux_for_graph_RL) - np.min(flux_for_graph_RL) * 0.2, np.max(flux_for_graph_RL) + np.max(flux_for_graph_RL) * 0.2)
axs[0].set_xlabel('Frequency, $GHz$', fontsize = 16)
axs[0].set_ylabel('Flux, $S.F.U.$', fontsize = 16)
axs[0].set_yscale('log')
# axs[0].set_xscale('log')
axs[0].legend(fontsize = 16)

# new_ax = axs[0].twinx()
# new_ax.plot(plot_freqs, flux_density_left - ya_left, color = 'red')

# второй график для нечетных чисел
axs[1].plot(plot_freqs, flux_density_right, 'D', label = f"Наблюдаемый спектр", linewidth = 8, color = 'darkblue', markersize=10, zorder = 2)
axs[1].plot(plot_freqs, ya_right, linestyle = '--', linewidth = 4, zorder = 1)
if psf_calibration == True:
    axs[1].plot(plot_freqs, correction_psf_ya_right, linestyle = '--', linewidth = 4, zorder = 1)
    axs[1].plot(plot_freqs, correction_psf_right, 'o', label = f"Наблюдаемый спектр\nс поправкой на ширину\nдиаграммы направленности", linewidth = 8, color = 'firebrick', markersize=10, zorder = 2)
axs[1].set_title('RCP', fontsize = 16)
axs[1].grid(True)
axs[1].set_xlim(np.min(plot_freqs) - np.min(plot_freqs) * 0.09, np.max(plot_freqs) + np.min(plot_freqs) * 0.09)
axs[1].set_ylim(np.min(flux_for_graph_RL) - np.min(flux_for_graph_RL) * 0.2, np.max(flux_for_graph_RL) + np.max(flux_for_graph_RL) * 0.2)
axs[1].set_xlabel('Frequency, $GHz$', fontsize = 16)
axs[1].set_ylabel('Flux, $S.F.U.$', fontsize = 16)
axs[1].set_yscale('log')
# axs[1].set_xscale('log')
axs[1].legend(fontsize = 16)

axs[0].xaxis.set_ticks(plot_freqs)
axs[0].set_xticklabels(plot_freqs, rotation=45, ha='right', fontsize = 10)
axs[1].xaxis.set_ticks(plot_freqs)
axs[1].set_xticklabels(plot_freqs, rotation=45, ha='right', fontsize = 10)

fig_I = plt.figure(num="Intensity")
ax = plt.gca()
ax.plot(plot_freqs, flux_density_left + flux_density_right, 'D', label = f"Наблюдаемый спектр", linewidth = 8, color = 'darkblue', markersize=10, zorder = 2)
ax.plot(plot_freqs, ya_left + ya_right, linestyle = '--', linewidth = 4, zorder = 1)
ax.plot(plot_freqs, correction_psf_ya_left + correction_psf_ya_right, linestyle = '--', linewidth = 4, zorder = 1)
ax.plot(plot_freqs, correction_psf_left + correction_psf_right, 'o', label = f"Наблюдаемый спектр\nс поправкой на ширину\nдиаграммы направленности", linewidth = 8, color = 'firebrick', markersize=10, zorder = 2)
ax.set_ylim(np.min(flux_for_graph_I) - np.min(flux_for_graph_I) * 0.2, np.max(flux_for_graph_I) + np.max(flux_for_graph_I) * 0.2)
ax.set_xlabel('Frequency, $GHz$', fontsize = 16)
ax.set_ylabel('Flux, $S.F.U.$', fontsize = 16)
ax.grid(True)
ax.set_yscale('log')
ax.legend(fontsize = 16)

ax.xaxis.set_ticks(plot_freqs)
ax.set_xticklabels(plot_freqs, rotation=45, ha='right', fontsize = 10)

plt.tight_layout()
plt.show()