import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from astropy.io import fits
import os, sys, re
import shutil
from scipy import constants, stats
import scipy.optimize as opt

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 25

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

import logging
logging.basicConfig(filename = 'logs.log',  filemode='a', level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.info(f'Start of the program to search for the brightness temperature of a calm sun')

##########
directory = "D:/datasets/20.01.22/times/20220120T055430"
mode = 'saved_settings'     # 'saved_settings' or 'calculation'
##########

logging.info(f'Path to files: {directory}')

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

frequency = 2.4 # GHz
Tb = zirin.getTbAtFrequency(frequency) # получить яркостную температуру на частоте 2.4 GHz
Sfu = zirin.getSfuAtFrequency(frequency) # получить значение яркости на частоте 2.4 GHz в единицах SFU

pattern = re.compile(r'(?<=[_.])\d{4,5}(?=[_.])')
correction_factor_brightness_array = []

# функция для извлечения цифр из названия файла
def extract_number(filename):
    match = pattern.search(filename)
    return int(match.group())

files = sorted(os.listdir(directory), key=extract_number)

logging.info(f'Find {len(files)} files')
logging.info(f'List files: \n {files}')

# Определяем место для сохранения
try:
    os.mkdir(f'{directory}_calibrated_brightness')
except:
    shutil.rmtree(f'{directory}_calibrated_brightness')
    os.mkdir(f'{directory}_calibrated_brightness')

for index, image in enumerate(files):
    
    if index % 2 != 0:
        pass
    
    elif index % 2 == 0:    

        # Считывание файлов
        data1 = fits.open(f'{directory}/{files[index]}', ignore_missing_simple=True)
        data2 = fits.open(f'{directory}/{files[index + 1]}', ignore_missing_simple=True)
        print(files[index] + '  ' + files[index + 1])
        
        header1 = data1[0].header
        header2 = data2[0].header
        
        img1 = data1[0].data
        img2 = data2[0].data
        data1.close()
        data2.close()

        current_frequency = int(re.search(r'(?<=[_.])\d{4,5}(?=[_.])', str(files[index])).group())
        Tb = zirin.getTbAtFrequency(current_frequency/1000) # получить яркостную температуру на частоте

        if mode == 'saved_settings':
            
            correction_factor_brightness_array = np.load(f'correction_factor_{directory[-15:-7]}.npy')
            correction_factor_brightness = correction_factor_brightness_array[index//2]
            
            fits.writeto(f'{directory}_calibrated_brightness/{files[index][:-4]}_calibrated_brightness.fits', img1 * correction_factor_brightness, overwrite=True, header=header1)
            logging.info(f"Image {index+1}: {files[index]} - saved")
            fits.writeto(f'{directory}_calibrated_brightness/{files[index + 1][:-4]}_calibrated_brightness.fits', img2 * correction_factor_brightness, overwrite=True, header=header2)
            logging.info(f"Image {index+2}: {files[index + 1]} - saved")
        
        elif mode == 'calculation':

            # создаем массивы координат пикселей
            x1, y1 = np.indices(img1.shape)
            x2, y2 = np.indices(img2.shape)

            # определяем параметры окружности
            center_x, center_y = 512, 512
            radius = 350

            # вычисляем расстояние от каждой точки до центра окружности
            distance_from_center1 = np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)
            distance_from_center2 = np.sqrt((x2 - center_x)**2 + (y2 - center_y)**2)

            # создаем маску, где значения True соответствуют точкам, входящим в окружность
            mask1 = distance_from_center1 <= radius
            mask2 = distance_from_center2 <= radius

            # вырезаем данные внутри круга
            data_inside_circle1 = img1[mask1]
            data_inside_circle2 = img2[mask2]
            data_inside_circle1 = np.round(data_inside_circle1, -2)
            data_inside_circle2 = np.round(data_inside_circle2, -2)
            
            mode1 = stats.mode(data_inside_circle1, nan_policy = 'omit')
            mode2 = stats.mode(data_inside_circle2, nan_policy = 'omit')

            logging.info(f"For image {index+1} mode intensivity: {str(mode1.mode)} [standard:{int(Tb*1000)}], count: {mode1.count}")
            logging.info(f"For image {index+2} mode intensivity: {str(mode2.mode)} [standard:{int(Tb*1000)}], count: {mode2.count}")
            
            correction_factor_brightness = 1 / (((mode1.mode + mode2.mode)/2) / (Tb * 1000))
            correction_factor_brightness_array.append(correction_factor_brightness)
            
            logging.info(f"For image {index + 1}, {index + 2} correction factor: {str(correction_factor_brightness)}")

        fits.writeto(f'{directory}_calibrated_brightness/{files[index][:-4]}_calibrated_brightness.fits', img1 * correction_factor_brightness, overwrite=True, header=header1)
        logging.info(f"Image {index+1}: {files[index]} - saved")
        fits.writeto(f'{directory}_calibrated_brightness/{files[index + 1][:-4]}_calibrated_brightness.fits', img2 * correction_factor_brightness, overwrite=True, header=header2)
        logging.info(f"Image {index+2}: {files[index + 1]} - saved")
        
np.save(f'correction_factor_{directory[-15:-7]}.npy', correction_factor_brightness_array)
            
fig = plt.figure()
ax = fig.gca()
plt.plot(correction_factor_brightness_array, linestyle = '-', linewidth = 3)
plt.grid()
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
# plt.show()