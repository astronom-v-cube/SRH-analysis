from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import os

import logging
# logging.basicConfig(filename = 'logs.log',  filemode='a', level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s', encoding = "UTF-8", datefmt='%d-%b-%y %H:%M:%S')
logging.basicConfig(filename = 'logs.log',  filemode='a', level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.info(f'Start')

# Каталог из которого будем брать файлы
directory = 'aligned'
logging.info(f'Путь к файлам: {directory}')
files = sorted(os.listdir(directory))
logging.info(f'Найдено {len(files)} файлов')
logging.info(f'Файлы: \n {files}')

# отображение всех значений матрицы в консоль
np.set_printoptions(threshold=np.inf)

# Загрузка fits файла
# fits_image_file = fits.open('data/9_srh_V_2022-05-11T05_57_45_9400.fit', ignore_missing_simple=True)
# # Получение матрицы numpy из данных fits
# fits_data = fits_image_file[0].data

####### округление до целого
# result = []
# for i in fits_data:
#     rounded_matrix = np.round(i)
#     result.append(rounded_matrix)
#########

# получение границ кропа через срезы из строк и каждой строки
# stroka_1 = 310
# stroka_2 = 360
# stolbec_1 = 105
# stolbec_2 = 165

stroka_1 = 0
stroka_2 = 1023
stolbec_1 = 0
stolbec_2 = 1023   
    
def multiple_crope_images_display(input_matrix_list_files, NX=4, NY=4):
    # https://teletype.in/@pythontalk/matplotlib_subplot_tutorial
    fig, axs = plt.subplots(NY, NX, sharex=True, sharey=True, figsize=(NX*3,NY*3))

    for i in range(0, len(files)):
        print(input_matrix_list_files[i])
        fits_image_file = fits.open(f'{directory}/{input_matrix_list_files[i]}', ignore_missing_simple=True)[0].data

        # if i % 2 == 0:
        #     plt.title(f'freq {freq}\nStoks I')
        # else:  
        #     plt.title(f'freq {freq}\nStoks V') 
        
        ax = axs[i//NX, i%NX]
        print(i//NX, i%NX)
        
        freq = str(input_matrix_list_files[i])[-10:-6]
        print(freq)
        i_or_v = str(input_matrix_list_files[i])[-5]
        print(f'рисую в клетке {i+1}')
        ax.imshow((fits_image_file)[stroka_1:stroka_2, stolbec_1:stolbec_2], origin='lower', cmap='plasma', interpolation='gaussian')
        ax.set_title(f'freq {freq}\nStoks {i_or_v}') 
        # plt.title(f'{input_matrix_list_files[i]}') 
        # plt.colorbar()
        # origin='lower' - расположение начал координат – точки [0,0]
        # plt.gca().invert_yaxis()
        ax.axis('off') 
        plt.pause(0.5)
    
    fig.tight_layout() 
    plt.show()

multiple_crope_images_display(files)

logging.info(f'Stop')
