import numpy as np
from astropy.io import fits
import os

import logging
logging.basicConfig(filename = 'logs.log',  filemode='a', level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.info(f'Start of the program to search intensity spectrum of a sun')

##########
directory = 'aligned'
##########
logging.info(f'Path to files: {directory}')

files = sorted(os.listdir(directory))
logging.info(f'Find {len(files)} files')
logging.info(f'List files: \n {files}')

def find_intensity_in_point(matrix : np.ndarray, point : tuple) -> float:
    """     
    The function returns the intensity value at a specific point. 
    The input arguments of the function are the matrix `matrix` (two-dimensional array) and the point `point` (coordinates).
    The function returns three values: x, y and the intensity value in them.
    """
    x, y = point
    return x, y, matrix[y][x]

coordinates = (512, 512)
intensivity_list = []

for image in range(len(files)):
    # Считывание файлов
    data = fits.open(f'{directory}/{files[image]}', ignore_missing_simple=True)
    img = data[0].data
    data.close()

    x, y, intensivity =  find_intensity_in_point(img, coordinates)
    intensivity_list.append(intensivity)

print(intensivity_list)

    