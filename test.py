import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from astropy.io import fits
import os, sys
import shutil


directory = 'test_dataset_2'
# Загрузка первого файла и нахождение координат отличительного признака
files = sorted(os.listdir(directory))
hdul1 = fits.open(f'{directory}/{files[1]}')
img1 = hdul1[0].data  # данные первого изображения

import numpy as np

def find_max_around_point(matrix, point, size):

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
    
    return (max_col, max_row, max_value)

max_indices = find_max_around_point(img1, (185, 725), 25)
print(max_indices)

