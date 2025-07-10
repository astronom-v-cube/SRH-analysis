import logging
import os
import re

import matplotlib.patches as patches
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.patches import Circle
from tqdm import tqdm
import skimage

from analise_utils import Extract, Monitoring, OsOperations, ArrayOperations

Extract = Extract()
Monitoring.start_log('logs')
# logging.info(f'Start program show of the solar on difference frequency')

# directory = "/mnt/astro/14may_new_612_times/20240514T014200_OWM_aligned"
vcenter = 5000

# # logging.info(f'Path to files: {directory}')

# files = sorted(os.listdir(directory), key=lambda x: Extract.extract_number(x))
# logging.info(f'Search {len(files)} files')
# logging.info(f'Files: \n {files}')

# Загрузка fits файла
path = "J:/20240514_hr/23400/srh_20240514T020659_23400_LCP.fits"
fits_image_file = fits.open(path)
# # Получение матрицы numpy из данных fits
fits_data = fits_image_file[0].data
cropped_data = fits_data[873:898, 385:415]
x_min, x_max = 873, 898  # по оси X (столбцы)
y_min, y_max = 385, 415  # по оси Y (строки)

levels = np.linspace(0.1 * np.max(fits_data), 0.995 * np.max(fits_data), 7)

fig = plt.figure(figsize=(9, 9))
ax = plt.gca()
plt.contour(fits_data, levels=levels)

plt.imshow(
    fits_data,
    origin='lower',
    cmap='plasma',
    vmin=1e4,
    vmax=fits_data.max(),
    # extent=[x_min, x_max, y_min, y_max]  # границы в исходных координатах
)  # norm=TwoSlopeNorm(vmin=0, vcenter=vcenter, vmax=300000),
rectangle1 = patches.Rectangle((882-0.5, 393-0.5), 3, 3, linewidth=2, edgecolor='k', facecolor='none', zorder = 5)
ax.add_patch(rectangle1)
rectangle2 = patches.Rectangle((884-0.5, 401-0.5), 3, 3, linewidth=2, edgecolor='k', facecolor='none', zorder = 5)
ax.add_patch(rectangle2)
rectangle3 = patches.Rectangle((887-0.5, 397-0.5), 3, 3, linewidth=2, edgecolor='k', facecolor='none', zorder = 5)
ax.add_patch(rectangle3)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.tight_layout()
plt.savefig(f'areas/{path[25:40]}.png')
# plt.show()
