from analise_utils import ArrayOperations, Extract, Monitoring, OsOperations
import os

directory = f"D:/datasets/14.05.24"


# f, ff = OsOperations.freq_sorted_1st_two_files_in_folders(directory)
# print(f, ff)


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Размер матрицы
size = 512

# Центр гауссового источника
center_x, center_y = size // 2, size // 2

# Стандартные отклонения по осям x и y
std_x, std_y = 50, 50

# Создаем координатные сетки
x = np.linspace(0, size - 1, size)
y = np.linspace(0, size - 1, size)
x, y = np.meshgrid(x, y)


# Параметры распределения
mean = [center_x, center_y]
cov = [[std_x**2, 0], [0, std_y**2]]

# Создаем распределение
rv = multivariate_normal(mean, cov)
gaussian_matrix = rv.pdf(np.dstack((x, y)))

# Нормализуем матрицу для визуализации
gaussian_matrix /= np.max(gaussian_matrix)
max_col, max_row, max_value = ArrayOperations.calculate_weighted_centroid(gaussian_matrix, (256, 200), 0.5, ((190, 320), (190, 320)))
print(max_col, max_row, max_value)


# Отображаем матрицу
plt.imshow(gaussian_matrix, cmap='hot', interpolation='nearest')
contour = plt.contour(gaussian_matrix, levels=[0.5], colors='blue')
plt.title('512x512 Matrix with Gaussian Source')
# plt.xlim(190, 320)
# plt.ylim(190, 320)
plt.show()

# v = []
# for i in range (0, 60):
#     vcenter = - 1500 * (i + 1) + 200000
#     v.append(vcenter)

# plt.plot(v)
# plt.show()