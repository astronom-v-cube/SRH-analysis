# Одиночный гаусс 
""" import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Функция для создания fits файла с овальным источником
def create_fits_file(file_path, shape=(50, 50), amplitude=1.0, x0=25, y0=25, a=5, b=10, noise_std=0.05):
    y, x = np.ogrid[:shape[0], :shape[1]]
    elliptical_source = amplitude * np.exp(-((x - x0) ** 2) / (2 * a ** 2) - ((y - y0) ** 2) / (2 * b ** 2))
    data = elliptical_source + np.random.normal(0, noise_std, shape)

    hdu = fits.PrimaryHDU(data)
    hdul = fits.HDUList([hdu])
    hdul.writeto(file_path, overwrite=True)

# Функция для аппроксимации данных гауссианом
def gaussian(x, amplitude, x0, y0, sigma_x, sigma_y, theta):
    x_rot = (x[0] - x0) * np.cos(theta) - (x[1] - y0) * np.sin(theta)
    y_rot = (x[0] - x0) * np.sin(theta) + (x[1] - y0) * np.cos(theta)
    exponent = -(x_rot ** 2 / (2 * sigma_x ** 2) + y_rot ** 2 / (2 * sigma_y ** 2))
    return amplitude * np.exp(exponent)

# Создание fits файла с овальным источником
fits_file_path = "oval_source.fits"
create_fits_file(fits_file_path)

# Чтение fits файла
hdul = fits.open("srh_20230715T072856_8600_LCP.fit")
data = hdul[0].data
data = data[620:680, 240:300]

np.set_printoptions(threshold=np.inf)

# Аппроксимация гауссианом
x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
# initial_guess = (0.5, data.shape[1] / 2, data.shape[0] / 2, 1, 1, 0)
initial_guess = (data.max(), 32, 26, 3, 5, 0)
bounds = ([data.min(), 0, 0, 1, 1, -np.pi], [data.max(), 1024, 1024, 10, 10, np.pi])

# Используем ravel(), чтобы вытянуть массив данных в одномерный
popt, _ = curve_fit(gaussian, (x.ravel(), y.ravel()), data.ravel(), p0=initial_guess, bounds = bounds)

# Вывод параметров
print("Amplitude:", popt[0])
print("Center X:", popt[1])
print("Center Y:", popt[2])
print("Sigma X:", popt[3])
print("Sigma Y:", popt[4])
print("Rotation Angle (theta):", popt[5])

# Создание гауссиана на основе параметров аппроксимации
gaussian_model = gaussian((x, y), *popt)

# Отображение оригинальных данных и аппроксимации
plt.figure(figsize=(8, 4))

plt.subplot(1, 3, 1)
plt.imshow(data, cmap='viridis', origin='lower')
plt.colorbar(label='Intensity')
plt.title('Original Data')
contour_levels = np.linspace(gaussian_model.min(), gaussian_model.max(), 3)
plt.contour(gaussian_model, levels=contour_levels, colors='red', linestyles='solid')

# Отображение аппроксимации гауссианом
plt.subplot(1, 3, 2)
plt.imshow(gaussian_model, cmap='viridis', origin='lower')
plt.colorbar(label='Intensity')
plt.title('Gaussian Fit')

# Отображение уровней аппроксимации
contour_levels = np.linspace(gaussian_model.min(), gaussian_model.max(), 3)
plt.contour(gaussian_model, levels=contour_levels, colors='red', linestyles='solid')


# Отображение аппроксимации гауссианом
plt.subplot(1, 3, 3)
plt.imshow(gaussian_model - data, cmap='viridis', origin='lower')
plt.colorbar(label='Intensity')
plt.title('Residual')
contour_levels = np.linspace(gaussian_model.min(), gaussian_model.max(), 3)
plt.contour(gaussian_model, levels=contour_levels, colors='red', linestyles='solid')

plt.tight_layout()
plt.show() """


""" import os
# функция для аппроксимации данных несколькими гауссианами
def gaussian_multi(x, *params):
    num_sources = len(params) // 6
    result = np.zeros_like(x[0], dtype=np.float64)
    
    for i in range(num_sources):
        amplitude, x0, y0, sigma_x, sigma_y, theta = params[i * 6:(i + 1) * 6]
        x_rot = (x[0] - x0) * np.cos(theta) - (x[1] - y0) * np.sin(theta)
        y_rot = (x[0] - x0) * np.sin(theta) + (x[1] - y0) * np.cos(theta)
        exponent = -(x_rot ** 2 / (2 * sigma_x ** 2) + y_rot ** 2 / (2 * sigma_y ** 2))
        result += amplitude * np.exp(exponent)
    
    return result

# Создание fits файла с несколькими овальными источниками
fits_file_path = "multi_oval_sources.fits"
sources = [(1.0, 25, 25, 5, 10), (0.5, 35, 15, 3, 5)]  # Пример нескольких источников

# Чтение fits файла
hdul = fits.open("srh_20230715T072748_7400_RCP.fit")
data = hdul[0].data
data = data[620:680, 250:300]

# Аппроксимация несколькими гауссианами
x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
initial_guess = np.array(
    [
        (data.max()/2, 22, 28, 8, 10, 0), 
        (data.max()/2, 28, 32, 3, 5, 0), 
        # (data.max()/2, 5, 20, 1, 4, np.pi)
     ]
    )
bounds = (
    [data.min(), 0, 0, 0, 0, -np.pi] * len(initial_guess),
    [data.max(), data.shape[1], data.shape[0], 15, 15, np.pi] * len(initial_guess)
    )

# Используем ravel(), чтобы вытянуть массив данных в одномерный
popt, _ = curve_fit(gaussian_multi, (x.ravel(), y.ravel()), data.ravel(), p0=initial_guess.flatten(), bounds=bounds)

# Вывод параметров
for i in range(len(popt) // 6):
    print(f"Source {i + 1} parameters:")
    print("Amplitude:", popt[i * 6])
    print("Center X:", popt[i * 6 + 1])
    print("Center Y:", popt[i * 6 + 2])
    print("Sigma X:", popt[i * 6 + 3])
    print("Sigma Y:", popt[i * 6 + 4])
    print("Rotation Angle (theta):", popt[i * 6 + 5])

# Создание модели на основе параметров аппроксимации
gaussian_model = gaussian_multi((x, y), *popt)

# Отображение оригинальных данных и аппроксимации
plt.figure(figsize=(12, 4))

# Находим максимальное положительное значение
max_positive_value = np.max(gaussian_model[gaussian_model > 0])

# contour_levels = np.linspace(gaussian_model.min(), gaussian_model.max(), 3)
# Задаем уровень контура как 0.5 от максимального положительного значения
contour_levels = [0.5 * max_positive_value] # type: ignore


plt.subplot(1, 3, 1)
plt.imshow(data, cmap='viridis', origin='lower')
plt.colorbar(label='Intensity', orientation='horizontal', aspect=25)
plt.title('Original Data')
plt.contour(gaussian_model, levels=contour_levels, colors='red', linestyles='solid')

plt.subplot(1, 3, 2)
plt.imshow(gaussian_model, cmap='viridis', origin='lower')
plt.colorbar(label='Intensity', orientation='horizontal', aspect=25)
plt.title('Gaussian Fit')
plt.contour(gaussian_model, contour_levels, colors='red', linestyles='solid')


plt.subplot(1, 3, 3)
plt.imshow(gaussian_model - data, cmap='viridis', origin='lower')
plt.colorbar(label='Intensity', orientation='horizontal', aspect=25)
plt.title('Residual')
plt.contour(gaussian_model, levels=contour_levels, colors='red', linestyles='solid')

plt.tight_layout()
plt.show()

contours = plt.contour(np.maximum(gaussian_model, 0), levels=[contour_levels], colors='red', linestyles='solid')

# Используем метод findobj для поиска контуров
for collection in plt.findobj(contours):
    path = collection.get_paths()[0]
    vertices = path.vertices
    codes = path.codes

    # Создаем объект Path
    contour_path = Path(vertices, codes)

    # Подсчитываем пиксели внутри контура
    x, y = np.meshgrid(np.arange(gaussian_model.shape[1]), np.arange(gaussian_model.shape[0]))
    points = np.column_stack((x.flatten(), y.flatten()))
    inside_pixels = contour_path.contains_points(points)

    # Площадь в пикселях
    area_in_pixels = np.sum(inside_pixels)

    print(f"Площадь в пикселях: {area_in_pixels}") """


import os
import re

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.path import Path
from scipy.optimize import curve_fit

directory = "D:/08-26-46"

pattern = re.compile(r'(?<=[_.])\d{4,5}(?=[_.])')

def extract_number(filename):
    match = pattern.search(filename)
    freqs.add(int(match.group()))
    return int(match.group())

freqs = set()
files = sorted(os.listdir(directory), key=extract_number)
freqs = sorted(list(freqs))

# функция для аппроксимации данных несколькими гауссианами
def gaussian_multi(x, *params):
    num_sources = len(params) // 6
    result = np.zeros_like(x[0], dtype=np.float64)
    
    for i in range(num_sources):
        amplitude, x0, y0, sigma_x, sigma_y, theta = params[i * 6:(i + 1) * 6]
        x_rot = (x[0] - x0) * np.cos(theta) - (x[1] - y0) * np.sin(theta)
        y_rot = (x[0] - x0) * np.sin(theta) + (x[1] - y0) * np.cos(theta)
        exponent = -(x_rot ** 2 / (2 * sigma_x ** 2) + y_rot ** 2 / (2 * sigma_y ** 2))
        result += amplitude * np.exp(exponent)
    
    return result

for i, file in enumerate(files):

    # Чтение fits файла
    hdul = fits.open(f'{directory}/{file}')
    data = hdul[0].data
    data = data[600:680, 310:380]

    # Аппроксимация несколькими гауссианами
    x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    initial_guess = np.array(
        [
            (data.max()*0.8, 19, 32, 8, 10, 0), 
            (data.max()*0.8, 33, 46, 3, 5, 0), 
            (data.max()*0.8, 44, 45, 1, 4, np.pi)
            
            # (data.max()/2, 19, 32, 8, 10, 0), 
            # (data.max()/2, 28, 35, 3, 5, 0), 
            # (data.max()/2, 35, 47, 1, 4, 0),
            # (data.max()/2, 44, 45, 1, 4, 0)
        ]
        )
    bounds = (
        [0, 0, 0, 0, 0, -np.pi] * len(initial_guess),
        [data.max(), data.shape[1], data.shape[0], 30, 30, np.pi] * len(initial_guess)
        )

    # Используем ravel(), чтобы вытянуть массив данных в одномерный
    popt, _ = curve_fit(gaussian_multi, (x.ravel(), y.ravel()), data.ravel(), p0=initial_guess.flatten(), bounds=bounds)

    # Вывод параметров
    for i in range(len(popt) // 6):
        print(f"Source {i + 1} parameters:")
        print("Amplitude:", popt[i * 6])
        print("Center X:", popt[i * 6 + 1])
        print("Center Y:", popt[i * 6 + 2])
        print("Sigma X:", popt[i * 6 + 3])
        print("Sigma Y:", popt[i * 6 + 4])
        print("Rotation Angle (theta):", popt[i * 6 + 5])

    # Создание модели на основе параметров аппроксимации
    gaussian_model = gaussian_multi((x, y), *popt)

    # Отображение оригинальных данных и аппроксимации
    plt.figure(figsize=(12, 4))

    # Находим максимальное положительное значение
    max_positive_value = np.max(gaussian_model[gaussian_model > 0])

    # contour_levels = np.linspace(gaussian_model.min(), gaussian_model.max(), 3)
    # Задаем уровень контура как 0.5 от максимального положительного значения
    contour_levels = [0.5 * max_positive_value] # type: ignore


    plt.subplot(1, 3, 1)
    plt.imshow(data, cmap='viridis', origin='lower')
    plt.colorbar(label='Intensity', orientation='horizontal', aspect=25)
    plt.title('Original Data')
    plt.contour(gaussian_model, levels=contour_levels, colors='red', linestyles='solid')

    plt.subplot(1, 3, 2)
    plt.imshow(gaussian_model, cmap='viridis', origin='lower')
    plt.colorbar(label='Intensity', orientation='horizontal', aspect=25)
    plt.title('Gaussian Fit')
    plt.contour(gaussian_model, contour_levels, colors='red', linestyles='solid')

    plt.subplot(1, 3, 3)
    plt.imshow(gaussian_model - data, cmap='viridis', origin='lower')
    plt.colorbar(label='Intensity', orientation='horizontal', aspect=25)
    plt.title('Residual')
    plt.contour(gaussian_model, levels=contour_levels, colors='red', linestyles='solid')

    plt.tight_layout()
    plt.show()

    contours = plt.contour(np.maximum(gaussian_model, 0), levels=[0.5 * max_positive_value], colors='red', linestyles='solid')

    # Используем метод findobj для поиска контуров
    for collection in plt.findobj(contours):
        path = collection.get_paths()[0]
        vertices = path.vertices
        codes = path.codes

        # Создаем объект Path
        contour_path = Path(vertices, codes)

        # Подсчитываем пиксели внутри контура
        x, y = np.meshgrid(np.arange(gaussian_model.shape[1]), np.arange(gaussian_model.shape[0]))
        points = np.column_stack((x.flatten(), y.flatten()))
        inside_pixels = contour_path.contains_points(points)

        # Площадь в пикселях
        area_in_pixels = np.sum(inside_pixels)

        print(f"Площадь в пикселях: {area_in_pixels}")