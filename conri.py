from astropy import units as u
import sunpy.map
import sunpy.data.sample
import sunpy
from sunpy.net import Fido, attrs as a
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from astropy.time import Time
from scipy.ndimage import gaussian_filter
from datetime import datetime as dt
from reproject import reproject_interp
from astropy.wcs import WCS
from astropy.io import fits
import os

def normalize_array(arr, new_min, new_max, power=5):
    """
    Нормирует значения двумерного массива на заданный диапазон только если значение не равно Nan.

    Parameters:
    arr (numpy.ndarray): Исходный двумерный массив.
    new_min (float): Минимальное значение нового диапазона.
    new_max (float): Максимальное значение нового диапазона.

    Returns:
    numpy.ndarray: Нормированный массив.
    """
    normalized = np.full(arr.shape, np.nan)
    old_min = np.nanmin(arr)
    old_max = np.nanmax(arr)

    mask = arr != np.nan
    # нормализуем значения в диапазоне [0, 1]
    transformed_values = (arr[mask] - old_min) / (old_max - old_min)

    # логарифмическое преобразование
    # transformed_values = np.log1p(transformed_values * (np.exp(power) - 1)) / np.log1p(np.exp(power) - 1)
    # normalized[mask] = new_min + transformed_values * (new_max - new_min)

    transformed_values = (np.tanh(transformed_values * 6) + 1) / 2
    normalized[mask] = new_min + transformed_values * (new_max - new_min)

    return normalized

from astropy.io.fits import Header

def merge_fits_headers(header1: Header, header2: Header) -> Header:
    """
    Объединяет два FITS хэдера: берет информацию о наблюдениях и инструменте из первого,
    а информацию о системе координат из второго.

    Parameters:
        header1 (Header): Первый FITS хэдер (информация о наблюдениях и инструменте).
        header2 (Header): Второй FITS хэдер (информация о системе координат).

    Returns:
        Header: Новый объединённый хэдер.
    """
    new_header = Header(header2)
    corrected_keys = ['DATE-OBS', 'DATE', 'TELESCOP', 'WAVELNTH']
    for key in corrected_keys:
        if key in header1:
            new_header[key] = header1[key]
    return new_header

def delete_files(file_paths):
    for file_path in file_paths:
        try:
            os.remove(file_path)
            print(f"Удален файл: {file_path}")
        except FileNotFoundError:
            print(f"Файл не найден: {file_path}")
        except Exception as e:
            print(f"Ошибка при удалении файла {file_path}: {e}")

def create_coordinate_binding_img(path_to_srh):

    date_object = dt.strptime(path_to_srh.split('/')[-1].split('_')[-3], "%Y%m%dT%H%M%S")
    time = Time(date_object)

    # hmi_query = Fido.search(a.Time(time - 40 * u.second, time + 40 * u.second),
    #                         a.Instrument('HMI'),
    #                         a.Physobs('intensity'),
    #                         a.Wavelength(6173 * u.angstrom))

    # hmi_files = Fido.fetch(hmi_query)
    # hmi_file = hmi_files[0]

    hmi_file = '/mnt/data/Документы/SRH-analysis/temp_fits/hmi.ic_45s.2024.05.14_02_00_45_TAI.continuum.fits'
    hmi_data = fits.open(hmi_file)[1]
    srh_data = fits.open(path_to_srh)[0]
    array, footprint = reproject_interp(hmi_data, srh_data.header)
    new_coord_head = merge_fits_headers(hmi_data.header, srh_data.header)
    hmi_map = sunpy.map.Map(array, new_coord_head)

    # # Попытка сделать модель спадания яркости и ее компенсацию
    # hmi_map = sunpy.map.Map(hmi_file)
    dimensions = (hmi_map.dimensions.x.value, hmi_map.dimensions.y.value)
    # Создание сетки пиксельных координат
    yy, xx = np.meshgrid(np.arange(dimensions[1]), np.arange(dimensions[0]))
    center_x = hmi_map.reference_pixel.x.value
    center_y = hmi_map.reference_pixel.y.value
    r = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    # Радиус солнечного диска в пикселях
    radius = (hmi_map.rsun_obs.to('arcsec').value / hmi_map.scale.axis1.value) - 1
    # Пример модели лимбного потемнения: I = I0 * (1 - u * (1 - sqrt(1 - (r/R)^2)))
    u = 0.6 # Коэффициент лимбного потемнения (подбирается для конкретной длины волны)
    limb_darkening_model = 1 - u * (1 - (1 - (r / radius) ** 2) ** (1 / 2.4))
    normalized_data = hmi_map.data / limb_darkening_model
    normalized_map = sunpy.map.Map(normalized_data, hmi_map.meta)
    normalized_map.save('temp_fits/circle_normalized.fits', overwrite=True)

    hmi_data = fits.open('temp_fits/circle_normalized.fits')[0]

    array = normalize_array(hmi_data.data, srh_data.data.max()*1.5, 0)
    array = np.nan_to_num(array, nan=0)
    smoothed_data_hmi = gaussian_filter(array, sigma=(7, 7))
    # smoothed_data_hmi_fliped = np.flip(smoothed_data_hmi, axis=(0, 1))
    fits.writeto('temp_fits/smooth_hmi.fits', smoothed_data_hmi, new_coord_head, overwrite=True)
    # fits.writeto('temp_fits/smooth_hmi_fliped.fits', smoothed_data_hmi_fliped, new_coord_head, overwrite=True)

    # hmi_map = sunpy.map.Map('temp_fits/smooth_hmi_fliped.fits')
    # srh_map = sunpy.map.Map(path_to_srh)
    # # delete_files(hmi_files)

    # plt.imshow(hmi_map.data, cmap='plasma', origin='lower')
    # plt.colorbar(label='Intensity (Negative)')
    # plt.tight_layout()
    # plt.show()

    # hmi_map = sunpy.map.Map('temp_fits/smooth_hmi.fits')
    # srh_map = sunpy.map.Map(path_to_srh)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(projection=hmi_map)
    # srh_map.plot_settings['norm'] = matplotlib.colors.TwoSlopeNorm(vmin=0, vcenter=9e5, vmax=8e6)
    # srh_map.plot_settings['cmap'] = matplotlib.colormaps['plasma']
    # hmi_map.plot(axes=ax1, alpha=0.95)
    # srh_map.plot(axes=ax1, alpha=0.65)
    # plt.show()

    return smoothed_data_hmi

if __name__ == '__main__':
    create_coordinate_binding_img(path_to_srh = 'temp_fits/srh_20240514T014017_9200_LCP.fit')