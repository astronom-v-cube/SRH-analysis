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


def normalize_array(arr, r, new_min, new_max):
    """
    Нормирует значения двумерного массива на заданный диапазон только внутри круга.

    Parameters:
    arr (numpy.ndarray): Исходный двумерный массив.
    r (numpy.ndarray): Массив расстояний от центра.
    new_min (float): Минимальное значение нового диапазона.
    new_max (float): Максимальное значение нового диапазона.

    Returns:
    numpy.ndarray: Нормированный массив.
    """
    # Создаем маску для элементов внутри круга
    mask = r <= np.nanmax(r)  # Можно задать радиус, например, r <= radius

    # Нормализуем только элементы внутри круга
    normalized = np.full(arr.shape, np.nan)  # Инициализируем массив NaN
    old_min = np.nanmin(arr[mask])
    old_max = np.nanmax(arr[mask])

    # Проверяем, чтобы избежать деления на ноль
    if old_max > old_min:
        normalized[mask] = new_min + (arr[mask] - old_min) * (new_max - new_min) / (old_max - old_min)

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
    # Создаем копию первого хэдера для сохранения его структуры
    new_header = Header(header1)

    # Ключи, относящиеся к системе координат (их можно уточнить для конкретных задач)
    coord_keys = [
        "CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
        "CDELT1", "CDELT2", "CUNIT1", "CUNIT2", "CD1_1", "CD1_2", "CD2_1", "CD2_2",
        "PV1_1", "PV1_2", "PV2_1", "PV2_2", "LONPOLE", "LATPOLE"
    ]

    # Перенос ключей системы координат из второго хэдера
    for key in coord_keys:
        if key in header2:
            new_header[key] = header2[key]

    # Возвращаем объединенный хэдер
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

    hmi_query = Fido.search(a.Time(time - 40 * u.second, time + 40 * u.second),
                            a.Instrument('HMI'),
                            a.Physobs('intensity'),
                            a.Wavelength(6173 * u.angstrom))

    hmi_files = Fido.fetch(hmi_query)
    hmi_file = hmi_files[0]
    hmi_data = fits.open(hmi_file)[1]
    srh_data = fits.open(path_to_srh)[0]
    array, footprint = reproject_interp(hmi_data, srh_data.header)
    new_coord_head = merge_fits_headers(hmi_data.header, srh_data.header)
    hmi_data = sunpy.map.Map(array, new_coord_head)

    # Попытка сделать модель спадания яркости и ее компенсацию
    hmi_map = sunpy.map.Map(hmi_file)
    dimensions = (hmi_map.dimensions.x.value, hmi_map.dimensions.y.value)
    # Создание сетки пиксельных координат
    yy, xx = np.meshgrid(np.arange(dimensions[1]), np.arange(dimensions[0]))
    center_x = hmi_map.reference_pixel.x.value
    center_y = hmi_map.reference_pixel.y.value
    r = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    # Радиус солнечного диска в пикселях
    print(r)
    radius = (hmi_map.rsun_obs.to('arcsec').value / hmi_map.scale.axis1.value) + 25
    # Пример модели лимбного потемнения: I = I0 * (1 - u * (1 - sqrt(1 - (r/R)^2)))
    u = 0.56  # Коэффициент лимбного потемнения (подбирается для конкретной длины волны)
    limb_darkening_model = 1 - u * (1 - np.sqrt(1 - (r / radius) ** 2))
    normalized_data = hmi_map.data / limb_darkening_model
    normalized_map = sunpy.map.Map(normalized_data, hmi_map.meta)
    normalized_map.save('circle_normalized.fits', overwrite=True)

    hmi_data = fits.open('circle_normalized.fits')[0]
    
    

    array = np.nan_to_num(array, nan=0)
    smoothed_data_hmi = gaussian_filter(array, sigma=(7, 7))
    smoothed_data_hmi = normalize_array(r, smoothed_data_hmi, srh_data.data.max(), 0)

    
    fits.writeto('smooth_hmi.fits', smoothed_data_hmi, srh_data.header, overwrite=True)

    hmi_map = sunpy.map.Map('smooth_hmi.fits')
    srh_map = sunpy.map.Map('/mnt/astro/14may_new_612/6000/srh_20240514T014016_6000_LCP.fit')
    # delete_files(hmi_files)

    plt.imshow(smoothed_data_hmi, cmap='plasma', origin='lower')
    plt.colorbar(label='Intensity (Negative)')
    plt.show()


    fig = plt.figure()
    ax1 = fig.add_subplot(projection=hmi_map)
    srh_map.plot_settings['norm'] = matplotlib.colors.TwoSlopeNorm(vmin=0, vcenter=1e5, vmax=8e6)
    srh_map.plot_settings['cmap'] = matplotlib.colormaps['plasma']
    hmi_map.plot(axes=ax1, alpha=0.7)
    srh_map.plot(axes=ax1, alpha=0.5)
    plt.show()

if __name__ == '__main__':
    create_coordinate_binding_img(path_to_srh = 'test_dataset/srh_20220113T031122_6200_LCP.fit')