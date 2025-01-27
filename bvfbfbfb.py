import numpy as np
import sunpy.map
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt

# Загрузка изображения
smap = sunpy.map.Map('/home/dmitry/sunpy/data/hmi.ic_45s.2024.05.14_02_00_45_TAI.continuum.fits')

# Получение размеров карты в пикселях
dimensions = (smap.dimensions.x.value, smap.dimensions.y.value)

# Создание сетки пиксельных координат
yy, xx = np.meshgrid(np.arange(dimensions[1]), np.arange(dimensions[0]))

# Координаты центра солнечного диска в пикселях
center_x = smap.reference_pixel.x.value
center_y = smap.reference_pixel.y.value

# Вычисление расстояний в пикселях
r = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)

# Радиус солнечного диска в пикселях
radius = (smap.rsun_obs.to('arcsec').value / smap.scale.axis1.value) + 25

# Пример модели лимбного потемнения: I = I0 * (1 - u * (1 - sqrt(1 - (r/R)^2)))
u = 0.65  # Коэффициент лимбного потемнения (подбирается для конкретной длины волны)
limb_darkening_model = 1 - u * (1 - np.sqrt(1 - (r / radius) ** 2))

# Нормализация изображения
normalized_data = smap.data / limb_darkening_model

# Создаем новую карту
normalized_map = sunpy.map.Map(normalized_data, smap.meta)

# Сохранение/отображение результата
normalized_map.plot()
plt.show()