import numpy as np
import matplotlib.pyplot as plt

def running_mean(data, window_size):
    window = np.ones(window_size) / window_size
    smoothed_data = np.convolve(data, window, mode='same')
    return smoothed_data

# Пример данных
data = [3, 5, 1, 2, 6, 8, 2, 4, 7, 3]

# Применение бегущего среднего по трем точкам
smoothed_data = running_mean(data, window_size=3)

# Построение исходного и сглаженного графиков
plt.plot(data, label='Исходные данные')
plt.plot(smoothed_data, label='Сглаженные данные')
plt.legend()
plt.show()
