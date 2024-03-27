import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, PchipInterpolator
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

# Ваши данные
y = [994, 361, 468, 581, 282, 438, 614, 1017]
x = ['05:55:30', '05:56:30', '05:57:30', '05:58:00', '05:58:45', '05:59:45', '06:00:45', '06:02:00']

# Преобразование времени в числовые значения
x_numeric = np.array([60, 120, 180, 210, 255, 315, 375, 450]) - 60

# Создание кубического сплайна
cs = PchipInterpolator(x_numeric, y)

# Создание более плотного набора точек для гладкой кривой
x_smooth = np.linspace(x_numeric[0], x_numeric[-1], 1000)
y_smooth = cs(x_smooth)

# Создание графика
plt.plot(x_numeric, y, 'D', label='Исходные данные', color = 'red', markersize=10, zorder = 1)
plt.plot(x_smooth, y_smooth, label='Интерполированная кривая', linewidth = 6, zorder = 0)
plt.xlabel('Время, UT', fontsize=25)
plt.ylabel('Магнитное поле, Гс', fontsize=25)
# plt.title('График интерполяции', fontsize=32)
plt.legend(fontsize=16)
plt.grid(True)
# Задание пользовательских подписей по оси x
plt.xticks(x_numeric, x, rotation=45)
plt.show()


# Ваши данные
y = [22.1, 10.24, 9.9, 8.75, 8.48, 9.66, 10.11, 10.83]

# Создание кубического сплайна
cs = PchipInterpolator(x_numeric, y)

# Создание более плотного набора точек для гладкой кривой
x_smooth = np.linspace(x_numeric[0], x_numeric[-1], 1000)
y_smooth = cs(x_smooth)

# Создание графика
plt.plot(x_numeric, y, 'D', label='Исходные данные', color = 'red', markersize=10, zorder = 1)
plt.plot(x_smooth, y_smooth, label='Интерполированная кривая', linewidth = 6, zorder = 0)
plt.xlabel('Время', fontsize=25)
plt.ylabel(r'$\delta_1$', fontsize=25)
# plt.title('График интерполяции', fontsize=32)
plt.legend(fontsize=16)
plt.grid(True, axis='both')
# Задание пользовательских подписей по оси x
plt.xticks(x_numeric, x, rotation=80)
plt.ylim(0, 25)
# plt.show()

# Ваши данные
y = np.abs(np.array([91.92, 105.11, 123.71, 118.57, 113.13, 115.99, 130.2, 174.11]) - 180)

# Создание кубического сплайна
cs = PchipInterpolator(x_numeric, y)

# Создание более плотного набора точек для гладкой кривой
x_smooth = np.linspace(x_numeric[0], x_numeric[-1], 1000)
y_smooth = cs(x_smooth)

# Создание графика
plt.plot(x_numeric, y, 'D', label='Исходные данные', color = 'red', markersize=10, zorder = 1)
plt.plot(x_smooth, y_smooth, label='Интерполированная кривая', linewidth = 6, zorder = 0)
plt.xlabel('Время', fontsize=25)
plt.ylabel(r'$\theta$, град', fontsize=25)
# plt.title('График интерполяции', fontsize=32)
plt.xticks(x_numeric, x, rotation=80)
plt.legend(fontsize=16)
plt.grid(True, axis='both')
plt.ylim(0, 90)
# plt.show()
