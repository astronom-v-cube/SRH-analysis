import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, PchipInterpolator
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

# Ваши данные
y = [84, 79, 78, 84, 78, 51, 70, 83, 104, 147]
x = ['08:23:51', '08:24:00', '08:24:08', '08:24:13', '08:24:20', '08:24:39', '08:24:46',  '08:25:41', '08:25:45', '08:26:00']

# Преобразование времени в числовые значения
x_numeric = [0, 9, 17, 22, 29, 48, 55, 110-30, 114-30, 129-30]

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
plt.legend(fontsize=20)
plt.grid(True)
# Задание пользовательских подписей по оси x
plt.xticks(x_numeric, x, rotation=80)
plt.ylim(0, 160)
plt.show()


# Ваши данные
y = [4.97, 4.78, 4.97, 3.95, 4.26, 3.99, 4.28, 4.6, 4.3, 3.98]

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
plt.legend(fontsize=20)
plt.grid(True)
# Задание пользовательских подписей по оси x
plt.xticks(x_numeric, x, rotation=80)
plt.ylim(0, 6)
plt.show()

# Ваши данные
y = np.abs(np.array([135.32, 142.79, 132.12, 136.8, 138.62, 137.26, 136.7, 126.1, 129.86, 130.05]) - 180)

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
plt.legend(fontsize=20)
plt.grid(True)
# Задание пользовательских подписей по оси x
plt.xticks(x_numeric, x, rotation=80)
plt.ylim(0, 90)
plt.show()