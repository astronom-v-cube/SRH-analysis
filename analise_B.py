import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.rcParams.update({'font.size': 22})

# Ваши данные
y = [535, 537, 753, 877, 1035]
x = ['05:58:00 UT', '05:58:45 UT', '05:59:45 UT', '06:00:45 UT', '06:02:00 UT']

# Преобразование времени в числовые значения
x_numeric = [0, 45, 105, 165, 240]

# Создание кубического сплайна
cs = CubicSpline(x_numeric, y)

# Создание более плотного набора точек для гладкой кривой
x_smooth = np.linspace(x_numeric[0], x_numeric[-1], 100)
y_smooth = cs(x_smooth)

# Создание графика
plt.plot(x_numeric, y, 'D', label='Исходные данные', color = 'red', markersize=10, zorder = 1)
plt.plot(x_smooth, y_smooth, label='Интерполированная кривая', linewidth = 6, zorder = 0)
plt.xlabel('Время', fontsize=32)
plt.ylabel('Величина магнитного поля, Гс', fontsize=32)
# plt.title('График интерполяции', fontsize=32)
plt.legend(fontsize=25)
plt.grid(True)
# Задание пользовательских подписей по оси x
plt.xticks(x_numeric, x)
plt.show()


# Ваши данные
y = [5.07, 4.81, 4.76, 4.99, 5.4]
x = ['05:58:00 UT', '05:58:45 UT', '05:59:45 UT', '06:00:45 UT', '06:02:00 UT']

# Преобразование времени в числовые значения
x_numeric = [0, 45, 105, 165, 240]

# Создание кубического сплайна
cs = CubicSpline(x_numeric, y)

# Создание более плотного набора точек для гладкой кривой
x_smooth = np.linspace(x_numeric[0], x_numeric[-1], 100)
y_smooth = cs(x_smooth)

# Создание графика
plt.plot(x_numeric, y, 'D', label='Исходные данные', color = 'red', markersize=10, zorder = 1)
plt.plot(x_smooth, y_smooth, label='Интерполированная кривая', linewidth = 6, zorder = 0)
plt.xlabel('Время', fontsize=32)
plt.ylabel(r'$\delta_1$', fontsize=32)
# plt.title('График интерполяции', fontsize=32)
plt.legend(fontsize=25)
plt.grid(True)
# Задание пользовательских подписей по оси x
plt.xticks(x_numeric, x)
plt.show()