import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.rcParams.update({'font.size': 22})

fig, axes = plt.subplots(1, 2)

# Ваши данные
y = [535, 537, 753, 877, 1035]
x = ['05:58', '05:59', '06:00', '06:01', '06:02']

# Преобразование времени в числовые значения
x_numeric = [0, 45, 105, 165, 240]

# Создание кубического сплайна
cs = CubicSpline(x_numeric, y)

# Создание более плотного набора точек для гладкой кривой
x_smooth = np.linspace(x_numeric[0], x_numeric[-1], 241)
y_smooth = cs(x_smooth)
print(x_smooth)

# Создание графика
axes[0].plot(x_numeric, y, 'D', label='Исходные данные', color = 'red', markersize=10, zorder = 1)
axes[0].plot(x_smooth, y_smooth, label='Интерполированная кривая', linewidth = 6, zorder = 0)
axes[0].set_title('а', fontsize=32)
axes[0].set_xlabel('Время, UT', fontsize=32)
axes[0].set_ylabel('Величина магнитного поля, Гс', fontsize=32)
# plt.title('График интерполяции', fontsize=32)
axes[0].legend(fontsize=25)
axes[0].grid(True)
# Задание пользовательских подписей по оси x
axes[0].set_xticks([x_smooth[0], x_smooth[60], x_smooth[120], x_smooth[180], x_smooth[240]], x)



# Ваши данные
y = [5.07, 4.81, 4.76, 4.99, 5.4]
x = ['05:58', '05:59', '06:00', '06:01', '06:02']

# Преобразование времени в числовые значения
x_numeric = [0, 45, 105, 165, 240]

# Создание кубического сплайна
cs = CubicSpline(x_numeric, y)

# Создание более плотного набора точек для гладкой кривой
x_smooth = np.linspace(x_numeric[0], x_numeric[-1], 241)
y_smooth = cs(x_smooth)
ццццццццц
# Создание графика
axes[1].plot(x_numeric, y, 'D', label='Исходные данные', color = 'red', markersize=10, zorder = 1)
axes[1].plot(x_smooth, y_smooth, label='Интерполированная кривая', linewidth = 6, zorder = 0)
axes[1].set_title('б', fontsize=32)
axes[1].set_xlabel('Время, UT', fontsize=32)
axes[1].set_ylabel('$\delta_1$', fontsize=32)
# plt.title('График интерполяции', fontsize=32)
axes[1].legend(fontsize=25)
axes[1].grid(True)
# Задание пользовательских подписей по оси x
axes[1].set_xticks([x_smooth[0], x_smooth[60], x_smooth[120], x_smooth[180], x_smooth[240]], x)
plt.show()