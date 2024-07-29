import matplotlib.pyplot as plt
import pandas as pd

# Пример данных из файла
data = {
    'time': ['08:00:00', '08:15:00', '08:30:00', '08:45:00', '09:00:00'],
    'value': [10, 15, 20, 18, 25]
}

# Преобразование времени из строкового формата в формат времени
data['time'] = pd.to_datetime(data['time'])

# Создание графика
plt.plot(data['time'], data['value'])

# Установка вертикальной черты в заданной координате времени
desired_time = pd.to_datetime('08:52:40')
plt.axvline(x=desired_time, color='r', linestyle='--')

# Настройка формата времени на оси x (необязательно)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))

# Отображение графика
plt.show()
