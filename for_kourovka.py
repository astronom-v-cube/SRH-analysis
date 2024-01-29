import ftplib
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.ticker import FuncFormatter

# отображение всех значений матрицы в консоль
np.set_printoptions(threshold=np.inf)

download_and_delete_fits = False
replace_zero_average = True
custom_time_line = False

date = '16.07.2023'

SMALL_SIZE = 13
MEDIUM_SIZE = 25
BIGGER_SIZE = 25

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

date_spl = date.split('.')
filenames = [f'srh_0306_cp_{date_spl[2]}{date_spl[1]}{date_spl[0]}.fits', f'srh_0612_cp_{date_spl[2]}{date_spl[1]}{date_spl[0]}.fits', f'srh_1224_cp_{date_spl[2]}{date_spl[1]}{date_spl[0]}.fits']

if download_and_delete_fits:

    ftp = ftplib.FTP('ftp.rao.istp.ac.ru', 'anonymous', 'anonymous')
    ftp.cwd(f'SRH/corrPlot/{date_spl[2]}/{date_spl[1]}')

    for filename in filenames:
        with open(filename, 'wb') as file:
            try:
                ftp.retrbinary('RETR %s' % filename, file.write)
            except:
                print(f'Файл {filename} не найден на сервере')
                continue

def format_seconds(x, pos):
    hours = int(x // 3600)
    minutes = int((x % 3600) // 60)
    seconds = int(x % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def replace_zero_average(array_time, array_data):
    if replace_zero_average:
        for i in range(2, len(array_time) - 1):

            if array_time[i] == 0 or array_time[i] < array_time[i-1]:

                average_time = (array_time[i - 1] + array_time[i + 1]) / 2.0
                array_time[i] = average_time

                for z in range(3):
                    average_data = (array_data[z][i - 1] + array_data[z][i + 1]) / 2.0
                    array_data[z][i] = average_data

def clear_time_line(array_time):
    for i in range(0, len(array_time) - 1):
        if array_time[i] < array_time[i-1]:
            array_time[i] = None
    return array_time

fig0612, ((ax3)) = plt.subplots(1, 1, sharex=True)

""" index = 0

while index < len(filenames):
    filename = filenames[index]
    if os.path.isfile(filename):
        index += 1
    else:
        filenames.pop(index) """

fits_file_0306, fits_file_0612, fits_file_1224 = filenames[0], filenames[1], filenames[2]

try:
    hdul0612 = fits.open(fits_file_0612)

    freqs_0612 = np.array(hdul0612[1].data.copy(), dtype='float') / 1e6
    data_0612 = hdul0612[2].data.copy()

    hdul0612.close()

except:
    print('Один из файлов не найден, или возникла какая-то ошибка')

# if download_and_delete_fits:
    # for file in filenames:
    #     os.unlink(file)

colors = plt.cm.jet(np.linspace(0, 1, 16))

for freq in range(0, 16):

    if custom_time_line == False:

        replace_zero_average(data_0612[freq][0], data_0612[freq])
        ax3.plot(data_0612[freq][0], data_0612[freq][3], label=f'{freqs_0612[freq]} GHz', color=colors[freq])
        # ax3.plot(data_0612[freq][0], data_0612[freq][4], color=colors[freq])



for ax in [ax3]:

    if ax == 'axx':
        ax.set_ylabel('Correlation coefficient')
        ax.set_title('Correlation plot')
        ax.legend(ncol=4)
    else:
        ax.set_ylabel('S.F.U.', fontsize=25)
        ax.legend(ncol=4)

    ax.xaxis.set_major_formatter(FuncFormatter(format_seconds))
    ax.set_xlabel('UT', fontsize=25)
    ax.grid(True)

# Выведите график
plt.show()