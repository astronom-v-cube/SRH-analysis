from scipy.io import readsav
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from combination_of_timeless_moments import find_nearest_files
from analise_utils import MplFunction
MplFunction.set_mpl_rc()
sbc_raw = "F:/cbs_data_2hours.sav"
sbc_data = readsav(sbc_raw)

datetime_list = []
base_date = datetime(1900, 1, 1)  # Базовая дата
sbc_freqs = np.linspace(35.25, 39.75, 10)

for row_time in sbc_data['times']:
    hours = int(row_time[0].decode('utf-8'))
    minutes = int(row_time[1].decode('utf-8'))
    seconds = float(row_time[2].decode('utf-8'))

    # Создание объекта datetime с базовой датой
    time = base_date + timedelta(hours=hours, minutes=minutes, seconds=seconds)
    datetime_list.append(time)

if len(datetime_list) == len(sbc_data['datas']):
    for i in range(0, 10, 1):
        plt.plot(datetime_list, sbc_data['datas'].T[i], label=sbc_freqs[i], linewidth=3)
    plt.xlabel('Time')
    plt.ylabel('Data')
    plt.title('Data over Time')
    plt.yscale('log')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()
