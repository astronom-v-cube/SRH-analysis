import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from analise_utils import ArrayOperations, Monitoring, MplFunction

MplFunction.set_mpl_rc()
Monitoring.start_log('logs')

def create_distribution_histogram(input_file : str, replace_minus_to_zero = True, only_disk = False):
    """Построение гистограммы распределения значений яркостной температуры в ```.fits``` файле

    Args:
        input_file (str): путь к файлу
        replace_minus_to_zero (bool, optional): параметр, регулирующий замену отрицательных значений на нуль при подсчете. По умолчанию установлен True.
    """

    fig = plt.figure(num="Гистограмма распределения", figsize = (25, 15))
    fig.canvas.manager.window.showMaximized()

    if replace_minus_to_zero:
        data, header = ArrayOperations.replace_minus_to_zero(input_file)
        plt.xlim(0, data.max())
    else:
        hdul = fits.open(input_file)
        data = hdul[0].data
        header = hdul[0].header
        plt.xlim(data.min(), data.max())

    if only_disk:
        data = ArrayOperations.cut_sun_disk_data(data)

    Monitoring.header_info(header)
    plt.hist(data.flatten(), bins=data.shape[0])
    plt.title('Гистограмма распределения')
    plt.xlabel('Значение яркостной температуры')
    plt.ylabel('Количество пикселей')
    plt.yscale("log")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    create_distribution_histogram('test_dataset/srh_20220113T031122_6200_LCP.fit')
