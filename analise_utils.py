import re, os
import shutil
from typing import List, Union, Tuple
import warnings
from scipy import constants
import scipy.optimize as opt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter, NullFormatter
import numpy as np
import logging

class Extract:

    def __init__(self) -> None:
        self.freqs_pattern = re.compile(r'(?<=[_.])\d{4,5}(?=[_.])')   # паттерн извлечения частоты
        self.datetime_pattern = re.compile(r'\d{8}T\d{6}')             # паттерн для извлечения даты и времени
        self.polarization_pattern = re.compile(r'(RCP|LCP|R|L)')       # паттерн извлечения поляризации

    def extract_number(self, filename: str, freqs_array_name: set = None) -> int:
        """Функция для извлечения цифр - номеров частот из названия файла. Если передать в качестве второго аргумента ```set()```, то он будет наполняться выделенными номерами частот

        Args:
            filename (str): название ```.fits``` файла с данными
            freqs_array_name (set, optional): для наполнения. По умолчанию None.

        Returns:
            int: выделенное число
        """
        match = self.freqs_pattern.search(filename)
        if freqs_array_name is not None:
            freqs_array_name.add(int(match.group()))
        return int(match.group())

    def extract_datetime(self, filename: str) -> str:
        """Функция для извлечения даты и времени из названия файла.

        Args:
            filename (str): название ```.fits``` файла с данными или путь директории, вообщем - любая строка, содержащая дату и время в таком формате

        Returns:
            str: строка, содержащая извлеченную дату и время в формате "ГГГГММДДТЧЧММСС"
        """
        match = self.datetime_pattern.search(filename)
        return match.group()

class ZirinTb:
# Zirin H. et al.
# The Microwave Brightness Temperature Spectrum of the Quiet Sun
# The Astrophysical Journal. 1991

    def fitFunc(self, f, A, B, C):
        return A + B*f + C*f**-1.8

    def __init__(self):

        self.frequency = np.array([1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2, 3.6, 4.2, 5.0, 5.8, 7.0, 8.2, 9.4, 10.6, 11.8, 13.2, 14.8, 16.4, 18.0]) # frequency [GHz]
        self.Tb = np.array([70.5, 63.8, 52.2, 42.9, 32.8, 27.1, 24.2, 21.7, 19.4, 17.6, 15.9, 14.1, 12.9, 12.2, 11.3, 11.0, 10.8, 10.8, 10.7, 10.3]) # brightness temperature [1e3K]
        self.guess = [1, 1, 1]
        self.fitTbParams, _ = opt.curve_fit(self.fitFunc, self.frequency, self.Tb, p0=self.guess)
        self.solarDiskRadius = np.deg2rad(900/3600)

    def getTbAtFrequency(self, f):
        return self.fitFunc(f, self.fitTbParams[0],self.fitTbParams[1],self.fitTbParams[2])

    def getSfuAtFrequency(self, f):
        return 2*constants.k*self.getTbAtFrequency(f)*1e3/(constants.c/(f*1e9))**2 * np.pi*self.solarDiskRadius**2 / 1e-22

class FindIntensity:
    """
    Получение суммарной интенсивности из пикселя, пикселя и окружающей области (4, 9, size), от всего Солнца
    """

    @staticmethod
    def find_intensity_in_point(matrix: np.ndarray, point: tuple) -> float:
        """Интенсивность из одного пикселя

        Args:
            matrix (np.ndarray): массив данных изображения
            point (tuple): координаты точки

        Returns:
            float: интенсивность из пикселя
        """
        x, y = point
        return matrix[y][x]

    @staticmethod
    def find_intensity_in_four_point(matrix: np.ndarray, point: tuple) -> float:
        """Интенсивность из четырех пикселей. Берется указанная точка + три точки с правой стороны от него

        Args:
            matrix (np.ndarray): массив данных изображения
            point (tuple): координаты точки

        Returns:
            float: интенсивность из пикселя
        """
        x, y = point
        intensity_sum = matrix[y][x] + matrix[y][x+1] + matrix[y+1][x] + matrix[y+1][x+1]
        intensity_avg = intensity_sum
        return intensity_avg

    @staticmethod
    def find_intensity_in_nine_point(matrix: np.ndarray, point: tuple) -> float:
        """Интенсивность из девяти пикселей. Берется указанная точка + восемь точек вокруг нее

        Args:
            matrix (np.ndarray): массив данных изображения
            point (tuple): координаты точки

        Returns:
            float: интенсивность из пикселя
        """
        x, y = point
        intensity_sum = matrix[y+1][x-1] + matrix[y+1][x] + matrix[y+1][x+1] + matrix[y][x-1] + matrix[y][x] + matrix[y][x+1] + matrix[y-1][x-1] + matrix[y-1][x] + matrix[y-1][x+1]
        intensity_avg = intensity_sum
        return intensity_avg

    @staticmethod
    def find_intensity_in_alotof_point(matrix: np.ndarray, point: tuple) -> float:
        x, y = point
        data_array = matrix[y-20:y+20, x-12:x+12]
        fig = plt.figure()
        plt.imshow(data_array)
        plt.plot()
        mask = data_array > 0
        data_array = data_array[mask]
        intensity_avg = np.sum(data_array)
        return intensity_avg

    @staticmethod
    def find_intensity_in_sun_disk(matrix: np.ndarray, point: tuple) -> float:
        x1, y1 = np.indices(matrix.shape)
        center_x, center_y = 512, 512
        radius = 405

        # вычисляем расстояние от каждой точки до центра окружности
        distance_from_center = np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)
        mask = distance_from_center <= radius
        data_inside_circle = matrix[mask]
        mask = data_inside_circle > 0
        data_inside_circle = data_inside_circle[mask]
        print(len(data_inside_circle))
        intensity_avg = np.sum(data_inside_circle)
        return intensity_avg

class ArrayOperations:

    @staticmethod
    def find_max_around_point(matrix : np.ndarray, point : tuple, size : int) -> tuple:
        """Функция выполняет поиск максимального значения в матрице определенного размера вокруг заданной точки.

        Args:
            matrix (np.ndarray):  матрица (двумерный массив) - изображение
            point (tuple): указанная точка - координаты
            size (int): размер области поиска

        Returns:
            tuple: кортеж из трех значений: строкa ```max_row``` и столбец ```max_col``` максимального значения, а также значение ```max_value``` самого большого элемента в указанной области.
        """
        # Get the indices of the smaller region around the point
        row, col = point
        half_size = size // 2
        row_indices = range(max(0, row - half_size), min(matrix.shape[0], row + half_size + 1))
        col_indices = range(max(0, col - half_size), min(matrix.shape[1], col + half_size + 1))

        # Get the smaller region from the original matrix
        smaller_matrix = matrix[np.ix_(row_indices, col_indices)]

        # Find the maximum value in the smaller region
        max_value = np.max(smaller_matrix)

        # Get the coordinates of the maximum value in the original matrix
        max_indices = np.unravel_index(np.argmax(smaller_matrix), smaller_matrix.shape)
        max_row = row_indices[max_indices[0]]
        max_col = col_indices[max_indices[1]]

        return (max_row, max_col, max_value)

    @staticmethod
    def find_center_of_mass(matrix: np.ndarray, point: tuple, size: int) -> tuple:
        """Функция выполняет поиск центра тяжести области методом взвешенного среднего. Функция вычисляет матрицу меньшего размера вокруг указанной точки и находит максимальное значение в этой области. Затем она устанавливает пороговое значение как фракцию максимального значения и находит индексы элементов матрицы, которые превышают этот порог. Если ни один элемент не превышает порог, центр тяжести вычисляется как центр области. В противном случае, используются индексы элементов, превышающих порог, для вычисления центра тяжести методом среднего.

        Args:
            matrix (np.ndarray):  матрица (двумерный массив)
            point (tuple): указанная точка - координаты
            size (int): размер области поиска

        Returns:
            tuple: кортеж из двух значений: строка ```center_row``` и столбец ```center_col``` координат центра тяжести, определенного по элементам матрицы в указанной области, а также значение ```max_value``` самого большого элемента в указанной области
        """
        # Get the indices of the smaller region around the point
        row, col = point
        half_size = size // 2
        row_indices = range(max(0, row - half_size), min(matrix.shape[0], row + half_size + 1))
        col_indices = range(max(0, col - half_size), min(matrix.shape[1], col + half_size + 1))

        # Get the smaller region from the original matrix
        smaller_matrix = matrix[np.ix_(row_indices, col_indices)]
        max_value = np.max(smaller_matrix)

        # Calculate threshold as a fraction of the maximum value in the matrix
        threshold = np.max(smaller_matrix) * 0.1  # for example, we can use 10% of the maximum value as threshold

        # Find indices of elements above the threshold
        high_value_indices = np.where(smaller_matrix > threshold)

        # If no elements exceed the threshold, return the center of the region
        if len(high_value_indices[0]) == 0:
            center_row = row_indices[len(row_indices) // 2]
            center_col = col_indices[len(col_indices) // 2]
        else:
            # Calculate center of mass using indices of elements above threshold
            center_row = np.mean(row_indices[high_value_indices[0]])
            center_col = np.mean(col_indices[high_value_indices[1]])

        return (center_row, center_col, max_value)

    @staticmethod
    def calculate_weighted_centroid(matrix: np.ndarray, point: tuple, target_level: float, xy_limits : tuple = None) -> tuple:
        """Функция выполняет вычисление средневзвешенного значения координат внутри контура вокруг заданной точки. Внимание! Контур всегда должен быть полностью отображен в области поиска!

        Args:
            matrix (np.ndarray): матрица (двумерный массив) - изображение
            point (tuple): указанная точка - координаты
            level (float): уровень контура
            xy_limits (turple): область поиска контура (по умолчанию - все изображение)

        Returns:
            tuple: кортеж из двух значений: координаты (```weighted_x```, ```weighted_y```) - средневзвешенные координаты внутри контура.

        Raises:
            ValueError: если точка не находится внутри контура.
        """
        if xy_limits != None:
            crop_matrix = matrix[xy_limits[1][0]:xy_limits[1][1], xy_limits[0][0]:xy_limits[0][1]]
        else:
            crop_matrix = matrix
        contours = plt.contour(crop_matrix, levels=[crop_matrix.max() * target_level], extent=[xy_limits[0][0], xy_limits[0][1], xy_limits[1][0], xy_limits[1][1]])

        # Найти контур нужного уровня, содержащий точку
        point_in_contour = False
        for collection, level in zip(contours.collections, contours.levels):
            if level == crop_matrix.max() * target_level:
                for path in collection.get_paths():
                    if path.contains_point(point):
                        contour_path = path
                        point_in_contour = True
                        break
                if point_in_contour:
                    break

        if not point_in_contour:
            raise ValueError(f"Точка не находится внутри контура уровня {target_level}")

        # Создание сетки координат
        y, x = np.mgrid[:matrix.shape[0], :matrix.shape[1]]
        points = np.vstack((x.flatten(), y.flatten())).T

        # Создание маски для точек внутри контура
        mask = contour_path.contains_points(points).reshape(matrix.shape)

        # Извлечение координат и значений данных внутри контура
        x_coords = x[mask]
        y_coords = y[mask]
        values = matrix[mask]

        weighted_x = np.average(x_coords, weights=values)
        weighted_y = np.average(y_coords, weights=values)

        return (int(round(weighted_y)), int(round(weighted_x)), values.max())

class MplFunction:

    @staticmethod
    def set_mpl_rc():
        """Функция, применяющая выверенные стандарты отображения MPL для удобства: размеры шрифтов, линий и пр."""

        SMALL_SIZE = 18
        MEDIUM_SIZE = 20
        BIGGER_SIZE = 25

        plt.rc('font', size=MEDIUM_SIZE)                    # размер шрифта по умолчанию
        plt.rc('axes', titlesize=BIGGER_SIZE)               # размер шрифта заголовков
        plt.rc('axes', labelsize=MEDIUM_SIZE)               # размер шрифта подписей осей
        plt.rc('xtick', labelsize=SMALL_SIZE)               # размер шрифта меток по оси x
        plt.rc('ytick', labelsize=SMALL_SIZE)               # размер шрифта меток по оси y
        plt.rc('legend', fontsize=MEDIUM_SIZE)              # размер шрифта легенды
        plt.rc('figure', titlesize=BIGGER_SIZE)             # размер шрифта заголовка фигуры
        plt.rc('lines', linewidth=6)                        # толщина линий
        plt.rc('grid', linestyle='--', linewidth=1)       # стиль и толщина линий сетки
        plt.rc('axes', grid=True)                           # отображение сетки на осях

        # Отключение предупреждений о попытке установки неположительных границ оси на логарифмической шкале при очистке графика
        warnings.filterwarnings('ignore', message='UserWarning: Attempt to set non-positive xlim on a log-scaled axis will be ignored.')

    @staticmethod
    def remove_ticks_in_log(axes):
        """shit code - не работает на 2+ осях, не ведаю почему, но код пусть будет
        Args:
            axes (_type_): _description_
        """
        axes.xaxis.set_major_formatter(ScalarFormatter())
        axes.xaxis.set_minor_formatter(NullFormatter())
        axes.xaxis.set_minor_locator(ticker.NullLocator())

class Monitoring:

    @staticmethod
    def start_log(name_file):
        logging.basicConfig(filename = f'{name_file}.log',  filemode='a', level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
        # encoding = "UTF-8"

    @staticmethod
    def logprint(info_msg):
        print(info_msg)
        logging.info(info_msg)

class ConvertingArrays:
    """
    Преобразования массивов - сглаживание, аппроксимация, вычет подложки, пересчет в другую систему единиц, преобразвание в принт-строку, ...
    """
    @staticmethod
    def arr2str4print(arr: Union[np.ndarray, List]) -> str:
        """Преобразование массива чисел в принт-строку

        Args:
            arr (ndarray): массив чисел

        Returns:
            str: строка в удобном для вывода / логирования виде
        """
        return ", ".join(arr.astype(str))

    # def arr2str4print(arr):
    # return np.array2string(arr, separator=', ', formatter={'float_kind': lambda x: "%.2f" % x})

    @staticmethod
    def running_mean(data: np.ndarray, window_size: int) -> np.ndarray:
        """Сглаживание массива скользящим усредняющим окном в указанное число элементов

        Args:
            array (np.ndarray): исходный массив

        Returns:
            np.ndarray: сглаженный массив
        """
        window = np.ones(window_size) / window_size
        smoothed_data = np.convolve(data, window, mode='same')
        return smoothed_data

    @staticmethod
    def variable_running_mean(array: np.ndarray) -> np.ndarray:
        """Сглаживание массива скользящим усредняющим окном от 1 до 3 элементов ("окно" для вычисления среднего значения фактически изменяется в зависимости от позиции текущего элемента данных, для крайних значений данных используется среднее значение только из соседних элементов)

        Args:
            array (np.ndarray): исходный массив

        Returns:
            np.ndarray: сглаженный массив
        """
        smoothed_data = []
        for i in range(len(array)):
            if i == 0:
                smoothed_data.append((array[i] + array[i + 1]) / 2)
            elif i == len(array) - 1:
                smoothed_data.append((array[i - 1] + array[i]) / 2)
            else:
                smoothed_data.append((array[i - 1] + array[i] + array[i + 1 ]) / 3)

        return np.array(smoothed_data)

    @staticmethod
    def gs_approximation(array_sfu : np.ndarray, array_freqs : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Аппроксимация массива уравнением, описывающим спектр гиросинхротронного излучения (Peak Frequency Dynamics in Solar Microwave Bursts, 2008). На выходе первый[0] элемент - аппроксимированный массив, второй[1] - список восстановленных параметров уравнения: [a1, a2, a3, a4]

        Args:
            array_sfu (np.ndarray): исходный массив интенсивностей
            array_freqs (np.ndarray): исходный массив частот

        Returns:
            np.ndarray: аппроксимированный массив
            np.ndarray: восстановленные параметры уравнения
        """

        def spectrum_function(array_freqs, a1, a2, a3, a4):
            return a1 * array_freqs**a2 * (1 - np.exp(-a3 * array_freqs**-a4))
        popt, pcov = curve_fit(spectrum_function, array_freqs, array_sfu)
        a1, a2, a3, a4 = popt
        print(f'Coeff: a1, a2, a3, a4 = {a1, a2, a3, a4}')
        approximation_arr = spectrum_function(array_freqs, *popt)
        return ((approximation_arr, np.array([a1, a2, a3, a4])))

    @staticmethod
    def background_subtraction(arr_left_flux: np.ndarray, arr_right_flux: np.ndarray, arr_freqs: np.ndarray):
        """Вычет подложки из области вспышки по интерполированной функции Zirin at el. Функция возвращает исходные массивы преобразованными

        Args:
            arr_left_flux (ndarray): массив данных левополяризованного излучения
            arr_right_flux (ndarray): массив данных правополяризованного излучения
            arr_freqs (ndarray): массив частот наблюдений

        Returns:
            ndarray: 2 преобразованных массива
        """
        zirin = ZirinTb()
        for index, freq in enumerate(arr_freqs):
            arr_left_flux[index]  = (arr_left_flux[index]  - zirin.getTbAtFrequency(freq/1000))
            arr_right_flux[index] = (arr_right_flux[index] - zirin.getTbAtFrequency(freq/1000))
        logging.info(f'Flux in sfu for LCP minus solar disk flux: [{ConvertingArrays.arr2str4print(arr_left_flux)}]')
        logging.info(f'Flux in sfu for RCP minus solar disk flux: [{ConvertingArrays.arr2str4print(arr_right_flux)}]')
        return arr_left_flux, arr_right_flux

    @staticmethod
    def Tb2sfu(arr_one_polarization: Union[np.ndarray, List], arr_freqs: Union[np.ndarray, List]) -> np.ndarray:
        """Функция пересчета яркостной температуры в плотность солнечного потока ```sfu```

        Args:
            arr_one_polarization (Union[np.ndarray, List]): массив одной из поляризаций (левая/правая)
            arr_freqs (Union[np.ndarray, List]): массив частот наблюдений

        Returns:
            np.ndarray: преобразованный массив
        """
        return (((2 * 1.38*1e-16 * (np.array(arr_freqs) * 1e6) ** 2) / (3e10)**2) * np.array(arr_one_polarization) * ((2.4/3600*0.01745)**2) * 1e19)

class OsOperations:

    @staticmethod
    def create_place(path: str, postfix: str = ''):
        """Создает папку с указанным путем, если она уже существует - удаляет ее и создает заново. Вторым агрументом можно передать постфикс для пути

        Args:
            path (str): путь к месту расположения
            postfix (str, optional): постфикс для пути, отделяется от пути нижнем подчеркиванием ```_```. По умолчанию пустая строка: ''
        """
        new_path = f'{path}_{postfix}' if postfix else path
        try:
            os.mkdir(new_path)
        except FileExistsError:
            shutil.rmtree(new_path)
            os.mkdir(new_path)

    @staticmethod
    def abс_sorted_files_in_folder(path_to_folder) -> List[str]:
        """Создает список файлов в указанной папке, отсортированных по АЛФАВИТУ / ВРЕМЕНИ

        Args:
            path_to_folder (str): путь к месту расположения файлов
        Returns:
            files (list): список файлов в указанной папке
        """
        files = sorted(os.listdir(path_to_folder))
        return files

    @staticmethod
    def freq_sorted_files_in_folder(path_to_folder) -> Tuple[List[str], List[int]]:
        """Создает список файлов в указанной папке, отсортированных по ЧАСТОТАМ

        Args:
            path_to_folder (str): путь к месту расположения файлов
        Returns:
            files (list): список файлов в указанной папке
            freqs (list): список частот от меньшей к большей
        """
        extract = Extract()
        freqs = set()
        files = sorted(os.listdir(path_to_folder), key=lambda x: extract.extract_number(x, freqs))
        freqs = sorted(list(freqs))
        return files, freqs

    @staticmethod
    def freq_sorted_1st_two_files_in_folders(path_to_folder : str, type_path : str = 'relative') -> Tuple[List[str], List[int]]:
        """Создает список из двух первых (по алфавиту/времени) файлов в каждой из подпапок в указанной папке, отсортированный по ЧАСТОТАМ

        Args:
            path_to_folder (str): путь к месту расположения папок с файлами
            type_path (str): желаемый путь на выходе: только имена файлов ```relative``` или полный ```full```
        Returns:
            files (list): список из двух первых файлов в каждой из подпапок
            freqs (list): список частот от меньшей к большей
        """
        extract = Extract()
        freqs = set()
        list_of_1st_two_files = []
        list_folders_in_folder = sorted(os.listdir(path_to_folder))
        for freq_folder in list_folders_in_folder:
            list_of_files = sorted(os.listdir(f'{path_to_folder}/{freq_folder}'))
            try:
                list_of_1st_two_files.append(f'{path_to_folder}/{freq_folder}/{list_of_files[0]}') if type_path == 'full' else list_of_1st_two_files.append(f'{list_of_files[0]}')
                list_of_1st_two_files.append(f'{path_to_folder}/{freq_folder}/{list_of_files[1]}') if type_path == 'full' else list_of_1st_two_files.append(f'{list_of_files[1]}')
            except IndexError as err:
                Monitoring.logprint(f"Похоже, есть пустая папка. Проверьте папку {freq_folder}. Ошибка: {err}")
        files = sorted(list_of_1st_two_files, key=lambda x: extract.extract_number(x, freqs))
        freqs = sorted(list(freqs))
        return files, freqs
