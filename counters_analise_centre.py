import os

import matplotlib
import matplotlib.patches as patches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from astropy.io import fits
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

matplotlib.rcParams.update({'font.size': 20})

# Путь к папке с данными
data_path = r"D:/datasets/16.07.23/counters"

# Частота наблюдений
vcenter = 200000

# Функция для обработки одного момента времени
def process_time_moment(folder, frequency):
    lcp_file = None
    rcp_file = None

    # Поиск нужных файлов в папке
    for file in os.listdir(folder):
        if f"{frequency}_LCP_calibrated_brightness._aligned.fits" in file:
            lcp_file = os.path.join(folder, file)
        if f"{frequency}_RCP_calibrated_brightness._aligned.fits" in file:
            rcp_file = os.path.join(folder, file)

    if not lcp_file or not rcp_file:
        return None

    # Чтение данных из файлов
    lcp_data = fits.getdata(lcp_file)
    rcp_data = fits.getdata(rcp_file)

    # Суммирование данных
    summed_data = lcp_data + rcp_data

    return summed_data

def calculate_weighted_centroid(data, contour):
    # Создание пути контура
    contour_path = mpath.Path(contour)

    # Создание сетки координат
    y, x = np.mgrid[:data.shape[0], :data.shape[1]]
    points = np.vstack((x.flatten(), y.flatten())).T

    # Создание маски для точек внутри контура
    mask = contour_path.contains_points(points).reshape(data.shape)

    # Извлечение координат и значений данных внутри контура
    x_coords = x[mask]
    y_coords = y[mask]
    values = data[mask]

    # Вычисление средневзвешенного значения для координат x и y
    weighted_x = np.average(x_coords, weights=values)
    weighted_y = np.average(y_coords, weights=values)

    return weighted_x, weighted_y

# Функция для создания анимации
def create_animation(data_list, folders, frequency, levels=[0.5, 0.75, 0.8, 0.9, 0.975]):
    fig, ax = plt.subplots()
    # Создаем список для хранения координат точки максимума
    max_points = []
    weighted_centroids = []

    def update(frame):
        ax.clear()
        data = data_list[frame]
        max_value = np.max(data)
        contour_levels = [level * max_value for level in levels]
        contours = ax.contour(data, levels=contour_levels, colors='red')

        # Поиск центра последнего уровня
        if len(contours.allsegs[-1]) > 0:
            largest_contour = contours.allsegs[-1][0]
            contour05 = contours.allsegs[0][0]
            centroid = np.mean(largest_contour, axis=0)

            weighted_centroid = calculate_weighted_centroid(data, contour05)
            weighted_centroids.append(weighted_centroid)

            ax.plot(centroid[0], centroid[1], 'bx')
            max_points.append(centroid)
            ax.imshow((data)[0:1024, 0:1024], origin='lower', cmap='plasma',  norm=TwoSlopeNorm(vmin=5000, vcenter=vcenter), extent=[0, data.shape[1], 0, data.shape[0]])
            ax.set_xlim(320, 335)
            ax.set_ylim(623, 639)

        ax.set_title(f"{folders[frame][30:45]}")

    ani = FuncAnimation(fig, update, frames=len(data_list), repeat=True)

    matplotlib.rcParams['animation.ffmpeg_path'] = "ffmpeg.exe"
    writer = animation.FFMpegWriter(fps = 1.5, bitrate=2500)
    ani.save(f'analise/videos/{frequency}.mp4', writer=writer)

    # if max_points:
    #     fig = plt.figure(figsize=(12, 8))
    #     ax = plt.gca()
    #     max_points = np.array(max_points)
    #     ax.plot(max_points[:, 0]*2.45, max_points[:, 1]*2.45, 'bo-')
    #     ax.set_title(f"Trajectory of Maximum Point on {frequency}", fontsize=30)
    #     for i, point in enumerate(max_points):
    #         ax.annotate(str(i + 1), (point[0], point[1]), textcoords="offset points", xytext=(5, 5), ha='center')
    #     plt.savefig(f"analise/pics/{frequency}.png", dpi=750)

    if max_points:

        legends = ["1 - 08:23:51", "2 - 08:24:00", "3 - 08:24:08", "4 - 08:24:13", "5 - 08:24:20", "6 - 08:24:39", "7 - 08:24:46", "8 - 08:25:00", "9 - 08:25:12", "10 - 08:25:24", "11 - 08:25:41", "12 - 08:25:45", "13 - 08:26:00"]

        fig = plt.figure(figsize=(12, 12))
        ax = plt.gca()
        max_points = np.array(max_points)
        weighted_centroids = np.array(weighted_centroids)

        # Смещаем центр координат на центр изображения и переводим в секунды дуги
        max_points_arcsec = (max_points[1:] - np.array([512, 512])) * 2.45
        weighted_centroids_arcsec = (weighted_centroids[1:] - np.array([512, 512])) * 2.45

        cmap = matplotlib.colormaps.get_cmap('brg')
        arrowprops = {'arrowstyle': '->', 'connectionstyle': 'angle3'}

        # ax.plot(max_points_arcsec[:, 0], max_points_arcsec[:, 1], 'bo-', linewidth = 4)
        for i, point in enumerate(weighted_centroids_arcsec):
            color = cmap(i / len(weighted_centroids_arcsec))
            ax.scatter(weighted_centroids_arcsec[i][0], weighted_centroids_arcsec[i][1], label = legends[i], s=150, color=color, alpha=1, zorder = 5)
            ax.annotate(str(i + 1), (point[0], point[1]), fontweight="bold", textcoords="offset points", xytext=(10, 10), ha='left' if i < 10 else 'right', zorder=8)

        rectangle = patches.Rectangle((-460.6 - (7.35/2), 284.2 - (7.35/2)), 7.35, 7.35, linewidth=3, edgecolor='r', facecolor='none', zorder = 5)
        ax.add_patch(rectangle)
        ax.plot(weighted_centroids_arcsec[:, 0], weighted_centroids_arcsec[:, 1], '--', linewidth = 3, c='olivedrab', zorder = 3)

        import matplotlib.ticker as ticker
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.grid(which='major', color = 'k', linewidth = 1.5, zorder=0)
        # ax.set_xlim(max_points_arcsec[:, 0].min(), max_points_arcsec[:, 0].max())
        # ax.set_ylim(max_points_arcsec[:, 1].min(), max_points_arcsec[:, 1].max())
        # ax.set_xlim(-464.5, -457.5)
        # ax.set_ylim(281.5, 288.5)

        ax.set_xlim(-464.5, -455)
        ax.set_ylim(280, 289.5)
        ax.set_xlabel('X $(arcsec)$', fontsize=25)
        ax.set_ylabel('Y $(arcsec)$', fontsize=25)

        plt.legend(fontsize=20, framealpha=1, loc='center right')
        plt.tight_layout()
        plt.savefig(f"analise/pics/{frequency/1000}.png", dpi=750)
        plt.savefig(f"analise/pics/{frequency/1000}.eps", dpi=750)

    plt.close()

# Список папок с данными
folders = sorted([os.path.join(data_path, folder) for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, folder))])

# [2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 5200, 5400, 5600, 5800, 6200, 6600, 7000, 7400, 7800, 8200, 8600, 9000, 9400, 10200, 10600, 11000, 11400, 11800]

for frequency in tqdm.tqdm([11800]):

    # Обработка данных для всех моментов времени
    data_list = []
    for folder in folders:
        data = process_time_moment(folder, frequency)
        if data is not None:
            data_list.append(data)

    # Создание анимации
    create_animation(data_list, folders, frequency)
