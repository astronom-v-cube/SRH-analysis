import matplotlib.pyplot as plt
from astropy.io import fits
from config import *
from matplotlib.widgets import Slider
from matplotlib.colors import TwoSlopeNorm
from analise_utils import Monitoring, OsOperations
import logging
import sys

class Alignment():

    def __init__(self):
        self.directory = directory
        self.files, self.freqs = OsOperations.freq_sorted_files_in_folder(directory) if folder_mode == 'one_folder' else OsOperations.freq_sorted_1st_two_files_in_folders(directory)
        self.iterable = range(0, len(self.files), 2)
        self.im, self.ax, self.click_coords = None, None, None
        self.coordinates_of_control_point, self.click_coords = [], []

    # Функция для обработки событий клика мыши
    def onclick(self, event):
        if event.dblclick:
            # Очистка списка координат кликов
            self.click_coords.clear()
            # Добавление координат клика в список
            self.click_coords.append((int(event.xdata), int(event.ydata)))
            plt.close()

    # Обновление цветовой шкалы и изображения при изменении положения ползунка
    def update(self, val):
        vmin = self.im.norm.vmin
        vmax = self.slider.val
        self.im.set_clim(vmin, vmax)
        self.ax.figure.canvas.draw_idle()

    def interactive_choise_AO(self):

        square_psf = list()

        for i in self.iterable:

            hdul1 = fits.open(f'{self.directory}/{self.freqs[i//2] if folder_mode == "folder_with_folders" else ""}/{self.files[i]}', ignore_missing_simple=True)
            hdul2 = fits.open(f'{self.directory}/{self.freqs[(i+1)//2] if folder_mode == "folder_with_folders" else ""}/{self.files[i+1]}', ignore_missing_simple=True)
            data1 = hdul1[0].data
            data2 = hdul2[0].data

            header1 = hdul1[0].header
            psf_a = header1['PSF_ELLA']
            psf_b = header1['PSF_ELLB']
            square_psf.append(int(3.1415 * psf_a * psf_b))

            I = data1 + data2

            fig, self.ax = plt.subplots(figsize=(12, 12))
            vcenter = -500 * (i + 1) + 80000
            vcenter = - 1500 * (i + 1) + 200000
            self.im = self.ax.imshow(I, origin='lower', cmap='plasma', extent=[0, I.shape[1], 0, I.shape[0]], norm=TwoSlopeNorm(vmin=0, vcenter=vcenter, vmax=200000))
            cropped_I = I[y_limit[0]:y_limit[1], x_limit[0]:x_limit[1]]
            # contour_levels = np.linspace(cropped_I.max() * 0.5, cropped_I.max(), 7)
            contour_levels = [cropped_I.max() * 0.5]
            self.ax.contour(cropped_I, levels=contour_levels, colors='k', extent=[x_limit[0], x_limit[1], y_limit[0], y_limit[1]])
            self.ax.set_xlim(x_limit) if len(x_limit) != 0 else logging.info(f'Limits for X not found')
            self.ax.set_ylim(y_limit) if len(y_limit) != 0 else logging.info(f'Limits for Y not found')
            # mplcursors.cursor(hover=True)
            plt.title(f'{self.files[i][:-4] + "  +  " + self.files[i+1][:-4]}')
            # fig.colorbar(im)
            # fig.tight_layout()

            # Создание слайдера для редактирования границ цветовой шкалы
            ax_slider = plt.axes([0.25, 0.03, 0.5, 0.01], facecolor='lightgoldenrodyellow')
            self.slider = Slider(ax_slider, 'Threshold', self.im.norm.vmin, self.im.norm.vmax, valinit=self.im.norm.vmax/2)
            self.slider.on_changed(self.update)
            # Привязка обработчика событий клика мыши к графику
            cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
            # Отображение на полный экран
            mng = plt.get_current_fig_manager()
            mng.full_screen_toggle()
            plt.show()

            # Вывод последней координаты двойного клика пользователя
            if len(self.click_coords) > 0:
                Monitoring.logprint(f"For image {i+1} last double click coordinates: {str(self.click_coords[-1])}")
                self.coordinates_of_control_point.append(self.click_coords[-1])
                self.coordinates_of_control_point.append(self.click_coords[-1])
                plt.close()

            else:
                Monitoring.logprint(f"No double click coordinates recorded")
                sys.exit()

        return self.coordinates_of_control_point, square_psf

    def create_intensity_list(self):
        intensity_maps_list = list()
        for i in self.iterable:

            hdul1 = fits.open(f'{directory}/{self.freqs[i//2] if folder_mode == "folder_with_folders" else ""}/{self.files[i]}', ignore_missing_simple=True)
            hdul2 = fits.open(f'{directory}/{self.freqs[(i+1)//2] if folder_mode == "folder_with_folders" else ""}/{self.files[i+1]}', ignore_missing_simple=True)
            data1 = hdul1[0].data
            data2 = hdul2[0].data

            I = data1 + data2
            intensity_maps_list.append(I)

        return intensity_maps_list