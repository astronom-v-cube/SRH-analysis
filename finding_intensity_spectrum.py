import logging
import os
import re
import sys
import platform

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib import ticker
from matplotlib.ticker import NullFormatter, ScalarFormatter
from tqdm import tqdm
from scipy.io import readsav
from datetime import datetime, timedelta


from analise_utils import (ConvertingArrays, Extract, FindIntensity, Monitoring, MplFunction, OsOperations, ZirinTb)
from combination_of_timeless_moments import find_nearest_files

extractor = Extract()
zirin = ZirinTb()
MplFunction.set_mpl_rc()
Monitoring.start_log('14052024')
logging.info(f'Start of the program to search intensity spectrum of a sun')

############################################
##### Values #####
beginning_of_path = '/mnt/astro_data' if platform.system() == 'Linux' else 'F:'
sbc_raw = f'{beginning_of_path}/cbs_data_2hours.sav'

times_list = [
    '20240514T020434',
    # '20240514T020446',
    # '20240514T020456',
    # '20240514T020501',
    # '20240514T020520',
    # '20240514T020530',
    # '20240514T020537',
    # '20240514T020544',
    # '20240514T020548',
    # '20240514T020555',
    # '20240514T020606',
    # '20240514T020613',
    # '20240514T020623',
    # '20240514T020641',
    # '20240514T020650',
    # '20240514T020657',
    ]

# times_list = [
#     '20240514T013000',
#     '20240514T012000',
#     ]


directories = [None] * len(times_list)
for index, time in enumerate(times_list):
    directories[index] = f'J:/20240514_hr_times/{time}'

polynomial_degree = 4
# coordinates = (324, 628)
# coordinates = (890, 563)
# coordinates = (893, 565)
# coordinates = (659, 300)
coordinates = (883, 394)
##### Params #####
running_mean = False
polinom_approx = False
gs_approx = False  #чет не работает
gm_approx = True
psf_calibration = False
background_Zirin_subtraction = False
intensity_plot = True
save_graphs = True
use_sbc_data = True
number_of_used_pixel = 9
alternation_subscription_x = True
#############################################

if save_graphs:
    if not os.path.isdir('intensity_graphs'):
        OsOperations.create_place('intensity_graphs')

if psf_calibration:
    psf_npz_file = np.load('psf_square.npz')
    freqs_npz = sorted(psf_npz_file['freqs'])
    psf_square = psf_npz_file['psf_square']


for directory in tqdm(directories, desc='Times analise', position=0, leave=True):
    print(directory.split('/')[-1])

    if os.path.isdir(directory) == False:
        try:
            print('_'.join(directory.rsplit('_', 1)[:-1]))
            find_nearest_files('_'.join(directory.rsplit('_', 1)[:-1]), directory.split('/')[-1], directory)
        except Exception as err:
            print(err)
            print('Папка не существует, создать не удалось')
            sys.exit()

    logging.info(f'Path to files: {directory}')
    freqs = set()
    files = sorted(os.listdir(directory), key=lambda x: extractor.extract_number(x, freqs))
    logging.info(f'Find {len(files)} files')
    logging.info(f'List files: \n {files}')
    freqs = np.array(sorted(list(freqs)))
    if psf_calibration == True:
        if (freqs_npz != freqs).any():
            logging.warning(f"Freqs in npz is not freqs in directory")
            sys.exit()

    logging.info(f'Working with freqs: [{ConvertingArrays.arr2str4print(freqs)}]')

    intensivity_list_L, intensivity_list_R = [], []

    for image_index, image_name in enumerate(files):

        try:
            r_or_l = re.search(r'(RCP|LCP|R|L)', str(files[image_index])).group()
        except AttributeError:
            logging.warning(f'{files[image_index]} - where is the polarization in the name? interrupting work...')
            sys.exit()

        data = fits.open(f'{directory}/{files[image_index]}', ignore_missing_simple=True)
        img = data[0].data
        data.close()

        if number_of_used_pixel == 1:
            intensivity = FindIntensity.find_intensity_in_point(img, coordinates)
        elif number_of_used_pixel == 4:
            intensivity = FindIntensity.find_intensity_in_four_point(img, coordinates)
        elif number_of_used_pixel == 9:
            intensivity = FindIntensity.find_intensity_in_nine_point(img, coordinates)

        logging.info(f'{files[image_index]} - {intensivity}')

        if r_or_l == 'RCP' or r_or_l == 'R':
            intensivity_list_R.append(intensivity)
        elif r_or_l == 'LCP' or r_or_l == 'L':
            intensivity_list_L.append(intensivity)

    intensivity_list_L, intensivity_list_R = np.array(intensivity_list_L), np.array(intensivity_list_R)

    if background_Zirin_subtraction == True:
        intensivity_list_L, intensivity_list_R = ConvertingArrays.background_subtraction(intensivity_list_L, intensivity_list_R, freqs)

    # Конвертация яркостной температуры в плотность потока
    flux_density_left  = ConvertingArrays.Tb2sfu(intensivity_list_L, freqs)
    flux_density_right = ConvertingArrays.Tb2sfu(intensivity_list_R, freqs)
    logging.info(f'The basic value flux in sfu for LCP: [{ConvertingArrays.arr2str4print(flux_density_left)}]')
    logging.info(f'The basic value flux in sfu for RCP: [{ConvertingArrays.arr2str4print(flux_density_right)}]')

    if use_sbc_data:
        sbc_data = readsav(sbc_raw)['datas'].T
        datetime_list = []
        base_date = datetime(int(directory.split('/')[-1][0:4]), int(directory.split('/')[-1][4:6]), int(directory.split('/')[-1][6:8]))
        sbc_freqs = np.linspace(35250, 39750, 10)

        for row_time in readsav(sbc_raw)['times']:
            hours = int(row_time[0].decode('utf-8'))
            minutes = int(row_time[1].decode('utf-8'))
            seconds = float(row_time[2].decode('utf-8'))
            time = base_date + timedelta(hours=hours, minutes=minutes, seconds=seconds)
            datetime_list.append(time)

        # Найти индекс ближайшего времени
        target_time = datetime.strptime(directory.split('/')[-1], "%Y%m%dT%H%M%S")
        time_diffs = [abs((dt - target_time).total_seconds()) for dt in datetime_list]
        nearest_index = int(np.argmin(time_diffs))
        nearest_time = datetime_list[nearest_index]

        # Получить частотные данные для ближайшего момента
        sbc_values = sbc_data[:, nearest_index]  # shape (10,), соответствующее sbc_freqs
        correct_sbc = np.array([2743, 2865, 2952, 3059, 3088, 3224, 3231, 3365, 3544, 3593])
        sbc_values = (sbc_values - correct_sbc) * 0.065
        flux_density_left = np.concatenate((flux_density_left, sbc_values/2), axis=0)
        flux_density_right = np.concatenate((flux_density_right, sbc_values/2), axis=0)
        freqs = np.concatenate((freqs, sbc_freqs), axis=0)

        print(f"Ближайшее время: {nearest_time}")
        # for freq, val in zip(sbc_freqs, sbc_values):
        #     print(f"{freq:.1f} MHz: {val}")

    if running_mean:
        flux_density_left = ConvertingArrays.variable_running_mean(flux_density_left)
        flux_density_right = ConvertingArrays.variable_running_mean(flux_density_right)
        logging.info(f'Flux in sfu for LCP with smoothing: [{ConvertingArrays.arr2str4print(flux_density_left)}]')
        logging.info(f'Flux in sfu for RCP with smoothing: [{ConvertingArrays.arr2str4print(flux_density_right)}]')

    if gs_approx:
        flux_density_left_approx = ConvertingArrays.gs_approximation(flux_density_left, freqs)[0]
        flux_density_right_approx = ConvertingArrays.gs_approximation(flux_density_right, freqs)[0]
        logging.info(f'Flux in sfu for LCP with GS approximation: [{ConvertingArrays.arr2str4print(flux_density_left_approx)}]')
        logging.info(f'Flux in sfu for RCP with GS approximation: [{ConvertingArrays.arr2str4print(flux_density_right_approx)}]')

    if gm_approx:
        try:
            flux_density_left_gm_approx, gm_left_plot_freqs, gm_left_plot_arr = ConvertingArrays.gamma_approximation(flux_density_left, freqs)
            flux_density_right_gm_approx, gm_right_plot_freqs, gm_right_plot_arr = ConvertingArrays.gamma_approximation(flux_density_right, freqs)
        except RuntimeError:
            print('Not sucsess approximation, use running mean...')
            temp_left = ConvertingArrays.variable_running_mean(flux_density_left)
            temp_right = ConvertingArrays.variable_running_mean(flux_density_right)
            flux_density_left_gm_approx, gm_left_plot_freqs, gm_left_plot_arr = ConvertingArrays.gamma_approximation(temp_left, freqs)
            flux_density_right_gm_approx, gm_right_plot_freqs, gm_right_plot_arr = ConvertingArrays.gamma_approximation(temp_right, freqs)
        except Exception as err:
            print(f'Error approximation: {err}')
            sys.exit()
        logging.info(f'Flux in sfu for LCP with gamma approximation???: [{ConvertingArrays.arr2str4print(flux_density_left_gm_approx)}]')
        logging.info(f'Flux in sfu for RCP with gamma approximation???: [{ConvertingArrays.arr2str4print(flux_density_right_gm_approx)}]')

        logging.info(f'Freqs L: [{ConvertingArrays.arr2str4print(gm_left_plot_freqs)}]')
        logging.info(f'Freqs R: [{ConvertingArrays.arr2str4print(gm_right_plot_freqs)}]')
        logging.info(f'Fluxxxx L: [{ConvertingArrays.arr2str4print(gm_left_plot_arr)}]')
        logging.info(f'Fluxxxx R: [{ConvertingArrays.arr2str4print(gm_right_plot_arr)}]')

    if psf_calibration:
        correction_psf_left = flux_density_left * psf_square
        correction_psf_right = flux_density_right * psf_square
        logging.info(f'Correction from psf flux in sfu for LCP: [{ConvertingArrays.arr2str4print(correction_psf_left)}]')
        logging.info(f'Correction from psf flux in sfu for RCP: [{ConvertingArrays.arr2str4print(correction_psf_right)}]')
        correction_psf_polynom_left = np.polyfit(freqs, np.log(correction_psf_left), polynomial_degree)
        correction_psf_ya_left = np.exp(np.polyval(correction_psf_polynom_left, freqs))
        correction_psf_polynom_right = np.polyfit(freqs, np.log(correction_psf_right), polynomial_degree)
        correction_psf_ya_right = np.exp(np.polyval(correction_psf_polynom_right, freqs))
        logging.info(f'Finish flux with correction from psf in sfu for LCP - polifit approximation: [{ConvertingArrays.arr2str4print(correction_psf_ya_left)}]')
        logging.info(f'Finish flux with correction from psf in sfu for RCP - polifit approximation: [{ConvertingArrays.arr2str4print(correction_psf_ya_right)}]')
    else:
        correction_psf_left, correction_psf_right = np.array([]), np.array([])

    # необходимо для настройки области отображения, лимитов по осям
    flux_for_graph_RL = np.concatenate((flux_density_left, flux_density_right, correction_psf_left, correction_psf_right), axis=0)
    flux_for_graph_I = np.concatenate((flux_density_left + flux_density_right, correction_psf_left + correction_psf_right), axis=0)



    if polinom_approx:
        # polynom_left = np.polyfit(freqs, np.log(flux_density_left * correction_quate_sun_left), polynomial_degree)
        polynom_left = np.polyfit(freqs, np.log(flux_density_left), polynomial_degree)
        ya_left = np.exp(np.polyval(polynom_left, freqs))
        # polynom_right = np.polyfit(freqs, np.log(flux_density_right * correction_quate_sun_right), polynomial_degree)
        polynom_right = np.polyfit(freqs, np.log(flux_density_right), polynomial_degree)
        ya_right = np.exp(np.polyval(polynom_right, freqs))

        logging.info(f'Finish flux in s.f.u for LCP - polifit: [{ConvertingArrays.arr2str4print(ya_left)}]')
        logging.info(f'Finish flux in s.f.u for RCP - polifit: [{ConvertingArrays.arr2str4print(ya_right)}]')

    ######## График поляризаций #######
    fig_LR, LR_axs = plt.subplots(1, 2, num="L and R polarization", figsize=(27, 15), sharex=True, sharey=True)
    # fig.suptitle(f'{directory[9:24]}')
    plot_freqs = freqs/1000

    LR_axs[0].plot(plot_freqs, flux_density_left, 'o', label = f"Наблюдаемый спектр", color = 'darkblue', markersize=12, markerfacecolor='none', markeredgewidth=4, zorder = 2)
    if polinom_approx:
        LR_axs[0].plot(plot_freqs, ya_left, linestyle = '--', zorder = 1)
    if psf_calibration == True:
        LR_axs[0].plot(plot_freqs, correction_psf_ya_left, linestyle = '--', zorder = 1)
        LR_axs[0].plot(plot_freqs, correction_psf_left, 'o', label = f"Наблюдаемый спектр\nс поправкой на ширину\nдиаграммы направленности", linewidth = 8, color = 'firebrick', markersize=12, markerfacecolor='none', markeredgewidth=4, zorder = 2)
    if gs_approx == True:
        LR_axs[0].plot(plot_freqs, flux_density_left_approx, linestyle = '--', zorder = 1)
    if gm_approx == True:
        LR_axs[0].plot(gm_left_plot_freqs/1000, gm_left_plot_arr, linestyle = '--', zorder = 1)

    LR_axs[0].set_title('LCP', fontweight='bold')
    LR_axs[0].set_ylabel('Flux density, $sfu$')

    LR_axs[1].plot(plot_freqs, flux_density_right, 'o', label = f"Наблюдаемый спектр", linewidth = 8, color = 'darkblue', markersize=12, markerfacecolor='none', markeredgewidth=4, zorder = 2)
    if polinom_approx:
        LR_axs[1].plot(plot_freqs, ya_right, linestyle = '--', zorder = 1)
    if psf_calibration == True:
        LR_axs[1].plot(plot_freqs, correction_psf_ya_right, linestyle = '--', zorder = 1)
        LR_axs[1].plot(plot_freqs, correction_psf_right, 'o', label = f"Наблюдаемый спектр\nс поправкой на ширину\nдиаграммы направленности", color = 'firebrick', markersize=12, markerfacecolor='none', markeredgewidth=4, zorder = 2)
    if gs_approx == True:
        LR_axs[1].plot(plot_freqs, flux_density_right_approx, linestyle = '--', zorder = 1)
    if gm_approx == True:
        LR_axs[1].plot(gm_right_plot_freqs/1000, gm_right_plot_arr, linestyle = '--', zorder = 1)

    LR_axs[1].set_title('RCP', fontweight='bold')
    # axs[1].set_ylabel('Flux density, $sfu$')

    for ax in LR_axs:
        ax.set_xlabel('Frequency, $GHz$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which="both", linestyle='--')
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.xaxis.set_ticks(plot_freqs)
        ax.set_xticklabels(plot_freqs, rotation=75, ha='right', fontsize=12, zorder=0)
        ax.set_xlim(np.min(plot_freqs) - np.log10(np.min(plot_freqs) * 0.5), np.max(plot_freqs) + np.log10(np.max(plot_freqs) * 0.5))
        ax.set_ylim(np.min(flux_for_graph_RL) - np.min(flux_for_graph_RL) * 0.1, np.max(flux_for_graph_RL) + np.max(flux_for_graph_RL) * 0.1)
        ax.legend()
    plt.tight_layout()
    if save_graphs:
        plt.savefig(f'intensity_graphs/LCP_RCP_{extractor.extract_datetime(directory)}.png', dpi = 300)
    else:
        plt.show()
    plt.close()

    ######## График интенсивности ########
    if intensity_plot:
        fig_I = plt.figure(num="Intensity", figsize=(12, 9))
        I_ax = plt.gca()
        I_ax.plot(plot_freqs, flux_density_left + flux_density_right, 'o', label = f"Наблюдаемый спектр", linewidth = 8, color = 'darkblue', markersize=12, markerfacecolor='none', markeredgewidth=4, zorder = 2)
        if polinom_approx:
            I_ax.plot(plot_freqs, ya_left + ya_right, linestyle = '--', zorder = 1)
        if psf_calibration == True:
            I_ax.plot(plot_freqs, correction_psf_ya_left + correction_psf_ya_right, linestyle = '--', zorder = 1)
            I_ax.plot(plot_freqs, correction_psf_left + correction_psf_right, 'o', label = f"Наблюдаемый спектр\nс поправкой на ширину\nдиаграммы направленности", color = 'firebrick', markersize=10, zorder = 2)
        if gm_approx == True:
            I_ax.plot(gm_right_plot_freqs/1000, gm_left_plot_arr + gm_right_plot_arr, linestyle = '--', zorder = 1, label = f"Аппроксимация")
        I_ax.set_xlim(np.min(plot_freqs) - np.log10(np.min(plot_freqs) * 0.5), np.max(plot_freqs) + np.log10(np.max(plot_freqs) * 0.5))
        I_ax.set_ylim(np.min(flux_for_graph_I) - np.min(flux_for_graph_I) * 0.2, np.max(flux_for_graph_I) + np.max(flux_for_graph_I) * 0.2)
        I_ax.set_xlabel('Frequency, $GHz$')
        I_ax.set_ylabel('Flux density, $sfu$')
        I_ax.grid(True, which="both", linestyle='--')
        I_ax.set_yscale('log')
        I_ax.set_xscale('log')
        I_ax.legend()

        I_ax.xaxis.set_major_formatter(ScalarFormatter())
        I_ax.xaxis.set_minor_formatter(NullFormatter())
        I_ax.xaxis.set_minor_locator(ticker.NullLocator())
        I_ax.xaxis.set_ticks(plot_freqs)
        if alternation_subscription_x:
            labels = [str(plot_freqs[i]) if i % 2 == 0 else "" for i in range(len(plot_freqs))]
            I_ax.set_xticklabels(labels, rotation=60, ha='right')
        else:
            I_ax.set_xticklabels(plot_freqs, rotation=60, ha='right')
        plt.tight_layout()
        if save_graphs:
            plt.savefig(f'intensity_graphs/I_{extractor.extract_datetime(directory)}.png', dpi = 300)
        else:
            plt.show()
        plt.close()


