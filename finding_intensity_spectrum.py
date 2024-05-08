import logging
import os, sys
import re
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter, NullFormatter
import numpy as np
from astropy.io import fits
from analise_utils import ZirinTb, FindIntensity, MplFunction, Monitoring, Extract, ConvertingArrays, OsOperations

extractor = Extract()
zirin = ZirinTb()
MplFunction.set_mpl_rc()
Monitoring.start_log('230122_logs')
logging.info(f'Start of the program to search intensity spectrum of a sun')

############################################
##### Values #####
directories = ["D:/datasets/20.01.22/times/20220120T055630_calibrated_brightness_COM_aligned", "D:/datasets/20.01.22/times/20220120T055730_calibrated_brightness_COM_aligned"]
polynomial_degree = 3
# coordinates = (324, 628)
coordinates = (890, 563)
coordinates = (881, 566)
##### Params #####
running_mean = False
psf_calibration = False
background_Zirin_subtraction = False
intensity_plot = False
save_graphs = True
#############################################

if save_graphs:
    OsOperations.create_place(f'intensity_graph_{extractor.extract_datetime(directories[0])[0:7]}')

psf_npz_file = np.load('psf_square.npz')
freqs_npz = sorted(psf_npz_file['freqs'])
psf_square = psf_npz_file['psf_square']
# psf_square = np.array([11.1460177 ,  9.72418879,  8.5339233 ,  7.57227139,  6.74631268, 6.04867257,  5.46902655,  4.96902655,  4.49115044,  4.12831858, 3.5280236 ,  3.23156342,  3.11651917,  2.99852507,  2.78466077, 2.76548673,  2.59734513,  2.47640118,  2.22713864,  2.01917404, 1.83185841,  1.70943953,  1.55899705,  1.42625369,  1.32300885, 1.21976401,  1.13126844,  1.0560472, 1]) # 16.07.23
psf_square = np.array([1.75, 1.7, 1.65, 1.6, 1.55, 1.5, 1.45, 1.4, 1.35, 1.3, 1.2, 1.15, 1.1, 1.05, 1])  # 20.01.22 (9800 = 1.25, 10200 = 1.2)

for directory in directories:
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

    if running_mean:
        flux_density_left = ConvertingArrays.variable_running_mean(flux_density_left)
        flux_density_right = ConvertingArrays.variable_running_mean(flux_density_right)
        logging.info(f'Flux in sfu for LCP with smoothing: [{ConvertingArrays.arr2str4print(flux_density_left)}]')
        logging.info(f'Flux in sfu for RCP with smoothing: [{ConvertingArrays.arr2str4print(flux_density_right)}]')

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
    plot_freqs = freqs/1000

    correction_quate_sun_left = [1.03838439, 0.95999654, 1.09428362, 0.97976959, 0.9899581, 0.87707643, 0.9681845, 0.98712561, 1.03482719, 1.06164441, 1.06819119, 1.02623596, 0.97603417, 1.00623033, 0.95329644]
    correction_quate_sun_right = [1.03798513, 1.01688483, 1.04706266, 0.94311813, 0.94967881, 0.9540217, 0.94383052, 1.02094715, 0.99510278, 1.05711966, 1.10484654, 0.96657375, 1.07459668, 0.96531071, 0.94302758]

    # polynom_left = np.polyfit(freqs, np.log(flux_density_left * correction_quate_sun_left), polynomial_degree)
    polynom_left = np.polyfit(freqs, np.log(flux_density_left), polynomial_degree)
    ya_left = np.exp(np.polyval(polynom_left, freqs))
    # polynom_right = np.polyfit(freqs, np.log(flux_density_right * correction_quate_sun_right), polynomial_degree)
    polynom_right = np.polyfit(freqs, np.log(flux_density_right), polynomial_degree)
    ya_right = np.exp(np.polyval(polynom_right, freqs))

    logging.info(f'Finish flux in s.f.u for LCP - polifit: [{ConvertingArrays.arr2str4print(ya_left)}]')
    logging.info(f'Finish flux in s.f.u for RCP - polifit: [{ConvertingArrays.arr2str4print(ya_right)}]')

    ######## График поляризаций #######
    fig_LR, LR_axs = plt.subplots(1, 2, num="L and R polarization", figsize=(18, 9), sharex=True, sharey=True)
    # fig.suptitle(f'{directory[9:24]}')

    LR_axs[0].plot(plot_freqs, flux_density_left, 'o', label = f"Наблюдаемый спектр", color = 'darkblue', markersize=12, markerfacecolor='none', markeredgewidth=4, zorder = 2)
    LR_axs[0].plot(plot_freqs, ya_left, linestyle = '--', zorder = 1)
    if psf_calibration == True:
        LR_axs[0].plot(plot_freqs, correction_psf_ya_left, linestyle = '--', zorder = 1)
        LR_axs[0].plot(plot_freqs, correction_psf_left, 'o', label = f"Наблюдаемый спектр\nс поправкой на ширину\nдиаграммы направленности", linewidth = 8, color = 'firebrick', markersize=12, markerfacecolor='none', markeredgewidth=4, zorder = 2)
    LR_axs[0].set_title('LCP', fontweight='bold')
    LR_axs[0].set_ylabel('Flux density, $sfu$')

    LR_axs[1].plot(plot_freqs, flux_density_right, 'o', label = f"Наблюдаемый спектр", linewidth = 8, color = 'darkblue', markersize=12, markerfacecolor='none', markeredgewidth=4, zorder = 2)
    LR_axs[1].plot(plot_freqs, ya_right, linestyle = '--', zorder = 1)
    if psf_calibration == True:
        LR_axs[1].plot(plot_freqs, correction_psf_ya_right, linestyle = '--', zorder = 1)
        LR_axs[1].plot(plot_freqs, correction_psf_right, 'o', label = f"Наблюдаемый спектр\nс поправкой на ширину\nдиаграммы направленности", color = 'firebrick', markersize=12, markerfacecolor='none', markeredgewidth=4, zorder = 2)
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
        ax.set_xticklabels(plot_freqs, rotation=60, ha='right')
        ax.set_xlim(np.min(plot_freqs) - np.log10(np.min(plot_freqs) * 0.3), np.max(plot_freqs) + np.log10(np.max(plot_freqs) * 0.3))
        ax.set_ylim(np.min(flux_for_graph_RL) - np.min(flux_for_graph_RL) * 0.1, np.max(flux_for_graph_RL) + np.max(flux_for_graph_RL) * 0.1)
        ax.legend()
    plt.tight_layout()
    if save_graphs:
        plt.savefig(f'intensity_graph_{extractor.extract_datetime(directories[0])[0:7]}/LCP_RCP_{extractor.extract_datetime(directory)}.png', dpi = 300)

    ######## График интенсивности ########
    if intensity_plot:
        fig_I = plt.figure(num="Intensity", figsize=(12, 9))
        I_ax = plt.gca()
        I_ax.plot(plot_freqs, flux_density_left + flux_density_right, 'o', label = f"Наблюдаемый спектр", linewidth = 8, color = 'darkblue', markersize=12, markerfacecolor='none', markeredgewidth=4, zorder = 2)
        I_ax.plot(plot_freqs, ya_left + ya_right, linestyle = '--', zorder = 1)
        if psf_calibration == True:
            I_ax.plot(plot_freqs, correction_psf_ya_left + correction_psf_ya_right, linestyle = '--', zorder = 1)
            I_ax.plot(plot_freqs, correction_psf_left + correction_psf_right, 'o', label = f"Наблюдаемый спектр\nс поправкой на ширину\nдиаграммы направленности", color = 'firebrick', markersize=10, zorder = 2)
        I_ax.set_xlim(np.min(plot_freqs) - np.log10(np.min(plot_freqs) * 0.3), np.max(plot_freqs) + np.log10(np.max(plot_freqs) * 0.3))
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
        I_ax.set_xticklabels(plot_freqs, rotation=60, ha='right')
        
        plt.tight_layout()
        if save_graphs:
            plt.savefig(f'intensity_graph_{extractor.extract_datetime(directories[0])[0:7]}/I_{extractor.extract_datetime(directory)}.png', dpi = 300)
            
    plt.show()