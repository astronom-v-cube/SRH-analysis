from astropy.io import fits
from analise_utils import Monitoring, OsOperations, FitsOperations, Variables
import logging
import os, sys
from tqdm import tqdm
import re

Monitoring.start_log('IV2RL')
root_directory = "C:/fits/"

def list_directories(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

directories = list_directories(root_directory)
logging.info(f'{directories}')

for directory in tqdm(directories, desc='Freqs convert', leave=True):
    logging.info(f'Start working with {directory}')
    # проверка для системы имен кода Анфиногентова
    fits_files = OsOperations.abс_sorted_files_in_folder(os.path.join(root_directory, directory))
    I_fits_files, V_fits_files = [], []
    for filename in fits_files:
        if filename.split('_')[3] == 'I':
            I_fits_files.append(filename)
        elif filename.split('_')[3] == 'V':
            V_fits_files.append(filename)
        else:
            print(f'Что-то не то с {filename}')
            sys.exit()
    OsOperations.create_place(os.path.join(root_directory, str(FitsOperations.folder_name_anf2globa(directory))))

    for index in tqdm(range(len(I_fits_files)), desc='Files convert', leave=False):
        FitsOperations.IV2RL(os.path.join(root_directory, directory), I_fits_files[index], V_fits_files[index], str(FitsOperations.folder_name_anf2globa(directory)), deleteIV=False)