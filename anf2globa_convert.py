# from astropy.io import fits
# from analise_utils import Monitoring, OsOperations, FileOperations, Variables
# import logging
# import os, sys
# from tqdm import tqdm
# import re

# Monitoring.start_log('IV2RL')
# root_directory = "J:/results2/images"

# def list_directories(path):
#     return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

# directories = list_directories(root_directory)
# logging.info(f'{directories}')

# for directory in tqdm(directories, desc='Freqs convert', leave=True):
#     logging.info(f'Start working with {directory}')
#     # проверка для системы имен кода Анфиногентова
#     fits_files = OsOperations.abс_sorted_files_in_folder(os.path.join(root_directory, directory))
#     I_fits_files, V_fits_files = [], []
#     for filename in fits_files:
#         if filename.split('_')[3] == 'I':
#             I_fits_files.append(filename)
#         elif filename.split('_')[3] == 'V':
#             V_fits_files.append(filename)
#         else:
#             print(f'Что-то не то с {filename}')
#             sys.exit()
#     OsOperations.create_place(os.path.join(root_directory, str(FileOperations.folder_name_anf2globa(directory))))

#     for index in tqdm(range(len(I_fits_files)), desc='Files convert', leave=False):
#         FileOperations.IV2RL(os.path.join(root_directory, directory), I_fits_files[index], V_fits_files[index], str(FileOperations.folder_name_anf2globa(directory)), deleteIV=False)


from astropy.io import fits
from analise_utils import Monitoring, OsOperations, FileOperations, Variables
import logging
import os, sys
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

Monitoring.start_log('IV2RL')
root_directory = "J:/results2/images"
tqdm.set_lock(Lock())

def list_directories(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def process_directory(directory):
    logging.info(f'Start working with {directory}')
    directory_path = os.path.join(root_directory, directory)
    fits_files = OsOperations.abс_sorted_files_in_folder(directory_path)

    I_fits_files, V_fits_files = [], []
    for filename in fits_files:
        try:
            pol = filename.split('_')[3]
            if pol == 'I':
                I_fits_files.append(filename)
            elif pol == 'V':
                V_fits_files.append(filename)
            else:
                logging.error(f'Unknown polarization in file name: {filename}')
                return
        except IndexError:
            logging.error(f'Invalid file name format: {filename}')
            return

    output_folder = str(FileOperations.folder_name_anf2globa(directory))
    output_path = os.path.join(root_directory, output_folder)
    OsOperations.create_place(output_path)

    for i in tqdm(range(len(I_fits_files)), desc=f'{directory}: Files convert', leave=False):
        FileOperations.IV2RL(
            directory_path,
            I_fits_files[i],
            V_fits_files[i],
            output_folder,
            deleteIV=False
        )

def main():
    directories = list_directories(root_directory)
    logging.info(f'Directories found: {directories}')

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_directory, d): d for d in directories}

        for future in tqdm(as_completed(futures), total=len(futures), desc='Freqs convert'):
            dir_name = futures[future]
            try:
                future.result()
            except Exception as exc:
                logging.error(f'{dir_name} generated an exception: {exc}')

if __name__ == "__main__":
    main()
