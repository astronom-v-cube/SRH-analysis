import os
import shutil
from datetime import datetime
from tqdm import tqdm
from analise_utils import OsOperations, Extract

extract = Extract()

source_folder = "E:/testdataset"
target_time = "20240514T020138"
destination_folder = f"{source_folder}_times/{target_time}"

OsOperations.create_place(destination_folder)

def find_nearest_files(src_folder, target_time, dest_folder):
    target_time = datetime.strptime(target_time, "%Y%m%dT%H%M%S")

    for root, dirs, files in os.walk(src_folder):
        lcp_nearest_file = None
        rcp_nearest_file = None
        min_difference = float('inf')

        for file in files:
            file_time_str = file.split("_")[1]
            file_time = datetime.strptime(file_time_str, "%Y%m%dT%H%M%S")
            time_difference = abs((file_time - target_time).total_seconds())

            if time_difference <= min_difference:
                min_difference = time_difference
                if extract.extract_polarization(file) == "LCP":
                    lcp_nearest_file = os.path.join(root, file)
                elif extract.extract_polarization(file) == "RCP":
                    rcp_nearest_file = os.path.join(root, file)

        if lcp_nearest_file:
            shutil.copy(lcp_nearest_file, dest_folder)
            print(f'Файл {lcp_nearest_file} перемещен')
        if rcp_nearest_file:
            shutil.copy(rcp_nearest_file, dest_folder)
            print(f'Файл {rcp_nearest_file} перемещен')

if __name__ == "__main__":
    find_nearest_files(source_folder, target_time, destination_folder)
