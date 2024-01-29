import os
import shutil
from datetime import datetime

# source_folder = "D:/16.07.23 point clean"
# target_time = "20230716T082420"

source_folder = "D:/datasets/20.01.22/fix_table"
target_time = "20220120T055430"
destination_folder = f"D:/datasets/20.01.22/times/{target_time}"

# Определяем место для сохранения
try:
    os.mkdir(destination_folder)
except:
    shutil.rmtree(destination_folder)
    os.mkdir(destination_folder)

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
                if file.endswith("_LCP.fit"):
                    lcp_nearest_file = os.path.join(root, file)
                elif file.endswith("_RCP.fit"):
                    rcp_nearest_file = os.path.join(root, file)

        if lcp_nearest_file:
            print(lcp_nearest_file)
            shutil.copy(lcp_nearest_file, dest_folder)
        if rcp_nearest_file:
            print(rcp_nearest_file)
            shutil.copy(rcp_nearest_file, dest_folder)

if __name__ == "__main__":
    find_nearest_files(source_folder, target_time, destination_folder)
