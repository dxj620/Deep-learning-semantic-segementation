import os

import shutil

file_path = r'D:\dataset\Graduation_project\Graduation_result\512_final_GID'
file_list = os.listdir(file_path)
k = 1
for file in file_list:
    if k == 1:
        k += 1
        continue
    else:
        k += 1
        file_next = os.path.join(file_path, file)
        for file_next_part in os.listdir(file_next):
            if file_next_part.split('_')[-1] in ['12', '16', '20', '24', '28', '32', '36', '40']:
                for file_last in os.listdir(os.path.join(file_next, file_next_part)):
                    for number in ['0.4', '0.5', '0.7', '0.8', '0.9']:
                        # print(file_last.find(number))
                        if file_last.find(number) != -1:
                            if os.path.isdir(os.path.join(os.path.join(file_next, file_next_part), file_last)):
                                shutil.rmtree(os.path.join(os.path.join(file_next, file_next_part), file_last))
                            else:
                                print(os.path.join(os.path.join(file_next, file_next_part), file_last))
