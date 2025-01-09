import os
import random
import shutil

# 源文件夹和目标文件夹路径
source_folder_a = '../hifi-gan-master/generated_files__withgan'
source_folder_b = '../hifi-gan-master-propose/generated_files'
source_folder_c = '../APNet2-main/output_withgan'
source_folder_d = '../APNet2-main-propose/output'
source_folder_e = '../FreeV-main/output_withgan'
source_folder_f = '../FreeV-main-propose/output'
source_folder_g = '../iSTFTNet-pytorch-master/generated_files_withgan'
source_folder_h = '../iSTFTNet-pytorch-master-propose/generated_files'
source_folder_i = '../vocos/output_withgan'
source_folder_j = '../vocos-propose/output'

target_folder_a = './abx/hifi'
target_folder_b = './abx/hifi+'
target_folder_c = './abx/apnet2'
target_folder_d = './abx/apnet2+'
target_folder_e = './abx/freev'
target_folder_f = './abx/freev+'
target_folder_g = './abx/istftnet'
target_folder_h = './abx/istftnet+'
target_folder_i = './abx/vocos'
target_folder_j = './abx/vocos+'

wav_files = [f for f in os.listdir(source_folder_a) if f.endswith('.wav')]

selected_files = random.sample(wav_files, 20)

# 遍历选中的文件
for file in selected_files:
    # 构建文件路径
    source_file_a = os.path.join(source_folder_a, file)
    source_file_b = os.path.join(source_folder_b, file)
    source_file_c = os.path.join(source_folder_c, file)
    source_file_d = os.path.join(source_folder_d, file)
    source_file_e = os.path.join(source_folder_e, file)
    source_file_f = os.path.join(source_folder_f, file)
    source_file_g = os.path.join(source_folder_g, file)
    source_file_h = os.path.join(source_folder_h, file)
    source_file_i = os.path.join(source_folder_i, file)
    source_file_j = os.path.join(source_folder_j, file)
    os.makedirs(target_folder_a, exist_ok=True)
    os.makedirs(target_folder_b, exist_ok=True)
    os.makedirs(target_folder_c, exist_ok=True)
    os.makedirs(target_folder_d, exist_ok=True)
    os.makedirs(target_folder_e, exist_ok=True)
    os.makedirs(target_folder_f, exist_ok=True)
    os.makedirs(target_folder_g, exist_ok=True)
    os.makedirs(target_folder_h, exist_ok=True)
    os.makedirs(target_folder_i, exist_ok=True)
    os.makedirs(target_folder_j, exist_ok=True)
    # 复制文件到目标文件夹b
    target_file_a = os.path.join(target_folder_a, file)
    shutil.copyfile(source_file_a, target_file_a)
    target_file_b = os.path.join(target_folder_b, file)
    shutil.copyfile(source_file_b, target_file_b)
    target_file_c = os.path.join(target_folder_c, file)
    shutil.copyfile(source_file_c, target_file_c)
    target_file_d = os.path.join(target_folder_d, file)
    shutil.copyfile(source_file_d, target_file_d)
    target_file_e = os.path.join(target_folder_e, file)
    shutil.copyfile(source_file_e, target_file_e)
    target_file_f = os.path.join(target_folder_f, file)
    shutil.copyfile(source_file_f, target_file_f)
    target_file_g = os.path.join(target_folder_g, file)
    shutil.copyfile(source_file_g, target_file_g)
    target_file_e = os.path.join(target_folder_e, file)
    shutil.copyfile(source_file_e, target_file_e)
    target_file_f = os.path.join(target_folder_f, file)
    shutil.copyfile(source_file_f, target_file_f)
    target_file_g = os.path.join(target_folder_g, file)
    shutil.copyfile(source_file_g, target_file_g)
    target_file_h = os.path.join(target_folder_h, file)
    shutil.copyfile(source_file_h, target_file_h)
    target_file_i = os.path.join(target_folder_i, file)
    shutil.copyfile(source_file_i, target_file_i)
    target_file_j = os.path.join(target_folder_j, file)
    shutil.copyfile(source_file_j, target_file_j)
print('Copy process completed.')
