import os
import numpy as np
import shutil
def get_dataset_filelist(input_training_wav_list):
    training_files=[]
    filelist=os.listdir(input_training_wav_list)
    for files in filelist:
      
      
      training_files.append(files)
    return training_files
 


source_folder_a = './abx/hifi'
source_folder_b = './abx/hifi+'
source_folder_c = './abx/apnet2'
source_folder_d = './abx/apnet2+'
source_folder_e = './abx/freev'
source_folder_f = './abx/freev+'
source_folder_g = './abx/istftnet'
source_folder_h = './abx/istftnet+'
source_folder_i = './abx/vocos'
source_folder_j = './abx/vocos+'
file_list = get_dataset_filelist(source_folder_a)
# D_source_dir = '../logamp_detach'
# E_source_dir = '../logamp_vits' # HiFi-GAN V2
# F_source_dir = '../logamp_vits_80' # uSE-GAN V2

target_folder_a = './A'
target_folder_b = './B'
target_folder_c = './C'
target_folder_d = './D'
target_folder_e = './E'
target_folder_f = './F'
target_folder_g = './G'
target_folder_h = './H'
target_folder_i = './I'
target_folder_j = './J'
# D_target_dir='../mos_wav/logamp_detach'
# E_target_dir='../mos_wav/logamp_vits'
# F_target_dir='../mos_wav/logamp_vits_80'

for i in range(len(file_list)):
	if i < 9:
		name = '0' + str(i+1)
	else:
		name = str(i+1)
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
	shutil.copyfile(os.path.join(source_folder_a, file_list[i]), os.path.join(target_folder_a, 'A_'+name+'.wav'))
	shutil.copyfile(os.path.join(source_folder_b, file_list[i]), os.path.join(target_folder_b, 'B_'+name+'.wav'))
	shutil.copyfile(os.path.join(source_folder_c, file_list[i]), os.path.join(target_folder_c, 'C_'+name+'.wav'))
	shutil.copyfile(os.path.join(source_folder_d, file_list[i]), os.path.join(target_folder_d, 'D_'+name+'.wav'))
	shutil.copyfile(os.path.join(source_folder_e, file_list[i]), os.path.join(target_folder_e, 'E_'+name+'.wav'))
	shutil.copyfile(os.path.join(source_folder_f, file_list[i]), os.path.join(target_folder_f, 'F_'+name+'.wav'))
	shutil.copyfile(os.path.join(source_folder_g, file_list[i]), os.path.join(target_folder_g, 'G_'+name+'.wav'))
	shutil.copyfile(os.path.join(source_folder_h, file_list[i]), os.path.join(target_folder_h, 'H_'+name+'.wav'))
	shutil.copyfile(os.path.join(source_folder_i, file_list[i]), os.path.join(target_folder_i, 'I_'+name+'.wav'))
	shutil.copyfile(os.path.join(source_folder_j, file_list[i]), os.path.join(target_folder_j, 'J_'+name+'.wav'))
	# shutil.copyfile(os.path.join(D_source_dir, file_list[i]), os.path.join(D_target_dir, 'D_'+name+'.wav'))
	# shutil.copyfile(os.path.join(E_source_dir, file_list[i]), os.path.join(E_target_dir, 'B_'+name+'.wav'))
	# shutil.copyfile(os.path.join(F_source_dir, file_list[i]), os.path.join(F_target_dir, 'F_'+name+'.wav'))
