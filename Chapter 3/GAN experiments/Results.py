
import numpy as np
import matplotlib.pyplot as plt
# import dccp
import json
import os
from mpl_toolkits import mplot3d
import pandas as pd
# import pickle
from Sphere_packing import *

from tabulate import tabulate


def report_results(files, name_save = ''):
    keys_access = ['GAN_model', 'mode_collapse', 'pac_num', 'filled', 'num_epochs', 'noise_dim', 'learning_rate_g',
                   'learning_rate_d',
                   'target_points', 'dataset_kind', 'dataset_include_tumor',
                   'dataset_size', 'size_wanted', 'shape', 'batch_size',
                   'step_report', 'softSparsity', 'diversity', 'connectedFlag', 'space_dim', 'in_channels_dim',
                   'dimension', 'itr_critic']
    return_dataframe_iso = pd.DataFrame(columns=keys_access + ['file', 'epoch'] + ['acc_with_target_unique','diversity_mean', 'diversity_std',
                                                             'count_ratio',  'count_per_tumor_list_mean', 'count_per_tumor_list_std'])
    return_dataframe_tumor = pd.DataFrame(columns=keys_access + ['file', 'epoch'] +  ['correct_spheres_percentage','KLD_ratio','KLD_points','KLD_points_new',
                                                                 'KLD_ratio_convex','OMEGA1_mean', 'OMEGA1_std','OMEGA2_mean', 'OMEGA2_std',
                                          'OMEGA3_mean', 'OMEGA3_std', 'points_size_mean', 'points_size_std', 'points_size_new_mean', 'points_size_new_std'])

    dir = '/Ryerson/Projects/tumorGAN/GAN_simple_3D/logs/'
    return_dataframe_tumor.to_csv(dir + 'pd_result_tumor_' + name_save + '.csv')
    return_dataframe_iso.to_csv(dir + 'pd_result_iso_' + name_save + '.csv')


    for dir_file in files:
        print('file', dir_file)
        results = extract_config(dir, [dir_file], keys_access)
        print(results)
        step = results[0]['step_report']
        epoch = results[0]['num_epochs']
        target_points = results[0]['target_points']
        files_epoch = list(range(0, epoch, step)) + [epoch-1]
        flag = 0
        for i in files_epoch:
            if not os.path.exists(dir + str(dir_file) + '/'+ 'result_dict_epoch' + str(i) + '.pkl'):
                print('result_dict_epoch' + str(i) + '.pkl'  + 'does not exist')
            else:
                if target_points == 1:
                    file = dir + str(dir_file) + '/'+ 'result_dict_epoch' + str(i) + '.pkl'
                    print(file)
                    results1 = check_files_convexity(file)
                elif target_points == 7:
                    file = [dir + str(dir_file) + '/'+ 'result_dict_epoch' + str(i) + '.pkl']
                    print(file)

                    results1 = check_files_sphere_filled_iso(file)
                results_new = results[0]
                results_new['file'] = dir_file
                results_new['epoch'] = i
                results_new.update(results1)
                df = pd.DataFrame(results_new)
                if target_points == 7:
                    return_dataframe_iso = return_dataframe_iso.append(results_new,ignore_index=True)
                    df.to_csv(dir + 'pd_result_iso_' + name_save + '.csv', mode='a', index=False, header=False)
                else:
                    return_dataframe_tumor = return_dataframe_tumor.append(results_new,ignore_index=True)
                    df.to_csv(dir + 'pd_result_tumor_' + name_save + '.csv', mode='a', index=False, header=False)

    print(tabulate(return_dataframe_tumor, headers='keys', tablefmt='pretty'))
    print(tabulate(return_dataframe_iso, headers='keys', tablefmt='pretty'))

    return_dataframe_tumor.to_csv(dir + 'pd_result_final_tumor' + '.csv')
    return_dataframe_iso.to_csv(dir + 'pd_result_final_iso' + '.csv')

def check_tumor_iso(ids, files_epoch, dataset_include_tumor, name_data, dir='/Ryerson/Projects/tumorGAN/GAN_simple_3D/logs/'):
    for id in ids:
        file_names = [dir + str(id) + '/' + 'result_dict_epoch' + str(k) + '.pkl' for k in files_epoch]
        check_files_sphere_filled_iso(file_names,dataset_include_tumor,name_data)

def check_tumor(ids, files_epoch, neme_data, dir='/Ryerson/Projects/tumorGAN/GAN_simple_3D/logs/',save_omega=False):
    for id in ids:
        file_names = [dir + str(id)+'/' +'result_dict_epoch' + str(k) + '.pkl' for k in files_epoch]
        for index, file in enumerate(file_names):
            print(file)
            check_files_convexity(file, neme_data, id_name=str(id) + '_' + str(files_epoch[index]),save_omega=save_omega)


# files = [162,163,164,165] #list(range(1,162))
keys_access = ['GAN_model', 'd_thresh','beta','mode_collapse', 'pac_num', 'filled', 'num_epochs', 'noise_dim', 'learning_rate_g',
               'learning_rate_d',
               'target_points', 'dataset_kind', 'dataset_include_tumor',
               'target_points', 'dataset_kind', 'dataset_include_tumor',
               'dataset_size', 'size_wanted', 'shape', 'batch_size',
               'step_report', 'softSparsity', 'diversity', 'connectedFlag', 'space_dim', 'in_channels_dim',
               'dimension', 'itr_critic']

dir = '/Ryerson/Projects/tumorGAN/dbraingen_master/logs/'
dataset_include_tumor = True
step_report = 100
num_epoch = 500
ids = [54]
files_epoch = [100,200] #list(range(0,num_epoch, step_report)) + [num_epoch-1]
name_data = 'Matlab'
check_tumor(ids, files_epoch, name_data, dir=dir,save_omega=True)
