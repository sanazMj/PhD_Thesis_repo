import h5py
import json
import os
import glob
import pickle
from helper import get_points_from_matrix


def get_tumor_coord(x_limit,Data_path_input="/ryerson/projects/tumorGAN/Data/"):
    '''
    returns the tumor coordinates from the generated files from Matlab
    :param x_limit: the limit of shapes' height/width/length
    :param Data_path_input: directory
    :return: ile_names, min_indexes_tumor, max_indexes_tumor
    '''
    min_indexes_tumor = []
    max_indexes_tumor = []
    min_indexes_OAR = []
    max_indexes_OAR = []
    file_names = []
    for itr, file_name in enumerate(glob.glob(Data_path_input + "*.mat")):
        print(itr)
        if itr == 100:
            break
        data = sio.loadmat(file_name)
        data_file = data['tumor_struct']
        tumors = data_file['maskTumor_cell'][0][0]
        OARs = data_file['maskOAR_cell'][0][0]
        all_points_tumor = []
        all_points_OAR = []
        all_points = []
        flag = 0
        for i in range((tumors.shape[0])):
            if np.sum(tumors[i][0]) > 0:
                points = get_points_from_matrix(tumors[i][0], condition=1)
                tumor_indice_specific = np.where(tumors[i][0] == 1)
                if len(np.unique(tumor_indice_specific[0])) == 1 or len(
                        np.unique(tumor_indice_specific[1])) == 1 or len(
                        np.unique(tumor_indice_specific[2])) == 1:
                    print('there is a 2D tumor here skip!')
                    flag = 1
                    continue
                if flag == 0:
                    if len(all_points) == 0:
                        all_points = points
                    else:
                        all_points = np.concatenate((all_points, points), axis=0)

        for i in range((OARs.shape[0])):
            if np.sum(OARs[i][0])>0:
                points = get_points_from_matrix(OARs[i][0], condition=1)
                tumor_indice_specific = np.where(OARs[i][0] == 1)
                if len(np.unique(tumor_indice_specific[0])) == 1 or len(
                        np.unique(tumor_indice_specific[1])) == 1 or len(
                        np.unique(tumor_indice_specific[2])) == 1:
                    print('there is a 2D OARs here skip!')
                    flag = 1
                    continue
                if flag == 0:
                    if len(all_points) == 0:
                        all_points = points
                    else:
                        all_points = np.concatenate((all_points, points), axis=0)


        if len(all_points) > 0 and flag == 0:
            tumor_min = [np.min(all_points[:, 0]), np.min(all_points[:, 1]), np.min(all_points[:, 2])]
            tumor_max =[np.max(all_points[:, 0]), np.max(all_points[:, 1]), np.max(all_points[:, 2])]
            tumor_width = np.array(tumor_max)-np.array(tumor_min)


            if  all(tumor_width < x_limit):
                file_names.append(file_name)
                min_indexes_tumor.append(tumor_min)
                max_indexes_tumor.append(tumor_max)

    return file_names, min_indexes_tumor, max_indexes_tumor


def create_python_dataset_from_matfiles(file_list_names, output_file_name,min_indexes_tumor=[0,0,0],
                                        max_indexes_tumor=[240,240,160], size=(240, 240, 160)
                                        ,Data_path= "/ryerson/projects/tumorGAN/Data/", values=[1,2]):
    '''
    Load .mat files. append them and create a pickle file as a python dataset
    :param file_list_names: files' names
    :param output_file_name: the final file's name
    :param min_indexes_tumor: the minimun indexes that contains the shape
    :param max_indexes_tumor: the maximum indexes that contains the shape
    :param size: size of the data
    :param Data_path: directory
    :param values: what to fill in the shape and isocenters
    :return:
    '''
    file_list = []
    dict_index = {}
    Dict_property = {}
    f = h5py.File(Data_path + 'Dataset_'+ output_file_name +'.h5', 'a')
    if len(file_list_names) == 0:
        file_list_names = glob.glob(Data_path_input + "*.mat")

    for itr, file_name in enumerate(file_list_names):
        print(itr,file_name)
        temp = np.zeros(size,dtype=np.float16)
        Dict_temp = {}
        file_list.append(file_name)

        data = sio.loadmat(file_name)
        data_file = data['tumor_struct']

        dict_index[str(itr)] = file_name

        tumors = data_file['maskTumor_cell'][0][0]
        OARs = data_file['maskOAR_cell'][0][0]


        for i in range((tumors.shape[0])):
            tumor = tumors[i][0]

            if len(file_list_names) > 0:
                temp1 = np.zeros(size, dtype=np.int8)
                temp1[:max_indexes_tumor[itr][0]-min_indexes_tumor[itr][0],
                :max_indexes_tumor[itr][1] - min_indexes_tumor[itr][1],
                :max_indexes_tumor[itr][2] - min_indexes_tumor[itr][2]] = tumor[min_indexes_tumor[itr][0]:max_indexes_tumor[itr][0],
                                min_indexes_tumor[itr][1]:max_indexes_tumor[itr][1],
                                min_indexes_tumor[itr][2]:max_indexes_tumor[itr][2]]
                tumor = temp1

            temp[ tumor == 1] = values[0] #1

        for j in range((OARs.shape[0])):
            OAR = OARs[j][0]

            if np.sum(OAR)>0:

                if len(file_list_names) > 0:
                    temp1 = np.zeros(size, dtype=np.int8)
                    temp1[:max_indexes_tumor[itr][0]-min_indexes_tumor[itr][0],
                    :max_indexes_tumor[itr][1] - min_indexes_tumor[itr][1],
                    :max_indexes_tumor[itr][2] - min_indexes_tumor[itr][2]] = OAR[min_indexes_tumor[itr][0]:max_indexes_tumor[itr][0],
                                    min_indexes_tumor[itr][1]:max_indexes_tumor[itr][1],
                                    min_indexes_tumor[itr][2]:max_indexes_tumor[itr][2]]
                    OAR = temp1

                temp[OAR == 1] = values[1] #2

        temp1 =  temp.reshape(1, size[0], size[1], size[2])

        if itr == 0:
            # Create the dataset at first
            f.create_dataset('data', data=temp1, compression="gzip", chunks=True, maxshape=(None, size[0], size[1], size[2]))

        else:
            # Append new data to it
            f['data'].resize((f['data'].shape[0] + temp1.shape[0]), axis=0)
            f['data'][-temp1.shape[0]:] = temp1

    f.close()

    with open(Data_path + 'data_index_' + output_file_name+ '.pkl', 'wb') as handle:
        pickle.dump(dict_index, handle, protocol=pickle.HIGHEST_PROTOCOL)




