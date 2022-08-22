
import numpy as np
import pandas as pd
import h5py
import json
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.autograd as autograd
from torch.autograd.variable import Variable
import math
from scipy.ndimage.measurements import label
from scipy.ndimage import generate_binary_structure
from scipy.spatial import ConvexHull, convex_hull_plot_2d,  distance
from constants import Constant_dict, Constant_dict_keys


def get_points_from_matrix(temp,condition=1, dim_count=3):
    '''
    Returns the coordinates of points inside a matrix equal to condition
    :param temp: input
    :param condition:
    :param dim_count: dimension
    :return:
    '''
    temp_indice_specific = np.where(temp==condition)
    points = []
    if len(temp_indice_specific[0])>0:
        tempsize = temp_indice_specific[0].shape[0]
        if len(temp.shape)==3:
            a = temp_indice_specific[0].reshape(tempsize, 1)
            b = temp_indice_specific[1].reshape(tempsize, 1)
            c =  temp_indice_specific[2].reshape(tempsize, 1)
            points = np.concatenate((a, b, c), axis=1)

        elif dim_count==3:
            a = temp_indice_specific[1].reshape(tempsize, 1)
            b = temp_indice_specific[2].reshape(tempsize, 1)
            c = temp_indice_specific[3].reshape(tempsize, 1)
            points = np.concatenate((a, b, c), axis=1)

        elif dim_count==4:
            a1 = temp_indice_specific[0].reshape(tempsize, 1)
            a = temp_indice_specific[1].reshape(tempsize, 1)
            b = temp_indice_specific[2].reshape(tempsize, 1)
            c = temp_indice_specific[3].reshape(tempsize, 1)
            points = np.concatenate((a1, a, b, c), axis=1)

    return points


def extract_tumors_OARs(sample, conditions, obj='tumor'):
    sample = sample.reshape(sample.shape[1],sample.shape[2],sample.shape[3])
    tumor_temp = np.where(sample==conditions[0], 1, 0)
    OAR_temp = np.where(sample==conditions[1], 1, 0)
    flag = 0
    count_tum = 1
    count_OAR = 1
    tumor_indice_specific = np.where(tumor_temp == 1)
    if len(np.unique(tumor_indice_specific[0])) == 1 or len(np.unique(tumor_indice_specific[1])) == 1 or len(
            np.unique(tumor_indice_specific[2])) == 1:
        print('there is a 2D tumor here skip!')
        flag = 1
    if len(tumor_indice_specific[0]) == 0:
        # print('No tumor')
        count_tum = 0
        flag = 1
    tumor_indice_specific = np.where(OAR_temp == 1)
    if len(np.unique(tumor_indice_specific[0])) == 1 or len(np.unique(tumor_indice_specific[1])) == 1 or len(
            np.unique(tumor_indice_specific[2])) == 1:
        print('there is a 2D OAR here skip!')
        flag = 1
    if len(tumor_indice_specific[0]) == 0:
        # print('No OAR')
        count_OAR= 0
        flag = 1
    if flag==0:
        tumor_temp_list = []
        OAR_temp_list = []
        s = generate_binary_structure(3,2)
        if count_tum == 1:
            labeled_array_tumor, tumor_count = label(tumor_temp,structure=s)
            # print('tumor count', tumor_count)
            for i in range(1, tumor_count + 1):
                tumor_temp_list.append(np.where(labeled_array_tumor == i, 1, 0))
        if count_OAR == 1:
            labeled_array_OAR, OAR_count = label(OAR_temp,structure=s)
            # print('OAR count', OAR_count)
            for i in range(1,OAR_count+1):
                OAR_temp_list.append(np.where(labeled_array_OAR == i, 1, 0))

        return tumor_temp, OAR_temp, tumor_temp_list, OAR_temp_list
    else:
        return tumor_temp, OAR_temp,[],[]



def get_target_distances(tumor_points, target_points):
    '''
    returns the target distances and target points of the data
    :param tumor_points: points of tumors
    :param target_points: number of target isocenter
    :return: Target distances and target points of the data
    '''
    real_center, radius_list, [min_x, min_y, min_z], [max_x, max_y, max_z] = get_center_tumor(tumor_points)
    [rx,ry,rz] = radius_list
    if target_points == 7:
        target_points_list = np.array([[min_x + np.round(rx / 2), real_center[1], real_center[2]],
                                     [max_x - np.round(rx / 2), real_center[1], real_center[2]],
                                     [real_center[0], min_y + np.round(ry / 2), real_center[2]],
                                     [real_center[0], max_y - np.round(ry / 2), real_center[2]],
                                     [real_center[0], real_center[1], min_z + np.round(rz / 2)],
                                     [real_center[0], real_center[1], max_z - np.round(rz / 2)],
                                     [real_center[0], real_center[1], real_center[2]]
                                     ])
        target_distances =   [(np.sqrt(radius_list[0] ** 2 + radius_list[1] ** 2) / 2),
                            (np.sqrt(radius_list[0] ** 2 + radius_list[2] ** 2) / 2),
                            (np.sqrt(radius_list[1] ** 2 + radius_list[2] ** 2) / 2),
                            radius_list[0], radius_list[1], radius_list[2]]
    elif target_points == 13:
        target_points_list = np.array(
            [[min_x + np.round(rx / 2), real_center[1], real_center[2]],
             [max_x - np.round(rx / 2), real_center[1], real_center[2]],
             [real_center[0], min_y + np.round(ry / 2), real_center[2]],
             [real_center[0], max_y - np.round(ry / 2), real_center[2]],
             [real_center[0], real_center[1], min_z + np.round(rz / 2)],
             [real_center[0], real_center[1], max_z -np.round( rz / 2)],
             [min_x, real_center[1], real_center[2]],
             [max_x, real_center[1], real_center[2]],
             [real_center[0], min_y, real_center[2]],
             [real_center[0], max_y, real_center[2]],
             [real_center[0], real_center[1], min_z],
             [real_center[0], real_center[1], max_z],
             [real_center[0], real_center[1], real_center[2]]
             ])
        target_distances = [(np.sqrt(radius_list[0] ** 2 + radius_list[1] ** 2) / 2),
                            (np.sqrt(radius_list[0] ** 2 + radius_list[2] ** 2) / 2),
                            (np.sqrt(radius_list[1] ** 2 + radius_list[2] ** 2) / 2),
                            (np.sqrt(radius_list[0] ** 2 + radius_list[1] ** 2)),
                            (np.sqrt(radius_list[0] ** 2 + radius_list[2] ** 2)),
                            (np.sqrt(radius_list[1] ** 2 + radius_list[2] ** 2)),
                            (np.sqrt((radius_list[0] / 2) ** 2 + radius_list[1] ** 2)),
                            (np.sqrt((radius_list[0] / 2) ** 2 + radius_list[2] ** 2)),
                            (np.sqrt((radius_list[0]) ** 2 + (radius_list[1] / 2) ** 2)),
                            (np.sqrt((radius_list[0]) ** 2 + (radius_list[2] / 2) ** 2)),
                            (np.sqrt((radius_list[1] / 2) ** 2 + radius_list[2] ** 2)),
                            (np.sqrt((radius_list[1]) ** 2 + (radius_list[2] / 2) ** 2)),
                            3 * radius_list[0] / 2, 3 * radius_list[1] / 2, 3 * radius_list[2] / 2,
                            radius_list[0], radius_list[1], radius_list[2],
                            2 * radius_list[0], 2 * radius_list[1], 2 * radius_list[2]]
    target_distances = np.unique(target_distances)
    return target_distances, target_points_list


