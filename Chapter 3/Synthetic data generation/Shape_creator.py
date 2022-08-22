
# https://medium.com/analytics-vidhya/generating-3d-point-clouds-of-geometric-shapes-spheres-pyramids-and-cubes-using-python-ada8590f16c5


import pandas as pd
import numpy as np
import random
from scipy.spatial import distance
import pickle
import h5py

from helper import get_points_from_matrix

def create_shape(center, radius, dimension_mat, shape, filled_val=1):
    '''
    Create sphere/ellipsoide shapes
    :param center:
    :param radius:
    :param dimension_mat: an initial cube to fill the points
    :param shape:
    :param filled_val:
    :return:
    '''
    for x in range(center[0] - radius[0], center[0] + radius[0] + 1):
      for y in range(center[1] - radius[1], center[1] + radius[1] + 1):
          for z in range(center[2] - radius[2], center[2] + radius[2] + 1):
              if (0<= x and x<= dimension_mat.shape[0]-1) and  (0<= y and y<= dimension_mat.shape[1]-1) and  (0<= z and z<= dimension_mat.shape[2]-1):

                  if shape == 'sphere'  and dimension_mat[x, y, z] == 0 and round(distance.euclidean([x, y, z], center)) <= radius[0]:
                    dimension_mat[x, y, z] = filled_val
                  elif shape == 'ellipsie' and dimension_mat[x, y, z] == 0:
                    a = ((x - center[0]) ** 2) / (radius[0] ** 2)
                    b = ((y - center[1]) ** 2) / (radius[1] ** 2)
                    c = ((z - center[2]) ** 2) / (radius[2] ** 2)
                    if round(a+b+c) <= 1:
                      dimension_mat[x, y, z] = filled_val
    return dimension_mat


def create_sphere_cube(center, radius, dimension_mat, filled_val,shape, space_dim=3, filled=True):
    if space_dim == 2:
        for x in range(center[0] - radius, center[0] + radius + 1):
            for y in range(center[1] - radius, center[1] + radius + 1):
                if filled:
                    if shape == 'sphere' and round(distance.euclidean([x, y], center)) <= radius:
                        dimension_mat[x, y] = filled_val
                    elif shape == 'cube':
                        dimension_mat[x, y]  = filled_val
                else:
                    if shape == 'sphere' and round(distance.euclidean([x, y], center)) == radius:
                        dimension_mat[x, y]  = filled_val
                    elif shape == 'cube':
                        dimension_mat[x, y]  = filled_val
    elif space_dim == 3:
        # print('inside space dim', center, radius, shape)
        for x in range(center[0] - radius, center[0] + radius + 1):
            for y in range(center[1] - radius, center[1] + radius + 1):
                for z in range(center[2] - radius, center[2] + radius + 1):
                    if filled:
                        if shape == 'sphere' and dimension_mat[x, y, z] == 0 and round(distance.euclidean([x, y, z], center)) <= radius:
                            # print(x,y, z,'value', dimension_mat[x, y, z])
                            dimension_mat[x, y, z] = filled_val
                            # print(x,y,z,'filled', filled_val)
                        elif shape == 'cube':
                            dimension_mat[x, y, z] = filled_val
                    else:
                        if shape == 'sphere' and dimension_mat[x, y, z] == 0 and round(distance.euclidean([x, y, z], center)) == radius:
                            dimension_mat[x, y, z] = filled_val
                        elif shape == 'cube':
                            dimension_mat[x, y, z] = filled_val
    return dimension_mat

def create_sphere(cx,cy,cz, r, resolution=360):
    '''
    create sphere with center (cx, cy, cz) and radius r
    '''
    phi = np.linspace(0, 2*np.pi, 2*resolution)
    theta = np.linspace(0, np.pi, resolution)

    theta, phi = np.meshgrid(theta, phi)

    r_xy = r*np.sin(theta)
    x = cx + np.cos(phi) * r_xy
    y = cy + np.sin(phi) * r_xy
    z = cz + r * np.cos(theta)

    return np.stack([x,y,z])



def shape_gen(space_dim, dimension, n_samples, centers,radius_list, additional_points, filled_vals = [0,1],filled=True, shape='sphere'):
    '''
    Function generates spheres
    :param space_dim:  space dimension
    :param dimension: height/width/length
    :param n_samples:  number of generated samples
    :param centers: center of shapes
    :param radius_list:  radiuses
    :param additional_points: target points to be marked and filled with spheres
    :param filled_vals: what values to be filled with
    :param filled: is the shape filled or not
    :param shape: sphere
    :return:  center_mat_sphere, center_mat, imgs
    '''
    if space_dim == 2:
        if filled_vals[0] == 0:
            imgs = np.zeros((n_samples, dimension[0], dimension[1]))
            center_mat = np.zeros((n_samples, dimension[0], dimension[1]))
            center_mat_sphere = np.zeros((n_samples, dimension[0], dimension[1]))

        elif filled_vals[0] == -1:
            imgs = -1 * np.ones((n_samples, dimension[0], dimension[1]))
            center_mat = -1 * np.ones((n_samples, dimension[0], dimension[1]))
            center_mat_sphere = np.zeros((n_samples, dimension[0], dimension[1]))


    elif space_dim == 3:
        if filled_vals[0] == 0:
            imgs = np.zeros((n_samples, dimension[0],dimension[1],dimension[2]))
            center_mat = np.zeros((n_samples,  dimension[0],dimension[1],dimension[2]))
            center_mat_sphere = np.zeros((n_samples, dimension[0], dimension[1],dimension[2]))

        elif filled_vals[0] == -1:
            imgs = -1*np.ones((n_samples, dimension[0], dimension[1], dimension[2]))
            center_mat = -1*np.ones((n_samples, dimension[0], dimension[1], dimension[2]))
            center_mat_sphere = np.zeros((n_samples, dimension[0], dimension[1],dimension[2]))


    for i in range(n_samples):
        if space_dim == 2:
            # center_mat[i,centers[i,0],centers[i,1]] = 1
            for j in range(additional_points.shape[1]):
                center_mat[i, additional_points[i,j,0], additional_points[i,j,1]] = 1
                center = [additional_points[i,j,0], additional_points[i,j,1]]
                if dataset_kind == 'iso_sphere_full':
                    center_mat_sphere[i] = create_sphere_cube(center, int(radius_list[i]/2), center_mat_sphere[i], j+1, shape,
                                                       space_dim, filled)

            imgs[i] = create_sphere_cube(centers[i], radius_list[i], imgs[i], filled_vals[1], shape, space_dim, filled)

        elif space_dim == 3:
            for j in range(len(additional_points[0])):
                center = [int(additional_points[i][j][0]), int(additional_points[i][j][1]),
                           int(additional_points[i][j][2])]
                center_mat[i, center[0],center[1],center[2]] = 1
                if dataset_kind == 'iso_sphere_full':
                    center_mat_sphere[i] = create_sphere_cube(center, int(radius_list[i] / 2), center_mat_sphere[i],
                                                                    j + 1, shape,
                                                                    space_dim, filled)

            imgs[i] = create_sphere_cube(centers[i], radius_list[i], imgs[i], filled_vals[1], shape,
                                               space_dim, filled)

    return center_mat_sphere, center_mat, imgs


def shape_gen_ellipsie(space_dim, dimension, n_samples, centers,  radius_list_x,radius_list_y,radius_list_z, additional_points, filled_vals = [0,1],filled=True):
    '''
    Function generates ellipsoids
    :param space_dim: space dimension
    :param dimension: height/width/length
    :param n_samples: number of generated samples
    :param centers: center of shapes
    :param radius_list_x: radiuses for x axis
    :param radius_list_y:  radiuses for y axis
    :param radius_list_z:  radiuses for z axis
    :param additional_points: target points to be marked and filled with spheres
    :param filled_vals: what values to be filled with
    :param filled: is the shape filled or not
    :return: center_mat, imgs
    '''

    if space_dim == 3:
        if filled_vals[0] == 0:
            imgs = np.zeros((n_samples, dimension[0],dimension[1],dimension[2]))
            center_mat = np.zeros((n_samples,  dimension[0],dimension[1],dimension[2]))
        elif filled_vals[0] == -1:
            imgs = -1*np.ones((n_samples, dimension[0], dimension[1], dimension[2]))
            center_mat = -1*np.ones((n_samples, dimension[0], dimension[1], dimension[2]))

    for i in range(n_samples):

            for j in range(len(additional_points[0])):
                center_mat[i, int(additional_points[i][j][0]), int(additional_points[i][j][1]),
                           int(additional_points[i][j][2])] = 1
            for x in range(centers[i,0]-radius_list_x[i], centers[i,0]+radius_list_x[i]+1):
                for y in range(centers[i, 1] - radius_list_y[i], centers[i, 1] + radius_list_y[i]+1):
                    for z in range(centers[i, 2] - radius_list_z[i], centers[i, 2] + radius_list_z[i]+1):
                        a = ((x - centers[i, 0]) ** 2) / (radius_list_x[i] ** 2)
                        b = ((y - centers[i, 1]) ** 2) / (radius_list_y[i] ** 2)
                        c = ((z - centers[i, 2]) ** 2) / (radius_list_z[i] ** 2)
                        if filled:

                            if round(a+b+c) <= 1:
                                imgs[i,x,y,z] = filled_vals[1]

                        else:
                            if round(a+b+c) == 1:
                                imgs[i,x,y,z] = filled_vals[1]


    return center_mat, imgs

def get_random_shape_info_ellipsie(space_dim, dimension,target_points, n_samples):
    '''
    Function returns the center, radiuses and target points of the shape
    :param space_dim: dimension of space
    :param dimension: shape dimension
    :param target_points: target number of subspheres
    :param n_samples: number of generated samples
    :return: centers, radius_list_x, radius_list_y, radius_list_z, additional_points

    '''
    centers = []
    additional_points = []
    radius_list_x= []
    radius_list_y= []
    radius_list_z= []

    constraints =[]
    for i in range(n_samples):
        additional_points_temp = []
        center = np.random.randint(0,dimension[0],size=(1,space_dim))
        for j in range(space_dim):
            constraints = constraints + [dimension[j] - 1 - center[0, j]]
            constraints = constraints + [center[0, j] - 1]

        while np.min(constraints)<4:
            center = np.random.randint(0, dimension[0], size=(1, space_dim))
            constraints = []
            for j in range(space_dim):
                constraints = constraints + [dimension[j] - 1 - center[0, j]]
                constraints = constraints +[center[0, j] - 1]

        r =  np.random.randint(size=(3),low=2,high=np.min(constraints))
        if target_points == 7:
            additional_points_temp.append(center.tolist()[0])
            for j in range(space_dim):
                change = np.zeros((1, space_dim))
                change[0,j] = 1

                additional_points_temp.append((center + int(r[j]/2)*change).tolist()[0])
                additional_points_temp.append((center - int(r[j]/ 2) * change).tolist()[0])

        elif target_points == 10:
            additional_points_temp.append(center.tolist()[0])
            for j in range(space_dim):
                change = np.zeros((1, space_dim))
                change[0, j] = 1

                for delta in range(-r[j],  r[j] + 1):
                    temp = (center + int(delta) * change).tolist()[0]
                    temp1 = [int(x) for x in temp]
                    additional_points_temp.append(temp1)
        elif target_points == 13:
            additional_points_temp.append(center.tolist()[0])
            for j in range(space_dim):
                change = np.zeros((1, space_dim))
                change[0, j] = 1

                for delta in [-r[j],-int(r[j]/2),int(r[j]/2),r[j]]:
                    temp = (center + int(delta) * change).tolist()[0]
                    temp1 = [int(x) for x in temp]
                    additional_points_temp.append(temp1)
        else:
            additional_points_temp.append(center.tolist())

        centers.append(center.reshape(space_dim).tolist())

        radius_list_x.append(r[0])
        radius_list_y.append(r[1])
        radius_list_z.append(r[2])
        additional_points.append(additional_points_temp)

    return np.array(centers), np.array(radius_list_x),  np.array(radius_list_y), np.array(radius_list_z),(additional_points)


def get_random_shape_info( space_dim, dimension,target_points, n_samples):
    '''
    Function returns the center, radius and target points of the sphere

    :param space_dim: dimension of space
    :param dimension: shape dimension
    :param target_points: target number of subspheres
    :param n_samples: number of generated samples
    :return: centers, radius_list, additional_points
    '''
    centers = []
    additional_points = []
    radius_list= []
    constraints =[]
    for i in range(n_samples):
        additional_points_temp = []
        center = np.random.randint(0,dimension[0],size=(1,space_dim))
        for j in range(space_dim):
            constraints = constraints + [dimension[j] - 1 - center[0, j]]
            constraints = constraints + [center[0, j] - 1]

        while np.min(constraints)<4:
            center = np.random.randint(0, dimension[0], size=(1, space_dim))
            constraints = []
            for j in range(space_dim):
                constraints = constraints + [dimension[j] - 1 - center[0, j]]
                constraints = constraints +[center[0, j] - 1]

        r =  np.random.randint(3,np.min(constraints))
        if target_points == 7:
            additional_points_temp.append(center.tolist()[0])
            for j in range(space_dim):
                change = np.zeros((1, space_dim))
                change[0,j] = 1
                additional_points_temp.append((center + int(r/2)*change).tolist()[0])
                additional_points_temp.append((center - int(r / 2) * change).tolist()[0])

        elif target_points == 10:
            additional_points_temp.append(center.tolist()[0])
            for j in range(space_dim):
                change = np.zeros((1, space_dim))
                change[0, j] = 1
                for delta in range(-r,  r + 1):
                    temp = (center + int(delta) * change).tolist()[0]
                    temp1 = [int(x) for x in temp]
                    additional_points_temp.append(temp1)
        elif target_points == 13:
            additional_points_temp.append(center.tolist()[0])
            for j in range(space_dim):
                change = np.zeros((1, space_dim))
                change[0, j] = 1
                for delta in [-r,-int(r/2),int(r/2),r]:
                    temp = (center + int(delta) * change).tolist()[0]
                    temp1 = [int(x) for x in temp]
                    additional_points_temp.append(temp1)
        else:
            additional_points_temp.append(center.tolist()[0])

        centers.append(center.reshape(space_dim).tolist())
        radius_list.append(r)
        additional_points.append(additional_points_temp)

    return np.array(centers), np.array(radius_list), (additional_points)


def shape_gen_without_tumor_isos_integers(dimension, n_samples, min_ratio, target_points, space_dim):
    '''
    Function creates mixture of  sphere/ellipsoid shapes

    :param dimension: height,width, length
    :param n_samples: number of generated samples
    :param min_ratio: smallest ratio of subsphere coverage/entire space
    :param target_points: number of target subspheres
    :param space_dim: space dimension
    :return: dimension_mat, center_mat, center_list, radius_list, additional_points_list
    '''
    temp, center_list, radius_list = fill_cube_with_spheres_diff_rs_ns(dimension, int(n_samples), min_ratio)
    center_list_new, radius_list_new, additional_points_list = get_target_points(center_list, radius_list, target_points=target_points,space_dim=space_dim)
    dimension_mat = np.zeros((len(radius_list_new), dimension[0], dimension[1],dimension[2]))
    center_mat = np.zeros((len(radius_list_new), dimension[0], dimension[1],dimension[2]))
    j = 0
    while j < n_samples:
        new_centers = additional_points_list[j]
        new_radius_list = [int(radius_list_new[j][0]/4),int(radius_list_new[j][1]/4), int(radius_list_new[j][2]/4) ]

        if new_radius_list[0] == new_radius_list[1] and new_radius_list[2] == new_radius_list[1]:
            shape = 'sphere'
        else:
            shape = 'ellipsie'
        flag = 0
        if new_radius_list[0] >=2 and new_radius_list[1] >=2 and new_radius_list[2] >=2 :
            flag = flag + 0
        else:
            flag = flag + 1
        if flag == 0:
            for index, center in enumerate(new_centers):
                x = center[0]
                y = center[1]
                z = center[2]
                if (0 <= x and x <= (dimension[0] - 1)) and (0 <= y and y <= (dimension[1] - 1)) and (0 <= z and z <= (dimension[2] - 1)):
                    flag = flag + 0
                else:
                    flag = flag + 1
        if flag == 0:
            for index, center in enumerate(new_centers):
                    center = list(map(int, center))
                    center_mat[j][center[0], center[1], center[2]] = 1
                    dimension_mat[j] = create_shape(center, new_radius_list, dimension_mat[j], shape, filled_val=index+1)
            print('created', j, '/', n_samples)
        j += 1
    return dimension_mat, center_mat, center_list, radius_list, additional_points_list

def shape_gen_tumor_integers(dimension, n_samples, min_ratio, target_points, space_dim,min_r):
    temp, center_list, radius_list = fill_cube_with_spheres_diff_rs_ns(dimension, int(n_samples), min_ratio, min_r=min_r)
    center_list_new = [j for i in center_list for j in i]
    radius_list_new = [j for i in radius_list for j in i]
    print(len(center_list_new), len(radius_list_new))
    dimension_mat = np.zeros((n_samples, dimension[0], dimension[1],dimension[2]))
    center_mat = np.zeros((n_samples, dimension[0], dimension[1],dimension[2]))
    j = 0
    while j < n_samples:
        center = center_list_new[j]
        new_radius_list  = radius_list_new[j]
        if new_radius_list[0] == new_radius_list[1] and new_radius_list[2] == new_radius_list[1]:
            shape = 'sphere'
        else:
            shape = 'ellipsie'
        flag = 0
        if new_radius_list[0] >=2 and new_radius_list[1] >=2 and new_radius_list[2] >=2 :
            flag = flag + 0
        else:
            flag = flag + 1
        if flag == 0:
            x = center[0]
            y = center[1]
            z = center[2]
            if (0 <= x and x <= (dimension[0] - 1)) and (0 <= y and y <= (dimension[1] - 1)) and (0 <= z and z <= (dimension[2] - 1)):
                flag = flag + 0
            else:
                flag = flag + 1
        if flag == 0:
            center = list(map(int, center))
            center_mat[j][center[0], center[1], center[2]] = 1
            dimension_mat[j] = create_shape(center, new_radius_list, dimension_mat[j], shape, filled_val=1)
            print('created', j, '/', n_samples)
        j += 1

    return dimension_mat, center_mat, center_list, radius_list


def save_random_points(dataset_kind, space_dim, dimension,n_samples,target_points,filled, filled_vals=[0,1],
                       Data_path = '/ryerson/projects/',shape='sphere', name_choice = '', min_ratio=0.4):
    '''
    function generates sphere/ellipsoid 3D shapes with target isocenters and saves them as dictionary
    :param space_dim: dimension of space, 3, 2
    :param dimension: width.length, height of shape
    :param n_samples: number of generated samples
    :param target_points: number of target isocenters
    :param filled: is it a filled sphere or not
    :param filled_vals: what values should it be filled with
    :param Data_path: directory
    :param shape: sphere/ellipsoid/Both
    :param name_choice: prefix of name
    :param min_ratio: ratio of subspheres/ entire sphere
    :return: saved files
    '''
    dict_data = {}
    if shape == 'ellipsoid':
        centers, radius_list_x, radius_list_y, radius_list_z, additional_points = get_random_shape_info_ellipsie(space_dim, dimension, target_points, n_samples)
        centers_temp, s = shape_gen_ellipsie(space_dim, dimension,n_samples, centers, radius_list_x, radius_list_y, radius_list_z, additional_points,filled_vals=filled_vals,filled=filled)
    elif shape == 'sphere':
        centers, radius_list, additional_points = get_random_shape_info(space_dim, dimension, target_points, n_samples)
        center_mat_sphere, centers_temp, s = shape_gen(space_dim, dimension,n_samples, centers, radius_list, additional_points, shape=shape, filled_vals=filled_vals,filled=filled)
    elif shape == 'Both':
        center_mat_sphere, centers_temp, centers, radius_list, additional_points_list= shape_gen_without_tumor_isos_integers(dimension, n_samples, min_ratio, target_points, space_dim)
    dict_data['centers'] = centers
    dict_data['centers_temp'] = centers_temp
    dict_data['center_mat_sphere'] = center_mat_sphere

    if shape != 'Both':
        dict_data['data'] = s
    if shape == 'ellipsoid':
        dict_data['radius_list_x'] = radius_list_x
        dict_data['radius_list_y'] = radius_list_y
        dict_data['radius_list_z'] = radius_list_z

    else:
        dict_data['radius_list'] = radius_list

    string_dimension = [str(int) for int in dimension]
    string_dimension = "_".join(string_dimension)
    if filled == False:
        with open(Data_path + name_choice + dataset_kind + '_' + shape + '_'+ string_dimension + '_' +str(n_samples)+ '_' +str(target_points)+'_not_filled_'+'_data.pkl', 'wb') as handle:
            pickle.dump(dict_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(Data_path + name_choice + dataset_kind + '_'+ shape + '_'+ string_dimension + '_' +str(n_samples)+ '_' +str(target_points)+'_data.pkl', 'wb') as handle:
            pickle.dump(dict_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_random_points(dataset_kind, space_dim, dimension, n_samples, target_points, shape,filled_vals=[0,1],filled=True,name_choice = '',
                      Data_path='ryerson/projects/tumorGAN/Data/'
                      ):
    '''
    Load the saved data from pickle files
    :param space_dim: dimension of space, 3, 2
    :param dimension: width.length, height of shape
    :param n_samples: number of generated samples
    :param target_points: number of target isocenters
    :param filled: is it a filled sphere or not
    :param filled_vals: what values should it be filled with
    :param Data_path: directory
    :param shape: sphere/ellipsoid/Both
    :param name_choice: prefix of name
    :return:  center_mat_sphere, centers_temp, s,centers,radius_list
    '''
    dict_data = {}
    string_dimension = [str(int) for int in dimension]
    string_dimension = "_".join(string_dimension)
    if filled == False:
        with open(Data_path + name_choice + dataset_kind + '_'+ shape + '_'+ string_dimension + '_' +str(n_samples)+ '_' +str(target_points)+'_not_filled_'+'_data.pkl','rb') as f:
            hf = pickle.load(f)
    else:
        with open(Data_path + name_choice + dataset_kind + '_'+ shape + '_'+ string_dimension + '_' +str(n_samples)+ '_' +str(target_points)  +'_data.pkl','rb') as f:
            hf = pickle.load(f)
    s  =[]
    centers = hf['centers']
    centers_temp = hf['centers_temp']
    center_mat_sphere = hf['center_mat_sphere']
    if 'data' in list(hf.keys()):
        s = hf['data']
    if 'radius_list' in list(hf.keys()):
        radius_list = hf['radius_list']
    else:
        radius_list = [hf['radius_list_x'],hf['radius_list_y'],hf['radius_list_z']]

    return center_mat_sphere, centers_temp, s,centers,radius_list


def find_point_on_perimeter(points):
    a_temp = np.array([[1,0,0],[0,1,0],[0,0,1],
                       [-1,0,0],[0,-1,0],[0,0,-1]])
    primeter_temp = []
    flag = 0
    for i in range(len(points)):
        flag = 0
        for j in range(len(a_temp)):
            if flag == 0:
                a = points[i] + a_temp[j]
                if not (points==a).all(1).any():
                    primeter_temp.append(points[i])
                    flag = 1

    return primeter_temp


from GAN_simple_3D.Sphere_creation_optimizer import *


def shape_gen_tumor(dimension, n_samples, min_ratio, target_points, space_dim, min_r=8):
    temp, center_list, radius_list = fill_cube_with_spheres_diff_rs_ns(dimension, int(n_samples), min_ratio, min_r=min_r)
    center_list_new = [j for i in center_list for j in i]
    radius_list_new = [j for i in radius_list for j in i]
    print(len(center_list_new), len(radius_list_new))
    # center_list_new, radius_list_new, additional_points_list = get_target_points(center_list, radius_list, target_points=target_points,space_dim=space_dim)
    dimension_mat = np.zeros((len(center_list_new), dimension[0], dimension[1], dimension[2]))
    center_mat = np.zeros((len(center_list_new), dimension[0], dimension[1], dimension[2]))
    for j in range(len(center_list_new)):
        center = center_list_new[j]
        radius_list = radius_list_new[j]
        if radius_list[0] == radius_list[1] and radius_list[2] == radius_list[1]:
            shape = 'sphere'
        else:
            shape = 'ellipsie'

        center = list(map(int, center))
        center_mat[j][center[0], center[1], center[2]] = 1
        dimension_mat[j] = create_shape(center, radius_list, dimension_mat[j], shape, filled_val=1)
        # visualize_multi_colored_isos(dimension_mat[j], 'tumor', 0, j, [], [], set_lim=False, save=False)
        # break
    return dimension_mat, center_mat, center_list_new, radius_list_new


def eliminate_width(center_mat_sphere, centers1, imgs, centers_list, radius_list, condition=0):
    width_all = []
    elim_indexes = []
    size = [condition, condition, condition]
    result_temp = []
    count = 0
    for i in range(center_mat_sphere.shape[0]):
        # print(count)
        a = np.where(center_mat_sphere[i] != 0, 1, 0)
        tumor_points = get_points_from_matrix(a, condition=1)
        if condition == 0:
            result_temp = []
            if len(tumor_points) <= 0:
                elim_indexes.append(i)
        # print(len(tumor_points))
        else:
            if len(tumor_points) > 0:
                width = [np.max(tumor_points[:, 0]) - np.min(tumor_points[:, 0]),
                         np.max(tumor_points[:, 1]) - np.min(tumor_points[:, 1]),
                         np.max(tumor_points[:, 2]) - np.min(tumor_points[:, 2])]
                if np.max(width) <= condition:
                    width_all.append(width)
                    temp1 = np.zeros(size, dtype=np.int8)
                    temp1[:np.max(tumor_points[:, 0]) - np.min(tumor_points[:, 0]),
                    :np.max(tumor_points[:, 1]) - np.min(tumor_points[:, 1]),
                    :np.max(tumor_points[:, 2]) - np.min(tumor_points[:, 2])] = center_mat_sphere[i][
                                                                                np.min(tumor_points[:, 0]): np.max(
                                                                                    tumor_points[:, 0]),
                                                                                np.min(tumor_points[:, 1]): np.max(
                                                                                    tumor_points[:, 1]),
                                                                                np.min(tumor_points[:, 2]): np.max(
                                                                                    tumor_points[:, 2])]
                    if count == 0:
                        result_temp = temp1.reshape(1, condition, condition, condition)
                    else:
                        temp1 = temp1.reshape(1, condition, condition, condition)
                        result_temp = np.concatenate((result_temp, temp1))
                    count += 1
                else:
                    elim_indexes.append(i)


            else:
                elim_indexes.append(i)

    # print(np.max(width_all), np.min(width_all))
    a = [radius_list[i] for i in range(center_mat_sphere.shape[0]) if i not in elim_indexes]
    b = [centers_list[i] for i in range(center_mat_sphere.shape[0]) if i not in elim_indexes]

    center_mat_sphere = np.delete(center_mat_sphere, elim_indexes, axis=0)
    return center_mat_sphere,  centers1, imgs, b, a , result_temp







