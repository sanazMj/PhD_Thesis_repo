import argparse
from Shape_creator import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_kind", type=str, default="iso_sphere_full", help="dataset_kind is_sphere_full or Matlab (Tumors)")
parser.add_argument("--dataset_size", type=int, default=40000, help="dataset_size")

parser.add_argument("--shape_choice", type=str, default="Both", help="shape_choice can be sphere, ellipsie, Both")
parser.add_argument("--dimension_num", type=int, default=3, help="Space Dimension")
parser.add_argument("--dimension", type=list, default= [16, 16, 16], help="dimension of shapes")
parser.add_argument("--target_points", type=int, default=7, help="target_points 7 for 3D volumes, 23 for tumors")
parser.add_argument("--name_choice", type=str, default="name", help="prefix name for saving plots")
parser.add_argument("--dir", type=str, default='ryerson/projects/', help="directory")

opt = parser.parse_args()
print(opt)

dataset_kind = opt.dataset_kind
dimension_num = opt.dimension_num
dimension = opt.dimension
dataset_size =opt.dataset_size
target_points = opt.target_points
shape_choice = opt.shape_choice
name_choice = opt.name_choice
dir = opt.dir
###############################################################
################## For 3D connected volume generation #########
###############################################################

# provide information to create the random points and save them as data
save_random_points(dataset_kind, dimension_num, dimension, dataset_size, target_points, shape=shape_choice,
                       filled_vals=[0, 1], filled=True,
                       Data_path=dir)
# Load data
center_mat_sphere, centers, imgs, centers_list, radius_list = get_random_points(dataset_kind, dimension_num,
                                                                                     dimension,
                                                                                     dataset_size, target_points,
                                                                                     name_choice=name_choice,
                                                                                     shape=shape_choice,Data_path=dir)



###############################################################
################## For 3D connected tumor generation#########
###############################################################
from Create_dataset import * 



x_limit = dimension[0]
Data_path_input = dir # Path of generated tumor shapes using Matlab

file_names, min_indexes_tumor, max_indexes_tumor = get_tumor_coord(x_limit,Data_path_input)
create_python_dataset_from_matfiles(file_names,name_choice,min_indexes_tumor,
                                     max_indexes_tumor,size=(x_limit,x_limit,x_limit),
                                     Data_path=Data_path_input,values=[1,1])

