import torch
from main_sphere_ellipsoid import ex
import sacred
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_size", type=int, default=100000, help="dataset_size")
parser.add_argument("--target_points", type=int, default=7, help="target_points")
parser.add_argument("--num_epochs", type=int, default=600, help="num_epochs")
parser.add_argument("--shape", type=str, default='Both', help="shape")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--size_wanted", type=int, default=40000, help="changed size of dataset")
parser.add_argument("--connectivity", type=bool, default=True, help="Add connection loss or not")
parser.add_argument("--dir", type=str, default='ryerson/projects/', help="directory")

opt = parser.parse_args()
print(opt)
Num = 5
for j in range(Num):
        torch.cuda.empty_cache()
        ex.run(config_updates={'dataset_size':opt.dataset_size, 'size_wanted':opt.size_wanted,'batch_size':opt.batch_size,
       'num_epochs': num_epochs,'connectivity':opt.connectivity, 'Data_path':opt.dir,'target_points':opt.target_points})



