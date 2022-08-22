import torch
from Main_AlphaWGAN_connectedspheres import ex
import sacred
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_kind", type=str, default='Matlab', help="dataset_kind")
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
for i in range(Num):
     ex.run(config_updates={'dataset_kind':opt.dataset_kind, 'connectivity':opt.connectivity})



