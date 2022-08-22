import sacred
import torch
from GAN_synthetic_data import ex
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--Model_type", type=str, default="Vanilla", help="Model type can be Conditional Vanilla'")
parser.add_argument("--Model_structure", type=str, default="FF", help="Model Structure can be FF , Convolutional")
parser.add_argument("--dataset", type=str, default='ring', help="Dataset kind ring or grid")
parser.add_argument("--n_mixture", type=int, default=8, help="number of modes, 8, 25, 36")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--test_noise_dim", type=int, default=2000, help="number of generated test samples)
parser.add_argument("--n_features", type=int, default=128, help="size of noise")
parser.add_argument("--dir", type=str, default='ryerson/projects/', help="directory")
parser.add_argument("--num_iterations", type=int, default=5, help="num_iterations")

opt = parser.parse_args()
print(opt)

for j in range(opt.num_iterations):
     print('GAN ring', j)
     torch.cuda.empty_cache()
     ex.run(config_updates={'num_epochs': opt.num_epochs, 'dataset': opt.dataset, 'n_mixture':opt.n_mixture, 'dir':opt.dir})
