
import torch
from GAN import ex
# from VARNET import ex
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--Model_type", type=str, default="Vanilla", help="Model type can be Conditional Vanilla'")
parser.add_argument("--Model_structure", type=str, default="FF", help="Model Structure can be FF , Convolutional")
parser.add_argument("--mode_collapse", type=str, default=" ", help="mode collapse can be PacGAN, minibatch")
parser.add_argument("--img_size", type=str, default='32', help="img_size")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--num_test_sample", type=int, default=26000, help="number of generated test samples)
parser.add_argument("--n_features", type=int, default=100, help="size of noise")
parser.add_argument("--dir", type=str, default='ryerson/projects/', help="directory")

opt = parser.parse_args()
print(opt)


Num = 5
for j in range(Num):

    torch.cuda.empty_cache()
    ex.run(config_updates={'num_epochs': opt.num_epochs,  'Model_structure': opt.Model_structure})


