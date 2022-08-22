from Main import ex
import sacred
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--Model_type", type=str, default="Conditional", help="Model type can be Conditional Vanilla'")
parser.add_argument("--Model_structure", type=str, default="FF", help="Model Structure can be FF , Convolutional")
parser.add_argument("--categorization", type=int, default=2, help="Data categorization, 2 for 9x9, 2.3 for 19x19 and 39x39, 8 for 8 cats")

parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--Pixel_Full", type=int, default=9, help="size of image height")
parser.add_argument("--full_image", type=bool, default=True, help="Using full_image or partial image")
parser.add_argument("--ndf", type=int, default=2048, help="Num Discriminator Features")
parser.add_argument("--ngf", type=int, default=512, help="Num Generator Features")
parser.add_argument("--channels", type=int, default=512, help="number of network channels")
parser.add_argument("--zdim", type=int, default=100, help="size of noise")

opt = parser.parse_args()
print(opt)

Num = 5
for i in range(Num):
     ex.run(config_updates={'num_epochs': opt.num_epochs, 'mode_collapse': '', 'categorization': opt.categorization, 'Pixel_Full': opt.Pixel_Full})


