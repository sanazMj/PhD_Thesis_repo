
[Conditional GAN MINST.ipynb](https://github.com/RyersonU-DataScienceLab/Sanaz_GAN_EES/blob/main/MNIST%20experiments/Conditional%20GAN%20MINST.ipynb): 
Runs conditional vanilla GAN on MNIST data. One-hot encoding label is used as auxiliary information (e.g., digit 3 one-hot encoding is [0,0,0,1,0,0,0,0,0,0]).

[Convolutional GAN MINST.ipynb](https://github.com/RyersonU-DataScienceLab/Sanaz_GAN_EES/blob/main/MNIST%20experiments/Convolutional%20GAN%20MINST.ipynb):
Runs convolutional GAN on MNIST data. 

[cDCGAN-Type1.ipynb](https://github.com/RyersonU-DataScienceLab/Sanaz_GAN_EES/blob/main/MNIST%20experiments/cDCGAN-Type1.ipynb):
Runs conditional convolutional GAN on MNIST data. There are different GAN structures on how to add the labels for conditioning. In this type and through out our entire work,
we add merge the conditions with noise for the generator and with the real data for the discriminator. (For instance, one can add the label in the middle of the network).

[Conditional GAN MINST - Mixture of Labels.ipynb](https://github.com/RyersonU-DataScienceLab/Sanaz_GAN_EES/blob/main/MNIST%20experiments/Conditional%20GAN%20MINST%20-%20Mixture%20of%20Labels.ipynb):
Runs conditional vanilla GAN structure on MNIST dataset. The labels are designed to be a mixutre of two different numbers. The code investigates the effect of 
a mixture of labels on the generated shapes. (e.g., can a new digit be generated to have the characteristics of both digits 3 and 5?)
