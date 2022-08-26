# Dataset:
The EES dataset used in this project is private and provided by [Communication Research Centre Canada](https://ised-isde.canada.ca/site/communications-research-centre-canada/en/about-crc). The details on the dataset's structure is provided in [paper](http://dx.doi.org/10.1007/s00521-020-05656-2). One can use the following directions to create a synthetic version of the dataset:
* 9x9 dataset generation process:
  * Generate every binary pattern in a 5x5 square. 
  * Use half of the square, rotate it and combine the rotations to have eight fold symmetry.
# Training:
You can train the model by calling [run_multiple_experiments.py](https://github.com/sanazMj/PhD_Thesis_repo/blob/main/Chapter%201/EES%20experiments/run_multiple_experiments.py). Set the configuration details as you wish. 

Configuations include:
* Model_structure: { 'Convolutional' ,'FF'}
* Model_type :{'Conditional', 'Vanilla'}
* categorization : {2, 2.3, 8}
* full_image: {False, True} 
* Pixel_Full:{9, 19}
* num_epochs
* zdim: dimension of the noise vector
* batch_size
* channels
*  ndf = 2048 # Num Discriminator Features
*  ngf = 512 # Num Generator Features

The run_multiple_experiments.py will generate three folders called (logs, results and runs). The logs folder contains four files:
* config.json : Contains the configurations above.
* cout.txt: Contains all the printed texts in the shell.
* metrics.json: Contains the metrics saved by "save_generation_distribution" function, including accuracy, number of unique items.
* run.json: Used for tensorboard. 
# Results:
Run [Results.py](https://github.com/sanazMj/PhD_Thesis_repo/blob/main/Chapter%201/EES%20experiments/Results.py) to create the tables and figures in the paper. The file reads the metrics.json files of specified logs and illustrates the results.

Or, Use [MakeFile](https://github.com/sanazMj/PhD_Thesis_repo/blob/main/Chapter%201/EES%20experiments/Makefile) to run all the necessary files to create different tables of the thesis.
