
This repository contains the code for the second chapter of my PhD thesis, called "Novel Generative Adversarial Network Architectures for Generating Discrete Image Data".
This repository contains the codes for the following publication:

Mohammadjafari, S., Cevik, M. \& Basar, A. VARGAN: variance enforcing network enhanced GAN, Applied Intelligence (2022). http://dx.doi.org/10.1007/s10489-022-03199-8
# Repository structure
There are three folders available, each containing the codes for different datasets. 

[Synthetic data experiments](https://github.com/RyersonU-DataScienceLab/Sanaz_VARGAN/tree/main/Synthetic%20data%20experiments) folder contains the experiments for synthetic ring and grids.

[Stacked MNIST experiments](https://github.com/RyersonU-DataScienceLab/Sanaz_VARGAN/tree/main/stacked%20MNIST%20experiments) folder contains the experiments for stacked MNIST data.

[EES experiments](https://github.com/RyersonU-DataScienceLab/Sanaz_VARGAN/tree/main/EES%20experiments) folder contains files on EES training for 9x9 and 19x19 designs. 

# Structure of experiments
All the experiments pretty much follow the same routine. 
* Creating the dataset
* Running the run_multiple_exp.py files. 

This file uses Sacred library to run multiple experiments using different setup arguments. After the model is trained, a number of test samples are generated and evaluated to generate specific metrics. The results are saved in logs folder under the experiments' id in different files such as cout.txt and metrics.json.

* Running the metric extraction or results files to extract the results from the saved files and either report the average performance over number of experiments or plot the specific plots.


