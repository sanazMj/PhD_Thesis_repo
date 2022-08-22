# Dataset:
A sample of the dataset can be created using the functions
```
sample_gen_ring2D(batch_size, n_mixture=8, std=0.01, radius=1.0) 
sample_gen_Grid2D(batch_size, n_mixture=25, std=0.05)
sample_gen_Grid2D(batch_size, n_mixture=36, std=0.05)
```
from [utils.py](https://github.com/RyersonU-DataScienceLab/Sanaz_VARGAN/blob/main/Synthetic%20data%20experiments/utils.py) file.
The first function creates a 2D ring with 8 modes and the second and thirds one create a synthetic grid data with 25 and 36 modes.
# Training:
You can train the model by calling [run_multiple_experiments.py](https://github.com/RyersonU-DataScienceLab/Sanaz_VARGAN/blob/main/Synthetic%20data%20experiments/run_multiple_exps.py). Set the configuration details as you wish. 
```
python run_multiple_exps.py --num_epochs 50 --dataset 'grid' --n_mixture 36
```

# Results:
Use [metric_extraction.py](https://github.com/RyersonU-DataScienceLab/Sanaz_VARGAN/blob/main/Synthetic%20data%20experiments/metric_extraction.py) to print seven different metrics for your experiments based on the name of files in your log folder (Files are created by Sacred).
```
python metric_extraction.py --path '\ryerson\projects\' --files [1,2,3]
```

Combine these seven metrics and pass them to [Results.py](https://github.com/RyersonU-DataScienceLab/Sanaz_VARGAN/blob/main/Synthetic%20data%20experiments/Results.py) to plot the figures.
```
python Results.py --temp [] --model 'GAN' --structure 'FF' --n_mixture 36 --name 'grid_36'
```
Use MakeFile to run all the experiments for Table 3.2.
