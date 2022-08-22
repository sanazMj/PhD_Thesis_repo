# Dataset:
The stacked version of MNIST data can be created by calling following function from [utils.py](https://github.com/RyersonU-DataScienceLab/Sanaz_VARGAN/blob/main/Stacked%20MNIST%20experiments/utils.py).
```
load_data_create(num_training_sample, img_size)
```

# Training:
You can train the model by calling [run_multiple_exp.py](https://github.com/RyersonU-DataScienceLab/Sanaz_VARGAN/blob/main/Stacked%20MNIST%20experiments/run_multiple_exp.py). Set the configuration details as you wish. 
```
python run_multiple_exps.py --num_epochs 50 
```
# Results:
Run the metrix_extraction_MNIST.py to find the results over seven different metrics.
```
python metric_extraction_MNIST.py --path '\ryerson\projects' --files [1,2,3]
```
