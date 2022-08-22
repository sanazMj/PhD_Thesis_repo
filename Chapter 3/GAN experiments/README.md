# Common used files between the baseline GANs and AlphaWGANs.
* [Load_data.py](https://github.com/sanazMj/PhD_Thesis_repo/blob/main/Chapter%203/GAN%20experiments/Load_data.py) loads the data from .h5 file
* [helper.py](https://github.com/sanazMj/PhD_Thesis_repo/blob/main/Chapter%203/GAN%20experiments/helper.py) and utils_pre contain useful functions used in all the codes.
* [constants.py](https://github.com/sanazMj/PhD_Thesis_repo/blob/main/Chapter%203/GAN%20experiments/constants.py) contains a dictionary of training data attributes used for evaluation.
* [Evaluation.py](https://github.com/sanazMj/PhD_Thesis_repo/blob/main/Chapter%203/GAN%20experiments/Evaluation.py) contains a set of functions used for evaluating the saved dictionaries (logged files).
* [Results.py](https://github.com/sanazMj/PhD_Thesis_repo/blob/main/Chapter%203/GAN%20experiments/Results.py) is used to evaluate a set of files.

Both experiments are evaulated at each 100 epochs over 10000 generated samples. Generator creates 10000 samples from noise and the samples are evaluated and a dictionary of evaluation metrics is saved under /logs/resultsdict_epoch(epoch).pkl. After the model is trained, run the Results.py over this .pkl files to see the results. 
## Vanilla GAN experiments
Run the run_multiple_experiments by defining the dataset kind: {'sphere', 'Matlab'} and number of target points: {1, 7}. 
```
python run_multiple_experiments.py --num_epochs 50 --target_points 1
```

## AlphaWGAN experiments
Run the run_multiple_experiments.py by defining the main function (depending on the model and dataset). 
```
python run_multiple_experiments.py --num_epochs 50 --target_points 1
```

