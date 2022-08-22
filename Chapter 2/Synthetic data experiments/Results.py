import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.style.use("seaborn")
import random
import os
import re
import json
from utils import fill_df_epochs, getBoxPlots
from metric_extraction import *
# Run metric extraction and create the GAN_ring based on that.
# GAN_ring is the list of all seven metrics reported in metric extraction
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='GAN', help="model name")
parser.add_argument("--structure", type=str, default='FF', help="structure of network")
parser.add_argument("--n_mixture", type=int, default=8, help="number of modes")
parser.add_argument("--name", type=str, default='name', help="file prefix name")
parser.add_argument("--files", type=list, default=[1,2,3], help="log files")
parser.add_argument("--path", type=str, default='ryerson/projects/', help="logs path")
opt = parser.parse_args()
print(opt)
num_modes_list, num_modes_thresh_list, kl_thresh_list, KL_Score_thresh_list, KL_non_thresh_list, KL_Score_no_thresh_list, High_quality_samples_list = metric_report(opt.files, opt.path)
temp =[num_modes_list, num_modes_thresh_list, kl_thresh_list, KL_Score_thresh_list, KL_non_thresh_list, KL_Score_no_thresh_list, High_quality_samples_list]
df_result = fill_df_epochs(temp, opt.model, opt.structure, opt.n_mixture)
print(df_result)
getBoxPlots(df_result,'Mode','Mode_' + opt.name, 'Number of modes','cat' ,'epochs_coded', 'Epochs')
getBoxPlots(df_result,'KL','KL_' + opt.name, 'KL divergence','cat' ,'epochs_coded', 'Epochs')
getBoxPlots(df_result,'HQ','HQ_' + opt.name, 'Percentage of high quality samples','cat' ,'epochs_coded', 'Epochs')
