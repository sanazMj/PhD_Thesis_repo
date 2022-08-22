from scipy.stats import mannwhitneyu, ranksums
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.style.use("seaborn")
import random
import os
import re
import json
import pandas
import argparse
from Utils.utils_def import *
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=int, default='Aug', help="Model names")
parser.add_argument("--file_names", type=list, default=[1], help="file_names in logs")
parser.add_argument("--iterations", type=list, default=list(range(1,6)), help="iterations")

opt = parser.parse_args()
print(opt)


def get_metrics(a, model_name, Cat, iterations):
    df_number = pd.DataFrame(columns=range(6))
    df_number.columns = ['Mode', 'Acc', 'Acc_wo_unknowns', 'Model', 'Cat', 'itr']
    # print(list(a[0,:]))
    df_number['Mode'] = list(a[0])
    df_number['Acc'] = list(a[1])
    df_number['Acc_wo_unknowns'] = list(a[2])

    df_number['Cat'] = [Cat] * len(a[0])
    df_number['Model'] = [str(model_name)] * len(a[0])
    df_number['itr'] = iterations
    return df_number


path =  os.getcwd() + '/logs/'
keys_list = ["(0.0, 1.0)", "(1.0, 0.0)","high_pass", "low_pass", "high_pass_without_unknowns_accuracy_without_unknowns","low_pass_without_unknowns_accuracy_without_unknowns"]
file_names = opt.file_names
model_name = opt.model_name # Define based on parameters
iterations = opt.iterations # Define based on how many files you have with the same configurations
df_result = pd.DataFrame()
df_result = pd.DataFrame(columns=range(6))
df_result.columns = ['Mode', 'Acc', 'Acc_wo_unknowns', 'Model', 'Cat', 'itr']
for file in range(iterations):
    f = open(path + str(file_name + file) + '/metrics.json')
    data = json.load(f)
    a_HP = [data["(0.0, 1.0)"]["values"][-1], data["high_pass"]["values"][-1], data["high_pass_without_unknowns_accuracy_without_unknowns"]["values"][-1]]
    a_LP = [data["(1.0, 0.0)"]["values"][-1], data[k]["low_pass"][-1], data["low_pass_without_unknowns_accuracy_without_unknowns"]["values"][-1]]
    df1 = get_metrics(a_HP, model_name, 'HP', iterations)
    df2 = get_metrics(a_LP, model_name, 'LP', iterations)
    df_result = pd.concat((df_result, df1, df2), axis=0, ignore_index=True)
print(df_result)
getBoxPlots(df_result, 'Mode','unique_' + model_name,'Percentage of unique designs',1.1, 'Model', 'Cat')
getBoxPlots(df_result, 'Acc','acc_' + model_name ,'Accuracy',1.1, 'Model', 'Cat')
getBoxPlots(df_result, 'Acc_wo_unknowns','Acc_wo_unknowns_' + model_name,'Percentage of unique designs',1.1, 'Model', 'Cat')