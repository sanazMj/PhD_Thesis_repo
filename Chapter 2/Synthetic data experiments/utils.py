# %%

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

from torch.autograd.variable import Variable
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.distributions as D

from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp

import random
import matplotlib.pyplot as plt
import imageio
import numpy as np
import math
from functools import reduce
# %%
def get_p_value(x,y, mode):
  U1, p = mannwhitneyu(x, y,alternative=mode)
  print(U1,p)

  def getBoxPlotsComp(gdf, y_col, name, y_label, legend_title):
      fig = plt.figure(figsize=(12, 5))
      sns.set_context("paper", font_scale=1.4)

      # sns.boxplot(x='n_series', y="ND", hue="selection_mode", data=df_tmp)
      g = snsa.boxplot(x='size', y=y_col, hue="cat", data=gdf).set(
          xlabel='Target Mode',
          ylabel=y_label
      )

      # Drawing a horizontal line at point 1.25
      # g.axhline(gdTargetVal)
      # plt.legend(bbox_to_anchor=(0.02,1.2), ncol=4,fontsize=12 )
      # if y_col == 'Mode':
      #   plt.ylim([0,40])
      plt.legend(bbox_to_anchor=(0.8, 1.4), ncol=4, title=legend_title)
      # plt.title(gsTitle)

      plt.tight_layout()  # for printing out x-y labels properly
      # plt.savefig(gsfn)
      plt.savefig('boxplot' + name + '.pdf', bbox_inches='tight')
      # plt.savefig('boxplot'+name+'.pdf')

def getBoxPlots_modeComp(gdf, y_col, name, y_label):
  fig, axs = plt.subplots(1, 3, figsize=(12, 5))
  # axs.xaxis.label.set_size(12)

  g = sns.boxplot(x='cat', y=y_col, data=gdf[gdf['size'] == 8], ax=axs[0]).set(
      xlabel='Model',
      ylabel=y_label
  )
  axs[0].set_ylim(0, 37)
  axs[0].tick_params(axis='x', rotation=45)

  axs[0].title.set_text('ring mode 8')
  g1 = sns.boxplot(x='cat', y=y_col, data=gdf[gdf['size'] == 25], ax=axs[1]).set(
      xlabel='Model',
      ylabel=y_label
  )
  axs[1].set_ylim(0, 37)
  axs[1].tick_params(axis='x', rotation=45)
  axs[1].title.set_text('grid mode 25')

  g2 = sns.boxplot(x='cat', y=y_col, data=gdf[gdf['size'] == 36], ax=axs[2]).set(
      xlabel='Model',
      ylabel=y_label
  )
  axs[2].set_ylim(0, 37)
  axs[2].tick_params(axis='x', rotation=45)

  axs[2].title.set_text('grid mode 36')

  plt.legend(bbox_to_anchor=(0.8, 1.3), ncol=5)

  plt.tight_layout()  # for printing out x-y labels properly
  plt.savefig('boxplot' + name + '.pdf', bbox_inches='tight')


def fill_f_epochs(a, cat_name, model_name, size, var_coef, slope, coef,
                  level_line, epochs):
    df = pd.DataFrame(columns=range(12))
    df.columns = ['Mode', 'KL', 'HQ', 'cat', 'Model',
                  'var_coef', 'Level', 'epochs', 'epochs_coded', 'coef', 'slope', 'size']

    df['Mode'] = a[0]
    df['KL'] = a[2]
    df['HQ'] = a[-1]
    df['epochs'] = epochs
    df['size'] = size
    df['var_coef'] = var_coef
    df['Level'] = level_line
    df['coef'] = coef
    df['slope'] = slope
    df['cat'] = cat_name
    df['Model'] = model_name
    for i in range(len(df)):

        if df.iloc[i, 7] < 5:
            df.iloc[i, 8] = '<5'
        elif 5 <= df.iloc[i, 7] < 10:
            df.iloc[i, 8] = '5-10'
        elif 10 <= df.iloc[i, 7] < 15:
            df.iloc[i, 8] = '10-15'
        elif 15 <= df.iloc[i, 7] < 25:
            df.iloc[i, 8] = '15-25'
        elif 25 <= df.iloc[i, 7]:
            df.iloc[i, 8] = '25-50'

    return df


def fill_df_epochs(a, cat_name, model_name, size):
    Max_range = []
    if a.shape[1] == 60:

        for i in range(a.shape[2]):
            if i % 6 != 1 or i % 60 == 1:
                Max_range.append(i)

        duration = len(Max_range) // (a.shape[1] // 60)
        epochs = list(range(duration)) * int(a.shape[1] // 60)
    elif a.shape[2] > 51:

        b = [1, 52, 103, 154, 205, 256, 307, 358, 408]
        for i in range(a.shape[2]):
            if i not in b:
                Max_range.append(i)
        epochs = list(range(len(Max_range))) * a.shape[1]
        indexes = a[:, :, Max_range]
        indexes = indexes.reshape(-1, indexes.shape[1] * indexes.shape[2])

    else:
        Max_range = list(range(a.shape[2]))
        epochs = list(range(len(Max_range))) * a.shape[1]
        indexes = a[:, :, Max_range]
        indexes = indexes.reshape(-1, indexes.shape[1] * indexes.shape[2])

    cat_name_list = [cat_name] * len(epochs)
    model_name_list = [model_name] * len(epochs)
    size_list = [size] * len(epochs)
    var_coef = [1] * len(epochs)
    slope = [1] * len(epochs)
    coef = [1] * len(epochs)
    level_line = [1] * len(epochs)

    df = pd.DataFrame(columns=range(12))
    df.columns = ['Mode', 'KL', 'HQ', 'cat', 'Model',
                  'var_coef', 'Level', 'epochs', 'epochs_coded', 'coef', 'slope', 'size']

    df['Mode'] = indexes[0]
    df['KL'] = indexes[2]
    df['HQ'] = indexes[-1]
    df['epochs'] = epochs
    df['size'] = size_list
    df['var_coef'] = var_coef
    df['Level'] = level_line
    df['coef'] = coef
    df['slope'] = slope
    df['cat'] = cat_name_list
    df['Model'] = model_name_list

    for i in range(len(df)):

        if df.iloc[i, 7] < 5:
            df.iloc[i, 8] = '<5'
        elif 5 <= df.iloc[i, 7] < 10:
            df.iloc[i, 8] = '5-10'
        elif 10 <= df.iloc[i, 7] < 15:
            df.iloc[i, 8] = '10-15'
        elif 15 <= df.iloc[i, 7] < 25:
            df.iloc[i, 8] = '15-25'
        elif 25 <= df.iloc[i, 7]:
            df.iloc[i, 8] = '25-50'

    return df


def getBoxPlots(gdf, y_col, name, y_label, x_name, hue_name, legend_title):
    fig = plt.figure(figsize=(12, 5))
    sns.set_context("paper", font_scale=1.4)

    g = sns.boxplot(x=x_name, y=y_col, hue=hue_name, data=gdf).set(
        xlabel='Model',
        ylabel=y_label
    )

    plt.legend(bbox_to_anchor=(0.75, 1.3), ncol=6, title=legend_title)

    plt.tight_layout()
    plt.savefig('boxplot' + name + '.pdf', bbox_inches='tight')


def get_metrics(a, cat_name, model_name, size, epoch):
    df = pd.DataFrame(columns=range(9))
    df.columns = ['Mode-mean', 'Mode-std', 'KL-mean', 'KL-std', 'HQ-mean',
                  'HQ-std', 'cat', 'Model', 'size']

    df['Mode-mean'] = [np.round(np.mean(a[0, :, epoch]), 4)]
    df['Mode-std'] = [np.round(np.std(a[0, :, epoch]), 4)]
    df['KL-mean'] = [np.round(np.mean(a[2, :, epoch]), 4)]
    df['KL-std'] = [np.round(np.std(a[2, :, epoch]), 4)]
    df['HQ-mean'] = [np.round(np.mean(a[-1, :, epoch]), 4)]
    df['HQ-std'] = [np.round(np.std(a[-1, :, epoch]), 4)]
    df['size'] = size
    df['cat'] = cat_name
    df['Model'] = model_name
    return df


def getBoxPlots(gdf, y_col, name, y_label, x_name, hue_name, legend_title):
    fig = plt.figure(figsize=(12, 5))
    sns.set_context("paper", font_scale=1.4)

    # sns.boxplot(x='n_series', y="ND", hue="selection_mode", data=df_tmp)
    g = sns.boxplot(x=x_name, y=y_col, hue=hue_name, data=gdf).set(
        xlabel='Model',
        ylabel=y_label
    )

    # plt.legend(bbox_to_anchor=(0.8,1.3), ncol=5)x_legend=0.65, y_legend=1.3
    plt.legend(bbox_to_anchor=(0.75, 1.3), ncol=6, title=legend_title)
    # plt.title(gsTitle)

    plt.tight_layout()  # for printing out x-y labels properly
    # plt.savefig(gsfn)
    plt.savefig('boxplot' + name + '.pdf', bbox_inches='tight')


def euclidean_distance(gVec, fVec):
    return np.sqrt((gVec[0] - fVec[0]) ** 2 + (gVec[1] - fVec[1]) ** 2)


def classify(points, centroids, labels, std_dev):
    Labels_list = []# it does  matter how far the point is from the centroid. It should be less than 3*std
    lables_without_thresh =[] # it does not matter how far the point is from the centroid.
    Min_Dist = []
    for i in range(points.shape[0]):
        min_dist = np.inf
        for j in range(centroids.shape[0]):
            distance = euclidean_distance(points[i, :], centroids[j, :])
            if distance < min_dist:
                min_dist = distance
                label = labels[j]
        lables_without_thresh.append(label)

        if min_dist > 3 * std_dev:
            label = 'None'
        Min_Dist.append(min_dist)
        Labels_list.append(label)

    return np.array(Labels_list), np.array(Min_Dist), np.array(lables_without_thresh)


# %%

class JSD:
    def KLD(self, p, q):
        if 0 in q:
            raise ValueError
        return sum(_p * log(_p / _q) for (_p, _q) in zip(p, q) if _p != 0)

    def JSD_core(self, p, q):
        M = [0.5 * (_p + _q) for _p, _q in zip(p, q)]
        return 0.5 * self.KLD(p, M) + 0.5 * self.KLD(q, M)


# %%
def create_prob(labels, labels_without_thresh, n_mixtures):
    modes = {}
    for i in labels:
        if i != 'None':
            if i in list(modes.keys()):
                modes[i] += 1
            else:
                modes[i] = 1
    num_mode = len(modes.keys())

    p = np.zeros(n_mixtures)
    p = list(modes.values())
    p = p / np.sum(p)

    modes1 = {}
    for i in labels_without_thresh:
        if i in list(modes1.keys()):
            modes1[i] += 1
        else:
            modes1[i] = 1
    num_mode1 = len(modes1.keys())

    p1 = np.zeros(n_mixtures)
    p1 = list(modes1.values())
    p1 = p1 / np.sum(p1)
    return p, num_mode, p1, num_mode1

def evaluate_modes_reverse_KL(generator, test_samples, q, q_without_thresh, n_mixtures, tensor_loc, label, std_dev=0.01):

    generated_points = torch.tensor([])
    if test_samples > 2000:
        divider = 2000
    else:
        divider = test_samples
    for rounds in range(test_samples//divider):
        test_noise = noise(divider)
        img = generator(test_noise).cpu().detach()
        generated_points = torch.cat([generated_points,img],0)

    labels, min_dist, labels_without_thresh = classify(generated_points, tensor_loc, label, std_dev)
    # labels_training, min_dist_training, labels_without_thresh_training = classify(training_data, tensor_loc, label, std_dev)

    high_quality_samples = sum(min_dist < 3 * std_dev) / generated_points.shape[0]
    p, num_mode, p1, num_mode1 = create_prob(labels, labels_without_thresh, n_mixtures)
    # q, q_num_mode, q1, q_num_mode1 = create_prob(labels_training, labels_without_thresh_training, n_mixtures)

    # print(p, q)
    # print(p1, q_without_thresh)
    print(' Number of modes %d  Number of modes without thresh %d ' % (num_mode, num_mode1))
    KL_thresh, KL_Score_thresh, KL_non_thresh, KL_Score_non_thresh= JSD().KLD(p, q_without_thresh), JSD().JSD_core(p, q_without_thresh), JSD().KLD(p1, q_without_thresh), JSD().JSD_core(p1, q_without_thresh)
    print('KL thresh', KL_thresh)
    print('KL_Score thresh', KL_Score_thresh)

    print('KL non-thresh', KL_non_thresh)
    print('KL_Score non-thresh', KL_Score_non_thresh)

    print('High quality samples', high_quality_samples)
    return num_mode, KL_thresh, KL_Score_thresh, KL_non_thresh, KL_Score_non_thresh, high_quality_samples


def evaluate_high_quality_samples(generated_points, n_mixtures, tensor_loc, label, std_dev=0.01):
    labels, min_dist = classify(generated_points, tensor_loc, label)
    return sum(min_dist < 3 * std_dev) / generated_points.shape[0]


# %%

def sample_gen_Grid2D(batch_size, n_mixture=25, std=0.05):
    n_point = int(np.sqrt(n_mixture))
    x = []
    distance = 2/n_point
    for i in range(n_point):
        for j in range(n_point):
            x.append([-1 + distance * i, -1 + distance * j])

    tensor_loc = torch.tensor(x).type(torch.float32)
    cat = D.Categorical(torch.ones(n_mixture))

    # x = [([xi, yi]) for xi, yi in zip(xs.ravel(), ys.ravel())]
    # tensor_loc = torch.tensor([([xi, yi]) for xi, yi in zip(xs.ravel(), ys.ravel())])
    labels = list(range(n_mixture))

    x = [([std, std]) for i in range(n_mixture)]
    tensor_sd = torch.tensor([([std, std]) for i in range(n_mixture)])

    comps = D.Independent(D.Normal(tensor_loc, tensor_sd), 1)

    data = D.MixtureSameFamily(cat, comps)
    Data_sampled = data.sample(range(batch_size, batch_size + 1))
    # samples_labels = classify(Data_sampled, tensor_loc, labels)

    return Data_sampled, tensor_loc, labels


def sample_gen_ring2D(batch_size, n_mixture=8, std=0.01, radius=1.0):
    thetas = 2 * np.pi / 8 * np.array(list(range(8)))
    # thetas = np.linspace(0, 2 * np.pi-0.2, n_mixture)
    xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
    cat = D.Categorical(torch.ones(n_mixture))

    x = [([xi, yi]) for xi, yi in zip(xs.ravel(), ys.ravel())]
    tensor_loc = torch.tensor([([xi, yi]) for xi, yi in zip(xs.ravel(), ys.ravel())])

    x = [([std, std]) for i in range(n_mixture)]
    tensor_sd = torch.tensor([([std, std]) for i in range(n_mixture)])

    labels = list(range(n_mixture))
    comps = D.Independent(D.Normal(tensor_loc, tensor_sd), 1)

    data = D.MixtureSameFamily(cat, comps)
    Data_sampled = data.sample(range(batch_size, batch_size + 1))
    # samples_labels = classify(Data_sampled, tensor_loc, labels)
    return Data_sampled, tensor_loc, labels


def noise(n, n_features=128):
    return Variable(torch.randn(n, n_features)).cuda()


def make_ones(size):
    data = Variable(torch.ones(size, 1))
    return data.cuda()


def make_zeros(size):
    data = Variable(torch.zeros(size, 1))
    return data.cuda()


def train_discriminator(optimizer, discriminator, criterion, real_data, fake_data):
    n = real_data.size(0)

    optimizer.zero_grad()

    prediction_real = discriminator(real_data)
    # print(prediction_real)
    error_real = criterion(prediction_real, make_ones(n))
    # print(error_real)
    error_real.backward()

    prediction_fake = discriminator(fake_data)
    error_fake = criterion(prediction_fake, make_zeros(n))

    error_fake.backward()
    optimizer.step()

    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(optimizer, discriminator, criterion, fake_data):
    n = fake_data.size(0)
    optimizer.zero_grad()

    prediction = discriminator(fake_data)
    error = criterion(prediction, make_ones(n))

    error.backward()
    optimizer.step()

    return error, prediction


# def train_varnet(optimizer, varnet, criterion, real_data, fake_data):
#     optimizer.zero_grad()
#
#     # print('v',real_data.shape, fake_data.shape)
#
#     n = real_data.shape[0]
#     m = fake_data.shape[0]
#     # print(n,m)
#
#     prediction_real = varnet(real_data)
#     # print(prediction_real.shape, make_ones(n).shape)
#     error_real = criterion(prediction_real, make_ones(n))
#     error_real.backward()
#
#     # print(fake_data.shape)
#     prediction_fake = varnet(fake_data)
#     # print(prediction_fake.shape, make_zeros(m).shape)
#     error_fake = criterion(prediction_fake, make_zeros(m))
#
#     error_fake.backward()
#     optimizer.step()
#
#     return error_real + error_fake, prediction_real, prediction_fake
def train_varnet(optimizer, varnet, criterion, real_data_reshape, total_label_diff, real_data, num_var, pac_var, batch_size, n_mixture, same_creation_type):
    optimizer.zero_grad()

    # print('v',real_data.shape, fake_data.shape)

    if same_creation_type == 0 or same_creation_type == 1:
        n = real_data_reshape.shape[0]
        prediction_real = varnet(real_data_reshape)
        error_real = criterion(prediction_real, make_ones(n))

    else:
        prediction_real = varnet(real_data_reshape[0],real_data_reshape[1], real_data_reshape[2] )
        error_real = criterion(prediction_real, total_label_diff)
    # print(prediction_real.shape, make_ones(n).shape)
    # print(make_ones(n))
    # print(prediction_real)
    # print(criterion(prediction_real, make_ones(n)))
    error_real.backward()


    error_fake = 0.0
    # num_var = get_divisibles(batch_size)
    # print(num_var)
    # if same_creation_type == 0:
    #     num_var = [1,2,4,5,10,20,25,50,100]
    # elif same_creation_type == 1:
    #     num_var = [1,2,4,5,10,20,25,50]
    # elif same_creation_type == 2:
    #     num_var = [1,2,4,5,10,20,25]
    # if n_mixture == 8:
    #     num_var = [1, 2, 4]
    # elif n_mixture == 25:
    #     num_var = [1, 5, 10, 20]
    # elif n_mixture == 36:
    #     num_var = [1, 2, 4, 8, 12, 16, 18, 24]
    if same_creation_type == 0 or same_creation_type == 1:
        fake_data_same, label = create_sim_data(real_data,batch_size, pac_var, n_mixture, type=same_creation_type )
        prediction_fake = varnet(fake_data_same)
        error_fake += criterion(prediction_fake, label)

    else:
        fake_data_same, label = create_sim_data_new(real_data,batch_size, pac_var, n_mixture, type=same_creation_type )
        prediction_fake = varnet(fake_data_same[0], fake_data_same[1],fake_data_same[2])
        # print(prediction_fake.shape, label.shape)
        error_fake += criterion(prediction_fake, label)
    # print(num_var)
    # fake_data_same, label = create_sim_data(real_data,  num_var[np.random.randint(0,len(num_var)-1,1)[0]],batch_size, type=same_creation_type )
    # num_var = [1, 2, 4, 5]
    # fake_data_same, label = create_sim_data(real_data, 1)
        # print(fake_data_same.shape, label.shape)


    # print(label, label.type)
    # print(prediction_fake)
    # print(criterion(prediction_fake, label))
    # print(fake_data.shape)


    error_fake.backward()
    optimizer.step()

    return error_real + error_fake, prediction_real, prediction_fake



def train_generator_varnet(optimizer, discriminator, varnet, criterion, var_coef, pac_var, fake_data, fake_data1, label1, same_creation_type=0):
    n = fake_data.size(0)

    optimizer.zero_grad()
    # print(fake_data.shape, fake_data1.shape)
    prediction_d = discriminator(fake_data)
    error_d = criterion(prediction_d, make_ones(n))


    if same_creation_type == 0 or same_creation_type == 1:
        prediction_v = varnet(fake_data1)
        error_v = criterion(prediction_v, make_ones(fake_data1.shape[0])) * var_coef

    else:
        prediction_v = varnet(fake_data1[0], fake_data1[1], fake_data1[2])
        error_v = criterion(prediction_v, label1) * var_coef



    # error_v = criterion(prediction_v, make_ones(fake_data1.shape[0])) * var_coef
    error = error_v + error_d

    error.backward()
    optimizer.step()

    return error_v + error_d, prediction_d, prediction_v

def create_sim_data(imgs, batch_size,  pac_var, n_mixture, type = 0):
    '''
    Construct the similar images for VARNET training
    :param imgs: generator output
    :return: The first element of generator ouput is repeated and passed
    '''
    if type == 0:
        num_var = [1, 2, 4, 5, 10, 20, 25, 50, 100]
    elif type == 1 :
        if n_mixture == 8:
            num_var = [1, 2, 4]
        elif n_mixture == 25:
            num_var = [1, 2, 4, 5, 10, 20]
        elif n_mixture == 36:
            num_var = [1, 2, 4, 9, 12, 18, 24]
    # elif type == 1:
    #     num_var = [1, 2, 4, 5, 10, 20, 25, 50]

    num = num_var[np.random.randint(0, len(num_var) - 1, 1)[0]]
    indexes = random.sample(list(range(imgs.shape[0])), num)

    # indexes = np.random.randint(0,imgs.shape[0]-1, num)
    fake_data = imgs[indexes, :]
    fake_data = fake_data.repeat(imgs.shape[0]//len(indexes), 1)
    if type == 0 or type == 1:
        if num == 1:
            label = torch.tensor([0.0])
        else:
            label = torch.tensor([0.51 - (1./ num)])


    fake_data_same_reshape = fake_data.reshape(fake_data.shape[0] // pac_var,fake_data.shape[1] * pac_var)
    fake_data_same_reshape = fake_data_same_reshape.cuda()

    label = label.repeat(fake_data_same_reshape.shape[0], 1)
    label = label.cuda()


    return fake_data_same_reshape, label

def create_sim_data_new(imgs, batch_size, pac_var, n_mixture, type = 0):
    '''
    Construct the similar images for VARNET training
    :param imgs: generator output
    :return: The first element of generator ouput is repeated and passed
    '''
    if type == 0 :
        num_var = [1, 2, 4, 5, 10, 20, 25, 50, 100]
    elif type == 1 :
        if n_mixture == 8:
            num_var = [1, 2, 4]
        elif n_mixture == 25:
            num_var = [1, 2, 4, 5, 10, 20]
        elif n_mixture == 36:
            num_var = [1, 2, 4, 9, 12, 18, 24]
        nums = num_var[np.random.randint(0, len(num_var) - 1, 1)[0]]

    if type == 2:
        if n_mixture == 8:
            num_var = [1, 2, 3, 4, 5]
            num_var_multiple_select = [1,2,4]
            nums = [3, num_var_multiple_select[np.random.randint(0, len(num_var_multiple_select) - 1, 1)[0]], 5]
        elif n_mixture == 25:
            num_var = [1, 2, 4, 5, 10, 15, 20]
            num_var_multiple_select = [1, 2, 4]
            num_var_multiple_select2 = [5, 10, 20]

            nums = [ num_var_multiple_select[np.random.randint(0, len(num_var_multiple_select) - 1, 1)[0]], 15,
                     num_var_multiple_select2[np.random.randint(0, len(num_var_multiple_select2) - 1, 1)[0]]]
        elif n_mixture == 36:
            num_var = [1, 2, 4, 8, 12, 16, 18, 24]
            num_var_multiple_select = [1, 2, 4, 8, 12, 24]
            nums = [16,18,num_var_multiple_select[np.random.randint(0, len(num_var_multiple_select) - 1, 1)[0]]]
    # print(nums)
    fake_data_list = []


    for i, num in enumerate(nums):
        indexes = np.random.randint(0,imgs.shape[0]-1, num)
        fake_data = imgs[indexes, :]
        fake_data = fake_data.repeat(imgs.shape[0]//(len(indexes)*len(nums)), 1)
        if num == 1:
            label = torch.tensor([0.0])
        else:
            label = torch.tensor([0.51 - (1./ num)])
        fake_data = fake_data.reshape(fake_data.shape[0] // pac_var[i], fake_data.shape[1] * pac_var[i]).cuda()
        fake_data_list.append(fake_data)
        label = label.repeat(fake_data.shape[0], 1)
        if i == 0:
            total_label = label
        else:
            total_label = torch.cat((total_label,label), axis = 0)
        # print(label.shape, total_label.shape, fake_data_list[i].shape)
    total_label = total_label.cuda()

    #
    # fake_data_same_reshape = fake_data_same.reshape(fake_data_same.shape[0] // pac_var,fake_data_same.shape[1] * pac_var)
    # fake_data_same_reshape = fake_data_same_reshape.cuda()
    # print(nums)
    # print(fake_data_list[0].shape, fake_data_list[1].shape, fake_data_list[2].shape, total_label.shape)
    # print(total_label)

    return fake_data_list, total_label


def create_dif_data(real_data, pac_var, batch_size, type= 0):
    if type == 0 or type == 1:
        real_data_final = real_data.reshape(real_data.shape[0] // pac_var, real_data.shape[1] * pac_var).cuda()
        total_label = make_ones(real_data_final.shape[0])
    elif type == 2:
        real_data_final = []
        for index, pac_var_element in enumerate(pac_var):
            real_data_temp = real_data[(batch_size//3 )* index : (batch_size//3 )* (index+1)]
            real_data_reshape = real_data_temp.reshape(real_data_temp.shape[0] // pac_var_element, real_data_temp.shape[1] * pac_var_element).cuda()
            real_data_final.append(real_data_reshape)
            # print(real_data_temp.shape, real_data_reshape.shape )
            label = make_ones(real_data_reshape.shape[0])

            if index == 0:
                total_label = label
            else:
                total_label = torch.cat((total_label, label), axis=0)
    # print(real_data_final[0].shape, real_data_final[1].shape, real_data_final[2].shape, total_label.shape)
    # print(total_label)
    return real_data_final, total_label

def get_divisibles(num):
    divisibles = []
    for i in range(1, int(np.sqrt(num)) +1):
        if num%i == 0:
            divisibles.append(i)
            divisibles.append(int(num/i))
    return list(np.sort(np.unique(divisibles)))

def lcm(a,b):
    return int(a*b/math.gcd(a,b))

def get_lcms(*array):
    return reduce(lcm, array)



class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims,Minibatch_kind = 'L1 Norm', mean=False):
        super(MinibatchDiscrimination, self).__init__()
        self.Minibatch_kind = Minibatch_kind
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        init.normal(self.T, 0, 1)

    def forward(self, x):
        # x is NxA
        # T is AxBxC
        # print(x.shape)
        # print(self.T.view(self.in_features, -1).shape)
        matrices = x.mm(self.T.view(self.in_features, -1))
        # print(matrices.shape)
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC


        # print('temp',temp[0])
        if self.Minibatch_kind == 'L1 Norm':
            norm = torch.abs(M - M_T).sum(3)
            expnorm = torch.exp(-norm)
        elif self.Minibatch_kind == 'L2 Norm':
            norm = ((torch.abs(M - M_T))**2).sum(3)
            expnorm = torch.exp(-norm)
        elif self.Minibatch_kind == 'identical':
            norm = torch.abs(M - M_T).sum(3)
            expnorm = torch.exp(-norm*100)
        elif self.Minibatch_kind == 'Cosine':
            cos1 = torch.nn.CosineSimilarity(dim=3)
            norm = cos1(M, M_T)
            expnorm = torch.exp(-norm)


        o_b = (expnorm.sum(0) - 1)   # NxB, subtract self distance
        # o_b = (exp_Sum - 1)
        if self.mean:
            o_b /= x.size(0) - 1
        # print('minibatch ob ', o_b.shape)
        x = torch.cat([x, o_b], 1)
        return x

def label_selection(labels, num):

    unique_labels, indices = np.unique(labels, return_index=True)
    # Choose unique designs without repetition
    if num < len(unique_labels):
        indexes = indices[random.sample(range(len(indices)), num)]
    else:
        indexes = random.sample(range(len(labels)), num)

    return indexes