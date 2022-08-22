
from torch.autograd.variable import Variable
import torch
import pickle
def pickle_load(dir):
    """Loads an object from given directory using pickle"""
    with open(dir, 'rb') as handle:
        obj = pickle.load(handle)
    return obj

def non_zero_dict(class_dict_list):
    len_non_zero = 0
    print(class_dict_list)
    for i in range(len(class_dict_list)):
            if class_dict_list[i] != 0:
                len_non_zero += 1
    return len_non_zero

def noise(size, zdim, pixel=False):
    # if pixel:
    #     n = Variable(torch.randn(size, pixel*pixel))
    # else:
    n = Variable(torch.randn(size, zdim))
    if torch.cuda.is_available(): return n.cuda()
    return n

def get_p_value(x,y, mode):
  # print(x,y)
  if len(x)==len(y) and np.allclose(x,y):
    p = 0.0001
    U1 = 0.0001
  elif np.allclose(x[:min(len(x),len(y))],y[:min(len(x),len(y))]):
    p = 0.0001
    U1 = 0.0001
  else:
    U2, p2 = ranksums(x, y,alternative=mode)
    U1, p = mannwhitneyu(x, y,alternative=mode)
  return U1,p, U2, p2


def create_p_val_df(temp1, temp2, Model1, Model2):
  df = pd.DataFrame(columns=range(5))
  df.columns = ['Model1', 'Model2', 'Mode', 'Acc', 'Acc_without']

  df['Model1'] = Model1
  df['Model2'] = Model2
  # s1, p1 = get_p_value(temp1[0,:,-1], temp2[0,:,-1],'less')
  s2, p2 = get_p_value(temp1[0,:], temp2[0,:],'greater')
  # s3, p3 = get_p_value(temp1[0,:,-1], temp2[0,:,-1],'two-sided')

  df['Mode'] = [p2] #[min(p1,p2)]
  # print(np., df['Mode'])
  # s1, p1 = get_p_value(temp1[2,:,-1], temp2[2,:,-1],'less')
  s2, p2 = get_p_value(temp1[1,:], temp2[1,:],'greater')
  # s3, p3 = get_p_value(temp1[0,:,-1], temp2[0,:,-1],'two-sided')

  df['Acc'] =  [p2] #[min(p1,p2)]
  # s1, p1 = get_p_value(temp1[-1,:,-1], temp2[-1,:,-1],'less')
  s2, p2 = get_p_value(temp1[2,:], temp2[2,:],'greater')
  # s3, p3 = get_p_value(temp1[0,:,-1], temp2[0,:,-1],'two-sided')

  # df['HQ'] =  [min(p1,p2)]
  df['Acc_without'] =  [p2]

  return df

def create_p_values_df(Models,Model_names):
  result_mode = np.zeros((len(Models),len(Models)))
  result_KL = np.zeros((len(Models),len(Models)))
  result_HQ = np.zeros((len(Models),len(Models)))
  result_Acc_without = np.zeros((len(Models),len(Models)))
  df_all = []
  count = 0
  # for i in range(0,len(Models)-1):
  #   for j in range(i+1, len(Models)):
  for i in range(len(Models)):
    for j in range(len(Models)):
      # print(Model_names[i],Model_names[j])
      df = create_p_val_df(Models[i],Models[j], [Model_names[i]],[Model_names[j]])
      result_mode[i,j] = df['Mode']
      result_Acc[i,j] = df['Acc']
      result_Acc_without[i,j] = df['Acc_without']
      if count == 0:
        df_all = df
      else:
        df_all = pd.concat([df_all, df],axis=0, ignore_index=True)
      count += 1
  return df_all,result_mode, result_Acc, result_Acc_without



def plt_heatmap(Labels,temp, name):
  plt.figure(figsize=(10,10))
  yticklabels = Labels
  # mask = np.where(temp==0,1,0)
  yticks = [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5]
  heat_map = sns.heatmap( temp, linewidth = 1 , annot = True, xticklabels = yticklabels, yticklabels=yticklabels)
  # plt.title( "HeatMap using Seaborn Method" )
  plt.xticks(yticks,rotation=90)
  plt.yticks(yticks,rotation=0,va="center")
  # plt.savefig(name + '.pdf', bbox_inches='tight')
  plt.show()

def getBoxPlots_unique(gdf, x_col, y_col, name, y_label):
  fig = plt.figure(figsize=(12, 5))
  sns.set_context("paper", font_scale=1.4)

  # sns.boxplot(x='n_series', y="ND", hue="selection_mode", data=df_tmp)
  g = sns.boxplot(x=x_col, y=y_col, data=gdf).set(
      xlabel='Model',
      ylabel=y_label
  )

  # plt.legend(bbox_to_anchor=(0.8,1.3), ncol=5)
  # plt.title(gsTitle)

  plt.tight_layout()  # for printing out x-y labels properly
  plt.setp(g.get_xticklabels(), rotation=90)
  # plt.savefig(gsfn)
  plt.show()
  # plt.savefig('boxplot'+name + '.pdf', bbox_inches='tight')

def getBoxPlots(gdf, y_col, name, y_label, ylim, x_name, hue_name):
    fig = plt.figure(figsize=(12,5))
    sns.set_context("paper", font_scale=1.4)

    # sns.boxplot(x='n_series', y="ND", hue="selection_mode", data=df_tmp)
    ax = sns.boxplot(x = x_name, y=y_col ,hue = hue_name, data=gdf)
    ax.set_xlabel('Models')
    ax.set_ylabel(y_label)
    ax.set_ylim(0,ylim)
    # plt.legend(bbox_to_anchor=(0.8,1.3), ncol=5)
    plt.legend(bbox_to_anchor=(1.17,1), ncol=1, title='Hidden layers')
    # plt.title(gsTitle)
    plt.setp(ax.get_xticklabels(), rotation=90)
    plt.tight_layout() # for printing out x-y labels properly
    # plt.savefig(gsfn)
    # plt.show()
    plt.savefig('boxplot'+name + '.pdf', bbox_inches='tight')

def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

def create_test_samples(categorization, num_condition, pixel, zdim, test_samples_per_category=1000):
    """
        Creates labels for test samples"""
    num_test_samples = test_samples_per_category * num_condition
    # zdim = 100
    test_samples = noise(num_test_samples, zdim, pixel)

    labels_of_test_samples = Variable(torch.zeros(num_test_samples, categorization)).cuda()
    for i in range(num_test_samples):
        cat_idx = i % categorization
        labels_of_test_samples[i][cat_idx] = 1
    return test_samples, labels_of_test_samples

def convert_tuple_binary_to_int(input_tuple):
    temp = ''
    for i in range(len(input_tuple)):
        temp += str(int(input_tuple[i]))
    return temp
def oct2array(octList, even_flag=False, side=None):
    """converts list of octant values to square array,
    size is determined by length of the octant values and the even_flag
    """
    if side is None:
        side = parDim2side(len(octList), even_flag=even_flag)
    c1 = side // 2
    c2 = int((side + 1) // 2)

    # mask=np.concatenate(([np.concatenate((np.zeros(side-1-k),np.ones(k+1))) for k in range(c2)],np.zeros((c1,r))))
    return np.concatenate(([np.concatenate(
        (np.zeros(side - 1 - k), octList[(k * (k + 1)) // 2:((k + 2) * (k + 1)) // 2])) for k in range(c2)],
                           np.zeros((c1, side))))


def eightfold_sym2(array_in):
    """produces array with 8 fold symmetry based on the contents of the first octant of  array_in
    Added support for even length sides
    """
    array = array_in.copy()
    r = array.shape[0]
    c1 = r // 2
    c2 = (r + 1) // 2

    mask = np.concatenate(
        ([np.concatenate((np.zeros(r - 1 - k), np.ones(k + 1))) for k in range(c2)], np.zeros((c1, r))))

    marray = array * mask
    # print(marray)
    marray += np.rot90(np.rot90(np.transpose(marray)))
    for k in range(c2):
        marray[c2 - 1 - k, c1 + k] /= 2
    # print(marray)
    marray += np.fliplr(marray)
    if np.mod(r, 2) == 1:
        for k in range(c2):
            marray[k, c2 - 1] /= 2
        # print(marray)
    marray += np.flipud(marray)
    if np.mod(r, 2) == 1:
        for k in range(r):
            marray[c2 - 1, k] /= 2
    return marray

