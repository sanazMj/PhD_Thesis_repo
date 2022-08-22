import numpy as np
def metric_report(files, path):
    num_modes_list = []
    num_modes_thresh_list = []
    kl_thresh_list = []
    KL_Score_thresh_list = []
    KL_non_thresh_list = []
    KL_Score_no_thresh_list = []
    High_quality_samples_list = []
    for num in files:
        num_modes = []
        num_modes_thresh = []
        kl_thresh = []
        KL_Score_thresh = []
        KL_non_thresh = []
        KL_Score_no_thresh = []
        High_quality_samples = []
        my_lines = []
        # print(path + str(num) + '/cout.txt')
        f = open(path + str(num) + '/cout.txt', 'rt')
        for my_line in f:
            my_lines.append(my_line.rstrip('\n'))
        # a = f.read()
        f.close()
        for i in range(len(my_lines)):
            Number_modes_point = my_lines[i].find('Number of modes')
            Number_modes_no_thresh_point = my_lines[i].find('Number of modes without thresh')
            kl_thresh_point = my_lines[i].find('KL thresh')
            KL_Score_thresh_point = my_lines[i].find('KL_Score thresh')
            KL_non_thresh_point = my_lines[i].find('KL non-thresh')
            KL_Score_no_thresh_point = my_lines[i].find('KL_Score non-thresh')
            High_quality_samples_point = my_lines[i].find('High quality samples')

            if Number_modes_point != -1:
                # print('a',Number_modes_point)
                # print(my_lines[i][Number_modes_point + 16: Number_modes_no_thresh_point-2])
                num_modes.append(int(my_lines[i][Number_modes_point + 16: Number_modes_no_thresh_point - 2]))

            if Number_modes_no_thresh_point != -1:
                # print('b',Number_modes_no_thresh_point)
                # print(my_lines[i][Number_modes_no_thresh_point + 31: kl_thresh_point])
                num_modes_thresh.append(int(my_lines[i][Number_modes_no_thresh_point + 31: kl_thresh_point]))

            if kl_thresh_point != -1:
                # print('c',kl_thresh_point,my_lines[i][kl_thresh_point+ 10: KL_Score_thresh_point])
                if my_lines[i][kl_thresh_point + 10: KL_Score_thresh_point] == '':
                    kl_thresh.append(0)
                else:
                    kl_thresh.append(np.round(float(my_lines[i][kl_thresh_point + 10: KL_Score_thresh_point]), 3))

            if KL_Score_thresh_point != -1:
                # print('d',my_lines[i][KL_Score_thresh_point+ 16: KL_non_thresh_point ])
                KL_Score_thresh.append(np.round(float(my_lines[i][KL_Score_thresh_point + 16: KL_non_thresh_point]), 3))

            if KL_non_thresh_point != -1:
                # print('e',my_lines[i][KL_non_thresh_point+ 14: KL_Score_no_thresh_point])
                KL_non_thresh.append(
                    np.round(float(my_lines[i][KL_non_thresh_point + 14: KL_Score_no_thresh_point]), 3))

            if KL_Score_no_thresh_point != -1:
                # print('f',my_lines[i][KL_Score_no_thresh_point+ 20: High_quality_samples_point])
                KL_Score_no_thresh.append(
                    np.round(float(my_lines[i][KL_Score_no_thresh_point + 20: High_quality_samples_point]), 3))

            if High_quality_samples_point != -1:
                # print('g',my_lines[i][High_quality_samples_point + 21: my_lines[i].find('Epoch')])
                High_quality_samples.append(
                    np.round(float(my_lines[i][High_quality_samples_point + 21: my_lines[i].find('Epoch')]), 3))

        num_modes_list.append(num_modes)
        num_modes_thresh_list.append(num_modes_thresh)
        kl_thresh_list.append(kl_thresh)
        KL_Score_thresh_list.append(KL_Score_thresh)
        KL_non_thresh_list.append(KL_non_thresh)
        KL_Score_no_thresh_list.append(KL_Score_no_thresh)
        High_quality_samples_list.append(High_quality_samples)
    return  num_modes_list, num_modes_thresh_list, kl_thresh_list, KL_Score_thresh_list, KL_non_thresh_list, KL_Score_no_thresh_list, High_quality_samples_list






import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--files", type=list, default=[1,2,3], help="log files")
parser.add_argument("--path", type=str, default='ryerson/projects/', help="logs path")

opt = parser.parse_args()
print(opt)

path =  opt.path #Define this
files = opt.files
num_modes_list, num_modes_thresh_list, kl_thresh_list, KL_Score_thresh_list, KL_non_thresh_list, KL_Score_no_thresh_list, High_quality_samples_list = metric_report(files, path)
print(num_modes_list)
print(',', num_modes_thresh_list)
print(',', kl_thresh_list)
print(',', KL_Score_thresh_list)
print(',', KL_non_thresh_list)
print(',', KL_Score_no_thresh_list)
print(',', High_quality_samples_list)


