import numpy as np
import csv

path ='F:\deeplearning_dataset\\ashrae-energy-prediction\\'
def load_data(file):
    data_list =[]
    openfile = open(path+file,'r')
    read = csv.reader(openfile)
    for line in read:
        data_list.append(line)
    data_list = np.array(data_list)
    return data_list


train_data = load_data('train.csv')
print(train_data.shape)

# test_data = load_data('test.csv')
# print(test_data.shape)

