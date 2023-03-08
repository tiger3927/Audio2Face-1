import numpy as np
import os

def propare_data(project_dir):
    """ Concat the data together and split them to training and testing sets
    Args:
        project_dir: str, project dir contains lpc and bs_value
    """
    dataSet_dir = os.path.join(project_dir, 'dataSet_mine')
    if not os.path.exists(dataSet_dir):
        os.mkdir(dataSet_dir)

    train_data = []
    val_data = []
    train_label_var_mouth = []
    val_label_var_mouth = []
    train_label_var_other = []
    val_label_var_other = []

    name_list = [fn.split('.')[0] for fn in os.listdir('./lpc')]
    data_path_list = [os.path.join(project_dir, 'lpc', i + '.npy') for i in name_list]
    label_path_list = [os.path.join(project_dir, 'bs_value', 'BS_' + i + '.npy') for i in name_list]

    data = np.zeros((1, 32, 64, 1))
    label = np.zeros((1, 63))

    # generate data and label
    for i in range(len(data_path_list)):
        data_temp = np.load(data_path_list[i])
        label_temp = np.load(label_path_list[i])

        data = np.vstack((data, data_temp))
        label = np.vstack((label, label_temp))
        print(len(data_temp), len(label_temp), len(data_temp) == len(label_temp))

    data = data[1:]
    label = label[1:]

    # split data in to train and val
    train_data.extend(data[:-5000,:])
    val_data.extend(data[-5000:])
    train_label_var_mouth.extend(label[:-5000, -28:-1])
    val_label_var_mouth.extend(label[-5000:, -28:-1])
    train_label_var_other.extend(label[:-5000, -52:-28])
    val_label_var_other.extend(label[-5000:, -52:-28])

    print(np.array(train_data).shape)
    print(np.array(val_data).shape)
    print(np.array(train_label_var_mouth).shape)
    print(np.array(val_label_var_mouth).shape)
    print(np.array(train_label_var_other).shape)
    print(np.array(val_label_var_other).shape)

    # save data and label to npy
    np.save(os.path.join(dataSet_dir, 'train_data.npy'), np.array(train_data))
    np.save(os.path.join(dataSet_dir, 'val_data.npy'), np.array(val_data))
    np.save(os.path.join(dataSet_dir, 'train_label_var_mouth.npy'), np.array(train_label_var_mouth))
    np.save(os.path.join(dataSet_dir, 'val_label_var_mouth.npy'), np.array(val_label_var_mouth))
    np.save(os.path.join(dataSet_dir, 'train_label_var_other.npy'), np.array(train_label_var_other))
    np.save(os.path.join(dataSet_dir, 'val_label_var_other.npy'), np.array(val_label_var_other))

if __name__ == '__main__':
    project_dir = r'./'
    propare_data(project_dir)