import numpy as np
import os
from tqdm import tqdm

def prepare_data(project_dir):
    data_dir = os.listdir('/home/ubuntu/data')
    total_len = 0
    for idx in tqdm(range(15)):
        """ Concat the data together and split them to training and testing sets
        Args:
            project_dir: str, project dir contains lpc and bs_value
        """
        dataSet_dir = './dataSet_mead'
        if not os.path.exists(dataSet_dir):
            os.mkdir(dataSet_dir)

        train_data = []
        val_data = []
        train_label_var_mouth = []
        val_label_var_mouth = []

        data = np.zeros((400000, 32, 64, 1))
        label = np.zeros((400000, 42))
        data_len = 0
        error  = []
        for id in data_dir[idx * 3: idx * 3 + 3]:
            for emo in os.listdir('/home/ubuntu/data/{}/blendshape/front/'.format(id)):
                for level in os.listdir('/home/ubuntu/data/{}/blendshape/front/{}/'.format(id, emo)):
                    data_name_list = [fn.split('.')[0] for fn in os.listdir('/home/ubuntu/data/{}/video/front/{}/{}'.format(id, emo, level)) if fn.split('.')[1] == 'wav']
                    data_path_list = [os.path.join(project_dir, '/home/ubuntu/data/{}/video/front/{}/{}'.format(id, emo, level), i + '.npy') for i in data_name_list]
                    label_path_list = [os.path.join(project_dir, '/home/ubuntu/data/{}/blendshape/front/{}/{}'.format(id, emo, level), "BS_" + i + '_bs.npy') for i in data_name_list]

                    # generate data and label
                    for i in range(len(data_path_list)):
                        try:
                            data_temp = np.load(data_path_list[i])
                            label_temp = np.load(label_path_list[i])

                            data[data_len:data_len+data_temp.shape[0]] = data_temp
                            label[data_len:data_len+data_temp.shape[0]] = label_temp
                            data_len += data_temp.shape[0]
                            # print(len(data_temp), len(label_temp), len(data_temp) == len(label_temp))
                        except:
                            print("error")
                            error.append(data_path_list)
        print('error', error)

        data = data[0:data_len]
        label = label[0:data_len]
        total_len += data_len
        print('data', data.shape)
        print('label', label.shape)

        # split data in to train and val
        train_data.extend(data[:])
        # val_data.extend(data[:])
        train_label_var_mouth.extend(label[:])
        # val_label_var_mouth.extend(label[-100000:])

        print(np.array(train_data).shape)
        # print(np.array(val_data).shape)
        print(np.array(train_label_var_mouth).shape)
        # print(np.array(val_label_var_mouth).shape)

        # save data and label to npy
        # np.save(os.path.join(dataSet_dir, 'train_data_{}.npy'.format(idx)), np.array(train_data))
        # np.save(os.path.join(dataSet_dir, 'val_data.npy'), np.array(val_data))
        np.save(os.path.join(dataSet_dir, 'train_label_var_head_{}.npy'.format(idx)), np.array(train_label_var_mouth))
        # np.save(os.path.join(dataSet_dir, 'val_label_var_mouth.npy'), np.array(val_label_var_mouth))
    print(total_len)

if __name__ == '__main__':
    project_dir = r''
    prepare_data(project_dir)