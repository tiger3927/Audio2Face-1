import numpy as np
import os
import json

def bs2npy(fp):
    """Convert weight txt to numpy file
    Args:
        fp: str, bs weight txt file path
    """
    # Open with special format
    file = json.load(open(fp, 'r'))
    output_data = [frame[0]["blendshapes"]for frame in file]
    return np.array(output_data)

def bss2npys(data_dir, output_dir):
    """Convert weight txt to numpy file in a dictionary
    Args:
        data_dir: str, data dir with bs weight txt file
        output_dir: str, ouput dir
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # List all txt file
    file_names = os.listdir(data_dir)
    file_paths = [fn for fn in file_names if fn.split('.')[1] == 'json']
    # for fn in file_names:
    #     if fn.split('.')[1] == 'npy':
    #         os.remove(os.path.join(data_dir, fn))

    for fn in file_paths:
        fp = os.path.join(data_dir, fn)

        output_data = bs2npy(fp)

        np.save(os.path.join(output_dir, 'BS_{}.npy'.format(fn.split('.')[0])), output_data)
        print(output_data.shape, 'npy array saved to ', os.path.join(output_dir, 'BS_{}.npy'.format(fn.split('.')[0])))

if __name__ == '__main__':
    data_dir = "/home/ubuntu/data"
    for id in os.listdir(data_dir):
        for emo in os.listdir('/home/ubuntu/data/{}/blendshape/front/'.format(id)):
            for level in os.listdir('/home/ubuntu/data/{}/blendshape/front/{}/'.format(id, emo)):
                input_dir = '/home/ubuntu/data/{}/blendshape/front/{}/{}'.format(id, emo, level)
                output_dir = '/home/ubuntu/data/{}/blendshape/front/{}/{}'.format(id, emo, level)
                print('read data from', input_dir)
                bss2npys(input_dir, output_dir)