import numpy as np
import os

def txt2npy(fp):
    """Convert weight txt to numpy file
    Args:
        fp: str, bs weight txt file path
    """
    # Open with special format
    file = open(fp, 'r', encoding='ISO-8859-1')
    output_data = []
    num = 0
    for line in file:
        # Drop known words
        line = line.strip()
        items = line.split(',')
        num += 1
        # Start form line 23
        if num > 22:
            output_data.append([float(a) for a in items[1:]])
    return np.array(output_data)

def txts2npys(data_dir, output_dir):
    """Convert weight txt to numpy file in a dictionary
    Args:
        data_dir: str, data dir with bs weight txt file
        output_dir: str, ouput dir
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # List all txt file
    file_names = os.listdir(data_dir)
    file_paths = [fn for fn in file_names if fn.split('.')[1] == 'txt']

    for fn in file_paths:
        fp = os.path.join(data_dir, fn)

        output_data = txt2npy(fp)

        np.save(os.path.join(output_dir, 'BS_{}.npy'.format(fn.split('.')[0])), output_data)
        print(output_data.shape, 'npy array saved to ', os.path.join(output_dir, 'BS_{}.npy'.format(fn.split('.')[0])))

if __name__ == '__main__':
    data_dir = r'./Participant Data'
    output_dir = r'./bs_value'
    print('read data from', data_dir)
    txts2npys(data_dir, output_dir)