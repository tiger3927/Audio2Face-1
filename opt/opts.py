import argparse

def parse_opt():
    parser = argparse.ArgumentParser()

    #basic
    parser.add_argument('--gpu', type=int, default=0, help='run gpu')
    parser.add_argument('--id', type=str, default='test', help='experiment id')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    # learning setting
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='learning rate decay')
    parser.add_argument('--bs', type=int, default=32, help='batch size')
    parser.add_argument('--epoch', type=int, default=200, help='max epoch')

    # dataset
    parser.add_argument('--dataset', type=str, default='dataSet_mine', help='train dataset')
    parser.add_argument('--output_path', type=str, default='./output_mine', help='output path')
    parser.add_argument('--output_feature', type=str, default='mouth', help='mouth or other')

    # finetune
    parser.add_argument('--finetune', action='store_true', help='whether finetune')
    parser.add_argument('--model_path', type=str, default='', help='model path')
    
    args = parser.parse_args()
    return args

def parse_opt_test():
    parser = argparse.ArgumentParser()

    # Basic
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--location', choice = ['mouth','other'], help='mouse or other location')

    # Path
    parser.add_argument('--model_path', type=str, default='', help='model path')
    parser.add_argument('--input_wav', type=str, default='', help='model path')
    parser.add_argument('--output_json', type=str, default='', help='output json')
    parser.add_argument('--output_size', choice = [27, 24], help='output size, 27 for mouth, 24 for other')
    parser.add_argument('--fps', type=int, default=30, help='fps')

    args = parser.parse_args()
    return args