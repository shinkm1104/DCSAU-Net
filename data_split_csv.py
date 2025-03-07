import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime


def pre_csv(data_path,frac):
    # np.random.seed(114)
    # np.random.seed(109)
    # np.random.seed(10)
    # np.random.seed(18)
    # np.random.seed(119)
    # np.random.seed(11)
    np.random.seed(4)
    folder_ids = [f for f in os.listdir(data_path) 
                  if os.path.isdir(os.path.join(data_path, f))]
    # image_ids = os.listdir(data_path)
    data_size = len(folder_ids)
    train_size = int(round(len(folder_ids) * frac, 0))
    train_set = np.random.choice(folder_ids,train_size,replace=False)

    ds_split = []
    for img_id in folder_ids:
        if img_id in train_set:
            ds_split.append('train')
        else:
            ds_split.append('test')
    
    ds_dict = {'image_id':folder_ids,
               'category':ds_split 
        }
    df = pd.DataFrame(ds_dict)
    current_time = datetime.now().strftime('%y%m%d_%H%M%S')
    # 파일 이름에 날짜와 시간을 추가
    file_name = f'src/test_train_data_{current_time}.csv'
    df.to_csv(file_name, index=False)
    os.chmod(file_name, 0o775)
    print('Number of train sample: {}'.format(len(train_set)))
    print('Number of test sample: {}'.format(data_size-train_size))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--dataset', type=str, default='data/', help='the path of dataset')
    parser.add_argument('--dataset', type=str, default='../datasets/DSB2018/image', help='the path of images') # issue 16
    parser.add_argument('--size', type=float, default=0.9, help='the size of your train set')
    args = parser.parse_args()
    os.makedirs('src/',exist_ok=True)
    pre_csv(args.dataset,args.size)
