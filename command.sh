#!/usr/bin/sh

### 콘다 실행
# conda activate pytorch

### 데이터셋 csv 생성
# dataset="/userHome/userhome4/kyoungmin/dataset/BraTS2021_Training_Data/converted_images"
dataset="/userHome/userhome4/kyoungmin/dataset/data"
# size는 트레이닝 비율 default=0.9(90%)
python data_split_csv.py --dataset ${dataset} --size 0.8


### csv 
# src="/userHome/userhome4/kyoungmin/code/DCSAU-Net/src/"
csv_dir="./src/"
# csvfile=${csv_dir}test_train_data.csv
csvfile=$(ls -t ${csv_dir}test_train_data_*.csv | head -n 1)
echo "학습할 csvfile는 : " ${csvfile}

# 하이퍼 파라미터 설정
loss="dice"
# batch=16
batch=16
lr=0.0001 # 학습률, 높을수록 업데이트 크기가 큼
epoch=200 # 에포크
# export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb=256"

dataset="/userHome/userhome4/kyoungmin/dataset/data/"

# 최신 CSV 파일을 사용해 Python 스크립트 실행
# CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python train.py --dataset ${dataset} --csvfile ${csvfile} --loss ${loss} --batch ${batch}
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset ${dataset} --csvfile ${csvfile} --loss ${loss} --batch ${batch} --lr ${lr} --epoch ${epoch}
# CUDA_LAUNCH_BLOCKING=1 python train.py --dataset ${dataset} --csvfile ${csvfile} --loss ${loss} --batch ${batch}
# python train.py --dataset ${dataset} --csvfile ${csvfile} --loss ${loss} --batch ${batch}

#     parser.add_argument('--dataset', default='data/',type=str, help='the path of dataset')
#     parser.add_argument('--csvfile', default='src/test_train_data.csv',type=str, help='two columns [image_id,category(train/test)]')
#     parser.add_argument('--model',default='save_models/epoch_last.pth', type=str, help='the path of model')
#     parser.add_argument('--debug',default=True, type=bool, help='plot mask')

# python eval_multiple.py --dataset ${dataset} --csvfile ${csvfile}
python eval_multiple.py --dataset ../../dataset/data --csvfile ${csvfile} --model ./save_models/epoch_last.pth --debug True
# python train.py --dataset ./dataset/data --csvfile /userHome/userhome4/kyoungmin/code/DCSAU-Net/src/test_train_data_241213_203902.csv --loss dice --batch 16 --lr 0.0001 --epoch 200
# python train.py --dataset ./dataset/data --csvfile /userHome/userhome4/kyoungmin/code/DCSAU-Net/src/test_train_data_241213_203902.csv --loss dice --batch 16 --lr 0.001 --epoch 200