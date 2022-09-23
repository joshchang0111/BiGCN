#! /bin/bash
## Environment setup
#unzip -d ./data/Weibo ./data/Weibo/weibotree.txt.zip
#pip install -U torch==1.4.0 numpy==1.18.1
#pip install -r requirements.txt

## Reproduce the experimental results.
#python ./model/Weibo/BiGCN_Weibo.py 100
#python ./model/Twitter/BiGCN_Twitter.py Twitter15 100
#python ./model/Twitter/BiGCN_Twitter.py Twitter16 100

## Modified script
#python ./model/Twitter/BiGCN_Twitter.py --dataset_name Twitter15
#python ./model/Twitter/BiGCN_Twitter.py --dataset_name Twitter16

#python ./model/Twitter/BiGCN_Twitter.py --dataset_name Twitter15 --flatten
#python ./model/Twitter/BiGCN_Twitter.py --dataset_name Twitter16 --flatten

#########################
## Train on my dataset ##
#########################
python ./model/BiGCN.py --dataset_name twitter15
python ./model/BiGCN.py --dataset_name twitter16
python ./model/BiGCN.py --dataset_name semeval2019