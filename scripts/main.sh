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

if [ $(hostname) = "josh-System-Product-Name" ]; then
	data_root="/mnt/hdd1/projects/BiGCN/dataset/processedV2"
	output_root="/mnt/hdd1/projects/BiGCN"
elif [ $(hostname) = "yisyuan-PC2" ]; then
	export CUDA_VISIBLE_DEVICES=0
	data_root="/home/joshchang/project/BiGCN/data/processedV2"
	output_root="/home/joshchang/project/BiGCN/results"
fi
#else
#	#export CUDA_VISIBLE_DEVICES=0
#	output_root="/nfs/home/joshchang/projects/RumorV2/results"
#fi

#########################
## Train on my dataset ##
#########################
python ./model/BiGCN.py --dataset_name twitter15 --data_root "$data_root" --output_root "$output_root"
python ./model/BiGCN.py --dataset_name twitter16 --data_root "$data_root" --output_root "$output_root"
python ./model/BiGCN.py --dataset_name semeval2019 --data_root "$data_root" --output_root "$output_root"