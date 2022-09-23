## Generate graph data and store in /data/Weibograph
#python ./Process/getWeibograph.py

## Generate graph data and store in /data/Twitter15graph
#python ./Process/getTwittergraph.py Twitter15

## Generate graph data and store in /data/Twitter16graph
#python ./Process/getTwittergraph.py Twitter16

## Flatten tree data
#python ./Process/preprocess.py --flatten --dataset Twitter15
#python ./Process/preprocess.py --flatten --dataset Twitter16

#python ./Process/getTwittergraph.py --dataset_name Twitter16
#python ./Process/getTwittergraph.py --dataset_name Twitter15
#python ./Process/getTwittergraph.py --dataset_name Twitter16 --flatten
#python ./Process/getTwittergraph.py --dataset_name Twitter15 --flatten

##########################################################
## Create datasets for BiGCN from my datasets (RumorV2) ##
##########################################################
#python ./Process/preprocess.py --create_from_my_dataset --dataset semeval2019
#python ./Process/preprocess.py --create_from_my_dataset --dataset twitter15
#python ./Process/preprocess.py --create_from_my_dataset --dataset twitter16

python ./Process/getTwittergraph.py --dataset_name semeval2019
python ./Process/getTwittergraph.py --dataset_name twitter15
python ./Process/getTwittergraph.py --dataset_name twitter16