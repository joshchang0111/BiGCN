import os
import csv
import ipdb
import argparse
import numpy as np
import pandas as pd
import preprocessor as pre

from tqdm import tqdm
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

tqdm.pandas()

def parse_args():
	parser = argparse.ArgumentParser(description="Rumor Detection")

	## What to do
	parser.add_argument("--flatten", action="store_true")
	parser.add_argument("--create_from_my_dataset", action="store_true")

	## Others
	parser.add_argument("--dataset", type=str, default="Twitter16", choices=["Twitter15", "Twitter16", "PHEME", "semeval2019", "twitter15", "twitter16"])
	parser.add_argument("--data_root", type=str, default="/mnt/hdd1/projects/BiGCN/dataset")

	args = parser.parse_args()

	return args

def flatten(args):
	print("Create flatten tree data `data.TD_RvNN.vol_5000.flatten.txt` of {}".format(args.dataset))

	path_in  = "{}/{}/data.TD_RvNN.vol_5000.txt".format(args.data_root, args.dataset)
	path_out = "{}/{}/data.TD_RvNN.vol_5000.flatten.txt".format(args.data_root, args.dataset)

	## Format: [source_id, parent_idx, self_idx, num_parent, max_seq_len, text]

	out_lines = []
	with open(path_in, "r") as f:
		for line in f.readlines():
			line = line.strip().rstrip()
			cols = line.split("\t")
			#if cols[1] != "None": ## parent_idx is not None (not source tweet)
			#	cols[1] = "1"
			cols[1] = "None"
			cols[3] = "0" ## num_parent should be 1
			out_lines.append("\t".join(cols))

	with open(path_out, "w") as f:
		for line in out_lines:
			f.write("{}\n".format(line))

def create_from_my_dataset(args):
	print("Create datasets for BiGCN from my datasets (RumorV2)")
	print("Dataset: {}".format(args.dataset))

	def clean_text(line):
		## Remove @, reduce length, handle strip
		tokenizer = TweetTokenizer(strip_handles=False, reduce_len=True)
		line = " ".join(tokenizer.tokenize(line))
		
		## Remove url, emoji, mention, prserved words, only preserve smiley
		#pre.set_options(pre.OPT.URL, pre.OPT.EMOJI, pre.OPT.MENTION, pre.OPT.RESERVED)
		pre.set_options(pre.OPT.URL, pre.OPT.RESERVED)
		line = pre.tokenize(line)
		
		## Remove non-sacii 
		line = "".join([i if ord(i) else "" for i in line]) ## remove non-sacii
		return line

	def text_to_idx_count(text, table):
		keys = table.keys()
		counts = {}
		for i in text.split():
			if i in keys:
				idx = table[i]
				if idx not in counts.keys():
					counts[idx] = 1
				else:
					counts[idx] += 1
		line = []
		for item in counts.items():
			line.append("{}:{}".format(item[0], item[1]))
		return " ".join(line)

	print("\n1). Create data.TD_RvNN.vol_5000.txt")
	path_in  = "../RumorV2/dataset/processedV2/{}/data.csv".format(args.dataset)
	path_out = "{}/processedV2/{}/data.TD_RvNN.vol_5000.txt".format(args.data_root, args.dataset)

	## Target Format: [source_id, parent_idx, self_idx, num_parent, max_seq_len, text]
	data_df = pd.read_csv(path_in)
	data_df["text"] = data_df["text"].progress_apply(lambda r: "{} <end>".format(clean_text(r).rstrip()))

	###############################
	## Convert text to idx:count ##
	###############################
	## Build table by idf
	corpus = data_df["text"].to_list()
	vectorizer = TfidfVectorizer(token_pattern=r'\S+')
	X = vectorizer.fit_transform(corpus)
	indices = np.argsort(vectorizer.idf_) # sort from large to small by IDF
	feature_names = vectorizer.get_feature_names_out()
	top_n = 5000 # Find the top n words by IDF
	top_features = [feature_names[i] for i in indices[:top_n]]
	table = {}
	idx = 0
	for feature in top_features:
		table[feature] = idx
		idx += 1

	## Iteratively write each row
	fw = open(path_out, "w")
	for idx, row in data_df.iterrows():
		write_line = "{}\t{}\t{}\t{}\t{}\t{}\n".format(
			row["source_id"], 
			row["parent_idx"], 
			row["self_idx"], 
			row["num_parent"], 
			row["max_seq_len"], 
			text_to_idx_count(row["text"], table)
		)
		fw.write(write_line)
	fw.close()

	print("\n2). Create 5-fold")

	def output_fold(train_or_test, fold):
		path_in  = "../RumorV2/dataset/processedV2/{}/split_{}".format(args.dataset, fold)
		path_out = "{}/processedV2/{}/split_{}".format(args.data_root, args.dataset, fold)
		os.makedirs(path_out, exist_ok=True)

		fw = open("{}/{}.txt".format(path_out, train_or_test), "w")
		label_df = pd.read_csv("{}/{}.csv".format(path_in, train_or_test))
		for idx, row in label_df.iterrows():
			fw.write("{}\t{}\n".format(row["source_id"], row["label_veracity"]))
		fw.close()

	folds = range(9) if args.dataset == "PHEME" else range(5)
	for fold in tqdm(folds):
		output_fold(train_or_test="train", fold=fold)
		output_fold(train_or_test="test", fold=fold)

if __name__ == "__main__":
	args = parse_args()

	if args.flatten:
		flatten(args)
	elif args.create_from_my_dataset:
		create_from_my_dataset(args)