import os
import sys
import copy
import ipdb
import argparse
import numpy as np
from tqdm import tqdm

sys.path.append(os.getcwd())

import torch as th
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader

from tools.evaluate import *
from tools.earlystopping import EarlyStopping
from Process.process import *
from Process.rand5fold import *

class TDrumorGCN(th.nn.Module):
	def __init__(self,in_feats,hid_feats,out_feats):
		super(TDrumorGCN, self).__init__()
		self.conv1 = GCNConv(in_feats, hid_feats)
		self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		x1=copy.copy(x.float())
		x = self.conv1(x, edge_index)
		x2=copy.copy(x)
		rootindex = data.rootindex
		root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
		batch_size = int(max(data.batch) + 1)
		for num_batch in range(batch_size):
			index = (th.eq(data.batch, num_batch))
			root_extend[index] = x1[rootindex[num_batch]]
		x = th.cat((x,root_extend), 1)

		x = F.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.conv2(x, edge_index)
		x = F.relu(x)
		root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
		for num_batch in range(batch_size):
			index = (th.eq(data.batch, num_batch))
			root_extend[index] = x2[rootindex[num_batch]]
		x = th.cat((x,root_extend), 1)
		x= scatter_mean(x, data.batch, dim=0)

		return x

class BUrumorGCN(th.nn.Module):
	def __init__(self,in_feats,hid_feats,out_feats):
		super(BUrumorGCN, self).__init__()
		self.conv1 = GCNConv(in_feats, hid_feats)
		self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

	def forward(self, data):
		x, edge_index = data.x, data.BU_edge_index
		x1 = copy.copy(x.float())
		x = self.conv1(x, edge_index)
		x2 = copy.copy(x)

		rootindex = data.rootindex
		root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
		batch_size = max(data.batch) + 1
		for num_batch in range(batch_size):
			index = (th.eq(data.batch, num_batch))
			root_extend[index] = x1[rootindex[num_batch]]
		x = th.cat((x,root_extend), 1)

		x = F.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.conv2(x, edge_index)
		x = F.relu(x)
		root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
		for num_batch in range(batch_size):
			index = (th.eq(data.batch, num_batch))
			root_extend[index] = x2[rootindex[num_batch]]
		x = th.cat((x,root_extend), 1)

		x= scatter_mean(x, data.batch, dim=0)
		return x

class Net(th.nn.Module):
	def __init__(self, args, in_feats, hid_feats, out_feats):
		super(Net, self).__init__()
		self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
		self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
		self.fc = th.nn.Linear((out_feats + hid_feats) * 2, args.n_classes)

	def forward(self, data):
		TD_x = self.TDrumorGCN(data)
		BU_x = self.BUrumorGCN(data)
		x = th.cat((BU_x,TD_x), 1)
		x=self.fc(x)
		x = F.log_softmax(x, dim=1)
		return x

def train_GCN(args, treeDic, x_test, x_train, TDdroprate, BUdroprate, lr, weight_decay, patience, n_epochs, batchsize, dataname, iter):
	model = Net(args, 5000, 64, 64).to(device)
	BU_params=list(map(id,model.BUrumorGCN.conv1.parameters()))
	BU_params += list(map(id, model.BUrumorGCN.conv2.parameters()))
	base_params=filter(lambda p:id(p) not in BU_params,model.parameters())
	optimizer = th.optim.Adam([
		{'params':base_params},
		{'params':model.BUrumorGCN.conv1.parameters(),'lr':lr/5},
		{'params': model.BUrumorGCN.conv2.parameters(), 'lr': lr/5}
	], lr=lr, weight_decay=weight_decay)
	
	print("Start training...")
	model.train()
	train_losses = []
	val_losses = []
	train_accs = []
	val_accs = []
	early_stopping = EarlyStopping(args, patience=patience, verbose=True)
	for epoch in range(n_epochs):
		traindata_list, testdata_list = loadBiData(args, dataname, treeDic, x_train, x_test, TDdroprate, BUdroprate)
		train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=5)
		test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=5)
		avg_loss = []
		avg_acc = []
		batch_idx = 0
		tqdm_train_loader = tqdm(train_loader)
		for Batch_data in tqdm_train_loader:
			Batch_data.batch = Batch_data.batch.long()
			Batch_data.to(device)
			out_labels= model(Batch_data)
			finalloss=F.nll_loss(out_labels,Batch_data.y)
			loss=finalloss
			optimizer.zero_grad()
			loss.backward()
			avg_loss.append(loss.item())
			optimizer.step()
			_, pred = out_labels.max(dim=-1)
			correct = pred.eq(Batch_data.y).sum().item()
			train_acc = correct / len(Batch_data.y)
			avg_acc.append(train_acc)
			print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,epoch, batch_idx,
																								 loss.item(),
																								 train_acc))
			batch_idx = batch_idx + 1

		train_losses.append(np.mean(avg_loss))
		train_accs.append(np.mean(avg_acc))

		temp_val_losses = []
		temp_val_accs = []
		temp_val_Acc_all, \
		temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
		temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
		temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
		temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
		model.eval()
		tqdm_test_loader = tqdm(test_loader)
		for Batch_data in tqdm_test_loader:
			Batch_data.batch = Batch_data.batch.long()
			Batch_data.to(device)
			val_out = model(Batch_data)
			val_loss  = F.nll_loss(val_out, Batch_data.y)
			temp_val_losses.append(val_loss.item())
			_, val_pred = val_out.max(dim=1)
			correct = val_pred.eq(Batch_data.y).sum().item()
			val_acc = correct / len(Batch_data.y)

			if args.n_classes == 3:
				Acc_all, \
				Acc1, Prec1, Recll1, F1, \
				Acc2, Prec2, Recll2, F2, \
				Acc3, Prec3, Recll3, F3 = evaluation3class(val_pred, Batch_data.y)
				Acc4, Prec4, Recll4, F4 = 0., 0., 0., 0.
			elif args.n_classes == 4:
				Acc_all, \
				Acc1, Prec1, Recll1, F1, \
				Acc2, Prec2, Recll2, F2, \
				Acc3, Prec3, Recll3, F3, \
				Acc4, Prec4, Recll4, F4 = evaluation4class(val_pred, Batch_data.y)
			else:
				raise NotImplementedError("--n_classes == {} is not implemented!".format(args.n_classes))
			
			temp_val_Acc_all.append(Acc_all), \
			temp_val_Acc1.append(Acc1), temp_val_Prec1.append(Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
			temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(Recll2), temp_val_F2.append(F2), \
			temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(Recll3), temp_val_F3.append(F3)
			temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(Recll4), temp_val_F4.append(F4)
			temp_val_accs.append(val_acc)

		val_losses.append(np.mean(temp_val_losses))
		val_accs.append(np.mean(temp_val_accs))
		print("Epoch {:05d} | Val_Loss {:.4f} | Val_Accuracy {:.4f} | Val_F1-Macro {:.4f}".format(
			epoch, np.mean(temp_val_losses), np.mean(temp_val_accs), 
			(np.mean(temp_val_F1) + np.mean(temp_val_F2) + np.mean(temp_val_F3) + np.mean(temp_val_F4)) / args.n_classes))

		res = [
			'acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
			'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1), np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
			'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2), np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
			'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3), np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
			'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4), np.mean(temp_val_Recll4), np.mean(temp_val_F4))
		]
		print('results:', res)
		early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
					   np.mean(temp_val_F3), np.mean(temp_val_F4), model, 'BiGCN', dataname)
		accs = np.mean(temp_val_accs)
		F1 = np.mean(temp_val_F1)
		F2 = np.mean(temp_val_F2)
		F3 = np.mean(temp_val_F3)
		F4 = np.mean(temp_val_F4)
		if early_stopping.early_stop:
			print("Early stopping")
			accs = early_stopping.accs
			F1 = early_stopping.F1
			F2 = early_stopping.F2
			F3 = early_stopping.F3
			F4 = early_stopping.F4
			break
			
	return train_losses, val_losses, train_accs, val_accs, accs, F1, F2, F3, F4

def parse_args():
	parser = argparse.ArgumentParser(description="Rumor Detection")

	## Settings
	parser.add_argument("--flatten", action="store_true")

	## Others
	parser.add_argument("--lr", type=float, default=0.0005)
	parser.add_argument("--weight_decay", type=float, default=1e-4)
	parser.add_argument("--patience", type=int, default=10)
	parser.add_argument("--n_epochs", type=int, default=200)
	parser.add_argument("--batch_size", type=int, default=128)
	parser.add_argument("--iterations", type=int, default=10)
	parser.add_argument("--TDdroprate", type=float, default=0.2)
	parser.add_argument("--BUdroprate", type=float, default=0.2)
	parser.add_argument("--model", type=str, default="GCN")

	parser.add_argument("--dataset_name", type=str, default="twitter16") ##, choices=["PHEME", "semeval2019", "twitter15", "twitter16"])
	parser.add_argument("--data_root", type=str, default="/mnt/hdd1/projects/BiGCN/dataset/processedV2") ## "/mnt/hdd1/projects/BiGCN/dataset/ori"
	parser.add_argument("--output_root", type=str, default="/mnt/hdd1/projects/BiGCN")
	parser.add_argument("--n_classes", type=int, default=4, help="PHEME/twitter15/twitter16: 4, semeval2019: 3")
	parser.add_argument("--n_fold", type=int, default=5, help="twitter15/twitter16/semeval2019: 5, PHEME: 9")

	args = parser.parse_args()

	return args

if __name__ == "__main__":
	args = parse_args()
	device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

	treeDic = loadTree(args, args.dataset_name)

	total_accs, total_NR_F1, total_FR_F1, total_TR_F1, total_UR_F1 = [], [], [], [], []
	for iter in range(args.iterations):

		## Train 5-fold
		accs, NR_F1, FR_F1, TR_F1, UR_F1 = [], [], [], [], []
		for fold_idx in range(args.n_fold):
			print("\n{}: Fold [{}]".format(args.dataset_name, fold_idx))

			## Load fixed 5-fold
			fold_test, fold_train = loadfoldlist(args, args.dataset_name, fold_idx)

			train_losses, val_losses, train_accs, val_accs0, acc, F1, F2, F3, F4 = \
			train_GCN(
				args, treeDic, fold_test, fold_train, 
				args.TDdroprate, args.BUdroprate, args.lr, args.weight_decay, 
				args.patience, args.n_epochs, args.batch_size, args.dataset_name, iter
			)

			## Record result of each fold
			accs.append(acc)
			FR_F1.append(F1)
			TR_F1.append(F2)
			UR_F1.append(F3)
			NR_F1.append(F4)

		## Record average result of 5-fold for current iteration
		total_accs.append(np.mean(accs))
		total_FR_F1.append(np.mean(FR_F1))
		total_TR_F1.append(np.mean(TR_F1))
		total_UR_F1.append(np.mean(UR_F1))
		total_NR_F1.append(np.mean(NR_F1))

	print("Total_Test_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
		np.mean(total_accs), 
		np.mean(total_NR_F1), 
		np.mean(total_FR_F1), 
		np.mean(total_TR_F1), 
		np.mean(total_UR_F1))
	)

	###########################
	## NEW: Log Final Result ##
	###########################
	if not os.path.isfile("{}/result.tsv".format(args.output_root)):
		with open("{}/result.tsv".format(args.output_root), "w") as fw:
			fw.write("{:15s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\n".format(
				"Dataset", "Accuracy", "F1-Macro", "F1-NR", "F1-FR", "F1-TR", "F1-UR"))

	with open("{}/result.tsv".format(args.output_root), "a") as fw:
		fw.write("{:15s}\t{:<10.4f}\t{:<10.4f}\t{:<10.4f}\t{:<10.4f}\t{:<10.4f}\t{:<10.4f}\n".format(
				args.dataset_name, sum(total_accs) / args.iterations, 
				(sum(total_NR_F1) + sum(total_FR_F1) + sum(total_TR_F1) + sum(total_UR_F1)) / (args.n_classes * args.iterations), 
				sum(total_NR_F1) / args.iterations, 
				sum(total_FR_F1) / args.iterations, 
				sum(total_TR_F1) / args.iterations, 
				sum(total_UR_F1) / args.iterations
			)
		)


