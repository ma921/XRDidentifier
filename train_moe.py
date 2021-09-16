import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from models.net1d import Net1D, MyDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import pandas as pd
import sys
from utils import *
from utils_moe import *

if __name__ == "__main__": 
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch', type=int, default=64)
	parser.add_argument('--epoch', type=int, default=100)
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--data_path', type=str, default='./pickles')
	parser.add_argument('--split_ratio', type=float, default=0.7)
	parser.add_argument('--save_model', type=str, default='moe_nll.pt')
	args = parser.parse_args()

	# arguments
	input_size = 6000
	l_experts = os.listdir('./pretrainedCPU')
	data_path = args.data_path
	batch_size = args.batch
	device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# loading MoE model
	print('loading models...')
	model = MoE_HDM(input_size, l_experts, noisy_gating=True, k=len(l_experts))
	model.to(device)
	
	# setup components
	dataloaders = setup_dataloaders(data_path, batch_size, r_split=0.7, n_train=50000)
	optimizer = optim.Adam(model.parameters(), lr=1e-3)
	loss_func = nn.NLLLoss()
	loss_func = loss_func.to(device)

	results = np.zeros([args.epoch, 5])
	best_acc = 0.0
	for epoch in range(args.epoch):
		# training phase
		# Initialising the indicators and models
		running_loss = 0.0
		running_corrects = 0.0
		# model setting
		model.train()
		for i in range(model.num_experts):
			model.experts[i].eval()
		model.zero_grad()

		for batch_idx, batch in enumerate(dataloaders[0]):
			sys.stdout.flush() # for writing results to a log file

			# prediction
			input, label = tuple(t.to(device) for t in batch)
			pred, aux_loss = model(input)
			loss = loss_func(pred, label)
			total_loss = loss + aux_loss

			# proceeding the training conditions 
			optimizer.zero_grad()
			total_loss.backward()
			optimizer.step()

			# calculating the metrics
			pred_labels = pred.max(1)[1]
			running_corrects += torch.sum(pred_labels == label).item()
			running_loss += loss.item()

			# report for each batch
			print('[Train] epoch: '+str(epoch)+' | batch: '+str(batch_idx)+' | loss: '+str(loss.item()))

		# report for training phase
		n_train = round(n_samples * args.split_ratio)
		epoch_loss = running_loss / n_train
		epoch_acc = running_corrects / n_train
		results[epoch, :3] = np.array([epoch, epoch_loss, epoch_acc])
		print('[Train total] epoch: '+str(epoch)+' | loss: '+str(epoch_loss)+' | acc:'+str(epoch_acc))

		# validation phase
		# Initialising the indicators and models
		running_loss = 0.0
		running_corrects = 0.0
		model.eval()
		model.zero_grad()

		with torch.no_grad():
			for batch_idx, batch in enumerate(dataloader_val):
				sys.stdout.flush() # for writing results to a log file
                
				# prediction
				input, label = tuple(t.to(device) for t in batch)
				pred, aux_loss = model(input)
				loss = loss_func(pred, label)
				total_loss = loss + aux_loss

				# calculating the metrics
				pred_labels = pred.max(1)[1]
				running_corrects += torch.sum(pred_labels == label).item()
				running_loss += loss.item()

				# report for each batch
				print('[Val] epoch: '+str(epoch)+' | batch: '+str(batch_idx)+' | loss: '+str(loss.item()))

			# report for test phase
			n_test = round(n_samples * (1 - args.split_ratio)/2)
			epoch_loss = running_loss / n_test
			epoch_acc = running_corrects / n_test
			results[epoch, 3:] = np.array([epoch_loss, epoch_acc])
			print('[Val total] epoch: '+str(epoch)+' | loss: '+str(epoch_loss)+' | acc:'+str(epoch_acc))

		if best_acc <= epoch_acc:
			best_acc = epoch_acc
			print('///////////////////////////////////////////////////////////////////////')
			print('-------->   Best model has been replaced. epoch: '+str(epoch)+' best_acc: '+str(epoch_acc))
			print('///////////////////////////////////////////////////////////////////////')
			model_path = './models/'+args.save_model.split('.')[0]+'_epoch'+str(epoch)+'.pt'
			torch.save(model.state_dict(), model_path)
			print('best model has been saved in '+model_path)

			# Recording the learning curve
			df = pd.DataFrame(results, columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
			df.to_csv(args.save_model.split('.')[0]+'.csv')
			print('The learning curve has been recorded.')

			# Test
			print('test will start')
			model.eval()
			model.zero_grad()

			# Initialising the indicators
			running_loss = 0.0
			running_corrects = 0.0
			with torch.no_grad():
				for batch_idx, batch in enumerate(dataloader_test):
					sys.stdout.flush() # for writing results to a log file

					# prediction
					input, label = tuple(t.to(device) for t in batch)
					pred, aux_loss = model(input)
					loss = loss_func(pred, label)
					total_loss = loss + aux_loss

					# calculating the metrics
					pred_labels = pred.max(1)[1]
					running_corrects += torch.sum(pred_labels == label).item()
					running_loss += loss.item()

					# report for each batch
					print('[Test] epoch: '+str(epoch)+' | batch: '+str(batch_idx)+' | loss: '+str(loss.item()))

			# report for test phase
			n_test = round(n_samples * (1 - args.split_ratio)/2)
			epoch_loss = running_loss / n_test
			epoch_acc = running_corrects / n_test
			results[epoch, 3:] = np.array([epoch_loss, epoch_acc])
			print('[Test total] epoch: '+str(epoch)+' | loss: '+str(epoch_loss)+' | acc:'+str(epoch_acc))
    
		# Recording the learning curve
		df = pd.DataFrame(results, columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
		df.to_csv(args.save_model.split('.')[0]+'.csv')

	print('ALL TRAINING PROCESSES HAVE BEEN COMPLETED.')
	print('Test phase will start')
