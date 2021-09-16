import numpy as np
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import random
import pickle
from torch.utils.data import DataLoader, Dataset
from torch.distributions.normal import Normal
from models.net1d import Net1D, MyDataset
from utils import *
import pickle

class expert_model(nn.Module):
    def __init__(self, expert_raw, n_class_all, label_convert):
        super(expert_model, self).__init__()
        self.expert_raw = expert_raw
        self.n_class_all = n_class_all
        self.label_convert = label_convert
        self.device = 'cuda'

    def forward(self, x):
        out = self.expert_raw(x)
        batch = x.size(0)
        out2 = torch.ones(batch, self.n_class_all).to(self.device) * (out.min() - out.var())
        for idx, label_conv in enumerate(self.label_convert):
            out2[:, label_conv] = out[:, idx]
        return out2

def HDM(model_name):
    # interpret as variables
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    element = model_name.split('_')[1].split('.pt')[0]
    model_path = './pretrainedCPU/'+model_name
    df_label = pd.read_csv('./labels/label_'+element+'.csv', index_col=0)
    n_class = len(np.unique(df_label.iloc[:,0]))

    # load expert model
    expert = Net1D(
        in_channels=1,
        base_filters=64,
        ratio=1.0,
        filter_list=[64,160,160,400,400,1024,1024],
        m_blocks_list=[2,2,2,3,3,4,4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        n_classes=n_class,
        verbose=False)
    expert.dense = AdaCos(1024, n_class)
    expert.load_state_dict(torch.load(model_path))
    
    # adjust prediction for the whole MoE model
    l_labels = list(df_label.iloc[:,0])
    materials = [
        ''.join(
            sorted(
                df_label.index[l_labels.index(label)].split(' ')
            )
        ) for label in range(n_class)
    ]

    df_all = pd.read_csv('./all_labels.csv', index_col=0)
    label_names = list(df_all.index)
    label_convert = [label_names.index(material) for material in materials]
    n_class_all =len(label_names)
    
    # wrap raw expert model into adjusted model
    model = expert_model(expert, n_class_all, label_convert)
    model.to(device)
    model.eval()
    model.zero_grad()

    return model

class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = sorted_experts[index_sorted_experts[:, 1],0]
        # calculate num samples that each expert gets
        self._part_sizes = list((gates > 0).sum(0).cpu().numpy())
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1).unsqueeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)


    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp().to(self.device)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates).to(self.device)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True).to(self.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()


    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class MoE_HDM(nn.Module):
    def __init__(self, input_size, l_experts, noisy_gating=True, k=4):
        super(MoE_HDM, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = len(l_experts)
        self.list_experts = l_experts
        self.input_size = input_size
        self.k = k
        
        # instantiate experts
        self.w_gate = nn.Parameter(torch.zeros(input_size, self.num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, self.num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.normal = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        self.experts = nn.ModuleList(
            [
                HDM(model_name) for model_name in self.list_experts
            ]
        )
        
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.Tensor([0])
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat,0 , threshold_positions_if_out), 1)
        # is each value currently in the top k.
        prob_if_in = self.normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = self.normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon) * train)
            noisy_logits = clean_logits + ( torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, train=True, loss_coef=1e-2):
        gates, load = self.noisy_top_k_gating(x, train)
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, loss

class HDM_preprocessed():
    def __init__(self,
                 model_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.element = model_name.split('_')[1].split('.pt')[0]
        self.n_class = len(pd.read_csv('./all_labels.csv', index_col=0))

    def forward(self, x):
        input = np.zeros([len(x), self.n_class])
        for idx, name in enumerate(list(x)):
            conv_name = './extraction/'+name.split('.pkl')[0]+'_'+self.element+'.pkl'
            with open(conv_name, 'rb') as web:
                pred = pickle.load(web)
            input[idx] = pred
        pred = torch.from_numpy(input).float().to(self.device)
        return pred

class MoE_preprocessed(nn.Module):
    def __init__(self, input_size, l_experts, noisy_gating=True, k=4):
        super(MoE_preprocessed, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = len(l_experts)
        self.list_experts = l_experts
        self.input_size = input_size
        self.k = k
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # instantiate experts
        self.w_gate = nn.Parameter(torch.zeros(input_size, self.num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, self.num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.normal = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        self.experts = [
            HDM_preprocessed(model_name) for model_name in self.list_experts
        ]

        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.Tensor([0])
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat,0 , threshold_positions_if_out), 1)
        # is each value currently in the top k.
        prob_if_in = self.normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = self.normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        input = np.zeros([len(x),6000])
        for idx, name in enumerate(list(x)):
            with open('./pickles/'+name, 'rb') as web:
                pred = pickle.load(web)
            input[idx] = pred
        x = torch.from_numpy(input).float().to(self.device)

        clean_logits = x @ self.w_gate
        if self.noisy_gating:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon) * train)
            noisy_logits = clean_logits + ( torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, train=True, loss_coef=1e-2):
        gates, load = self.noisy_top_k_gating(x, train)
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        #expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i].forward(x) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, loss

def split_datalist(data_path, r_split=0.7):
    l_all = os.listdir(data_path)
    n_test = int(len(l_all)*(1-r_split)/2)
    l_val = random.choices(l_all, k=n_test)
    l_test_tmp = list(set(l_all) ^ set(l_val))
    l_test = random.choices(l_test_tmp, k=n_test)
    l_train = list(set(l_test_tmp) ^ set(l_test))
    
    return l_train, l_val, l_test

class MoEDataset(Dataset):
    def __init__(self, data_list, train=True, n_train=10):
        self.data_list = data_list
        self.train = train
        self.n_train = n_train

    def __getitem__(self, index):
        if self.train:
            index = random.randint(0, len(self.data_list))
        pickle_path = './pickles/'+self.data_list[index]
        with open(pickle_path, 'rb') as web:
            data = pickle.load(web)
        data = torch.from_numpy(data)
        label = torch.tensor(int(self.data_list[index].split('_')[0]), dtype=torch.long)
        return (data, label)

    def __len__(self):
        length = len(self.data_list)
        if self.train:
            length = self.n_train
        return length

class MoEDataset_extract(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        name = self.data_list[index]
        pickle_path = './pickles/'+name
        with open(pickle_path, 'rb') as web:
            data = pickle.load(web)
        data = torch.from_numpy(data)
        label = torch.tensor(int(name.split('_')[0]), dtype=torch.long)
        return (data, label, name)

    def __len__(self):
        return len(self.data_list)

def setup_dataloaders(data_path, batch_size, r_split=0.7, n_train=100):
    l_train, l_val, l_test = split_datalist(data_path, r_split=r_split)
    dataset_train = MoEDataset(l_train, train=True, n_train=n_train)
    dataset_val = MoEDataset(l_val, train=False)
    dataset_test = MoEDataset(l_test, train=False)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    return (dataloader_train, dataloader_val, dataloader_test)

class MoEDataset_preprocessed(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        name = self.data_list[index]
        label = torch.tensor(int(name.split('_')[0]), dtype=torch.long)
        return (name, label)

    def __len__(self):
        return len(self.data_list)

def split_datalist_preprocessed(data_csv_path, r_split=0.7):
    l_conv = np.array(pd.read_csv(data_csv_path, index_col=0)).squeeze().tolist()
    
    # split train
    label_all = [pkl.split('_')[0] for pkl in l_conv]
    label_unique = np.unique(label_all)
    l_train = [
        l_conv[label_all.index(label)] for label in label_unique
    ]
    l_rest = list(set(l_conv) ^ set(l_train))
    n_train_rest = int(2 * (r_split - 0.5) * len(l_rest))
    l_train_rest = random.choices(l_rest, k=n_train_rest)
    l_train = l_train + l_train_rest

    # split val/test
    l_rest = list(set(l_rest) ^ set(l_train_rest))
    n_val = int(0.5 * len(l_rest))
    l_val = random.choices(l_rest, k=n_val)
    l_test = list(set(l_rest) ^ set(l_val))

    return l_train, l_val, l_test

def setup_dataloaders_preprocessed(data_csv_path, batch_size, r_split=0.7):
    l_train, l_val, l_test = split_datalist_preprocessed(data_csv_path, r_split=r_split)
    dataset_train = MoEDataset_preprocessed(l_train)
    dataset_val = MoEDataset_preprocessed(l_val)
    dataset_test = MoEDataset_preprocessed(l_test)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    return (dataloader_train, dataloader_val, dataloader_test)
