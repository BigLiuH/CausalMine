"""
Unified CRDHNS Run and Export Script
This file integrates all dependencies into a single module for easy execution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import yaml
import logging
import argparse
import scipy.sparse as sp
from collections import defaultdict
from time import strftime, localtime, time
import math
import heapq
from numba import jit
from random import shuffle, choice, sample, randint
from torch.distributions import Beta
import warnings
warnings.filterwarnings('ignore')

# ========================================================================================
# CONFIGURATION AND PARAMETERS
# ========================================================================================

class ModelConf(object):
    """Configuration manager for loading YAML config files"""
    def __init__(self, file):
        self.config = {}
        self.read_configuration(file)

    def __getitem__(self, item):
        if not self.contain(item):
            print(f'Parameter {item} is not found in the configuration file!')
            exit(-1)
        return self.config[item]

    def contain(self, key):
        return key in self.config

    def read_configuration(self, file):
        if not os.path.exists(file):
            print('Config file is not found!')
            raise IOError
        with open(file, 'r') as f:
            try:
                self.config = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(f"Error in configuration file: {exc}")
                raise IOError


# Parse command line arguments
parser = argparse.ArgumentParser(description="Unified CRDHNS")
parser.add_argument('--n_candidates', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
parser.add_argument('--n_hid', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
parser.add_argument('--dims', type=int, default=256, help='DNN dims')
parser.add_argument('--norm', type=bool, default=True, help='Normalize input')
parser.add_argument('--emb_size', type=int, default=256, help='Embedding size')
parser.add_argument('--steps', type=int, default=20, help='Diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='Noise schedule')
parser.add_argument('--noise_scale', type=float, default=0.1, help='Noise scale')
parser.add_argument('--noise_min', type=float, default=0.0001, help='Noise lower bound')
parser.add_argument('--noise_max', type=float, default=0.02, help='Noise upper bound')
parser.add_argument('--sampling_noise', type=bool, default=False, help='Sampling with noise')
parser.add_argument('--sampling_steps', type=int, default=0, help='Sampling steps')
parser.add_argument('--reweight', type=bool, default=True, help='Reweight timesteps')
parser.add_argument('--initial_alpha', type=int, default=1, help='Initial alpha')
parser.add_argument('--initial_beta', type=int, default=9, help='Initial beta')
parser.add_argument('--final_alpha', type=int, default=9, help='Final alpha')
parser.add_argument('--final_beta', type=int, default=1, help='Final beta')
parser.add_argument('--cl_weight', type=float, default=0.1, help='CL loss weight')
parser.add_argument('--min_mix_ratio', type=float, default=0, help='Min mix ratio')
parser.add_argument('--num_steps', type=int, default=20, help='Num steps')
parser.add_argument('--stride', type=int, default=4, help='Stride')
parser.add_argument('--sample_start', type=int, default=10, help='Sample start')
parser.add_argument('--diffusion_loss_weight', type=float, default=0.000001, help='Diffusion loss weight')
parser.add_argument('--temp', type=float, default=2, help='Temperature')
parser.add_argument('--positive_noise', type=float, default=0.01, help='Positive noise')

args = parser.parse_args([])  # No args passed, use defaults

# ========================================================================================
# DATA LOADING
# ========================================================================================

class FileIO(object):
    """File I/O utilities"""
    @staticmethod
    def write_file(dir, file, content, op='w'):
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir + file, op) as f:
            f.writelines(content)

    @staticmethod
    def delete_file(file_path):
        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def load_data_set(file, rec_type):
        if rec_type == 'graph':
            data = []
            with open(file) as f:
                for line in f:
                    items = line.strip().split()
                    user_id = items[0]
                    item_id = items[1]
                    weight = float(items[2]) if len(items) > 2 else 1.0
                    data.append([user_id, item_id, weight])
        elif rec_type == 'sequential':
            data = {}
            with open(file) as f:
                for line in f:
                    items = line.strip().split(':')
                    seq_id = items[0]
                    data[seq_id] = items[1].split()
        return data

    @staticmethod
    def load_user_list(file):
        user_list = []
        with open(file) as f:
            for line in f:
                user_list.append(line.strip().split()[0])
        return user_list

    @staticmethod
    def load_social_data(file):
        social_data = []
        with open(file) as f:
            for line in f:
                items = line.strip().split()
                user1 = items[0]
                user2 = items[1]
                weight = float(items[2]) if len(items) > 2 else 1
                social_data.append([user1, user2, weight])
        return social_data


# ========================================================================================
# GRAPH AND DATA CLASSES
# ========================================================================================

class Graph(object):
    """Graph utilities"""
    def __init__(self):
        pass

    @staticmethod
    def normalize_graph_mat(adj_mat):
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat


class Data(object):
    """Base data class"""
    def __init__(self, conf, training, test):
        self.config = conf
        self.training_data = training
        self.test_data = test


class Interaction(Data, Graph):
    """User-Item interaction data"""
    def __init__(self, conf, training, test):
        Graph.__init__(self)
        Data.__init__(self, conf, training, test)

        self.user = {}
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        self.training_set_u = defaultdict(dict)
        self.training_set_i = defaultdict(dict)
        self.test_set = defaultdict(dict)
        self.test_set_item = set()

        self.__generate_set()
        self.user_num = len(self.training_set_u)
        self.item_num = len(self.training_set_i)
        self.ui_adj = self.__create_sparse_bipartite_adjacency()
        self.norm_adj = self.normalize_graph_mat(self.ui_adj)
        self.interaction_mat = self.__create_sparse_interaction_matrix()

    def __generate_set(self):
        for user, item, rating in self.training_data:
            if user not in self.user:
                user_id = len(self.user)
                self.user[user] = user_id
                self.id2user[user_id] = user
            if item not in self.item:
                item_id = len(self.item)
                self.item[item] = item_id
                self.id2item[item_id] = item
            self.training_set_u[user][item] = rating
            self.training_set_i[item][user] = rating

        for user, item, rating in self.test_data:
            if user in self.user and item in self.item:
                self.test_set[user][item] = rating
                self.test_set_item.add(item)

    def __create_sparse_bipartite_adjacency(self, self_connection=False):
        n_nodes = self.user_num + self.item_num
        user_np = np.array([self.user[pair[0]] for pair in self.training_data])
        item_np = np.array([self.item[pair[1]] for pair in self.training_data]) + self.user_num
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np)), shape=(n_nodes, n_nodes), dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat

    def __create_sparse_interaction_matrix(self):
        row = np.array([self.user[pair[0]] for pair in self.training_data])
        col = np.array([self.item[pair[1]] for pair in self.training_data])
        entries = np.ones(len(row), dtype=np.float32)
        return sp.csr_matrix((entries, (row, col)), shape=(self.user_num, self.item_num), dtype=np.float32)

    def get_user_id(self, u):
        return self.user.get(u)

    def get_item_id(self, i):
        return self.item.get(i)

    def training_size(self):
        return len(self.user), len(self.item), len(self.training_data)

    def test_size(self):
        return len(self.test_set), len(self.test_set_item), len(self.test_data)

    def contain(self, u, i):
        return u in self.user and i in self.training_set_u[u]

    def user_rated(self, u):
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())

    def item_rated(self, i):
        return list(self.training_set_i[i].keys()), list(self.training_set_i[i].values())

    def row(self, u):
        k, v = self.user_rated(self.id2user[u])
        vec = np.zeros(self.item_num, dtype=np.float32)
        for item, rating in zip(k, v):
            vec[self.item[item]] = rating
        return vec

    def col(self, i):
        k, v = self.item_rated(self.id2item[i])
        vec = np.zeros(self.user_num, dtype=np.float32)
        for user, rating in zip(k, v):
            vec[self.user[user]] = rating
        return vec

    def matrix(self):
        m = np.zeros((self.user_num, self.item_num), dtype=np.float32)
        for u, u_id in self.user.items():
            vec = np.zeros(self.item_num, dtype=np.float32)
            k, v = self.user_rated(u)
            for item, rating in zip(k, v):
                vec[self.item[item]] = rating
            m[u_id] = vec
        return m


# ========================================================================================
# LOGGING
# ========================================================================================

class Log(object):
    """Logging utility"""
    def __init__(self, module, filename):
        self.logger = logging.getLogger(module)
        self.logger.setLevel(level=logging.INFO)
        if not os.path.exists('./log/'):
            os.makedirs('./log/')
        handler = logging.FileHandler('./log/' + filename + '.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def add(self, text):
        self.logger.info(text)


# ========================================================================================
# NEURAL NETWORK MODULES
# ========================================================================================

def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class MLP(nn.Module):
    """MLP for diffusion denoising"""
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(MLP, self).__init__()
        self.dropout = None
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1]
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError(f"Unimplemented timestep embedding type {self.time_type}")

        out_dims_temp = self.out_dims
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in self.in_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.out_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x, timesteps):
        device = x.device
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x).to(device)
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)

        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)

        return h


# ========================================================================================
# DIFFUSION PROCESS
# ========================================================================================

def betas_from_linear_variance(steps, variance, max_beta=0.999):
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """Compute KL divergence between two gaussians"""
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None

    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def mean_flat(tensor):
    """Take mean over all non-batch dimensions"""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class DiffusionProcess(nn.Module):
    """Diffusion process for hard negative generation"""
    def __init__(self, noise_schedule, noise_scale, noise_min, noise_max, steps, device, keep_num=10, causal_weight=0.1):
        super(DiffusionProcess, self).__init__()
        self.noise_schedule = noise_schedule
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.device = device
        self.keep_num = keep_num
        self.causal_weight = causal_weight

        self.Lt_record = torch.zeros(steps, keep_num, dtype=torch.float64).to(device)
        self.Lt_count = torch.zeros(steps, dtype=int).to(device)
        self.beta_nums = torch.tensor(self.betas_num(), dtype=torch.float64).to(self.device)
        self.diffusion_setting()

    def betas_num(self):
        st_bound = self.noise_scale * self.noise_min
        e_bound = self.noise_scale * self.noise_max
        if self.noise_schedule == "linear" or self.noise_schedule == "linear-var":
            return np.linspace(st_bound, e_bound, self.steps, dtype=np.float64)
        else:
            return betas_from_linear_variance(self.steps, np.linspace(st_bound, e_bound, self.steps, dtype=np.float64))

    def diffusion_setting(self):
        alphas = 1.0 - self.beta_nums
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(self.device)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]).to(self.device)
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).to(self.device)]).to(self.device)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (self.beta_nums * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
        self.posterior_mean_coef1 = (self.beta_nums * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)).to(self.device)
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        ).to(self.device)

    def causal_regularization(self, emb_t, emb_t_prev, t):
        assert emb_t.shape == emb_t_prev.shape
        logvar_t = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, (emb_t.shape[0], *([1] * (len(emb_t.shape) - 1)))
        ).expand_as(emb_t)
        logvar_t_prev = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t - 1, (emb_t_prev.shape[0], *([1] * (len(emb_t_prev.shape) - 1)))
        ).expand_as(emb_t_prev)
        zero_tensor = torch.zeros_like(emb_t_prev)
        logvar_t_prev = torch.where(
            (t == 0).view(-1, *([1] * (len(emb_t_prev.shape) - 1))), zero_tensor, logvar_t_prev
        )
        kl_div = normal_kl(mean1=emb_t, logvar1=logvar_t, mean2=emb_t_prev, logvar2=logvar_t_prev)
        return self.causal_weight * torch.mean(kl_div)

    def caculate_losses(self, model, emb_s, reweight=False):
        batch_size, device = emb_s.size(0), emb_s.device
        ts, pt = self.sample_timesteps(batch_size, device, 'uniform')
        noise = torch.randn_like(emb_s)
        emb_t = self.forward_process(emb_s, ts, noise)

        t_prev = torch.clamp(ts - 1, min=0)
        emb_t_prev = self.forward_process(emb_s, t_prev, noise)

        terms = {}
        model_output = model(emb_t, ts).to(device)
        assert model_output.shape == emb_s.shape

        mse = mean_flat((emb_s - model_output) ** 2)
        weight = torch.tensor([1.0] * len(model_output)).to(device)
        causal_reg = self.causal_regularization(model_output, emb_t_prev, ts)
        terms["loss"] = weight * mse + causal_reg
        terms["pred_xstart"] = model_output
        return terms

    def p_sample(self, model, emb_s, steps, sampling_noise=False):
        assert steps <= self.steps
        if steps == 0:
            emb_t = emb_s
        else:
            t = torch.tensor([steps - 1] * emb_s.shape[0]).to(emb_s.device)
            emb_t = self.q_sample(emb_s, t).float()

        indices = list(range(steps))[::-1]

        if self.noise_scale == 0.:
            for i in indices:
                t = torch.tensor([i] * emb_t.shape[0]).to(emb_s.device)
                emb_t = model(emb_t, t)
            return emb_t

        for i in indices:
            t = torch.tensor([i] * emb_t.shape[0]).to(emb_s.device)
            out = self.p_mean_variance(model, emb_t, t)
            if sampling_noise:
                noise = torch.randn_like(emb_t)
                nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(emb_t.shape) - 1))))
                emb_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            else:
                emb_t = out["mean"]
        return emb_t

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start, dtype=x_start.dtype)
        sqrt_alpha_bar = torch.sqrt(self.alphas_cumprod[t].view(-1, 1))
        sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alphas_cumprod[t].view(-1, 1))
        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    def sample_timesteps(self, batch_size, device, method='uniform', uniform_prob=0.001):
        if method == 'importance':
            if not (self.Lt_count == self.keep_num).all():
                return self.sample_timesteps(batch_size, device, method='uniform')
            Lt_sqrt = torch.sqrt(torch.mean(self.Lt_record ** 2, axis=-1))
            pt_all = Lt_sqrt / torch.sum(Lt_sqrt)
            pt_all *= 1 - uniform_prob
            pt_all += uniform_prob / len(pt_all)
            assert pt_all.sum(-1) - 1. < 1e-5
            t = torch.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)
            return t, pt
        elif method == 'uniform':
            t = torch.randint(0, self.steps, (batch_size,), device=device).long()
            pt = torch.ones_like(t).float()
            return t, pt
        else:
            raise ValueError

    def forward_process(self, emb_s, t, noise=None):
        if noise is None:
            noise = torch.randn_like(emb_s)
        assert noise.shape == emb_s.shape
        return (
                self._extract_into_tensor(self.sqrt_alphas_cumprod, t, emb_s.shape) * emb_s
                + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, emb_s.shape) * noise
        )

    def q_posterior_mean_variance(self, emb_s, emb_t, t):
        assert emb_s.shape == emb_t.shape
        device = self.device
        emb_s = emb_s.to(device)
        emb_t = emb_t.to(device)
        t = t.to(device)

        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, emb_t.shape) * emb_s
            + self._extract_into_tensor(self.posterior_mean_coef2, t, emb_t.shape) * emb_t
        )

        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, emb_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, emb_t.shape
        )

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t):
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, t)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)
        pred_xstart = model_output

        model_mean, _, _ = self.q_posterior_mean_variance(emb_s=pred_xstart, emb_t=x, t=t)
        assert (model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def SNR(self, t):
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)


# ========================================================================================
# LOSS FUNCTIONS
# ========================================================================================

def contrastive_loss(anchor, positive, negative, temp=0.2):
    anchor = F.normalize(anchor, dim=1)
    positive = F.normalize(positive, dim=1)
    negative = F.normalize(negative, dim=1)
    pos_logits = torch.sum(anchor * positive, dim=1) / temp
    neg_logits = torch.einsum('bd,nd->bn', anchor, negative) / temp
    return -pos_logits.mean() + torch.logsumexp(neg_logits, dim=1).mean()


def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2) / emb.shape[0]
    return emb_loss * reg


# ========================================================================================
# TORCH UTILITIES
# ========================================================================================

class TorchGraphInterface(object):
    @staticmethod
    def convert_sparse_mat_to_tensor(X):
        coo = X.tocoo()
        coords = np.array([coo.row, coo.col])
        i = torch.LongTensor(coords)
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)


# ========================================================================================
# ALGORITHM UTILITIES
# ========================================================================================

@jit(nopython=True)
def find_k_largest(K, candidates):
    """Find K largest values in candidates using heap"""
    n_candidates = []
    for iid, score in enumerate(candidates[:K]):
        n_candidates.append((score, iid))
    heapq.heapify(n_candidates)
    for iid, score in enumerate(candidates[K:]):
        if score > n_candidates[0][0]:
            heapq.heapreplace(n_candidates, (score, iid + K))
    n_candidates.sort(reverse=True)
    ids = [item[1] for item in n_candidates]
    scores = [item[0] for item in n_candidates]
    return ids, scores


# ========================================================================================
# EVALUATION METRICS
# ========================================================================================

class Metric(object):
    @staticmethod
    def hits(origin, res):
        hit_count = {}
        for user in origin:
            items = list(origin[user].keys())
            predicted = [item[0] for item in res[user]]
            hit_count[user] = len(set(items).intersection(set(predicted)))
        return hit_count

    @staticmethod
    def hit_ratio(origin, hits):
        total_num = 0
        for user in origin:
            items = list(origin[user].keys())
            total_num += len(items)
        hit_num = 0
        for user in hits:
            hit_num += hits[user]
        return round(hit_num / total_num, 5)

    @staticmethod
    def precision(hits, N):
        prec = sum([hits[user] for user in hits])
        return round(prec / (len(hits) * N), 5)

    @staticmethod
    def recall(hits, origin):
        recall_list = [hits[user] / len(origin[user]) for user in hits]
        recall = round(sum(recall_list) / len(recall_list), 5)
        return recall

    @staticmethod
    def NDCG(origin, res, N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    DCG += 1.0 / math.log(n + 2, 2)
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG += 1.0 / math.log(n + 2, 2)
            sum_NDCG += DCG / IDCG
        return round(sum_NDCG / len(res), 5)


def ranking_evaluation(origin, res, N):
    measure = []
    for n in N:
        predicted = {}
        for user in res:
            predicted[user] = res[user][:n]
        indicators = []
        if len(origin) != len(predicted):
            print('The Lengths of test set and predicted set do not match!')
            exit(-1)
        hits = Metric.hits(origin, predicted)
        hr = Metric.hit_ratio(origin, hits)
        indicators.append('Hit Ratio:' + str(hr) + '\n')
        prec = Metric.precision(hits, n)
        indicators.append('Precision:' + str(prec) + '\n')
        recall = Metric.recall(hits, origin)
        indicators.append('Recall:' + str(recall) + '\n')
        NDCG = Metric.NDCG(origin, predicted, n)
        indicators.append('NDCG:' + str(NDCG) + '\n')
        measure.append('Top ' + str(n) + '\n')
        measure += indicators
    return measure


# ========================================================================================
# SAMPLER
# ========================================================================================

def next_batch_pairwise(data, batch_size, n_negs=1):
    training_data = data.training_data.copy()
    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    while ptr < data_size:
        batch_end = min(ptr + batch_size, data_size)
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ptr = batch_end
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])
        yield u_idx, i_idx, j_idx


# ========================================================================================
# RECOMMENDER BASE CLASSES
# ========================================================================================

class Recommender(object):
    def __init__(self, conf, training_set, test_set, **kwargs):
        self.config = conf
        self.data = Data(self.config, training_set, test_set)

        model_config = self.config['model']
        self.model_name = model_config['name']
        self.ranking = self.config['item.ranking.topN']
        self.emb_size = int(self.config['embedding.size'])
        self.maxEpoch = int(self.config['max.epoch'])
        self.batch_size = int(self.config['batch.size'])
        self.lRate = float(self.config['learning.rate'])
        self.reg = float(self.config['reg.lambda'])
        self.output = self.config['output']

        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        self.model_log = Log(self.model_name, f"{self.model_name} {current_time}")

        self.result = []
        self.recOutput = []

    def initializing_log(self):
        self.model_log.add('### model configuration ###')
        config_items = self.config.config
        for k in config_items:
            self.model_log.add(f"{k}={str(config_items[k])}")

    def print_model_info(self):
        print('Model:', self.model_name)
        print('Embedding Dimension:', self.emb_size)
        print('Maximum Epoch:', self.maxEpoch)
        print('Learning Rate:', self.lRate)
        print('Batch Size:', self.batch_size)
        print('Regularization Parameter:', self.reg)

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def test(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def evaluate(self, rec_list):
        pass

    def execute(self):
        self.initializing_log()
        self.print_model_info()
        print('Initializing and building model...')
        self.build()
        print('Training Model...')
        self.train()
        print('Testing...')
        rec_list = self.test()
        print('Evaluating...')
        self.evaluate(rec_list)


class GraphRecommender(Recommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super(GraphRecommender, self).__init__(conf, training_set, test_set, **kwargs)
        self.data = Interaction(conf, training_set, test_set)
        self.bestPerformance = []
        self.topN = [int(num) for num in self.ranking]
        self.max_N = max(self.topN)

    def print_model_info(self):
        super(GraphRecommender, self).print_model_info()
        print(f'Training Set Size: (user number: {self.data.training_size()[0]}, '
              f'item number: {self.data.training_size()[1]}, '
              f'interaction number: {self.data.training_size()[2]})')
        print(f'Test Set Size: (user number: {self.data.test_size()[0]}, '
              f'item number: {self.data.test_size()[1]}, '
              f'interaction number: {self.data.test_size()[2]})')
        print('=' * 80)

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def test(self):
        def process_bar(num, total):
            try:
                rate = float(num) / total
                ratenum = int(50 * rate)
                print(f'\rProgress: [{"+" * ratenum}{" " * (50 - ratenum)}]{ratenum * 2}%', end='', flush=True)
            except:
                pass

        rec_list = {}
        user_count = len(self.data.test_set)
        for i, user in enumerate(self.data.test_set):
            candidates = self.predict(user)
            rated_list, _ = self.data.user_rated(user)
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8
            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        return rec_list

    def evaluate(self, rec_list):
        if len(self.data.test_set) == 0:
            print('Test set is empty. Skipping evaluation.')
            return
        self.recOutput.append('userId: recommendations in (itemId, ranking score) pairs\n')
        for user in self.data.test_set:
            line = user + ':' + ''.join(
                f" ({item[0]},{item[1]})" for item in rec_list[user]
            )
            line += '\n'
            self.recOutput.append(line)
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        out_dir = self.output
        file_name = f"{self.config['model']['name']}@{current_time}-top-{self.max_N}items.txt"
        FileIO.write_file(out_dir, file_name, self.recOutput)
        print('The result has been output to ', os.path.abspath(out_dir))
        file_name = f"{self.config['model']['name']}@{current_time}-performance.txt"
        self.result = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        self.model_log.add('###Evaluation Results###')
        self.model_log.add(''.join(self.result))
        FileIO.write_file(out_dir, file_name, self.result)
        print(f'The result of {self.model_name}:\n{"".join(self.result)}')

    def fast_evaluation(self, epoch):
        print('Evaluating the model...')
        rec_list = self.test()
        measure = ranking_evaluation(self.data.test_set, rec_list, self.topN)

        performance = {}
        current_top = None
        for m in measure:
            m = m.strip()
            if m.startswith('Top'):
                current_top = m.split()[-1]
            elif ':' in m:
                k, v = m.split(':')
                metric_name = f"{k}@{current_top}"
                performance[metric_name] = float(v)

        target_metrics = {
            k: v for k, v in performance.items()
            if k in ['Recall@10', 'NDCG@10', 'Recall@20', 'NDCG@20']
        }

        if self.bestPerformance:
            improved = all(
                target_metrics[k] >= self.bestPerformance[1].get(k, float('-inf'))
                for k in target_metrics
            )
            if improved:
                self.bestPerformance = [epoch + 1, target_metrics]
                self.save()
        else:
            self.bestPerformance = [epoch + 1, target_metrics]
            self.save()

        print('-' * 80)
        print(f'Real-Time Ranking Performance (Top-{self.max_N} Item Recommendation)')
        measure_str = ', '.join([f'{k}: {v}' for k, v in target_metrics.items()])
        print(f'*Current Performance*\nEpoch: {epoch + 1}, {measure_str}')
        if self.bestPerformance:
            bp = ', '.join([f'{k}: {v}' for k, v in self.bestPerformance[1].items()])
            print(f'*Best Performance*\nEpoch: {self.bestPerformance[0]}, {bp}')
        print('-' * 80)
        return measure


# ========================================================================================
# LGCN ENCODER
# ========================================================================================

class LGCN_Encoder(nn.Module):
    """LightGCN Encoder for user-item embeddings"""
    def __init__(self, data, emb_size, n_layers):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        import numpy as np
        import os

        item_emb_path = 'F:/PyCharmProject/CRDHNS/dataset/my_data/ciRNAEmbed.npy'
        user_emb_path = 'F:/PyCharmProject/CRDHNS/dataset/my_data/miRNAEmbed.npy'

        if os.path.exists(user_emb_path) and os.path.exists(item_emb_path):
            print(f"Loading pretrained embeddings from {user_emb_path} and {item_emb_path}...")
            try:
                raw_user_emb = np.load(user_emb_path)
                raw_item_emb = np.load(item_emb_path)

                print("Aligning embeddings with dataset IDs...")

                aligned_user_emb = np.zeros((self.data.user_num, self.latent_size), dtype=np.float32)
                aligned_item_emb = np.zeros((self.data.item_num, self.latent_size), dtype=np.float32)

                for ext_id, int_id in self.data.user.items():
                    try:
                        original_idx = int(ext_id)
                        if original_idx < raw_user_emb.shape[0]:
                            aligned_user_emb[int_id] = raw_user_emb[original_idx]
                    except ValueError:
                        pass

                for ext_id, int_id in self.data.item.items():
                    try:
                        original_idx = int(ext_id)
                        if original_idx < raw_item_emb.shape[0]:
                            aligned_item_emb[int_id] = raw_item_emb[original_idx]
                    except ValueError:
                        pass

                embedding_dict = nn.ParameterDict({
                    'user_emb': nn.Parameter(torch.FloatTensor(aligned_user_emb)),
                    'item_emb': nn.Parameter(torch.FloatTensor(aligned_item_emb)),
                })
                print("Pretrained embeddings loaded successfully.")
                return embedding_dict
            except Exception as e:
                print(f"Error loading pretrained embeddings: {e}")

        # Random initialization
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings


# ========================================================================================
# CRDHNS MODEL
# ========================================================================================

class CRDHNS(GraphRecommender):
    """CRDHNS Model - Diffusion-based Hard Negative Generation"""
    def __init__(self, conf, training_set, test_set):
        super(CRDHNS, self).__init__(conf, training_set, test_set)
        yaml_config = self.config['CRDHNS']
        self.n_layers = int(yaml_config['n_layer'])
        self.pre_train_item_emb = None

        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers)

        self.Diffusion = DiffusionProcess(
            args.noise_schedule, args.noise_scale, args.noise_min,
            args.noise_max, args.steps, device
        ).to(device)

        output_dims = [args.dims] + [args.n_hid]
        input_dims = output_dims[::-1]
        self.MLP = MLP(input_dims, output_dims, args.emb_size, time_type="cat", norm=args.norm).to(device)

        self.MLP_opt = torch.optim.Adam([{'params': self.MLP.parameters(), 'weight_decay': 0}], lr=args.lr)

        from collections import defaultdict
        self.result = defaultdict(list)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        initial_alpha, initial_beta = args.initial_alpha, args.initial_beta
        final_alpha, final_beta = args.final_alpha, args.final_beta
        cl_weight = args.cl_weight
        min_mix_ratio = args.min_mix_ratio

        for epoch in range(self.maxEpoch):
            progress = epoch / self.maxEpoch
            alpha = initial_alpha + (final_alpha - initial_alpha) * progress
            beta = initial_beta + (final_beta - initial_beta) * progress

            model.train()

            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch

                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[torch.tensor(user_idx)], rec_item_emb[torch.tensor(pos_idx)], rec_item_emb[torch.tensor(neg_idx)]
                rand_neg_samples = neg_item_emb

                mix_ratio = Beta(alpha, beta).sample((len(user_idx), 1)).to(pos_item_emb.device)
                mix_ratio = torch.clamp((1 - min_mix_ratio) * mix_ratio + min_mix_ratio, 0, 1)

                terms = self.Diffusion.caculate_losses(self.MLP, pos_item_emb, args.reweight)

                with torch.no_grad():
                    num_steps = args.num_steps
                    stride = args.stride
                    sample_steps = list(range(args.sample_start, num_steps + 1, stride))

                    all_neg_samples = []
                    scores = []
                    device = pos_item_emb.device
                    user_emb = user_emb.to(device)

                    for step in sample_steps:
                        step_neg_emb = self.Diffusion.p_sample(self.MLP, pos_item_emb, step, False).to(device)
                        step_scores = torch.sigmoid(torch.sum(user_emb * step_neg_emb, dim=1))
                        all_neg_samples.append(step_neg_emb)
                        scores.append(step_scores)

                    all_neg_samples = torch.stack(all_neg_samples).to(device)
                    scores = torch.stack(scores).to(device)
                    hard_neg_idx = torch.argmax(scores, dim=0)
                    hard_neg_samples = all_neg_samples[hard_neg_idx, torch.arange(len(hard_neg_idx), device=device)]

                mixed_neg = mix_ratio * hard_neg_samples + (1 - mix_ratio) * rand_neg_samples

                cl_loss = contrastive_loss(
                    anchor=pos_item_emb,
                    positive=pos_item_emb + args.positive_noise * torch.randn_like(pos_item_emb),
                    negative=torch.cat([hard_neg_samples, rand_neg_samples], dim=0),
                    temp=args.temp
                )

                bpr_loss = -torch.log(torch.sigmoid(
                    torch.sum(user_emb * (pos_item_emb - mixed_neg), dim=1)
                )).mean()

                total_loss = (
                    bpr_loss +
                    cl_weight * cl_loss +
                    args.diffusion_loss_weight * terms["loss"].mean() +
                    l2_reg_loss(self.reg, model.embedding_dict['user_emb'][torch.tensor(user_idx)], model.embedding_dict['item_emb'][torch.tensor(pos_idx)], model.embedding_dict['item_emb'][torch.tensor(neg_idx)]) / self.batch_size
                )

                optimizer.zero_grad()
                self.MLP_opt.zero_grad()
                total_loss.backward()
                optimizer.step()
                self.MLP_opt.step()

            with torch.no_grad():
                self.user_emb, self.item_emb = model()
                self.fast_evaluation(epoch)

        self.user_emb, self.item_emb = model()

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.detach().cpu().numpy()

    @torch.no_grad()
    def generate_hard_negatives(self, user_emb, pos_item_emb):
        """Generate hard negatives for external use"""
        self.MLP.eval()
        self.Diffusion.eval()

        num_steps = args.num_steps
        stride = args.stride
        sample_steps = list(range(args.sample_start, num_steps + 1, stride))

        all_neg_samples = []
        scores = []
        device = pos_item_emb.device

        for step in sample_steps:
            step_neg_emb = self.Diffusion.p_sample(self.MLP, pos_item_emb, step, False).to(device)
            step_scores = torch.sigmoid(torch.sum(user_emb * step_neg_emb, dim=1))
            all_neg_samples.append(step_neg_emb)
            scores.append(step_scores)

        all_neg_samples = torch.stack(all_neg_samples).to(device)
        scores = torch.stack(scores).to(device)
        hard_neg_idx = torch.argmax(scores, dim=0)
        hard_neg_samples = all_neg_samples[hard_neg_idx, torch.arange(len(hard_neg_idx), device=device)]

        self.MLP.train()
        self.Diffusion.train()

        return hard_neg_samples


# ========================================================================================
# MAIN EXECUTION
# ========================================================================================

def run_and_export():
    """Main execution function"""
    os.chdir(os.path.dirname(__file__) or '.')

    # 1. Load Configuration
    print("Loading configuration...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    conf_path = os.path.join(current_dir, 'conf', 'CRDHNS.yaml')
    conf = ModelConf(conf_path)

    # 2. Load Data
    print("Loading data...")
    training_data = FileIO.load_data_set(conf['training.set'], conf['model']['type'])
    test_data = FileIO.load_data_set(conf['test.set'], conf['model']['type'])

    # 3. Initialize Model
    print("Initializing CRDHNS model...")
    model = CRDHNS(conf, training_data, test_data)

    # 4. Train the model
    print("Starting training...")
    model.execute()

    # 5. Generate Hard Negatives
    print("\n" + "="*50)
    print("Training complete. Starting hard negative generation...")
    print("="*50)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    if hasattr(model, 'user_emb') and hasattr(model, 'item_emb'):
        user_emb_all = model.user_emb
        item_emb_all = model.item_emb
    else:
        user_emb_all, item_emb_all = model.model()

    print("Generating negatives and finding neighbors (User-by-User strategy)...")

    model.MLP.eval()
    model.Diffusion.eval()

    if not isinstance(item_emb_all, torch.Tensor):
        item_emb_all = torch.tensor(item_emb_all)
    item_emb_all = item_emb_all.to(device)
    items_norm = F.normalize(item_emb_all, p=2, dim=1)

    final_triplets = []

    user_list = list(model.data.training_set_u.keys())
    total_users = len(user_list)

    with torch.no_grad():
        for i, user_str in enumerate(user_list):
            if i % 100 == 0:
                print(f"Processed {i}/{total_users} users...")

            u_id_internal = model.data.user[user_str]
            pos_items_str = list(model.data.training_set_u[user_str].keys())
            pos_items_internal = [model.data.item[item] for item in pos_items_str]

            num_pos = len(pos_items_internal)
            if num_pos == 0:
                continue

            batch_u = torch.tensor([u_id_internal] * num_pos, dtype=torch.long).to(device)
            batch_i = torch.tensor(pos_items_internal, dtype=torch.long).to(device)

            batch_user_emb = user_emb_all[batch_u]
            batch_pos_item_emb = item_emb_all[batch_i]

            neg_emb = model.generate_hard_negatives(batch_user_emb, batch_pos_item_emb)
            neg_norm = F.normalize(neg_emb, p=2, dim=1)

            sim_scores = torch.matmul(neg_norm, items_norm.t())
            mask_indices = torch.tensor(pos_items_internal, dtype=torch.long).to(device)
            sim_scores[:, mask_indices] = -1e9

            used_negatives = set()
            user_results = []

            K = min(num_pos + 20, model.data.item_num)
            topk_scores, topk_indices = torch.topk(sim_scores, k=K, dim=1)
            topk_indices = topk_indices.cpu().numpy()

            for idx in range(num_pos):
                # Try to find a valid negative for interaction 'idx'
                candidates = topk_indices[idx]
                found = False
                for cand in candidates:
                    # Convert numpy int to Python int for dict lookup
                    cand_int = int(cand)
                    cand_str = model.data.id2item[cand_int]
                    # Skip if: 1) already used for this user, 2) equals user ID, 3) equals positive item
                    if cand_int not in used_negatives and cand_str != user_str and cand_str != pos_items_str[idx]:
                        neg_item_internal = cand_int
                        used_negatives.add(cand_int)
                        found = True
                        break

                if not found:
                    # Fallback: find ANY valid candidate
                    for cand in candidates:
                        cand_int = int(cand)
                        cand_str = model.data.id2item[cand_int]
                        if cand_str != user_str and cand_str != pos_items_str[idx]:
                            neg_item_internal = cand_int
                            found = True
                            break

                # Skip this interaction if no valid negative found
                if not found:
                    continue

                # Map back to string IDs
                pos_item_str = pos_items_str[idx]
                neg_item_str = model.data.id2item[neg_item_internal]
                final_triplets.append((user_str, pos_item_str, neg_item_str))

    # Save triplets
    print("Saving User-Positive-Negative Item triplets...")
    pairs_path = './dataset/hard_negative_pairs.txt'

    with open(pairs_path, 'w') as f:
        for u_str, p_str, n_str in final_triplets:
            f.write(f"{u_str},{p_str},{n_str}\n")

    print(f"Successfully saved {len(final_triplets)} triplets to {pairs_path}")

    print("Example mapping (First 5):")
    for i in range(min(5, len(final_triplets))):
        print(f"Triplet: {final_triplets[i]}")


if __name__ == '__main__':
    run_and_export()
