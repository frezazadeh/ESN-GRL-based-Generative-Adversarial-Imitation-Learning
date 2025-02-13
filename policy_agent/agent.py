import gym
import torch
import numpy as np
from torch.distributions import Categorical
from torch_geometric.data import Data
from itertools import permutations
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch_geometric.nn import GCNConv, LayerNorm, global_add_pool
import random

# Enable some TF32 options for speed (if available)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.empty_cache()

class Utils:
    @staticmethod
    def set_seed(seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

class GraphDataProcessor:
    @staticmethod
    def create_graph_data(data, index_pairs):
        # Create fully connected edge index for nodes (here using all permutations)
        edge_index = list(permutations(range(len(data)), 2))
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        # Create node features using the provided index pairs
        node_features = torch.tensor([[data[i], data[j]] for i, j in index_pairs], dtype=torch.float)
        return Data(x=node_features, edge_index=edge_index)

    @staticmethod
    def construct_index_pairs(num_nodes):
        index_pairs = [(i, i + 1) for i in range(0, num_nodes - 1, 2)]
        index_pairs.extend((i, num_nodes - i - 1) for i in range(num_nodes // 2))
        return index_pairs

class EchoStateNetwork(nn.Module):
    def __init__(self, input_dim, reservoir_size, spectral_radius=0.9, sparsity=0.5, leaky_rate=0.2):
        super(EchoStateNetwork, self).__init__()
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.leaky_rate = leaky_rate

        self.W_in = (torch.rand(reservoir_size, input_dim) - 0.5) * 2 / input_dim

        W = torch.rand(reservoir_size, reservoir_size) - 0.5
        mask = torch.rand(reservoir_size, reservoir_size) > sparsity
        W[mask] = 0

        eigenvector = torch.rand(reservoir_size, 1)
        for _ in range(50):  # Power iteration to estimate max eigenvalue
            eigenvector = W @ eigenvector
            eigenvector = eigenvector / eigenvector.norm()
        max_eigenvalue = eigenvector.norm()
        self.W = W * (spectral_radius / max_eigenvalue)

        self.register_buffer("state", torch.zeros(reservoir_size))

    def forward(self, x):
        device = x.device
        self.state = self.state.to(device)
        self.W_in = self.W_in.to(device)
        self.W = self.W.to(device)

        self.state = (1 - self.leaky_rate) * self.state + self.leaky_rate * torch.tanh(self.W_in @ x + self.W @ self.state)
        self.state = self.state / (self.state.norm(dim=0, keepdim=True).clamp(min=1e-6))
        return self.state

class GraphReinforceAgent(nn.Module):
    def __init__(self, input_dimension, output_dimension, esn_reservoir_size=500, hidden_layer_dimension=128, learning_rate=0.0005):
        super(GraphReinforceAgent, self).__init__()
        self.esn = EchoStateNetwork(input_dim=input_dimension, reservoir_size=esn_reservoir_size)
        self.graph_convolution_layer = GCNConv(2, hidden_layer_dimension)
        self.hidden_linear_layer = nn.Linear(hidden_layer_dimension + esn_reservoir_size, hidden_layer_dimension)
        self.output_layer = nn.Linear(hidden_layer_dimension, output_dimension)
        self.normalization_layer = LayerNorm(hidden_layer_dimension)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.1)
        self.experience_memory = []

    def store_transition(self, transition):
        self.experience_memory.append(transition)

    def forward(self, node_features, edge_index, esn_state):
        # Pass through GCN and then pool all nodes (assume one graph per sample)
        node_features = F.relu(self.graph_convolution_layer(node_features, edge_index))
        batch = torch.zeros(node_features.size(0), dtype=torch.long).to(node_features.device)
        node_features = global_add_pool(self.normalization_layer(node_features), batch)
        node_features = torch.cat((node_features, esn_state), dim=1)
        node_features = F.relu(self.hidden_linear_layer(node_features))
        output = self.output_layer(node_features)
        return F.log_softmax(output, dim=1)

    def optimize(self, discount_factor):
        running_discounted_reward = 0
        discounted_rewards = []
        # Compute discounted rewards (backward pass)
        for reward, log_prob in reversed(self.experience_memory):
            running_discounted_reward = reward + discount_factor * running_discounted_reward
            discounted_rewards.insert(0, running_discounted_reward)
        discounted_rewards = np.array(discounted_rewards)
        rewards_mean, rewards_std_dev = discounted_rewards.mean(), discounted_rewards.std()
        self.optimizer.zero_grad()
        policy_loss = 0
        for (reward, log_prob), Gt in zip(self.experience_memory, discounted_rewards):
            normalized_reward = (Gt - rewards_mean) / (rewards_std_dev + 1e-6)
            policy_loss += -log_prob * normalized_reward
        policy_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.experience_memory = []
