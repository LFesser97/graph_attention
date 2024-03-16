"""
node_classification.py
"""

import os.path as osp
import time
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WebKB, HeterophilousGraphDataset
from torch_geometric.nn import GATConv, global_mean_pool, GATv2Conv
from torch_geometric.utils import to_networkx, from_networkx, to_dense_adj

from GraphRicciCurvature.OllivierRicci import OllivierRicci

import networkx as nx


class GAT(torch.nn.Module):
    """
    Create a GAT model for node classification.
    """
    def __init__(self, num_features, num_classes, num_layers, num_heads):
        super(GAT, self).__init__()
        self.convs = ModuleList()
        self.convs.append(GATv2Conv(num_features, 10, heads=num_heads, dropout=0.6))
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(10 * num_heads, 10, heads=num_heads, dropout=0.6))
        self.lin = torch.nn.Linear(10 * num_heads, num_classes)
        # self.attention = None

    def forward(self, data):
        """
        Forward pass of the GAT model.
        """
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs):
            # rel_edge_index = edge_index[:, data.edge_type == i]
            x = F.elu(conv(x, edge_index))
        return x
    
    def attention(self, data):
        """
        Return the attention scores of the GAT model using
        return_attention_scores=True.
        """
        x, edge_index = data.x, data.edge_index
        attention = []
        for i, conv in enumerate(self.convs):
            # rel_edge_index = edge_index[:, data.edge_type == i]
            x, att = conv(x, edge_index, return_attention_weights=True)
            attention.append(att)
            x = F.elu(x)
        return attention


class Experiment:
    def __init__(self, dataset, num_layers, num_heads=1):
        self.dataset = dataset
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.create_model()
        self.epochs = 0

    def create_model(self):
        """
        Create a GAT model with the given number of layers and heads
        for the given number of features and classes.
        """
        num_features = self.dataset.num_features
        num_classes = self.dataset.num_classes
        model = GAT(num_features, num_classes, self.num_layers, self.num_heads).to(self.device)
        return model
    
    def train(self, epochs=100):
        """
        Train the model for the given number of epochs.
        """
        model = self.model
        device = self.device
        data = self.dataset[0].to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        train_mask = [i for i in range(len(data.y)) if i % 2 == 0]
        for epoch in tqdm(range(epochs)):
            optimizer.zero_grad()
            out = model(data)
            # loss = criterion(out[data.train_mask], data.y[data.train_mask])
            # loss = criterion(out, data.y)
            loss = criterion(out[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()
        self.epochs += epochs

        # evaluate the model on the test set
        # model.eval()
        # with torch.no_grad():
            # logits = model(data)
            # pred = logits.argmax(1)
            # test_acc = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
        print(f"Model trained for {self.epochs} total epochs.")
        # print(f"Test accuracy: {test_acc:.4f}")
        self.model = model

    def create_attention_digraph(self, input_graph, layer):
        """
        Create a directed graph from the attention scores of the given layer.
        """
        attention_scores = self.model.attention(input_graph)[layer-1]
        H = to_networkx(input_graph)
        G = nx.DiGraph()
        num_edges = input_graph.num_edges + input_graph.num_nodes
        G.add_nodes_from(range(input_graph.num_nodes))
        # add edges from source to target with attention score as weight
        for i, edge in enumerate(H.edges):
            u, v = edge
            att = attention_scores[1][i].item()
            G.add_edge(u, v, weight=att)
        # add self-loops with attention score as weight
        for i in range(input_graph.num_nodes):
            att = attention_scores[1][i+input_graph.num_edges].item()
            G.add_edge(i, i, weight=att)
        return G, attention_scores