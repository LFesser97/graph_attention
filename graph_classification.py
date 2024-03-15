"""
graph_classification.py
"""

import os.path as osp
import time
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.nn import ModuleList

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WebKB, HeterophilousGraphDataset
from torch_geometric.nn import GATConv, global_mean_pool, GATv2Conv
from torch_geometric.utils import to_networkx, from_networkx, to_dense_adj

from GraphRicciCurvature.OllivierRicci import OllivierRicci

import networkx as nx


class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers, num_heads):
        """
        Create a GAT model with the given number of layers and heads
        for the given number of features and classes.
        """
        super(GAT, self).__init__()
        self.convs = ModuleList()
        self.convs.append(GATv2Conv(num_features, 8, heads=num_heads, dropout=0.6))
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(8 * num_heads, 8, heads=num_heads, dropout=0.6))
        self.lin = torch.nn.Linear(8 * num_heads, num_classes)
        # self.attention = None

    def forward(self, data):
        """
        Forward pass of the GAT model.
        """
        x, edge_index = data.x, data.edge_index
        # self.attention = []
        for i, conv in enumerate(self.convs):
            x = F.elu(conv(x, edge_index))
            # self.attention.append(conv.att)
        x = global_mean_pool(x, data.batch)
        x = F.elu(self.lin(x))
        return F.log_softmax(x, dim=1)
    
    def attention(self, data):
        """
        Return the attention scores of the GAT model using
        return_attention_scores=True.
        """
        x, edge_index = data.x, data.edge_index
        attention = []
        for i, conv in enumerate(self.convs):
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
        Create a GAT model for the given dataset with the given number of layers and heads.
        The dataset is one of Mutag, Enzymes, or Proteins, and has type list.

        Each element of the list is of the form Data(edge_index=[2, 38], x=[17, 7], edge_attr=[38, 4], y=[1])
        """
        dataset = self.dataset
        num_features = dataset[0].num_features
        # we only run binary classification experiments for now
        num_classes = 2
        model = GAT(num_features, num_classes, self.num_layers, self.num_heads).to(self.device)
        return model
    
    def train(self, epochs=100):
        """
        Train the GAT model on the given binary classification dataset
        for the given number of epochs.
        """
        model = self.model
        device = self.device
        dataset = self.dataset
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(epochs):
            for data in dataset:
                data = data.to(device)
                optimizer.zero_grad()
                y = data.y.to(device)
                out = model(data)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
        optimizer.zero_grad()
        self.epochs += epochs
        print(f"Model trained for {self.epochs} total epochs.")
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

    
def compute_edge_curvatures(graph):
    """
    Compute the Ollivier-Ricci curvature of a given directed and weighted graph.
    """
    curvatures = {}
    orc = OllivierRicci(graph, alpha=0, verbose="ERROR")
    orc.compute_ricci_curvature()
    for edge in graph.edges:
        if edge[0] != edge[1]:
            curvatures[edge] = orc.G.edges[edge]["ricciCurvature"]
    return curvatures