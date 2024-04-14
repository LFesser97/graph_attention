"""
graph_classification.py
"""

import os.path as osp
import time
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import ModuleList

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WebKB, HeterophilousGraphDataset
from torch_geometric.nn import GATConv, global_mean_pool, GATv2Conv, GCNConv
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
        self.convs.append(GATv2Conv(num_features, 32, heads=num_heads, dropout=0.6))
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(32 * num_heads, 32, heads=num_heads, dropout=0.6))
        self.lin = torch.nn.Linear(32 * num_heads, num_classes)
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
    

class GCN(torch.nn.Module):
    """
    Create a GCN model for node classification
    with hidden layers of size 32.
    """
    def __init__(self, num_features, num_classes, num_layers):
        super(GCN, self).__init__()
        self.convs = ModuleList()
        self.convs.append(GCNConv(num_features, 32))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(32, 32))
        self.lin = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        """
        Forward pass of the GCN model.
        """
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, data.batch)
        x = F.relu(self.lin(x))
        return F.log_softmax(x, dim=1)
    

class GATGCN(torch.nn.Module):
    """
    Create a GAT-GCN model for node classification
    with hidden layers of size 32.
    """
    def __init__(self, num_features, num_classes, num_layers, num_heads = 1, gat_layers = [1]):
        """
        Create a GCN model with the given number of layers
        that uses GAT for the layers in gat_layers.
        """
        super(GATGCN, self).__init__()
        self.gat_layers = gat_layers
        self.convs = ModuleList()
        if 0 in gat_layers:
            self.convs.append(GATv2Conv(num_features, 32, heads=num_heads))
        else:
            self.convs.append(GCNConv(num_features, 32))
        for i in range(1, num_layers):
            if i in gat_layers:
                self.convs.append(GATv2Conv(32, 32, heads=num_heads))
            else:
                self.convs.append(GCNConv(32, 32))
        self.lin = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        """
        Forward pass of the GAT-GCN model.
        """
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, data.batch)
        x = F.relu(self.lin(x))
        return F.log_softmax(x, dim=1)
    
    def attention(self, data):
        """
        Return the attention scores of the GAT layers in
        the GAT-GCN model using return_attention_scores=True.
        """
        x, edge_index = data.x, data.edge_index
        attention = []
        for i, conv in enumerate(self.convs):
            if i in self.gat_layers:
                x, att = conv(x, edge_index, return_attention_weights=True)
                attention.append(att)
                x = F.relu(x)
            else:
                attention.append(None) # TODO: Implement 'Attention' for GCN layers
                x = F.relu(conv(x, edge_index))
        return attention
    

class Experiment:
    def __init__(self, dataset, num_layers, num_heads=1, model_type="GAT", gat_layers=[]):
        self.dataset = dataset
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.model_type = model_type
        self.gat_layers = gat_layers
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
        if self.model_type == "GAT":
            model = GAT(num_features, num_classes, self.num_layers, self.num_heads).to(self.device)
        elif self.model_type == "GCN":
            model = GCN(num_features, num_classes, self.num_layers).to(self.device)
        elif self.model_type == "GATGCN":
            model = GATGCN(num_features, num_classes, self.num_layers, self.num_heads, self.gat_layers).to(self.device)
        return model
    
    def train(self, epochs=100):
        """
        Train the GAT model on the given binary classification dataset
        for the given number of epochs.
        """
        model = self.model
        device = self.device
        dataset = self.dataset
        # randomly choose 80% of the dataset for training
        elements_to_choose = int(0.8 * len(dataset))
        train_dataset = random.sample(dataset, elements_to_choose)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in tqdm(range(epochs)):
            for data in train_dataset:
                data = data.to(device)
                optimizer.zero_grad()
                y = data.y.to(device)
                out = model(data)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
        optimizer.zero_grad()
        self.epochs += epochs

        # test the model
        model.eval()
        test_dataset = [data for data in dataset if data not in train_dataset]
        correct = 0
        for data in test_dataset:
            data = data.to(device)
            y = data.y.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct += int((pred == y).sum())
        acc = correct / len(test_dataset)
        print(f"Model trained for {self.epochs} total epochs.")
        print(f"Accuracy: {acc}")
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


class VisualizationMethods:
    """
    An abstract class with static
    methods for visualizing attention graphs.
    """

    @staticmethod
    def visualize_attention_digraph(G, deviation=False):
        """
        Visualize the attention scores of the given directed graph
        with edges with low attention scores in blue and edges with
        high attention scores in red. Set the max attention score
        to 1 and the min attention score to 0.
        """
        if deviation:
            # replace the edge weights with the difference from 1/deg(u)
            G = AttentionDeviation.compute_gcn_deviation(G)
        pos = nx.spring_layout(G)  # Define the layout
        weights = [G[u][v]['weight'] for u, v in G.edges()]  # Extract edge weights

        # Create a color map based on edge weights
        cmap = plt.cm.coolwarm  # You can choose any colormap you like
        normalize = plt.Normalize(vmin=min(weights), vmax=max(weights))
        edge_colors = [cmap(normalize(weight)) for weight in weights]

        # Draw nodes and edges
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=50)
        nx.draw_networkx_edges(G, pos, width=2, edge_color=edge_colors)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')

        # Show the plot
        plt.axis('off')
        plt.show()
        
                
    @staticmethod
    def visualize_attention_differences(G):
        """
        Create a new graph where for each pair of undirected edges (u, v) and (v, u),
        we add a new undirected edge with weight equal to the absolute difference
        between the attention scores of the original edges. Visualize the new graph
        with edges with low attention differences in blue and edges with high attention
        differences in red. Set the max attention difference to 1 and the min attention
        difference to 0.
        """
        H = nx.Graph()
        for u, v in G.edges:
            if u != v:
                w1 = G[u][v]["weight"]
                w2 = G[v][u]["weight"]
                H.add_edge(u, v, weight=abs(w1 - w2))
        edge_colors = [H[u][v]["weight"] for u, v in H.edges]
        edge_colors = [(1 - w, 0, w) for w in edge_colors]
        nx.draw(H, with_labels=True, node_color="lightgrey", node_size=50, edge_color=edge_colors, width=2.0, edge_cmap=plt.cm.Reds)
        plt.show()

    
class AttentionDeviation:
    """
    An abstract class with static
    methods for computing attention deviations.
    """

    @staticmethod
    def compute_gcn_deviation(attention_graph, absolute=True):
        """
        For each directed edge in the attention graph,
        compute the absolute difference from 1/deg(u),
        where deg(u) is the degree of the target node.
        """
        H = nx.DiGraph()
        for u, v in attention_graph.edges:
            w = attention_graph[u][v]["weight"]
            deg = attention_graph.in_degree(u)
            assert(deg > 1)
            if absolute:
                H.add_edge(u, v, weight=abs(w - 1/deg))
            else:
                H.add_edge(u, v, weight=w - 1/deg)
        return H

    @staticmethod
    def compute_percentiles(values):
        min_val = np.min(values)
        percentile_25 = np.percentile(values, 25)
        median = np.percentile(values, 50)
        percentile_75 = np.percentile(values, 75)
        max_val = np.max(values)
        
        return min_val, percentile_25, median, percentile_75, max_val

    @staticmethod
    def plot_box_plots(data_dict, ylabel="Deviation from GCN Attention"):
        # Get keys and values from the dictionary
        keys = list(data_dict.keys())
        values = list(data_dict.values())
        
        # Create box plots
        plt.figure(figsize=(10, 6))
        plt.boxplot(values, labels=keys)
        plt.xlabel('Degree of the Target Node')
        plt.ylabel(ylabel)
        plt.ylim(0, 1)
        # plt.title('Box plots of values for each key')
        plt.grid(True)
        plt.show()