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
from torch_geometric.utils import to_networkx, from_networkx, to_dense_adj, to_undirected

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
    
    def node_similarity(self, data):
        """
        Compute the node similarity of the GAT model
        after each layer using the compute_node_similarity
        function.
        """
        x, edge_index = data.x, data.edge_index
        similarity = []
        similarity.append(compute_node_similarity(x))
        for i, conv in enumerate(self.convs):
            # use the compute_node_similarity function to compute the node similarity
            x = F.elu(conv(x, edge_index))
            similarity.append(compute_node_similarity(x))
        return similarity


class Experiment:
    def __init__(self, dataset, num_layers, num_heads=1):
        # remove self-loops and convert to undirected graph
        x = dataset.data.x
        y = dataset.data.y
        dataset.data.edge_index = to_undirected(dataset.data.edge_index)
        G = to_networkx(dataset.data)
        self_loops = list(nx.selfloop_edges(G))
        G.remove_edges_from(self_loops)
        dataset.data = from_networkx(G)
        dataset.data.x = x
        dataset.data.y = y

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
        # create a list of training nodes by randomly choosing 50% of the nodes
        train_mask = torch.randperm(data.num_nodes) < 0.5 * data.num_nodes
        for epoch in tqdm(range(epochs)):
            optimizer.zero_grad()
            out = model(data)
            # loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss = criterion(out[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()
        self.epochs += epochs

        # create a list of test nodes by randomly choosing 25% of the nodes
        test_mask = torch.randperm(data.num_nodes) < 0.25 * data.num_nodes
        # evaluate the model on the test set
        model.eval()
        with torch.no_grad():
            logits = model(data)
            pred = logits.argmax(1)
            # test_acc = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
            test_acc = pred[test_mask].eq(data.y[test_mask]).sum().item() / test_mask.sum().item()
        print(f"Model trained for {self.epochs} total epochs.")
        print(f"Test accuracy: {test_acc:.4f}")
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
           att = attention_scores[1][i+input_graph.num_edges-1].item()
           G.add_edge(i, i, weight=att)
        return G, attention_scores
    
    def get_node_similarity(self, input_graph) -> list:
        """
        Compute the node similarity of the given graph
        after each layer of the model.
        """
        return self.model.node_similarity(input_graph)


def visualize_attention_digraph(G):
    """
    Visualize the attention scores of the given directed graph
    with edges with low attention scores in blue and edges with
    high attention scores in red. Set the max attention score
    to 1 and the min attention score to 0.
    """
    edge_colors = [G[u][v]["weight"] for u, v in G.edges]
    edge_colors = [(1 - w, 0, w) for w in edge_colors]
    # add a legend for the edge colors to the plot
    fig, ax = plt.subplots()
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, label="Attention Score")
    nx.draw(G, with_labels=False, node_color="lightgrey", node_size=50, edge_color=edge_colors, width=2.0, edge_cmap=plt.cm.Reds)
    plt.show()


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
        w1 = G[u][v]["weight"]
        w2 = G[v][u]["weight"]
        H.add_edge(u, v, weight=abs(w1 - w2))
    edge_colors = [H[u][v]["weight"] for u, v in H.edges]
    edge_colors = [(1 - w, 0, w) for w in edge_colors]
    nx.draw(H, with_labels=True, node_color="lightgrey", node_size=500, edge_color=edge_colors, width=2.0, edge_cmap=plt.cm.Reds)
    plt.show()


def compute_node_similarity(X):
    """
    Given a matrix of the node features,
    compute \|X - 1 gamma_X \|_F where gamma_X
    is 1^T X / n and 1 is the vector of ones.
    """
    n = X.shape[0]
    gamma_X = torch.ones(n) @ X / n
    gamma_X = gamma_X.unsqueeze(0).t()
    return torch.norm(X - torch.ones(n) @ gamma_X, p="fro")