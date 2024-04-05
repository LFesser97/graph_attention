"""
node_classification.py

TODO: Organize code for individual experiements into classes
"""

import os.path as osp
import time
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WebKB, HeterophilousGraphDataset
from torch_geometric.nn import GATConv, global_mean_pool, GATv2Conv, GCNConv
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
    
    def node_similarity(self, data, device):
        """
        Compute the node similarity of the GAT model
        after each layer using the compute_node_similarity
        function.
        """
        x, edge_index = data.x, data.edge_index
        similarity = []
        similarity.append(compute_node_similarity(x, device))
        for i, conv in enumerate(self.convs):
            # use the compute_node_similarity function to compute the node similarity
            x = F.elu(conv(x, edge_index))
            similarity.append(compute_node_similarity(x, device))
        return similarity
    

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
        return x
    
    def node_similarity(self, data, device):
        """
        Compute the node similarity of the GCN model
        after each layer using the compute_node_similarity
        function.
        """
        x, edge_index = data.x, data.edge_index
        similarity = []
        similarity.append(compute_node_similarity(x, device))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            similarity.append(compute_node_similarity(x, device))
        return similarity


class Experiment:
    def __init__(self, dataset, num_layers, num_heads=1, model_type="GAT"):
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
        self.model_type = model_type
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
        if self.model_type == "GAT":
            model = GAT(num_features, num_classes, self.num_layers, self.num_heads).to(self.device)
        elif self.model_type == "GCN":
            model = GCN(num_features, num_classes, self.num_layers).to(self.device)
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
    
    def get_node_similarity(self, input_graph) -> list:
        """
        Compute the node similarity of the given graph
        after each layer of the model.
        """
        return self.model.node_similarity(input_graph, self.device)


class VisualizationMethods:
    """
    An abstract class with static
    methods for visualizing attention graphs.
    """

    @staticmethod
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
    def compute_gcn_deviation(attention_graph):
        """
        For each directed edge in the attention graph,
        compute the absolute difference from 1/deg(u),
        where deg(u) is the degree of the target node.
        """
        H = nx.DiGraph()
        for u, v in attention_graph.edges:
            # if u != v:
            w = attention_graph[u][v]["weight"]
            deg = attention_graph.in_degree(u)
            assert(deg > 1)
            H.add_edge(u, v, weight=abs(w - 1/deg))
            # H.add_edge(u, v, weight=1/deg)
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


class FeatureDistance:
    """
    An abstract class with static methods for computing
    the feature distance between nodes in a graph.
    """

    @staticmethod
    def create_feature_distance_dict(graph, deviation_graph):
        """
        For each edge in the given graph, compute the feature distance
        between the source and target nodes. The input if of the form
        Data(edge_index=[2, 554], num_nodes=183, x=[183, 1703], y=[183])
        
        Return a dictionary of the form {(u, v) : {'feature_distance': d}}.
        """
        feature_distances = {}
        for i in range(graph.edge_index.shape[1]):
            u, v = graph.edge_index[:, i].tolist()
            x_u = graph.x[u]
            x_v = graph.x[v]
            d = torch.norm(x_u - x_v, p=2).item()
            feature_distances[(u, v)] = {'feature_distance': d}

        # add self-loops with feature distance 0
        for i in range(graph.num_nodes):
            feature_distances[(i, i)] = {'feature_distance': 0}

        # add attention deviations for each edge
        feature_distances = FeatureDistance.add_attention_deviations(feature_distances, deviation_graph)
        return feature_distances

    @staticmethod
    def add_attention_deviations(distance_dict, deviation_graph):
        """
        For each edge in the given deviation graph, add the
        attention deviation to the corresponding edge in the
        distance dictionary.
        """
        for (u, v) in distance_dict.keys():
            d = deviation_graph[u][v]["weight"]
            distance_dict[(u, v)]['attention_deviation'] = d
        return distance_dict

    @staticmethod
    def plot_scatter(feature_dict):
        # Extract x and y values from the list of tuples
        x_values = [feature_dict[edge]['feature_distance'] for edge in feature_dict.keys()]
        y_values = [feature_dict[edge]['attention_deviation'] for edge in feature_dict.keys()]
        
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(x_values, y_values)
        plt.xlabel('Feature Distance (L2 Norm)')
        plt.ylabel('Attention Deviation')
        # plt.title('Scatter Plot')
        plt.grid(True)
        plt.show()

    @staticmethod
    def adjust_feature_distance_dict(feature_dict):
        """
        Adjust the feature distance dictionary by removing
        one of the two edges for each pair of undirected edges.

        Keep only one of the two edges in the dictionary with the
        absolute difference in attention scores.
        """
        adjusted_feature_dict = {}
        for (u, v) in feature_dict.keys():
            if (u, v) in adjusted_feature_dict.keys() or (v, u) in adjusted_feature_dict.keys():
                continue
            else:
                d1 = feature_dict[(u, v)]['attention_deviation']
                d2 = feature_dict[(v, u)]['attention_deviation']
                # compute the absolute difference in attention scores
                attention_difference = abs(d1 - d2)
                adjusted_feature_dict[(u, v)] = {'feature_distance': feature_dict[(u, v)]['feature_distance'], 'devistion_difference': attention_difference}
        return adjusted_feature_dict

    @staticmethod
    def plot_scatter_feature_distance_adjusted(feature_dict):
        # Extract x and y values from the list of tuples
        x_values = [feature_dict[edge]['feature_distance'] for edge in feature_dict.keys()]
        y_values = [feature_dict[edge]['devistion_difference'] for edge in feature_dict.keys()]
        
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(x_values, y_values)
        plt.xlabel('Feature Distance (L2 Norm)')
        plt.ylabel('Attention Deviation Difference')
        # plt.title('Scatter Plot')
        plt.grid(True)
        plt.show()


class EigenvectorCentrality:
    """
    An abstract class with static methods for
    computing the eigenvector centrality of nodes in a graph.
    """

    @staticmethod
    def create_eigvec_centrality_dict(graph, deviation_graph):
        """
        For each edge in the given graph, compute the eigenvector centrality
        of the source and target nodes. The input is of the form
        Data(edge_index=[2, 554], num_nodes=183, x=[183, 1703], y=[183])

        Return a dictionary of the form {(u, v) : {'eigvec_centrality_u': c},
        eigvec_centrality_v': c}, 'attention_deviation': d}.
        """
        # for each node in the graph, compute the eigenvector centrality
        G = to_networkx(graph)
        eigvec_centrality = nx.eigenvector_centrality(G)
        eigvec_centrality_dict = {}
        for i in range(graph.edge_index.shape[1]):
            u, v = graph.edge_index[:, i].tolist()
            c_u = eigvec_centrality[u]
            c_v = eigvec_centrality[v]
            eigvec_centrality_dict[(u, v)] = {'eigvec_centrality_u': c_u, 'eigvec_centrality_v': c_v}

        # add self-loops with eigenvector centrality c_u = c_v
        for i in range(graph.num_nodes):
            c = eigvec_centrality[i]
            eigvec_centrality_dict[(i, i)] = {'eigvec_centrality_u': c, 'eigvec_centrality_v': c}

        # add attention deviations for each edge
        eigvec_centrality_dict = FeatureDistance.add_attention_deviations(eigvec_centrality_dict, deviation_graph)
        return eigvec_centrality_dict

    @staticmethod
    def plot_scatter_eigvec(eigvec_dict):
        # Extract x and y values from the list of tuples
        x_values = [eigvec_dict[edge]['eigvec_centrality_u'] for edge in eigvec_dict.keys()]
        y_values = [eigvec_dict[edge]['attention_deviation'] for edge in eigvec_dict.keys()]
        
        # Create scatter plot, use a log scale for the x-axis
        plt.figure(figsize=(8, 6))
        plt.scatter(x_values, y_values)
        plt.xlabel('Eigenvector Centrality of Source Node')
        plt.ylabel('Attention Deviation')
        plt.xscale('log')
        # plt.title('Scatter Plot')
        plt.grid(True)
        plt.show()


class BetweennessCentrality:
    """
    An abstract class with static methods for
    computing the betweenness centrality of nodes in a graph.
    """

    @staticmethod
    def create_betweenness_dict(graph, deviation_graph):
        """
        For each edge in the given graph, compute the betweenness centrality
        of the source and target nodes. The input is of the form
        Data(edge_index=[2, 554], num_nodes=183, x=[183, 1703], y=[183])

        Return a dictionary of the form {(u, v) : {'betweenness_u': b},
        betweenness_v': b}, 'attention_deviation': d}.
        """
        # for each node in the graph, compute the betweenness centrality
        G = to_networkx(graph)
        betweenness_centrality = nx.betweenness_centrality(G)
        betweenness_dict = {}
        for i in range(graph.edge_index.shape[1]):
            u, v = graph.edge_index[:, i].tolist()
            b_u = betweenness_centrality[u]
            b_v = betweenness_centrality[v]
            betweenness_dict[(u, v)] = {'betweenness_u': b_u, 'betweenness_v': b_v}

        # add self-loops with betweenness centrality b_u = b_v
        for i in range(graph.num_nodes):
            b = betweenness_centrality[i]
            betweenness_dict[(i, i)] = {'betweenness_u': b, 'betweenness_v': b}

        # add attention deviations for each edge
        betweenness_dict = FeatureDistance.add_attention_deviations(betweenness_dict, deviation_graph)
        return betweenness_dict

    @staticmethod
    def plot_scatter_betweenness(betweenness_dict):
        # Extract x and y values from the list of tuples
        x_values = [betweenness_dict[edge]['betweenness_u'] for edge in betweenness_dict.keys()]
        y_values = [betweenness_dict[edge]['attention_deviation'] for edge in betweenness_dict.keys()]
        
        # Create scatter plot, use a log scale for the x-axis
        plt.figure(figsize=(8, 6))
        plt.scatter(x_values, y_values)
        plt.xlabel('Betweenness Centrality of Source Node')
        plt.ylabel('Attention Deviation')
        plt.xscale('log')
        # plt.title('Scatter Plot')
        plt.grid(True)
        plt.show()


class ORC:
    """
    An abstract class with static methods for 
    computing the Ollivier-Ricci curvature of edges in a graph.
    """

    @staticmethod
    def create_orc_dict(graph, deviation_graph):
        """
        For each edge in the given graph, compute the Ollivier-Ricci curvature
        of the edge. The input is of the form Data(edge_index=[2, 554], num_nodes=183,
        x=[183, 1703], y=[183])

        Return a dictionary of the form {(u, v) : {'orc': c}, 'attention_deviation': d}.
        """
        # turn the graph into an undirected networkx graph and remove self-loops
        G = to_networkx(graph)
        G = G.to_undirected()
        self_loops = list(nx.selfloop_edges(G))
        G.remove_edges_from(self_loops)

        # compute the Ollivier-Ricci curvature of the graph
        orc = OllivierRicci(G, alpha=0.5, verbose="ERROR")
        orc.compute_ricci_curvature()
        orc_dict = {}
        for i in range(graph.edge_index.shape[1]):
            u, v = graph.edge_index[:, i].tolist()
            if u != v:
                try:
                    c = orc.G[u][v]['ricciCurvature']
                    orc_dict[(u, v)] = {'orc': c}
                except KeyError:
                    c = orc.G[v][u]['ricciCurvature']
                    orc_dict[(u, v)] = {'orc': c}

        # add attention deviations for each edge
        orc_dict = FeatureDistance.add_attention_deviations(orc_dict, deviation_graph)
        return orc_dict

    @staticmethod
    def plot_scatter_orc(orc_dict):
        # Extract x and y values from the list of tuples
        x_values = [orc_dict[edge]['orc'] for edge in orc_dict.keys()]
        y_values = [orc_dict[edge]['attention_deviation'] for edge in orc_dict.keys()]
        
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(x_values, y_values)
        plt.xlabel('Ollivier-Ricci Curvature of Edge')
        plt.ylabel('Attention Deviation')
        plt.xlim(-2, 1)
        # plt.title('Scatter Plot')
        plt.grid(True)
        plt.show()

    @staticmethod
    def adjust_orc_dict(orc_dict):
        """
        Adjust the Ollivier-Ricci curvature dictionary by removing
        one of the two edges for each pair of undirected edges.

        Keep only one of the two edges in the dictionary with the
        absolute difference in attention scores.
        """
        adjusted_orc_dict = {}
        for (u, v) in orc_dict.keys():
            if (u, v) in adjusted_orc_dict.keys() or (v, u) in adjusted_orc_dict.keys():
                continue
            else:
                w1 = orc_dict[(u, v)]['attention_deviation']
                w2 = orc_dict[(v, u)]['attention_deviation']
                # compute the absolute difference in attention scores
                attention_difference = abs(w1 - w2)
                adjusted_orc_dict[(u, v)] = {'orc': orc_dict[(u, v)]['orc'], 'devistion_difference': attention_difference}
        return adjusted_orc_dict

    @staticmethod
    def plot_scatter_orc_adjusted(orc_dict):
        # Extract x and y values from the list of tuples
        x_values = [orc_dict[edge]['orc'] for edge in orc_dict.keys()]
        y_values = [orc_dict[edge]['devistion_difference'] for edge in orc_dict.keys()]
        
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(x_values, y_values)
        plt.xlabel('Ollivier-Ricci Curvature of Edge')
        plt.xlim(-2, 1)
        plt.ylabel('Attention Deviation Difference')
        # plt.title('Scatter Plot')
        plt.grid(True)
        plt.show()


class NodeLevelAccuracy:
    """
    An abstract class with static methods for
    computing the node-level accuracy of GCN or GAT.
    """
    @staticmethod
    def compute_node_level_accuracy(model_type, data, num_layers = 4, num_runs=10):
        """
        Train a GCN or GAT model for the given number of runs
        and compute for each node the number of times it was
        correctly classified.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        graph = data[0].to(device)
        # create a dictionary with nodes as keys and the number of times
        # they were correctly classified as values
        node_accuracy = {i: [] for i in range(graph.num_nodes)}
        epochs = 100

        for _ in range(num_runs):
            if model_type == "GCN":
                model = GCN(data.num_features, data.num_classes, num_layers).to(device)
            elif model_type == "GAT":
                model = GAT(data.num_features, data.num_classes, num_layers, num_heads=1).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
            model.train()
            criterion = torch.nn.CrossEntropyLoss()
            
            # create a list of training nodes by randomly choosing 50% of the nodes
            train_mask = torch.randperm(graph.num_nodes) < 0.5 * graph.num_nodes
            for epoch in tqdm(range(epochs)):
                optimizer.zero_grad()
                out = model(graph)
                loss = criterion(out[train_mask].to(device), graph.y[train_mask].to(device))
                loss.backward()
                optimizer.step()

            optimizer.zero_grad()

            # create a list of test nodes by randomly choosing 25% of the nodes
            test_mask = torch.randperm(graph.num_nodes) < 0.25 * graph.num_nodes
            # evaluate the model on each node in the test set
            model.eval()
            with torch.no_grad():
                logits = model(graph)
                pred = logits.argmax(1)
                for i in range(graph.num_nodes):
                    if pred[i] == graph.y[i]:
                        node_accuracy[i].append(1)
                    else:
                        node_accuracy[i].append(0)
        return {i: sum(node_accuracy[i]) / len(node_accuracy[i]) for i in range(graph.num_nodes)}


def compute_node_similarity(X, device):
    """
    Given a matrix of the node features,
    compute \|X - 1 gamma_X \|_F where gamma_X
    is 1^T X / n and 1 is the vector of ones.
    """
    n = X.shape[0]
    gamma_X = torch.ones(n).to(device) @ X / n
    outer_product = torch.ger(torch.ones(n).to(device), gamma_X)
    return torch.norm(X - outer_product, p="fro")


def compute_projection_matrix(X, device):
    """
    Compute the projection matrix P_X = X (X^T X)^{-1} X^T
    for the given matrix X.
    """
    return X @ torch.inverse(X.t() @ X) @ X.t()