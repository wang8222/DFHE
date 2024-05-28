import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import networkx as nx
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, query_dim, key_dim, value_dim):
        super().__init__()

        self.query_transform = nn.Linear(query_dim, key_dim)
        self.key_transform = nn.Linear(key_dim, key_dim)
        self.value_transform = nn.Linear(value_dim, value_dim)



    def forward(self, query, key, value):
        query = self.query_transform(query)
        key = self.key_transform(key)
        value = self.value_transform(value)

        query = query.unsqueeze(1)  # 添加维度以匹配批量矩阵乘法的要求
        key = key.unsqueeze(1)  # 添加维度以匹配批量矩阵乘法的要求
        key = key.permute(0, 2, 1)  # 调整维度顺序

        att_scores = torch.bmm(query, key)
        # 对相似度进行Softmax归一化
        att_softmax = F.softmax(att_scores, dim=-1)
        value = value.unsqueeze(1)  # 添加维度以匹配批量矩阵乘法的要求
        # 上下文向量计算
        context = torch.bmm(att_softmax, value)

        return context


# 从 TXT 文件读取游走数据
def load_walks_from_file(filepath):
    with open(filepath, 'r') as f:
        return [line.strip().split() for line in f.readlines()]

def build_graph_from_walks(walks):
    unique_nodes = list(set(node for walk in walks for node in walk))
    node_to_idx = {node: i for i, node in enumerate(unique_nodes)}

    # 使用整数ID更新 walks
    updated_walks = [[node_to_idx[node] for node in walk] for walk in walks]

    # 使用 walks 构建图
    G = nx.Graph()
    for walk in updated_walks:
        for i in range(len(walk) - 1):
            if G.has_edge(walk[i], walk[i + 1]):
                G[walk[i]][walk[i + 1]]['weight'] += 1
            else:
                G.add_edge(walk[i], walk[i + 1], weight=1)
    return G,node_to_idx

def build_graph_from_walks_meta(walks,node_to_idx):
    updated_walks = [[node_to_idx[node] for node in walk] for walk in walks]
    G = nx.Graph()
    for walk in updated_walks:
        for i in range(len(walk) - 1):
            if G.has_edge(walk[i], walk[i + 1]):
                G[walk[i]][walk[i + 1]]['weight'] += 1
            else:
                G.add_edge(walk[i], walk[i + 1], weight=1)
    return G
# 将 networkx 图转为 PyG 图
def convert_nx_to_pyg(G):
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    embedding_dim = 64
    node_embeddings = nn.Embedding(G.number_of_nodes(), embedding_dim)
    x = node_embeddings(torch.arange(G.number_of_nodes()))
    return Data(x=x, edge_index=edge_index)

# Models
# class GCNEncoder(nn.Module):#编码器
#     def __init__(self, in_channels, hidden_channels):
#         super(GCNEncoder, self).__init__()
#         self.conv1 = GCNConv(in_channels, 128)
#         self.conv2 = GCNConv(128, 128)
#
#     def forward(self, x, edge_index):
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv2(x, edge_index)
#         return x
class AttGCNEncoder(nn.Module):

  def __init__(self, in_feats, hidden_feats):
    super().__init__()

    self.conv1 = None
    self.conv2 = None

    self.att = Attention(hidden_feats, hidden_feats, hidden_feats)

    self.in_feats = 64
    self.hidden_feats = 64

  def forward(self, x, edge_index):
      self.conv1 = GCNConv(self.in_feats, self.hidden_feats)
      h = F.relu(self.conv1(x, edge_index))
      # h = F.batch_norm(F.relu(h))  # 添加BN

      h = F.dropout(F.relu(h), p=0.5, training=self.training)
      self.conv2 = GCNConv(self.hidden_feats, self.hidden_feats)
      # h = F.batch_norm(F.relu(h))  # 添加BN

      h = F.relu(self.conv2(h, edge_index))

      # h1为conv1的输出,作为Attention的query
      h1 = h

      h = self.att(h1, h, h)

      return h
# class GCNDecoder(nn.Module):#解码器重建节点表示
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(GCNDecoder, self).__init__()
#         self.fc1 = nn.Linear(in_channels, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, 128)
#         self.relu = nn.ReLU()
#
#     def forward(self, z):
#         z = self.relu(self.fc1(z))
#         z = self.relu(self.fc2(z))
#         return self.fc3(z)
class MLPDecoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64, 64)

        self.relu = nn.ReLU()


    def forward(self, z):
        h = self.relu(self.fc1(z))
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.relu(self.fc2(h))
        return self.fc3(h)

# class GCNAutoencoder(nn.Module):#组合
#     def __init__(self, encoder, decoder):
#         super(GCNAutoencoder, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
class GCAE(nn.Module):

    def __init__(self):
        in_feats = 64  # 根据您的需求设置输入特征维度
        hidden_feats = 64  # 根据您的需求设置隐藏特征维度

        super(GCAE, self).__init__()
        self.encoder = AttGCNEncoder(in_feats, hidden_feats)
        self.decoder = MLPDecoder(64, 128, 64)

    # 前向传播
    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        return self.decoder(z)

# Training function
def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    x_hat = model(data.x, data.edge_index)
    target = data.x[:,:64].unsqueeze(1)
    loss = F.mse_loss(x_hat, target)
    loss.backward(retain_graph=True)
    optimizer.step()
    return loss.item()


def save_embeddings(model, data, node_to_idx, filepath='D:/.embedding'):
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)

    idx_to_node = {i: node for node, i in node_to_idx.items()}

    with open(filepath, 'w') as f:
        for idx, embed in enumerate(embeddings):
            node = idx_to_node[idx]
            embed_list = [str(e) for e in embed.tolist()]
            f.write(f"{node} {' '.join(embed_list)}\n")


def save_embedding(embeddings, node_to_idx, filepath='embeddings.txt'):
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    with open(filepath, 'w') as f:
        for idx, embed in enumerate(embeddings.weight):
            node = idx_to_node[idx]
            embed_list = [str(e) for e in embed.tolist()]
            f.write(f"{node} {' '.join(embed_list)}\n")



def reindex_graph_with_mapping(G):
    mapping = {node: i for i, node in enumerate(G.nodes())}
    reverse_mapping = {i: node for i, node in enumerate(G.nodes())}
    G_reindexed = nx.relabel_nodes(G, mapping)
    return G_reindexed, reverse_mapping



# graph_rw_random.txt
# Main
filepath = 'D:/.txt'
walks = load_walks_from_file(filepath)
G ,node_to_idx= build_graph_from_walks(walks)
data = convert_nx_to_pyg(G)

# Define global embeddings
num_global_nodes = len(node_to_idx)
global_embeddings = nn.Embedding(num_global_nodes, 64)
# encoder = GCNEncoder(16, 16)
# decoder = GCNDecoder(16, 16)
# model = GCNAutoencoder(encoder, decoder)
encoder = AttGCNEncoder(128, 64)
decoder = MLPDecoder(64,128,128)
model = GCAE()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    loss = train(model, data, optimizer)+1e-3*sum(p**2 for p in model.parameters())
    print(f"Epoch {epoch}, Loss: {loss}")
# Save embeddings
save_embeddings(model, data, node_to_idx)
