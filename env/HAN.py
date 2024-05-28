import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import scipy.sparse as sp
import dgl
from dgl.nn.pytorch import GATConv, GraphConv

class SemanticAttention(nn.Module):#语义注意力机制
    def __init__(self, in_size, hidden_size=64):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(#两个线性层和一个Tanh激活函数
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):#元路径重要性注意力,z为一个三维，路径数、项目数、维度
        w = self.project(z).mean(0)  # mean（0）第一维度求平均
        beta = torch.softmax(w, dim=0)  # (M, 1)，查询向量
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)


class HANLayer(nn.Module):
    """
    HAN layer.
    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types每个元路径都是边类型的列表
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability
    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features
    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, in_size, out_size, layer_num_heads, dropout, hidden_size, threshold):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleDict()#存储基于元路径邻接矩阵的每个GAT层
        self.rest_layers = nn.ModuleDict()
        self.mp_weights = nn.ParameterDict()#存储元路径权重
        self.in_size, self.out_size, self.layer_num_heads, self.dropout = in_size, out_size, layer_num_heads, dropout
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.sg_dict = dict()
        self.large_graph = dict()
        self.threshold = threshold
        self.useless_flag = False

        self.project = nn.Sequential(
            nn.Linear(out_size * layer_num_heads, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1, bias=False)
        )


    def reset(self):
        self.gat_layers.load_state_dict(self.rest_layers.state_dict())

    def metapath_reachable_graph(self, g, metapath):
        adj = 1

        for etype in metapath:

            # adj = adj * g.adj(etype=etype, scipy_fmt='csr', transpose=False)

            # 更新后的代码
            adj_matrix = g.adj(etype=etype).to_dense()#获取该关系类型下的邻接矩阵表示并从从稀疏格式转换成密集格式张量
            # adj_matrix_t = torch.transpose(adj_matrix, 0, 1)#对张量进行转置
            adj_csr = sp.csr_matrix(adj_matrix.numpy())#将tensor转换成numpy数组

            print('adj_csr.shape:',adj_csr.shape)
            adj = adj * adj_csr

        # adj = (adj != 0).tocsr()
        srctype = g.to_canonical_etype(metapath[0])[0]
        dsttype = g.to_canonical_etype(metapath[-1])[2]

        # if len(metapath) > 1:
        #     if metapath[1] == '1':
        #         adj = adj > 100
        #     elif metapath[1] == '3':
        #         adj = adj > 2

        node_pairs = adj.nonzero()#获取邻接矩阵adj中所有非零元素的位置。
        # nonzero()函数返回一个元组，其中包含了非零元素的行索引和列索引。这些索引被存储在node_pairs变量中

        density = adj.nnz / adj.shape[0] / adj.shape[0]#密度被定义为邻接矩阵中非零元素的数量除以邻接矩阵的总元素数量
        print('Density: ', density, end=' ')
        if density > self.threshold:
            return density

        modified_node_pairs = (node_pairs[0], node_pairs[1])#创建新的元组，它的元素是node_pairs元组的第一个和第二个元素
        if len(metapath) > 1:
            #adjb2 = g.adj(etype='2', scipy_fmt='csr', transpose=False) > 0
            topk_indices = np.argsort(adj.data)[-sum(adj.shape) * 3000:]#获取adj.data中最大的sum(adj.shape) * 5000个元素的索引
            #np.argsort()函数返回的是数组值从小到大的索引值2150

            node_pairs = (node_pairs[0][topk_indices], node_pairs[1][topk_indices])#卡一下5000，加不加无所谓

            # adjb = sp.csr_matrix(([True] * len(node_pairs[0]), node_pairs), shape=adjb2.shape)
            # adjb -= adjb.multiply(adjb2)
            # node_pairs = adjb.nonzero()
            modified_node_pairs = (node_pairs[0], node_pairs[1])

            # '''
            #     Select Top-k from each row.
            # '''
            # ml = adj.tolil()
            # random_pairs = ([], [])
            # for i, x in enumerate(ml.rows):
            #     if x:
            #         x = [(x2, ml.data[i][i2]) for i2, x2 in enumerate(x)]
            #         x = sorted(x, key=lambda y: y[1], reverse=True)
            #         x = [y[0] for y in x]
            #         for idx in range(min(len(x), 10)):
            #             random_pairs[0].append(i)
            #             # random_pairs[1].append(np.random.choice(x) + adj.shape[0])
            #             random_pairs[1].append(x[idx] + adj.shape[0])
            # random_pairs = (np.array(random_pairs[0]), np.array(random_pairs[1]))

            # if metapath[1] == '3':
            #     modified_node_pairs = (np.concatenate([modified_node_pairs[0], random_pairs[0]]),
            #                            np.concatenate([modified_node_pairs[1], random_pairs[1]]))

        print('#Edges: ', len(modified_node_pairs[0]), end=' ')

        new_g = dgl.graph(modified_node_pairs, num_nodes=sum(adj.shape), device='cpu').to_simple()
        #转换为了简单图，多重变合并为单一边。
        # # copy srcnode features
        # new_g.nodes[srctype].data.update(g.nodes[srctype].data)
        # # copy dstnode features
        # if srctype != dsttype:
        #     new_g.nodes[srctype].data.update(g.nodes[dsttype].data)

        new_g = dgl.to_bidirected(new_g)#再将每一个边添加一条反向的边

        return new_g
#------------------------------------------------------------------------------------------------------------
    def forward(self, g, h, meta_pathset, optimizer, b_ids, test=False):#h是节点特征

        self.useless_flag = False

        meta_paths = list(tuple(meta_path) for meta_path in meta_pathset)



        semantic_embeddings = []#语义嵌入
        device = 'cuda'if torch.cuda.is_available() else 'cpu'

        for meta_path in meta_paths[:]:#加载信息的循环
            mp = list(map(str, meta_path))
            if ''.join(mp) not in self.sg_dict and ''.join(mp) not in self.large_graph:
                #这行代码检查当前元路径是否已经在sg_dict或large_graph中。如果不在，那么它会执行以下的代码块。
                tim1 = time.time()
                print("Meta-path: ", str(mp), end=' ')
                # graph = dgl.metapath_reachable_graph(g, meta_path)
                graph = self.metapath_reachable_graph(g, meta_path)#获取由元路径定义的可达图
                # print('graph:',self.metapath_reachable_graph(g, meta_path))
                if isinstance(graph, float):#检查graph是否是一个浮点数。如果是，那么它会执行以下的代码块
                    self.large_graph[''.join(mp)] = graph#将graph添加到large_graph中，键是元路径的字符串表示
                    meta_paths.remove(meta_path)
                    meta_pathset.remove(list(meta_path))
                    self.useless_flag = True
                    print("Prepare dense meta-path graph: ", time.time() - tim1)
                    continue
                self.sg_dict[''.join(mp)] = graph#这行代码将graph添加到sg_dict中，键是元路径的字符串表示
                gatconv = nn.ModuleDict({''.join(mp): GATConv(self.in_size, self.out_size, self.layer_num_heads,
                                                              self.dropout, self.dropout,
                                                              allow_zero_in_degree=True).to(device)})

                self.gat_layers.update(gatconv)
                self.rest_layers.update(copy.deepcopy(gatconv))
                optimizer.add_param_group({'params': gatconv.parameters()})
                print("Average degree: ", graph.number_of_edges() / graph.number_of_nodes(), end=' ')
                print("Prepare meta-path graph (s): ", time.time() - tim1)
            elif ''.join(mp) in self.large_graph:
                self.useless_flag = True
                meta_pathset.remove(list(meta_path))
                meta_paths.remove(meta_path)

        # print("Processed Meta-path Set: ", meta_paths)

        for i, meta_path in enumerate(meta_paths):#形成语义嵌入的循环
            mp = list(map(str, meta_path))#元路径中的每个元素转换为字符串
            graph = self.sg_dict[''.join(mp)]

            sampler = dgl.dataloading.MultiLayerNeighborSampler([500])#多层邻居采样器，每个节点邻居选500个
            dataloader = dgl.dataloading.DataLoader(
                graph, torch.LongTensor(list(set(b_ids.tolist()))), sampler, device=torch.device(device),
                batch_size=len(b_ids),
                drop_last=False)
            for input_nodes, output_nodes, blocks in dataloader:
                emb = self.gat_layers[''.join(mp)](blocks[0], h[input_nodes]).flatten(1)
                #对输入节点的特征h[input_nodes]应用GAT层进行转换，并将结果从第二个维度扁平化，然后将结果存储在emb中
                if emb.shape[0] != len(b_ids):#检查emb的形状是否与b_ids的长度相等
                    c = torch.zeros((h.shape[0], self.out_size)).to(device)
                    c[output_nodes] = emb
                    emb = c[b_ids]
                semantic_embeddings.append(emb)
        # if len(semantic_embeddings) == 0:
        #     semantic_embeddings.append(torch.rand((h.shape[0], self.out_size * self.layer_num_heads)).to(device))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HAN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_heads, dropout, threshold=0.75):
        super(HAN, self).__init__()



        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(in_size, hidden_size, num_heads[0], dropout, hidden_size=32, threshold=threshold))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(hidden_size * num_heads[l - 1],
                                        hidden_size, num_heads[l], dropout, hidden_size=32, threshold=threshold))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h, meta_paths, optimizer, b_ids, test=False):

        for gnn in self.layers:
            h = gnn(g.cpu(), h, meta_paths, optimizer, b_ids, test)
        return self.predict(h)

    def get_embedding(self, g, h, meta_paths, optimizer, b_ids, test=False):
        for gnn in self.layers:
            h = gnn(g.cpu(), h, meta_paths, optimizer, b_ids, test)
        return h

    def reset(self):
        for gnn in self.layers:
            gnn.reset()
