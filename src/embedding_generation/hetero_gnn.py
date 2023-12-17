import os
import copy
import torch
import deepsnap
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from sklearn.metrics import f1_score
from deepsnap.hetero_gnn import forward_op
from torch_sparse import SparseTensor, matmul

from src.data_modelling.deepsnap_datasets import create_deepsnap_dataset

class HeteroGNNConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels_src, in_channels_dst, out_channels):
        super(HeteroGNNConv, self).__init__(aggr="sum")

        self.in_channels_src = in_channels_src
        self.in_channels_dst = in_channels_dst
        self.out_channels = out_channels

        self.lin_dst = None
        self.lin_src = None

        self.lin_update = None

        self.lin_src = nn.Linear(self.in_channels_src, self.out_channels, bias=False)
        self.lin_dst = nn.Linear(self.in_channels_dst, self.out_channels, bias=False)
        self.lin_update = nn.Linear(self.out_channels * 2, self.out_channels, bias=False)

    def forward(
        self,
        node_feature_src,
        node_feature_dst,
        edge_index,
        size=None
    ):
        return self.propagate(edge_index, size=size, node_feature_dst=node_feature_dst, node_feature_src=node_feature_src)


    def message_and_aggregate(self, edge_index, node_feature_src):

        return matmul(edge_index, node_feature_src, reduce=self.aggr)
    

    def update(self, aggr_out, node_feature_dst):
        node_feature_dst = self.lin_dst(node_feature_dst)
        aggr_out = self.lin_src(aggr_out)
        aggr_out = self.lin_update(torch.cat((node_feature_dst, aggr_out), dim=1))
        return aggr_out


class HeteroGNNWrapperConv(deepsnap.hetero_gnn.HeteroConv):
    def __init__(self, convs, args, aggr="mean"):
        super(HeteroGNNWrapperConv, self).__init__(convs, None)
        self.aggr = aggr

        # Map the index and message type
        self.mapping = {}

        # A numpy array that stores the final attention probability
        self.alpha = None

        self.attn_proj = None

        if self.aggr == "attn":

            self.attn_proj = nn.Sequential(
                nn.Linear(args['hidden_size'], args['attn_size']),
                nn.Tanh(),
                nn.Linear(args['attn_size'], 1, bias=False)
            )

    def reset_parameters(self):
        super(HeteroGNNWrapperConv, self).reset_parameters()
        if self.aggr == "attn":
            for layer in self.attn_proj.children():
                layer.reset_parameters()

    def forward(self, node_features, edge_indices):
        message_type_emb = {}
        for message_key, message_type in edge_indices.items():
            src_type, edge_type, dst_type = message_key
            node_feature_src = node_features[src_type]
            node_feature_dst = node_features[dst_type]
            edge_index = edge_indices[message_key]
            message_type_emb[message_key] = (
                self.convs[message_key](
                    node_feature_src,
                    node_feature_dst,
                    edge_index,
                )
            )
        node_emb = {dst: [] for _, _, dst in message_type_emb.keys()}
        mapping = {}
        for (src, edge_type, dst), item in message_type_emb.items():
            mapping[len(node_emb[dst])] = (src, edge_type, dst)
            node_emb[dst].append(item)
        self.mapping = mapping
        for node_type, embs in node_emb.items():
            if len(embs) == 1:
                node_emb[node_type] = embs[0]
            else:
                node_emb[node_type] = self.aggregate(embs)
        return node_emb

    def aggregate(self, xs):
        # Here, xs is a list of tensors (embeddings) with respect to message
        # type aggregation results.

        if self.aggr == "mean":
            xs = torch.stack(xs, dim=0)
            return xs.mean(dim=0)

        elif self.aggr == "attn":
            N = xs[0].shape[0] # Number of nodes for that node type
            M = len(xs) # Number of message types for that node type

            x = torch.cat(xs, dim=0).view(M, N, -1) # M * N * D
            z = self.attn_proj(x).view(M, N) # M * N * 1
            z = z.mean(1) # M * 1
            alpha = torch.softmax(z, dim=0) # M * 1

            # Store the attention result to self.alpha as np array
            self.alpha = alpha.view(-1).data.cpu().numpy()

            alpha = alpha.view(M, 1, 1)
            x = x * alpha
            return x.sum(dim=0)


def generate_convs(hetero_graph, conv, hidden_size, first_layer=False):
    # returns a dictionary of `HeteroGNNConv`
    # layers where the keys are message types. `hetero_graph` is deepsnap `HeteroGraph`
    # object and the `conv` is the `HeteroGNNConv`.
    convs = {}

    for message_type in hetero_graph.message_types:
      if first_layer:
        in_channels_src = hetero_graph.num_node_features(message_type[0])
        in_channels_dst = hetero_graph.num_node_features(message_type[2])
      else:
        in_channels_src = hidden_size
        in_channels_dst = hidden_size
      convs[message_type] = conv(in_channels_src, in_channels_dst, hidden_size)

    return convs


class HeteroGNN(torch.nn.Module):
    def __init__(self, hetero_graph, args, aggr="mean"):
        super(HeteroGNN, self).__init__()

        self.aggr = aggr
        self.hidden_size = args['hidden_size']

        self.convs1 = None
        self.convs2 = None

        self.bns1 = nn.ModuleDict()
        self.bns2 = nn.ModuleDict()
        self.relus1 = nn.ModuleDict()
        self.relus2 = nn.ModuleDict()
        self.post_mps = nn.ModuleDict()

        self.convs1 = HeteroGNNWrapperConv(generate_convs(hetero_graph, HeteroGNNConv, self.hidden_size, first_layer=True), aggr=self.aggr, args=args)
        self.convs2 = HeteroGNNWrapperConv(generate_convs(hetero_graph, HeteroGNNConv, self.hidden_size), aggr=self.aggr, args=args)
        for node_type in hetero_graph.node_types:
          self.bns1[node_type] = nn.BatchNorm1d(self.hidden_size, eps=1)
          self.bns2[node_type] = nn.BatchNorm1d(self.hidden_size, eps=1)
          self.relus1[node_type] = nn.LeakyReLU()
          self.relus2[node_type] = nn.LeakyReLU()
          self.post_mps[node_type] = nn.Linear(self.hidden_size, args['label_size'])
          

    def get_embeddings(self, node_feature, edge_index, last_relu=True):
        x = node_feature

        x = self.convs1(x, edge_index)
        x = forward_op(x, self.bns1)
        x = forward_op(x, self.relus1)
        x = self.convs2(x, edge_index)
        x = forward_op(x, self.bns2)
        if last_relu:
            x = forward_op(x, self.relus2)
        return x


    def forward(self, node_feature, edge_index):
        x = node_feature

        x = self.convs1(x, edge_index)
        x = forward_op(x, self.bns1)
        x = forward_op(x, self.relus1)
        x = self.convs2(x, edge_index)
        x = forward_op(x, self.bns2)
        x = forward_op(x, self.relus2)
        x = forward_op(x, self.post_mps)

        return x

    def loss(self, preds, y, indices):

        loss = 0
        loss_func = F.cross_entropy

        for node_type in preds.keys():
            pred, gt, idx = preds[node_type], y[node_type], indices[node_type]
            loss += loss_func(pred[idx], gt[idx].float())

        return loss
    

def train(model, optimizer, hetero_graph, train_index):
    model.train()
    optimizer.zero_grad()
    preds = model(hetero_graph.node_feature, hetero_graph.edge_index)

    loss = None

    y = hetero_graph.node_label
    loss = model.loss(preds, y, train_index)

    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, graph, target_table, index, best_model=None, best_val=0):
    model.eval()
    accs = dict()

    preds = model(graph.node_feature, graph.edge_index)
    num_node_types = 0
    micro = 0
    macro = 0
    for node_type in preds:
        if node_type != target_table:
            continue
        idx = index[node_type]

        pred = preds[node_type][idx]
        pred = pred.max(1)[1]
        label_np = graph.node_label[node_type][idx].cpu().numpy()
        pred_np = pred.cpu().numpy()
        micro = f1_score(np.argmax(label_np, axis=1), pred_np, average='micro')
        macro = f1_score(np.argmax(label_np, axis=1), pred_np, average='macro')
        num_node_types += 1

        accs[node_type] = (micro, macro)
    
    if accs[target_table][0] > best_val:
        best_val = accs[target_table][0]
        best_model = copy.deepcopy(model)

    return accs, best_model, best_val


def train_hetero_gnn(dataset_name, target_table, masked_tables, model_save_dir='models/hetero_gnn', k=10, epochs=200):
    best_model = None
    best_val = 0
    args = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'hidden_size': 64,
        'epochs': epochs,
        'weight_decay': 1e-05,
        'lr': 0.008,
          'attn_size': 32,
        'label_size': k
        }
    
    hetero_graph = create_deepsnap_dataset(dataset_name, target_table, masked_tables, k=k)

    # select 90% of random nodes as train nodes
    train_index = dict()
    test_index = dict()
    for node_type in hetero_graph.node_label.keys():
        train_index[node_type] = torch.randperm(hetero_graph.num_nodes(node_type))[:int(hetero_graph.num_nodes(node_type) * 0.9)]
        test_index[node_type] = torch.arange(hetero_graph.num_nodes(node_type))[int(hetero_graph.num_nodes(node_type) * 0.9):]
    # Node feature and node label to device
    for key in hetero_graph.node_feature:
        hetero_graph.node_feature[key] = hetero_graph.node_feature[key].to(args['device'])
    for key in hetero_graph.node_label:
        hetero_graph.node_label[key] = hetero_graph.node_label[key].to(args['device'])

    # Edge_index to sparse tensor and to device
    for key in hetero_graph.edge_index:
        edge_index = hetero_graph.edge_index[key]
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(hetero_graph.num_nodes(key[0]), hetero_graph.num_nodes(key[2])))
        hetero_graph.edge_index[key] = adj.t().to(args['device'])

    model = HeteroGNN(hetero_graph, args, aggr="attn").to(args['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    pbar = tqdm(range(args['epochs']))
    for epoch in pbar:
        loss = train(model, optimizer, hetero_graph, train_index)
        accs, best_model, best_val = test(model, hetero_graph, target_table, test_index, best_model, best_val)

        pbar.set_description(f"Epoch {epoch + 1}: loss {round(loss, 5)}, val micro {round(accs[target_table][0] * 100, 2)}%, val macro {round(accs[target_table][1] * 100, 2)}%")

    model = best_model
    model_save_path = os.path.join(model_save_dir, dataset_name)
    os.makedirs(model_save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_save_path, f"model_{target_table}.pt"))

        
def compute_hetero_gnn_embeddings(hetero_graph, dataset_name, target_table, model_save_dir='models/hetero_gnn', k=10):
    args = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'hidden_size': 64,
        'weight_decay': 1e-05,
        'lr': 0.008,
          'attn_size': 32,
        'label_size': k
        }
    model_save_path = os.path.join(model_save_dir, dataset_name, f"model_{target_table}.pt")

    # Node feature to device
    for key in hetero_graph.node_feature:
        hetero_graph.node_feature[key] = hetero_graph.node_feature[key].to(args['device'])

    # Edge_index to sparse tensor and to device
    for key in hetero_graph.edge_index:
        if isinstance(hetero_graph.edge_index[key], SparseTensor):
            continue
        edge_index = hetero_graph.edge_index[key]
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(hetero_graph.num_nodes(key[0]), hetero_graph.num_nodes(key[2])))
        hetero_graph.edge_index[key] = adj.t().to(args['device'])

    model = HeteroGNN(hetero_graph, args, aggr="attn").to(args['device'])
    model.load_state_dict(torch.load(model_save_path))

    # save last layer embeddings
    with torch.no_grad():
        last_layer_embeddings = model.get_embeddings(hetero_graph.node_feature, hetero_graph.edge_index, last_relu=False)
    return last_layer_embeddings[target_table].cpu().detach().numpy()
    

def main():
    target_table = 'test'
    masked_tables = ['test']
    dataset_name = 'rossmann-store-sales'
    train_hetero_gnn(dataset_name, target_table, masked_tables)
    hetero_graph = create_deepsnap_dataset(dataset_name, target_table, masked_tables, k=10)
    embeddings = compute_hetero_gnn_embeddings(hetero_graph, dataset_name, target_table)
    np.save(f'data/embeddings/{target_table}_embeddings.npy', embeddings)

    target_table = 'molecule'
    dataset_name = 'mutagenesis'
    masked_tables = ['molecule', 'atom', 'bond']
    train_hetero_gnn(dataset_name, target_table, masked_tables)
    hetero_graph = create_deepsnap_dataset(dataset_name, target_table, masked_tables, k=10)
    compute_hetero_gnn_embeddings(hetero_graph, dataset_name, target_table)
    embeddings = compute_hetero_gnn_embeddings(hetero_graph, dataset_name, target_table)
    np.save(f'data/embeddings/{target_table}_embeddings.npy', embeddings)

if __name__ == '__main__':
    main()