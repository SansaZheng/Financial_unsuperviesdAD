"""Exact per-view receptive-field minibatches, preserving full native windows."""
from itertools import combinations
import numpy as np
from scipy import sparse
from scipy.special import softmax
import torch
from finad_recent.himvh import neighborhoods


class NativeContext:
    def __init__(self, x, views, node_ids=None):
        self.x = x; self.views = views; self.n = len(x)
        self.ids = np.arange(len(x)) if node_ids is None else np.asarray(node_ids)
        self.adj = [neighborhoods(e, len(x)) for e in views]
        self.incidence = []
        for e in views:
            row = e.reshape(-1); col = np.repeat(np.arange(len(e)), e.shape[1])
            self.incidence.append(sparse.csr_matrix((np.ones(len(row), dtype=np.int8), (row, col)), shape=(len(x), len(e))))
        sizes = [np.asarray(a.sum(1)).ravel() for a in self.adj]
        identity = np.zeros(len(x)); count = 0
        for a, b in combinations(range(len(views)), 2):
            intersection = np.asarray(self.adj[a].multiply(self.adj[b]).sum(1)).ravel()
            union = sizes[a]+sizes[b]-intersection
            identity += np.where(union > 0, 1-intersection/np.maximum(union, 1), 0); count += 1
        self.identity = identity/max(count, 1)
        self.sizes = sizes

    def enhanced(self, x, nodes):
        prob = []
        for a, size in zip(self.adj, self.sizes):
            mean = a[nodes]@x/np.maximum(size[nodes, None], 1)
            isolated = size[nodes] == 0; mean[isolated] = x[nodes[isolated]]
            prob.append(softmax(mean.astype(np.float64), axis=1).clip(1e-12))
        feat = np.zeros(len(nodes)); count = 0
        for a, b in combinations(range(len(self.views)), 2):
            middle = (prob[a]+prob[b])/2
            feat += .5*(middle*(2*np.log(middle)-np.log(prob[a])-np.log(prob[b]))).sum(1); count += 1
        return np.column_stack([x[nodes], self.identity[nodes], feat/max(count, 1), np.zeros(len(nodes))]).astype(np.float32)

    def plan(self, targets, layers=3):
        targets = np.asarray(targets); branches = []
        for e, incidence in zip(self.views, self.incidence):
            nodes = np.unique(targets); reverse = []
            for _ in range(layers):
                selected_edges = np.unique(incidence[nodes].indices)
                members = e[selected_edges]
                inputs = np.union1d(nodes, members.reshape(-1))
                reverse.append((inputs, nodes, np.searchsorted(inputs, members)))
                nodes = inputs
            branches.append(list(reversed(reverse)))
        return branches


def block_forward(net, enhanced, branches, targets, device, feature_nodes=None):
    representations = []
    for layers, blocks in zip(net.branches, branches):
        inputs = blocks[0][0]
        features = inputs if feature_nodes is None else np.searchsorted(feature_nodes, inputs)
        h = net.project(torch.as_tensor(enhanced[features], device=device))
        for layer, (input_ids, output_ids, e) in zip(layers, blocks):
            h = layer(h, torch.as_tensor(e, device=device))
            h = h[torch.as_tensor(np.searchsorted(input_ids, output_ids), device=device)]
        order = np.searchsorted(blocks[-1][1], targets)
        representations.append(h[torch.as_tensor(order, device=device)])
    base = torch.softmax(net.gate(torch.relu(torch.cat(representations, dim=1))), dim=1)
    shared = torch.tanh(net.mod_project(torch.stack(representations).sum(0)))
    mod = torch.cat([layer(shared) for layer in net.modulation], dim=1)
    weights = (base+mod).abs(); weights = weights/weights.sum(1, keepdim=True).clamp_min(1e-8)
    return net.decoder((torch.stack(representations, dim=1)*weights[..., None]).sum(1))


def feature_groups(task, width):
    if task == 'banksim': groups = [np.arange(0, 1), np.arange(1, 9), np.arange(9, 13), np.arange(13, 45), np.arange(45, 61)]
    elif task == 's_ffsd': groups = [np.arange(0, 1), np.arange(1, 17), np.arange(17, 33), np.arange(33, 49)]
    else: raise ValueError(task)
    assert np.array_equal(np.concatenate(groups), np.arange(width))
    return groups


def assignments(ids, salt, groups):
    z = np.asarray(ids, dtype=np.uint64)+np.uint64(salt & ((1 << 64)-1))+np.uint64(0x9e3779b97f4a7c15)
    z = (z^(z >> np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
    z = (z^(z >> np.uint64(27)))*np.uint64(0x94d049bb133111eb)
    return ((z^(z >> np.uint64(31))) % np.uint64(groups)).astype(np.int64)


def masked(x, fields, groups):
    result = x.copy()
    for i, cols in enumerate(groups):
        result[np.ix_(np.flatnonzero(fields == i), cols)] = 0
    return result
