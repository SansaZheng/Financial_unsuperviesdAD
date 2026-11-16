"""Independent HIMVH Eq.1-18 implementation with a denoising objective.

This is HIMVH-Zero, NOT the supervised WWW2026 classifier. No official source
was linked in the verified paper. Label discrepancy and embeddings are zero.
"""
from itertools import combinations
import numpy as np
from scipy import sparse
from scipy.special import softmax
import torch
from torch import nn
from .huge import aggregate


def hyperedges(keys, window=4):
    """Input rows already chronological, stable ties; short groups have no edge."""
    groups = {}
    for i, key in enumerate(keys):
        groups.setdefault(str(key), []).append(i)
    edges = []
    for group in groups.values():
        if len(group) >= window:
            edges.extend(np.lib.stride_tricks.sliding_window_view(np.asarray(group), window))
    return np.asarray(edges, dtype=np.int64).reshape(-1, window)


def neighborhoods(edges, n):
    if not len(edges):
        return sparse.csr_matrix((n, n), dtype=np.float64)
    row = np.repeat(edges, edges.shape[1], axis=1).reshape(-1)
    col = np.tile(edges, (1, edges.shape[1])).reshape(-1)
    keep = row != col
    a = sparse.coo_matrix((np.ones(keep.sum()), (row[keep], col[keep])), shape=(n, n)).tocsr()
    a.data[:] = 1
    return a


def discrepancies(x, views):
    """Eq.4-9: Jaccard and literal reverse-KL Eq.6, no label channel."""
    adj = [neighborhoods(e, len(x)) for e in views]
    sizes = [np.asarray(a.sum(1)).ravel() for a in adj]
    prob = []
    for a, size in zip(adj, sizes):
        mean = a@x/np.maximum(size[:, None], 1)
        mean[size == 0] = x[size == 0]
        prob.append(softmax(mean.astype(np.float64), axis=1).clip(1e-12))
    ids = np.zeros(len(x)); feats = np.zeros(len(x)); count = 0
    for a, b in combinations(range(len(views)), 2):
        intersection = np.asarray(adj[a].multiply(adj[b]).sum(1)).ravel()
        union = sizes[a]+sizes[b]-intersection
        ids += np.where(union > 0, 1-intersection/np.maximum(union, 1), 0)
        middle = (prob[a]+prob[b])/2
        feats += .5*(middle*(2*np.log(middle)-np.log(prob[a])-np.log(prob[b]))).sum(1)
        count += 1
    extra = np.column_stack([ids/max(count, 1), feats/max(count, 1), np.zeros(len(x))])
    return np.column_stack([x, extra]).astype(np.float32)


class NoveltyLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Linear(width, width, bias=False)

    def forward(self, h, edges):
        total = torch.zeros_like(h)
        count = h.new_zeros(len(h))
        projected = self.weight(h)
        for e in edges.split(4096):
            nodes = h[e]; center = nodes.mean(1, keepdim=True)
            variance = (nodes-center).square().mean(1)
            gate = torch.softmax(variance, dim=-1)
            novelty = ((nodes-center).square()*gate[:, None, :]).sum(-1)
            alpha = torch.softmax(novelty, dim=1)
            weighted = alpha[..., None]*projected[e]
            # Exclude the recipient's own message (Eq.14), without inventing
            # a second normalization not specified in the paper.
            message = weighted.sum(1, keepdim=True)-weighted
            total = total.index_add(0, e.reshape(-1), message.reshape(-1, h.shape[1]))
            count = count.index_add(0, e.reshape(-1), h.new_ones(e.numel()))
        # Nodes in several sliding hyperedges use a mean (explicit completion
        # of an aggregation detail not specified by Eq.14).
        return torch.sigmoid(h+total/count.clamp_min(1)[:, None])


class HimvhNet(nn.Module):
    def __init__(self, width, nviews, hidden=256, layers=3):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(width+3, hidden), nn.LayerNorm(hidden))
        self.branches = nn.ModuleList([nn.ModuleList([NoveltyLayer(hidden) for _ in range(layers)]) for _ in range(nviews)])
        self.gate = nn.Linear(nviews*hidden, nviews)
        self.mod_project = nn.Linear(hidden, hidden)
        self.modulation = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(nviews)])
        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(.1), nn.Linear(hidden, width))

    def forward(self, enhanced, views):
        initial = self.project(enhanced)
        representations = []
        for layers, edges in zip(self.branches, views):
            h = initial
            for layer in layers:
                h = layer(h, edges)
            representations.append(h)
        base = torch.softmax(self.gate(torch.relu(torch.cat(representations, dim=1))), dim=1)
        shared = torch.tanh(self.mod_project(torch.stack(representations).sum(0)))
        modulation = torch.cat([layer(shared) for layer in self.modulation], dim=1)
        weights = (base+modulation).abs()
        weights = weights/weights.sum(1, keepdim=True).clamp_min(1e-8)
        fused = (torch.stack(representations, dim=1)*weights[..., None]).sum(1)
        return self.decoder(fused)


class HIMVHZero:
    def __init__(self, width, nviews, seed=42, device='cpu', steps=200, hidden=256, layers=3):
        self.width, self.nviews, self.seed, self.device, self.steps = width, nviews, seed, device, steps
        self.hidden, self.layers = hidden, layers
        torch.manual_seed(seed)
        self.net = HimvhNet(width, nviews, hidden, layers).to(device)
        self.info = dict(method='himvh_zero_reconstruction', steps=steps, hidden=hidden, layers=layers, window=4,
                         beta=1, lr=.001, mask_probability=.2, training='denoising MSE', scoring='clean input reconstruction MSE',
                         label_discrepancy=0, label_embedding=0, official_code=False)

    def fit(self, x, views):
        enhanced = torch.as_tensor(discrepancies(x, views), device=self.device)
        target = torch.as_tensor(x, device=self.device)
        edges = [torch.as_tensor(e, device=self.device) for e in views]
        opt = torch.optim.Adam(self.net.parameters(), lr=.001)
        rng = torch.Generator(device=self.device).manual_seed(self.seed)
        losses = []
        self.net.train()
        for step in range(self.steps):
            masked = enhanced.clone()
            mask = torch.rand(target.shape, generator=rng, device=self.device) < .2
            masked[:, :self.width] = target.masked_fill(mask, 0)
            opt.zero_grad()
            reconstruction = self.net(masked, edges)
            # Same train target rows as the tabular comparators.
            sample = torch.randperm(len(x), generator=rng, device=self.device)[:512]
            loss = (reconstruction[sample]-target[sample]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('HIMVH-Zero nonfinite loss')
            loss.backward(); opt.step(); losses.append(float(loss.detach()))
        self.info.update(loss_first=losses[0], loss_last=losses[-1], train_hyperedges=[len(e) for e in views])
        return self

    def score(self, x, views):
        self.net.eval()
        enhanced = torch.as_tensor(discrepancies(x, views), device=self.device)
        edges = [torch.as_tensor(e, device=self.device) for e in views]
        with torch.no_grad():
            prediction = self.net(enhanced, edges)
            return (prediction-enhanced[:, :self.width]).square().mean(1).cpu().numpy().astype(np.float64)

    def save(self, path):
        torch.save(dict(width=self.width, nviews=self.nviews, seed=self.seed, steps=self.steps, hidden=self.hidden,
                        layers=self.layers, state=self.net.state_dict(), info=self.info), path)

    @classmethod
    def load(cls, path, device='cpu'):
        d = torch.load(path, map_location=device, weights_only=True)
        model = cls(d['width'], d['nviews'], d['seed'], device, d['steps'], d['hidden'], d['layers'])
        model.net.load_state_dict(d['state']); model.info = d['info']
        return model
