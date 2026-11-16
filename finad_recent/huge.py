"""HUGE PyTorch port, checked against the pinned MIT-licensed author formulas.

Upstream: CampanulaBells/HUGE-GAD bdf8d2aec6455b48d7322c0713e31d2f6fecc1b7.
See third_party/HUGE_LICENSE. Transfer train-graph weights to held-out graphs;
the paper instead trains transductively on each benchmark graph.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def bidirected_union(edges, n):
    e = np.unique(np.concatenate(edges, axis=1).T, axis=0)
    e = e[e[:, 0] != e[:, 1]]
    return np.unique(np.concatenate([e, e[:, ::-1]], axis=0), axis=0).T


def aggregate(values, destination, n):
    return values.new_zeros((n, *values.shape[1:])).index_add(0, destination, values)


def halo(x, edges):
    """Author coeff-weighted HALO, including both original epsilons."""
    a, b = x[edges[0]], x[edges[1]]
    coefficient = (a-b).abs()+1e-9
    a, b = coefficient*a, coefficient*b
    h = (a-b).square().sum(1).sqrt()/(a.square().sum(1)+b.square().sum(1)+1e-6).sqrt()
    count = torch.bincount(edges[1], minlength=len(x))
    result = aggregate(h, edges[1], len(x))/count.clamp_min(1)
    return torch.where(count > 0, result, -torch.ones_like(result))


def loop_edges(edges, n):
    diagonal = torch.arange(n, device=edges.device).repeat(2, 1)
    return torch.cat([edges, diagonal], dim=1)


def similarity(z, edges, negative=False):
    a, b = edges
    norm = F.normalize(z, dim=1)
    sims = (norm[a]*norm[b]).sum(1)
    count = torch.bincount(b, minlength=len(z)).to(z.dtype)
    summed = aggregate(sims, b, len(z))
    positive = summed/count.clamp_min(1)
    if not negative:
        return positive
    # Algebraically identical to summing the full non-neighbor cosine matrix.
    total = norm@norm.sum(0)
    non_neighbors = len(z)-count
    neg = (total-summed)/non_neighbors.clamp_min(1)
    return positive, neg


def rank_loss(pred, guidance):
    difference = pred[None, :]-pred[:, None]
    target = (guidance[None, :]-guidance[:, None] > 0).to(pred.dtype)
    return ((1-target)*difference+F.softplus(-difference)).mean()


def alignment(mlp, gnn, edges):
    a, b = edges
    keep = a != b
    a, b = a[keep], b[keep]
    ip = ((1+F.cosine_similarity(mlp[a], mlp[b]))/2).clamp_min(1e-7)
    target = ((1+F.cosine_similarity(gnn[a].detach(), gnn[b].detach()))/2).clamp_min(1e-7)
    degree = torch.bincount(b, minlength=len(mlp)).clamp_min(1)
    return (target*(target.log()-ip.log())/degree[b]).sum()/len(mlp)


class HugeNet(nn.Module):
    def __init__(self, width, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.PReLU(), nn.Linear(hidden, hidden), nn.PReLU())
        self.gcn_weight = nn.Parameter(torch.empty(hidden, hidden))
        self.gcn_bias = nn.Parameter(torch.zeros(hidden))
        self.activation = nn.PReLU()
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.gcn_weight)

    def forward(self, x, edges):
        z = self.mlp(x)
        a, b = edges
        degree = torch.bincount(a, minlength=len(x)).to(x.dtype).clamp_min(1).rsqrt()
        message = (z@self.gcn_weight)*degree[:, None]
        g = aggregate(message[a], b, len(x))*degree[:, None]+self.gcn_bias
        return z, self.activation(g)


class HUGE:
    def __init__(self, width, seed=42, device='cpu', steps=300):
        self.width, self.seed, self.device, self.steps = width, seed, device, steps
        torch.manual_seed(seed)
        self.net = HugeNet(width).to(device)
        self.info = dict(method='huge_financial_inductive', steps=steps, hidden=128, lr=.0005,
                         kd_param=.5, rank_batch=512, upstream_commit='bdf8d2aec6455b48d7322c0713e31d2f6fecc1b7')

    def fit(self, x, edges):
        x = torch.as_tensor(x, device=self.device)
        e = torch.as_tensor(edges, device=self.device, dtype=torch.long)
        h = halo(x, e).detach(); e = loop_edges(e, len(x))
        opt = torch.optim.Adam(self.net.parameters(), lr=.0005)
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        losses = []
        self.net.train()
        for step in range(self.steps):
            opt.zero_grad()
            mlp, gnn = self.net(x, e)
            sample = torch.randperm(len(x), generator=generator, device=self.device)[:512]
            loss = x.new_zeros(())
            for z in (mlp, gnn):
                pos, neg = similarity(z, e, True)
                pred = -pos
                loss = loss+(rank_loss(pred[sample], h[sample])+F.softplus(pred+neg).mean())/2
            loss = loss+.5*alignment(mlp, gnn, e)
            if not torch.isfinite(loss):
                raise FloatingPointError('HUGE nonfinite loss')
            loss.backward(); opt.step(); losses.append(float(loss.detach()))
        self.info['loss_first'], self.info['loss_last'] = losses[0], losses[-1]
        return self

    def score(self, x, edges):
        self.net.eval()
        with torch.no_grad():
            x = torch.as_tensor(x, device=self.device)
            e = loop_edges(torch.as_tensor(edges, device=self.device, dtype=torch.long), len(x))
            # Author fraud score is NEGATIVE neighbor cosine, with self-loops.
            return -similarity(self.net.mlp(x), e).cpu().numpy().astype(np.float64)

    def save(self, path):
        torch.save(dict(width=self.width, seed=self.seed, steps=self.steps, state=self.net.state_dict(), info=self.info), path)

    @classmethod
    def load(cls, path, device='cpu'):
        d = torch.load(path, map_location=device, weights_only=True)
        model = cls(d['width'], d['seed'], device, d['steps'])
        model.net.load_state_dict(d['state']); model.info = d['info']
        return model
