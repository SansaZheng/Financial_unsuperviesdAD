"""Native per-view exact ego training and consistent masked-field detection."""
import numpy as np
import torch
from finad_recent.himvh import HimvhNet
from .hypergraph import NativeContext, block_forward, assignments, masked, feature_groups


class HIMVHRepair:
    def __init__(self, width, task, seed=42, device='cpu', native=False, consistent=False, steps=800):
        self.width, self.task, self.seed, self.device = width, task, seed, device
        self.native, self.consistent, self.steps = native, consistent, steps
        self.groups = feature_groups(task, width)
        torch.manual_seed(seed)
        self.net = HimvhNet(width, 4).to(device)
        self.info = dict(native_context=native, consistent_masked_field=consistent, steps=steps, batch=128,
                         target_exposures=steps*128, hidden=256, layers=3, window=4, lr=.001,
                         decoder_normalization='original BatchNorm', supervised_labels=False)

    def fit(self, x, views, target_rows, node_ids=None):
        context = NativeContext(x, views, node_ids)
        clean = None if self.consistent else context.enhanced(x, np.arange(len(x)))
        opt = torch.optim.Adam(self.net.parameters(), lr=.001)
        rng = np.random.default_rng(self.seed+1234)
        losses = []
        self.net.train()
        for step in range(self.steps):
            targets = rng.choice(target_rows, size=min(128, len(target_rows)), replace=False)
            plan = context.plan(targets)
            needed = np.unique(np.concatenate([b[0][0] for b in plan]))
            salt = self.seed*1000003+step*10007
            if self.consistent:
                fields = assignments(context.ids, salt, len(self.groups))
                value = masked(x, fields, self.groups)
                enhanced = context.enhanced(value, needed)
            else:
                enhanced = clean[needed].copy()
                # Same20% marginal raw-coordinate corruption, with deterministic
                # node/feature hashing so context variants share target masks.
                keys = (context.ids[needed, None]*self.width+np.arange(self.width)).reshape(-1)
                mask = assignments(keys, salt, 5).reshape(len(needed), -1) == 0
                enhanced[:, :self.width][mask] = 0
            opt.zero_grad()
            prediction = block_forward(self.net, enhanced, plan, targets, self.device, needed)
            truth = torch.as_tensor(x[targets], device=self.device)
            if self.consistent:
                error = (prediction-truth).square()
                per_row = torch.zeros(len(targets), device=self.device)
                for g, cols in enumerate(self.groups):
                    rows = np.flatnonzero(fields[targets] == g)
                    if len(rows):
                        ti = torch.as_tensor(rows, device=self.device)
                        per_row[ti] = error[ti][:, cols].mean(1)
                loss = per_row.mean()
            else:
                loss = (prediction-truth).square().mean()
            if not torch.isfinite(loss): raise FloatingPointError('HIMVH repair nonfinite loss')
            loss.backward(); opt.step(); losses.append(float(loss.detach()))
        self.info.update(loss_first=losses[0], loss_last=losses[-1], context_rows=len(x),
                         train_targets=len(target_rows), hyperedges=[len(e) for e in views])
        return self

    def components(self, x, views, node_ids=None):
        ctx = NativeContext(x, views, node_ids)
        nodes = np.arange(len(x)); result = np.zeros((len(x), len(self.groups)), dtype=np.float64)
        self.net.eval()
        edges = [torch.as_tensor(e, device=self.device) for e in views]
        base = assignments(ctx.ids, self.seed+991, len(self.groups))
        passes = len(self.groups) if self.consistent else 1
        with torch.no_grad():
            for turn in range(passes):
                fields = (base+turn) % len(self.groups)
                value = masked(x, fields, self.groups) if self.consistent else x
                enhanced = torch.as_tensor(ctx.enhanced(value, nodes), device=self.device)
                predicted = self.net(enhanced, edges).cpu().numpy()
                error = (predicted-x)**2
                for g, cols in enumerate(self.groups):
                    rows = fields == g if self.consistent else np.ones(len(x), bool)
                    result[rows, g] = error[rows][:, cols].mean(1)
        return result

    def score_components(self, comp, flat=False):
        if flat or not self.consistent:
            return np.average(comp, axis=1, weights=[len(c) for c in self.groups])
        return comp.mean(1)

    def save(self, path):
        torch.save(dict(width=self.width, task=self.task, seed=self.seed, native=self.native, consistent=self.consistent,
                        steps=self.steps, state=self.net.state_dict(), info=self.info), path)

    @classmethod
    def load(cls, path, device='cpu'):
        d = torch.load(path, map_location=device, weights_only=True)
        model = cls(d['width'], d['task'], d['seed'], device, d['native'], d['consistent'], d['steps'])
        model.net.load_state_dict(d['state']); model.info = d['info']; return model
