"""SD-FMM-Zero: author patch Transformer + unlabeled wavelet corruption.

Architecture adapted from ys-Dai/SD-FMM at 0d1a2158ee849fdc31a127bd049e622174f5c893
(MIT; third_party/SD_FMM_LICENSE). Unlike the release, no publisher anomaly
labels/normal fragment selection/validation F1 are used. Synthetic corruption
targets are self-generated, not real labels. No claim of exact paper reproduction.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pywt

LENGTH = 160


def sequence_starts(ids):
    ids = np.asarray(ids)
    changes = np.r_[True, ids[1:] != ids[:-1]]
    return np.maximum.accumulate(np.where(changes, np.arange(len(ids)), 0))


def contexts(x, starts, rows):
    """Causal inclusive windows, repeat first value at cold starts; no future."""
    rows = np.asarray(rows)
    index = rows[:, None]+np.arange(-LENGTH+1, 1)[None, :]
    index = np.maximum(index, starts[rows, None])
    return np.asarray(x[index], dtype=np.float32)


def synthesize(windows, rng):
    """Corrupt a 13-row fragment in wavelet space; mark only injected support.

All source windows may contain unknown anomalies; never call them verified
normal. Amplitudes are estimated solely from each unlabeled source fragment.
This generic multivariate generator replaces the author's few-shot domain
generator, which cannot be transferred without annotated anomaly examples.
"""
    altered = windows.copy()
    masks = np.zeros(windows.shape[:2], np.float32)
    original_fragments, changed_fragments = [], []
    for i in range(len(windows)):
        begin = int(rng.integers(0, LENGTH-13+1)); end = begin+13
        fragment = windows[i, begin:end].copy()
        coefficients = pywt.wavedec(fragment, 'db2', axis=0, level=2, mode='symmetric')
        amplitude = np.maximum(np.std(fragment, axis=0), .1)
        for level in range(1, len(coefficients)):
            coefficients[level] += rng.choice([-1, 1], size=coefficients[level].shape)*rng.uniform(1, 3, size=coefficients[level].shape)*amplitude
        changed = np.clip(pywt.waverec(coefficients, 'db2', axis=0, mode='symmetric')[:13], -8, 8)
        altered[i, begin:end] = changed
        masks[i, begin:end] = 1
        original_fragments.append(fragment); changed_fragments.append(changed)
    return altered, masks, np.asarray(original_fragments), np.asarray(changed_fragments)


class SDNet(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.embedding = nn.Linear(width*4, 64)
        position = torch.arange(5000, dtype=torch.float32)[:, None]
        div = torch.exp(torch.arange(0, 64, 2)*(-math.log(10000)/64))
        pe = torch.zeros(5000, 64)
        pe[:, 0::2] = torch.sin(position*div); pe[:, 1::2] = torch.cos(position*div)
        self.register_buffer('pe', pe[None])
        self.dropout = nn.Dropout(.2)
        layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=.2, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.output = nn.Linear(64, 4)
        self.fragment_expansion = nn.Linear(13, 160)
        self.extractor = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))

    def forward(self, x, features=False):
        if x.shape[1] == 13:
            x = self.fragment_expansion(x.transpose(1, 2)).transpose(1, 2)
        if x.shape[1] != LENGTH:
            raise ValueError('SD-FMM requires length 160 or fragment length 13')
        x = x.reshape(len(x), 40, -1)
        z = self.encoder(self.dropout(self.embedding(x)+self.pe[:, :40]))
        if features:
            return self.extractor(z.mean(1))
        return self.output(z).reshape(len(x), LENGTH)  # logits for stable BCE


class SDFMMZero:
    def __init__(self, width, seed=42, device='cpu', steps=400):
        self.width, self.seed, self.device, self.steps = width, seed, device, steps
        torch.manual_seed(seed)
        self.net = SDNet(width).to(device)
        self.info = dict(method='sdfmm_zero_synthetic', steps=steps, batch=16, lr=.001,
                         context=160, patch=4, hidden=64, heads=4, layers=2, feedforward=128,
                         dropout=.2, contrastive_weight=.5, temperature=.5, fragment_length=13,
                         generator='db2 unlabeled local detail corruption', causal=True,
                         upstream_commit='0d1a2158ee849fdc31a127bd049e622174f5c893')

    def fit(self, x, starts, indices):
        rng = np.random.default_rng(self.seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=.001)
        losses = []
        self.net.train()
        for step in range(self.steps):
            selected = rng.choice(indices, size=min(16, len(indices)), replace=False)
            original = contexts(x, starts, selected)
            changed, mask, normal_fragment, changed_fragment = synthesize(original, rng)
            batch = torch.as_tensor(np.concatenate([original, changed]), device=self.device)
            target = torch.as_tensor(np.concatenate([np.zeros_like(mask), mask]), device=self.device)
            opt.zero_grad()
            prediction = self.net(batch)
            # Balanced original/corrupted windows. No label-dependent normal filtering.
            main = F.binary_cross_entropy_with_logits(prediction, target)
            fragments = torch.as_tensor(np.concatenate([normal_fragment, changed_fragment]), device=self.device)
            a, b = self.net(fragments, features=True).chunk(2)
            similarity = F.cosine_similarity(a, b, dim=1)/.5
            contrastive = -torch.log(1-torch.sigmoid(similarity)+1e-8).mean()
            loss = main+.5*contrastive
            if not torch.isfinite(loss):
                raise FloatingPointError('SD-FMM-Zero nonfinite loss')
            loss.backward(); opt.step(); losses.append(float(loss.detach()))
        self.info.update(loss_first=losses[0], loss_last=losses[-1])
        return self

    def score(self, x, starts, rows, batch=256):
        self.net.eval()
        scores = []
        with torch.no_grad():
            for offset in range(0, len(rows), batch):
                value = torch.as_tensor(contexts(x, starts, rows[offset:offset+batch]), device=self.device)
                # Only the last timestamp is evaluated; earlier positions see no
                # future relative to this target, even with bidirectional attention.
                scores.append(self.net(value)[:, -1].sigmoid().cpu().numpy())
        return np.concatenate(scores).astype(np.float64)

    def save(self, path):
        torch.save(dict(width=self.width, seed=self.seed, steps=self.steps, state=self.net.state_dict(), info=self.info), path)

    @classmethod
    def load(cls, path, device='cpu'):
        d = torch.load(path, map_location=device, weights_only=True)
        model = cls(d['width'], d['seed'], device, d['steps'])
        model.net.load_state_dict(d['state']); model.info = d['info']
        return model
