"""Label-free role-semantic alignment for a native bipartite payment graph."""
import numpy as np
import torch
from finad.preprocessing import Preprocessor
from finad_recent.huge import similarity


def canonical(raw, role):
    # Corresponding count/mean/std/max of outgoing payments for payers and
    # incoming payments for payees. Role one-hots are not behavioral attributes.
    out = np.empty((*raw.shape[:2], 4), dtype=np.float32)
    out[:, role] = raw[:, role][:, :, [2, 4, 6, 8]]
    out[:, ~role] = raw[:, ~role][:, :, [3, 5, 7, 9]]
    return out


class RoleNormalizer:
    def fit(self, raw, role):
        x = canonical(raw, role)
        self.scalers = [Preprocessor().fit(x[:, role == r].reshape(-1, 4)) for r in [False, True]]
        return self

    def transform(self, raw, role):
        x = canonical(raw, role); result = np.empty_like(x)
        for r, pre in zip([False, True], self.scalers):
            result[:, role == r] = pre.transform(x[:, role == r].reshape(-1, 4)).reshape(4, -1, 4)
        return result.transpose(1, 0, 2).reshape(raw.shape[1], 16)

    def save(self, folder):
        for i, pre in enumerate(self.scalers): pre.save(folder/f'role_{i}.npz')

    @classmethod
    def load(cls, folder):
        o = cls(); o.scalers = [Preprocessor.load(folder/f'role_{i}.npz') for i in range(2)]; return o


def score_without_self(model, x, edges):
    # Recompute directly: a zero embedding has self-cosine0, not1, so an
    # unconditional algebraic +1 correction would be invalid for that case.
    model.net.eval()
    with torch.no_grad():
        value=torch.as_tensor(x,device=model.device)
        e=torch.as_tensor(edges,device=model.device,dtype=torch.long)
        return -similarity(model.net.mlp(value),e).cpu().numpy().astype(np.float64)
