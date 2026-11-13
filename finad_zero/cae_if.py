"""Independent CAE -> Isolation Forest core adaptation, not unavailable paper architecture."""
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import IsolationForest
import joblib


class CAE(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(),
                                     nn.Conv1d(16, 4, 3, stride=2, padding=1), nn.ReLU())
        self.decoder = nn.Sequential(nn.Conv1d(4, 16, 3, padding=1), nn.ReLU(),
                                     nn.Upsample(size=features, mode='linear', align_corners=False),
                                     nn.Conv1d(16, 1, 3, padding=1))

    def forward(self, x):
        return self.decoder(self.encoder(x[:, None]))[:, 0]


class CAEIF:
    def __init__(self, seed=42, device='cpu', steps=400):
        self.seed, self.device, self.steps = seed, device, steps

    def encode(self, z):
        self.model.eval()
        with torch.inference_mode():
            return np.concatenate([self.model.encoder(torch.as_tensor(z[i:i+1024], device=self.device)[:, None])
                                   .flatten(1).cpu().numpy() for i in range(0, len(z), 1024)])

    def fit(self, z):
        torch.manual_seed(self.seed)
        self.features = z.shape[1]
        self.model = CAE(self.features).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        rng = np.random.default_rng(self.seed)
        self.losses = []
        self.model.train()
        for step in range(self.steps):
            b = torch.as_tensor(z[rng.integers(len(z), size=min(256, len(z)))], device=self.device)
            loss = (self.model(b)-b).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite CAE objective')
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if step % 100 == 0 or step+1 == self.steps:
                self.losses.append({'step': step+1, 'loss': float(loss.detach().cpu())})
        self.forest = IsolationForest(n_estimators=100, max_samples=min(256, len(z)),
                                      random_state=self.seed, contamination='auto', n_jobs=2).fit(self.encode(z))
        self.info = dict(method='cae_if_core', steps=self.steps, input_axis='fixed engineered feature order',
                         channels=[1, 16, 4], kernel=3, stride=2, latent=self.encode(z[:1]).shape[1],
                         losses=self.losses, n_estimators=100, max_samples=min(256, len(z)))
        return self

    def score(self, z):
        return -self.forest.score_samples(self.encode(z))

    def save(self, path):
        torch.save({'features': self.features, 'seed': self.seed, 'steps': self.steps,
                    'state': self.model.state_dict(), 'info': self.info}, str(path)+'.pt')
        joblib.dump(self.forest, str(path)+'.joblib')

    @classmethod
    def load(cls, path, device='cpu'):
        p = torch.load(str(path)+'.pt', weights_only=True, map_location=device)
        obj = cls(p['seed'], device, p['steps']); obj.features = p['features']
        obj.model = CAE(obj.features).to(device); obj.model.load_state_dict(p['state'])
        obj.forest = joblib.load(str(path)+'.joblib'); obj.info = p['info']
        return obj
