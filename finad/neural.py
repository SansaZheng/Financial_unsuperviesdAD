"""MLP adaptations of reconstruction AE, Gaussian VAE and one-class Deep SVDD.

These reproduce the core objectives on numeric financial records, not the
original image architectures, original training schedules, or published scores.
"""
import math
import numpy as np
import torch
from torch import nn

METHODS = ('ae', 'vae', 'deep_svdd')


class Autoencoder(nn.Module):
    def __init__(self, features, latent):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(features, 128), nn.ReLU(), nn.Linear(128, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, 128), nn.ReLU(), nn.Linear(128, features))

    def forward(self, x):
        return self.decoder(self.encoder(x))


class VariationalAutoencoder(nn.Module):
    def __init__(self, features, latent):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(features, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.mean = nn.Linear(64, latent); self.log_variance = nn.Linear(64, latent)
        self.decoder = nn.Sequential(nn.Linear(latent, 64), nn.ReLU(), nn.Linear(64, 128),
                                     nn.ReLU(), nn.Linear(128, features))

    def encode(self, x):
        hidden = self.encoder(x)
        return self.mean(hidden), self.log_variance(hidden).clamp(-12, 8)

    @staticmethod
    def kl(mean, log_variance):
        return .5*torch.sum(mean.square()+log_variance.exp()-1-log_variance, dim=1)


class SVDDNetwork(nn.Module):
    def __init__(self, features, latent):
        super().__init__()
        # Bias-free layers, unbounded LeakyReLU, no batch normalization.
        self.encoder = nn.Sequential(nn.Linear(features, 128, bias=False), nn.LeakyReLU(.1),
            nn.Linear(128, 64, bias=False), nn.LeakyReLU(.1), nn.Linear(64, latent, bias=False))

    def forward(self, x):
        return self.encoder(x)


def make_network(method, features, latent):
    return {'ae': Autoencoder, 'vae': VariationalAutoencoder, 'deep_svdd': SVDDNetwork}[method](features, latent)


class NeuralDetector:
    def __init__(self, method, seed=42, device='cpu', steps=400, pretrain_steps=200):
        if method not in METHODS:
            raise ValueError('Unknown neural method')
        self.method, self.seed, self.device = method, seed, device
        self.steps, self.pretrain_steps = steps, pretrain_steps

    def _batch(self, z, rng):
        indices = rng.integers(0, len(z), size=min(256, len(z)))
        return torch.as_tensor(z[indices], device=self.device)

    def fit(self, z):
        torch.manual_seed(self.seed)
        self.features = z.shape[1]; self.latent = min(16, max(1, self.features//2))
        self.model = make_network(self.method, self.features, self.latent).to(self.device)
        rng = np.random.default_rng(self.seed)
        self.center = None; self.losses = []
        if self.method == 'deep_svdd':
            # Pretrain the SAME bias-free encoder by reconstructing unlabelled inputs.
            decoder = nn.Sequential(nn.Linear(self.latent, 64, bias=False), nn.LeakyReLU(.1),
                nn.Linear(64, 128, bias=False), nn.LeakyReLU(.1), nn.Linear(128, self.features, bias=False)).to(self.device)
            opt = torch.optim.Adam(list(self.model.parameters())+list(decoder.parameters()), lr=1e-3, weight_decay=5e-4)
            for step in range(self.pretrain_steps):
                batch = self._batch(z, rng)
                loss = (decoder(self.model(batch))-batch).square().mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            self.model.eval()
            with torch.no_grad():
                total = torch.zeros(self.latent, device=self.device)
                for start in range(0, len(z), 2048):
                    total += self.model(torch.as_tensor(z[start:start+2048], device=self.device)).sum(0)
                self.center = total/len(z)
                self.center[torch.abs(self.center) < .1] = torch.where(
                    self.center[torch.abs(self.center) < .1] < 0, -.1, .1)
            del opt, decoder
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3,
                                     weight_decay=5e-7 if self.method == 'deep_svdd' else 0.)
        for step in range(self.steps):
            batch = self._batch(z, rng)
            if self.method == 'ae':
                loss = (self.model(batch)-batch).square().mean()
            elif self.method == 'vae':
                mean, log_variance = self.model.encode(batch)
                latent = mean+torch.exp(.5*log_variance)*torch.randn_like(mean)
                reconstruction = self.model.decoder(latent)
                nll = .5*((batch-reconstruction).square()+math.log(2*math.pi)).sum(1)
                loss = (nll+self.model.kl(mean, log_variance)).mean()
            else:
                loss = (self.model(batch)-self.center).square().sum(1).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite neural training objective')
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            if step % 100 == 0 or step == self.steps-1:
                self.losses.append({'step': step+1, 'loss': float(loss.detach().cpu())})
        self.model.eval()
        # Fixed antithetic common-random-number samples: independent of query
        # ordering/chunk size, unlike sampling a fresh posterior at each call.
        eps = np.random.default_rng(1729).normal(size=(4, self.latent)).astype(np.float32)
        self.eps = torch.as_tensor(np.concatenate([eps, -eps]), device=self.device)
        self.info = dict(method=self.method, features=self.features, latent=self.latent, steps=self.steps,
                         pretrain_steps=self.pretrain_steps if self.method == 'deep_svdd' else 0,
                         objective={'ae': 'mean_squared_reconstruction', 'vae': 'negative_gaussian_ELBO',
                                    'deep_svdd': 'one_class_fixed_center'}[self.method],
                         mc_samples=8 if self.method == 'vae' else 0, losses=self.losses)
        return self

    def score(self, z):
        self.model.eval()
        with torch.inference_mode():
            x = torch.as_tensor(z, device=self.device)
            if self.method == 'ae':
                score = (self.model(x)-x).square().mean(1)
            elif self.method == 'deep_svdd':
                score = (self.model(x)-self.center).square().sum(1)
            else:
                mean, log_variance = self.model.encode(x)
                sigma = torch.exp(.5*log_variance)
                nll = torch.zeros(len(x), device=self.device)
                for eps in self.eps:
                    prediction = self.model.decoder(mean+sigma*eps)
                    nll += .5*((x-prediction).square()+math.log(2*math.pi)).sum(1)/len(self.eps)
                score = nll+self.model.kl(mean, log_variance)
            return score.cpu().numpy().astype(np.float64)

    def save(self, path):
        torch.save({'method': self.method, 'seed': self.seed, 'features': self.features, 'latent': self.latent,
                    'steps': self.steps, 'pretrain_steps': self.pretrain_steps,
                    'state_dict': self.model.state_dict(), 'center': self.center,
                    'eps': self.eps, 'info': self.info}, path)

    @classmethod
    def load(cls, path, device='cpu'):
        payload = torch.load(path, map_location=device, weights_only=True)
        obj = cls(payload['method'], payload['seed'], device, payload['steps'], payload['pretrain_steps'])
        obj.features, obj.latent = payload['features'], payload['latent']
        obj.model = make_network(obj.method, obj.features, obj.latent).to(device)
        obj.model.load_state_dict(payload['state_dict']); obj.model.eval()
        obj.center, obj.eps, obj.info = payload['center'], payload['eps'], payload['info']
        return obj
