"""Constraint-preserving financial synthesis and target-aligned BCE ablations."""
import numpy as np
import pywt
import torch
from torch.nn import functional as F
from finad_recent.sdfmm import SDFMMZero, contexts, synthesize


def point_features(value):
    value = np.asarray(value, dtype=np.float64)
    parts = [value, np.diff(value, prepend=value[:1])]
    for width in [4, 16, 32]:
        windows = np.lib.stride_tricks.sliding_window_view(np.pad(value, (width-1, 0), constant_values=np.nan), width)
        parts.extend([np.nanmean(windows, 1), np.nanstd(windows, 1), np.nanmax(windows, 1)-np.nanmin(windows, 1)])
    return np.column_stack(parts).astype(np.float32)


def gain_curve(length, rng):
    # db2 detail modulation of a positive ramp, not independent feature noise.
    noise = rng.normal(size=16)
    coefficients = pywt.wavedec(noise, 'db2', level=2, mode='symmetric')
    coefficients[0] *= 0
    detail = np.abs(pywt.waverec(coefficients, 'db2', mode='symmetric')[:length])
    detail /= max(float(detail.max()), 1e-8)
    profile = np.linspace(.1, 1, length)+.25*detail
    amplitude = float(rng.uniform(.1, 1.))
    sign = int(rng.choice([-1, 1]))
    return np.exp(sign*amplitude*profile), np.exp(amplitude*profile)


def coherent(raw_windows, elapsed, task, rng):
    altered = raw_windows.copy()
    for i in range(len(altered)):
        valid = min(160, int(elapsed[i])+1); begin_valid = 160-valid
        length = min(13, valid); start = 160-length
        gain, activity = gain_curve(length, rng)
        if task.startswith('sapienza_'):
            # Preserve clock coordinates and price mean<=maximum. Activity
            # bursts are shared across count/volume fields; price can rise/fall.
            altered[i, start:, :5] *= activity[:, None]
            altered[i, start:, 5] *= gain
            price_scale = max(float(np.median(np.abs(raw_windows[i, begin_valid:, 6])))*.01, 1e-6)
            altered[i, start:, 6:8] = np.maximum(0, (raw_windows[i, start:, 6:8]+price_scale)*gain[:, None]-price_scale)
            if begin_valid > 0 and start == begin_valid:
                altered[i, :begin_valid] = altered[i, begin_valid]
        else:
            price = raw_windows[i, begin_valid:, 0].copy()
            scale = max(float(np.std(price)), .05*float(np.median(np.abs(price))), .01)
            price[-length:] = np.maximum(0, (price[-length:]+scale)*gain-scale)
            rebuilt = point_features(price)
            altered[i, start:] = rebuilt[-length:]
            if begin_valid > 0 and start == begin_valid:
                altered[i, :begin_valid] = rebuilt[0]
    return altered


def transform_windows(raw, pre):
    return pre.transform(raw.reshape(-1, raw.shape[-1])).reshape(raw.shape)


class SDRepair(SDFMMZero):
    def __init__(self, width, task, seed=42, device='cpu', endpoint=False, generator=False, contrastive=.5, steps=400):
        super().__init__(width, seed, device, steps)
        self.task, self.endpoint, self.generator, self.contrastive = task, endpoint, generator, contrastive
        self.info.update(task=task, endpoint_loss=endpoint, coherent_generator=generator, contrastive_weight=contrastive,
                         sampling_rng='seed+1234 separate from augmentation seed+10000',
                         generator='financial shared gain + db2 modulation' if generator else 'legacy per-column db2')

    def fit(self, raw, starts, indices, pre):
        sampling = np.random.default_rng(self.seed+1234); augmentation = np.random.default_rng(self.seed+10000)
        opt = torch.optim.Adam(self.net.parameters(), lr=.001)
        losses = []; mains = []; contrasts = []; visible = []
        self.net.train()
        for _ in range(self.steps):
            rows = sampling.choice(indices, size=min(16, len(indices)), replace=False)
            original_raw = contexts(raw, starts, rows)
            original = transform_windows(original_raw, pre)
            if self.generator:
                changed_raw = coherent(original_raw, rows-starts[rows], self.task, augmentation)
                changed = transform_windows(changed_raw, pre)
                mask = (np.max(np.abs(changed-original), axis=-1) > 1e-6).astype(np.float32)
                normal_fragment, changed_fragment = original[:, -13:], changed[:, -13:]
            else:
                changed, mask, normal_fragment, changed_fragment = synthesize(original, augmentation)
            batch = torch.as_tensor(np.concatenate([original, changed]), device=self.device)
            target = torch.as_tensor(np.concatenate([np.zeros_like(mask), mask]), device=self.device)
            opt.zero_grad(); prediction = self.net(batch)
            main = F.binary_cross_entropy_with_logits(prediction[:, -1], target[:, -1]) if self.endpoint else F.binary_cross_entropy_with_logits(prediction, target)
            fragments = torch.as_tensor(np.concatenate([normal_fragment, changed_fragment]), device=self.device)
            a, b = self.net(fragments, features=True).chunk(2)
            per_pair = -torch.log(1-torch.sigmoid(F.cosine_similarity(a, b, dim=1)/.5)+1e-8)
            keep = torch.as_tensor(np.max(np.abs(normal_fragment-changed_fragment), axis=(1, 2)) > 1e-6, device=self.device)
            contrast = per_pair[keep].mean() if keep.any() else per_pair.sum()*0
            loss = main+self.contrastive*contrast
            if not torch.isfinite(loss): raise FloatingPointError('SD repair nonfinite loss')
            loss.backward(); opt.step()
            losses.append(float(loss.detach())); mains.append(float(main.detach())); contrasts.append(float(contrast.detach())); visible.append(float(mask[:, -1].mean()))
        self.info.update(loss_first=losses[0], loss_last=losses[-1], main_first=mains[0], main_last=mains[-1],
                         contrast_first=contrasts[0], contrast_last=contrasts[-1], visible_endpoint_fraction=float(np.mean(visible)))
        return self

    def score_raw(self, raw, starts, rows, pre, batch=256):
        self.net.eval(); scores = []
        with torch.no_grad():
            for i in range(0, len(rows), batch):
                x = transform_windows(contexts(raw, starts, rows[i:i+batch]), pre)
                scores.append(self.net(torch.as_tensor(x, device=self.device))[:, -1].sigmoid().cpu().numpy())
        return np.concatenate(scores).astype(np.float64)

    def save(self, path):
        torch.save(dict(width=self.width, task=self.task, seed=self.seed, endpoint=self.endpoint, generator=self.generator,
                        contrastive=self.contrastive, steps=self.steps, state=self.net.state_dict(), info=self.info), path)

    @classmethod
    def load(cls, path, device='cpu'):
        d = torch.load(path, map_location=device, weights_only=True)
        m = cls(d['width'], d['task'], d['seed'], device, d['endpoint'], d['generator'], d['contrastive'], d['steps'])
        m.net.load_state_dict(d['state']); m.info = d['info']; return m
