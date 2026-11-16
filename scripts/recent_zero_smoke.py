"""Synthetic numerical, pinned-author architecture and resource qualification."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import numpy as np
import torch
from torch import nn
from finad_recent.data import ROOT, write
from finad_recent.huge import HUGE, bidirected_union
from finad_recent.himvh import HIMVHZero, hyperedges
from finad_recent.sdfmm import SDFMMZero, SDNet, sequence_starts


def upstream_sd_equivalence(device):
    commit = '0d1a2158ee849fdc31a127bd049e622174f5c893'
    content = urllib.request.urlopen(f'https://raw.githubusercontent.com/ys-Dai/SD-FMM/{commit}/train.py', timeout=60).read()
    expected = json.loads((ROOT/'docs/recent_zero_upstream.json').read_text())['SD-FMM']['files']['train.py']
    assert hashlib.sha256(content).hexdigest() == expected
    # Execute only the two public class definitions, never top-level data reads
    # or the author's training/evaluation entry points.
    tree = ast.parse(content.decode())
    selected = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ['PositionalEncoding', 'ContrastiveTimeSeriesTransformer']]
    assert len(selected) == 2
    namespace = dict(torch=torch, nn=nn, patch_size=4, len_model=40)
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<pinned-author-classes>', 'exec'), namespace)
    torch.manual_seed(9)
    author = namespace['ContrastiveTimeSeriesTransformer'](dropout=.2).to(device).eval()
    port = SDNet(1).to(device).eval()
    state = {}
    for key, value in author.state_dict().items():
        renamed = key.replace('pos_encoder.pe', 'pe').replace('transformer_encoder.', 'encoder.').replace('output_layer.', 'output.').replace('feature_extractor.', 'extractor.')
        state[renamed] = value
    port.load_state_dict(state)
    maximum = 0.
    with torch.no_grad():
        for n in [13, 160]:
            x = torch.randn(3, n, 1, device=device)
            for features in [False, True]:
                a = author(x, get_features=features)
                b = port(x, features=features)
                if not features:
                    b = b.sigmoid()
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
                maximum = max(maximum, float((a-b).abs().max()))
    return dict(status='passed', train_source_sha256=expected, forward_max_abs=maximum,
                lengths=[13, 160], branches=['risk', 'contrastive'])


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu'); p.add_argument('--resource-profile', action='store_true')
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if args.device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    result = dict(upstream_sd=upstream_sd_equivalence(args.device), models={})
    rng = np.random.default_rng(7)
    for name in ['huge', 'himvh', 'sdfmm']:
        start = time.monotonic()
        if args.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        n = 8192 if args.resource_profile and name == 'himvh' else 512
        width = 61 if name == 'himvh' else 12
        x = rng.normal(size=(n, width)).astype(np.float32)
        if name == 'huge':
            pairs = np.stack([np.arange(n), np.roll(np.arange(n), 1)])
            e = bidirected_union([pairs], n)
            model = HUGE(width, device=args.device, steps=2).fit(x, e)
            inputs = (x, e)
        elif name == 'himvh':
            views = [hyperedges(rng.integers(0, 30, n)) for _ in range(4)]
            model = HIMVHZero(width, 4, device=args.device, steps=2).fit(x, views)
            # Test inference at the largest real transaction phase (label free).
            if args.resource_profile:
                x = rng.normal(size=(120000, width)).astype(np.float32)
                views = [hyperedges(rng.integers(0, 50, len(x))) for _ in range(4)]
            inputs = (x, views)
        else:
            starts = sequence_starts(np.repeat(np.arange(n//16), 16))
            model = SDFMMZero(width, device=args.device, steps=2).fit(x, starts, np.arange(n))
            inputs = (x, starts, np.arange(n))
        path = args.output/(name+'.pt'); model.save(path)
        loaded = type(model).load(path, args.device)
        a = model.score(*inputs); b = loaded.score(*inputs)
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
        assert np.isfinite(a).all()
        result['models'][name] = dict(seconds=time.monotonic()-start, rows=len(a), reload_max_abs=float(np.max(np.abs(a-b))),
                                     peak_cuda_gb=torch.cuda.max_memory_allocated()/2**30 if args.device.startswith('cuda') else None)
        print(f'SMOKE {name} passed', flush=True)
        del model, loaded
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    result.update(status='passed', device=args.device, torch=torch.__version__)
    write(args.output/'qualification.json', result)


if __name__ == '__main__':
    main()
