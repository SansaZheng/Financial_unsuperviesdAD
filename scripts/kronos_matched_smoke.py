"""Target-free real-weight qualification and throughput measurement, not results."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from finad_kronos.assets import fetch_assets, load_predictor
from finad_kronos.adapter import forecast
from finad_kronos.experiment import load_values, verify_inputs, write

p = argparse.ArgumentParser(); p.add_argument('--data', type=Path, required=True)
p.add_argument('--assets', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
p.add_argument('--device', default='cuda:0'); a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
if a.device.startswith('cuda'):
    torch.cuda.set_device(a.device); torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
inputs = verify_inputs(a.data, ['banksim']); receipt = fetch_assets(a.assets)
predictor = load_predictor(a.assets, a.device)
values, starts, targets, _, cold = load_values(a.data/'banksim', 42, a.output)
from finad_kronos.adapter import history
t = targets['calibration'][:8]; h, _ = history(values, t, np.zeros(8, dtype=int), starts)
first, _ = forecast(predictor, h, t-starts[t], 42)
again, _ = forecast(predictor, h, t-starts[t], 42)
np.testing.assert_array_equal(first, again)
loaded = load_predictor(a.assets, a.device)
reloaded, _ = forecast(loaded, h, t-starts[t], 42)
np.testing.assert_array_equal(first, reloaded)
del loaded
rng = np.random.default_rng(1986); timings = []
for batch in (1024, 1024, 1024, 1024):
    h = rng.normal(size=(batch, 32)); positions = np.arange(32, 32+batch)
    started = time.monotonic(); forecast(predictor, h, positions, 43)
    if a.device.startswith('cuda'): torch.cuda.synchronize()
    timings.append(time.monotonic()-started)
result = dict(status='passed', target_free=True, real_weights=True, exact_input_files=len(inputs),
              frozen_assets=receipt, repeated_forecast_max_abs=float(np.max(np.abs(first-again))),
              reload_max_abs=float(np.max(np.abs(first-reloaded))), batch_size=1024, batch_seconds=timings,
              feature_forecasts_per_second=3072/sum(timings[1:]),
              device=a.device, torch=torch.__version__, cold_start_rule=cold,
              cuda_devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
              peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if a.device.startswith('cuda') else None)
write(a.output/'qualification.json', result)
print(json.dumps(result), flush=True)
