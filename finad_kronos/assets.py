"""Fetch, hash and load the exact official source and frozen weights externally."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import urllib.request

OFFICIAL_COMMIT = '67b630e67f6a18c9e9be918d9b4337c960db1e9a'
SOURCE = {
    'model/__init__.py': 'f8f856ca3fedadcaac97e196be23d1aeda1c3c9ffe8903d66d43ea3bcac6240c',
    'model/kronos.py': '0a5f90282e2039c2de0771473419715c845def154896dbd0f5747837e6241032',
    'model/module.py': 'a07edbadc0e96804c8158c021bbc6063bb7cc43b34d7fc470d5c8ff2005a409f',
}
WEIGHTS = {
    'Kronos-mini/config.json': '70daca2cb11e3a979dd6b8ac12ee08e2aace877acf28f5b8dfb4fe5609736201',
    'Kronos-mini/model.safetensors': 'a7d5f37e2e9fbd9891f7d7d4f72574512dd1f704fee14223e0a8cd0fbf54197c',
    'Kronos-Tokenizer-2k/config.json': '0b30a443affb03e05a876a083857de9164f899feb7b4d261da02c485c9a3e3b6',
    'Kronos-Tokenizer-2k/model.safetensors': 'b97ec46b3b72160509e289183eaf7bdf5f0dac5bb9b49522f6d46638a99a8717',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(2**20):
            h.update(chunk)
    return h.hexdigest()


def fetch_assets(root):
    root = Path(root)
    receipt = {'official_commit': OFFICIAL_COMMIT, 'sha256': {}}
    for group, entries in [('official', SOURCE), ('weights', WEIGHTS)]:
        for name, expected in entries.items():
            path = root/group/name
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                if group == 'official':
                    url = f'https://raw.githubusercontent.com/shiyu-coder/Kronos/{OFFICIAL_COMMIT}/{name}'
                else:
                    model, filename = name.split('/', 1)
                    url = f'https://huggingface.co/NeoQuasar/{model}/resolve/main/{filename}'
                temporary = path.with_suffix(path.suffix+'.download')
                with urllib.request.urlopen(url, timeout=180) as src, temporary.open('wb') as dst:
                    while chunk := src.read(2**20):
                        dst.write(chunk)
                if sha(temporary) != expected:
                    temporary.unlink()
                    raise ValueError(f'Pinned asset changed: {name}')
                temporary.replace(path)
            actual = sha(path)
            if actual != expected:
                raise ValueError(f'Asset hash mismatch: {name}')
            receipt['sha256'][f'{group}/{name}'] = actual
    (root/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def load_predictor(root, device):
    root = Path(root)
    fetch_assets(root)
    source = str((root/'official').resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    official = importlib.import_module('model')
    if Path(official.__file__).resolve() != (root/'official/model/__init__.py').resolve():
        raise ValueError('An unrelated Python model module shadows the pinned Kronos source')
    model = official.Kronos.from_pretrained(str(root/'weights/Kronos-mini')).eval().requires_grad_(False)
    tokenizer = official.KronosTokenizer.from_pretrained(str(root/'weights/Kronos-Tokenizer-2k')).eval().requires_grad_(False)
    return official.KronosPredictor(model, tokenizer, device=device, max_context=512, clip=5)
