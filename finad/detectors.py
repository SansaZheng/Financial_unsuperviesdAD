"""Detector construction/loading; modern methods are opt-in."""
from .classical import ClassicalDetector, METHODS as CLASSICAL
from .neural import NeuralDetector, METHODS as NEURAL

MODERN=('rgld2026','rttad2026')
METHODS=CLASSICAL+NEURAL+MODERN


def make_detector(method,seed,device,steps,pretrain_steps):
    if method in CLASSICAL:return ClassicalDetector(method,seed)
    if method in NEURAL:return NeuralDetector(method,seed,device,steps,pretrain_steps)
    if method=='rgld2026':
        from .rgld import RGLD
        return RGLD(seed)
    if method=='rttad2026':
        from .rttad import RTTAD
        return RTTAD(seed,device,steps)
    raise ValueError(f'Unknown method: {method}')


def extension(method):return 'joblib' if method in CLASSICAL or method=='rgld2026' else 'pt'


def load_detector(method,path,device='cpu'):
    if method in CLASSICAL:return ClassicalDetector.load(path)
    if method in NEURAL:return NeuralDetector.load(path,device)
    if method=='rgld2026':
        from .rgld import RGLD
        return RGLD.load(path)
    if method=='rttad2026':
        from .rttad import RTTAD
        return RTTAD.load(path,device)
    raise ValueError(f'Unknown method: {method}')
