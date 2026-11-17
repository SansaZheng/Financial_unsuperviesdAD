"""Train-only role channels; no target labels or evaluation-set fitting."""
import json
import numpy as np
from finad.preprocessing import Preprocessor
from finad_repair.roles import canonical, RoleNormalizer

VARIANTS = ('legacy','role','numeric','aligned','centered','dual','dual_center','dual_role')

class RoleAdapter:
    def __init__(self, mode):
        if mode not in VARIANTS: raise ValueError(mode)
        self.mode = mode

    @staticmethod
    def validate(raw, role):
        raw=np.asarray(raw);role=np.asarray(role)
        if raw.ndim!=3 or raw.shape[0]!=4 or raw.shape[2]!=10 or role.shape!=(raw.shape[1],) or role.dtype!=bool or not np.isfinite(raw).all():
            raise ValueError('Need finite 4 x N x10 and boolean roles')
        if not role.any() or role.all(): raise ValueError('Both roles required')

    def fit(self, raw, role):
        self.validate(raw,role)
        self.common=Preprocessor().fit(raw.reshape(-1,10))
        self.pooled=Preprocessor().fit(canonical(raw,role).reshape(-1,4))
        self.relative=RoleNormalizer().fit(raw,role)
        return self

    @staticmethod
    def flatten(x): return x.transpose(1,0,2).reshape(x.shape[1],-1)

    def transform(self, raw, role):
        self.validate(raw,role)
        common=self.common.transform(raw.reshape(-1,10)).reshape(raw.shape)
        numeric=self.flatten(common[:,:,2:])
        if self.mode=='legacy': return self.flatten(common)
        if self.mode=='numeric': return numeric
        aligned=self.flatten(self.pooled.transform(canonical(raw,role).reshape(-1,4)).reshape(4,-1,4))
        if self.mode=='aligned': return aligned
        relative=self.relative.transform(raw,role)
        if self.mode=='role': return relative
        centered=canonical(raw,role).astype(np.float64)
        for r,pre in zip([False,True],self.relative.scalers):
            centered[:,role==r]=(centered[:,role==r]-pre.center)/self.pooled.scale
        centered=self.flatten(np.clip(centered,-8,8).astype(np.float32))
        if self.mode=='centered': return centered
        extra={'dual':aligned,'dual_center':centered,'dual_role':relative}[self.mode]
        return np.column_stack([numeric/np.sqrt(32),extra/np.sqrt(16)]).astype(np.float32)

    def save(self, folder):
        folder.mkdir(parents=True,exist_ok=True)
        (folder/'adapter.json').write_text(json.dumps({'mode':self.mode})+'\n')
        self.common.save(folder/'preprocessor.npz');self.pooled.save(folder/'pooled.npz');self.relative.save(folder)

    @classmethod
    def load(cls,folder):
        obj=cls(json.loads((folder/'adapter.json').read_text())['mode'])
        obj.common=Preprocessor.load(folder/'preprocessor.npz');obj.pooled=Preprocessor.load(folder/'pooled.npz');obj.relative=RoleNormalizer.load(folder)
        return obj
