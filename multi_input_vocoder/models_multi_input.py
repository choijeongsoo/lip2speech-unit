# adapted from https://github.com/jik876/hifi-gan

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "speech-resynthesis"))
from models import CodeGenerator
sys.path.pop(0)

class MelCodeGenerator(CodeGenerator):
    def __init__(self, h):
        super().__init__(h)
        self.dict = nn.Embedding(h.num_embeddings, h.embedding_dim)
        self.multispkr = h.get('multispkr', None)

        embedder_dim = h.get("embedder_dim", None)
        if self.multispkr and not embedder_dim:
            self.spkr = nn.Embedding(h.get("num_speakers", 200), h["embedding_dim"])
        elif embedder_dim:
            self.spkr = nn.Linear(embedder_dim, h["embedding_dim"])

        self.layer = nn.Sequential(
            nn.ConvTranspose1d(h.embedding_dim, h.embedding_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(h.embedding_dim, h.embedding_dim)

    def forward(self, **kwargs):
        x = kwargs['mel']

        code = self.dict(kwargs['code'])
        code = self.layer(code.permute(0,2,1)).permute(0,2,1)
        code = self.dropout(code)
        code = self.fc(code)
        code = code.permute(0,2,1)

        x = torch.cat([x, code], dim=1)

        if self.multispkr:
            spkr = self.spkr(kwargs['spkr'])
            spkr = self._upsample(spkr, x.shape[-1])
            x = torch.cat([x, spkr], dim=1)

        for k, feat in kwargs.items():
            if k in ['spkr', 'code', 'mel']:
                continue

            feat = self._upsample(feat, x.shape[-1])
            x = torch.cat([x, feat], dim=1)

        return super(CodeGenerator, self).forward(x)