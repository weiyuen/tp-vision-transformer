import torch
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from PIL import Image
from torch import nn
from torch import Tensor


class PatchEmbeddings(nn.Module):
    def __init__(self, image_size=224, patch_size=16, emb_size=768):
        super().__init__()
        # NN to project images into patches.
        self.project = nn.Sequential(
            nn.Conv2d(3, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e h w -> b (h w) e')
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.position_emb = nn.Parameter(torch.randn((image_size // patch_size)**2 + 1, emb_size))
        
    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.project(x)
        
        # Create cls token for each sample in batch.
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        
        # Prepend cls token and add position embedding.
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.position_emb
        return x
    
    
class TPMHA(nn.Module):
    def __init__(self, emb_size=768, n_heads=8, dropout=0):
        super().__init__()
        self.emb_size = emb_size
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.qkvr = nn.Linear(emb_size, emb_size*4)
        self.projection = nn.Linear(emb_size, emb_size)

        
    def forward(self, x, mask=None):
        # Split x into QKVR.
        qkvr = self.qkvr(x)
        qkvr = rearrange(qkvr, 'b n (h d qkvr) -> (qkvr) b h n d', h=self.n_heads, qkvr=4)
        q, k, v, r = qkvr[0], qkvr[1], qkvr[2], qkvr[3]
        
        # Attention mechanism
        energy = torch.einsum('bhqd, bhkd -> bhqk', q, k)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.dropout(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, v)
        
        # TPR
        out = out * r
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
        
        return out
    
    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        x_copy = x
        x = self.fn(x, **kwargs)
        x += x_copy
        return x
    
    
class FeedForward(nn.Module):
    def __init__(self, emb_size=768, expansion=4, dropout=0.):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size * expansion, emb_size)
        )
        
    def forward(self, x):
        x = self.ff(x)
        return x
    
    
class EncoderBlock(nn.Module):
    def __init__(self, emb_size=768, dropout=0., ff_expansion=4, ff_dropout=0., **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            Residual(nn.Sequential(
                TPMHA(emb_size, **kwargs),
                nn.Dropout(dropout)
            )),
            nn.LayerNorm(emb_size),
            Residual(nn.Sequential(
                FeedForward(emb_size, expansion=ff_expansion, dropout=ff_dropout),
                nn.Dropout(dropout)
            )),
            nn.LayerNorm(emb_size)
        )
        
    def forward(self, x):
        return self.encoder(x)
    
    
class Encoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[EncoderBlock(**kwargs) for _ in range(depth)])
        
        
class ClassificationHead(nn.Module):
    def __init__(self, emb_size=768, n_classes=1000):
        super().__init__()
        self.cls_head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes),
        )
        
    def forward(self, x):
        cls_token = x[:, 0, :]
        out = self.cls_head(cls_token)
        logits = F.softmax(out, dim=1)
        return logits
    
    
class TPViT(nn.Module):
    def __init__(
        self,
        patch_size = 16,
        emb_size = 768,
        image_size = 224,
        depth = 12,
        n_classes = 1000,
        **kwargs
    ):
        super().__init__()
        self.model = nn.Sequential(
            PatchEmbeddings(image_size, patch_size, emb_size),
            Encoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )
        
    def forward(self, x):
        return self.model(x)