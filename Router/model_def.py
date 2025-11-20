# model_def.py
import torch
import torch.nn as nn

class EncoderLayerWithAttn(nn.Module):
    def __init__(self, d_model, nhead, dim_ff=256, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff  = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, need_weights=False):
        attn_out, attn_weights = self.mha(x, x, x, need_weights=need_weights, average_attn_weights=False)
        x = self.norm1(x + self.drop(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.drop(ff_out))
        return x, attn_weights


class TabTransformerCLS(nn.Module):
    def __init__(self, seq_len, num_classes, d_model=128, nhead=4, depth=2, dim_ff=256, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        self.value_embed = nn.Linear(1, d_model)
        self.feature_embed = nn.Embedding(seq_len + 1, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.layers = nn.ModuleList([
            EncoderLayerWithAttn(d_model, nhead, dim_ff, dropout) for _ in range(depth)
        ])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )
        self.last_attn = None

    def forward(self, x, capture_attn=False):
        B, F = x.shape
        assert F == self.seq_len
        v = self.value_embed(x.unsqueeze(-1))
        ids = torch.arange(1, F + 1, device=x.device).unsqueeze(0).expand(B, F)
        fe = self.feature_embed(ids)
        tok = v + fe
        cls = self.cls_token.expand(B, 1, self.d_model)
        cls_id = torch.zeros((B, 1), dtype=torch.long, device=x.device)
        cls_emb = cls + self.feature_embed(cls_id)
        xseq = torch.cat([cls_emb, tok], dim=1)
        xseq = self.dropout(xseq)

        attn_weights_last = None
        for i, layer in enumerate(self.layers):
            need = capture_attn and (i == len(self.layers) - 1)
            xseq, aw = layer(xseq, need_weights=need)
            if need:
                attn_weights_last = aw

        if capture_attn:
            self.last_attn = attn_weights_last

        cls_out = xseq[:, 0, :]
        logits = self.head(cls_out)
        return logits
