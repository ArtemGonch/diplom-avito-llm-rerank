"""
UR4Rec retriever (paper §3.2, Eq.4).

Block: MHAtt -> CrossAtt(Z) -> FFN
Self-attn/FFN init from BERT; cross-attn random init.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertLayer

from .masks import bidirectional_mask, build_extended_mask, proxy_item_mask


class RetrieverBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def _attn(
        self,
        layer: nn.MultiheadAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        out, _ = layer(
            query,
            key,
            value,
            attn_mask=attn_mask,
            need_weights=False,
        )
        return out

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor | None,
        self_mask: torch.Tensor | None,
        use_cross: bool,
    ) -> torch.Tensor:
        # Self-attention
        sm = build_extended_mask(self_mask, x.size(0), 1)
        sa = self._attn(self.self_attn, x, x, x, sm)
        x = self.norm1(x + self.dropout(sa))

        if use_cross and z is not None:
            ca = self._attn(self.cross_attn, x, z, z, None)
            x = self.norm2(x + self.dropout(ca))

        ff = self.ffn(x)
        x = self.norm3(x + self.dropout(ff))
        return x


class UR4RecRetriever(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_proxies: int = 8,
        num_layers: int = 6,
        num_heads: int = 12,
        dropout: float = 0.1,
        bert_name: str = "bert-base-uncased",
        init_from_bert: bool = True,
        aggr_dim: int | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_proxies = num_proxies
        self.aggr_dim = aggr_dim or hidden_size * (1 + 10)  # user + up to 10 item texts

        self.blocks = nn.ModuleList()
        if init_from_bert:
            from transformers import BertModel

            bert = BertModel.from_pretrained(bert_name)
            n_layers = min(num_layers, len(bert.encoder.layer))
        else:
            bert = None
            n_layers = num_layers
        for i in range(n_layers):
            blk = RetrieverBlock(hidden_size, num_heads, dropout)
            if bert is not None:
                self._init_block_from_bert(blk, bert.encoder.layer[i])
            self.blocks.append(blk)
        del bert

        self.proxies = nn.Parameter(torch.randn(num_proxies, hidden_size) * 0.02)
        self.z_proj = nn.Linear(self.aggr_dim, hidden_size)
        self.match_head = nn.Linear(hidden_size, 1)

    @staticmethod
    def _init_block_from_bert(blk: RetrieverBlock, src: BertLayer) -> None:
        """Init self-attn + FFN from BERT; cross-attn stays random (paper §3.2)."""
        bert_attn = src.attention.self
        blk.self_attn.in_proj_weight.data.copy_(
            torch.cat(
                [bert_attn.query.weight, bert_attn.key.weight, bert_attn.value.weight],
                dim=0,
            )
        )
        blk.self_attn.in_proj_bias.data.copy_(
            torch.cat(
                [bert_attn.query.bias, bert_attn.key.bias, bert_attn.value.bias],
                dim=0,
            )
        )
        blk.self_attn.out_proj.weight.data.copy_(src.attention.output.dense.weight.data)
        blk.self_attn.out_proj.bias.data.copy_(src.attention.output.dense.bias.data)
        blk.ffn[0].weight.data.copy_(src.intermediate.dense.weight.data)
        blk.ffn[0].bias.data.copy_(src.intermediate.dense.bias.data)
        blk.ffn[3].weight.data.copy_(src.output.dense.weight.data)
        blk.ffn[3].bias.data.copy_(src.output.dense.bias.data)
        blk.norm1.weight.data.copy_(src.attention.output.LayerNorm.weight.data)
        blk.norm1.bias.data.copy_(src.attention.output.LayerNorm.bias.data)
        blk.norm3.weight.data.copy_(src.output.LayerNorm.weight.data)
        blk.norm3.bias.data.copy_(src.output.LayerNorm.bias.data)

    def _memory_tokens(self, e_aggr: torch.Tensor) -> torch.Tensor:
        """Normalise UR4Rec knowledge memory to [batch, memory_tokens, hidden].

        New runs pass the Eq. 5 user/item vectors as separate tokens.  The 2-D
        branch keeps old checkpoints/callers loadable, but is a legacy fallback
        with degenerate single-key cross-attention.
        """
        if e_aggr.dim() == 3:
            if e_aggr.size(-1) != self.hidden_size:
                raise ValueError(
                    "UR4Rec memory token width must equal hidden_size: "
                    f"got {e_aggr.size(-1)} vs {self.hidden_size}"
                )
            return e_aggr
        if e_aggr.dim() == 2:
            return self.z_proj(e_aggr).unsqueeze(1)
        raise ValueError(f"Expected e_aggr rank 2 or 3, got shape {tuple(e_aggr.shape)}")

    def _run_blocks(
        self,
        x: torch.Tensor,
        z: torch.Tensor | None,
        mask: torch.Tensor | None,
        use_cross: bool,
    ) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, z, mask, use_cross)
        return x

    def forward_preference_filter(
        self,
        e_aggr: torch.Tensor,
        *,
        return_proxy_tokens: bool = False,
    ) -> torch.Tensor:
        """Eq.6: e_pref = Retriever(P, e_aggr); X0=P, cross-attn to Z."""
        b = e_aggr.size(0)
        x = self.proxies.unsqueeze(0).expand(b, -1, -1)
        z = self._memory_tokens(e_aggr)
        mask = bidirectional_mask(self.num_proxies, x.device)
        out = self._run_blocks(x, z, mask, use_cross=True)
        if return_proxy_tokens:
            return out
        return out.mean(dim=1)

    def forward_item_only(self, h_item: torch.Tensor) -> torch.Tensor:
        """Eq.7: item-only path, no cross-attention. Returns [B, D]."""
        if h_item.dim() == 1:
            h_item = h_item.unsqueeze(0)
        x = h_item.unsqueeze(1)
        out = self._run_blocks(x, None, None, use_cross=False)
        return out[:, 0, :]

    def forward_joint(
        self,
        h_items: torch.Tensor,
        e_aggr: torch.Tensor,
        mode: Literal["aug", "pim"] = "aug",
    ) -> torch.Tensor:
        """
        Eq.3.3 / Eq.8: Retriever([P; h_i], e_aggr).
        h_items: [B, L, D] or [B, D] for single item
        Returns per-item aug [B, L, D] (or [B, D] when L=1) or PIM logits [B].
        """
        if h_items.dim() == 2:
            h_items = h_items.unsqueeze(1)
        b, l, _ = h_items.shape
        proxies = self.proxies.unsqueeze(0).expand(b, -1, -1)
        x = torch.cat([proxies, h_items], dim=1)
        z = self._memory_tokens(e_aggr)
        mask = proxy_item_mask(self.num_proxies, l, x.device)
        out = self._run_blocks(x, z, mask, use_cross=True)
        proxy_out = out[:, : self.num_proxies, :]
        item_out = out[:, self.num_proxies :, :]
        if mode == "aug":
            if l == 1:
                return proxy_out.mean(dim=1)
            return item_out
        logits = self.match_head(proxy_out).squeeze(-1)
        return logits.mean(dim=1)
