import torch
from torch import nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import seaborn as sns

from FusionLayer.position_embedding import SinusoidalPositionalEmbedding
from FusionLayer.multihead_attention import MultiheadAttention


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    return nn.LayerNorm(embedding_dim)


def fill_with_neg_inf(t):
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1 + abs(dim2 - dim1))
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_mask = attn_mask

        self.self_attn = MultiheadAttention(embed_dim, num_heads, attn_dropout)
        self.fc1 = Linear(embed_dim, 4 * embed_dim)
        self.fc2 = Linear(4 * embed_dim, embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(embed_dim) for _ in range(2)])
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.attn_weights = None  # <--- attention 儲存

    def forward(self, x, x_k=None, x_v=None):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None

        if x_k is None and x_v is None:
            x, attn_weights = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True)
            x, attn_weights = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)

        self.attn_weights = attn_weights.detach().cpu()  # 儲存 attention 權重

        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0,
                 res_dropout=0.0, embed_dropout=0.0, attn_mask=False):
        super().__init__()
        self.dropout = embed_dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        self.attn_mask = attn_mask

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, attn_dropout, relu_dropout, res_dropout, attn_mask)
            for _ in range(layers)
        ])
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

        self.attn_weights_all = []  # <--- 所有 attention 權重

    def forward(self, x_in, x_in_k=None, x_in_v=None):
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.embed_positions is not None:
                x_k += self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)
                x_v += self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)

        self.attn_weights_all.clear()  # 清空上一輪的記錄

        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            else:
                x = layer(x)
            self.attn_weights_all.append(layer.attn_weights)

        if self.normalize:
            x = self.layer_norm(x)
        return x

    def get_attention_maps(self):
        return self.attn_weights_all


### 🟢 可視化函數
def plot_attention(attn_weights, layer=0, head=0, title="Attention Map"):
    """
    attn_weights: List of [num_heads, seq_len, seq_len]
    """
    weights = attn_weights[layer][head]  # shape: [seq_len, seq_len]
    plt.figure(figsize=(8, 6))
    sns.heatmap(weights, cmap='viridis')
    plt.title(f"{title} - Layer {layer} Head {head}")
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.tight_layout()
    plt.show()


### ✅ 測試用
if __name__ == '__main__':
    encoder = TransformerEncoder(embed_dim=768, num_heads=4, layers=2, attn_dropout=0.1, relu_dropout=0.1,
                                 res_dropout=0.1, embed_dropout=0.2, attn_mask=True)
    x = torch.rand(32, 1, 768)  # [seq_len, batch, dim]
    out = encoder(x)

    # 畫 attention heatmap：第 0 層，第 0 個 head
    plot_attention(encoder.get_attention_maps(), layer=0, head=0)

    print('Output shape:', out.shape)
