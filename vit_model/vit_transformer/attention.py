# vit_model/transformer/attention.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, config, vis=False, mode="sa"):  # 默认用 self-attention
        super().__init__()
        self.vis = vis
        self.mode = mode
        self.num_heads = config.transformer["num_heads"]
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_dim = self.num_heads * self.head_dim

        self.query = nn.Linear(config.hidden_size, self.all_head_dim)
        self.key = nn.Linear(config.hidden_size, self.all_head_dim)
        self.value = nn.Linear(config.hidden_size, self.all_head_dim)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = nn.Dropout(config.transformer["dropout_rate"])
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def compute_attention(self, query_input, key_input, value_input):
        q = self.transpose_for_scores(self.query(query_input))
        k = self.transpose_for_scores(self.key(key_input))
        v = self.transpose_for_scores(self.value(value_input))

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_probs = self.softmax(attn_scores)
        attn_probs = self.attn_dropout(attn_probs)

        context = torch.matmul(attn_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(query_input.size(0), -1, self.all_head_dim)
        output = self.out(context)
        output = self.proj_dropout(output)
        return output, attn_probs if self.vis else None

    def forward(self, *inputs):
        if len(inputs) == 1:
            # self-attention 模式
            hidden_states = inputs[0]
            output, attn_weights = self.compute_attention(hidden_states, hidden_states, hidden_states)
            return output, None, attn_weights
        elif len(inputs) == 2:
            # cross-attention 模式
            query_input, key_value_input = inputs
            out1, attn_weights = self.compute_attention(query_input, key_value_input, key_value_input)
            out2, _ = self.compute_attention(key_value_input, query_input, query_input)
            return out1, out2, attn_weights
        else:
            raise ValueError("Attention forward expects 1 or 2 input tensors.")

