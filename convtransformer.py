import torch
import torch.nn as nn 
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = self.generate_encoding(d_model, max_len)

    def generate_encoding(self, d_model, max_len):
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.encoding[:seq_len, :]


class MultiHeadAttnetionLayer(nn.Module):
    def __init__(self, in_c, hidden_c, n_heads, dropout_ratio):
        super().__init__()
        
        self.hidden_dim = hidden_c 
        self.n_heads = n_heads 
        self.dropout_ratio = dropout_ratio
        self.head_dim = hidden_c // n_heads  
        
        self.conv_q = nn.Conv1d(in_c, hidden_c, kernel_size=1)
        self.conv_k = nn.Conv1d(in_c, hidden_c, kernel_size=1)
        self.conv_v = nn.Conv1d(in_c, hidden_c, kernel_size=1)
    
        self.dropout = nn.Dropout(dropout_ratio)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

        self.conv_o = nn.Conv1d(hidden_c, in_c, kernel_size=1)
        
        
    def forward(self, query, key, value):
        # q,k,v: (B, 6, 30)
        
        batch_size = query.shape[0]

        Q = self.conv_q(query).transpose(1,2)
        K = self.conv_k(key).transpose(1,2)
        V = self.conv_v(value).transpose(1,2)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)

        energy = torch.matmul(Q, K.permute(0,1,3,2)) / self.scale 

        attention = torch.softmax(energy, dim=-1)
        
        x = torch.matmul(self.dropout(attention), V)
        
        x = x.permute(0 ,2, 1, 3).contiguous()

        x = x.view(batch_size, -1, self.hidden_dim).transpose(1,2)
        
        x = self.conv_o(x)
        
        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hidden_dim, pf_dim, dropout_ratio):
        super().__init__()
        
        self.fc_1 = nn.Linear(hidden_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_ratio)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        x = self.dropout(self.gelu(self.fc_1(x)))
        x = self.fc_2(x)
        return x 


class EncoderLayer(nn.Module):
    def __init__(self, in_c, hidden_c, n_heads, pf_dim, dropout_ratio):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(in_c)
        self.ff_layer_norm = nn.LayerNorm(in_c)
        
        self.self_attention = MultiHeadAttnetionLayer(in_c, hidden_c, n_heads, dropout_ratio)
        
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(in_c, pf_dim, dropout_ratio)
        self.dropout = nn.Dropout(dropout_ratio)
    def forward(self, x):
        # x: (B, 6, 30)
        
        _x, _ = self.self_attention(x, x, x)
        # _x: (B, 6, 30)
        
        x, _x = x.transpose(1,2), _x.transpose(1,2)
        # x: (B, 30, 6), _x: (B, 30, 6)
        
        x = self.self_attn_layer_norm(x + self.dropout(_x))
        # x: (B, 30, 6)
        
        _x = self.positionwise_feedforward(x)
        # _x: (B, 30, 6)
        
        x = self.ff_layer_norm(x + self.dropout(_x)).transpose(1,2)
        # x: (B, 6, 30)
        return x

class ConvTransformer(nn.Module):
    def __init__(self, in_c, hidden_c, n_heads, pf_dim, n_layers, mlp_in, mlp_hidden, mlp_out, dropout_ratio):
        super().__init__()
        
        self.positional_encoding = PositionalEncoding(in_c)
        self.layers = nn.ModuleList([EncoderLayer(in_c, hidden_c, n_heads, pf_dim, dropout_ratio) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout_ratio)
        self.mlp = MLPblock(input_dim=mlp_in, hidden_dim=mlp_hidden, output_dim=mlp_out, dropout_ratio=dropout_ratio)
    
    def forward(self, x):
        # x: (B, 30, 6)
        batch_size = x.shape[0]
        
        x = self.dropout(self.positional_encoding(x)).transpose(1,2)
        # x: (B, 6, 30)
        
        for layer in self.layers:
            x = layer(x)
        
        # x: (B, 6, 30)
        x = x.transpose(1,2).view(batch_size, -1)
        # x: (B, 30*6)
        x = self.mlp(x)
        return x

class MLPblock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_ratio):
        super().__init__()
        self.mlp_block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.mlp_block(x)

if __name__ == "__main__":
    # (B, 30, 6) -> Positional encoding -> (B, 30, 6) -> (B, 6, 30) -> Attention -> (B, 6, 30) -> (B, 30, 6) -> FFW -> (B, 30, 6)
    # ConvTransformer: (B, 30, 6) -> (B, 30, 6)
    input_data = torch.randn(32, 30, 6) # (B, seq_len, emb_dim)
    
    in_c = input_data.shape[-1] # 6
    hidden_c = 64
    n_heads = 8 
    pf_dim = 64
    n_layers= 6
    dropout_ratio = 0.1
    mlp_in = 30 * 6
    mlp_hidden = 32
    mlp_out = 3

    model = ConvTransformer(in_c, hidden_c, n_heads, pf_dim, n_layers, mlp_in=mlp_in, mlp_hidden=mlp_hidden, mlp_out=mlp_out, dropout_ratio=dropout_ratio)
    trans_output = model(input_data)
    print(trans_output.shape)