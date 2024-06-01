import torch
import torch.nn as nn
import math


class InputEmbedding(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super.__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PE(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super.__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # crrate matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # use log for simplify the computation and make it more stable
        # create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # denominator
        div_term  = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # apply sin to even pos
        pe[:, 0::2] = torch.sin(position  * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        # register tesnor in buffer of model, saved in the model file
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    

class LayerNorm(nn.Module):
    def __init__(self, eps: float= 10**-6 ) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std(dim=-1, keepdim=True)
        return self.alpha * (x -  mean) / (std + self.eps) + self.bias
    


class FFBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 b2

    def forward(self, x):
        # (b, seq_len, d_model) -> (b, seq_len, d_ff) -> (b, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model:int, h: int, dropout: float) -> None:
        super().__init__()

        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, 'd_model is not divisible by h'

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):
        d_k = query.shape[-1]

        # (b, h, seq_len, d_k) -> (b, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask == 0 , -1e9)

        attention_scores = attention_scores.softmax(dim = -1) # (b, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (b, seq_len, d_model) -> (b, seq_len, d_model)
        key = self.w_k(k)
        value = self.w_v(v)

        #  (b, seq_len, d_model) -> (b, seq_len h, d_k) -> (b, h, seq_len, d_k)
        query  = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key  = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value  = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)

        # (b, h, seq_len, d_k)  -> (b, seq_len, h, d_k) -> (b, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (b, seq_len, d_mdoel) - > (b, seq_len, d_model)
        return self.w_o(x)
    

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):

        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttentionBlock, ff_block: FFBlock, dropout: float) -> None:
        super.__init__()
        self.self_attention_block = self_attention_block
        self.ff_block = ff_block
        self.residudal_connections = nn.ModuleList([ResidualConnection(dropout=dropout) for _ in range(2)])

    def forward(self, x, src_mask):

        x = self.residudal_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residudal_connections[1](x, self.ff_block)
        return x 
    

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super.__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock,
                 ff_block: FFBlock, dropout: float) -> None:
        super.__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.ff_block = ff_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout=dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x = self.residual_connections[2](x, lambda x: self.ff_block)
        return x
    

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super.__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)
    
class ProjLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super.__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (b, seq_len, d_model) -> (b, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)
    

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, 
                 src_emb: InputEmbedding, tgt_emb: InputEmbedding,
                 src_pos: PE, tgt_pos: PE, 
                 proj_layer: ProjLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj_layer = proj_layer

    def encode(self, src, src_mask):
        src  = self.src_emb(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decide(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_emb(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.proj_layer(x) 
    

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, 
                      d_model: int=512, N: int=6,
                      h:int=8, dropout: float=0.1, d_ff:int=2048):
    src_emb = InputEmbedding(d_model, src_vocab_size)
    tgt_emb = InputEmbedding(d_model,  tgt_vocab_size)

    src_pos = PE(d_model, src_seq_len, dropout=dropout)
    tgt_pos = PE(d_model, tgt_seq_len, dropout=dropout)

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        ff_block = FFBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, ff_block, dropout)
        encoder_blocks.append(encoder_block)


    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        ff_block = FFBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, ff_block, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    proj_layer = ProjLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_emb, tgt_emb, src_pos, tgt_pos, proj_layer)

    # init params
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer