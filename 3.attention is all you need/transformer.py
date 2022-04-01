from pyexpat import model
from sre_parse import _OpSubpatternType
from turtle import forward
from numpy import mat, outer
import torch
import math
import torch.nn.functional as F
import copy

class Embedder(torch.nn.Module):
    def __init__(self,vocab_size,d_model):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size,d_model)

    def forward(self,x):

        return self.embed(x)

class PositonalEncoder(torch.nn.Module):

    def __init__(self,d_model,max_seq_len=80):
        super().__init__()
        self.d_model = d_model

        pe_matrix = torch.zeros(max_seq_len,d_model)

        for pos in range(max_seq_len):
            for i in range(0,d_model):
                pe_matrix [pos,i] = math.sin(pos / 1000**(2*i/d_model))
                pe_matrix[pos,i+1] = math.cos(pos/1000**(2*i/d_model))

            pe_matrix = pe_matrix.unsqueeze(0)
            self.register_buffer_buffer("pe",pe_matrix)
    
    def forward(self,x):

        seq_len = x.size()[1]
        x = x + self.pe[:,:seq_len]
        return x

def scaled_dot_product_attention(q,k,v, mask = None, dropout = None):
    
    attention_score = torch.matmul(q,k.tanspose(-2,-1))/math.sqrt(q.shape[-1])

    if mask is not None:
        attention_score = attention_score.masked_fill(mask == 0, value=-1e9)

    attention_weight = F.softmax(attention_score,dim = -1)

    if dropout is not None:
        attention_weight = dropout(attention_weight)
    
    output = torch.matmul(attention_weight,v)

    return output

class MultiheasAttention(torch.nn.Module):
    def __init__(self,n_heads,d_model,dropout = 0.1):
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = self.d_v = d_model // n_heads

        self.q_linear_layer = []
        self.k_linear_layer = []
        self.v_linear_layer = []

        for i in range(n_heads):
            self.q_linear_layer.append(torch.nn.Linear(d_model, self.d_k))
            self.k_linear_layer.append(torch.nn.Linear(d_model,self.d_k))
            self.v_linear_layer.append(torch.nn.Linear(d_model,self.d_v))

        self.dropout = torch.nn.Dropout(dropout)
        self.out = torch.nn.Linear(n_heads*self.d_v,d_model)

    def forward(self,q,k,v,mask= None):
        multi_head_attention_outputs = []
        for q_linear,k_linear,v_linear in zip(self.q_linear_layer,self.k_linear_layer,self.v_linear_layer):
            new_q = q_linear(q)
            new_k = k_linear(k)
            new_v = v_linear(v)

            head_v = scaled_dot_product_attention(new_q,new_k,new_v,mask,self.dropout)
            multi_head_attention_outputs.append(head_v)
        concat = torch.cat(multi_head_attention_outputs,-1)
        output = self.out(concat)

        return output
    
class FeedForward(torch.nn.Module):
    def __init__(self,d_model,d_ff=2048,dropout=0.1):
        super().__init__()
        self.linear_1 = torch.nn.Linear(d_model,d_ff)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear_2 = torch.nn.Linear(d_ff,d_model)

    def forward(self,x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class LayerNorm(torch.nn.Module):
    def __init__(self,d_model,eps = 1e-6):
        super().__init__()

        self.d_model = d_model
        self.alpha = torch.nn.parameter(torch.ones(self.d_model))
        self.beta = torch.nn.parameter(torch.zero(self.d_model))
        self.eps = eps
    
    def forward(self,x):
        x_hat = (x-x.mean(dim = -1, keepdim = True))/(x.std(dim = -1,keepdim = True) + self.eps)
        x_tilde = self.alpha*x_hat + self.beta

        return x_tilde



class EncoderLayer(torch.nn.Module):
    def __init__(self,d_model,n_heads,dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model)
        self.multi_head_attention = MultiheasAttention(n_heads,d_model)
        self.feed_forward = FeedForward(d_model)
        self.dropout_1 = torch.nn.Dropout(dropout)
        self.dropout_2 = torch.nn.Dropout(dropout)
    def forward(self,x,mask):

        x = x + self.dropout_1(self.multi_head_attention(x,x,x,mask))
        x = x + self.dropout_2(self.feed_forward(x))

        x = self.norm_2(x)

        return x

class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model)
        self.norm_3 = LayerNorm(d_model)
        
        self.dropout_1 = torch.nn.Dropout(dropout)
        self.dropout_2 = torch.nn.Dropout(dropout)
        self.dropout_3 = torch.nn.Dropout(dropout)
        
        self.multi_head_attention_1 = MultiheasAttention(n_heads, d_model)
        self.multi_head_attention_2 = MultiheasAttention(n_heads, d_model)
        
        self.feed_forward = FeedForward(d_model)
        
    def forward(self, x, encoder_output, src_mask, trg_mask):
        x = self.dropout_1(self.multi_head_attention_1(x, x, x, trg_mask))
        x = x + self.norm_1(x)
        
        x = self.dropout_2(self.multi_head_attention_2(x, encoder_output, encoder_output, src_mask))
        x = x + self.norm_2(x)
        
        x = self.dropout_3(self.feed_forward(x))
        x = x + self.norm_3(x)
        
        return x

def clone_layer(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, d_model, N, n_heads):
        super().__init__()
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositonalEncoder(d_model)
        self.encoder_layers = clone_layer(EncoderLayer(d_model, n_heads), N)
        self.norm = LayerNorm(d_model)
        
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for encoder in self.encoder_layers:
            x = encoder(x, mask)
        return self.norm(x)


class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, d_model, N, n_heads):
        super().__init__()
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositonalEncoder(d_model)
        self.decoder_layers = clone_layer(DecoderLayer(d_model, n_heads), N)
        self.norm = LayerNorm(d_model)
        
    def forward(self, trg, encoder_output, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for decoder in self.decoder_layers:
            x = decoder(x, encoder_output, src_mask, trg_mask)
        return self.norm(x)

class Transformer(torch.nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, N, n_heads):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, N, n_heads)
        self.decoder = Decoder(trg_vocab_size, d_model, N, n_heads)
        self.linear = torch.nn.Linear(d_model, trg_vocab_size)
        
    def forward(self, src, trg, src_mask, trg_mask):
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(trg, encoder_output, src_mask, trg_mask)
        output = self.linear(decoder_output)
        return output

