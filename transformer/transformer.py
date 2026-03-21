import torch
import torch.nn as nn
import math
import copy

# 词嵌入(ok)
class Embedding(nn.Module):
    def __init__(self,vocab_size,embedding_size):
        super().__init__()
        self.vocab_size=vocab_size
        self.embedding_size=embedding_size
        self.embedding = nn.Embedding(vocab_size,embedding_size)
    def forward(self,x):
        embeded = self.embedding(x)
        result = embeded*math.sqrt(self.embedding_size) # 两个作用
        return result

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self,embedding_size,dropout_p,max_len):
        super().__init__()
        self.embedding_size=embedding_size
        self.dropout = nn.Dropout(p=dropout_p)
        # 初始化位置编码矩阵(max_len,embedding_size)
        pe = torch.zeros(max_len,embedding_size)
        # 定义位置矩阵(max_len,1)
        temp=torch.arange(0,max_len).unsqueeze(1)
        # 定义转化矩阵(1,embedding_size/2)
        div=torch.exp(torch.arange(0,embedding_size,2)*-math.log(10000.0)/embedding_size)
        # 赋值(max_len,embedding_size/2)
        position=temp*div
        # 赋值
        pe[:,0::2] = torch.sin(position)
        pe[:,1::2] = torch.cos(position)
        pe=pe.unsqueeze(0)
        # 注册缓冲区
        self.register_buffer('pe',pe)
    def forward(self,x): # x为embedding后结果
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)

# 自注意力函数(ok)
def self_attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask < 0, -1e9)
    atten_weight = torch.softmax(scores, dim=-1)
    if dropout is not None:
        atten_weight = dropout(atten_weight)
    return torch.matmul(atten_weight, value)

#clone函数(ok)
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

#多头自注意力(ok)
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, embedding_dim, dropout=0.1,is_causal=False):
        super().__init__()
        #不符合则报错
        assert embedding_dim % n_head == 0
        self.n_head = n_head
        self.head_dim = embedding_dim // n_head
        #克隆linear层
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        #初始化atten_weight
        self.atten_weight=None
        self.dropout = nn.Dropout(dropout)

        self.is_causal = is_causal  # 新增：是否是causal mask

    def create_causal_mask(self, seq_len):
        """创建causal mask（防止偷看未来）"""
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        return mask

    def forward(self, query, key, value, mask=None):

        self.batch_size = query.size(0)
        # 重点
        query, key, value = [model(item).view(self.batch_size,-1,self.n_head,self.head_dim).transpose(1,2)
                             for model,item in zip(self.linears, (query, key, value))]

        # 如果causal attention，生成mask
        if self.is_causal:
            seq_len = query.size(2)  # 当前序列长度
            mask = self.create_causal_mask(seq_len).to(query.device)
            # 扩展维度：[seq_len, seq_len] -> [1, 1, seq_len, seq_len]
            mask = mask.unsqueeze(0).unsqueeze(0)

        #调用自注意力方法
        atten_result= self_attention(query, key, value, mask=mask,dropout=self.dropout)
        #最终形状变换+linear
        result=atten_result.transpose(1,2).contiguous().view(self.batch_size,-1, self.n_head*self.head_dim)
        result=self.linears[-1](result)
        return result

#定义前馈全连接层(ok)
class FeedForward(nn.Module):
    def __init__(self, embedding_dim, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, embedding_dim)
    def forward(self, x):#x为第一个子层的输出
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

#定义层规范化层(ok)
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps)+self.b_2

# 定义子层连接层（post-norm）(ok)
class SublayerConnection(nn.Module):
    def __init__(self, size ,dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.size = size
        self.norm = LayerNorm(size)
    def forward(self, x, sublayer):
        result=x+self.dropout(self.norm(sublayer(x)))
        return result

# 定义编码器层(这是一层完整的编码器层，所以需要传入完整参数：atten、feedforward、size、dropout)(ok)
class EncoderLayer(nn.Module):
    def __init__(self, size, self_atten,feed_forward,dropout=0.1):
        super().__init__()
        self.self_atten=self_atten
        self.feed_forward=feed_forward
        self.size = size
        self.sublayers = clones(SublayerConnection(size, dropout),2)
    def forward(self, x):
        x=self.sublayers[0](x, lambda x:self.self_atten(x,x,x))
        x=self.sublayers[1](x, lambda x:self.feed_forward(x))
        return x

#定义编码器(ok)
class Encoder(nn.Module):
    def __init__(self, layer,N):
        super().__init__()
        self.layers=clones(layer,N)
        self.N=N
        self.norm=LayerNorm(layer.size)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

#定义解码器层（由三个子层连接结构构成）(ok)
class DecoderLayer(nn.Module):
    def __init__(self,size,self_atten,src_atten,feed_forward,dropout=0.1):
        super().__init__()
        self.size = size
        self.self_atten=self_atten
        self.src_atten=src_atten
        self.feed_forward=feed_forward
        self.dropout=nn.Dropout(dropout)
        self.sublayers=clones(SublayerConnection(size,dropout),3)
    def forward(self,y,encoder_output):
        #将y送入第一个子层连接结构得到多头自注意力+norm+add之后的结果
        result1=self.sublayers[0](y,lambda y:self.self_atten(y,y,y))
        #将result1送入第二个子层连接结构得到自注意力+norm+add之后的结果
        result2=self.sublayers[1](result1,lambda result1 :self.src_atten(result1,encoder_output,encoder_output))
        #将result2送入第三个子层连接结构得到前馈全连接+norm+add之后的结果
        result=self.sublayers[2](result2,lambda result2:self.feed_forward(result2))
        return result

#定义解码器(ok)
class Decoder(nn.Module):
    def __init__(self,layer,N):
        super().__init__()
        self.layers=clones(layer,N)
        self.N=N
        self.norm=LayerNorm(layer.size)
    def forward(self,y,encoder_output):
        for layer in self.layers:
            y=layer(y,encoder_output)
        return self.norm(y)

#定义输出层(ok)
class Generator(nn.Module):
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.linear = nn.Linear(d_model,vocab_size)
    def forward(self,input):
        return torch.log_softmax(self.linear(input), dim=-1)





