import torch
import torch.nn as nn
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


#填充标记
PAD=0
#起始标志
SOS=1
#结束标志
EOS=2
#未知字符
UNK=3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = 1044 # 词表大小

path= r'C:\Users\zhang\Desktop\AI_display\transformer\financial_corpus.txt'

def load_data(path):
    with open(path,'r',encoding='utf-8') as f:
        whole_text = f.read()
        with open(path, 'r', encoding='utf-8') as f:
            data=f.readlines()
    data=data[5:]

    # 去掉每行末尾的换行符
    data = [line.rstrip('\n') for line in data]


    #按照奇偶索引切片取出内容和摘要
    content=data[0:len(data):2]
    summary=data[1:len(data):2]

    #初始化pair对
    pairs=[]

    #将内容和对应的摘要加入pair对
    for i in range(len(content)):
        pairs.append([content[i],summary[i]])


    return whole_text, pairs

def word2vec():
    whole_text, _ = load_data(path)
    word2vec_dict={'PAD': 0, 'SOS': 1,'EOS': 2, 'UNK': 3}

    for word in whole_text:
        if word not in word2vec_dict:
            word2vec_dict[word]=len(word2vec_dict)

    index2word = {v: k for k, v in word2vec_dict.items()}

    return index2word, word2vec_dict



class TransformerSummarizer(nn.Module):
    """完整的Transformer摘要生成器"""

    def __init__(self, vocab_size, d_model=256, n_head=4, num_layers=6,
                 max_src_len=512, max_tgt_len=100, dropout=0.1):
        super().__init__()

        # 保存配置
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_head = n_head
        self.num_layers = num_layers
        self.dropout = dropout

        # 1. Embedding层（共享权重或分开）
        self.src_embedding = Embedding(vocab_size, d_model)
        self.tgt_embedding = Embedding(vocab_size, d_model)

        # 2. 位置编码
        self.src_pos_encoding = PositionalEncoding(
            embedding_size=d_model,
            dropout_p=dropout,
            max_len=max_src_len
        )
        self.tgt_pos_encoding = PositionalEncoding(
            embedding_size=d_model,
            dropout_p=dropout,
            max_len=max_tgt_len
        )

        # 3. 编码器
        encoder_layer = EncoderLayer(
            size=d_model,
            self_atten=MultiHeadAttention(n_head=n_head, embedding_dim=d_model, dropout=dropout),
            feed_forward=FeedForward(embedding_dim=d_model, d_ff=512, dropout=dropout),
            dropout=dropout
        )
        self.encoder = Encoder(layer=encoder_layer, N=num_layers)

        # 4. 解码器
        decoder_layer = DecoderLayer(
            size=d_model,
            self_atten=MultiHeadAttention(n_head=n_head, embedding_dim=d_model, dropout=dropout, is_causal=True),
            src_atten=MultiHeadAttention(n_head=n_head, embedding_dim=d_model, dropout=dropout, is_causal=False),
            feed_forward=FeedForward(embedding_dim=d_model, d_ff=512, dropout=dropout),
            dropout=dropout
        )
        self.decoder = Decoder(layer=decoder_layer, N=num_layers)

        # 5. 输出层
        self.generator = Generator(d_model=d_model, vocab_size=vocab_size)

        # 6. 初始化参数
        self._init_parameters()

    def _init_parameters(self):
        """参数初始化"""
        for p in self.parameters():
            if p.dim() > 1: # 只初始化权重（维度>1），不初始化偏置（维度=1）
                nn.init.xavier_uniform_(p)

    def encode(self, src):
        """编码器前向传播"""
        # src: [batch, src_len]
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.src_pos_encoding(src_emb)
        memory = self.encoder(src_emb)
        return memory

    def decode(self, tgt, memory):
        """解码器前向传播"""
        # tgt: [batch, tgt_len] (包含SOS)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_pos_encoding(tgt_emb)
        decoder_output = self.decoder(tgt_emb, memory)
        return decoder_output

    def forward(self, src, tgt): # 训练过程批量预测
        """完整的前向传播（训练用）"""
        # src: [batch, src_len] - 文章
        # tgt: [batch, tgt_len] - 摘要（包含SOS和EOS）

        # 1. 编码
        memory = self.encode(src)

        # 2. 解码（输入去掉最后一个token: EOS）
        decoder_input = tgt[:, :-1]  # 去掉EOS
        decoder_output = self.decode(decoder_input, memory)

        # 3. 生成
        output = self.generator(decoder_output)  # [batch, tgt_len-1, vocab_size]

        return output

    def generate(self, src, max_len=100, temperature=1.0): # 测试过程逐token预测
        """生成摘要（推理用）"""
        batch_size = src.size(0)

        # 编码
        memory = self.encode(src)

        # 初始化tgt（只有SOS）
        tgt = torch.full((batch_size, 1), SOS, device=src.device, dtype=torch.long)

        for _ in range(max_len - 1):
            # 解码
            decoder_output = self.decode(tgt, memory)

            # 预测下一个token
            # 先用temperature进行调节，然后用exp转换为真实概率（因为要用multinomial方法采样）
            next_token_logits = self.generator(decoder_output[:, -1:])  / temperature
            next_token_logits = torch.exp(next_token_logits)
            next_token = torch.multinomial(next_token_logits.squeeze(1), num_samples=1)  # 采样

            # 添加到序列
            tgt = torch.cat([tgt, next_token], dim=1)

            # 检查是否所有样本都生成了EOS
            if torch.all(next_token == EOS):
                break

        return tgt[:, 1:]  # 去掉SOS


def single_text_test(single_text,temperature,model_path = r'C:\Users\zhang\Desktop\AI_display\transformer\summary_300(best).pth', max_len=100):


    # 加载词表

    index2word, word2index = word2vec()

    # 加载测试数据
    _, pairs = load_data(path)


    # 编码函数（用词表）
    def encode_text(text):
        tokens = []
        for char in text[:512]:
            if char in word2index:
                tokens.append(word2index[char])
            else:
                tokens.append(UNK)  # 用UNK常量


        return tokens

    # 解码函数
    def decode_tokens(tokens, show_unk=True):
        """
        解码token序列为文本

        参数:
            tokens: token序列
            show_unk: 是否显示UNK标记
        """
        text = []

        for token in tokens:

            # 1. 转为python标量
            if isinstance(token, torch.Tensor):
                token = token.item()

            # 2. 检查是否在词表中
            if token in index2word:
                word = index2word[token]

            # 3. 处理不同标记
                if word == 'PAD':
                    continue  # 填充标记，不显示
                elif word == 'SOS':
                    continue  # 起始标记，不显示
                elif word == 'EOS':
                    break  # 结束标记，停止解码
                elif word == 'UNK':
                    if show_unk:
                        text.append('<UNK>')  # 显示UNK标记
                    # 或者: text.append('□')  # 显示为方框
                else:
                    text.append(word)  # 正常字
            else:
                # 如果token不在词表中（不应该发生）
                if show_unk:
                    text.append('<UNK>')

        return ''.join(text)

    # 加载模型

    model = TransformerSummarizer(vocab_size=vocab_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()


    # 编码
    src_token = encode_text(single_text)

    if len(src_token) > 512:
        src_token = src_token[:512]

    src_tensor = torch.tensor(src_token, dtype=torch.long)

    src_tensor = src_tensor.unsqueeze(0)

    # 预测
    with torch.no_grad():
        generated_tokens = model.generate(
            src_tensor,
            max_len=max_len,
            temperature=temperature
        )

    # 关键修改：将 [1, 100] 变成 [100]
    generated_tokens = generated_tokens.squeeze(0)

    # 解码和显示结果
    gen_text = decode_tokens(generated_tokens)

    return gen_text


if __name__ == '__main__':
    gen_text = single_text_test(single_text='近日，中国人民银行宣布了一系列旨在促进经济稳定增长的货币政策调整。首先，央行决定下调金融机构存款准备金率0.5个百分点，以释放更多流动性支持实体经济。此举预计将向市场注入约1万亿元人民币的资金。此外，央行还表示将通过定向中期借贷便利（TMLF）工具，为中小银行提供低成本资金，以增强其服务小微企业和民营企业的能力。同时，为了进一步加强金融监管，银保监会发布新规，要求商业银行严格控制房地产贷款比例，并加强对互联网金融平台的监管力度，确保金融市场健康发展。在保险领域，监管机构鼓励保险公司开发更多创新型产品，满足消费者多样化需求。资本市场方面，证监会推出多项措施，包括优化股票发行注册制、扩大外资参与A股市场的渠道等，以提高市场活力。金融科技领域，央行正积极推进数字货币的研发与试点，预计将在未来几年内逐步推广使用。国际贸易方面，中国政府继续推动"一带一路"倡议，加强与沿线国家的经贸合作，促进全球经济复苏。这些政策的出台，体现了政府对宏观经济调控的决心，旨在实现高质量发展。',temperature=0.7)
    print(gen_text)