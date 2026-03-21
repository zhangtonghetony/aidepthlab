# 导包
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import re
import json

# 指定训练硬件
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')
# 填充标记
PAD=0
# 起始标志
SOS=1
# 结束标志
EOS=2
# 未知字符
UNK=3
# 最大句子长度
Max_length=150
# 数据文件路径
path=r'C:\Users\zhang\Desktop\AI_display\gru\corpus_train.txt'

# 定义模型训练超参数
my_lr=0.001


def dynamic_teacher_forcing(epoch, total_epochs=50):
    """
    动态调整Teacher Forcing比例（50轮训练策略）

    策略：训练前期用高TF（学正确模式），后期用低TF（让模型自己生成）

    Args:
        epoch: 当前轮次（0-based）
        total_epochs: 总训练轮数
    Returns:
        tf_ratio: 当前轮次的Teacher Forcing比例
    """
    if epoch < 30:  # 前30轮：高TF（建立基础模式）
        return 0.8
    elif epoch < 40:  # 中间10轮：中等TF（逐步放手）
        return 0.5
    else:  # 最后10轮：低TF（让模型自己飞）
        return 0.2

# 定义字符串清洗函数
def normalize(s):
    s=s.lower().strip()
    s=re.sub(r'([?.!。，,？！：:])', r' \1 ', s)
    s = re.sub(r'[ ]+', " ", s)  # 将连续空格合并成单个空格
    #print(s)
    return s

# 定义函数，获得pair对
def get_pair(path):
    my_pair = []
    with open(path,'r',encoding='utf-8') as f:
        data=f.readlines()
        for i in data:
            chinese, english=i.strip().split(' ',1)# todo:split()方法可传入maxsplit参数控制最大切分次数
            # 在这里调用清洗函数（中文不需要，因为按字符分词）
            english = normalize(english)
            my_pair.append([chinese, english.strip()])
    return my_pair

# 定义函数，获取语料长度信息
def get_len():
    my_pair=get_pair(path)
    chinese_len = []
    english_len = []
    for i in my_pair:
        chinese_len.append(len(i[0]))
        english_len.append(len(i[1]))
    # 打印中英文语料平均和最大长度
    print(f"中文平均长度: {np.mean(chinese_len):.1f} 字")
    print(f"英文平均长度: {np.mean(english_len):.1f} 字符")
    print(f"中文最大长度: {max(chinese_len)} 字")
    print(f"英文最大长度: {max(english_len)} 字符")

#构建语料中英文字典
def word2vec():
    my_pair=get_pair(path)
    english_word2index = {'PAD': 0, 'SOS': 1,'EOS': 2, 'UNK': 3}
    english_word2index_n = 4
    chinese_word2index = {'PAD': 0, 'SOS': 1,'EOS': 2, 'UNK': 3}
    chinese_word2index_n = 4
    #chinese_char_freq = {}
    for pair in my_pair:
        for char in pair[0]:
            if char not in chinese_word2index:
                chinese_word2index[char] = chinese_word2index_n
                chinese_word2index_n = chinese_word2index_n + 1
            # chinese_char_freq[char] = chinese_char_freq.get(char, 0) + 1  # todo:字典的get方法
        for word in pair[1].split(' '):
            if word not in english_word2index:
                english_word2index[word] = english_word2index_n
                english_word2index_n = english_word2index_n + 1
    # print('去重后英文单词总数：', len(english_word2index))
    # print('去重后中文字符单词总数：', len(chinese_word2index))
    # sorted_chars = sorted(chinese_char_freq.items(), key=lambda x: x[1], reverse=True)#todo:sorted内置函数
    # for char, freq in sorted_chars[:20]:
    #     print(f"'{char}': {freq}次")
    chinese_index2word={v:k for k,v in chinese_word2index.items()}
    english_index2word={v:k for k,v in english_word2index.items()}
    return (my_pair,chinese_word2index,english_word2index,chinese_index2word,
            english_index2word,len(chinese_word2index),len(english_word2index))

#构建dataset对象
class TranslationDataset(Dataset):
    def __init__(self, my_pair):
        super().__init__()
        self.my_pair = my_pair
        self.sample_len = len(my_pair)
    def __len__(self):
        return self.sample_len
    def __getitem__(self, index):
        result=word2vec()
        # 异常值修正
        index = min(max(0, index), self.sample_len - 1)
        # 根据索引取出样本
        x = self.my_pair[index][0]
        y = self.my_pair[index][1]
        # 文本数值化
        x2index = [result[1][word] for word in x]
        y2index = [result[2][word] for word in y.split(' ')]
        y2index.append(EOS)
        #句子长度补齐（先加EOS再补齐）
        if len(x2index)<Max_length:
            x2index=x2index+[0]*(Max_length-len(x2index))
        if len(y2index)<Max_length:
            y2index=y2index+[0]*(Max_length-len(y2index))
        # 封装为张量
        tensor_x = torch.tensor(x2index, dtype=torch.long, device=device)
        tensor_y = torch.tensor(y2index, dtype=torch.long, device=device)
        return tensor_x, tensor_y

#构建dataloader
def get_dataloader():
    my_dataset=TranslationDataset(my_pair=get_pair(path))
    train_dataloader = DataLoader(my_dataset, batch_size=16, shuffle=True,drop_last=True)
    return train_dataloader

#定义encoder类
class EncoderGRU(nn.Module):
    def __init__(self,vocab_size,batch_size,hidden_size,num_layers=1):
        super().__init__()
        self.vocab_size=vocab_size
        self.batch_size=batch_size
        self.hidden_size=hidden_size
        self.num_layers=1
        self.embedding = nn.Embedding(vocab_size,hidden_size)
        self.gru = nn.GRU(input_size=hidden_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
    def forward(self,x,h0):
        output,hn=self.gru(self.embedding(x),h0)
        return output,hn
    def init_hidden(self):
        h0 = torch.zeros(self.num_layers,self.batch_size,self.hidden_size,device=device)
        return h0

#定义带attention的解码器类（需大改，因为batch_size不再为1）
class Attention_decoder(nn.Module):
    def __init__(self, eng_vocab_size, hidden_size=256, num_layers=1,
                 max_length=150, dropout_p=0.1):
        super().__init__()
        self.vocab_size = eng_vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_length = max_length

        # 词嵌入层
        self.embedding = nn.Embedding(eng_vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)

        # Attention机制（适配批量训练）
        self.attention = nn.Linear(hidden_size * 2, 1)  # 计算Attention分数
        self.attention_combine = nn.Linear(hidden_size * 2, hidden_size)

        # GRU层
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # 输出层
        self.output_layer = nn.Linear(hidden_size, eng_vocab_size)

    def forward(self, decoder_input, encoder_hidden, encoder_outputs):
        """
        批量训练版Attention Decoder

        Args:
            decoder_input: [batch_size, 1] - 当前解码的token（Teacher Forcing时）
            encoder_hidden: [num_layers, batch_size, hidden_size] - Encoder最后隐状态
            encoder_outputs: [batch_size, seq_len, hidden_size] - Encoder所有隐状态
        Returns:
            output: [batch_size, vocab_size] - 下一个token的预测概率
            hidden: [num_layers, batch_size, hidden_size] - 新的隐状态
            attention_weights: [batch_size, 1, seq_len] - Attention权重
        """
        batch_size = decoder_input.size(0)

        # 1. 词嵌入
        embedded = self.embedding(decoder_input)  # [batch_size, 1, hidden_size]
        embedded = self.dropout(embedded)

        # 2. GRU一步前向传播
        output, hidden = self.gru(embedded, encoder_hidden)  # output: [batch_size, 1, hidden_size]

        # 3. Attention机制
        # 将Decoder输出扩展到Encoder序列长度
        output_expanded = output.expand(-1, encoder_outputs.size(1), -1)  # [batch_size, enc_len, hidden_size]

        # 计算Attention分数
        energy_input = torch.cat((output_expanded, encoder_outputs), dim=2)  # [batch_size, enc_len, hidden_size*2]
        energy = torch.tanh(self.attention(energy_input))  # [batch_size, enc_len, 1]

        # 计算Attention权重
        attention_weights = F.softmax(energy, dim=1)  # [batch_size, enc_len, 1]

        # Attention加权求和
        context = torch.bmm(attention_weights.transpose(1, 2), encoder_outputs)  # [batch_size, 1, hidden_size]

        # 4. 合并Decoder输出和Attention上下文
        combined_input = torch.cat((output, context), dim=2)  # [batch_size, 1, hidden_size*2]
        attention_output = F.relu(self.attention_combine(combined_input))  # [batch_size, 1, hidden_size]

        # 5. 最终预测
        output = self.output_layer(attention_output.squeeze(1))  # [batch_size, vocab_size]
        output = F.log_softmax(output, dim=1)

        return output, hidden, attention_weights

    def init_hidden(self, batch_size):
        """根据batch_size动态初始化隐状态"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h0




def translate2eng(encoder, decoder, chinese_sentence, max_length=150):
    """
    Args:
        encoder: 训练好的编码器
        decoder: 训练好的解码器
        chinese_sentence: 中文句子
        expected_english: 期望英文翻译（用于BLEU评分，可选）
        max_length: 最大生成长度
    Returns:
        english_translation: 英文翻译结果
        bleu_score: BLEU分数（如有参考翻译）
    """
    encoder.eval()
    decoder.eval()
    result = word2vec()
    vocab_size = result[5]
    eng_vocab_size = result[6]
    with torch.no_grad():
        # 中文预处理
        chinese_indices = []
        for char in chinese_sentence:
            chinese_indices.append(result[1].get(char, UNK))

        # 补齐或截断到Max_length
        if len(chinese_indices) < Max_length:
            chinese_indices += [PAD] * (Max_length - len(chinese_indices))
        else:
            chinese_indices = chinese_indices[:Max_length]

        # 编码
        x = torch.tensor(chinese_indices, dtype=torch.long,device=device).unsqueeze(0)
        h0 = encoder.init_hidden()
        encoder_outputs, encoder_hidden = encoder(x, h0)

        # 4. 解码（独立停止）
        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor([[SOS]], device=device)
        english_tokens = []

        for t in range(max_length):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            top_token = torch.argmax(decoder_output, dim=1).item()

            if top_token == EOS:
                break
            elif top_token == PAD:  # 关键修复：跳过PAD
                # 如果预测PAD，跳过不添加到结果中
                # 但继续用PAD作为下一个输入（保持维度一致）
                decoder_input = torch.tensor([[PAD]], device=device)
                continue
            else:
                english_tokens.append(top_token)
                # 测试时用预测概率最大的词的索引当做下一个输入
                decoder_input = torch.tensor([[top_token]], device=device)

        # 5. 转为英文
        english_words = [result[4].get(token, '<UNK>') for token in english_tokens]
        english_translation = ' '.join(english_words)


        return english_translation

path1=r'C:\Users\zhang\Desktop\AI_display\gru\translation_encoder.pth'
path2=r'C:\Users\zhang\Desktop\AI_display\gru\translation_decoder.pth'

# 生成函数
def single_text_translate(text,path1=path1,path2=path2):
    result=word2vec()
    vocab_size = result[5]
    eng_vocab_size = result[6]
    hidden_size = 256
    encoder = EncoderGRU(vocab_size, 1,hidden_size).to(device)
    decoder = Attention_decoder(eng_vocab_size).to(device)
    # 加载模型
    encoder.load_state_dict(torch.load(path1))
    decoder.load_state_dict(torch.load(path2))

    translation = translate2eng(encoder, decoder, text)

    #统计UNK数量
    unk_count = translation.count('<UNK>')
    word_count = len(translation.split())
    unk_ratio = unk_count / word_count if word_count > 0 else 0

    return translation, unk_ratio


if __name__=='__main__':
    translation , unk = single_text_translate('股票市场今日开盘后出现大幅震荡。')
    print(translation)
    print(unk)
    print(len(translation.split(' ')))
    #debug_first_token_prediction()
    #train()




