import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import BertTokenizer,BertModel
from torch.optim import AdamW
import time
import shap
import pandas as pd

path= 'financial_comments_corpus.txt'


data = []

# 简单的转换逻辑，将txt文件转换为csv文件，适配“真香定律”
def transform2csv():
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            text = lines[i].strip()
            label = int(lines[i+1].strip()) - 1 # 将标签归零化，因为PyTorch 的 CrossEntropyLoss 默认要求标签从 0 开始
            data.append({'text': text, 'label': label}) # pandas生成scv文件标准格式：[{},{},...]
    pd.DataFrame(data).to_csv('financial_comments_corpus.csv', index=False) # index=False使得索引不作为scv文件的列

# 获取数据
def get_data(): # load_dataset方法封装了len/getitem方法，还新增了切片取数据功能

    '''
    :return: 训练dataset和测试dataset
    '''

    # 读取训练数据集
    train_dataset = load_dataset('csv',data_files="financial_comments_corpus.csv", split="train")
    #test_dataset = load_dataset('csv',data_files="./test.csv", split="train")

    return train_dataset

# 加载分词器
bert_tokenizer = BertTokenizer.from_pretrained('C:/Users/zhang/Desktop/AI-learning/models/bert-base-chinese')

# 加载模型
bert_model=BertModel.from_pretrained('C:/Users/zhang/Desktop/AI-learning/models/bert-base-chinese')

# 自定义函数处理dataset中的数据
def collate_fn(batch):
    """
    :param batch: 是一个列表，每个元素是 dataset[i] 返回的一个字典
                  格式类似: [{'text': '...', 'label': 0}, {'text': '...', 'label': 1}, ...]
    :return: inputs_ids,attention_mask,token_type_ids,labels_y
    """

    # 1. 提取出当前 batch 里的所有文本和所有标签
    # batch 里的数据是 [{'text':t1, 'label':l1}, {'text':t2, 'label':l2}...]
    # 我们需要把它们拆解成两个列表：[t1, t2] 和 [l1, l2]
    texts = [item['text'] for item in batch]
    labels = [item['label'] for item in batch]

    # 2. 使用 bert_tokenizer 进行批量分词处理（这是最核心的一步）
    # 我们在这里调用 batch_encode_plus
    inputs = bert_tokenizer.batch_encode_plus(
        texts,
        padding='max_length',      # 自动补齐
        truncation=True,           # 自动截断：超过 BERT 默认 512 或指定长度的会被切掉
        max_length=128,            # 设定最大长度
        return_tensors='pt' ,      # 非常重要：直接返回 PyTorch 张量，而不是普通的 list
        return_length=True
    )

    # 3. 将标签也转换为张量
    # 注意：CrossEntropyLoss 需要 LongTensor 类型的标签
    labels_y = torch.tensor(labels, dtype=torch.long)

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']

    return  input_ids,attention_mask,token_type_ids,labels_y

# 获取dataloader
def get_dataloder():

    '''
    :return:训练dataloader
    '''

    train_dataset = get_data()
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                                  collate_fn=collate_fn,drop_last=True)
    return train_dataloader

# 预训练模型+微调
class MyModel(nn.Module):
    def __init__(self,pretrained_model):
        super().__init__()
        self.bert_model = pretrained_model
        self.fc_1 = nn.Linear(768,256)
        self.fc_2 = nn.Linear(256,3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self,input_ids,attention_mask,token_type_ids):
        # 将上述三个参数送入预训练模型
        result = self.bert_model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)

        # 使用pooler_output
        output = self.fc_1(result['pooler_output'])
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc_2(output)

        return output


def train_model():
    # 1. 准备数据和实例化模型
    train_dataloader = get_dataloder()
    my_model = MyModel(pretrained_model=bert_model)

    # --- 核心修改：解冻最后 4 层 Transformer (Layer 8, 9, 10, 11) ---
    # 先冻结所有参数
    for param in my_model.parameters():
        param.requires_grad = False

    # 解冻线性层 (fc_1, fc_2)
    for param in my_model.fc_1.parameters(): param.requires_grad = True
    for param in my_model.fc_2.parameters(): param.requires_grad = True

    # 解冻 BERT 的最后 4 层
    # BERT 内部层的命名通常包含 'layer.8', 'layer.9', 'layer.10', 'layer.11'
    unfreeze_layers = ['layer.8', 'layer.9', 'layer.10', 'layer.11', 'pooler']  # pooler层通常也建议解冻
    for name, param in my_model.bert_model.named_parameters():
        if any(layer in name for layer in unfreeze_layers):
            # 打印layer的name
            print(name)
            param.requires_grad = True

    # 2. 核心设置：差异化学习率 (Differential Learning Rate)
    # 给 BERT 层较小的 lr 防止破坏预训练特征，给 FC 层稍大的 lr 加速收敛
    bert_params = [p for n, p in my_model.named_parameters() if p.requires_grad and 'bert_model' in n]
    fc_params = [p for n, p in my_model.named_parameters() if p.requires_grad and 'fc_' in n]

    optim = AdamW([
        {'params': bert_params, 'lr': 2e-5},  # BERT层：微调
        {'params': fc_params, 'lr': 1e-4}  # 线性层：精练
    ], eps=1e-8)

    # 3. 损失函数
    weights = torch.tensor([1.5, 1.0, 1.0])  # 确保 weight 也在同一个设备上
    loss_func = nn.CrossEntropyLoss(weight=weights)
    my_model.train()
    epochs = 4  # 解冻后可以适当增加 1 个 epoch 观察效果
    loss_list = []

    print(f"开始训练 (已解冻最后4层)...")
    for epoch in range(epochs):
        start = time.time()
        total_loss = 0.0
        acc = 0.0
        idx = 0

        for i, (input_ids, attention_mask, token_type_ids, labels_y) in enumerate(train_dataloader):
            # 数据送入设备
            input_ids, attention_mask, token_type_ids, labels_y = \
                input_ids, attention_mask, token_type_ids, labels_y

            # 前向传播
            output = my_model(input_ids, attention_mask, token_type_ids)
            loss = loss_func(output, labels_y)

            # 反向传播
            optim.zero_grad()
            loss.backward()
            optim.step()

            # 计算准确率
            predicts = torch.argmax(output, dim=1)
            acc += (predicts == labels_y).sum().item() / len(labels_y)
            total_loss += loss.item()
            idx += 1

        avg_loss = total_loss / idx
        avg_acc = (acc / idx) * 100
        print(
            f'--- Epoch {epoch + 1}/{epochs} | 用时: {time.time() - start:.2f}s | 平均Loss: {avg_loss:.4f} | 平均Acc: {avg_acc:.2f}% ---')
        loss_list.append(total_loss)

    # 4. 保存
    torch.save(my_model.state_dict(), 'my_model.pth')
    print("针对解冻层的权重已保存。")
    return loss_list

if __name__ == '__main__':
    # train_dataset = get_data()
    # print(train_dataset[0])
    # train_dataloader = get_dataloder()
    # for input_ids,attention_mask,token_type_ids,labels_y in train_dataloader:
    #     print(input_ids.shape)
    #     print(labels_y.shape)
    #     break
    # transform2csv()
    df = pd.read_csv('financial_comments_corpus.csv')
    print(df['label'].value_counts())
    #train_model()

