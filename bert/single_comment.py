import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F

model_dir = r"C:\Users\zhang\Desktop\AI_display\bert\bert_risk_model"  # 存放权重和配置的文件夹
tokenizer = BertTokenizer.from_pretrained(model_dir)
bert_base = BertModel.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 严格按照原有模型结构定义，否则会报错
class MyModel(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.bert_model = pretrained_model
        self.fc_1 = nn.Linear(768, 256)
        self.fc_2 = nn.Linear(256, 3)  # 0:高, 1:中, 2:低
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask, token_type_ids):


        result = self.bert_model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids)
        # 使用 pooler_output
        output = self.fc_1(result['pooler_output'])
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc_2(output)
        return output

model = MyModel(bert_base)
model.load_state_dict(torch.load(f"{model_dir}/my_model.pth", map_location=device))
model.eval()  # 切换到评估模式，关闭 Dropout

class BertSingleComment():
    def __init__(self):
        self.model = model

    def predict_risk_level(self,comment : str):

        with torch.no_grad():
            # 对文本进行预处理
            inputs = tokenizer(comment, return_tensors="pt", padding=True,
                               truncation=True, max_length=128).to(device)

            logits = model(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])
        predict_label = torch.argmax(logits, dim=1).item()

        return predict_label

    def predict_probs(self, comments: list):
        """为 LIME 准备的新逻辑：列表输入，返回概率矩阵"""
        with torch.no_grad():
            # 这里的 comments 是 LIME 生成的几十个文本变体
            inputs = tokenizer(comments, return_tensors="pt", padding=True,
                                    truncation=True, max_length=128).to(device)

            # 得到 logits: [batch_size, 3]
            outputs = self.model(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])

            # 必须用 softmax 转化为概率，且转为 numpy 格式
            probs = F.softmax(outputs, dim=1)
            return probs.numpy()










single_comment_predict = BertSingleComment()