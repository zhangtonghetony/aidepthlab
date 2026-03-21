import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


# 1. 严格按照原有模型结构定义，否则会报错
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


# --- 环境准备 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = "./bert_risk_model"  # 存放权重和配置的文件夹

# 2. 加载 Tokenizer 和基础 BERT 骨架
print("正在初始化基础模型...")
tokenizer = BertTokenizer.from_pretrained(model_dir)
bert_base = BertModel.from_pretrained(model_dir)

# 3. 实例化并加载保存的 .pth 权重 (load_state_dict)
model = MyModel(bert_base)
# 关键：加载训练好的参数到模型中
model.load_state_dict(torch.load(f"{model_dir}/my_model.pth", map_location=device))
print(model.fc_1.weight[:1, :5]) # 看看这几个数，如果全是 0 或非常规律的数，说明没加载对。
model.to(device)
model.eval()  # 切换到评估模式，关闭 Dropout
print("权重加载成功！开始测试...\n")


# --- 测试数据处理 ---
def run_test(file_path):
    correct = 0
    total = 0
    # 统计每个类别的准确情况：{标签: [正确数, 总数]}
    stats = {0: [0, 0], 1: [0, 0], 2: [0, 0]}
    label_map = {0: "高风险", 1: "中风险", 2: "低风险"}

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with torch.no_grad():
        for line in lines:
            if '|' not in line: continue
            text, label_str = line.strip().split('|')
            true_label = int(label_str)

            # 对文本进行预处理
            inputs = tokenizer(text, return_tensors="pt", padding=True,
                               truncation=True, max_length=128).to(device)

            # 模型推理
            logits = model(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])
            pred_label = torch.argmax(logits, dim=1).item()

            # 统计
            total += 1
            stats[true_label][1] += 1
            if pred_label == true_label:
                correct += 1
                stats[true_label][0] += 1
                result_mark = "正确"
            else:
                result_mark = f"错误 (预测为:{pred_label})"

            print(f"{result_mark} | 实际:{true_label} | 文本: {text[:30]}...")

    # --- 输出最终报告 ---
    print("\n" + "=" * 40)
    print(f"测试总结 (共 {total} 条样本)")
    print(f"总准确率: {correct / total * 100:.2f}%")
    print("-" * 40)
    for lb, val in stats.items():
        recall = (val[0] / val[1] * 100) if val[1] > 0 else 0
        print(f"[{label_map[lb]}] 识别准确率: {recall:.2f}% ({val[0]}/{val[1]})")
    print("=" * 40)


# 运行测试
if __name__ == "__main__":
    run_test("test.txt")