from openai import OpenAI
import time
import json
import random

client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


# 先测试2条
TEST_MODE = False  # 设置为True先测试2条
if TEST_MODE:
    total_samples = 5
else:
    total_samples = 1000

# 打开文件准备写入
txt_file = open("financial_comments_corpus.txt", "a", encoding="utf-8")

# 示例数据
example_comment = "今天白酒板块集体回调，茅台缩量跌了 2 个点，目前看回补了之前的缺口。打算在 1600 附近再观察一下，长线分批建仓的逻辑没变，心态要稳。"

example_risk_level = '3'

styles = ["愤青散户", "专业基金经理", "理性的老股民", "激进的短线客", "迷茫的新手", "金融自媒体","金融新闻发言人","行业研究专家"]
platforms = ["雪球论坛", "微博评论", "朋友圈短评", "专业研报摘要", "股吧黑话","36氪论坛"]

success_count = 0
failed_count = 0

for i in range(total_samples):
    try:

        current_style = random.choice(styles)
        current_platform = random.choice(platforms)

        # 构建消息
        messages = [
            {"role": "system",
             "content": "你是金融评论写手。生成金融评论（大于50字，少于100字）和对应体现的风险等级（输出数字，3代表低风险，2代表中风险，1代表高风险）。输出必须是纯JSON：{'comment':'评论内容','risk_level':'风险等级（数字）'}。不要任何解释、标记、额外文本。必须遵守字数要求。"},
            #{"role": "assistant",
             #"content": json.dumps({"comment": example_comment, "risk_level": example_risk_level}, ensure_ascii=False)},
            #{"role": "user",
             #"content": "再生成一条金融评论和对应风险等级，自定主题，内容完全不要模仿示例，避免与示例相似的句式结构和表达方式，但格式一定要按照要求。强调：必须是原创内容，不能与示例有任何相似性，自行判断风险等级并输出，必须按照格式要求输出。同时，输出中风险和低风险评论的概率不能过低。"}
            {"role":"user","content":f"请模仿在{current_platform}上的发帖风格，以{current_style}的口吻，生成一条金融评论和对应风险等级，自定主题,格式一定要按照要求。同时，输出中风险和低风险评论的概率不能过低。必须遵守字数要求。"}
        ]

        # 调用API
        response = client.chat.completions.create(
            model="qwen3-max",
            messages=messages,
            temperature=1.2,      # 温度适当调高
            top_p=1.0,            # 必须是 1.0，允许模型选择所有可能的词


        )

        # 解析响应
        content = response.choices[0].message.content.strip()
        print(f"\n原始响应预览: {content[:10]}...")


        # 解析JSON
        data = json.loads(content)

        # 验证长度
        comment = data["comment"].strip()

        comment_len = len(comment)


        if comment_len > 100:
            print(f"文章过长({comment_len}字)，截断为100字")
            comment = comment[:100]


        if comment_len < 30:
            print(f"文章过短({comment_len}字)，跳过")
            failed_count += 1
            continue

        # 写入文件（不要加"文章:"前缀，保持干净）
        txt_file.write(f"{comment}\n")
        txt_file.write(f"{data["risk_level"]}\n")

        success_count += 1

        # 进度显示
        print(f"已生成 {success_count}/{total_samples} 条")
        print(f"   评论: {comment_len}字")


        # 防限流
        if (i + 1) % 10 == 0:
            time.sleep(1)
        if (i + 1) % 100 == 0:
            time.sleep(3)

    except json.JSONDecodeError as e:
        print(f"第{i + 1}条JSON解析失败: {e}")
        print(f"   原始内容: {content[:15]}...")
        failed_count += 1
    except KeyError as e:
        print(f"第{i + 1}条缺少字段: {e}")
        failed_count += 1
    except Exception as e:
        print(f" 第{i + 1}条失败: {type(e).__name__}: {e}")
        failed_count += 1

# 更新文件头部
txt_file.close()

print("=" * 50)
print(f"生成完成")
print(f"成功: {success_count} 条")
print(f"失败: {failed_count} 条")
