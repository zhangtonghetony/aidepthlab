from flask import Flask, render_template, request, Blueprint, jsonify
# 确保你的路径导入正确
from bert.single_comment import single_comment_predict
from lime.lime_text import LimeTextExplainer

bert_bp = Blueprint('bert', __name__)

# 初始化解释器
explainer = LimeTextExplainer(class_names=['High Risk', 'Medium Risk', 'Low Risk'])

@bert_bp.route('/bert', methods=['GET'])
def bert():
    return render_template('bert.html')


@bert_bp.route('/bert/predict', methods=['POST'])
def predict():
    # 1. 安全获取 JSON 数据
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"success": False, "message": "无效的请求格式"}), 400

    comment = data['text'].strip()

    # 2. 验证输入是否为空
    if not comment:
        return jsonify({"success": False, "message": "评论内容不能为空"}), 400

    try:
        # 3. 利用 BERT 模型进行推理
        # 假设 predict_risk_level 返回的是整数 0, 1, 2
        predict_label = single_comment_predict.predict_risk_level(comment)

        # 4. 使用映射表替代多个 if
        risk_map = {
            0: '高风险评论，需特别关注',
            1: '中风险评论',
            2: '低风险评论，无异常'
        }

        result = risk_map.get(predict_label, "未知风险等级")

        return jsonify({
            "success": True,
            "result": result,
            "label": predict_label  # 顺便传回原始标签，方便前端做样式区分
        }), 200

    except Exception as e:
        # 捕捉模型推理过程中的意外错误（如显存溢出、分词失败等）
        return jsonify({"success": False, "message": f"模型推理失败: {str(e)}"}), 500


@bert_bp.route('/bert/explain', methods=['POST'])
def explain():
    data = request.get_json()
    comment = data.get('text', '').strip()

    # 拿到用户预测的标签，看看我们要解释哪一类
    # 比如模型预测是 0 (高风险)，那我们就解释为什么是 0
    target_label = data.get('label', 0)

    if not comment:
        return jsonify({"success": False, "message": "没有可解释的内容"}), 400

    try:
        # 重要：LIME 调用的函数必须是返回概率矩阵的那个
        def predictor_fn(texts):
            return single_comment_predict.predict_probs(texts)

        # explain_instance 会在内部多次调用 predictor_fn
        exp = explainer.explain_instance(
            comment,
            predictor_fn,
            num_features=5,  # 返回最重要的 5 个词
            labels=(target_label,), # 解释指定的类别
            num_samples=100
        )

        # 提取该类别的权重列表
        weights = exp.as_list(label=target_label)

        return jsonify({
            "success": True,
            "weights": weights  # 格式：[["垃圾", 0.45], ["不错", -0.12]]
        })
    except Exception as e:
        return jsonify({"success": False, "message": f"解释生成失败: {str(e)}"}), 500

