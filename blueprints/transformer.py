from flask import render_template, request, Blueprint, jsonify
from transformer.single_text import single_text_test


transformer_bp = Blueprint('transformer', __name__)


@transformer_bp.route('/transformer', methods=['GET'])
def transformer_page():  # 建议函数名与路由一致
    return render_template('transformer.html')


@transformer_bp.route('/transformer/generate', methods=['POST'])
def transformer_generate():
    try:
        data = request.get_json()

        # 1. 严格校验
        if not data or 'text' not in data:
            return jsonify({"success": False, "message": "无效的请求：缺少文本"}), 400

        text = data['text'].strip()
        # 给温度一个默认值，并强制转为 float
        try:
            temperature = float(data.get('temperature', 1.0))
            if temperature <= 0: temperature = 0.1  # 防止除零错误
        except (ValueError, TypeError):
            temperature = 1.0

        if not text:
            return jsonify({"success": False, "message": "请输入待处理文本"}), 400

        # 2. 调用预测逻辑
        # 注意：如果 single_text_test 内部还在读文件/载模型，建议重构为只传 text 和 temp
        gen_text = single_text_test(text, temperature)

        # 3. 返回结果
        return jsonify({
            "success": True,
            "text": gen_text
        }), 200

    except Exception as e:
        # 捕捉模型推理过程中可能的报错（如 OOM 或 长度溢出）
        print(f"Prediction Error: {e}")
        return jsonify({"success": False, "message": "模型生成失败，请稍后重试"}), 500