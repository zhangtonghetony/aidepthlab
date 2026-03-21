from flask import render_template, request, Blueprint, jsonify

from gru.translation import single_text_translate

gru_bp = Blueprint('gru', __name__)

@gru_bp.route('/gru', methods=['GET'])
def gru():
    return render_template('gru.html')

@gru_bp.route('/gru/translate', methods=['POST'])
def gru_translate():
    data = request.get_json()

    # 严格校验
    if not data or 'text' not in data:
        return jsonify({"success": False, "message": "无效的请求：缺少文本"}), 400

    text = data['text'].strip()

    if not text:
        return jsonify({"success": False, "message": "请输入待处理文本"}), 400

    try:
        result = single_text_translate(text)

        return jsonify({"success": True, "result": result}),200

    except Exception as e:
        # 捕捉模型推理过程中可能的报错（如 OOM 或 长度溢出）
        print(f"Prediction Error: {e}")

        return jsonify({"success": False, "message": "模型生成失败，请稍后重试"}), 500


