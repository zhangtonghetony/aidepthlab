"""
Flask应用主入口
"""
from flask import Flask, render_template
from primary_rag_agent.utils.config import chroma_config

# 创建Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ai-is-best'

# 配置文件上传
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB限制
app.config['UPLOAD_FOLDER'] = chroma_config['upload_dir']



# 注册蓝图
from blueprints.main import main_bp

from blueprints.rag_agent import rag_agent_bp

from blueprints.login import login_bp

from blueprints.transformer import transformer_bp

from blueprints.bert import bert_bp

from blueprints.gru import gru_bp

app.register_blueprint(main_bp)

app.register_blueprint(rag_agent_bp)

app.register_blueprint(login_bp)

app.register_blueprint(bert_bp)

app.register_blueprint(transformer_bp)

app.register_blueprint(gru_bp)

# 404错误处理
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(port=5000)