# /blueprint/rag_agent.py
from flask import Blueprint, request, jsonify, current_app, render_template, session, send_from_directory
from datetime import datetime
import threading
import hashlib
import os
from primary_rag_agent.utils.log import logger
import uuid
from primary_rag_agent.rag.vector_store import vector_store
from primary_rag_agent.rag.rag import RagService
from primary_rag_agent.utils.config import chroma_config
from agent.agent import ReactAgent

# 创建蓝图
rag_agent_bp = Blueprint('rag_agent', __name__)

persist_directory = chroma_config['persist_dir']
upload_dir = chroma_config['upload_dir']

# MD5记录文件路径
MD5_FILE = chroma_config['MD5_FILE']
VECTOR_MD5_FILE = chroma_config['VECTOR_MD5_FILE']

# 全局上传锁，用于确保只有一个上传任务在处理向量化
upload_lock = threading.Lock()
# 当前向量化状态
vectorization_in_progress = False
# 向量化结果队列
vectorization_results = []

# 创建RAG服务实例
rag_service = RagService()

# 创建ReactAgent实例
react_agent = ReactAgent()

# 存储对话历史的最大轮数
MAX_HISTORY_ROUNDS = 5  # 不能过大，否则会超出session大小限制

# 报告存储目录
REPORTS_DIR = chroma_config['REPORTS_DIR']


def get_file_md5(file_path: str) -> str:
    """计算文件的MD5值"""
    chunk_size = 4096
    md5 = hashlib.md5()

    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)

    return md5.hexdigest()


def save_md5(md5_value: str, filename: str):
    """保存MD5值到记录文件"""
    with open(MD5_FILE, 'a', encoding='utf-8') as f:
        f.write(md5_value + '\n')


def check_md5(md5_value: str) -> bool:
    """检查MD5值是否已存在"""
    if not os.path.exists(MD5_FILE):
        return False

    with open(MD5_FILE, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if md5_value == line.strip():
                return True
    return False


def process_vectorization_async(file_path, original_filename):
    """异步处理向量化的线程函数
    Args:
        file_path: 上传的文件路径
        original_filename: 原始文件名
    """
    global vectorization_in_progress, vectorization_results

    try:
        logger.info(f"开始向量化处理: {original_filename} ({file_path})")

        # 处理单个文件的向量化
        success = vector_store.process_single_file(file_path, original_filename)

        if success:
            vectorization_results.append({
                'success': True,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'message': f'文件 "{original_filename}" 向量化完成'
            })
        else:
            vectorization_results.append({
                'success': False,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'message': f'文件 "{original_filename}" 向量化失败或已存在'
            })

    except Exception as e:
        vectorization_results.append({
            'success': False,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'message': f'向量化处理失败: {str(e)}'
        })
    finally:
        # 处理完成，释放锁
        with upload_lock:
            vectorization_in_progress = False


@rag_agent_bp.route('/rag_agent')
def rag_agent_page():
    """渲染RAG+Agent页面"""
    return render_template('rag_agent.html')


@rag_agent_bp.route('/rag_agent/api/upload', methods=['POST'])
def upload_file():
    """处理文件上传"""
    global vectorization_in_progress

    # 检查是否有文件上传
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'message': '没有选择文件'
        }), 400

    # 获取文件
    file = request.files['file']

    # 检查文件名
    if file.filename == '':
        return jsonify({
            'success': False,
            'message': '文件名为空'
        }), 400

    # 检查文件类型
    allowed_extensions = {'.txt', '.pdf'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        return jsonify({
            'success': False,
            'message': f'不支持的文件类型: {file_ext}，仅支持 txt 和 pdf 格式'
        }), 400

    # 检查文件大小 (限制10MB)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)  # 重置文件指针

    if file_size > 10 * 1024 * 1024:  # 10MB
        return jsonify({
            'success': False,
            'message': '文件过大，最大支持10MB'
        }), 400

    # 检查当前是否有向量化任务在进行
    with upload_lock:
        if vectorization_in_progress:
            return jsonify({
                'success': False,
                'message': '系统正在处理上一个文件的向量化，请稍后再试'
            }), 400

        # 设置向量化状态为进行中
        vectorization_in_progress = True

    try:
        # 生成唯一文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{uuid.uuid4().hex[:8]}{file_ext}"
        original_filename = file.filename  # 保存原始文件名用于日志

        # 确保上传目录存在
        os.makedirs(upload_dir, exist_ok=True)

        # 保存文件
        file_path = os.path.join(upload_dir, unique_filename)
        file.save(file_path)

        # 计算并检查MD5
        file_md5 = get_file_md5(file_path)

        if check_md5(file_md5):
            # 删除重复文件
            os.remove(file_path)
            # 释放向量化锁
            with upload_lock:
                vectorization_in_progress = False
            return jsonify({
                'success': False,
                'message': '该文件已存在于服务器中，无需重复上传'
            }), 400

        # 保存MD5记录
        save_md5(file_md5, unique_filename)

        # 启动异步向量化处理线程，传入文件信息
        vectorization_thread = threading.Thread(
            target=process_vectorization_async,
            args=(file_path, original_filename)
        )
        vectorization_thread.daemon = True
        vectorization_thread.start()

        # 返回成功响应
        return jsonify({
            'success': True,
            'message': '文件上传成功，正在后台进行向量化处理',
            'filename': unique_filename,
            'original_name': original_filename,
            'file_size': file_size,
            'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        # 发生异常时释放锁
        with upload_lock:
            vectorization_in_progress = False
        return jsonify({
            'success': False,
            'message': f'上传失败: {str(e)}'
        }), 500


@rag_agent_bp.route('/rag_agent/api/system-status', methods=['GET'])
def get_system_status():
    """获取系统状态（是否正在向量化）"""
    global vectorization_in_progress

    return jsonify({
        'success': True,
        'is_vectorizing': vectorization_in_progress,
        'message': '系统忙' if vectorization_in_progress else '系统空闲'
    })


def get_conversation_history(session_id):
    """从session中获取对话历史"""
    if 'conversation_history' not in session:
        session['conversation_history'] = []
    return session['conversation_history']


def add_to_history(session_id, role, content):
    """添加消息到历史记录，限制内容长度"""
    history = get_conversation_history(session_id)

    # 限制内容长度（最多100个字符），防止session大小超限
    if len(content) > 100:
        content = content[:100] + "..."

    # 限制历史记录长度
    if len(history) >= MAX_HISTORY_ROUNDS * 2:  # 每轮对话包含user和assistant两条消息
        history = history[2:]  # 删除最早的一轮对话

    history.append({
        'role': role,
        'content': content,
        'timestamp': datetime.now().isoformat(),
        'original_length': len(content)  # 可选：保存原始长度信息
    })

    session['conversation_history'] = history
    session.modified = True


def build_conversation_context(history):
    """构建会话上下文，用于Agent调用"""
    if not history:
        return ""

    # 将最近的几轮对话构建为上下文
    context_messages = []
    for msg in history[-MAX_HISTORY_ROUNDS * 2:]:  # 取最近N轮对话
        # 这里使用已截断的内容
        context_messages.append(f"{msg['role']}: {msg['content']}")

    return "\n".join(context_messages)


# 修改后的聊天API，使用Agent并保留历史记录
@rag_agent_bp.route('/rag_agent/api/chat', methods=['POST'])
def chat():
    """处理聊天请求，使用Agent并保留历史记录"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default_session')

        if not user_message:
            return jsonify({
                'success': False,
                'message': '消息不能为空'
            }), 400

        # 获取历史记录
        history = get_conversation_history(session_id)

        # 构建包含历史记录的查询
        context = build_conversation_context(history)
        full_query = ""
        if context:
            full_query = f"以下是对话历史记录，请根据历史记录理解上下文：\n{context}\n\n"
        full_query += f"用户提问：{user_message}"

        # 准备上下文参数
        agent_context = {
            'report': False,  # 初始化report为False
            'session_id': session_id,
            'history_length': len(history)
        }

        # 添加用户消息到历史记录
        add_to_history(session_id, 'user', user_message)

        # 调用Agent处理消息
        try:
            ai_response = react_agent.execute(full_query, agent_context)

            # 添加AI回复到历史记录
            add_to_history(session_id, 'assistant', ai_response)

            # 返回成功响应
            return jsonify({
                'success': True,
                'message': ai_response,
                'history_length': len(get_conversation_history(session_id))
            })

        except Exception as e:
            logger.error(f"Agent处理失败: {str(e)}")
            return jsonify({
                'success': False,
                'message': f'AI服务暂时不可用: {str(e)}'
            }), 500

    except Exception as e:
        logger.error(f"聊天处理失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'处理请求时出错: {str(e)}'
        }), 500


# 新增：下载报告API
@rag_agent_bp.route('/rag_agent/api/download-report/<filename>', methods=['GET'])
def download_report(filename):
    """下载生成的报告HTML文件"""
    try:
        # 安全检查，防止路径遍历攻击
        if '..' in filename or filename.startswith('/'):
            return jsonify({
                'success': False,
                'message': '文件名不合法'
            }), 400

        # 检查文件是否存在
        file_path = os.path.join(REPORTS_DIR, filename)
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'message': '文件不存在'
            }), 404

        # 检查是否是HTML文件
        if not filename.lower().endswith('.html'):
            return jsonify({
                'success': False,
                'message': '只能下载HTML格式的报告文件'
            }), 400

        # 返回文件
        return send_from_directory(
            REPORTS_DIR,
            filename,
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        logger.error(f"下载报告失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'下载失败: {str(e)}'
        }), 500


# 新增：获取报告列表API
@rag_agent_bp.route('/rag_agent/api/report-list', methods=['GET'])
def get_report_list():
    """获取属于当前用户且可下载的报告列表"""
    try:
        # 1. 获取当前登录用户的 user_id
        current_user = session.get('user_id')
        if not current_user:
            return jsonify({
                'success': False,
                'message': '未检测到登录状态，无法获取报告'
            }), 401

        if not os.path.exists(REPORTS_DIR):
            return jsonify({
                'success': True,
                'reports': []
            })

        reports = []
        user_prefix = f"{current_user}_" # 定义匹配前缀

        for filename in os.listdir(REPORTS_DIR):
            # 2. 核心逻辑：必须以 user_id 开头，且是 .html 文件
            if filename.startswith(user_prefix) and filename.lower().endswith('.html'):
                file_path = os.path.join(REPORTS_DIR, filename)
                file_stat = os.stat(file_path)

                # 提取标题：去掉 user_id 前缀和扩展名
                # 假设文件名格式是: {user_id}_{safe_title}_{timestamp}_{uuid}.html
                raw_title = os.path.splitext(filename)[0]
                display_title = raw_title.replace(user_prefix, "", 1)

                reports.append({
                    'filename': filename,
                    'title': display_title, # 展示给用户看的不带 ID 的标题
                    'size': file_stat.st_size,
                    'created': datetime.fromtimestamp(file_stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                    'modified': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                })

        # 按修改时间倒序排序
        reports.sort(key=lambda x: x['modified'], reverse=True)

        return jsonify({
            'success': True,
            'reports': reports
        })

    except Exception as e:
        logger.error(f"获取报告列表失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'获取报告列表失败: {str(e)}'
        }), 500

@rag_agent_bp.route('/rag_agent/api/clear-history', methods=['POST'])
def clear_history():
    """清空当前会话的对话历史"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default_session')

        if 'conversation_history' in session:
            session.pop('conversation_history')
            session.modified = True
            logger.info(f"已清空会话 {session_id} 的历史记录")

        return jsonify({
            'success': True,
            'message': '对话历史已清空',
            'history_length': 0
        })
    except Exception as e:
        logger.error(f"清空历史失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'清空历史失败: {str(e)}'
        }), 500


@rag_agent_bp.route('/rag_agent/api/get-history-length', methods=['GET'])
def get_history_length():
    """获取当前对话历史长度"""
    try:
        data = request.args
        session_id = data.get('session_id', 'default_session')

        history = get_conversation_history(session_id)
        return jsonify({
            'success': True,
            'history_length': len(history)
        })
    except Exception as e:
        logger.error(f"获取历史长度失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'获取历史长度失败: {str(e)}'
        }), 500


@rag_agent_bp.route('/rag_agent/api/get-history', methods=['GET'])
def get_history():
    """获取当前对话历史（调试用）"""
    try:
        data = request.args
        session_id = data.get('session_id', 'default_session')

        history = get_conversation_history(session_id)
        return jsonify({
            'success': True,
            'history': history,
            'count': len(history)
        })
    except Exception as e:
        logger.error(f"获取历史失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'获取历史失败: {str(e)}'
        }), 500


@rag_agent_bp.route('/rag_agent/api/load-history', methods=['GET'])
def load_history():
    """加载当前会话的历史记录到前端"""
    try:
        data = request.args
        session_id = data.get('session_id', 'default_session')

        history = get_conversation_history(session_id)

        # 将历史记录格式化为前端可用的格式
        formatted_history = []
        for msg in history:
            formatted_history.append({
                'role': msg['role'],
                'content': msg['content'],
                'timestamp': msg.get('timestamp', '')
            })

        return jsonify({
            'success': True,
            'history': formatted_history
        })
    except Exception as e:
        logger.error(f"加载历史失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'加载历史失败: {str(e)}'
        }), 500