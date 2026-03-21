# /blueprint/login.py
"""
登录注册模块
处理用户认证相关功能
"""

from flask import Blueprint, request, jsonify, session, redirect, url_for, render_template
import pymysql
import re

# 创建蓝图
login_bp = Blueprint('login', __name__)

# 数据库连接配置
DB_CONFIG = {
    'host': 'localhost',  # MySQL主机地址
    'user': 'root',  # MySQL用户名
    'password': '110606',  # MySQL密码
    'database': 'aidepthlab',  # 数据库名
    'charset': 'utf8mb4'  # 字符集，支持中文
}


def get_db():
    """
    获取数据库连接对象
    返回: pymysql连接对象
    """
    return pymysql.connect(**DB_CONFIG)


@login_bp.route('/login', methods=['GET'])
def login_page():
    """
    显示登录页面
    如果用户已登录，重定向到RAG Agent页面
    """
    if 'user_id' in session:  # 检查session中是否有user_id
        return redirect(url_for('rag_agent.rag_agent_page'))  # 已登录，跳转到RAG Agent页面(文件名.函数名)
    return render_template('login.html')  # 未登录，显示登录页面


@login_bp.route('/register', methods=['GET'])
def register_page():
    """
    显示注册页面
    如果用户已登录，重定向到RAG Agent页面
    """
    if 'user_id' in session:  # 检查session中是否有user_id
        return redirect(url_for('rag_agent.rag_agent_page'))  # 已登录，跳转到RAG Agent页面
    return render_template('login.html')  # 显示登录页面（包含注册表单）


@login_bp.route('/api/login', methods=['POST'])
def api_login():
    """
    处理登录请求的API接口
    请求方法: POST
    请求体: JSON格式，包含identifier和password
    返回: JSON格式的登录结果
    """
    # 获取前端发送的JSON数据
    data = request.get_json()
    identifier = data.get('identifier', '').strip()  # 用户名或邮箱
    password = data.get('password', '').strip()  # 密码

    # 验证输入是否为空
    if not identifier or not password:
        return jsonify({"success": False, "message": "请输入用户名/邮箱和密码"}), 400

    # 连接数据库
    db = get_db()
    # 创建游标，返回字典类型的结果
    cursor = db.cursor(pymysql.cursors.DictCursor)

    # 查询用户：通过用户名或邮箱查找
    cursor.execute("SELECT * FROM user_info WHERE username = %s OR email = %s", (identifier, identifier))
    user = cursor.fetchone()  # 获取第一条记录

    # 关闭数据库连接
    cursor.close()
    db.close()

    # 用户不存在
    if not user:
        return jsonify({"success": False, "message": "用户不存在"}), 401

    # 验证密码（明文比较，生产环境需要加密！）
    if user['password'] != password:
        return jsonify({"success": False, "message": "密码错误"}), 401

    # 登录成功，设置session
    session['user_id'] = user['user_id']  # 用户ID
    session['username'] = user['username']  # 用户名
    session['email'] = user['email']  # 邮箱

    # 返回成功响应
    return jsonify({
        "success": True,
        "message": "登录成功",
        "redirect_url": url_for('rag_agent.rag_agent_page'),  # 告诉前端要跳转到RAG Agent页面
        "user": {
            "id": user['user_id'],  # 用户ID
            "username": user['username'],  # 用户名
            "email": user['email']  # 邮箱
        }
    })


@login_bp.route('/api/register', methods=['POST'])
def api_register():
    """
    处理注册请求的API接口
    请求方法: POST
    请求体: JSON格式，包含username、email、password
    返回: JSON格式的注册结果
    """
    # 获取前端发送的JSON数据
    data = request.get_json()
    username = data.get('username', '').strip()  # 用户名
    email = data.get('email', '').strip()  # 邮箱
    password = data.get('password', '').strip()  # 密码

    # 验证必填字段
    if not username or not email or not password:
        return jsonify({"success": False, "message": "请填写所有必填字段"}), 400

    # 验证用户名长度
    if len(username) < 3 or len(username) > 20:
        return jsonify({"success": False, "message": "用户名长度需在3-20字符之间"}), 400

    # 验证密码长度
    if len(password) < 6:
        return jsonify({"success": False, "message": "密码至少需要6个字符"}), 400

    # 验证邮箱格式（简单的正则验证）
    email_pattern = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
    if not re.match(email_pattern, email):
        return jsonify({"success": False, "message": "邮箱格式无效"}), 400

    # 连接数据库
    db = get_db()
    cursor = db.cursor(pymysql.cursors.DictCursor)

    # 检查用户名是否已存在
    cursor.execute("SELECT * FROM user_info WHERE username = %s", (username,))
    if cursor.fetchone():
        cursor.close()
        db.close()
        return jsonify({"success": False, "message": "用户名已存在"}), 400

    # 检查邮箱是否已注册
    cursor.execute("SELECT * FROM user_info WHERE email = %s", (email,))
    if cursor.fetchone():
        cursor.close()
        db.close()
        return jsonify({"success": False, "message": "邮箱已被注册"}), 400

    # 插入新用户到数据库
    cursor.execute(
        "INSERT INTO user_info (username, email, password) VALUES (%s, %s, %s)",
        (username, email, password)  # 注意：这里是明文密码！
    )
    user_id = cursor.lastrowid  # 获取自增ID

    # 提交事务
    db.commit()

    # 关闭数据库连接
    cursor.close()
    db.close()

    # 注册成功，设置session
    session['user_id'] = user_id  # 用户ID
    session['username'] = username  # 用户名
    session['email'] = email  # 邮箱

    # 返回成功响应
    return jsonify({
        "success": True,
        "message": "注册成功",
        "redirect_url": url_for('rag_agent.rag_agent_page'),  # 告诉前端要跳转到RAG Agent页面
        "user": {
            "id": user_id,  # 新创建的用户ID
            "username": username,  # 用户名
            "email": email  # 邮箱
        }
    })

