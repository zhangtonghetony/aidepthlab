# /blueprint/main.py
from flask import Blueprint, render_template, jsonify, request


# 创建蓝图
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """首页路由 - 只渲染首页"""
    return render_template('main.html')