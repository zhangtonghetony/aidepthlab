from langchain_core.tools import tool
from primary_rag_agent.rag.rag import RagService
from datetime import datetime
import json
import markdown
import os
import uuid
import re
from primary_rag_agent.utils.log import logger
from flask import session

rag_service = RagService()

# 报告存储目录 - 绝对路径
REPORTS_DIR = r"C:\Users\zhang\Desktop\AI_display\static\report"

# 确保目录存在
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR, exist_ok=True)


@tool(description='从向量存储中检索参考资料')
def rag_summarize(query: str):
    """从向量库检索金融信息"""
    return rag_service.rag_summarize(query)


@tool(description='无入参，无返回值，调用后触发中间件为报告生产场景动态注入上下文信息，为后续提示词切换提供上下文信息')
def fill_context_for_report():
    """切换到报告生成模式的前置工具"""
    return 'fill_context_for_report已调用'


@tool(description='收集报告所需信息，为LLM生成报告提供数据')
def generate_report(user_requirements: str, report_type: str = None):
    """
    收集报告所需信息，返回给AI用于生成报告

    参数:
    - user_requirements: 用户的具体需求描述
    - report_type: 可选，报告类型

    返回:
    - JSON格式的报告数据，包含需求、标题、RAG数据等信息
    """
    try:
        # 1. 生成报告标题
        if report_type:
            report_title = f"{user_requirements} - {report_type}"
        else:
            report_title = f"{user_requirements} - 金融分析报告"

        # 2. 使用RAG获取相关信息
        logger.info(f"[generate_report] 开始检索: {user_requirements}, 报告类型: {report_type}")
        rag_query = f"{user_requirements} {report_type if report_type else ''}"
        rag_result = rag_service.rag_summarize(rag_query)
        logger.info(f"[generate_report] RAG检索完成，数据长度: {len(str(rag_result))}")

        # 3. 返回JSON数据给AI
        report_data = {
            "report_title": report_title,
            "user_requirements": user_requirements,
            "report_type": report_type or "金融分析报告",
            "rag_data": rag_result,
            "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "instructions": "请基于以上信息，结合你的金融专业知识，生成不少于800字的完整Markdown格式金融报告。报告应包含：执行摘要、宏观经济分析、行业分析、核心数据、风险评估、投资建议、结论等章节。"
        }

        result_json = json.dumps(report_data, ensure_ascii=False, indent=2)
        logger.info(f"[generate_report] 返回数据长度: {len(result_json)}")

        return result_json

    except Exception as e:
        error_msg = f"收集报告数据时出错: {str(e)}"
        logger.error(f"[generate_report] {error_msg}")
        return json.dumps({"error": error_msg}, ensure_ascii=False)


@tool(description='将Markdown报告保存为HTML文件到服务器')
def save_report_as_html(markdown_content: str, report_title: str = None,user_id : str=None ):
    """
    将Markdown报告保存为HTML文件到服务器

    参数:
    - markdown_content: Markdown格式的报告内容
    - report_title: 可选，报告标题，用于文件名
    - user_id: 给保存的HTML文件名中加上user_id，用于辨别报告归属

    返回:
    - JSON格式，包含文件保存信息
    """
    try:
        result = save_report_as_html_internal(markdown_content, report_title)
        return json.dumps({"success": True, "message": result}, ensure_ascii=False)
    except Exception as e:
        error_msg = f"保存报告工具调用失败: {str(e)}"
        logger.error(f"[save_report_as_html] {error_msg}")
        return json.dumps({"success": False, "error": error_msg}, ensure_ascii=False)


def save_report_as_html_internal(markdown_content: str, report_title: str = None) -> str:
    """
    内部函数：实际保存HTML文件的逻辑
    直接从 Flask Session 中获取 user_id 并作为文件名前缀
    """
    try:
        # 0. 从 session 获取 user_id，如果不存在则标记为 anonymous
        # 这样可以确保文件名始终具有可识别的归属前缀
        current_user = session.get('user_id', 'anonymous')

        # 1. 生成文件名
        if not report_title:
            report_title = f"金融报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 清理文件名中的特殊字符
        safe_title = re.sub(r'[^\w\u4e00-\u9fa5\s-]', '', report_title)
        safe_title = re.sub(r'[-\s]+', '_', safe_title)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_id = str(uuid.uuid4())[:8]

        # 将 user_id 放在文件名的最前面，方便后续使用 startswith 过滤查找
        filename = f"{current_user}_{safe_title}_{timestamp}_{report_id}"

        # 2. 保存Markdown文件
        md_filename = f"{filename}.md"
        md_path = os.path.join(REPORTS_DIR, md_filename)

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        logger.info(f"[save_report_as_html_internal] Markdown文件已保存: {md_path} (用户: {current_user})")

        # 3. 转换为HTML
        html_content = convert_to_html(markdown_content, report_title)

        # 4. 保存HTML文件
        html_filename = f"{filename}.html"
        html_path = os.path.join(REPORTS_DIR, html_filename)

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"[save_report_as_html_internal] HTML文件已保存: {html_path}")

        # 5. 返回成功信息
        result = f"""
✅ **报告保存成功！**

**文件信息:**
- 归属用户: `{current_user}`
- HTML文件: `{html_filename}`
- Markdown文件: `{md_filename}`
- 保存路径: `{html_path}`
- 保存时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**下载说明:**
1. 文件已保存在服务器，已与您的账户绑定。
2. 可以在"下载报告"页面查看您的专属历史报告。
        """

        return result

    except Exception as e:
        error_msg = f"❌ 保存HTML文件时出错: {str(e)}"
        logger.error(f"[save_report_as_html_internal] {error_msg}")
        return error_msg

def convert_to_html(markdown_text: str, title: str = "金融报告") -> str:
    """将Markdown转换为美观的HTML"""
    # 扩展配置
    extensions = ['extra', 'tables', 'fenced_code']

    # 转换Markdown
    md = markdown.Markdown(extensions=extensions, output_format='html5')
    html_body = md.convert(markdown_text)

    # 简单美观的HTML模板
    html_template = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{ 
            color: #2c3e50; 
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{ color: #34495e; border-bottom: 1px solid #eee; padding-bottom: 8px; }}
        h3 {{ color: #2c3e50; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        pre {{
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        code {{ background: #f8f9fa; padding: 2px 6px; border-radius: 3px; }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #7f8c8d;
            font-size: 0.9em;
            text-align: center;
        }}
        .success-box {{
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div style="color: #7f8c8d; margin-bottom: 30px;">
            生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
        </div>
        <div class="success-box">
            ✅ 本报告已自动保存为HTML文件，可下载或打印
        </div>
        {html_body}
        <div class="footer">
            <p>本报告由AI金融分析助手生成 | © {datetime.now().year}</p>
        </div>
    </div>
</body>
</html>'''

    return html_template