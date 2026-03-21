'''
总结服务：用户端将问题和参考资料打包提交给模型，模型进行总结回复
'''
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi
from primary_rag_agent.utils.config import rag_config
from langchain_core.prompts import PromptTemplate
from primary_rag_agent.rag.vector_store import vector_store
from primary_rag_agent.utils.config import prompt_config
from datetime import datetime

def load_summarize_prompt():
    summarize_path = prompt_config["rag_summarize_path"]

    with open(summarize_path, "r", encoding='utf-8') as f:
        summarize_prompt = f.read()
    return summarize_prompt


class RagService:
    def __init__(self):
        self.store = vector_store
        self.retriever = self.store.get_retriever()
        self.model = ChatTongyi(model=rag_config['chat_model'])
        self.prompt_text = load_summarize_prompt()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.chain = self.init_chain()

    def init_chain(self):
        # 修改：更新prompt模板变量，与使用方式匹配
        # 原来的模板变量是 {input} 和 {context}，保持不变
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def retriever_doc(self, query: str):
        """检索相关文档"""
        # 修改：增加错误处理和日志
        try:
            doc = self.retriever.invoke(query)
            return doc
        except Exception as e:
            print(f"文档检索失败: {str(e)}")
            return []

    def rag_summarize(self, query: str):
        """执行RAG总结
        Args:
            query: 用户查询，可能包含历史上下文
        """
        try:
            # 1. 检索相关文档
            context = self.retriever_doc(query)

            # 2. 格式化上下文
            context_str = ''
            count = 0
            for i in context:
                count += 1
                # 修改：修正字符串格式化错误
                context_str += f'[参考资料{count}] : {i.page_content}\n\n'

            # 如果没有检索到文档
            if not context_str:
                context_str = "未找到相关参考资料"

            # 3. 调用chain进行回答
            # 修改：根据你的prompt模板，需要传递正确的变量名
            # 检查你的prompt模板中使用的变量名
            result = self.chain.invoke({
                'input': query,  # 用户的问题（可能包含历史上下文）
                'context': context_str  # 检索到的参考资料
            })

            return result

        except Exception as e:
            error_msg = f"RAG总结失败: {str(e)}"
            print(error_msg)
            return f"抱歉，处理您的请求时出现错误: {str(e)}"


# 新增：简单的对话历史管理类（可选，可以在RagService中使用）
class ConversationManager:
    """管理对话历史"""

    def __init__(self, max_history_rounds=5):
        self.max_history_rounds = max_history_rounds
        self.history = []

    def add_message(self, role, content):
        """添加消息到历史"""
        self.history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })

        # 限制历史长度
        if len(self.history) > self.max_history_rounds * 2:  # 每轮包含一问一答
            self.history = self.history[-(self.max_history_rounds * 2):]

    def get_recent_context(self, current_question, include_rounds=None):
        """获取最近的对话上下文"""
        if include_rounds is None:
            include_rounds = self.max_history_rounds

        # 计算要包含的消息数量
        max_messages = include_rounds * 2

        # 获取最近的消息
        recent_history = self.history[-max_messages:] if len(self.history) > max_messages else self.history

        # 构建上下文字符串
        context_lines = []
        for msg in recent_history:
            if msg['role'] == 'user':
                context_lines.append(f"用户: {msg['content']}")
            elif msg['role'] == 'assistant':
                context_lines.append(f"助手: {msg['content']}")

        if context_lines:
            context_text = "\n".join(context_lines)
            return f"""之前的对话历史：
{context_text}

当前问题：{current_question}

请根据以上对话历史和当前问题回答："""
        else:
            return current_question

    def clear_history(self):
        """清空对话历史"""
        self.history = []

    def get_history_count(self):
        """获取当前历史记录数量"""
        return len(self.history)


# 创建全局RAG服务实例
rag_service_instance = RagService()