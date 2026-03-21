from langchain.agents import create_agent
from .middleware import load_main_prompt, report_prompt_switch, monitor_tool
from .tools import rag_summarize, generate_report, save_report_as_html
from langchain_community.chat_models.tongyi import ChatTongyi
from primary_rag_agent.utils.config import rag_config


class ReactAgent():
    def __init__(self):
        self.agent = create_agent(
            model=ChatTongyi(model=rag_config['chat_model']),
            system_prompt=load_main_prompt(),
            tools=[rag_summarize, generate_report,save_report_as_html],
            middleware=[monitor_tool, report_prompt_switch]
        )

    def execute(self, query: str, context: dict = None):
        """执行查询，非流式输出"""
        # 初始化context，设置report为False
        input_dict = {
            'messages': [{'role': 'user', 'content': query}]
        }

        # 如果提供了上下文，则使用，否则初始化为空
        if context is None:
            context = {'report': False}

        # 执行查询
        result = self.agent.invoke(input_dict, context=context)

        # 提取最终回复
        if result and 'messages' in result and len(result['messages']) > 0:
            latest_message = result['messages'][-1]
            if latest_message.content:
                return latest_message.content.strip()

        return "抱歉，未能生成有效的回复。"