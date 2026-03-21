from langchain.agents.middleware import wrap_tool_call, dynamic_prompt, ModelRequest
from primary_rag_agent.utils.log import logger
from primary_rag_agent.utils.config import prompt_config

def load_report_prompt():
    report_path = prompt_config["report_path"]

    with open(report_path, "r", encoding='utf-8') as f:
        summarize_prompt = f.read()
    return summarize_prompt



def load_main_prompt():
    main_prompt_path = prompt_config["main_prompt_path"]

    with open(main_prompt_path, "r", encoding='utf-8') as f:
        summarize_prompt = f.read()
    return summarize_prompt


@wrap_tool_call
def monitor_tool(request,handler):
    '''

    :param request: 请求的数据封装
    :param handler: 执行的函数本身
    :return: handler(request)
    '''
    logger.info(f"[tool]调用工具{request.tool_call['name']}")
    logger.info(f"[tool]传入参数{request.tool_call['args']}")

    # 通过工具调用监控中间件判断是否调用了fill_context_for_report（需在主提示词中写明），从而切换提示词
    if request.tool_call['name']=='fill_context_for_report':
        request.runtime.context['report'] = True


    return handler(request)


@dynamic_prompt # 每次生成提示词前调用此函数
def report_prompt_switch(request : ModelRequest):

    is_report = request.runtime.context.get('report', False)

    if is_report:
        return load_report_prompt()

    return load_main_prompt()
