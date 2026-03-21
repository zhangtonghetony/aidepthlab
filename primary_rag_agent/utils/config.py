import yaml
from primary_rag_agent.utils.path_tool import get_abs_path

'''
配置文件相关代码
'''

def load_rag_config(config_path:str = get_abs_path('config/rag_config.yml'),encoding='utf-8') -> dict:
    with open(config_path,'r',encoding=encoding) as f:
        rag_config = yaml.load(f,Loader=yaml.FullLoader)
    return rag_config # 返回的是dict类型


def load_chroma_config(config_path:str =get_abs_path('config/chroma_config.yml'),encoding='utf-8') -> dict:
    with open(config_path,'r',encoding=encoding) as f:
        chroma_config = yaml.load(f,Loader=yaml.FullLoader)
    return chroma_config


def load_prompt_config(config_path:str = get_abs_path('config/prompt_config.yml'),encoding='utf-8') -> dict:
    with open(config_path,'r',encoding=encoding) as f:
         prompt_config= yaml.load(f,Loader=yaml.FullLoader)
    return prompt_config

rag_config = load_rag_config()
chroma_config = load_chroma_config()
prompt_config = load_prompt_config()

if __name__ == '__main__':
    print(rag_config['chat_model'])
    print(type(rag_config))