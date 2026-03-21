import logging
import os
from datetime import datetime
from primary_rag_agent.utils.path_tool import get_abs_path
'''
日志配置文件
1.日志基础配置（获取路径、创建目录、格式配置）
2.logger = logging.getLogger(name)得到logger实例对象
3.为logger添加handler
'''

# 获取日志目录绝对路径
log_path = get_abs_path('logs')

# 确保日志目录存在
os.makedirs(log_path, exist_ok=True)

# 日志格式配置
default_format =logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

def get_logger(name:str = 'rag_agent',console_level:int = logging.INFO,file_level:int = logging.DEBUG,log_file=None ) -> logging.Logger:

    logger = logging.getLogger(name)

    logger.setLevel(console_level)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(default_format)

    logger.addHandler(console_handler)

    # 文件handler
    if log_file is None:
        file_handler = logging.FileHandler(os.path.join(log_path,f'{name}_{datetime.now().strftime("%Y%m%d")}.log'),encoding='utf-8')
        file_handler.setLevel(file_level)
        file_handler.setFormatter(default_format)

        logger.addHandler(file_handler)

    return logger

logger = get_logger()

if __name__ == '__main__':
    print(logger)
    print(type(logger))
    logger.debug('debug')
    logger.info('info')
