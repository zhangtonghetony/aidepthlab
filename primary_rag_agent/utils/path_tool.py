import os

'''
为整个工程提供统一的绝对路径
'''

def get_project_root():
    # 获取当前文件绝对路径
    current_file=os.path.abspath(__file__)
    # 获取文件所在文件夹绝对路径
    dir_path=os.path.dirname(current_file)
    # 获取工程根目录（到primary_rag_agent一层）
    root_path=os.path.dirname(dir_path)

    return root_path

def get_abs_path(path:str):
    '''
    :param path: 相对路径
    :return: 拼接后的绝对路径
    '''
    return os.path.join(get_project_root(), path)




if __name__ == '__main__':
    abs_path=get_abs_path('utils/log.py')
    print(abs_path)