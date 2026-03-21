from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from primary_rag_agent.utils.config import chroma_config
from primary_rag_agent.utils.config import rag_config
import os
import hashlib
from primary_rag_agent.utils.log import logger

persist_directory = chroma_config['persist_dir']
VECTOR_MD5_FILE = chroma_config['VECTOR_MD5_FILE']


class VectorStoreService():
    def __init__(self):
        # 直接创建嵌入模型
        self.embedding_model = DashScopeEmbeddings(
            model=rag_config['embedding_model']
        )

        self.chroma = Chroma(
            collection_name=chroma_config['collection_name'],
            embedding_function=self.embedding_model,
            persist_directory=persist_directory,
        )

        self.spliter = RecursiveCharacterTextSplitter(
            separators=chroma_config['separators'],
            chunk_size=chroma_config['chunk_size'],
            chunk_overlap=chroma_config['overlap'],
            length_function=len
        )

    def get_retriever(self):
        """获取检索器"""
        return self.chroma.as_retriever(search_kwargs={'k': chroma_config['k']})

    def _get_file_md5(self, file_path: str) -> str:
        """计算文件的MD5值"""
        chunk_size = 4096
        md5 = hashlib.md5()

        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)

        return md5.hexdigest()

    def _check_md5_exist(self, md5_for_check: str) -> bool:
        """检查MD5是否已存在于向量化记录"""
        # 使用单独的向量化MD5记录文件

        if not os.path.exists(VECTOR_MD5_FILE):
            return False

        with open(VECTOR_MD5_FILE, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if md5_for_check == line.strip():
                    return True
        return False

    def _save_md5(self, md5_value: str):
        """保存MD5值到记录文件"""
        with open(VECTOR_MD5_FILE, 'a', encoding='utf-8') as f:
            f.write(md5_value + '\n')

    def _load_document(self, file_path: str):
        """加载文档内容"""
        if file_path.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
            return loader.load()
        elif file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            return loader.load()
        return []

    def load_uploaded_files(self, upload_dir: str):
        """
        加载上传的文件到向量数据库

        Args:
            upload_dir: 上传目录路径

        Returns:
            tuple: (处理的文件数量, 跳过的文件数量, 总文件数)
        """

        if not os.path.exists(upload_dir):
            logger.warning(f"上传目录不存在: {upload_dir}")
            return 0, 0, 0

        # 记录统计信息
        processed_count = 0
        skipped_count = 0
        total_count = 0

        # 遍历上传目录
        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)

            # 跳过子目录
            if not os.path.isfile(file_path):
                continue

            total_count += 1

            # 计算MD5
            file_md5 = self._get_file_md5(file_path)

            # 检查是否已处理（已向量化）
            if self._check_md5_exist(file_md5):
                logger.info(f'文件已向量化过: {filename}')
                skipped_count += 1
                continue

            # 加载文档
            documents = self._load_document(file_path)

            if not documents:
                logger.warning(f'无法加载文档: {filename}')
                continue

            # 保存md5值到向量化记录
            self._save_md5(file_md5)

            # 分割文档
            split_documents = self.spliter.split_documents(documents)

            # 存入向量数据库
            self.chroma.add_documents(split_documents)

            logger.info(f'文件已向量化: {filename}')
            processed_count += 1

        # 记录处理结果
        logger.info(f"向量化处理完成: 总文件数={total_count}, 新处理={processed_count}, 跳过={skipped_count}")

        return processed_count, skipped_count, total_count

    def process_single_file(self, file_path: str, filename: str) -> bool:
        """
        处理单个文件向量化
        Returns: 是否成功
        """
        try:
            # 计算MD5
            file_md5 = self._get_file_md5(file_path)

            # 检查是否已向量化
            if self._check_md5_exist(file_md5):
                logger.info(f'文件已向量化过: {filename}')
                return False

            # 加载文档
            documents = self._load_document(file_path)

            if not documents:
                logger.warning(f'无法加载文档: {filename}')
                return False

            # 保存向量化MD5记录
            self._save_md5(file_md5)

            # 分割文档
            split_documents = self.spliter.split_documents(documents)

            # 存入向量数据库
            self.chroma.add_documents(split_documents)

            logger.info(f'文件已向量化: {filename}')

            return True

        except Exception as e:
            logger.error(f'文件向量化失败 {filename}: {str(e)}')

            return False


# 实例化类对象
vector_store = VectorStoreService()