from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import SpacyTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

loader = PyMuPDFLoader("./sample.pdf")
documents = loader.load()

text_splitter = SpacyTextSplitter(
    chunk_size=300,
    pipeline="ko_core_news_sm"
)
splitted_documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

database = Chroma(
    persist_directory="./.data",
    embedding_function=embeddings
)

database.add_documents(
    splitted_documents,
)

print("데이터베이스 생성이 완료되었습니다.")