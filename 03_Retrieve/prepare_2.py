from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import SpacyTextSplitter

loader = PyMuPDFLoader("./sample.pdf")
documents = loader.load()

text_splitter = SpacyTextSplitter(
    chunk_size=300,
    pipeline="ko_core_news_sm"
)
splitted_documents = text_splitter.split_documents(documents)

print(f"분할 전 문서 갯수: {len(documents)}")
print(f"분하 후 문서 개수: {len(splitted_documents)}")