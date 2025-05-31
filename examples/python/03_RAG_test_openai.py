# RAG(Retrieval-Augmented Generation) 샘플
# pdf 파일을 읽어서 vector 형태로 저장하고, 이를 활용하여 질의응답을 수행하는 샘플
# 이 샘플은 openAI의 gpt-3.5-turbo LLM을 사용 함.

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

pdf_filepath = "../data/api_manual1.pdf"
loader = PyPDFLoader(pdf_filepath)
pages = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
docs = text_splitter.split_documents(pages)
texts = text_splitter.split_text(pages[0].page_content)

recursive_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ",", " "], #우선순위
    chunk_size = 500,
    chunk_overlap = 200)

docs1 = recursive_splitter.split_documents(pages)
texts1 = recursive_splitter.split_text(pages[0].page_content)

split = loader.load_and_split(text_splitter=recursive_splitter)

print(texts1)

embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(split, embeddings)

vectorstore.save_local(folder_path="../data", index_name='api_manual1_faiss_openai')

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"), 
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

query = "휴대폰 인증 관련 API 알려줘"
result = qa_chain.invoke({"query": query})
print(result["result"])
