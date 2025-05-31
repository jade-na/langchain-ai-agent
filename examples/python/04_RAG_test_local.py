# RAG(Retrieval-Augmented Generation) 샘플
# pdf 파일을 읽어서 vector 형태로 저장하고, 이를 활용하여 질의응답을 수행하는 샘플
# 이 샘플은 LM Studio의 gemma3를 사용하고 HuggingFaceEmbedding을 사용.

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

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

# HuggingFace의 Embedding 모델을 직접 다운받아서 사용.
# gpu를 사용하면 속도를 향상 시킬 수 있음.
# gpu 사용을 위해서는 torch가 CUDA 활성화 상태이어야 함.
# cuda가 활성화되어 있지 않으면 torch 삭제 후 재 설치 필요.
# uv pip uninstall torch
# uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pc에 cuda가 설치되어 있는지 확인하는 방법. 파워쉘에서 다음 명령어 실행. nvidia-sml

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",         # 범용 모델
    model_kwargs={"device": "cuda"},         # 또는 "cuda"로 설정하면 GPU 사용 가능
    encode_kwargs={"normalize_embeddings": True}  # 벡터 정규화 (RAG에서 일반적으로 권장)
)

# 문서 분할 완료 후 embeddings 사용
vectorstore = FAISS.from_documents(split, embeddings)

# 저장해서 재사용 가능
vectorstore.save_local(folder_path="../data", index_name='api_manual1_faiss_huggingface')

llm = ChatOpenAI(openai_api_base="http://localhost:1234/v1", 
                 openai_api_key="lmstudio",
                 model_name="google/gemma-3-1b")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff",
    retriever=vectorstore.as_retriever())

query = "휴대폰 인증 관련 API 알려줘"
result = qa_chain.invoke({"query": query})
print(result["result"])
