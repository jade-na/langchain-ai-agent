from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",         # 범용 모델
    model_kwargs={"device": "cuda"},         # 또는 "cuda"로 설정하면 GPU 사용 가능
    encode_kwargs={"normalize_embeddings": True}  # 벡터 정규화 (RAG에서 일반적으로 권장)
)

## OJT a

vectorstore = FAISS.load_local(
    folder_path="../data",
    embeddings=embeddings,
    index_name="api_manual1_faiss_huggingface"
)

llm = ChatOpenAI(openai_api_base="http://localhost:1234/v1", 
                 openai_api_key="lmstudio",
                 model_name="google/gemma-3-1b")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)


print("----")

# query = "휴대폰 인증관련 API 알려줘 "
# query = "회원가입 휴대폰 인증 요청 API에 Request Body 알려줘?"
query = ("회원가입 휴대폰 인증 요청 API Spec에 대해서 알려줘 이쁘게 정렬해서 보여주면 좋고 ~ 반환은 json 구조로 해줄수잇나 ex) key value 로 ? ")
# query = "API 호출에 대한 내용이 있어? 있다면 이쁘게 보여줘"
result = qa_chain.invoke(query)
# print(result)
print(result["result"])



