from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings()

## OJT a

vectorstore = FAISS.load_local(
    folder_path="../data",
    embeddings=embeddings,
    index_name="api_manual1_faiss_openai"
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
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



