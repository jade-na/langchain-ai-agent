#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI  # ✅ 권장 방식
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI(openai_api_base="http://localhost:1234/v1", 
                 openai_api_key="lmstudio",
                 model_name="google/gemma-3-1b"
                 )

prompt = PromptTemplate(
    input_variables=["product"],
    template="다음 제품에 대한 마케팅 문구를 작성해주세요: {product}"
)

chain = LLMChain(llm=llm, prompt=prompt)

result = chain.run("스마트워치")
print(result)
