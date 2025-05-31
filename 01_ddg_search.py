from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.tools import DuckDuckGoSearchRun


llm = ChatOpenAI(openai_api_base="http://localhost:1234/v1", 
                 openai_api_key="lmstudio",
                 model_name = "google/gemma-3-1b"
                 )

# DuckDuckGo 검색 툴 추가
search = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="웹에서 검색할 수 있습니다."
    )
]

# 에이전트 초기화
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True  # ✅ 이 옵션 추가
)

# 명령어 실행
response = agent.run("오산시 시청의 한글 주소를 알려주세요.")
print(response)


