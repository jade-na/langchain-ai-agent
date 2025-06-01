from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory

# 1. 사용할 툴 정의
def greet(name: str) -> str:
    return f"안녕하세요, {name}님! 반갑습니다."

tools = [
    Tool(
        name="greet_user",
        func=greet,
        description="이름을 받아 인사말을 생성하는 도구입니다."
    )
]

# 2. 로컬 LLM 설정 (Gemma 모델 예시)
llm = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1", 
    openai_api_key="lmstudio",
    model_name="google/gemma-3-1b"
)

# 3. 메모리 설정
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 4. 에이전트 초기화
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# 5. 지속형 대화 루프
print("대화를 시작합니다. 종료하려면 '종료'라고 입력하세요.\n")

while True:
    user_input = input("👤 사용자: ")
    
    if user_input.strip().lower() in ["종료", "quit", "exit"]:
        print("🧠 대화를 종료합니다. 안녕히 가세요!")
        break
    
    response = agent.run(user_input)
    print(f"🤖 에이전트: {response}\n")
