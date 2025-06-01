# 0. 필수 임포트
from langchain_openai import ChatOpenAI      # langchain-openai 패키지
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain import hub
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults


load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

#search_tool1 = DuckDuckGoSearchRun(
#    name="Intermediate Answer",
#    api_wrapper=DuckDuckGoSearchAPIWrapper(backend="api")  # ✅ 해결
#)

search_tool2 = TavilySearchResults(name="Intermediate Answer")

tools = [search_tool2]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.SELF_ASK_WITH_SEARCH,
    verbose=True,
    handle_parsing_errors=True
)

print(agent.run("한국 최초의 우주인은 누구이고, 현재 하는 일은 무엇인가요?"))
