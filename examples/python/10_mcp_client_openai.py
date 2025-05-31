# mcp_use 라이브러리를 활용한 agent
# MCPClient와 local LLM을 결합한 ai agent

import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

async def main():
    # Load environment variables
    load_dotenv()
    
    # Create MCPClient from configuration dictionary
    client = MCPClient.from_config_file(
        os.path.join("../config/mcp_server_config.json")
    )

    # Create LLM
    llm = ChatOpenAI(model="gpt-4o")

    # Create agent with the client
    agent = MCPAgent(llm=llm, client=client, verbose=True, max_steps=30)

    # Run the query
    result = await agent.run(
        "장비에 '안녕하세요? secsgem' 라고 터미널 메시지를 보내줘.",
    )
    print(f"\nResult: {result}")

if __name__ == "__main__":
    asyncio.run(main())