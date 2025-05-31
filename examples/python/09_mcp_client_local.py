import asyncio
import os
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

async def main():
    # Create MCPClient from configuration dictionary
    client = MCPClient.from_config_file(
        os.path.join("../config/mcp_server_config.json")
    )

    # Create LLM
    llm = ChatOpenAI(openai_api_base="http://localhost:1234/v1", 
                    openai_api_key="lmstudio",
                    model_name="google/gemma-3-1b"
                    )

    # Create agent with the client
    agent = MCPAgent(llm=llm, client=client, verbose=True, max_steps=30)

    # Run the query
    result = await agent.run(
        "장비에 '안녕하세요? secsgem' 라고 터미널 메시지를 보내줘.",
    )
    print(f"\nResult: {result}")

if __name__ == "__main__":
    asyncio.run(main())