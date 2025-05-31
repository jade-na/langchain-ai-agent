# mcp_use 라이브러리를 활용한 대화형 agent
# MCPClient와 local LLM을 결합한 ai agent

import asyncio
import os
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient


class InteractiveMCPAgent:
    """대화형 MCP Agent 클래스"""
    
    def __init__(self, config_file: str = "../config/mcp_server_config.json"):
        self.config_file = config_file
        self.client = None
        self.agent = None
        self.conversation_history = []
        
    async def initialize(self):
        """Agent 초기화"""
        print("🤖 MCP Agent 초기화 중...")
        
        try:
            # Load environment variables
            load_dotenv()
            
            # Create MCPClient from configuration file
            self.client = MCPClient.from_config_file(
                os.path.join(self.config_file)
            )
            print("✅ MCP Client 생성 완료")

            # Create LLM
            self.llm = ChatOpenAI(model="gpt-4o")
            print("✅ LLM 연결 완료")

            # Create agent with the client
            self.agent = MCPAgent(
                llm=self.llm, 
                client=self.client, 
                verbose=True, 
                max_steps=30
            )
            print("✅ MCP Agent 생성 완료")
            
            return True
            
        except Exception as e:
            print(f"❌ 초기화 실패: {str(e)}")
            return False
    
    async def process_user_input(self, user_input: str) -> str:
        """사용자 입력 처리"""
        try:
            print(f"\n🔄 처리 중: {user_input}")
            
            # Run the query
            result = await self.agent.run(user_input)
            
            # 대화 히스토리에 추가
            self.conversation_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": user_input,
                "agent": result
            })
            
            return result
            
        except Exception as e:
            error_msg = f"처리 중 오류 발생: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def show_help(self):
        """도움말 표시"""
        help_text = """
🤖 MCP Agent 대화형 모드

📋 사용 가능한 명령어:
- help, 도움말: 이 도움말 표시
- history, 히스토리: 대화 기록 보기  
- clear, 초기화: 대화 기록 지우기
- quit, exit, 종료: 프로그램 종료

💡 사용 예시:
- "장비 상태를 확인해주세요"
- "장비에 '안녕하세요' 메시지를 보내주세요"
- "현재 시간을 알려주세요"

✨ 자연어로 원하는 작업을 요청하면 됩니다!
        """
        print(help_text)
    
    def show_history(self):
        """대화 기록 표시"""
        if not self.conversation_history:
            print("📝 대화 기록이 없습니다.")
            return
        
        print("\n📝 대화 기록:")
        print("=" * 80)
        
        for i, entry in enumerate(self.conversation_history, 1):
            print(f"\n[{i}] {entry['timestamp']}")
            print(f"👤 사용자: {entry['user']}")
            print(f"🤖 Agent: {entry['agent']}")
            print("-" * 80)
    
    def clear_history(self):
        """대화 기록 초기화"""
        self.conversation_history.clear()
        print("🗑️ 대화 기록이 초기화되었습니다.")
    
    async def run_interactive(self):
        """대화형 인터페이스 실행"""
        # 초기화
        if not await self.initialize():
            print("❌ 초기화에 실패했습니다. 프로그램을 종료합니다.")
            return
        
        # 시작 메시지
        print("\n" + "=" * 80)
        print("🚀 MCP Agent 대화형 모드 시작")
        print("=" * 80)
        print("💡 'help' 또는 '도움말'을 입력하면 사용법을 확인할 수 있습니다.")
        print("💡 'quit', 'exit', '종료'를 입력하면 프로그램이 종료됩니다.")
        print("=" * 80)
        
        # 초기 테스트 (선택적)
        print("\n🧪 초기 연결 테스트 중...")
        test_result = await self.process_user_input("사용 가능한 도구들을 알려주세요")
        print(f"🤖 테스트 결과: {test_result}")
        print("\n" + "=" * 80)
        
        # 메인 대화 루프
        while True:
            try:
                # 사용자 입력 받기
                user_input = input("\n👤 사용자: ").strip()
                
                # 빈 입력 처리
                if not user_input:
                    continue
                
                # 특별 명령어 처리
                if user_input.lower() in ['quit', 'exit', '종료']:
                    print("👋 프로그램을 종료합니다.")
                    break
                
                elif user_input.lower() in ['help', '도움말']:
                    self.show_help()
                    continue
                
                elif user_input.lower() in ['history', '히스토리', '기록']:
                    self.show_history()
                    continue
                
                elif user_input.lower() in ['clear', '초기화', '클리어']:
                    self.clear_history()
                    continue
                
                # 일반 요청 처리
                result = await self.process_user_input(user_input)
                print(f"🤖 Agent: {result}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Ctrl+C로 프로그램을 종료합니다.")
                break
                
            except Exception as e:
                print(f"❌ 예외 발생: {str(e)}")
                print("계속 사용하시겠습니까? (y/n): ", end="")
                choice = input().strip().lower()
                if choice in ['n', 'no', '아니오']:
                    break
        
        # 정리
        await self.cleanup()
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            if self.client and hasattr(self.client, 'close_all_sessions'):
                await self.client.close_all_sessions()
                print("🔧 MCP Client 세션 종료 완료")
        except Exception as e:
            print(f"⚠️ 정리 중 오류: {str(e)}")


# 추가: 배치 처리 모드
class BatchMCPAgent(InteractiveMCPAgent):
    """배치 처리용 MCP Agent"""
    
    async def run_batch(self, commands: List[str]):
        """여러 명령어를 배치로 처리"""
        if not await self.initialize():
            print("❌ 초기화 실패")
            return
        
        print(f"📦 배치 모드: {len(commands)}개 명령어 처리 시작")
        
        results = []
        for i, command in enumerate(commands, 1):
            print(f"\n[{i}/{len(commands)}] 처리 중: {command}")
            result = await self.process_user_input(command)
            results.append({
                "command": command,
                "result": result,
                "index": i
            })
            print(f"✅ 완료: {result[:100]}...")  # 처음 100자만 표시
        
        await self.cleanup()
        return results


async def main():
    """메인 함수 - 대화형 모드"""
    agent = InteractiveMCPAgent(config_file="../config/mcp_server_config.json")
    await agent.run_interactive()


async def batch_example():
    """배치 처리 예제"""
    batch_agent = BatchMCPAgent(config_file="../config/mcp_server_config.json")
    
    # 배치로 처리할 명령어들
    commands = [
        "장비 상태를 확인해주세요",
        "장비에 '배치 테스트 1' 메시지를 보내주세요",
        "현재 시간을 알려주세요",
        "장비에 '배치 테스트 완료' 메시지를 보내주세요"
    ]
    
    results = await batch_agent.run_batch(commands)
    
    # 결과 요약 출력
    print("\n" + "=" * 80)
    print("📊 배치 처리 결과 요약")
    print("=" * 80)
    for result in results:
        print(f"[{result['index']}] {result['command']}")
        print(f"    → {result['result'][:100]}...")
        print()


if __name__ == "__main__":
    # 대화형 모드 실행
    asyncio.run(main())
    
    # 배치 모드 실행 (주석 해제하여 사용)
    # asyncio.run(batch_example())