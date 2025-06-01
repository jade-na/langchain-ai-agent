
# LangChain AgentType 입문 가이드

> **대상**: LangChain 에이전트를 처음 접하는 개발자  
> **목표**: `AgentType` 열거형이 의미하는 바를 이해하고, 적절한 타입을 선택하여 간단한 예제를 실행할 수 있다.

---

## 1. 에이전트(Agent)란 무엇인가?
LangChain에서 **에이전트**는 LLM(대형 언어 모델)을 “추론 엔진”으로 활용하여  
**어떤 툴을 언제, 어떤 순서로 사용할지 스스로 결정**하는 실행 단위입니다.  
* 체인(Chain) – 실행 순서가 고정  
* 에이전트(Agent) – 실행 순서가 **동적** (LLM이 결정)

---

## 2. `AgentType`이란?
`initialize_agent()` 함수(0.1.x 이전 API)에서 **에이전트의 전략(알고리즘)을 지정**하기 위해 쓰는 열거형입니다.
LangChain 0.1.0 이후에는 `create_react_agent()`, `create_structured_chat_agent()` 같은 _builder_ 함수가 권장되지만,  
기존 코드나 학습 자료에는 아직 `AgentType`이 많이 남아 있습니다.

```python
from langchain.agents import initialize_agent, AgentType
# 예시
agent = initialize_agent(
    tools=tools,         # 사용할 툴 리스트
    llm=llm,             # LLM 인스턴스
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
```

---

## 3. 주요 AgentType 한눈에 보기

| AgentType (열거 값) | 요약 | 내부 전략 | 주 사용처 |
|--------------------|------|-----------|-----------|
| `ZERO_SHOT_REACT_DESCRIPTION` | 설명만 주면 **ReAct** 패턴으로 추론·행동 | 입력→(Thought/Action) 반복→Answer | 사전지식 없는 간단한 작업 |
| `ZERO_SHOT_REACT_CHAT` | 위와 동일하지만 **Chat 모델** 최적화 | 동일 | 챗 기반 LLM(OpenAI GPT‑4o 등) |
| `CONVERSATIONAL_REACT_DESCRIPTION` (또는 `CHAT_CONVERSATIONAL_REACT_DESCRIPTION`) | 대화 기록(메모리)을 고려하는 ReAct | 대화 context + ReAct | 챗봇, 고객지원 |
| `SELF_ASK` (`SELF_ASK_WITH_SEARCH`) | “질문을 나눠서 스스로 다시 물어보기” 패턴 | Self‑Ask + 검색 툴 | 복잡한 지식 탐색 |
| `OPENAI_FUNCTIONS` | **Function‑calling** 전용 라우터 | 함수 매핑 테이블 | OpenAI 함수 호출 |
| `STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION` | 구조화된 역할/도구 지시어 + ReAct | 시스템/도구 프롬프트 → ReAct | 명확한 역할 구분 필요 |
| (참고) `REACT_DOCSTORE` 등 | 문서 저장소 전용, 현재는 **Deprecated** | – | – |

---

## 4. AgentType 별 작동 흐름 살펴보기

### 4‑1. ZERO_SHOT_REACT_DESCRIPTION
1. **Prompt**: “도구 설명 + 최종 답변 지시”
2. **LLM**이 *Thought → Action → Observation* 루프 실행
3. 올바른 툴·순서 자동 결정  
```python
agent = initialize_agent(tools, llm,
                         AgentType.ZERO_SHOT_REACT_DESCRIPTION)
print(agent.run("서울 오늘 날씨 알려줘."))
```

### 4‑2. CONVERSATIONAL_REACT_DESCRIPTION
* 위 흐름에 **Memory**(예: `ConversationBufferMemory`)를 추가  
```python
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent = initialize_agent(tools, llm,
                         AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                         memory=memory)
```

### 4‑3. SELF_ASK
* ⚙️ 내부적으로 “질문 분해 → 검색 → 조합”을 반복  
* 자체 검색 툴(예: `ArxivSearchTool`, `SerpAPI`) 필요

### 4‑4. OPENAI_FUNCTIONS
* 메시지를 함수 스펙과 함께 LLM에 전달 ↔ LLM이 JSON으로 함수 호출 결정  
```python
agent = initialize_agent(tools, llm, AgentType.OPENAI_FUNCTIONS)
```

### 4‑5. STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
* “You are a helpful assistant”와 별도로 **도구별 설명 블록** 제공  
* JSON 형식으로 Tool 사용 명령 → 안전성과 제어력 ↑

---

## 5. 선택 가이드

| 필요 조건 | 추천 AgentType |
|-----------|---------------|
| 코드 한 줄로 빠르게 실험 | ZERO_SHOT_REACT_DESCRIPTION |
| **대화** 이력을 유지해야 함 | CONVERSATIONAL_REACT_DESCRIPTION |
| OpenAI **함수 호출 API** 활용 | OPENAI_FUNCTIONS |
| 복잡한 지식 탐구(검색) | SELF_ASK |
| Role / Tool을 **구조적**으로 제어 | STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION |

---

## 6. AgentType 이후의 변화
* **LangChain 0.1.0+** → `initialize_agent()` & `AgentType` **Deprecated**  
  * 예) `create_structured_chat_agent(llm, tools)`  
* 하지만 **기존 튜토리얼·샘플**은 AgentType을 사용하므로 의미 파악 필수!

---

## 7. 실습 과제
1. `ZERO_SHOT_REACT_DESCRIPTION` 예제를 직접 실행해 보기  
2. 동일한 툴셋으로 `OPENAI_FUNCTIONS` 타입으로 변경, 응답 차이 비교  
3. Memory 객체를 넣어서 `CONVERSATIONAL_REACT_DESCRIPTION` 타입과 비교

---

### 참고 문헌
* LangChain 공식 문서 — Agents 섹션  
* [LangChain AgentType API 레퍼런스]  
* 커뮤니티 튜토리얼 (Medium, GitHub Discussions 등)

---

**Happy LangChaining!** 🚀
