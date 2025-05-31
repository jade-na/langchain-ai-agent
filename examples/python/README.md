예제 코드를 실행하기 위해서 아래와 같이 진행 하세요.

## Installation uv

Install uv with our standalone installers:

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```


## 가상환경 설치 및 패키지 설치

uv를 설치하였다면 uv를 이용하여 가상환경을 초기화하고 가상환경에 예제에서 사용되는 패키지를 설치 합니다.

```bash
#가상 환경 초기화
uv venv .venv

#가상환경 활성화
.venv\Scripts\activate

#필요 패키지 설치
uv pip install -r requirements.txt

```

## 예제 실행

예제 실행을 위해서 LLM Server를 실행 시킨 후 파이썬 코드를 실행 시킵니다.

```bash

python 01_ddg_search.py

```