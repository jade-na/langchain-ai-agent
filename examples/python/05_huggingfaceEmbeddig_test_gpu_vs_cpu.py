#=================================================================
#
# HuggingFaceEmbedding 모델 사용 시 gpu vs cpu 성능 비교
#
#=================================================================
import time
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings

def test_embedding_performance(device="cuda"):
    print(f"\n[INFO] '{device}' 모드로 임베딩 모델 로딩 중...")

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",  # 또는 다른 HuggingFace 모델
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    test_texts = [
        "휴대폰 인증을 처리하는 API는 어떤 것들이 있나요?",
        "사용자 로그인 기능은 어떻게 구현하나요?",
        "REST API 보안에 대해 알려주세요.",
        "LangChain에서 Retrieval QA는 어떻게 구성하나요?",
        "FastAPI로 인증 서버를 만들 수 있나요?",
    ] * 100  # 총 500개 문장

    print(f"[INFO] 총 {len(test_texts)} 문장에 대해 임베딩 시작...")

    start = time.time()
    vectors = embeddings.embed_documents(test_texts)
    end = time.time()

    # 결과 정보 출력
    print(f"[RESULT] 벡터 개수: {len(vectors)}")
    print(f"[RESULT] 벡터 차원: {len(vectors[0])}")
    print(f"[RESULT] 수행 시간: {end - start:.2f}초")
    print(f"[RESULT] 초당 임베딩 수: {len(vectors)/(end-start):.2f} 문장/sec")

    # GPU 메모리 사용량 확인 (선택)
    if device == "cuda":
        print(f"[GPU] 현재 GPU 메모리 사용량: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

# 실행
test_embedding_performance(device="cuda")   # GPU 성능 측정
test_embedding_performance(device="cpu")    # CPU 성능 비교 (옵션)
