import os
import sys
from pathlib import Path

# cook_rag 디렉터리를 sys.path에 추가
cook_rag_dir = Path(__file__).resolve().parent / "cook_rag"
sys.path.insert(0, str(cook_rag_dir))

from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print("=" * 60)
    print("  쿡봇(CookBot) - 스마트 레시피 RAG 챗봇 웹 서버 시작")
    print("  모델: Ollama qwen3:8b | 임베딩: bge-m3")
    print(f"  브라우저 접속 주소: http://localhost:{port}")
    print("=" * 60)
    app.run(host="127.0.0.1", port=port, debug=False)
