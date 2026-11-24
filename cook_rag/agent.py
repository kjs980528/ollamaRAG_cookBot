import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "vector_db"

# 1. 질문 재작성 프롬프트 (대화 문맥 반영)
REPHRASE_TEMPLATE = """이전 대화 내용과 사용자의 새로운 질문을 분석하여, 요리책 문서 검색에 사용할 구체적이고 독립적인 검색 질문 하나를 작성하세요.

[이전 대화 내용]
{chat_history}

[사용자의 새로운 질문]
{question}

[작성 규칙]
1. 사용자가 "3번 레시피 알려줘", "그 요리 만드는 법은?", "거기에 들어가는 재료는?" 같이 이전 대화의 번호나 대명사('그거', '그 요리')를 언급한 경우, 이전 대화에서 가리키는 정확한 요리 이름을 파악하여 검색 질문을 완성하세요. (예: "궁중떡볶이 레시피 및 만드는 법")
2. 이전 대화와 무관한 새로운 질문이라면 원래 질문의 핵심 키워드를 유지하세요.
3. 부가 설명, 인사말, 따옴표 없이 오직 검색에 사용할 질문 한 줄만 출력하세요.
"""

# 2. 메인 셰프 답변 프롬프트
PROMPT_TEMPLATE = """당신은 '쿡봇(CookBot) - 스마트 레시피 셰프'입니다.
제공된 [요리책 참고자료]와 [이전 대화 내용]을 적극 활용하여 사용자의 질문 의도에 맞게 친절하고 센스 있게 답변하세요.

[이전 대화 내용]
{chat_history}

[요리책 참고자료]
{context}

[사용자 질문]
{question}

[답변 작성 가이드]
1. 질문 유형에 따른 답변 방식 (매우 중요):
   (A) '단순 추천 / 아이디어 요청'인 경우:
       - 사용자가 "요리 추천해줘", "달달한 거 뭐 있어?", "오늘 뭐 먹지?" 등 단순 추천만 요청하고, 구체적인 "만드는 법"이나 "레시피"를 묻지 않았다면:
       - **절대로 전체 재료 분량이나 1, 2, 3 단계별 조리 순서를 전부 나열하지 마세요.**
       - 참고자료에서 어울리는 요리 1~3가지를 골라 **요리 이름과 간단한 특징/소개(1~2줄)**만 산뜻하게 제안하세요.
       - 마지막에 "원하시는 요리의 상세 레시피나 조리법이 궁금하시면 언제든 말씀해 주세요!"라고 안내하세요.
   (B) '상세 레시피 / 조리법 요청'인 경우 (이전 추천 요리에 대한 레시피 요청 포함):
       - 사용자가 "만드는 법 알려줘", "레시피 알려줘", "3번 레시피 알려줘", "어떻게 만들어?", "재료 알려줘" 등을 명시적으로 요구한 경우:
       - 이전 대화에서 추천했던 번호나 요리가 있다면 그 요리를 명확히 짚어주며 안내하세요.
       - **요리 이름 & 특징**
       - **필요한 재료 및 분량**
       - **단계별 조리 순서(1, 2, 3...)**
       - **셰프의 꿀팁 & 주의사항**을 체계적으로 안내하세요.

2. 자연스러운 한국어 식재료 계량 단위 사용:
   - 식재료 수량을 안내할 때 한국 요리 관습에 맞는 자연스러운 단위를 사용하세요.
   - 특히 조개류(굴, 가리비, 전복, 바지락 등)나 달걀, 과채류에는 어색하게 '마리'를 쓰지 말고 반드시 **'개'**(또는 굴 알맹이는 '알/미')라는 자연스러운 단위를 쓰세요. (예: '굴 2개', '가리비 1개', '전복 2개')

3. 이미지 활용:
   - 참고자료에 명시된 [관련 이미지 링크] 중 설명 중인 요리와 맞는 이미지가 있다면 `![요리 사진](/static/recipe_images/파일명)` 형태로 본문에 삽입하세요.

4. 출처 안내 (필수):
   - 답변 마지막에 참고한 실제 PDF 파일명과 페이지를 명시하세요.
   - **주의: 절대로 '[요리책 이름]'이라는 글자를 그대로 출력하지 마세요.** 반드시 참고자료의 출처에 적힌 실제 파일 이름(예: `cook_recipebook.pdf`, `요아마미 레시피북.pdf` 등)을 적으세요.
   - 예시: "📖 출처: cook_recipebook.pdf (p.177)"
"""

class CookAgent:
    def __init__(self, model_name: str = "qwen3:8b", embedding_model: str = "bge-m3"):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.llm = ChatOllama(model=model_name, temperature=0.3)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        
        if not DB_DIR.exists():
            self.vectorstore = None
            self.retriever = None
        else:
            self.vectorstore = Chroma(
                persist_directory=str(DB_DIR),
                embedding_function=self.embeddings
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

        # 메인 답변 체인
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.chain = prompt | self.llm | StrOutputParser()

        # 질문 재작성 체인
        rephrase_prompt = ChatPromptTemplate.from_template(REPHRASE_TEMPLATE)
        self.rephrase_chain = rephrase_prompt | self.llm | StrOutputParser()

    def reload_db(self):
        if DB_DIR.exists():
            self.vectorstore = Chroma(
                persist_directory=str(DB_DIR),
                embedding_function=self.embeddings
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
            return True
        return False

    def format_chat_history(self, history: Optional[List[Dict[str, str]]]) -> str:
        if not history:
            return "(이전 대화 없음)"
        lines = []
        for msg in history[-6:]:  # 최근 3턴(6개 메시지)
            role = "사용자" if msg.get("role") == "user" else "셰프"
            content = msg.get("content", "").strip()
            # 내용이 너무 길면 축약
            if len(content) > 300:
                content = content[:300] + "..."
            lines.append(f"{role}: {content}")
        return "\n".join(lines) if lines else "(이전 대화 없음)"

    def ask(self, question: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        if not self.retriever:
            if not self.reload_db():
                return {
                    "answer": "레시피 벡터 DB가 아직 생성되지 않았습니다. 먼저 DB를 생성해주세요.",
                    "images": [],
                    "sources": []
                }

        chat_history_str = self.format_chat_history(history)
        search_query = question

        # 이전 대화가 있는 경우, 문맥을 반영하여 검색 질문 재작성
        if history and len(history) > 0:
            try:
                rewritten = self.rephrase_chain.invoke({
                    "chat_history": chat_history_str,
                    "question": question
                }).strip()
                if rewritten and len(rewritten) > 2:
                    print(f"[*] 검색어 재작성: '{question}' -> '{rewritten}'")
                    search_query = rewritten
            except Exception as e:
                print(f"[!] 질문 재작성 실패, 원래 질문 유지: {e}")
                search_query = question

        # 재작성된 질문으로 벡터 DB 검색
        docs = self.retriever.invoke(search_query)

        # Context 조립 및 이미지, 출처 수집
        context_parts = []
        collected_images = []
        sources = []
        seen_images = set()

        for doc in docs:
            metadata = doc.metadata
            source_name = metadata.get("source", "알 수 없음")
            page_num = metadata.get("page", 1)
            stem = metadata.get("stem", "")
            
            # 이미지 파싱
            images_raw = metadata.get("images", "[]")
            try:
                img_list = json.loads(images_raw)
            except Exception:
                img_list = []

            img_hints = [f"/static/recipe_images/{img}" for img in img_list]
            img_text = f"\n[해당 페이지 관련 이미지]: {', '.join(img_hints)}" if img_hints else "\n[해당 페이지 관련 이미지]: 없음"
            
            context_parts.append(
                f"{doc.page_content}{img_text}\n"
            )

            for img in img_list:
                if img not in seen_images:
                    seen_images.add(img)
                    collected_images.append({
                        "filename": img,
                        "url": f"/static/recipe_images/{img}",
                        "source": source_name,
                        "page": page_num
                    })

            sources.append({
                "source": source_name,
                "page": page_num,
                "stem": stem
            })

        context_str = "\n====================\n".join(context_parts)

        # LLM 응답 생성 (이전 대화 히스토리 전달)
        try:
            answer = self.chain.invoke({
                "chat_history": chat_history_str,
                "context": context_str,
                "question": question
            })
            
            # 혹시라도 LLM이 플레이스홀더를 그대로 출력한 경우 실제 소스 파일명으로 안전하게 치환
            if sources:
                primary_source = sources[0]["source"]
                answer = re.sub(r'\[요리책\s*이름\]', primary_source, answer)
                answer = re.sub(r'\[요리책\s*제목\]', primary_source, answer)
                answer = re.sub(r'\[요리책\]', primary_source, answer)

            # 조개류/어패류 식재료의 어색한 단위('마리')를 자연스러운 '개'로 자동 교정
            shellfish_list = ["굴", "가리비", "전복", "바지락", "홍합", "꼬막", "키조개", "소라", "백합", "조개"]
            for item in shellfish_list:
                answer = re.sub(rf'({item}[^\n\d]*\d+)\s*마리', r'\1개', answer)

        except Exception as e:
            answer = f"답변 생성 중 오류가 발생했습니다: {str(e)}"

        # '추천' 질문인 경우 불필요하게 많은 조리 과정 사진이 쏟아지지 않도록 최대 4장으로 제한
        is_recommendation = any(kw in question for kw in ["추천", "골라", "뭐 먹", "어떤 요리", "메뉴 제안", "아이디어"])
        if is_recommendation and len(collected_images) > 4:
            collected_images = collected_images[:4]

        return {
            "answer": answer,
            "images": collected_images,
            "sources": sources,
            "search_query": search_query
        }
