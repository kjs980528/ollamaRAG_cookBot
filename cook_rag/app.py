import os
import io
import re
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, url_for
import pymupdf

try:
    from agent import CookAgent
except ImportError:
    from cook_rag.agent import CookAgent

try:
    from extract_images import extract_recipe_images
    from build_db import build_vector_db
except ImportError:
    from cook_rag.extract_images import extract_recipe_images
    from cook_rag.build_db import build_vector_db

app = Flask(__name__, static_folder="static", template_folder="templates")
agent = CookAgent(model_name="qwen3:8b", embedding_model="bge-m3")

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR.parent / "cook_pdf"
IMAGES_DIR = BASE_DIR / "static" / "recipe_images"

# 동기화 상태 관리
sync_status_lock = threading.Lock()
sync_state = {
    "is_running": False,
    "step": "idle",
    "message": "",
    "error": None
}

def run_sync_task():
    global sync_state
    try:
        with sync_status_lock:
            sync_state["is_running"] = True
            sync_state["step"] = "extracting"
            sync_state["message"] = "새 요리책 PDF를 감지하여 이미지 및 증분 임베딩을 분석 중입니다..."
            sync_state["error"] = None

        # 스마트 증분 동기화 실행 (신규 파일만 감지하여 이미지 추출 및 add_documents 실행)
        sync_result = build_vector_db()

        # 에이전트 DB 리로드
        agent.reload_db()

        with sync_status_lock:
            sync_state["is_running"] = False
            sync_state["step"] = "done"
            sync_state["message"] = sync_result.get("message", "동기화가 완료되었습니다!")
    except Exception as e:
        with sync_status_lock:
            sync_state["is_running"] = False
            sync_state["step"] = "error"
            sync_state["error"] = str(e)
            sync_state["message"] = f"동기화 중 오류 발생: {str(e)}"

def sanitize_filename(name: str) -> str:
    return re.sub(r'[^\w\-_\.]', '_', name)

def get_all_pdf_files():
    files = list(PDF_DIR.glob("*.pdf")) + list(PDF_DIR.glob("*.PDF"))
    # 중복 제거 (대소문자 다른 중복 경로 방지)
    seen = set()
    unique_files = []
    for f in files:
        if f.name.lower() not in seen:
            seen.add(f.name.lower())
            unique_files.append(f)
    return unique_files

# PDF 파일 매핑 캐시
def get_pdf_map():
    pdf_map = {}
    for p in get_all_pdf_files():
        stem = sanitize_filename(p.stem)
        pdf_map[stem] = p
        pdf_map[p.name] = p
    return pdf_map

@app.route("/")
def index():
    # 등록된 요리책 목록 정보
    pdf_files = []
    for p in get_all_pdf_files():
        try:
            doc = pymupdf.open(p)
            pages = len(doc)
            doc.close()
        except Exception:
            pages = 0
        pdf_files.append({
            "name": p.name,
            "stem": sanitize_filename(p.stem),
            "pages": pages,
            "size_mb": round(p.stat().st_size / (1024 * 1024), 1)
        })
    return render_template("index.html", pdf_files=pdf_files)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    message = data.get("message", "").strip()
    history = data.get("history", [])
    
    if not message:
        return jsonify({"error": "메시지를 입력해주세요."}), 400

    try:
        result = agent.ask(message, history=history)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "answer": f"요청 처리 중 오류가 발생했습니다: {str(e)}",
            "images": [],
            "sources": []
        }), 500

@app.route("/api/status")
def status():
    db_ready = (BASE_DIR / "vector_db").exists()
    return jsonify({
        "status": "ready" if db_ready else "db_building",
        "model": "qwen3:8b",
        "embedding": "bge-m3",
        "db_exists": db_ready
    })

@app.route("/api/sync", methods=["POST"])
def start_sync():
    with sync_status_lock:
        if sync_state["is_running"]:
            return jsonify({"status": "already_running", "message": "이미 동기화 작업이 진행 중입니다."}), 400
        
        thread = threading.Thread(target=run_sync_task, daemon=True)
        thread.start()
        return jsonify({"status": "started", "message": "요리책 동기화 작업이 시작되었습니다."})

@app.route("/api/sync/status")
def get_sync_status():
    with sync_status_lock:
        return jsonify(sync_state)

@app.route("/api/page_image/<stem>/<int:page>")
def render_page_image(stem: str, page: int):
    """PDF 페이지를 고화질 이미지(PNG)로 온디맨드 렌더링하여 제공합니다."""
    pdf_map = get_pdf_map()
    target_pdf = pdf_map.get(stem)
    
    if not target_pdf or not target_pdf.exists():
        return jsonify({"error": "PDF를 찾을 수 없습니다."}), 404

    try:
        doc = pymupdf.open(target_pdf)
        if page < 1 or page > len(doc):
            return jsonify({"error": "페이지 번호가 범위를 벗어났습니다."}), 400

        page_obj = doc[page - 1]
        # DPI 150 (zoom 약 2.0배)
        mat = pymupdf.Matrix(1.5, 1.5)
        pix = page_obj.get_pixmap(matrix=mat)
        
        img_bytes = pix.tobytes("png")
        doc.close()

        return send_file(
            io.BytesIO(img_bytes),
            mimetype="image/png",
            as_attachment=False,
            download_name=f"{stem}_page_{page}.png"
        )
    except Exception as e:
        return jsonify({"error": f"페이지 렌더링 실패: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="127.0.0.1", port=port, debug=False)
