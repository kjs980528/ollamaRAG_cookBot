import os
import re
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pymupdf

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

try:
    from extract_images import extract_recipe_images, sanitize_filename
except ImportError:
    from cook_rag.extract_images import extract_recipe_images, sanitize_filename

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR.parent / "cook_pdf"
IMAGES_DIR = BASE_DIR / "static" / "recipe_images"
DB_PATH = BASE_DIR / "vector_db"
HISTORY_FILE = BASE_DIR / "sync_history.json"

def load_sync_history() -> Dict[str, Any]:
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_sync_history(history: Dict[str, Any]):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def get_all_pdf_files():
    files = list(PDF_DIR.glob("*.pdf")) + list(PDF_DIR.glob("*.PDF"))
    seen = set()
    unique_files = []
    for f in files:
        if f.name.lower() not in seen:
            seen.add(f.name.lower())
            unique_files.append(f)
    return unique_files

def check_pdf_changes() -> Tuple[List[Path], List[Path]]:
    """
    cook_pdf 디렉터리를 스캔하여
    (새로 추가/수정된 PDF 목록, 기존에 처리된 PDF 목록)을 반환합니다.
    """
    history = load_sync_history()
    all_pdfs = get_all_pdf_files()
    
    new_or_modified = []
    unchanged = []

    for pdf_path in all_pdfs:
        stat = pdf_path.stat()
        file_key = pdf_path.name
        file_mtime = int(stat.st_mtime)
        file_size = stat.st_size

        if file_key not in history:
            # 새로운 파일
            new_or_modified.append(pdf_path)
        else:
            rec = history[file_key]
            # 수정 시간이나 크기가 달라진 경우
            if rec.get("mtime") != file_mtime or rec.get("size") != file_size:
                new_or_modified.append(pdf_path)
            else:
                unchanged.append(pdf_path)

    return new_or_modified, unchanged

def get_page_images(images_dir: Path, stem: str, page_num: int):
    matched = []
    for f in images_dir.glob(f"{stem}_p{page_num}_img*.*"):
        matched.append(f.name)
    if not matched and page_num > 1:
        for f in images_dir.glob(f"{stem}_p{page_num-1}_img*.*"):
            matched.append(f.name)
    return matched

def parse_pdf_to_documents(pdf_files: List[Path], images_dir: Path) -> List[Document]:
    raw_docs = []
    for pdf_path in pdf_files:
        stem = sanitize_filename(pdf_path.stem)
        doc = pymupdf.open(pdf_path)
        print(f"[*] Parsing {pdf_path.name} ({len(doc)} pages)...")

        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            page = doc[page_idx]
            text = page.get_text().strip()

            if not text or len(text) < 15:
                page_imgs = get_page_images(images_dir, stem, page_num)
                if not page_imgs:
                    continue
                text = f"[{pdf_path.stem} - {page_num}페이지 요리 레시피 이미지 자료]"

            images = get_page_images(images_dir, stem, page_num)
            metadata = {
                "source": pdf_path.name,
                "stem": stem,
                "page": page_num,
                "images": json.dumps(images, ensure_ascii=False)
            }
            content = f"--- [출처: {pdf_path.name} | {page_num}페이지] ---\n{text}"
            raw_docs.append(Document(page_content=content, metadata=metadata))

        doc.close()
    return raw_docs

def incremental_sync() -> Dict[str, Any]:
    """
    새로 추가되거나 변경된 PDF만 감지하여 스마트하게 증분 동기화(Incremental Sync)를 수행합니다.
    """
    new_files, unchanged = check_pdf_changes()
    embeddings = OllamaEmbeddings(model="bge-m3")

    # DB가 아직 아예 없다면 전체 빌드 필요
    if not DB_PATH.exists():
        print("[!] 벡터 DB가 없습니다. 전체 요리책 빌드를 시작합니다.")
        return full_rebuild()

    # 새 파일이 전혀 없는 경우
    if not new_files:
        print("[+] 변경되거나 추가된 PDF가 없습니다. (이미 최신 상태)")
        return {
            "status": "up_to_date",
            "message": "새로 추가된 PDF가 없습니다. 모든 요리책이 최신 상태입니다.",
            "new_count": 0,
            "new_files": []
        }

    print(f"[*] {len(new_files)}개의 새로운 요리책 감지: {[f.name for f in new_files]}")

    # 1. 새 파일에 대해서만 요리 사진 추출
    extract_recipe_images(PDF_DIR, IMAGES_DIR, target_files=new_files)

    # 2. 새 파일 텍스트 파싱 및 청킹
    raw_docs = parse_pdf_to_documents(new_files, IMAGES_DIR)
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    new_chunks = splitter.split_documents(raw_docs)
    print(f"[*] 새 요리책에서 생성된 Chunk 수: {len(new_chunks)}")

    # 3. 기존 Chroma DB에 새 Chunk들만 추가 (add_documents)
    vectorstore = Chroma(persist_directory=str(DB_PATH), embedding_function=embeddings)
    if new_chunks:
        vectorstore.add_documents(new_chunks)
    print(f"[+] Chroma DB에 새 문서 {len(new_chunks)}개 증분 추가 완료!")

    # 4. 동기화 기록 갱신
    history = load_sync_history()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for f in new_files:
        stat = f.stat()
        history[f.name] = {
            "mtime": int(stat.st_mtime),
            "size": stat.st_size,
            "synced_at": now_str
        }
    save_sync_history(history)

    return {
        "status": "updated",
        "message": f"새 요리책 {len(new_files)}권({', '.join([f.name for f in new_files])})의 사진 추출 및 증분 임베딩을 완료했습니다!",
        "new_count": len(new_files),
        "new_files": [f.name for f in new_files]
    }

def full_rebuild() -> Dict[str, Any]:
    """전체 요리책을 처음부터 다시 빌드합니다."""
    all_pdfs = get_all_pdf_files()
    extract_recipe_images(PDF_DIR, IMAGES_DIR, target_files=all_pdfs)

    raw_docs = parse_pdf_to_documents(all_pdfs, IMAGES_DIR)
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    chunks = splitter.split_documents(raw_docs)

    embeddings = OllamaEmbeddings(model="bge-m3")
    if DB_PATH.exists():
        shutil.rmtree(DB_PATH)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_PATH)
    )

    history = {}
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for f in all_pdfs:
        stat = f.stat()
        history[f.name] = {
            "mtime": int(stat.st_mtime),
            "size": stat.st_size,
            "synced_at": now_str
        }
    save_sync_history(history)

    return {
        "status": "rebuilt",
        "message": f"총 {len(all_pdfs)}권의 요리책 전체 빌드를 완료했습니다.",
        "new_count": len(all_pdfs),
        "new_files": [f.name for f in all_pdfs]
    }

def build_vector_db():
    return incremental_sync()

if __name__ == "__main__":
    incremental_sync()
