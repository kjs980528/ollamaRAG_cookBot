import os
import re
from pathlib import Path
from typing import List, Optional
import pymupdf

def sanitize_filename(name: str) -> str:
    return re.sub(r'[^\w\-_\.]', '_', name)

def extract_recipe_images(pdf_dir: Path, output_dir: Path, target_files: Optional[List[Path]] = None, min_size: int = 120):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if target_files is not None:
        pdf_files = target_files
    else:
        pdf_files = list(pdf_dir.glob("*.pdf"))
    
    extracted_summary = {}

    for pdf_path in pdf_files:
        stem = sanitize_filename(pdf_path.stem)
        print(f"[*] Processing {pdf_path.name} for images...")
        doc = pymupdf.open(pdf_path)
        pdf_image_count = 0
        
        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            page = doc[page_idx]
            image_list = page.get_images(full=True)
            
            saved_for_page = 0
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    width = base_image["width"]
                    height = base_image["height"]
                    
                    if width < min_size or height < min_size:
                        continue
                    
                    if saved_for_page >= 4:
                        break

                    img_filename = f"{stem}_p{page_num}_img{img_idx+1}.{image_ext}"
                    img_filepath = output_dir / img_filename
                    
                    with open(img_filepath, "wb") as f:
                        f.write(image_bytes)
                    
                    saved_for_page += 1
                    pdf_image_count += 1
                except Exception:
                    continue
        
        extracted_summary[pdf_path.name] = pdf_image_count
        print(f"[+] {pdf_path.name}: 추출된 요리 이미지 {pdf_image_count}개")
        doc.close()

    return extracted_summary

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    pdf_dir = current_dir.parent / "cook_pdf"
    output_dir = current_dir / "static" / "recipe_images"
    extract_recipe_images(pdf_dir, output_dir)
