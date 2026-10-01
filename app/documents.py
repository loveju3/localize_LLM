from io import BytesIO

from pypdf import PdfReader


def parse_pdf(data, settings):
    if not data.startswith(b"%PDF-"):
        raise ValueError("僅接受 PDF 檔案")
    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise ValueError("目前不接受加密 PDF，請先解密")
        if len(reader.pages) > settings.max_pdf_pages:
            raise ValueError("PDF 超過頁數限制")
        chunks, warnings = [], []
        step = settings.chunk_chars - settings.chunk_overlap
        if step <= 0:
            raise ValueError("CHUNK_OVERLAP 必須小於 CHUNK_CHARS")
        for number, page in enumerate(reader.pages, 1):
            # Keep line breaks and page boundaries; no claims of table reconstruction.
            text = (page.extract_text() or "").replace("\x00", "").strip()
            if len(text) < 20:
                warnings.append(f"第 {number} 頁文字不足，可能是掃描頁、圖片或空白頁，需人工檢查／OCR")
                continue
            for start in range(0, len(text), step):
                part = text[start:start + settings.chunk_chars].strip()
                if part:
                    chunks.append({"page": number, "text": part})
                if len(chunks) > settings.max_chunks:
                    raise ValueError("文件片段數超過限制")
                if start + settings.chunk_chars >= len(text):
                    break
        if not chunks:
            raise ValueError("未取得可索引文字；掃描 PDF 需要先進行 OCR")
        return chunks, len(reader.pages), warnings
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("PDF 解析失敗，請確認檔案完整性") from exc
