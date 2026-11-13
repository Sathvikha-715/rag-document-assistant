import os
import tempfile

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader

_reader = None


def get_ocr_reader():
    """Load EasyOCR only the first time a scanned PDF needs it."""
    global _reader
    if _reader is None:
        import easyocr  # imported here so normal PDFs never need it
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def load_pdf(pdf_path: str):
    """
    1. Try normal text extraction.
    2. If the PDF has no readable text, fall back to OCR.
    """
    documents = PyMuPDFLoader(pdf_path).load()

    extracted_text = "".join(doc.page_content.strip() for doc in documents)
    if extracted_text:
        return documents

    # ---- OCR fallback (scanned PDFs) ----
    try:
        reader = get_ocr_reader()
    except Exception as e:
        raise RuntimeError(
            "This looks like a scanned PDF, but OCR is not available "
            f"in this environment ({e})."
        )

    pdf = fitz.open(pdf_path)
    ocr_documents = []

    try:
        for page_num, page in enumerate(pdf):
            pix = page.get_pixmap(dpi=200)  # 200 is faster than 300 and usually enough

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
                image_path = temp.name
            pix.save(image_path)

            try:
                text = "\n".join(
                    reader.readtext(image_path, detail=0, paragraph=True)
                )
                if text.strip():
                    ocr_documents.append(
                        Document(
                            page_content=text,
                            metadata={"page": page_num, "source": pdf_path},
                        )
                    )
            finally:
                os.remove(image_path)
    finally:
        pdf.close()

    return ocr_documents