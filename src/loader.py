import os
import tempfile

import fitz  # PyMuPDF
import easyocr

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader

# Initialize OCR reader once
reader = easyocr.Reader(['en'], gpu=False)


def load_pdf(pdf_path: str):
    """
    Loads a PDF.

    1. Try extracting text using PyMuPDFLoader.
    2. If no readable text exists, automatically perform OCR.
    """

    # -------------------------
    # Try normal text extraction
    # -------------------------
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    extracted_text = "".join(
        doc.page_content.strip()
        for doc in documents
    )

    # If readable text exists, return it
    if extracted_text:
        return documents

    print("No embedded text found. Running OCR...")

    # -------------------------
    # OCR Fallback
    # -------------------------
    pdf = fitz.open(pdf_path)
    ocr_documents = []

    for page_num, page in enumerate(pdf):

        # Render page as image
        pix = page.get_pixmap(dpi=300)

        with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False
        ) as temp:

            image_path = temp.name

        pix.save(image_path)

        try:
            # OCR
            text = "\n".join(
                reader.readtext(
                    image_path,
                    detail=0,
                    paragraph=True
                )
            )

            if text.strip():
                ocr_documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "page": page_num,
                            "source": pdf_path
                        }
                    )
                )

        finally:
            os.remove(image_path)

    pdf.close()

    return ocr_documents