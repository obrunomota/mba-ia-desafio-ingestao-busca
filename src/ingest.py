"""Ingestão: lê o PDF, divide em chunks, gera embeddings e persiste no PostgreSQL + pgVector."""

import logging
import os
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vectorstore import PROJECT_ROOT, get_embeddings, get_vector_store

PDF_PATH = os.getenv("PDF_PATH") or "document.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Na primeira ingestão não há collection para apagar; evita o aviso "Collection not found".
logging.getLogger("langchain_postgres").setLevel(logging.ERROR)


def resolve_pdf_path(path: str) -> Path:
    """Caminhos relativos são resolvidos a partir da raiz do projeto, independente do diretório atual."""
    pdf_path = Path(path)
    return pdf_path if pdf_path.is_absolute() else PROJECT_ROOT / pdf_path


def load_chunks(pdf_path: Path) -> list[Document]:
    pages = PyPDFLoader(str(pdf_path)).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        chunk.metadata = {k: v for k, v in chunk.metadata.items() if v not in ("", None)}
    return chunks


def ingest_pdf() -> None:
    pdf_path = resolve_pdf_path(PDF_PATH)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Arquivo PDF não encontrado: {pdf_path}")

    chunks = load_chunks(pdf_path)
    if not chunks:
        raise ValueError(f"Nenhum conteúdo extraído de {pdf_path.name}.")
    print(f"{pdf_path.name}: {len(chunks)} chunks gerados (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"{pdf_path.stem}-{i}" for i in range(len(chunks))]

    # Os embeddings são gerados (em lote) antes de tocar no banco: se a API falhar,
    # os dados de uma ingestão anterior permanecem intactos.
    vectors = get_embeddings().embed_documents(texts)

    store = get_vector_store(pre_delete_collection=True)
    store.add_embeddings(texts=texts, embeddings=vectors, metadatas=metadatas, ids=ids)
    print(f"Ingestão concluída: {len(ids)} vetores armazenados na collection '{store.collection_name}'.")


if __name__ == "__main__":
    try:
        ingest_pdf()
    except Exception as exc:
        sys.exit(f"Erro na ingestão: {exc}")
