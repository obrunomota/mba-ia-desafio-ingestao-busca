"""Configuração compartilhada: modelos da OpenAI e conexão com o PostgreSQL + pgVector."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-5.4-nano"
DB_CONNECT_TIMEOUT_SECONDS = 5


def require_env(name: str) -> str:
    """Retorna o valor da variável de ambiente ou falha com uma mensagem clara."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Variável de ambiente '{name}' não definida. Verifique o arquivo .env.")
    return value


def get_embeddings() -> OpenAIEmbeddings:
    require_env("OPENAI_API_KEY")
    return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL)


def get_llm() -> ChatOpenAI:
    require_env("OPENAI_API_KEY")
    return ChatOpenAI(model=os.getenv("OPENAI_LLM_MODEL") or DEFAULT_LLM_MODEL, temperature=0)


def get_vector_store(pre_delete_collection: bool = False) -> PGVector:
    """Cria o PGVector conectado à collection configurada.

    pre_delete_collection=True apaga a collection existente antes de usar,
    garantindo uma ingestão idempotente (sem chunks duplicados ou obsoletos).
    """
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=require_env("PG_VECTOR_COLLECTION_NAME"),
        connection=require_env("DATABASE_URL"),
        use_jsonb=True,
        pre_delete_collection=pre_delete_collection,
        # Sem timeout, a conexão pode ficar pendurada indefinidamente se o banco estiver fora do ar.
        engine_args={"connect_args": {"connect_timeout": DB_CONNECT_TIMEOUT_SECONDS}},
    )
