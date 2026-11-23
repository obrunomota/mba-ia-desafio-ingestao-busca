"""Busca: recupera os chunks mais relevantes no pgVector e monta a chain RAG com a LLM."""

import sys

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_postgres import PGVector

from vectorstore import get_llm, get_vector_store

TOP_K = 10

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def build_context_retriever(store: PGVector, k: int = TOP_K) -> RunnableLambda:
    """Vetoriza a pergunta, busca os k chunks mais similares e os concatena como contexto."""

    def retrieve(question: str) -> str:
        results = store.similarity_search_with_score(question, k=k)
        return "\n\n".join(doc.page_content.strip() for doc, _score in results)

    return RunnableLambda(retrieve)


def search_prompt(question: str | None = None) -> Runnable | str | None:
    """Monta a chain RAG (busca -> prompt -> LLM).

    Sem argumento, retorna a chain pronta para ser reutilizada a cada pergunta
    (conexão com o banco e clientes de IA são criados uma única vez).
    Com `question`, executa a chain e retorna a resposta diretamente.
    Retorna None se a inicialização falhar.
    """
    try:
        store = get_vector_store()
        llm = get_llm()
    except Exception as exc:
        print(f"Erro ao inicializar a busca: {exc}", file=sys.stderr)
        return None

    chain = (
        {"contexto": build_context_retriever(store), "pergunta": RunnablePassthrough()}
        | PromptTemplate.from_template(PROMPT_TEMPLATE)
        | llm
        | StrOutputParser()
    )

    return chain.invoke(question) if question else chain
