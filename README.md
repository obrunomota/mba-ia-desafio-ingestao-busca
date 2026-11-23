# Ingestão e Busca Semântica com LangChain e Postgres

Lê um PDF, guarda seus trechos como vetores no PostgreSQL + pgVector e responde perguntas no terminal **somente** com base no conteúdo do PDF.

- **Stack:** Python, LangChain, PostgreSQL + pgVector (Docker), OpenAI
- **Modelos:** `text-embedding-3-small` (embeddings) e `gpt-5.4-nano` (respostas)
- **Chunks:** 1000 caracteres, overlap de 150 — **Busca:** `k=10`

## Pré-requisitos

Python 3.10+, Docker e uma API Key da OpenAI.

## Como executar

```bash
# 1. Ambiente virtual e dependências
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Variáveis de ambiente (preencha OPENAI_API_KEY no .env)
cp .env.example .env

# 3. Banco de dados
docker compose up -d

# 4. Ingestão do PDF
python src/ingest.py

# 5. Chat
python src/chat.py
```

Digite `sair` para encerrar o chat.

## Exemplo

```
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento da empresa SuperTechIABrazil é R$ 10.000.000,00.

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

## Estrutura

```
├── docker-compose.yml   # PostgreSQL + pgVector
├── requirements.txt
├── .env.example
├── document.pdf         # PDF ingerido
└── src/
    ├── vectorstore.py   # Configuração compartilhada (modelos e banco)
    ├── ingest.py        # Ingestão do PDF
    ├── search.py        # Busca + prompt + LLM
    └── chat.py          # CLI
```

## Observações

- A ingestão é idempotente: rodar de novo recria a collection, sem duplicar chunks.
- Ao trocar o modelo de embeddings, apague o volume (`docker compose down -v`) e refaça a ingestão, pois a dimensão dos vetores muda.
