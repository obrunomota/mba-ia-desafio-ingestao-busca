# Ingestão e Busca Semântica com LangChain e Postgres

Solução do desafio técnico do MBA em Engenharia de Software com IA (Full Cycle).

O software:

- **Ingestão**: lê o `document.pdf`, divide o conteúdo em chunks de 1000 caracteres (overlap de 150), gera os embeddings e armazena os vetores no PostgreSQL com a extensão pgVector.
- **Busca**: um chat no terminal (CLI) recebe perguntas, busca os 10 trechos mais relevantes no banco vetorial e responde **somente** com base no conteúdo do PDF.

## Tecnologias

- Python 3.10+
- LangChain (`langchain-openai`, `langchain-postgres`, `langchain-community`, `langchain-text-splitters`)
- PostgreSQL 17 + pgVector (via Docker Compose)
- OpenAI
  - Embeddings: `text-embedding-3-small` (1536 dimensões)
  - LLM: `gpt-5.4-nano` (`temperature=0`)

Os modelos foram escolhidos após consultar os modelos disponíveis na API da OpenAI: são as opções mais leves e baratas, suficientes para o volume do desafio. Ambos podem ser trocados pelo `.env`.

## Estrutura do projeto

```
├── docker-compose.yml     # PostgreSQL + pgVector
├── requirements.txt       # Dependências
├── .env.example           # Template das variáveis de ambiente
├── src/
│   ├── vectorstore.py     # Configuração compartilhada: embeddings, LLM e conexão com o PGVector
│   ├── ingest.py          # Script de ingestão do PDF
│   ├── search.py          # Busca semântica + prompt + chain com a LLM
│   └── chat.py            # CLI para interação com o usuário
├── document.pdf           # PDF para ingestão
└── README.md
```

## Pré-requisitos

- Python 3.10 ou superior
- Docker e Docker Compose
- Uma API Key da OpenAI

## Como executar

### 1. Clonar o repositório

```bash
git clone https://github.com/obrunomota/mba-ia-desafio-ingestao-busca.git
cd mba-ia-desafio-ingestao-busca
```

### 2. Criar o ambiente virtual e instalar as dependências

Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Configurar as variáveis de ambiente

```bash
cp .env.example .env
```

Edite o `.env` e preencha a sua `OPENAI_API_KEY`. Os demais valores já vêm prontos para o banco do `docker-compose.yml`:

| Variável                    | Descrição                                      | Valor padrão                                                   |
| --------------------------- | ---------------------------------------------- | -------------------------------------------------------------- |
| `OPENAI_API_KEY`            | API Key da OpenAI                              | —                                                              |
| `OPENAI_EMBEDDING_MODEL`    | Modelo de embeddings                           | `text-embedding-3-small`                                       |
| `OPENAI_LLM_MODEL`          | Modelo de LLM usado para responder             | `gpt-5.4-nano`                                                 |
| `DATABASE_URL`              | Conexão com o PostgreSQL                       | `postgresql+psycopg://postgres:postgres@localhost:5432/rag`    |
| `PG_VECTOR_COLLECTION_NAME` | Nome da collection de vetores                  | `document_chunks`                                              |
| `PDF_PATH`                  | Caminho do PDF (relativo à raiz ou absoluto)   | `document.pdf`                                                 |

### 4. Subir o banco de dados

```bash
docker compose up -d
```

O serviço `bootstrap_vector_ext` cria a extensão `vector` automaticamente assim que o PostgreSQL fica saudável.

### 5. Executar a ingestão do PDF

```bash
python src/ingest.py
```

Saída esperada:

```
document.pdf: 67 chunks gerados (size=1000, overlap=150).
Ingestão concluída: 67 vetores armazenados na collection 'document_chunks'.
```

### 6. Rodar o chat

```bash
python src/chat.py
```

Digite `sair` (ou pressione `Ctrl+C`) para encerrar.

## Exemplo de uso

```
Faça sua pergunta (digite 'sair' para encerrar):

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento da SuperTechIABrazil é R$ 10.000.000,00.

PERGUNTA: Qual o ano de fundação da Alfa Energia S.A.?
RESPOSTA: O ano de fundação da Alfa Energia S.A. é 1972.

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

PERGUNTA: Qual é a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

## Como funciona

**Ingestão (`src/ingest.py`)**

1. `PyPDFLoader` carrega o PDF página a página.
2. `RecursiveCharacterTextSplitter` divide o texto em chunks de 1000 caracteres com overlap de 150.
3. Os embeddings de todos os chunks são gerados em lote.
4. Os vetores são gravados no pgVector via `PGVector`.

A ingestão é **idempotente**: a collection é recriada a cada execução, então rodar o script novamente não duplica chunks. Os embeddings são gerados *antes* de tocar no banco — se a API falhar, os dados da ingestão anterior permanecem intactos.

**Busca e chat (`src/search.py` e `src/chat.py`)**

1. A pergunta é vetorizada e `similarity_search_with_score(pergunta, k=10)` retorna os 10 chunks mais relevantes.
2. Os chunks são concatenados no `CONTEXTO` do prompt definido no desafio.
3. A LLM responde seguindo as regras do prompt, e a resposta é exibida no terminal.

O fluxo é montado como uma chain LCEL (`retriever | prompt | llm | parser`), criada uma única vez ao iniciar o chat e reutilizada a cada pergunta, sem reabrir conexões com o banco ou recriar os clientes da OpenAI.

## Solução de problemas

- **Erro de dimensão ao trocar o modelo de embeddings**: modelos diferentes geram vetores com dimensões diferentes. Altere `PG_VECTOR_COLLECTION_NAME` para uma nova collection ou remova o volume do banco (`docker compose down -v`) e refaça a ingestão.
- **`Erro ao inicializar a busca ... connection timeout expired`**: o banco não está acessível. Verifique se o container está de pé com `docker compose ps`.
- **Porta 5432 em uso**: outro PostgreSQL está rodando na máquina. Pare-o ou altere o mapeamento de porta no `docker-compose.yml` e na `DATABASE_URL`.
