"""CLI de chat: recebe perguntas no terminal e responde com base no conteúdo do PDF."""

from search import search_prompt

EXIT_COMMANDS = {"sair", "exit", "quit"}


def main() -> None:
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Faça sua pergunta (digite 'sair' para encerrar):\n")

    while True:
        try:
            question = input("PERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            break

        try:
            answer = chain.invoke(question)
        except Exception as exc:
            print(f"Erro ao processar a pergunta: {exc}\n")
            continue

        # O terminal não renderiza Markdown: remove marcadores de negrito da resposta.
        print(f"RESPOSTA: {answer.replace('**', '').strip()}\n")

    print("Até logo!")


if __name__ == "__main__":
    main()
