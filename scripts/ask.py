import argparse

from app.rag import ask_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question to Thesis RAG Bot")
    parser.add_argument("question", type=str)
    args = parser.parse_args()

    result = ask_question(args.question)
    print("\nAnswer:\n")
    print(result["answer"])
    print("\nSources:\n")
    for src in result["sources"]:
        print(f"- {src['source']} (page: {src['page']})")


if __name__ == "__main__":
    main()
