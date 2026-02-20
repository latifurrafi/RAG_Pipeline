# main.py

from rag_pipeline import RAGPipeline


def main():

    rag = RAGPipeline()

    print("\nAdaptive RAG CSV System Ready")
    print("Type 'exit' to quit\n")

    while True:

        query = input("Query: ")

        if query.lower() == "exit":
            break

        answer = rag.ask(query)

        print("\nAnswer:")
        print(answer)
        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()