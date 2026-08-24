import uuid

from .agent import run_agent
from .logging_config import setup_logging


def main():
    setup_logging()

    print("====================================")
    print("          KUDOO")
    print("====================================")
    print("Ask questions about your sales data.")
    print("Type 'quit' to exit.\n")

    conversation_id = str(uuid.uuid4())

    while True:
        user_question = input("You: ").strip()

        if user_question.lower() == "quit":
            print("\nGoodbye!")
            return

        if not user_question:
            continue

        result = run_agent(
            user_question=user_question,
            conversation_id=conversation_id,
        )

        print("\nKUDOO:")
        print(result["answer"])
        print()


if __name__ == "__main__":
    main()