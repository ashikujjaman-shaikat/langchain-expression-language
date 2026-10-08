"""Smart Q&A assistant built with LCEL.

Run:  python app.py   (needs Ollama running with llama3.2 and mistral:7b pulled)

Combines: with_fallbacks, assign, RunnableBranch (history-aware rephrasing + routing), streaming.
"""
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_ollama import ChatOllama

parser = StrOutputParser()
llm = ChatOllama(model="llama3.2", temperature=0).with_fallbacks(
    [ChatOllama(model="mistral:7b", temperature=0)]
)

# Rewrite follow-up questions so they stand alone (only when history exists)
rephrase = (
    ChatPromptTemplate.from_messages([
        ("system", "Rewrite the latest question as a standalone question using the chat history. Return only the question."),
        ("placeholder", "{chat_history}"),
        ("human", "{question}"),
    ])
    | llm
    | parser
)
standalone = RunnableBranch((lambda x: bool(x["chat_history"]), rephrase), itemgetter("question"))

# Classify the topic so the question can be routed to a specialist
classify = (
    ChatPromptTemplate.from_template(
        "Classify the question as exactly one word: python, math, or other.\n"
        "Question: {question}\nAnswer:"
    )
    | llm
    | parser
    | (lambda s: s.strip().lower())
)


def specialist(role: str):
    return ChatPromptTemplate.from_messages([("system", role), ("human", "{question}")]) | llm | parser


route = RunnableBranch(
    (lambda x: "python" in x["topic"], specialist("You are a Python mentor. Answer briefly with a short code example.")),
    (lambda x: "math" in x["topic"], specialist("You are a math tutor. Solve step by step.")),
    specialist("You are a helpful assistant. Answer concisely."),
)

chain = (
    RunnablePassthrough.assign(question=standalone)
    .assign(topic=classify)
    | route
)


def main() -> None:
    history: list[tuple[str, str]] = []
    print("Ask anything (empty line or 'exit' to quit).")
    while (question := input("\nYou: ").strip()) and question.lower() != "exit":
        answer = ""
        print("Bot: ", end="")
        for chunk in chain.stream({"question": question, "chat_history": history}):
            print(chunk, end="", flush=True)
            answer += chunk
        print()
        history += [("human", question), ("ai", answer)]


if __name__ == "__main__":
    main()
