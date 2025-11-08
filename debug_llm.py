"""Quick manual test for the local Ollama model via LangChain-Ollama ChatOllama.
Run this after ensuring `ollama list` shows `llama3.2:1b`.
"""
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


def main():
    llm = ChatOllama(model="llama3.2:1b", temperature=0.7)
    prompt = "Say hello in one short sentence."
    resp = llm.invoke([HumanMessage(content=prompt)])
    print("Model reply:", resp.content)


if __name__ == "__main__":
    main()
