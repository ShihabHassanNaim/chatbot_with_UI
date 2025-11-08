"""Backend LangGraph setup for the Streamlit chatbot.

Key points:
- Uses ChatOllama with local model `llama3.2:1b`.
- Maintains conversation state via the `add_messages` annotation.
- Returns only the newly generated assistant message each node run; LangGraph aggregates.
"""

from langgraph.graph import StateGraph, add_messages, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.7,
)


class ChatState(TypedDict):
    message: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    """Single chat node that takes the latest user message and produces an assistant reply.

    Adds basic defensive handling for cases where the Ollama stream returns no data.
    """
    messages = state["message"]
    last = messages[-1] if messages else HumanMessage(content="Hello")

    try:
        # Preferred: pass the list for full context
        response = llm.invoke(messages)
    except ValueError as e:
        # Retry with only the last message's content if we hit the 'No data received' edge case
        if "No data received" in str(e):
            response = llm.invoke(last)
        else:
            raise

    return {"message": [response]}


checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)