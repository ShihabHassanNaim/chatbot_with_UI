import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph_back import chatbot

CONFIG = {'configurable' : {'thread_id': 'thread-1'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []


for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])


user_input = st.chat_input("Send a message to the AI assistant:")

if user_input:
    # record and show user message
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # invoke the graph with only the new message; the graph's checkpointer keeps context
    state = chatbot.invoke({'message': [HumanMessage(content=user_input)]}, config=CONFIG)
    assistant_reply = state['message'][-1].content

    st.session_state['message_history'].append({'role': 'assistant', 'content': assistant_reply})
    with st.chat_message('assistant'):
        st.text(assistant_reply)