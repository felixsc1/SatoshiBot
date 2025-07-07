from backend.core import run_llm
import streamlit as st

if "user_prompt_history" not in st.session_state:
    st.session_state.user_prompt_history = []
if "chat_answers_history" not in st.session_state:
    st.session_state.chat_answers_history = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


st.header("Satoshi ₿ot")

prompt = st.text_input("Enter your prompt here", placeholder="What is Satoshi Nakamoto's stance on Libertarianism?")

def create_sources_string(sources):
    if not sources:
        return ""
    sources_list = list(sources)
    # sources_list.sort()
    sources_string = "**Sources:**\n"
    for i, source in enumerate(sources_list):
        # Get the URL and filename from metadata
        url = source.metadata["source_url"]
        display_name = source.metadata["filename"]
        
        # Create markdown link: [display_text](url)
        sources_string += f"{i+1}. [{display_name}]({url})\n"
    return sources_string

if prompt:
    with st.spinner("Thinking..."):
        generated_response = run_llm(prompt, chat_history=st.session_state.chat_history)
        sources = [doc for doc in generated_response["source_documents"]]
        formatted_response = f"**Answer:** {generated_response['result']}\n\n {create_sources_string(sources)}"

        st.session_state.user_prompt_history.append(prompt)
        st.session_state.chat_answers_history.append(formatted_response)
        st.session_state.chat_history.append((prompt, generated_response["result"]))

if st.session_state.chat_answers_history:
    for generated_response, user_prompt in zip(st.session_state.chat_answers_history, st.session_state.user_prompt_history):
        st.chat_message("user").write(user_prompt)
        st.chat_message("assistant").write(generated_response)

# print("test")


