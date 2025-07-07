from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains.history_aware_retriever import create_history_aware_retriever


def run_llm(query, chat_history=[]):
    embeddings = OllamaEmbeddings(model="granite-embedding:30m")
    loaded_vectorstore = FAISS.load_local("satoshi_vectorstore", embeddings, allow_dangerous_deserialization=True)
    llm = ChatOllama(model="llama3.2:1b", temperature=0)

    # Try custom prompt that is more specific for this use case
    template = """
    Use the provided context (Satoshi Nakamoto's emails, quotes, and forum posts) to answer the question. Cite specific sources (e.g., email date, forum post title) when referencing context. If the question refers to an unspecified person, assume it is Satoshi Nakamoto. If the context is insufficient or contradictory, state this clearly and do not speculate. Provide a concise, factual, and neutral answer.

    Context: {context}

    Question: {input}

    Answer:
    """
    custom_rag_prompt = ChatPromptTemplate.from_template(template)

    # Summarize chat history into a new refined question.
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    
    # Debug: Print the rephrased question by running the rephrase prompt directly
    # if chat_history:
    #     # print("=== DEBUG: Rephrase Prompt ===")
    #     # print(f"Original query: {query}")
    #     # print(f"Chat history: {chat_history}")
        
    #     # Run the rephrase prompt directly to see the output
    #     rephrased_question = rephrase_prompt.invoke({
    #         "input": query,
    #         "chat_history": chat_history
    #     })
    #     print(f"Rephrased question: {rephrased_question.to_string()}")
    #     print("=== END DEBUG ===")
    
    history_aware_retriever = create_history_aware_retriever(llm=llm, retriever=loaded_vectorstore.as_retriever(), prompt=rephrase_prompt)

    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    # stuff_documents_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    stuff_documents_chain = create_stuff_documents_chain(llm, custom_rag_prompt)
    qa = create_retrieval_chain(retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain)

    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    # just renaming the keys to be compatible with tutorial
    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"]
    }
    return new_result


if __name__ == "__main__":
    result = run_llm("What is Satoshis stance on Libertarianism?")
    print(result["result"])
