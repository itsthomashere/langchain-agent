import streamlit as st
from llama_index import StorageContext, load_index_from_storage
from llama_index.retrievers import VectorIndexRetriever


# load knowledge base from disk.
try:
    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="storage"))
except ValueError:
    st.info("Please refresh the page.", icon="♻️")
    st.stop()

# make the knowledge base into a query engine—an object that queries can be run on
query_engine = index.as_query_engine()

# Configure retriever
retriever = VectorIndexRetriever(
index=index,
similarity_top_k=5  # Modify this value to change top K retrievals
)

def answer_question(query):
    """Run a query on the query engine."""
    retrieved_nodes = retriever.retrieve(query)
    response = query_engine.query(query)
    response_text = response.response
    reference_list = []
    for node in response.source_nodes:
        title = node.node.metadata.get('title')
        page = node.node.metadata.get('page_number')
        percentage = f"{node.score * 100:.1f}%"
        reference_list.append(f"**{title}**, page {page} at {percentage} relevance.")
        # reference_list.append(f"{title}\n{page}\n{percentage}")

        st.toast(f"{title}\n{page}\n{percentage}", icon="ℹ️")


    return response_text + "\n\n" + "\n".join(reference_list)
