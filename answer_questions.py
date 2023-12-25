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
    for i, node in enumerate(response.source_nodes):

        title = f"\n**Source {i+1}** {node.node.metadata.get('title')}"
        page = f"**Page** {node.node.metadata.get('page_number')}"
        percentage = f"**Relevance** {node.score * 100:.1f}%"

        st.toast(f"{title}\n{page}\n{percentage}", icon="ℹ️")
        st.markdown(f"**{title}**")
        st.markdown(f"**page {page}**")
        st.markdown(f"**{percentage}% relevance**")

    return response_text # + "\n\n" + f"{title}\n{page}\n{percentage}"
