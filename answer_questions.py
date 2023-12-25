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
    # st.write(retrieved_nodes)
    response = query_engine.query(query)
    st.write(response)
    response_text = response.get_response().response
    st.write(response_text)
    # st.write(response.source_nodes[0].get_content())
    for i, node in enumerate(response.source_nodes):
    #     print("-----")
    #     text_fmt = node.node.get_content().strip().replace("\n", " ")[:1000]
        # st.write(f"Text:\t {text_fmt} ...")
        # st.write(f"Metadata:\t {node.node.metadata}")
    #     # print out the page number and the metadata
        # st.info('This is a purely informational message', icon="ℹ️")

        title = f"\n**Source {i+1}** {node.node.metadata.get('title')}"
        page = f"**Page** {node.node.metadata.get('page_number')}"
        percentage = f"**Relevance** {node.score * 100:.1f}%"

        st.toast(f"{title}\n{page}\n{percentage}", icon="ℹ️")
    # # print(response.get_formatted_sources())

    return response_text
