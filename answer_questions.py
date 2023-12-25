import streamlit as st
from llama_index import StorageContext, load_index_from_storage
from llama_index.retrievers import VectorIndexRetriever


# load knowledge base from disk.
index = load_index_from_storage(
    StorageContext.from_defaults(persist_dir="storage"))

# make the knowledge base into a query engine—an object that queries can be run on
query_engine = index.as_query_engine(streaming=True)

# Configure retriever
retriever = VectorIndexRetriever(
index=index,
similarity_top_k=5  # Modify this value to change top K retrievals
)

def answer_question(query):
    """Run a query on the query engine."""
    retrieved_nodes = retriever.retrieve(query)
    sources = [node.metadata.get('document_id') for node in retrieved_nodes]
    response = query_engine.query(query)
    st.write(response.)
        st.write(f"Title:\t {node.node.metadata.get('title')}")
        st.write(f"Page:\t {node.node.metadata.get('page_number')}")
        st.write(f"Score:\t {node.score:.3f}")

    # for node in response.source_nodes:
    #     print("-----")
    #     text_fmt = node.node.get_content().strip().replace("\n", " ")[:1000]
    #     # print(f"Text:\t {text_fmt} ...")
    #     # print(f"Metadata:\t {node.node.metadata}")
    #     # print out the page number and the metadata
    #     st.write(f"Title:\t {node.node.metadata.get('title')}")
    #     st.write(f"Page:\t {node.node.metadata.get('page_number')}")
    #     st.write(f"Score:\t {node.score:.3f}")
    # # print(response.get_formatted_sources())

    return query_engine.query(query)
