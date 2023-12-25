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
    # st.write(retrieved_nodes)
    response = query_engine.query(query)
    # st.write(response)
    st.write(response.get_response().response)
    # st.write(response.source_nodes[0].get_content())
    for node in response.source_nodes:
        title = node.node.metadata.get('title')
        page = node.node.metadata.get('page_number')
        score = node.score
        st.write(f"`{title}, page {page}, relevance {score*100:.2f}%`")
    #     print("-----")
    #     text_fmt = node.node.get_content().strip().replace("\n", " ")[:1000]
        # st.write(f"Text:\t {text_fmt} ...")
        # st.write(f"Metadata:\t {node.node.metadata}")
    #     # print out the page number and the metadata
        # st.write(f"{node.node.metadata.get('title')}", end=", ")
        # st.write(f"page:\t {node.node.metadata.get('page_number')}", end=", ")
        # st.write(f"Relevance:\t {node.score*100:.2f}%")
        # st.write(f"score:\t {node.score:.3f}")
        # score modified to percentage
    # # print(response.get_formatted_sources())

    # return query_engine.query(query), reference_list
