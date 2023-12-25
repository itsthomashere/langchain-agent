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

def extract_info(node):
    """Extracts the content, title, and page number from a node."""
    content = node.get('text', 'No content available')
    metadata = node.get('metadata', {})
    title = metadata.get('title', 'No title')
    page_number = metadata.get('page_number', 'No page number')
    return content, title, page_number

def print_info(nodes):
    """Prints the content, title, and page number for each node in the list."""
    for index, node in enumerate(nodes):
        content, title, page_number = extract_info(node.get('node', {}))
        print(f"Item {index}:")
        print(f"Title: {title}")
        print(f"Page Number: {page_number}")
        print("Content:")
        print(content)
        print("\n")

def answer_question(query):
    """Run a query on the query engine."""
    nodes = retriever.retrieve(query)
    for index, node in enumerate(nodes):
        content, title, page_number = extract_info(node.get('node', {}))
        st.write(f"Item {index}:")
        st.write(f"Title: {title}")
        st.write(f"Page Number: {page_number}")
        st.write("Content:")
        st.write(content)
        st.write("\n") 

    response = query_engine.query(query)
    st.write(response)
    st.write(response.source_nodes[0].get_content())
    for node in response.source_nodes:
    #     print("-----")
    #     text_fmt = node.node.get_content().strip().replace("\n", " ")[:1000]
        # st.write(f"Text:\t {text_fmt} ...")
        # st.write(f"Metadata:\t {node.node.metadata}")
    #     # print out the page number and the metadata
        st.write(f"Title:\t {node.node.metadata.get('title')}")
        st.write(f"Page:\t {node.node.metadata.get('page_number')}")
        st.write(f"Score:\t {node.score:.3f}")
    # # print(response.get_formatted_sources())

    # return query_engine.query(query), reference_list
