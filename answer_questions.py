from llama_index import StorageContext, load_index_from_storage

# load knowledge base from disk.
index = load_index_from_storage(
    StorageContext.from_defaults(persist_dir="storage"))

# make the knowledge base into a query engine—an object that queries can be run on
query_engine = index.as_query_engine()

def answer_question(query):
    """Run a query on the query engine."""
    return query_engine.query(query)
