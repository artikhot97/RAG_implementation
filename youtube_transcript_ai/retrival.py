
def data_retriever(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return retriever