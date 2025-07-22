from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(transcript):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    return chunks
