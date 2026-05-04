import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # type: ignore[import-not-found]
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data"
DB_PATH = "vector_db"

def main():

    print("Loading documents...")
    # DirectoryLoader globs are relative to DATA_PATH
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    print(f"Loaded {len(documents)} documents")
    if not documents:
        print(f"❌ No documents found under '{DATA_PATH}'. Nothing to ingest.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")
    if not chunks:
        print("❌ No chunks created. Nothing to ingest.")
        return

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=DB_PATH
    )

    # Chroma persists automatically when persist_directory is set.
    # Avoid unicode symbols to keep Windows consoles happy.
    print("Vector DB created successfully")


if __name__ == "__main__":
    main()