from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from datetime import datetime
from config.settings import Settings
from utils.logging import setup_logging

logger = setup_logging()

class ChromaService:
    def __init__(self, settings: Settings):
        embeddings = OllamaEmbeddings(model=settings.llm_model)
        self.vectorstore = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=embeddings,
            persist_directory=settings.chroma_persist_dir
        )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    def store_conversation(self, user_input: str, response: str):
        try:
            docs = [
                Document(
                    page_content=user_input,
                    metadata={"type": "user_input", "timestamp": datetime.now().isoformat()}
                ),
                Document(
                    page_content=response,
                    metadata={"type": "assistant_response", "timestamp": datetime.now().isoformat()}
                )
            ]
            split_docs = self.text_splitter.split_documents(docs)
            self.vectorstore.add_documents(split_docs)
            self.vectorstore.persist()
            logger.info("Stored conversation in ChromaDB")
        except Exception as e:
            logger.error(f"Error storing conversation: {str(e)}")

    def retrieve_context(self, query: str, num_docs: int = 5) -> str:
        try:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": num_docs})
            docs = retriever.get_relevant_documents(query)
            return "\n".join([doc.page_content for doc in docs]) if docs else "No context found"
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return ""