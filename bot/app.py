from llama_index.core import SimpleDirectoryReader   #To read the files from the directory
from llama_index.core.node_parser import SimpleNodeParser   #To split data into chunks
from llama_index.core import ServiceContext,StorageContext   #service_context for emdedding model with LLM ,storage_context is Vectore store
from llama_index.core import VectorStoreIndex    #indexing
from llama_index.llms.huggingface import (
    HuggingFaceInferenceAPI
)    # We are using inference api coz we are using cpu machine 
from llama_index.embeddings.langchain import LangchainEmbedding    
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from llama_index.core import load_index_from_storage

#Using ChromaDB
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

import os
from huggingface_hub import login
from dotenv import load_dotenv

# create client and a new collection
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")

def initialize_llm_model():
    # Load environment variables
    HF_TOKEN = os.getenv('hugging_face_token')
    # Login to Hugging Face
    login(token=HF_TOKEN)
    # Initialize LLM model
    llm = HuggingFaceInferenceAPI(
        model_name="HuggingFaceH4/zephyr-7b-alpha",
        api_key=HF_TOKEN
    )
    return llm

def initialize_embedding_model():
    # Load environment variables
    HF_TOKEN = os.getenv('hugging_face_token')
    # Initialize embedding model
    embed_model = LangchainEmbedding(
        HuggingFaceInferenceAPIEmbeddings(
            api_key=HF_TOKEN,
            model_name="thenlper/gte-large"
        )
    )
    return embed_model

def create_index():
    PERSIS_DIR = "./chroma_db"
    if not os.path.exists(PERSIS_DIR):
        #create
        document = SimpleDirectoryReader("data").load_data()
        parser = SimpleNodeParser()
        nodes = parser.get_nodes_from_documents(document)
        db = chromadb.PersistentClient(path=PERSIS_DIR)

        chroma_collection = db.get_or_create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        service_context = ServiceContext.from_defaults(llm=initialize_llm_model(), embed_model=initialize_embedding_model(), chunk_size=512)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes, service_context=service_context, storage_context=storage_context)
        index.storage_context.persist(persist_dir=PERSIS_DIR)
    else:
        db2 = chromadb.PersistentClient(path=PERSIS_DIR)
        chroma_collection = db2.get_or_create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        service_context = ServiceContext.from_defaults(llm=initialize_llm_model(), embed_model=initialize_embedding_model(), chunk_size=512)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            service_context=service_context
        )
    return index

def query_index(query):
    # Create or load the index
    index = create_index()
    # Initialize query engine
    query_engine = index.as_query_engine()
    # Query the index
    response = query_engine.query(query)
    return response

# Example usage:
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    # Query the index with a specific query
    response = query_index("define coil span")
    # Print the response
    print(response)