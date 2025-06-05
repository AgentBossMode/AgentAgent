from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
embedding = AzureOpenAIEmbeddings(
openai_api_type="azure",
openai_api_version=os.environ["OPENAI_API_EMBEDDING_VERSION"],
openai_api_key=os.environ["OPENAI_API_EMBEDDING_KEY"],
azure_endpoint=os.environ["AZURE_OPENAI_EMBEDDING_ENDPOINT"],
deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
model=os.environ["AZURE_OPENAI_EMBEDDING_MODEL"],
validate_base_url=True,
)

def get_embeddings():
    return embedding

def get_vector_store(embeddings: AzureOpenAIEmbeddings, index_name: str):
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    vector_store = PineconeVectorStore(index=pc.Index(index_name), embedding=embeddings)
    return vector_store