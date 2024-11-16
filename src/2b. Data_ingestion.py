import warnings
warnings.filterwarnings("ignore")


import chromadb
import torch
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext
from llama_index.core import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import ServiceContext
from llama_index.core import Settings

fpath = 'data'
model_path = 'HuggingFaceH4/zephyr-7b-alpha'
embed_model_path = 'BAAI/bge-small-en-v1.5'
db_folder = "./chroma_db"
db_name = 'DB_01'


def load_docs(fpath):

    loader = SimpleDirectoryReader(
        input_dir=fpath,
        recursive=True
    )

    documents = loader.load_data()

    # exclude some metadata from the LLM
    for doc in documents:
        doc.excluded_llm_metadata_keys = ["File Name", "Content Type", "Header Path"]

    return documents 

documents = load_docs(fpath)
splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)
nodes = splitter.get_nodes_from_documents(documents)

# initialize client, setting path to save data
db = chromadb.PersistentClient(path=db_folder)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# create collection
chroma_collection = db.get_or_create_collection(db_name)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
    nodes, storage_context=storage_context, embed_model=embed_model, show_progress=True
)
