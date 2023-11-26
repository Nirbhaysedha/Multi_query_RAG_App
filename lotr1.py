
import os
import chromadb
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings  #bge model used for embedding
from langchain.document_transformers import (
    EmbeddingsRedundantFilter,
    EmbeddingsClusteringFilter,
)
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter






model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print("Embedding Model Loaded..........")













loader_mac=PyPDFLoader("RAGGGG/How-to-implement-a-better-RAG/data/q4fy19-financial-tables.pdf")
documents_mac=loader_mac.load()
text_splitter_mac=RecursiveCharacterTextSplitter(chunk_size=1000,
                                                 chunk_overlap=100)
texts_mac=text_splitter_mac.split_documents(documents_mac)


loader_guide=PyPDFLoader("RAGGGG/How-to-implement-a-better-RAG/data/Lecture-Notes-Mid1.pdf")
documents_guide=loader_guide.load()
text_splitter_guide=RecursiveCharacterTextSplitter(chunk_size=1000,
                                                 chunk_overlap=100)
texts_guide=text_splitter_guide.split_documents(documents_guide)






mac_store=Chroma.from_documents(texts_mac,hf,collection_metadata={"hnsw:space":"cosine"},persist_directory="store/mac_cosine")


guide_store=Chroma.from_documents(texts_guide,hf,collection_metadata={"hnsw:space":"cosine"},persist_directory="store/guide_cosine")





#loading vector store

load_mac_store = Chroma(persist_directory="store/mac_cosine", embedding_function=hf)


load_guide_store = Chroma(persist_directory="store/guide_cosine", embedding_function=hf)









# merge pdf and perform semantic search
retriever_mac= load_mac_store.as_retriever(search_type = "similarity", search_kwargs = {"k":3, "include_metadata": True})

retriever_guide = load_guide_store.as_retriever(search_type = "similarity", search_kwargs = {"k":3, "include_metadata": True}
                                            )
# k as 3 retrieving 3 documents



lotr = MergerRetriever(retrievers=[retriever_mac, retriever_guide])






queru=" Explain Chapter 1 -- An Overview of Financial Management "
docs=lotr.get_relevant_documents(queru)


from dotenv import load_dotenv
load_dotenv()
import os
from langchain.llms import GooglePalm 
llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)





template =f"Answer this question {queru} using this information={docs} , make sure you answer it efficiently and make the answer concise !"

print(llm.predict(template))

