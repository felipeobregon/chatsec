
# set environment variables and api keys
import os
import getpass

if 'OPENAI_API_KEY' not in os.environ:
    os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key: ')

PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
PINECONE_ENV = os.environ['PINECONE_ENV']


# load
from langchain.document_loaders import UnstructuredEPubLoader
loader = UnstructuredEPubLoader('book.epub')
book = loader.load()

# chunk
from langchain.text_splitter import CharacterTextSplitter
ts = CharacterTextSplitter(chunk_size=1000)
docs = ts.split_documents(book)

# embed
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

embeddings = OpenAIEmbeddings()

import pinecone

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

index_name = "chat"

docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
