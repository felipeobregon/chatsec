'''

you need all your data as chunks of text.   

receive an EPUB and output a json file that contains
an array of chunks, include metadata

'''
# set environment variables and api keys
import os
import getpass

if 'OPENAI_API_KEY' not in os.environ:
    os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key: ')

PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
PINECONE_ENV = os.environ['PINECONE_ENV']

#PINECONE_API_KEY = getpass.getpass('Pinecone API Key:')
#PINECONE_ENV = getpass.getpass('Pinecone Environment:')

# load
from langchain.document_loaders import UnstructuredEPubLoader
loader = UnstructuredEPubLoader()
book = loader.load()

# chunk
from langchain.text_splitter import CharacterTextSplitter
ts = CharacterTextSplitter(chunk_size=1000)
docs = ts.split_documents(book)

# embed

