from langchain.document_loaders import UnstructuredEPubLoader

# goal: ask questions to a book

'''

you need all your data as chunks of text.   

receive an EPUB and output a json file that contains
an array of chunks, include metadata

'''
# load
loader = UnstructuredEPubLoader()

book = loader.load()

# chunk
from langchain.text_splitter import CharacterTextSplitter

ts = CharacterTextSplitter(chunk_size=1000)
docs = ts.split_documents(book)

# embed

