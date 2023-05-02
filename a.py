# load document
from langchain.document_loaders import UnstructuredEPubLoader

epub = UnstructuredEPubLoader('book.epub')

book = epub.load()


# chunk document

from langchain.text_splitter import CharacterTextSplitter

ts = CharacterTextSplitter(
    chunk_size=1000
)

docs = ts.split_documents(book)

# store chunks as embeddings

from langchain.vectorstores import Chroma

db = Chroma.from_documents(docs)

# perform similarity search on chunks

res = db.similarity_search('where was jeff bezos born?')

print(res[0])

# create a stuffing chain

