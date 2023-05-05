from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import os


# This is a CLI for sending queries to an index
# Documents that are similar to the query are retrieved

pinecone.init(api_key=os.environ['PINECONE_API_KEY'], environment=os.environ['PINECONE_ENV'])


index_name = input('Index name: ')
query = input('Query: ')

index = pinecone.Index(index_name)
embeddings = OpenAIEmbeddings()

# Creates an object representing the vector store
vs = Pinecone(index, embeddings.embed_query, "text")

res = vs.similarity_search(query)

print(res[0].page_content)