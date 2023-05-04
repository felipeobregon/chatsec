import pinecone
from openai import Embedding

# use embeddings

e = Embedding()
res = e.create(
    model = 'text-embedding-ada-002',
    input = 'Hello world'
)

print(res)

# Set up and query pinecone

pinecone.init(environment='us-west1-gcp-free')
index = pinecone.Index('chat')

#query_response = index.query(top_k=1)

#print(index.describe_index_stats())

v = [0] * 1536

q = index.query(top_k=1, vector=v)
print(q)

