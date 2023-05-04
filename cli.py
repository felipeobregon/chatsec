import pinecone

pinecone.init(environment='us-west1-gcp-free')
index = pinecone.Index('chat')

query_response = index.query(
    namespace='example-namespace',
    top_k=10,
    include_values=True,
    include_metadata=True,
    vector=[0.1, 0.2, 0.3, 0.4],
    filter={
        'genre': {'$in': ['comedy', 'documentary', 'drama']}
    }
)