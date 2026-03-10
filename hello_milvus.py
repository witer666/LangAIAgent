from pymilvus import MilvusClient, model
from pymilvus import Function,FunctionType

client = MilvusClient("milvus_hello.db")
# if client.has_collection(collection_name="hello_collection"):
#     client.drop_collection(collection_name="hello_collection")
# client.create_collection(
#     collection_name="hello_collection",
#     dimension=768,  # The vectors we will use in this demo has 768 dimensions
# )

embedding_fn = model.DefaultEmbeddingFunction()

docs = [
    "煤化工技术的发展与新型煤化工技术",
    "石油和化工行业智能优化制造若干问题及挑战",
    "我国现代煤化工产业发展现状及对石油化工产业的影响",
]

vectors = embedding_fn.encode_documents(docs)
# The output vector has 768 dimensions, matching the collection that we just created.
print("Dim:", embedding_fn.dim, vectors[0].shape)  # Dim: 768 (768,)

# Each entity has id, vector representation, raw text, and a subject label that we use
# to demo metadata filtering later.
data = [
    {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
    for i in range(len(vectors))
]

print("Data has", len(data), "entities, each with fields: ", data[0].keys())
print("Vector dim:", len(data[0]["vector"]))


res = client.insert(collection_name="hello_collection", data=data)

print(res)



query_vectors = embedding_fn.encode_queries(["煤化工技术的发展"])
# If you don't have the embedding function you can use a fake vector to finish the demo:
# query_vectors = [ [ random.uniform(-1, 1) for _ in range(768) ] ]

res = client.search(
    collection_name="hello_collection",  # target collection
    data=query_vectors,  # query vectors
    limit=2,  # number of returned entities
    output_fields=["text", "subject"],  # specifies fields to be returned
)

print(res)


res = client.query(
    collection_name="hello_collection",
    filter="text like '%化工%'",
    output_fields=["text", "subject"],
)
print(res)

res = client.delete(
    collection_name="demo_collection",
    filter="id == 0",
)

print(res)

bm25_function = Function(
    name="text_bm25_emb",  # Function name
    input_field_names=["text"],  # Name of the VARCHAR field containing raw text data
    output_field_names=[
        "sparse"
    ],  # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
    function_type=FunctionType.BM25,
)

schema = client.create_schema()
schema.add_function(bm25_function)