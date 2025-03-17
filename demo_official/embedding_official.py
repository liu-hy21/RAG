import pandas as pd
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from pymilvus import (
    AnnSearchRequest,
    WeightedRanker,
)

file_path = "quora_duplicate_questions.tsv"
df = pd.read_csv(file_path, sep="\t")
questions = set()
for _, row in df.iterrows():
    obj = row.to_dict()
    questions.add(obj["question1"][:512])
    questions.add(obj["question2"][:512])
    if len(questions) > 500:  # Skip this if you want to use the full dataset
        break

docs = list(questions)

print(docs[0])

ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
dense_dim = ef.dim["dense"]

# # Generate embeddings using BGE-M3 model
docs_embeddings = ef(docs)

# Connect to Milvus given URI
# connections.connect(uri="./milvus.db")
connections.connect(alias="default", host='localhost', port='19530')

# Specify the data schema for the new Collection
fields = [
    # Use auto generated id as primary key
    FieldSchema(
        name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100
    ),
    # Store the original text to retrieve based on semantically distance
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
    # Milvus now supports both sparse and dense vectors,
    # we can store each in a separate field to conduct hybrid search on both vectors
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
]
schema = CollectionSchema(fields)

# Create collection (drop the old one if exists)
col_name = "hybrid_demo"
if utility.has_collection(col_name):
    Collection(col_name).drop()
col = Collection(col_name, schema, consistency_level="Strong")

# To make vector search efficient, we need to create indices for the vector fields
sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
col.create_index("sparse_vector", sparse_index)
dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
col.create_index("dense_vector", dense_index)
col.load()

for i in range(0, len(docs), 50):
    batched_entities = [
        docs[i: i + 50],
        docs_embeddings["sparse"][i: i + 50],
        docs_embeddings["dense"][i: i + 50],
    ]
    col.insert(batched_entities)
print("Number of entities inserted:", col.num_entities)
