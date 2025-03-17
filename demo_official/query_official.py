from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from pymilvus import (
    AnnSearchRequest,
    WeightedRanker,
)


def dense_search(col, query_dense_embedding, limit=10):
    search_params = {"metric_type": "IP", "params": {}}
    res = col.search(
        [query_dense_embedding],
        anns_field="dense_vector",
        limit=limit,
        output_fields=["text"],
        param=search_params,
    )[0]
    return [hit.get("text") for hit in res]


def sparse_search(col, query_sparse_embedding, limit=10):
    search_params = {
        "metric_type": "IP",
        "params": {},
    }
    res = col.search(
        [query_sparse_embedding],
        anns_field="sparse_vector",
        limit=limit,
        output_fields=["text"],
        param=search_params,
    )[0]
    return [hit.get("text") for hit in res]


def hybrid_search(
        col,
        query_dense_embedding,
        query_sparse_embedding,
        sparse_weight=1.0,
        dense_weight=1.0,
        limit=10,
):
    dense_search_params = {"metric_type": "IP", "params": {}}
    dense_req = AnnSearchRequest(
        [query_dense_embedding], "dense_vector", dense_search_params, limit=limit
    )
    sparse_search_params = {"metric_type": "IP", "params": {}}
    sparse_req = AnnSearchRequest(
        [query_sparse_embedding], "sparse_vector", sparse_search_params, limit=limit
    )
    rerank = WeightedRanker(sparse_weight, dense_weight)
    res = col.hybrid_search(
        [sparse_req, dense_req], rerank=rerank, limit=limit, output_fields=["text"]
    )[0]
    return [hit.get("text") for hit in res]


def doc_text_formatting(docs):
    formatted_texts = []

    for doc in docs:
        # 直接将原始文档添加到结果列表中
        formatted_texts.append(doc)

    return formatted_texts


if __name__ == '__main__':
    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
    dense_dim = ef.dim["dense"]

    # Enter your search query
    query = input("Enter your search query: ")
    print(query)

    # Generate embeddings for the query
    query_embeddings = ef([query])
    # print(query_embeddings)

    # Specify the data schema for the new Collection
    connections.connect(alias="default", host='localhost', port='19530')
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
    col_name = "hybrid_demo"
    col = Collection(col_name, schema, consistency_level="Strong")

    dense_results = dense_search(col, query_embeddings["dense"][0])
    sparse_results = sparse_search(col, query_embeddings["sparse"]._getrow(0))
    hybrid_results = hybrid_search(
        col,
        query_embeddings["dense"][0],
        query_embeddings["sparse"]._getrow(0),
        sparse_weight=0.7,
        dense_weight=1.0,
    )

    # Hybrid search results
    print("\n**Hybrid Search Results:**")
    formatted_results = doc_text_formatting(hybrid_results)
    for result in formatted_results:
        print(result)
