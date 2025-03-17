import gradio as gr
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


def dense_search(col, query_dense_embedding, limit=5):  # 修改 limit 默认值为 5
    search_params = {"metric_type": "IP", "params": {}}
    res = col.search(
        [query_dense_embedding],
        anns_field="dense_vector",
        limit=limit,
        output_fields=["text"],
        param=search_params,
    )[0]
    return [hit.get("text") for hit in res]


def sparse_search(col, query_sparse_embedding, limit=5):  # 修改 limit 默认值为 5
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
        limit=5  # 修改 limit 默认值为 5
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


def process_query(query, history):
    # 初始化状态
    yield history + [[query, "正在检索相关文档..."]], []

    try:
        ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
        dense_dim = ef.dim["dense"]

        # Generate embeddings for the query
        query_embeddings = ef([query])

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

        final_response = "\n".join(hybrid_results)
        formatted_results = doc_text_formatting(hybrid_results)
        retrieval_results = [[i + 1, result] for i, result in enumerate(formatted_results)]

        new_history = history + [[query, final_response]]
        yield new_history, retrieval_results

    except Exception as e:
        error_msg = f"系统错误：{str(e)}"
        yield history + [[query, error_msg]], []


with gr.Blocks(title="智能文档助手", theme="soft") as demo:
    gr.Markdown("## 📚 智能文档问答系统（基于Milvus）")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, bubble_full_width=False)
            query_input = gr.Textbox(label="请输入问题", lines=2)

        with gr.Column(scale=2):
            retrieval_display = gr.DataFrame(
                headers=["文档ID", "内容片段"],
                datatype=["number", "str"],
                interactive=False,
                wrap=True,
                column_widths=["20%", "80%"]
            )

    with gr.Row():
        submit_btn = gr.Button("提交查询", variant="primary")
        clear_btn = gr.Button("清空对话", variant="secondary")

    # 交互逻辑
    query_input.submit(
        process_query,
        [query_input, chatbot],
        [chatbot, retrieval_display]
    )
    submit_btn.click(
        process_query,
        [query_input, chatbot],
        [chatbot, retrieval_display]
    )
    clear_btn.click(lambda: (None, []), None, [chatbot, retrieval_display])

if __name__ == "__main__":
    demo.launch()
