import requests
import json
import numpy as np
import gradio as gr
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import hashlib
import faiss
from rank_bm25 import BM25Okapi

# 配置参数
API_URL = "http://127.0.0.1:1234/v1/chat/completions"
# 使用 DPR 模型
EMBEDDING_MODEL = "facebook-dpr-question_encoder-multiset-base"
MILVUS_COLLECTION = "documents_collection"
TOP_K = 5  # 检索数量
SIMILARITY_THRESHOLD = 0.5  # 相似度阈值

# 初始化 Milvus 连接
connections.connect(alias="default", host='localhost', port='19530')
collection = Collection(MILVUS_COLLECTION)
collection.load()

# 初始化编码模型
encoder = SentenceTransformer(EMBEDDING_MODEL)

# 加载所有文档嵌入到 FAISS 索引中
all_embeddings = []
all_documents = []
for result in collection.query(expr="True", output_fields=["embedding", "chunk_text", "document_id", "chunk_type"]):
    embedding = result["embedding"]
    all_embeddings.append(embedding)
    all_documents.append({
        "text": result["chunk_text"],
        "doc_id": result["document_id"],
        "type": result["chunk_type"]
    })

all_embeddings = np.array(all_embeddings).astype('float32')
d = all_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(all_embeddings)

# 初始化 BM25 用于混合检索
tokenized_docs = [doc["text"].split(" ") for doc in all_documents]
bm25 = BM25Okapi(tokenized_docs)


def semantic_search(query, top_k=5):
    """语义检索函数，使用 FAISS 进行近似最近邻搜索"""
    query_embedding = encoder.encode(query).astype('float32')
    query_embedding = query_embedding.reshape(1, -1)

    distances, indices = index.search(query_embedding, top_k)

    formatted_results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        distance = distances[0][i]
        doc = all_documents[idx]

        if 1 - distance < SIMILARITY_THRESHOLD:
            continue

        formatted_results.append({
            "score": 1 - distance,
            "text": doc["text"],
            "doc_id": doc["doc_id"],
            "type": doc["type"]
        })

    return sorted(formatted_results, key=lambda x: x["score"], reverse=True)


def hybrid_search(query, top_k=5):
    """混合检索函数，结合 BM25 和 DPR"""
    # 基于词法的检索
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]  # 先筛选出两倍数量的候选文档

    # 基于向量的检索
    query_embedding = encoder.encode(query).astype('float32')
    query_embedding = query_embedding.reshape(1, -1)
    candidate_embeddings = all_embeddings[top_bm25_indices]
    candidate_index = faiss.IndexFlatL2(candidate_embeddings.shape[1])
    candidate_index.add(candidate_embeddings)
    distances, indices = candidate_index.search(query_embedding, top_k)

    formatted_results = []
    for i in range(len(indices[0])):
        idx = top_bm25_indices[indices[0][i]]
        distance = distances[0][i]
        doc = all_documents[idx]

        if 1 - distance < SIMILARITY_THRESHOLD:
            continue

        formatted_results.append({
            "score": 1 - distance,
            "text": doc["text"],
            "doc_id": doc["doc_id"],
            "type": doc["type"]
        })

    return sorted(formatted_results, key=lambda x: x["score"], reverse=True)


def build_prompt(query, context):
    """提示构建函数"""
    context_str = "\n".join([f"[doc#{doc['doc_id']}] {doc['text']}" for doc in context])
    return f"Answer with a single noun (not a sentence) in ≤5 words, English, based on:\n{context_str}\nQuestion: {query}"


def format_retrieval_results(results):
    """优化结果显示格式，只保留文档 ID 和内容片段"""
    return [[
        doc["doc_id"],
        doc["text"][:150] + "..." if len(doc["text"]) > 150 else doc["text"]
    ] for doc in results]


def rag_generation(query, history, use_hybrid=False):
    """带超时处理的增强流程"""
    try:
        # 1. 检索相关段落
        if use_hybrid:
            search_results = hybrid_search(query, TOP_K)
        else:
            search_results = semantic_search(query, TOP_K)

        # 2. 构建增强提示
        prompt = build_prompt(query, search_results[:3])  # 取前三相关

        # 3. 流式生成
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "messages": [{"role": "user", "content": prompt}],
                "model": "default",
                "temperature": 0.5,
                "stream": True
            },
            stream=True,
            timeout=15  # 缩短超时时间
        )
        response.raise_for_status()

        full_response = ""
        for chunk in response.iter_lines():
            if chunk:
                decoded = chunk.decode().lstrip('data: ').strip()
                if decoded == "[DONE]":
                    break
                try:
                    data = json.loads(decoded)
                    delta = data['choices'][0]['delta'].get('content', '')
                    full_response += delta
                    yield full_response, search_results
                except:
                    continue
    except Exception as e:
        yield f"请求失败：{str(e)}", []


# 创建增强界面
with gr.Blocks(title="智能文档助手", theme="soft") as demo:
    gr.Markdown("## 📚 智能文档问答系统（基于 Milvus 和 FAISS）")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, bubble_full_width=False)
            query_input = gr.Textbox(label="请输入问题", lines=2)
            use_hybrid_checkbox = gr.Checkbox(label="使用混合检索", value=False)

        with gr.Column(scale=2):
            retrieval_display = gr.DataFrame(
                headers=["文档 ID", "内容片段"],
                datatype=["number", "str"],
                interactive=False,
                wrap=True,
                column_widths=["20%", "80%"]
            )

    with gr.Row():
        submit_btn = gr.Button("提交查询", variant="primary")
        clear_btn = gr.Button("清空对话", variant="secondary")


    def process_query(query, history, use_hybrid):
        # 初始化状态
        yield history + [[query, "正在检索相关文档..."]], []

        try:
            # 执行 RAG 流程
            generator = rag_generation(query, history, use_hybrid)
            final_response = ""
            final_results = []

            for partial_res, results in generator:
                final_response = partial_res
                final_results = results
                yield history + [[query, final_response]], format_retrieval_results(final_results)

            # 最终更新
            new_history = history + [[query, final_response]]
            yield new_history, format_retrieval_results(final_results)

        except Exception as e:
            error_msg = f"系统错误：{str(e)}"
            yield history + [[query, error_msg]], []


    # 交互逻辑
    query_input.submit(
        process_query,
        [query_input, chatbot, use_hybrid_checkbox],
        [chatbot, retrieval_display]
    )
    submit_btn.click(
        process_query,
        [query_input, chatbot, use_hybrid_checkbox],
        [chatbot, retrieval_display]
    )
    clear_btn.click(lambda: (None, []), None, [chatbot, retrieval_display])

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
    )
