import requests
import json
import numpy as np
import gradio as gr
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import hashlib

# 配置参数
API_URL = "http://127.0.0.1:1234/v1/chat/completions"
EMBEDDING_MODEL = "paraphrase-MiniLM-L6-v2"
# MILVUS_COLLECTION = "doc_chunks_collection"
MILVUS_COLLECTION = "documents_collection"
TOP_K = 5  # 检索数量
SIMILARITY_THRESHOLD = 0.5  # 相似度阈值

# 初始化Milvus连接
connections.connect(alias="default", host='localhost', port='19530')
collection = Collection(MILVUS_COLLECTION)
collection.load()

# 初始化编码模型
encoder = SentenceTransformer(EMBEDDING_MODEL)


def semantic_search(query, top_k=5):
    """语义检索函数"""
    query_embedding = encoder.encode(query).tolist()

    search_params = {
        "metric_type": "L2",
        "params": {"ef": 64}
    }

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["chunk_text", "document_id"]
    )

    formatted_results = []
    for hits in results:
        for hit in hits:
            # 获取字段的正确方式
            chunk_text = hit.entity.chunk_text
            doc_id = hit.entity.document_id
            chunk_type = hit.entity.chunk_type

            if hit.distance < SIMILARITY_THRESHOLD:
                continue

            formatted_results.append({
                "score": 1 - hit.distance,
                "text": chunk_text,
                "doc_id": doc_id,
                "type": chunk_type
            })

    return sorted(formatted_results, key=lambda x: x["score"], reverse=True)


def build_prompt(query, context):
    """提示构建函数"""
    context_str = "\n".join([f"[doc#{doc['doc_id']}] {doc['text']}" for doc in context])
    return f"Answer with a single noun (not a sentence) in ≤5 words, English, based on:\n{context_str}\nQuestion: {query}"


def format_retrieval_results(results):
    """优化结果显示格式，只保留文档ID和内容片段"""
    return [[
        doc["doc_id"],
        doc["text"][:150] + "..." if len(doc["text"]) > 150 else doc["text"]
    ] for doc in results]


def rag_generation(query, history):
    """带超时处理的增强流程"""
    try:
        # 1. 检索相关段落
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


    def process_query(query, history):
        # 初始化状态
        yield history + [[query, "正在检索相关文档..."]], []

        try:
            # 执行RAG流程
            generator = rag_generation(query, history)
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
    demo.launch(
        server_name="127.0.0.1",
    )
