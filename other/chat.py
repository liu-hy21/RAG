import requests
import json
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatIP
from vectorDB import get_document_vector

# 配置参数
API_URL = "http://127.0.0.1:1234/v1/chat/completions"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 多语言小型模型
VECTOR_DIM = 384  # 与模型维度匹配
TOP_K = 3  # 检索数量

# 示例向量数据库（实际应从数据库加载）
documents = get_document_vector()

# 初始化模型和索引
encoder = SentenceTransformer(EMBEDDING_MODEL)
index = IndexFlatIP(VECTOR_DIM)
doc_embeddings = np.stack([doc["embedding"] for doc in documents])
index.add(doc_embeddings)


def semantic_search(query, top_k=3):
    """语义向量检索"""
    # 生成查询向量
    query_embedding = encoder.encode(query, convert_to_tensor=True).cpu().numpy()
    query_embedding = query_embedding.astype('float32').reshape(1, -1)

    # 相似度搜索
    distances, indices = index.search(query_embedding, top_k)

    # 组合检索结果
    results = []
    for idx, score in zip(indices[0], distances[0]):
        if idx >= 0:  # FAISS可能返回-1
            doc = documents[idx]
            results.append({
                "score": float(score),
                "document_text": doc["document_text"],
                "document_id": doc["document_id"]
            })
    return sorted(results, key=lambda x: x["score"], reverse=True)


def build_prompt(query, context):
    """构建增强提示"""
    context_str = "\n".join([f"[参考文档 {i + 1}] {doc['document_text']}"
                             for i, doc in enumerate(context)])
    return f"""基于以下参考文档回答问题：
{context_str}

问题：{query}
答案："""


def rag_generation(query, history):
    """RAG流程处理"""
    # 1. 检索相关文档
    search_results = semantic_search(query, TOP_K)

    # 2. 构建提示
    context = [res for res in search_results if res["score"] > 0.5]  # 相似度阈值
    prompt = build_prompt(query, context)

    # 3. 流式生成
    try:
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "messages": [{"role": "user", "content": prompt}],
                "model": "default",
                "temperature": 0.7,
                "stream": True
            },
            stream=True,
            timeout=30
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
                    yield full_response
                except:
                    continue
    except Exception as e:
        yield f"生成失败：{str(e)}"


def format_retrieval_results(results):
    """格式化检索结果用于表格显示（去除相似度，加长片段）"""
    return [[
        doc["document_id"],
        doc["document_text"][:250] + "..." if len(doc["document_text"]) > 250 else doc["document_text"]
    ] for doc in results]


# 创建增强版界面
with gr.Blocks(title="RAG对话系统", theme="soft") as demo:
    gr.Markdown("## 🧠 智能问答系统（检索增强版）")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(label="输入问题")
        # 修改界面部分
        with gr.Column(scale=1):
            retrieval_output = gr.DataFrame(
                label="📚 相关文档",
                headers=["文档ID", "内容片段"],
                datatype=["number", "str"],
                interactive=False,
                wrap=True,
                height=400,
                column_widths=["20%", "80%"]  # 调整列宽比例
            )

    with gr.Row():
        submit_btn = gr.Button("提交", variant="primary")
        clear_btn = gr.Button("清空对话")


    def respond(message, chat_history):
        # 显示加载状态
        yield chat_history + [[message, ""]], [["检索中...", "", ""]]

        # 执行RAG流程
        search_results = semantic_search(message, TOP_K)
        context = [res for res in search_results if res["score"] > 0.5]
        formatted_results = format_retrieval_results(context)

        generator = rag_generation(message, chat_history)
        response = ""
        for partial_res in generator:
            response = partial_res
            yield chat_history + [[message, response]], formatted_results

        # 最终更新
        new_history = chat_history + [[message, response]]
        yield new_history, formatted_results


    # 交互逻辑
    msg.submit(
        respond,
        [msg, chatbot],
        [chatbot, retrieval_output],
        show_progress="hidden"
    )
    submit_btn.click(
        respond,
        [msg, chatbot],
        [chatbot, retrieval_output],
        show_progress="hidden"
    )
    clear_btn.click(lambda: None, None, chatbot, queue=False)

demo.launch(server_name="127.0.0.1", share=False)
