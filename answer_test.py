import json
import requests
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 复用原始配置
API_URL = "http://127.0.0.1:1234/v1/chat/completions"
EMBEDDING_MODEL = "paraphrase-MiniLM-L6-v2"
MILVUS_COLLECTION = "documents_collection"
TOP_K = 5
SIMILARITY_THRESHOLD = 0.5

# 初始化Milvus连接
connections.connect(alias="default", host='localhost', port='19530')
collection = Collection(MILVUS_COLLECTION)
collection.load()

# 初始化编码模型
encoder = SentenceTransformer(EMBEDDING_MODEL)


def semantic_search(query):
    """调整后的语义搜索函数"""
    query_embedding = encoder.encode(query).tolist()

    search_params = {"metric_type": "L2", "params": {"ef": 64}}

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=TOP_K,
        output_fields=["chunk_text", "document_id"]
    )

    formatted = []
    for hits in results:
        for hit in hits:
            if hit.distance < SIMILARITY_THRESHOLD:
                continue
            formatted.append({
                "doc_id": hit.entity.document_id,
                "text": hit.entity.chunk_text,
                "score": 1 - hit.distance
            })
    return sorted(formatted, key=lambda x: x["score"], reverse=True)


def build_prompt(query, context):
    """提示构建函数"""
    context_str = "\n".join([f"[doc#{doc['doc_id']}] {doc['text']}" for doc in context])
    return f"Answer with a single noun (not a sentence) in ≤5 words, English, based on:\n{context_str}\nQuestion: {query}"


def generate_answer(prompt):
    """同步生成回答"""
    try:
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "messages": [{"role": "user", "content": prompt}],
                "model": "default",
                "temperature": 0.5,
                "stream": False
            },
            timeout=20
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"


def process_question(question):
    """处理单个问题"""
    try:
        # 语义检索
        results = semantic_search(question)
        if not results:
            return "No relevant documents found", []

        # 构建提示（取前3个结果）
        prompt = build_prompt(question, results[:3])

        # 生成回答
        answer = generate_answer(prompt)

        # 提取文档ID列表
        doc_ids = [str(res["doc_id"]) for res in results]  # 转换为字符串避免JSON序列化问题

        return answer, doc_ids
    except Exception as e:
        return f"Processing Error: {str(e)}", []


if __name__ == '__main__':
    # 文件路径配置
    input_file = "/Users/shione/python/nlp/project/jsonl/val.jsonl"
    output_file = "/Users/shione/python/nlp/project/jsonl/val_predict1.jsonl"

    # 统计输入文件的行数
    with open(input_file, 'r') as fin:
        total_lines = sum(1 for _ in fin)

    i = 0
    # 批量处理
    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        # 使用tqdm包装迭代器
        for line in tqdm(fin, total=total_lines, desc="Processing questions"):
            # 解析输入数据
            data = json.loads(line.strip())
            question = data["question"]

            # 处理问题
            answer, doc_ids = process_question(question)

            # 构建输出格式
            output_data = {
                "question": question,
                "answer": answer,
                "document_id": doc_ids,
            }

            # 写入结果
            fout.write(json.dumps(output_data, ensure_ascii=False) + "\n")
            i += 1
            if i == 10:
                break

    print(f"处理完成！结果已保存至 {output_file}")
