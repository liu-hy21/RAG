import jsonlines
import hashlib
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re
from nltk.tokenize import sent_tokenize

# 连接到 Milvus
connections.connect(alias="default", host='localhost', port='19530')

# 修改后的字段定义
fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="document_id", dtype=DataType.INT64),
    FieldSchema(name="chunk_hash", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=20),
    FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=60000),  # 扩大长度
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
]

schema = CollectionSchema(
    fields=fields,
    description="Optimized document chunks collection",
    enable_dynamic_field=True
)
collection_name = "documents_collection"


def semantic_chunking(text, max_words=256, max_chars=2500):
    """增强版分块函数"""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_word_count = 0
    current_char_count = 0

    for sentence in sentences:
        words = sentence.split()
        sentence_word_count = len(words)
        sentence_char_count = len(sentence)

        # 双重检查机制
        if (current_word_count + sentence_word_count > max_words) or \
                (current_char_count + sentence_char_count > max_chars):

            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_word_count = 0
                current_char_count = 0

        current_chunk.append(sentence)
        current_word_count += sentence_word_count
        current_char_count += sentence_char_count

        # 强制分割保护
        if current_char_count >= max_chars:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_word_count = 0
            current_char_count = 0

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def validate_chunk(chunk_text):
    """数据校验函数"""
    MAX_LENGTH = 60000

    if len(chunk_text) > MAX_LENGTH:
        truncated = chunk_text[:MAX_LENGTH - 3] + '...'
        print(f"截断超长文本块: {len(chunk_text)} -> {MAX_LENGTH}")
        return truncated

    if not chunk_text.strip():
        return None

    return chunk_text


def insert_optimized_data():
    # 删除旧集合
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"集合 {collection_name} 已重置")

    # 创建新集合
    collection = Collection(name=collection_name, schema=schema)

    # 创建索引
    index_params = {
        "metric_type": "L2",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200}
    }
    collection.create_index("embedding", index_params)

    # 加载模型
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')

    # 批量处理参数
    file_path = "/Users/shione/python/nlp/project/documents_markdown_clean.jsonl"
    batch_size = 200
    batch_data = []

    with jsonlines.open(file_path) as reader:
        total_docs = sum(1 for _ in reader)

    with jsonlines.open(file_path) as reader, \
            tqdm(total=total_docs, desc="Processing") as pbar:

        for obj in reader:
            doc_id = obj.get("document_id")
            text = obj['document_text']

            # 语义分块
            chunks = semantic_chunking(text)

            # 生成嵌入
            chunk_embeddings = model.encode(chunks)

            # 构建批量数据
            for chunk, emb in zip(chunks, chunk_embeddings):
                validated = validate_chunk(chunk)
                if not validated:
                    continue

                chunk_hash = hashlib.sha256(validated.encode()).hexdigest()

                batch_data.append({
                    "document_id": doc_id,
                    "chunk_hash": chunk_hash,
                    "chunk_type": "paragraph",  # 可根据需要添加类型检测
                    "chunk_text": validated,
                    "embedding": emb.tolist()
                })

            # 批量插入（带校验）
            if len(batch_data) >= batch_size:
                collection.insert(batch_data)
                batch_data = []

            pbar.update(1)

        # 插入剩余数据
        if batch_data:
            collection.insert(batch_data)

    # 数据持久化
    collection.flush()
    print(f"总共插入 {collection.num_entities} 个文本块")


if __name__ == "__main__":
    insert_optimized_data()
