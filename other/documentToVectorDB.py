import jsonlines
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 连接到 Milvus 服务器
connections.connect(alias="default", host='localhost', port='19530')

# 定义集合的字段
fields = [
    FieldSchema(name="document_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
]

# 创建集合的模式
schema = CollectionSchema(fields=fields, description="Document collection")
collection_name = "document_collection"


def split_text(text, max_length=256):
    """
    将长文本拆分为不超过 max_length 的段落。
    """
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def insert_data():
    # 如果集合存在，则删除后重新创建
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"集合 {collection_name} 已存在，删除后重新创建。")
    else:
        print(f"集合 {collection_name} 不存在，创建新的集合。")

    collection = Collection(name=collection_name, schema=schema)

    # 加载 SentenceTransformer 模型
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # 加载 JSONL 文件并插入数据
    file_path = "/Users/shione/python/nlp/project/documents.jsonl"
    batch_size = 100  # 每批插入的数据量
    batch_ids = []
    batch_embeddings = []

    # 统计文件中的总行数
    with jsonlines.open(file_path) as reader:
        total_lines = sum(1 for _ in reader)

    # 重新打开文件，读取数据并显示进度条
    with jsonlines.open(file_path) as reader, tqdm(total=total_lines, desc="Processing") as pbar:
        for i, obj in enumerate(reader):
            # 获取文档ID和文本内容；如果没有ID，则用循环计数代替
            doc_id = obj.get("id", i)
            text = obj.get("text", "")

            # 将长文本拆分为段落
            text_chunks = split_text(text)

            # 对每个段落生成嵌入
            chunk_embeddings = model.encode(text_chunks)

            # 对段落嵌入取平均值，得到文档的整体嵌入
            document_embedding = sum(chunk_embeddings) / len(chunk_embeddings)

            batch_ids.append(doc_id)
            batch_embeddings.append(document_embedding.tolist())

            # 当达到批量大小时，批量插入数据
            if (i + 1) % batch_size == 0:
                data = [
                    batch_ids,
                    batch_embeddings
                ]
                collection.insert(data)
                print(f"已插入 {i + 1} 条数据")
                batch_ids = []
                batch_embeddings = []

            # 更新进度条
            pbar.update(1)

        # 插入剩余的数据（如果有）
        if batch_ids:
            data = [
                batch_ids,
                batch_embeddings
            ]
            collection.insert(data)


if __name__ == "__main__":
    insert_data()
