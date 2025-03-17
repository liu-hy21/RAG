from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from sentence_transformers import SentenceTransformer

# 连接到 Milvus 服务
connections.connect(
    alias="default",
    host='localhost',
    port='19530'
)

# 定义集合的字段
fields = [
    FieldSchema(name="document_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="document_text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)  # 使用 all-MiniLM-L6-v2 模型，向量维度为 384
]

# 定义集合的模式
schema = CollectionSchema(fields=fields, description="This is a test collection")

# 定义集合名称
collection_name = "test_collection"


def insert_document_vector():
    # 创建集合
    collection = Collection(name=collection_name, schema=schema)

    # 加载文本嵌入模型
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 定义要插入的文本数据
    texts = [
        "LIU Hangyu has always had a special affection for furry little animals, especially the cute and soft rabbits. On a sunny weekend, he decided to go to a pet farm in the suburbs, where there is a special area allowing visitors to interact closely with various small animals.As soon as he entered the farm, LIU Hangyu rushed straight to the rabbit area. He saw a group of rabbits happily hopping on the grass, some snow - white, some gray, and some black - and - white. Just then, a tiny white rabbit caught his eye. This rabbit had long ears and eyes like two brown jewels. It was squatting in the corner, quietly munching on a carrot.LIU Hangyu walked over gently, crouched down, and reached out his hand to the little rabbit. Amazingly, the little rabbit wasn't scared. Instead, it hopped over to him, sniffed his hand with its nose, and then obediently lay down at his feet. LIU Hangyu felt a rush of joy in his heart. He slowly stroked the little rabbit's soft fur, and the little rabbit comfortably narrowed its eyes.In the following time, LIU Hangyu stayed with this little rabbit all the time. He went to the farm store and bought special rabbit food, feeding it to the little rabbit bit by bit. The little rabbit ate with relish and from time to time rubbed its head against LIU Hangyu's hand, as if expressing its love for him.",
    ]

    # 将文本转换为向量
    embeddings = model.encode(texts)

    # 生成 document_id
    document_ids = [1]

    # 构造要插入的数据
    data = [
        document_ids,
        texts,
        embeddings.tolist()
    ]

    # 插入数据
    insert_result = collection.insert(data)
    print(f"Inserted rows: {insert_result.insert_count}")

    # 刷新集合以确保数据持久化
    collection.flush()

    # 释放资源
    collection.release()

    # 断开与 Milvus 服务的连接
    connections.disconnect("default")


def get_document_vector():
    # 获取集合对象
    collection = Collection(name=collection_name)

    # 加载集合到内存（如果集合数据量较大，加载可能需要一些时间）
    collection.load()

    # 定义查询参数
    expr = "document_id > 0"  # 这里可以根据实际需求修改查询表达式
    output_fields = ["document_id", "document_text", "embedding"]  # 定义要返回的字段

    # 进行查询
    results = collection.query(
        expr=expr,
        output_fields=output_fields
    )

    # 释放集合内存
    collection.release()

    return results


if __name__ == '__main__':
    # 插入文档向量
    insert_document_vector()
    # get_document_vector()
