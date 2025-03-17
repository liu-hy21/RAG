from pymilvus import connections, Collection

# 配置参数
MILVUS_HOST = 'localhost'
MILVUS_PORT = '19530'
MILVUS_COLLECTION = 'documents_collection'

# 连接到 Milvus
try:
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    print("成功连接到 Milvus 数据库。")
except Exception as e:
    print(f"连接 Milvus 数据库失败: {e}")
    exit(1)

# 加载集合
try:
    collection = Collection(MILVUS_COLLECTION)
    collection.load()
    print(f"成功加载集合 {MILVUS_COLLECTION}。")
except Exception as e:
    print(f"加载集合 {MILVUS_COLLECTION} 失败: {e}")
    connections.disconnect("default")
    exit(1)

# 执行一个简单的查询
try:
    print("开始执行查询...")
    # 假设集合中有一个字段名为 'document_id'，我们查询 document_id 为 1 的记录
    query_expr = "document_id == 1"
    output_fields = ["chunk_text", "document_id"]
    results = collection.query(expr=query_expr, output_fields=output_fields)

    if results:
        print("查询成功，以下是查询结果：")
        for result in results:
            print(f"文档 ID: {result['document_id']}, 内容片段: {result['chunk_text']}")
    else:
        print("查询结果为空。")
except Exception as e:
    print(f"查询失败: {e}")

# 释放集合并断开连接
collection.release()
connections.disconnect("default")
