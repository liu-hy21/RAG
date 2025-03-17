from markdownify import markdownify as md
import jsonlines
from tqdm import tqdm


def html_to_md(input_file, output_file):
    # 统计文件中的总行数
    with jsonlines.open(input_file) as reader:
        total_lines = sum(1 for _ in reader)

    # 读取、转换并写入新的 JSONL 文件
    with jsonlines.open(input_file) as reader, jsonlines.open(output_file, mode='w') as writer:
        for obj in tqdm(reader, total=total_lines, desc="Processing"):
            # 将 HTML 转换为 Markdown
            markdown_text = md(obj['document_text'])
            # 创建新的字典，替换 document_text
            new_obj = {
                'document_id': obj['document_id'],
                'document_text': markdown_text
            }
            # 写入新的 JSONL 文件
            writer.write(new_obj)


if __name__ == '__main__':
    input_path = "/Users/shione/python/nlp/project/documents.jsonl"
    output_path = "/Users/shione/python/nlp/project/documents_markdown.jsonl"
    html_to_md(input_path, output_path)
