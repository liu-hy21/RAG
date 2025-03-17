import jsonlines
from tqdm import tqdm
import re


def clean_markdown(text):
    """
    清洗维基百科Markdown文本的核心函数
    返回保留关键内容且适合embedding的干净文本
    """
    # 预处理：合并换行符为统一格式
    text = re.sub(r'\r\n', '\n', text)

    # 第一阶段：移除维护模板和警告框
    patterns_to_remove = [
        # 移除维护模板（包含"template message"的区块）
        r'\n\|.*?template message.*?\n.*?\n',
        # 删除所有{{...}}模板（贪婪匹配）
        r'\{\{.*?\}\}',
        # 删除HTML注释
        r'<!--.*?-->',
        # 移除导航跳转栏
        r'Jump to:.*?\n',
        # 删除目录结构
        r'Contents\n-+\n\(hide\).*?(?=\n\n)',
        # 移除参考文献跳转标记
        r'\n\d+\. Jump up \^.*',
        # 删除表格结构（保留单元格内容）
        r'(\|.*?\n)+',
        # 移除编辑时间戳
        r'This page was last edited on.*?UTC',
        # 删除隐藏分类声明
        r'Hidden categories:.*?\]\]',
        # 移除参考文献区块
        r'References\n-+\n.*?1\..*?(?=\n\n|$)',
        # 删除页面底部链接
        r'Retrieved from ".*?\]\]',
        # 删除图片标记
        r'\[\[File:.*?\]\]',
        # 删除外部链接标记
        r'\[http.*?\]',
        # 删除小工具标记
        r'\n__.*?__\n',
        # 删除空列表项
        r'\n\*?\s*?\n'
    ]

    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

    # 第二阶段：处理特殊格式
    processing_steps = [
        # 标准化标题格式（将下划线转换为##）
        (r'\n(.*?)\n=+\n', r'\n# \1\n'),
        # 将二级标题转换为##
        (r'\n(.*?)\n-+\n', r'\n## \1\n'),
        # 合并连续空行
        (r'\n{3,}', '\n\n'),
        # 删除行内引用标记 [1]
        (r'\[\d+\]', ''),
        # 处理法律术语连字符
        (r'(\w+) - (\w+)', r'\1-\2'),
        # 删除残留的表格符号
        (r'^\|.*\n', ''),
        # 删除维基百科内部链接（保留显示文本）
        (r'\[\[(.*?\|)?(.*?)\]\]', r'\2'),
        # 清理残留的HTML标签
        (r'<.*?>', ''),
        # 删除行首行尾空白
        (r'^\s+|\s+$', '')
    ]

    for pattern, repl in processing_steps:
        text = re.sub(pattern, repl, text)

    # 第三阶段：选择性保留关键内容
    sections_to_keep = [
        r'## History.*?(?=\n##|$)',
        r'## Types.*?(?=\n##|$)',
        r'## Comparison.*?(?=\n##|$)',
        r'## Legal requirements.*?(?=\n##|$)',
        r'## Key Statistics.*?(?=\n##|$)'
    ]

    kept_sections = []
    for pattern in sections_to_keep:
        matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        kept_sections.extend(matches)

    # 如果找到关键章节，则重建文本
    if kept_sections:
        text = '\n\n'.join(kept_sections)

    # 最终清理
    final_cleanup = [
        # 删除技术术语注释
        (r'\([^)]*?technical note.*?\)', ''),
        # 统一引号格式
        (r'``(.*?)``', r'"\1"'),
        # 删除孤立符号
        (r'^[^a-zA-Z0-9]*$', ''),
        # 合并重复换行
        (r'\n{2,}', '\n\n')
    ]

    for pattern, repl in final_cleanup:
        text = re.sub(pattern, repl, text)

    return text.strip()


def mdTocleanMd(input_path, output_path):
    # 统计文件中的总行数
    with jsonlines.open(input_path) as reader:
        total_lines = sum(1 for _ in reader)

    with jsonlines.open(input_path) as reader, jsonlines.open(output_path, mode='w') as writer:
        for obj in tqdm(reader, desc="Processing", total=total_lines):
            # 清理Markdown文本
            cleaned_text = clean_markdown(obj['document_text'])
            # 创建新的字典，替换document_text
            new_obj = {
                'document_id': obj['document_id'],
                'document_text': cleaned_text
            }
            # 写入新的JSONL文件
            writer.write(new_obj)

    print(f"Markdown文本清理完成，输出文件路径：{output_path}")


if __name__ == '__main__':
    input_path = "/Users/shione/python/nlp/project/documents_markdown.jsonl"
    output_path = "/Users/shione/python/nlp/project/documents_markdown_clean.jsonl"
    mdTocleanMd(input_path, output_path)
