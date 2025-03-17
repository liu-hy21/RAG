import gradio as gr

# 模拟文档数据样例
sample_docs = [
    {"title": "文档1：知识图谱技术", "content": "知识图谱是一种用图结构建模实体及其关系的技术...",
     "source": "《人工智能导论》第3章"},
    {"title": "文档2：问答系统分类", "content": "问答系统可分为基于规则的、检索式的和生成式的...",
     "source": "QA系统综述论文"},
    {"title": "文档3：Gradio框架", "content": "Gradio是一个快速创建机器学习Web界面的Python库...",
     "source": "官方文档v3.0"}
]


# 文档展示的HTML模板
def format_docs_html(docs):
    html = "<div style='padding: 10px; background: #f5f5f5; border-radius: 5px;'>"
    for doc in docs:
        html += f"""
        <div style='margin: 10px 0; padding: 15px; background: white; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1)'>
            <h4 style='margin: 0 0 8px 0; color: #333;'>{doc['title']}</h4>
            <p style='margin: 0 0 8px 0; color: #666;'>{doc['content']}</p>
            <small style='color: #999;'>{doc['source']}</small>
        </div>
        """
    return html + "</div>"


# 处理用户输入（模拟实现）
def process_query(question):
    # 此处应连接实际问答系统，现在返回模拟结果
    answer = "这是一个示例答案：知识图谱（Knowledge Graph）是一种用图结构来组织和表示知识的技术。"

    # 格式化答案
    answer_html = f"""
    <div style='padding: 20px; background: #e8f5e9; border-radius: 5px; border: 1px solid #c8e6c9;'>
        <h3 style='margin-top: 0; color: #2e7d32;'>系统回答：</h3>
        <p style='margin: 0; line-height: 1.6;'>{answer}</p>
    </div>
    """

    # 返回格式化的文档和答案
    return format_docs_html(sample_docs), answer_html


# 界面布局
with gr.Blocks(title="KBQA系统交互界面", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 知识图谱问答系统")
    gr.Markdown("请输入您的问题，系统将检索相关文档并生成回答")

    with gr.Row():
        # 用户输入区
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label="输入问题",
                placeholder="请输入您的问题...",
                lines=3,
                max_lines=5
            )
            submit_btn = gr.Button("提交问题", variant="primary")

        # 结果显示区
        with gr.Column(scale=7):
            answer_output = gr.HTML(label="系统回答")
            docs_output = gr.HTML(label="参考文档")

    # 交互逻辑
    submit_btn.click(
        fn=process_query,
        inputs=question_input,
        outputs=[docs_output, answer_output]
    )

    # 示例问题
    examples = gr.Examples(
        examples=["什么是知识图谱？", "问答系统有哪些类型？"],
        inputs=question_input
    )

if __name__ == "__main__":
    demo.launch()
