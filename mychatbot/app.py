from flask import Flask, request, render_template, Response, stream_with_context
import requests
import threading
import webbrowser
import time
import json
from py2neo import Graph
from question_classifier import QuestionClassifier
from question_parser import QuestionPaser

app = Flask(__name__)

# 模型接口配置（Ollama 本地部署的 deepseek-r1:7b）
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "deepseek-r1:7b"

# 初始化图数据库连接（确保与你的 Neo4j 配置一致）
graph = Graph("bolt://127.0.0.1:7687", auth=("neo4j", "hly20050529"))

# 初始化问句分类器和解析器
classifier = QuestionClassifier()
parser = QuestionPaser()

@app.route('/')
def index():
    return render_template('index.html')

# 执行知识图谱查询
def search_knowledge_graph(question):
    classify_result = classifier.classify(question)
    if not classify_result:
        return None

    sqls = parser.parser_main(classify_result)
    if not sqls:
        return None

    answers = []
    for sql in sqls:
        question_type = sql['question_type']
        for query in sql['sql']:
            try:
                result = graph.run(query).data()
                for r in result:
                    keys = list(r.keys())
                    if len(keys) == 2:
                        answers.append(f"{r[keys[0]]} 的 {question_type} 是：{r[keys[1]]}")
                    elif len(keys) == 3:
                        answers.append(f"{r[keys[0]]} 的 {r[keys[1]]} 是：{r[keys[2]]}")
            except Exception as e:
                answers.append(f"[图谱查询错误] {str(e)}")

    return '\n'.join(answers) if answers else None

# 提供知识图谱查询信息给前端（用于渲染额外知识块）
@app.route('/chat_graph_info', methods=['POST'])
def chat_graph_info():
    user_input = request.json.get("message", "")
    kg_info = search_knowledge_graph(user_input)
    return {"kg_info": kg_info or ""}

# 主聊天接口（图谱增强 + 本地模型生成）
@app.route('/chat_stream', methods=['POST'])
def chat():
    user_input = request.json.get("message", "")

    messages = []

    # 查询知识图谱
    kg_info = search_knowledge_graph(user_input)
    if kg_info:
        messages.append({
            "role": "system",
            "content": f"以下是知识图谱提供的信息，请优先参考：\n{kg_info}"
        })

    messages.append({"role": "user", "content": user_input})

    # 构造请求体
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True
    }

    # 流式生成模型响应
    def generate():
        try:
            with requests.post(OLLAMA_URL, json=payload, stream=True) as resp:
                for line in resp.iter_lines():
                    if line:
                        data = json.loads(line.decode('utf-8'))
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
        except Exception as e:
            yield f"[错误] {str(e)}"

    return Response(stream_with_context(generate()), content_type='text/plain')

# 自动打开网页
def open_browser():
    time.sleep(1)
    webbrowser.open("http://127.0.0.1:5000")

# 启动服务
if __name__ == '__main__':
    threading.Thread(target=open_browser).start()
    app.run(debug=True)
