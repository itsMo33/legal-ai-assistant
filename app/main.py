from flask import Flask, request, jsonify, render_template
from app.legal_agent import ask_legal_question
import os

app = Flask(__name__, template_folder="templates")
conversation_history = []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    global conversation_history
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"answer": "الرجاء كتابة سؤال."})

    conversation_history.append({"role": "user", "content": question})

    if len(conversation_history) > 1:
        history_text = "\n".join([
            ("المستخدم: " if m["role"] == "user" else "المساعد: ") + m["content"]
            for m in conversation_history[-6:]
        ])
        full_question = "تاريخ المحادثة:\n" + history_text + "\n\nالسؤال الحالي: " + question
    else:
        full_question = question

    answer = ask_legal_question(full_question)
    conversation_history.append({"role": "assistant", "content": answer})
    return jsonify({"answer": answer})

@app.route("/clear", methods=["POST"])
def clear():
    global conversation_history
    conversation_history = []
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    print("الخادم يعمل على Render")
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)