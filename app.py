from flask import Flask, render_template, request, Response
import os
import asyncio
from main import PersistentMemoryAgent, get_tools
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Initialize agent lazily
agent = None


def get_agent():
    global agent
    if agent is None:
        DB_URI = os.getenv("DB_URL")
        llm = ChatOpenRouter(
            model="google/gemma-4-31b-it:free", api_key=os.getenv("OPENROUTER_API_KEY")
        )
        tools = asyncio.run(get_tools())
        agent = PersistentMemoryAgent(
            llm=llm, db_uri=DB_URI, thread_id="user_1", tools=tools
        )
        agent.set_system_message("You are a helpful assistant.")
    return agent


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")

    def generate():
        try:
            current_agent = get_agent()
            for chunk in current_agent.stream_messages(user_input):
                yield f"data: {chunk}\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
