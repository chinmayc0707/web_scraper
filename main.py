import psycopg
import tiktoken
import mcp
from psycopg.rows import dict_row
import os

# --------------------langchain------------------
from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter
from langchain.chat_models import BaseChatModel
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_mcp_adapters.client import MultiServerMCPClient

from dotenv import load_dotenv

# -----------------MCP libraries--------------------------
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

import asyncio

load_dotenv()


async def get_tools():

    client = MultiServerMCPClient(
        {
            "duckduckgo": {
                "command": "uvx",
                "args": ["duckduckgo-mcp-server"],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()

    return tools


class PersistentMemoryAgent:

    def __init__(
        self,
        llm: BaseChatModel,
        db_uri,
        max_context_tokens=32000,
        summary_threshold_ratio=0.75,
        keep_recent_messages=10,
        thread_id="default_thread",
        tools=[],
    ):

        self.db_uri = db_uri
        self.model_name = llm.model_name

        self.max_context_tokens = max_context_tokens

        self.summary_threshold = int(max_context_tokens * summary_threshold_ratio)

        self.keep_recent_messages = keep_recent_messages

        self.thread_id = thread_id
        self.system_message = "You are a helpful assistant."

        # =========================
        # TOKENIZER
        # =========================

        self.enc = tiktoken.get_encoding("cl100k_base")

        # =========================
        # LLM
        # =========================

        self.llm = llm

        # =========================
        # DATABASE
        # =========================

        self.conn = psycopg.connect(
            self.db_uri, autocommit=True, prepare_threshold=0, row_factory=dict_row
        )

        self.memory = PostgresSaver(self.conn)

        # Run only first time if needed
        self._setup_checkpoint_tables()

        # =========================
        # SUMMARY TABLE
        # =========================

        self._create_summary_table()

        # =========================
        # AGENT
        # =========================

        self.agent = create_agent(model=self.llm, tools=tools, checkpointer=self.memory)

        self.config = {"configurable": {"thread_id": self.thread_id}}

    # =====================================================
    # DATABASE
    # =====================================================
    def _setup_checkpoint_tables(self):

        queries = [
            """
            CREATE TABLE IF NOT EXISTS checkpoint_migrations (
                v INTEGER PRIMARY KEY
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                type TEXT,
                checkpoint JSONB NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}',
                PRIMARY KEY (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id
                )
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS checkpoint_blobs (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                channel TEXT NOT NULL,
                version TEXT NOT NULL,
                type TEXT NOT NULL,
                blob BYTEA,
                PRIMARY KEY (
                    thread_id,
                    checkpoint_ns,
                    channel,
                    version
                )
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS checkpoint_writes (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                channel TEXT NOT NULL,
                type TEXT,
                blob BYTEA NOT NULL,
                PRIMARY KEY (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    task_id,
                    idx
                )
            )
            """,
            """
            INSERT INTO checkpoint_migrations (v)
            VALUES (1)
            ON CONFLICT DO NOTHING
            """,
        ]

        with self.conn.cursor() as cur:

            for query in queries:
                cur.execute(query)

    def _create_summary_table(self):

        with self.conn.cursor() as cur:

            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_summaries (
                    thread_id TEXT PRIMARY KEY,
                    summary TEXT
                )
            """)

    # =====================================================
    # TOKEN COUNT
    # =====================================================

    def count_tokens(self, messages):

        text = ""

        for m in messages:
            text += str(m.content) + "\n"

        return len(self.enc.encode(text))

    def get_token_count(self):
        """Returns the total token count of the current conversation state."""
        state = self.agent.get_state(self.config)
        messages = state.values.get("messages", [])
        return self.count_tokens(messages)

    # =====================================================
    # SUMMARY
    # =====================================================

    def get_summary(self):

        with self.conn.cursor() as cur:

            cur.execute(
                """
                SELECT summary
                FROM chat_summaries
                WHERE thread_id=%s
                """,
                (self.thread_id,),
            )

            row = cur.fetchone()

            if row:
                return row["summary"]

        return ""

    def save_summary(self, summary):

        with self.conn.cursor() as cur:

            cur.execute(
                """
                INSERT INTO chat_summaries(thread_id, summary)
                VALUES (%s, %s)

                ON CONFLICT(thread_id)
                DO UPDATE SET summary=EXCLUDED.summary
            """,
                (self.thread_id, summary),
            )

    # =====================================================
    # SUMMARIZATION
    # =====================================================

    def summarize_if_needed(self, state):

        messages = state["messages"]

        total_tokens = self.count_tokens(messages)

        print(f"TOKENS: {total_tokens} / THRESHOLD: {self.summary_threshold}")

        if total_tokens < self.summary_threshold:
            return

        recent_messages = messages[-self.keep_recent_messages :]
        old_messages = messages[: -self.keep_recent_messages]

        old_text = "\n".join(
            [
                f"{getattr(m, 'type', 'message')}: {getattr(m, 'content', str(m))}"
                for m in old_messages
            ]
        )

        existing_summary = self.get_summary()

        prompt = f"""
Existing summary:

{existing_summary}

Summarize the following conversation compactly.

Preserve:
- important facts
- preferences
- tasks
- ongoing context
- user-specific memory

Conversation:

{old_text}
"""

        response = self.llm.invoke(prompt)

        new_summary = response.content

        self.save_summary(new_summary)

        print("SUMMARY UPDATED")

    # =====================================================
    # SYSTEM MESSAGE
    # =====================================================

    def set_system_message(self, message):
        """Sets the custom system message for the agent."""
        self.system_message = message

    # =====================================================
    # CHAT
    # =====================================================

    def invoke(self, user_input, verbose=False):
        """Helper that uses stream_messages to get the full response."""
        full_response = ""
        for chunk in self.stream_messages(user_input, verbose=verbose):
            full_response += chunk
        return full_response

    def stream_messages(self, user_input, verbose=False):
        """Streams the response from the agent token by token."""
        if verbose:
            print(f"--- DEBUG: Current Token Count: {self.get_token_count()} ---")

        summary = self.get_summary()
        system_prompt = (
            f"{self.system_message}\n\nLong-term conversation summary:\n\n{summary}"
        )

        inputs = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ]
        }

        # We use stream_mode="messages" to get token-by-token updates if supported by the adapter
        for msg, metadata in self.agent.stream(
            inputs, config=self.config, stream_mode="messages"
        ):
            # Handle tool calls if verbose is enabled
            if verbose and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    print(
                        f"\n--- DEBUG: Tool Call: {tool_call['name']} with args: {tool_call['args']} ---"
                    )

            # Only yield content from the AI's message
            if msg.content:
                yield msg.content

        # After streaming is complete, we check if summarization is needed.
        state = self.agent.get_state(self.config)

        if verbose:
            # Temporarily override summarize_if_needed's internal prints by wrapping it or checking here
            print(
                f"--- DEBUG: Post-response Token Count: {self.count_tokens(state.values.get('messages', []))} ---"
            )

        self.summarize_if_needed(state.values)

    # =====================================================
    # RESET
    # =====================================================

    def clear_summary(self):

        with self.conn.cursor() as cur:

            cur.execute(
                """
                DELETE FROM chat_summaries
                WHERE thread_id=%s
                """,
                (self.thread_id,),
            )

    # =====================================================
    # CLOSE
    # =====================================================

    def close(self):

        self.conn.close()


# =========================================================
# USAGE
# =========================================================

if __name__ == "__main__":

    DB_URI = os.getenv("DB_URL")
    llm = ChatOpenRouter(
        model="google/gemma-4-31b-it:free", api_key=os.getenv("OPENROUTER_API_KEY")
    )
    tools = asyncio.run(get_tools())
    agent = PersistentMemoryAgent(
        llm=llm, db_uri=DB_URI, thread_id="user_1", tools=tools
    )
    agent.set_system_message(
        "You are a web searching agent, use given tools to generate a detailed response of given query"
    )
    while True:

        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        for content in agent.stream_messages(user_input, verbose=True):
            print(content, end="", flush=True)

    # asyncio.run( get_tools())
