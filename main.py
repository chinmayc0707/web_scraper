import psycopg
import tiktoken
import mcp
from psycopg.rows import dict_row
import os

#--------------------langchain------------------
from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter
from langchain.chat_models import BaseChatModel
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_mcp_adapters.client import MultiServerMCPClient

from dotenv import load_dotenv
#-----------------MCP libraries--------------------------
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
        llm:BaseChatModel,
        db_uri,
        
        max_context_tokens=32000,
        summary_threshold_ratio=0.75,
        keep_recent_messages=10,
        thread_id="default_thread",
        tools=[]
    ):

        self.db_uri = db_uri
        self.model_name = llm.model_name

        self.max_context_tokens = max_context_tokens

        self.summary_threshold = int(
            max_context_tokens * summary_threshold_ratio
        )

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
        # =========================        self.tools = tools

    async def setup(self):
        self.conn = await psycopg.AsyncConnection.connect(
            self.db_uri,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row
        )

        self.memory = AsyncPostgresSaver(self.conn)

        # Run only first time if needed
        await self._setup_checkpoint_tables()

        # =========================
        # SUMMARY TABLE
        # =========================

        await self._create_summary_table()

        # =========================
        # AGENT
        # =========================

        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            checkpointer=self.memory
        )

        self.config = {
            "configurable": {
                "thread_id": self.thread_id
            }
        }

    # =====================================================
    # DATABASE
    # =====================================================
    async def _setup_checkpoint_tables(self):

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
            """
        ]

        async with self.conn.cursor() as cur:

            for query in queries:
                await cur.execute(query)

    async def _create_summary_table(self):

        async with self.conn.cursor() as cur:

            await cur.execute("""
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

    async def get_token_count(self):
        """Returns the total token count of the current conversation state."""
        state = await self.agent.aget_state(self.config)
        messages = state.values.get("messages", [])
        return self.count_tokens(messages)

    # =====================================================
    # SUMMARY
    # =====================================================

    async def get_summary(self):

        async with self.conn.cursor() as cur:

            await cur.execute(
                """
                SELECT summary
                FROM chat_summaries
                WHERE thread_id=%s
                """,
                (self.thread_id,)
            )

            row = await cur.fetchone()

            if row:
                return row["summary"]

        return ""

    async def save_summary(self, summary):

        async with self.conn.cursor() as cur:

            await cur.execute("""
                INSERT INTO chat_summaries(thread_id, summary)
                VALUES (%s, %s)

                ON CONFLICT(thread_id)
                DO UPDATE SET summary=EXCLUDED.summary
            """, (self.thread_id, summary))

    # =====================================================
    # SUMMARIZATION
    # =====================================================

    async def summarize_if_needed(self, state):

        messages = state["messages"]

        total_tokens = self.count_tokens(messages)

        print(f"TOKENS: {total_tokens} / THRESHOLD: {self.summary_threshold}")

        if total_tokens < self.summary_threshold:
            return

        recent_messages = messages[-self.keep_recent_messages:]
        old_messages = messages[:-self.keep_recent_messages]

        old_text = "\n".join([
            f"{getattr(m, 'type', 'message')}: {getattr(m, 'content', str(m))}"
            for m in old_messages
        ])

        existing_summary = await self.get_summary()

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

        response = await self.llm.ainvoke(prompt)

        new_summary = response.content

        await self.save_summary(new_summary)

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

    async def invoke(self, user_input, verbose=False):
        """Helper that uses stream_messages to get the full response."""
        full_response = ""
        async for chunk in self.stream_messages(user_input, verbose=verbose):
            full_response += chunk
        return full_response

    async def stream_messages(self, user_input, verbose=False):
        """Streams the response from the agent token by token."""
        if verbose:
            print(f"--- DEBUG: Current Token Count: {await self.get_token_count()} ---")

        summary = await self.get_summary()
        system_prompt = f"{self.system_message}\n\nLong-term conversation summary:\n\n{summary}"

        inputs = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        }

        # We use stream_mode="messages" to get token-by-token updates if supported by the adapter
        async for msg, metadata in self.agent.astream(inputs, config=self.config, stream_mode="messages"):
            # Handle tool calls if verbose is enabled
            if verbose and hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    print(f"\n--- DEBUG: Tool Call: {tool_call['name']} with args: {tool_call['args']} ---")

            # Only yield content from the AI's message
            if msg.content:
                yield msg.content

        # After streaming is complete, we check if summarization is needed.
        state = await self.agent.aget_state(self.config)
        
        if verbose:
            # Temporarily override summarize_if_needed's internal prints by wrapping it or checking here
            print(f"--- DEBUG: Post-response Token Count: {self.count_tokens(state.values.get('messages', []))} ---")
        
        await self.summarize_if_needed(state.values)

    # =====================================================
    # RESET
    # =====================================================

    async def clear_summary(self):

        async with self.conn.cursor() as cur:

            await cur.execute(
                """
                DELETE FROM chat_summaries
                WHERE thread_id=%s
                """,
                (self.thread_id,)
            )

    # =====================================================
    # CLOSE
    # =====================================================

    async def close(self):

        await self.conn.close()


# =========================================================
# USAGE
# =========================================================

async def main():
    DB_URI = os.getenv('DB_URL')
    llm=ChatOpenRouter(model='google/gemma-4-31b-it:free',api_key=os.getenv('OPENROUTER_API_KEY'))
    tools=await get_tools()
    agent = PersistentMemoryAgent(
        llm=llm,
        db_uri=DB_URI,
        thread_id="user_1",
        tools=tools
    )
    await agent.setup()
    agent.set_system_message("You are a web searching agent, use given tools to generate a detailed response of given query")
    while True:

        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        async for content in agent.stream_messages(user_input,verbose=True):
            print(content,end='',flush=True)

if __name__ == "__main__":
    asyncio.run(main())