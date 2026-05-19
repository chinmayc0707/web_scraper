import psycopg
import tiktoken
import mcp
from psycopg.rows import dict_row
import os
import uuid

#--------------------langchain------------------

from langchain_openrouter import ChatOpenRouter
from langchain.chat_models import BaseChatModel
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_mcp_adapters.client import MultiServerMCPClient
import psycopg_pool

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
        thread_id,
        tools,
        max_context_tokens=32000,
        summary_threshold=24000,
        keep_recent_messages=5
    ):

        self.db_uri = db_uri
        self.thread_id = thread_id
        self.tools = tools

        self.max_context_tokens = max_context_tokens
        self.summary_threshold = summary_threshold
        self.keep_recent_messages = keep_recent_messages

        self.system_message = "You are a helpful assistant."

        self.enc = tiktoken.get_encoding("cl100k_base")

        self.llm = llm

    async def setup(self):
        self.pool = psycopg_pool.AsyncConnectionPool(
            conninfo=self.db_uri,
            max_size=20,
            open=False,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            }
        )
        await self.pool.open()
        self.memory = AsyncPostgresSaver(self.pool)
        await self.memory.setup()

        await self._create_summary_table()

        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            checkpointer=self.memory
        )

        self.config = {
            "configurable": {
                "thread_id": self.thread_id
            }
        }

    async def _create_summary_table(self):
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS chat_summaries (
                        thread_id TEXT PRIMARY KEY,
                        summary TEXT
                    )
                """)

    def count_tokens(self, messages):
        text = ""
        for m in messages:
            text += str(m.content) + "\n"
        return len(self.enc.encode(text))

    async def get_token_count(self):
        state = await self.agent.aget_state(self.config)
        messages = state.values.get("messages", [])
        return self.count_tokens(messages)

    async def get_summary(self):
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
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
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    INSERT INTO chat_summaries(thread_id, summary)
                    VALUES (%s, %s)
                    ON CONFLICT(thread_id)
                    DO UPDATE SET summary=EXCLUDED.summary
                """, (self.thread_id, summary))

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

    def set_system_message(self, message):
        self.system_message = message

    async def invoke(self, user_input, verbose=False):
        full_response = ""
        async for chunk in self.stream_messages(user_input, verbose=verbose):
            full_response += chunk
        return full_response

    async def stream_messages(self, user_input, verbose=False):
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

        async for msg, metadata in self.agent.astream(inputs, config=self.config, stream_mode="messages"):
            if verbose and hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    print(f"\n--- DEBUG: Tool Call: {tool_call['name']} with args: {tool_call['args']} ---")
            if msg.content:
                yield msg.content

        state = await self.agent.aget_state(self.config)
        
        if verbose:
            print(f"--- DEBUG: Post-response Token Count: {self.count_tokens(state.values.get('messages', []))} ---")
        
        await self.summarize_if_needed(state.values)

    async def clear_summary(self):
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    DELETE FROM chat_summaries
                    WHERE thread_id=%s
                    """,
                    (self.thread_id,)
                )

    async def close(self):
        await self.pool.close()

async def run_agent():
    DB_URI = os.getenv('DB_URL')
    llm=ChatOpenRouter(model='google/gemma-4-31b-it:free',api_key=os.getenv('OPENROUTER_API_KEY'))
    tools=await get_tools()
    agent = PersistentMemoryAgent(
        llm=llm,
        db_uri=DB_URI,
        thread_id=f"user_{uuid.uuid4()}",
        tools=tools
    )
    await agent.setup()
    agent.set_system_message("You are a web searching agent, use given tools to generate a detailed response of given query")
    while True:
        try:
            user_input = input("You: ")
        except EOFError:
            break
        if user_input.lower() == "exit":
            break
        async for content in agent.stream_messages(user_input,verbose=True):
            print(content,end='',flush=True)
    await agent.close()

if __name__ == "__main__":
    asyncio.run(run_agent())
