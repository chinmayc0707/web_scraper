import psycopg
import tiktoken
from psycopg.rows import dict_row
import os
import re
import selectors
import sys
from datetime import date
from pathlib import Path
from uuid import uuid4

#--------------------langchain------------------
from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_mcp_adapters.client import MultiServerMCPClient

from dotenv import load_dotenv

import asyncio

load_dotenv()


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def require_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _new_event_loop():
    if os.name == "nt":
        return asyncio.SelectorEventLoop(selectors.SelectSelector())
    return asyncio.new_event_loop()


def _run_async(coro, async_method_name):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = _new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    coro.close()
    raise RuntimeError(
        f"This method cannot run inside an active event loop. "
        f"Use {async_method_name} instead."
    )


def _content_to_text(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "".join(text_parts)

    return ""


def _truncate_text(text, max_chars=1800):
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def _message_role(message):
    message_type = getattr(message, "type", "message")
    return {
        "human": "user",
        "ai": "assistant",
        "AIMessageChunk": "assistant",
        "tool": "tool",
        "system": "system",
    }.get(message_type, message_type)


def _is_time_sensitive_request(text):
    lowered = text.lower()
    markers = [
        "current",
        "latest",
        "today",
        "now",
        "recent",
        "breaking",
        "news",
        "affairs",
        "as of",
        "up to date",
        "uptodate",
    ]
    return any(marker in lowered for marker in markers)


def _is_detailed_report_request(text):
    lowered = text.lower()
    markers = [
        "report",
        "deep dive",
        "deep-dive",
        "white paper",
        "analysis paper",
        "long form",
        "long-form",
    ]
    return any(marker in lowered for marker in markers)


def _refers_to_prior_context(text):
    lowered = text.lower()
    markers = [
        "same",
        "that",
        "this",
        "it",
        "above",
        "previous",
        "earlier",
        "we discussed",
        "continue",
    ]
    return any(marker in lowered for marker in markers)


def _extract_search_queries(plan, user_input, max_queries=5):
    queries = []
    in_search_queries = False

    for line in plan.splitlines():
        stripped = line.strip()
        lowered_line = stripped.lower()

        if "search queries" in lowered_line:
            in_search_queries = True
            continue

        if in_search_queries and stripped.startswith("**") and "search queries" not in lowered_line:
            break

        if not in_search_queries:
            continue

        match = re.match(r"^(?:[-*]|\d+[.)])\s*(.+)$", stripped)
        if not match:
            continue
        candidate = match.group(1).strip()
        if not candidate or len(candidate) > 180:
            continue
        candidate = candidate.strip("\"'")
        if candidate not in queries:
            queries.append(candidate)

    if not queries:
        queries.append(f"{user_input} {date.today().strftime('%B %d, %Y')}")

    return queries[:max_queries]


def _tool_result_to_text(result):
    text = _content_to_text(result)
    if text:
        return text
    return str(result)


def _evidence_has_sources(evidence):
    return "URL:" in evidence or "http://" in evidence or "https://" in evidence


def _extract_urls(text, max_urls=3):
    urls = []
    for url in re.findall(r"https?://[^\s)>\]]+", text):
        cleaned = url.rstrip(".,;:")
        if cleaned not in urls:
            urls.append(cleaned)
        if len(urls) >= max_urls:
            break
    return urls


def _word_count(text):
    return len(re.findall(r"\b[\w'-]+\b", text))


def _report_sections_for(user_input):
    lowered = user_input.lower()

    if any(marker in lowered for marker in ("war", "conflict", "iran", "military", "security")):
        return [
            ("Executive Summary and Key Judgments", 500, 750),
            ("Background, Origins, and Chronology", 650, 900),
            ("Actors, Objectives, and Strategic Constraints", 650, 900),
            ("Military and Operational Assessment", 700, 1000),
            ("Diplomacy, Law, and Negotiation Tracks", 600, 850),
            ("Economic, Energy, and Market Consequences", 650, 900),
            ("Humanitarian, Social, and Regional Effects", 600, 850),
            ("International Reactions and Alliance Dynamics", 600, 850),
            ("Forward Scenarios and Indicators to Watch", 600, 850),
            ("Sources, Confidence Levels, and Evidence Gaps", 350, 550),
        ]

    return [
        ("Executive Summary and Key Judgments", 500, 750),
        ("Method, Scope, and Source Quality", 350, 550),
        ("Geopolitical Landscape and Security Flashpoints", 700, 1000),
        ("Global Economy, Finance, Trade, and Energy", 650, 900),
        ("Science, Technology, AI, and Strategic Innovation", 600, 850),
        ("Climate, Environment, Health, and Humanitarian Conditions", 650, 900),
        ("Regional Developments and Political Transitions", 600, 850),
        ("Timeline of Key Events", 500, 750),
        ("Outlook, Risks, and Scenarios", 650, 900),
        ("Sources, Confidence Levels, and Evidence Gaps", 350, 550),
    ]


def _format_report_todo(sections, active_index=None):
    lines = ["\n--- REPORT TODO ---"]
    for index, (title, _, _) in enumerate(sections):
        if active_index is None:
            marker = "[ ]"
        elif index < active_index:
            marker = "[x]"
        elif index == active_index:
            marker = "[~]"
        else:
            marker = "[ ]"
        lines.append(f"{marker} {index + 1}. {title}")
    lines.append("--- END TODO ---\n")
    return "\n".join(lines)


def _slugify_filename(text):
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:80] or "report"


def _basic_markdown_to_html(markdown_text):
    import html

    lines = markdown_text.splitlines()
    html_parts = []
    in_ul = False
    in_ol = False
    in_code = False
    code_lines = []

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            html_parts.append("</ul>")
            in_ul = False
        if in_ol:
            html_parts.append("</ol>")
            in_ol = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                html_parts.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
                code_lines = []
                in_code = False
            else:
                close_lists()
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not stripped:
            close_lists()
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            close_lists()
            level = len(heading.group(1))
            html_parts.append(f"<h{level}>{html.escape(heading.group(2))}</h{level}>")
            continue

        unordered = re.match(r"^[-*]\s+(.+)$", stripped)
        if unordered:
            if not in_ul:
                close_lists()
                html_parts.append("<ul>")
                in_ul = True
            html_parts.append(f"<li>{html.escape(unordered.group(1))}</li>")
            continue

        ordered = re.match(r"^\d+[.)]\s+(.+)$", stripped)
        if ordered:
            if not in_ol:
                close_lists()
                html_parts.append("<ol>")
                in_ol = True
            html_parts.append(f"<li>{html.escape(ordered.group(1))}</li>")
            continue

        close_lists()
        html_parts.append(f"<p>{html.escape(stripped)}</p>")

    close_lists()
    if in_code:
        html_parts.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")

    return "\n".join(html_parts)


def _markdown_to_html(markdown_text):
    try:
        import markdown

        return markdown.markdown(
            markdown_text,
            extensions=[
                "extra",
                "sane_lists",
                "tables",
                "fenced_code",
                "nl2br",
            ],
            output_format="html5",
        )
    except Exception:
        return _basic_markdown_to_html(markdown_text)


def _font_paths():
    candidates = [
        {
            "regular": r"C:\Windows\Fonts\arial.ttf",
            "bold": r"C:\Windows\Fonts\arialbd.ttf",
            "italic": r"C:\Windows\Fonts\ariali.ttf",
            "bold_italic": r"C:\Windows\Fonts\arialbi.ttf",
        },
        {
            "regular": r"C:\Windows\Fonts\calibri.ttf",
            "bold": r"C:\Windows\Fonts\calibrib.ttf",
            "italic": r"C:\Windows\Fonts\calibrii.ttf",
            "bold_italic": r"C:\Windows\Fonts\calibriz.ttf",
        },
    ]

    for group in candidates:
        regular = Path(group["regular"])
        if regular.exists():
            return group

    return None


def _configure_report_pdf():
    from fpdf import FPDF

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(16, 16, 16)
    pdf.add_page()

    font_family = "Helvetica"
    fonts = _font_paths()
    if fonts:
        font_family = "ReportFont"
        pdf.add_font(font_family, "", fonts["regular"])
        if Path(fonts["bold"]).exists():
            pdf.add_font(font_family, "B", fonts["bold"])
        if Path(fonts["italic"]).exists():
            pdf.add_font(font_family, "I", fonts["italic"])
        if Path(fonts["bold_italic"]).exists():
            pdf.add_font(font_family, "BI", fonts["bold_italic"])

    pdf.set_font(font_family, size=11)
    pdf.set_title("Detailed Research Report")
    pdf.set_author("PersistentMemoryAgent")
    return pdf, font_family


def _write_plain_markdown_pdf(pdf, markdown_text, font_family):
    in_code = False

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code = not in_code
            continue

        if not stripped:
            pdf.ln(4)
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading and not in_code:
            level = len(heading.group(1))
            size = max(12, 22 - (level * 2))
            pdf.set_font(font_family, "B", size=size)
            pdf.multi_cell(0, 7, heading.group(2))
            pdf.ln(2)
            pdf.set_font(font_family, size=11)
            continue

        list_item = re.match(r"^(?:[-*]|\d+[.)])\s+(.+)$", stripped)
        if list_item and not in_code:
            pdf.set_font(font_family, size=11)
            pdf.multi_cell(0, 5.5, f"- {list_item.group(1)}")
            continue

        plain = re.sub(r"(\*\*|__)(.*?)\1", r"\2", stripped)
        plain = re.sub(r"(\*|_)(.*?)\1", r"\2", plain)
        plain = re.sub(r"`([^`]+)`", r"\1", plain)
        plain = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", plain)

        pdf.set_font(font_family, "B" if in_code else "", size=10 if in_code else 11)
        pdf.multi_cell(0, 5.5, plain)
        pdf.set_font(font_family, size=11)


def _render_markdown_pdf(markdown_text, pdf_path):
    html = _markdown_to_html(markdown_text)
    pdf, font_family = _configure_report_pdf()
    try:
        pdf.write_html(html, ul_bullet_char="-")
    except Exception:
        pdf, font_family = _configure_report_pdf()
        _write_plain_markdown_pdf(pdf, markdown_text, font_family)
    pdf.output(str(pdf_path))


def _reports_dir():
    configured_dir = os.getenv("REPORTS_DIR")
    reports_dir = (
        Path(configured_dir).expanduser()
        if configured_dir
        else Path(__file__).resolve().parent / "reports"
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def _new_report_artifact_paths(user_input):
    reports_dir = _reports_dir()

    stamp = date.today().strftime("%Y-%m-%d")
    suffix = uuid4().hex[:8]
    base_name = f"{stamp}-{_slugify_filename(user_input)}-{suffix}"
    markdown_path = reports_dir / f"{base_name}.md"
    pdf_path = reports_dir / f"{base_name}.pdf"
    return markdown_path.resolve(), pdf_path.resolve()


def _write_report_markdown(markdown_path, markdown_text):
    markdown_path = Path(markdown_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_text, encoding="utf-8")
    return markdown_path.resolve()


def _write_report_pdf(markdown_path, pdf_path):
    markdown_text = Path(markdown_path).read_text(encoding="utf-8")
    _render_markdown_pdf(markdown_text, Path(pdf_path))
    return Path(pdf_path).resolve()


def _write_report_artifacts(markdown_text, user_input):
    markdown_path, pdf_path = _new_report_artifact_paths(user_input)

    _write_report_markdown(markdown_path, markdown_text)
    _write_report_pdf(markdown_path, pdf_path)
    return markdown_path.resolve(), pdf_path.resolve()


def _blocked_report_markdown(user_input, research_plan, evidence, reason):
    return (
        "# Detailed Research Report\n\n"
        f"**Report date:** {date.today().strftime('%B %d, %Y')}\n\n"
        f"**User request:** {user_input}\n\n"
        "**Generation status:** blocked before drafting\n\n"
        "## Why the Report Was Not Drafted\n\n"
        f"{reason}\n\n"
        "No fallback provider is enabled, and this request needs current, source-grounded facts. "
        "The agent therefore created this audit file instead of producing an unverified report.\n\n"
        "## Research Plan\n\n"
        f"{research_plan or '(no research plan was produced)'}\n\n"
        "## Search Evidence Returned\n\n"
        "```text\n"
        f"{_truncate_text(evidence or '(no evidence returned)', 12000)}\n"
        "```\n"
    )


def _turn_report_markdown(user_input, turn_output):
    cleaned_output = (turn_output or "").strip()
    if not cleaned_output:
        cleaned_output = "_No report content was produced for this turn._"

    return (
        "# Detailed Research Report\n\n"
        f"**Report date:** {date.today().strftime('%B %d, %Y')}\n\n"
        f"**User request:** {user_input}\n\n"
        "**Generation status:** exported after the conversation turn completed\n\n"
        f"{cleaned_output}\n"
    )


def _write_turn_report_artifacts(user_input, turn_output):
    return _write_report_artifacts(
        _turn_report_markdown(user_input, turn_output),
        user_input,
    )


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

    return await client.get_tools()
class PersistentMemoryAgent:

    def __init__(
        self,
        llm:BaseChatModel,
        db_uri,
        
        max_context_tokens=32000,
        summary_threshold_ratio=0.75,
        keep_recent_messages=10,
        thread_id="default_thread",
        tools=None,
        conn=None,
        memory=None,
    ):
        if conn is None or memory is None:
            raise RuntimeError(
                "PersistentMemoryAgent uses async MCP tools. "
                "Create it with `await PersistentMemoryAgent.create(...)`."
            )

        self.db_uri = db_uri
        self.model_name = getattr(
            llm,
            "model_name",
            getattr(llm, "model", llm.__class__.__name__)
        )

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
        # =========================

        self.conn = conn
        self.memory = memory
        self.tools = tools or []
        self.tool_by_name = {tool.name: tool for tool in self.tools}

        # =========================
        # AGENTS
        # =========================

        self.planner_agent = create_agent(
            model=self.llm,
            tools=[],
            checkpointer=self.memory
        )

        self.research_agent = create_agent(
            model=self.llm,
            tools=self.tools,
            checkpointer=self.memory
        )

        self.planner_config = {
            "configurable": {
                "thread_id": f"{self.thread_id}:planner"
            }
        }

        self.research_config = {
            "configurable": {
                "thread_id": self.thread_id
            }
        }
        self.config = self.research_config
        self.agent = self.research_agent

    @classmethod
    async def create(
        cls,
        llm:BaseChatModel,
        db_uri,
        max_context_tokens=32000,
        summary_threshold_ratio=0.75,
        keep_recent_messages=10,
        thread_id="default_thread",
        tools=None
    ):
        conn = await psycopg.AsyncConnection.connect(
            db_uri,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row
        )
        memory = AsyncPostgresSaver(conn)

        agent = cls(
            llm=llm,
            db_uri=db_uri,
            max_context_tokens=max_context_tokens,
            summary_threshold_ratio=summary_threshold_ratio,
            keep_recent_messages=keep_recent_messages,
            thread_id=thread_id,
            tools=tools,
            conn=conn,
            memory=memory,
        )

        await agent._setup_checkpoint_tables()
        await agent._create_summary_table()
        return agent

    # =====================================================
    # DATABASE
    # =====================================================
    async def _setup_checkpoint_tables(self):
        await self.memory.setup()

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

    def get_token_count(self):
        """Returns the total token count of the current conversation state."""
        return _run_async(self.aget_token_count(), "aget_token_count()")

    async def aget_token_count(self):
        """Returns the total token count of the current conversation state."""
        planner_state = await self.planner_agent.aget_state(self.planner_config)
        research_state = await self.research_agent.aget_state(self.research_config)
        messages = []
        messages.extend(planner_state.values.get("messages", []))
        messages.extend(research_state.values.get("messages", []))
        return self.count_tokens(messages)

    # =====================================================
    # SUMMARY
    # =====================================================

    def get_summary(self):
        return _run_async(self.aget_summary(), "aget_summary()")

    async def aget_summary(self):

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

    def save_summary(self, summary):
        _run_async(self.asave_summary(summary), "asave_summary()")

    async def asave_summary(self, summary):

        async with self.conn.cursor() as cur:

            await cur.execute("""
                INSERT INTO chat_summaries(thread_id, summary)
                VALUES (%s, %s)

                ON CONFLICT(thread_id)
                DO UPDATE SET summary=EXCLUDED.summary
            """, (self.thread_id, summary))

    async def adelete_thread(self, thread_id=None):
        """
        Deletes all data associated with a specific thread,
        including the long-term summary and the LangGraph checkpoints.
        """
        target_id = thread_id or self.thread_id

        async with self.conn.cursor() as cur:
            # 1. Delete the custom summary
            await cur.execute(
                "DELETE FROM chat_summaries WHERE thread_id = %s",
                (target_id,)
            )

            # 2. Delete the LangGraph checkpoints
            # LangGraph uses the 'thread_id' column in the checkpoints table
            await cur.execute(
                "DELETE FROM checkpoints WHERE thread_id = %s",
                (target_id,)
            )

        print(f"Thread {target_id} and its summary have been deleted.")

    # =====================================================
    # SUMMARIZATION
    # =====================================================

    def summarize_if_needed(self, state):
        _run_async(self.asummarize_if_needed(state), "asummarize_if_needed()")

    async def asummarize_if_needed(self, state):

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

        existing_summary = await self.aget_summary()

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

        await self.asave_summary(new_summary)

        print("SUMMARY UPDATED")

    async def aget_recent_context(self):
        state = await self.research_agent.aget_state(self.research_config)
        messages = state.values.get("messages", [])
        context_parts = []

        for message in messages[-self.keep_recent_messages:]:
            role = _message_role(message)
            if role == "system":
                continue

            content = _content_to_text(getattr(message, "content", ""))
            if not content:
                continue

            context_parts.append(f"{role}: {_truncate_text(content, 1200)}")

        return "\n\n".join(context_parts)

    def _fallback_research_plan(self, user_input):
        return f"""
Research objective:
- Answer the user's request accurately and as of {date.today().strftime('%B %d, %Y')}.

Search plan:
- Search for the user's topic with today's date.
- Prefer recent primary or reputable news sources.
- Cross-check important claims across multiple sources.

Output requirements:
- Give a detailed answer.
- Include dates for time-sensitive facts.
- Clearly say when evidence is limited or uncertain.

User request:
{user_input}
"""

    async def aplan_research(
        self,
        user_input,
        summary,
        recent_context,
        context_note,
        verbose=False
    ):
        today = date.today().strftime('%B %d, %Y')
        system_prompt = f"""
You are the planning part of a two-agent research system.

Today's date is {today}. Build a concise research plan for the researching agent.
Do not answer the user directly. Do not use tools.
Use conversation history only as context for the user's interests and continuity.
Do not treat prior assistant claims, summaries, or old tool outputs as verified facts.
If prior context contains factual claims relevant to the new request, list them as claims to verify.
If the user's new request is broad, do not narrow it to an old topic unless the user explicitly asks for that topic.

Your plan must include:
- research objective
- relevant conversation context to preserve
- freshness/date requirements
- search queries to run
- when to fetch full page content after search
- source quality and cross-check strategy
- citation/source requirements
- prior-context claims that need verification or correction
- final answer structure

For time-sensitive topics, force the researching agent to verify facts against today's date.
The researching agent must not make factual claims that are not supported by search or fetched source content.
"""

        planner_prompt = f"""
User request:
{user_input}

Long-term summary:
{summary or "(none)"}

Recent conversation context:
{recent_context or "(none)"}

Context handling note:
{context_note}
"""

        try:
            response = await self.planner_agent.ainvoke(
                {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": planner_prompt},
                    ]
                },
                config=self.planner_config
            )
        except Exception as exc:
            if verbose:
                print(f"\n--- DEBUG: Planner failed; using fallback plan: {exc} ---")
            return self._fallback_research_plan(user_input)

        messages = response.get("messages", [])
        if not messages:
            return self._fallback_research_plan(user_input)

        plan = _content_to_text(messages[-1].content).strip()
        return plan or self._fallback_research_plan(user_input)

    async def agather_research_evidence(self, user_input, research_plan, verbose=False):
        search_tool = self.tool_by_name.get("search")
        fetch_tool = self.tool_by_name.get("fetch_content")
        if search_tool is None:
            return "No search tool is available."

        evidence_parts = []
        queries = _extract_search_queries(research_plan, user_input)

        for index, query in enumerate(queries, start=1):
            if verbose:
                print(f"\n--- DEBUG: Evidence Search {index}: {query} ---")

            try:
                result = await search_tool.ainvoke({"query": query, "max_results": 5})
                result_text = _tool_result_to_text(result)
            except Exception as exc:
                result_text = f"Search failed: {exc}"

            if fetch_tool is not None and _evidence_has_sources(result_text):
                fetched_parts = []
                for url_index, url in enumerate(_extract_urls(result_text, max_urls=1), start=1):
                    if verbose:
                        print(f"\n--- DEBUG: Fetching source {url_index} for query {index}: {url} ---")
                    try:
                        fetched = await fetch_tool.ainvoke({
                            "url": url,
                            "start_index": 0,
                            "max_length": 5000,
                        })
                        fetched_parts.append(
                            f"Fetched source {url_index}: {url}\n"
                            f"{_truncate_text(_tool_result_to_text(fetched), 5000)}"
                        )
                    except Exception as exc:
                        fetched_parts.append(f"Fetch failed for {url}: {exc}")

                if fetched_parts:
                    result_text = result_text + "\n\n" + "\n\n".join(fetched_parts)

            evidence_parts.append(
                f"Query {index}: {query}\n{result_text}"
            )

        return "\n\n".join(evidence_parts)

    async def agenerate_report_section(
        self,
        user_input,
        research_plan,
        evidence,
        section_title,
        section_index,
        total_sections,
        min_words,
        max_words,
        prior_section_notes,
        memory_context,
        context_note,
    ):
        today = date.today().strftime('%B %d, %Y')
        prompt = f"""
You are the report-writing stage of a planning-and-research agent.

Write section {section_index} of {total_sections} for a detailed report.

User request:
{user_input}

Report section to write:
{section_index}. {section_title}

Target length:
- Minimum {min_words} words.
- Preferred range {min_words}-{max_words} words.
- Do not compress this into a summary. Use full paragraphs, detailed analysis, and clear subheadings.

Date and accuracy:
- The report date is {today}.
- Keep facts accurate to this date.
- Use only the provided research plan and gathered search evidence.
- Do not invent facts, dates, figures, operations, casualties, prices, or events.
- If evidence is thin, write an evidence-limited analysis and clearly say what could not be verified.
- Include URL citations inline for factual claims wherever the evidence includes URLs.

Continuity:
- This is one section of a longer 5-10 page report.
- Do not repeat the full report introduction in every section.
- Use the prior section notes only to avoid repetition and keep continuity.

Prior section notes:
{prior_section_notes or "(none yet)"}

Prior memory context:
{memory_context or "(none)"}

Context handling note:
{context_note}

Research plan:
{research_plan}

Gathered search evidence:
{_truncate_text(evidence, 12000)}

Now write only section {section_index}: {section_title}.
"""

        response = await self.llm.ainvoke(prompt)
        section_text = _content_to_text(response.content).strip()

        if _word_count(section_text) < min_words:
            expansion_prompt = f"""
The section below is too short for the requested detailed report.

Expand it to at least {min_words} words while preserving accuracy and citations.
Do not add unsupported facts. If evidence is limited, expand the analysis of implications,
uncertainties, confidence levels, and what should be verified next.

Section title:
{section_title}

Current draft:
{section_text}

Research evidence:
{_truncate_text(evidence, 10000)}
"""
            expanded_response = await self.llm.ainvoke(expansion_prompt)
            expanded_text = _content_to_text(expanded_response.content).strip()
            if _word_count(expanded_text) > _word_count(section_text):
                section_text = expanded_text

        return section_text

    async def agenerate_detailed_report(
        self,
        user_input,
        research_plan,
        evidence,
        memory_context,
        context_note,
        verbose=False,
    ):
        sections = _report_sections_for(user_input)
        total_sections = len(sections)
        report_parts = []
        prior_section_notes = ""

        yield "# Detailed Research Report\n"
        yield f"**Report date:** {date.today().strftime('%B %d, %Y')}\n"
        yield "**Generation mode:** section-by-section long-form report\n\n"
        yield _format_report_todo(sections)

        for index, (title, min_words, max_words) in enumerate(sections):
            yield _format_report_todo(sections, active_index=index)
            yield f"\n\n## {index + 1}. {title}\n\n"

            try:
                section_text = await self.agenerate_report_section(
                    user_input=user_input,
                    research_plan=research_plan,
                    evidence=evidence,
                    section_title=title,
                    section_index=index + 1,
                    total_sections=total_sections,
                    min_words=min_words,
                    max_words=max_words,
                    prior_section_notes=prior_section_notes,
                    memory_context=memory_context,
                    context_note=context_note,
                )
            except Exception as exc:
                section_text = (
                    f"Section generation failed: {exc}\n\n"
                    "This section could not be generated. Continue with the remaining sections if possible."
                )

            if _is_time_sensitive_request(user_input) and "http://" not in section_text and "https://" not in section_text:
                section_text += (
                    "\n\n**Citation note:** This section did not include URL citations. "
                    "Treat its factual claims as lower confidence unless they are supported elsewhere in the report."
                )

            report_parts.append(f"## {index + 1}. {title}\n\n{section_text}")
            yield section_text
            yield "\n\n"

            prior_section_notes = _truncate_text(
                prior_section_notes
                + f"\nSection {index + 1} completed: {title}\n"
                + _truncate_text(section_text, 900),
                4000,
            )

        yield _format_report_todo(sections, active_index=total_sections)
        full_report = "\n\n".join(report_parts)

        existing_summary = await self.aget_summary()
        report_memory = (
            f"{existing_summary}\n\n"
            f"Recent detailed report generated on {date.today().strftime('%B %d, %Y')} "
            f"for user request: {user_input}\n"
            f"Sections: {', '.join(title for title, _, _ in sections)}\n"
            f"Report excerpt: {_truncate_text(full_report, 2500)}"
        ).strip()
        await self.asave_summary(_truncate_text(report_memory, 10000))

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

    async def ainvoke(self, user_input, verbose=False):
        """Async helper that uses astream_messages to get the full response."""
        full_response = ""
        async for chunk in self.astream_messages(user_input, verbose=verbose):
            full_response += chunk
        return full_response

    def stream_messages(self, user_input, verbose=False):
        """Sync wrapper around astream_messages."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "stream_messages() cannot run inside an active event loop. "
                "Use astream_messages() instead."
            )

        loop = _new_event_loop()
        async_gen = self.astream_messages(user_input, verbose=verbose)

        try:
            asyncio.set_event_loop(loop)

            while True:
                try:
                    yield loop.run_until_complete(anext(async_gen))
                except StopAsyncIteration:
                    break
        finally:
            loop.run_until_complete(async_gen.aclose())
            asyncio.set_event_loop(None)
            loop.close()

    async def astream_messages(self, user_input, verbose=False):
        """Streams the response from the agent token by token."""
        if verbose:
            print(f"--- DEBUG: Current Token Count: {await self.aget_token_count()} ---")

        summary = await self.aget_summary()
        recent_context = await self.aget_recent_context()
        use_memory_context = not (
            _is_time_sensitive_request(user_input)
            and not _refers_to_prior_context(user_input)
        )

        if use_memory_context:
            summary_for_prompt = summary
            recent_context_for_prompt = recent_context
            context_note = (
                "Use memory for continuity, but verify factual claims with fresh sources before repeating them."
            )
        else:
            summary_for_prompt = _truncate_text(summary, 5000)
            recent_context_for_prompt = _truncate_text(recent_context, 2500)
            context_note = (
                "The user's request is broad and time-sensitive. Prior context is included only as summarized "
                "memory of what the agent previously did, not as factual evidence. Verify every factual claim "
                "against fresh source results before using it."
            )

        research_plan = await self.aplan_research(
            user_input,
            summary_for_prompt,
            recent_context_for_prompt,
            context_note,
            verbose=verbose
        )

        if verbose:
            print(f"\n--- DEBUG: Research Plan ---\n{research_plan}\n--- END PLAN ---")

        evidence = await self.agather_research_evidence(
            user_input,
            research_plan,
            verbose=verbose
        )

        if verbose:
            print(f"\n--- DEBUG: Gathered Evidence ---\n{_truncate_text(evidence, 6000)}\n--- END EVIDENCE ---")

        if _is_time_sensitive_request(user_input) and not _evidence_has_sources(evidence):
            blocked_reason = (
                "I could not verify current information from the search tools right now. "
                "DuckDuckGo MCP returned no source URLs, likely because of bot detection or a temporary search failure. "
                "No fallback provider is enabled, so I am not going to produce a current-affairs briefing from memory "
                "or unverified context."
            )
            if _is_detailed_report_request(user_input):
                yield _blocked_report_markdown(
                    user_input=user_input,
                    research_plan=research_plan,
                    evidence=evidence,
                    reason=blocked_reason,
                )
                return

            yield f"{blocked_reason}\n"
            return

        if _is_detailed_report_request(user_input):
            memory_context = (
                f"Long-term summary:\n{summary_for_prompt or '(none)'}\n\n"
                f"Recent context:\n{recent_context_for_prompt or '(none)'}"
            )
            async for content in self.agenerate_detailed_report(
                user_input=user_input,
                research_plan=research_plan,
                evidence=evidence,
                memory_context=memory_context,
                context_note=context_note,
                verbose=verbose,
            ):
                yield content
            return

        system_prompt = (
            "You are the researching part of a two-agent research system.\n"
            f"{self.system_message}\n\n"
            f"Today's date is {date.today().strftime('%B %d, %Y')}.\n\n"
            "For time-sensitive queries, use today's date above. "
            "Ignore stale dates from prior messages unless the user explicitly asks for them.\n\n"
            "Execute the provided research plan. Use `search` when the answer depends on current, recent, "
            "or externally verifiable facts. Use `fetch_content` for promising search results when snippets "
            "are not enough for detailed or high-confidence claims. Prefer reputable and current sources, "
            "cross-check major claims, and include exact dates for time-sensitive facts.\n\n"
            "Grounding rules:\n"
            "- Treat long-term summary and recent conversation context as memory, not evidence.\n"
            "- Treat the gathered search evidence below as the primary factual basis.\n"
            "- Verify prior-context factual claims before repeating them.\n"
            "- If fresh sources contradict prior context, prioritize the fresh sources and explain the correction.\n"
            "- Do not invent facts, dates, names, outcomes, statistics, or events beyond the tool results.\n"
            "- If current search results are too broad or weak, say the available results are insufficient.\n"
            "- End detailed research with a Sources section containing the URLs you relied on.\n"
            "- Mark uncertain claims as uncertain instead of presenting them as verified.\n\n"
            f"Context handling note:\n{context_note}\n\n"
            f"Long-term conversation summary:\n\n{summary_for_prompt or '(withheld for this query)'}"
        )

        inputs = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"User request:\n{user_input}\n\n"
                        f"Recent conversation context:\n{recent_context_for_prompt or '(withheld for this query)'}\n\n"
                        f"Research plan from planner:\n{research_plan}\n\n"
                        f"Gathered search evidence:\n{evidence}"
                    )
                }
            ]
        }

        answer_parts = []
        requires_cited_output = _is_time_sensitive_request(user_input)

        try:
            # MCP tools are async-only, so the agent must be driven with astream().
            async for msg, metadata in self.research_agent.astream(inputs, config=self.research_config, stream_mode="messages"):
                # Handle tool calls if verbose is enabled
                if verbose and hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        name = tool_call.get("name") or "<streaming>"
                        args = tool_call.get("args", {})
                        print(f"\n--- DEBUG: Tool Call: {name} with args: {args} ---")

                if getattr(msg, "type", None) in {"ai", "AIMessageChunk"} and msg.content:
                    text = _content_to_text(msg.content)
                    if text:
                        answer_parts.append(text)
                        if not requires_cited_output:
                            yield text
        except Exception as exc:
            if verbose:
                print(f"\n--- DEBUG: Streaming failed ({type(exc).__name__}); retrying without token streaming. ---")
            
            # FALLBACK: Use non-streaming ainvoke to get a stable result
            try:
                result = await self.research_agent.ainvoke(inputs, config=self.research_config)
                final_text = _content_to_text(result.content)
                if final_text:
                    yield final_text
            except Exception as fallback_exc:
                if verbose:
                    print(f"--- DEBUG: Fallback also failed: {fallback_exc} ---")
                raise exc from fallback_exc

            try:
                response = await self.research_agent.ainvoke(inputs, config=self.research_config)
            except Exception as fallback_exc:
                yield f"\nModel provider error: {fallback_exc}\n"
                return

            messages = response.get("messages", [])
            if messages:
                text = _content_to_text(messages[-1].content)
                if text:
                    answer_parts.append(text)
                    if not requires_cited_output:
                        yield text

        answer_text = "".join(answer_parts)
        if requires_cited_output:
            if "http://" in answer_text or "https://" in answer_text:
                yield answer_text
            else:
                yield (
                    "The researcher produced an uncited answer, so I withheld it. "
                    "For time-sensitive research, the final report must include source URLs from DuckDuckGo MCP. "
                    "No fallback provider is enabled.\n"
                )

        # After streaming is complete, we check if summarization is needed.
        state = await self.research_agent.aget_state(self.research_config)
        
        if verbose:
            # Temporarily override summarize_if_needed's internal prints by wrapping it or checking here
            print(f"--- DEBUG: Post-response Token Count: {self.count_tokens(state.values.get('messages', []))} ---")
        
        await self.asummarize_if_needed(state.values)

        # Save the final answer as a report
        if answer_text:
            _write_report_artifacts(answer_text, user_input)

    # =====================================================
    # RESET
    # =====================================================

    def clear_summary(self):
        _run_async(self.aclear_summary(), "aclear_summary()")

    async def aclear_summary(self):

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

    def close(self):
        _run_async(self.aclose(), "aclose()")

    async def aclose(self):
        await self.conn.close()


# =========================================================
# USAGE
# =========================================================

if __name__ == "__main__":
    llm=ChatOpenRouter(model='OpenRouter/free')
    async def main():
        DB_URI = require_env('DB_URL')
        llm=ChatOpenRouter(
            model=os.getenv('OPENROUTER_MODEL', 'google/gemma-4-31b-it:free'),
            api_key=require_env('OPENROUTER_API_KEY'),
            temperature=float(os.getenv('OPENROUTER_TEMPERATURE', '0.2')),
        )
        tools=await get_tools()
        agent = await PersistentMemoryAgent.create(
            llm=llm,
            db_uri=DB_URI,
            thread_id="user_1",
            tools=tools
        )
        agent.set_system_message(
            "You are a careful web research agent. Use the available tools to produce detailed, "
            "source-grounded answers that are accurate as of today's date."
        )
        
        try:
            while True:

                user_input = input("You: ")

                if user_input.lower() == "exit":
                    break

                turn_chunks = []
                try:
                    async for content in agent.astream_messages(user_input,verbose=True):
                        turn_chunks.append(content)
                        print(content,end='',flush=True)
                except Exception as exc:
                    error_text = (
                        "\n\n## Runtime Error\n\n"
                        f"The conversation turn failed before normal completion: {exc}\n"
                    )
                    turn_chunks.append(error_text)
                    print(error_text,end='',flush=True)

                if _is_detailed_report_request(user_input):
                    try:
                        markdown_path, pdf_path = _write_turn_report_artifacts(
                            user_input,
                            "".join(turn_chunks),
                        )
                        print(
                            "\n--- REPORT FILES ---\n"
                            f"Markdown: {markdown_path}\n"
                            f"PDF: {pdf_path}\n"
                            "--- END REPORT FILES ---",
                            flush=True,
                        )
                    except Exception as exc:
                        print(
                            "\n--- REPORT FILE ERROR ---\n"
                            f"Could not create report files after the conversation turn: {exc}\n"
                            "--- END REPORT FILE ERROR ---",
                            flush=True,
                        )

                print()
        finally:
            await agent.aclose()
    
    _run_async(main(), "main()")

    # asyncio.run( get_tools())
