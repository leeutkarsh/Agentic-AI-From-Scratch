# 🧠 Tutorial 2 — Give Your AI Agent Memory and Make It Safer

### LangChain + LangGraph + Ollama Cloud | Beginner to Intermediate

In **Tutorial 1** you built an agent that can think, call tools, read and write files, and loop until a task is finished.

In **Tutorial 2** we make that agent more *reliable*. Instead of piling on new tools, we upgrade the foundation you already have.

By the end of this tutorial, your agent will:

- ✅ Remember a conversation across multiple messages **and across program restarts**
- ✅ Save conversation history in a local **SQLite** database
- ✅ Read and write files **only inside a dedicated `workspace/` folder**
- ✅ Refuse to **overwrite existing files** by accident
- ✅ Support **multiple separate conversations** through simple terminal commands

We continue with the same stack: **Ollama Cloud + LangChain + LangGraph**.

---

## 📚 Table of Contents

- [Part A — The Concepts](#part-a--the-concepts) (what was missing in Tutorial 1, memory, checkpointers, safe workspace)
- [Part B — Setup](#part-b--setup) (project structure and dependencies)
- [Part C — Build the Improved Agent](#part-c--build-the-improved-agent) (the full `agent.py`)
- [Part D — Understand the New Code](#part-d--understand-the-new-code)
- [Part E — Run It!](#part-e--run-it) (seven tests)
- [Part F — Troubleshooting](#part-f--troubleshooting)
- [Part G — Important Limitations](#part-g--important-limitations)
- [Part H — What's Next: Tutorial 3](#part-h--whats-next-tutorial-3)

---

# PART A — The Concepts

## 1. The problem with your first agent

In Tutorial 1, every time you typed a question, the program ran something like this:

```python
result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
```

Look closely: each call starts with a **brand-new list containing only your latest message**. The agent never sees what you said before. So this happens:

```text
> My name is Rahul.
AI: Nice to meet you, Rahul!

> What is my name?
AI: I don't know your name.
```

This is not a bug in the model. **Language models are stateless**: every API call is independent, and the model only knows what is inside the request you send it. If you want it to "remember" the conversation, *your program* has to send the conversation history again on every call.

Tutorial 1 also had two safety problems:

1. The file tools could access **any path on your computer**, not just the project folder.
2. Writing to an existing file used `"w"` mode, which **silently erases the old contents**.

Tutorial 2 fixes all three problems.

## 2. What is "memory" in an AI agent?

There are two different ideas, and beginners often mix them up:

| Type | What it does | Example |
| --- | --- | --- |
| **Conversation memory** | Keeps the messages of one conversation so the agent can use them in later replies. | "My favorite language is Java." → later: "Which language did I mention?" |
| **Long-term knowledge memory** | Stores selected facts or documents so they can be looked up in *any* conversation. | Saving your coding preferences and retrieving them in a brand-new chat. |

In this tutorial we build the **first kind: persistent conversation memory**. "Persistent" means it survives after you close the program.

## 3. What is a checkpointer?

LangGraph agents are built as a **graph** that passes around a shared **state** (in our case, the list of messages). Think of that state as a notebook the graph writes in while it works.

A **checkpointer** saves snapshots of that notebook after each step. In a later run, LangGraph can load the saved notebook and continue where it left off.

We will use **SQLite**, a lightweight database that lives in a single local file. LangGraph's `SqliteSaver` stores checkpoints there. It is a good fit for small, synchronous projects like this one. See the [`SqliteSaver` reference](https://reference.langchain.com/python/langgraph.checkpoint.sqlite/SqliteSaver) for details.

Here is the flow:

```text
You type a message
        │
        ▼
LangGraph loads the saved state for this thread   (old messages)
        │
        ▼
New message is added to the state                 (old + new)
        │
        ▼
Agent runs:  model → tools (if needed) → model → final answer
        │
        ▼
SQLite saves the updated state
        │
        ▼
Next time, the conversation can be resumed
```

Three terms to remember:

| Term | Meaning |
| --- | --- |
| `checkpointer` | The component that saves and loads graph state. |
| `SqliteSaver` | A checkpointer that stores state in a local SQLite file. |
| `thread_id` | A label that identifies **which conversation** a saved state belongs to. |

> **Key idea:** Same `thread_id` → resume that conversation. Different `thread_id` → a separate, fresh conversation.

## 4. What is a safe workspace?

A **workspace** is the one folder your file tools are allowed to touch.

Instead of allowing paths like:

```text
C:\Users\YourName\Documents\important.txt
C:\Users\YourName\Desktop\private.txt
```

we restrict the agent to:

```text
my-first-agent/
└── workspace/
    ├── hello.txt
    └── poem.txt
```

We also stop the agent from writing over an existing file.

> ⚠️ This is an **introductory safety boundary**, not a full security sandbox. It makes the two file tools much more constrained than in Tutorial 1, but it does not make every possible action safe (more on this in Part G).

---

# PART B — Setup

## Step 1 — Update your project structure

Keep using your project from Tutorial 1. It will now look like this:

```text
my-first-agent/
├── agent.py
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
├── agent_memory.sqlite      ← created automatically
└── workspace/               ← created automatically
```

You do **not** need to create `agent_memory.sqlite` or `workspace/` yourself. The program creates them when it first runs.

## Step 2 — Install the SQLite integration

Open PowerShell in your project folder, activate your virtual environment, and install the new library:

```powershell
venv\Scripts\Activate.ps1
pip install langgraph-checkpoint-sqlite
```

This is the only new dependency for Tutorial 2.

## Step 3 — Update `requirements.txt`

```text
langchain-core
langchain-openai
langgraph
langgraph-checkpoint-sqlite
python-dotenv
```

Anyone cloning your project can then install everything with:

```powershell
pip install -r requirements.txt
```

## Step 4 — Check your `.env` and `.env.example`

Your real settings stay in `.env`:

```text
OLLAMA_API_KEY=your_real_api_key
OLLAMA_MODEL=your_working_tool_capable_model
OLLAMA_API_URL=https://ollama.com/v1
```

Use the same model that worked in Tutorial 1. It **must support tool calling**.

Your `.env.example` is the *public template* and must contain **placeholders only**:

```text
OLLAMA_API_KEY=your_api_key_here
OLLAMA_MODEL=your_tool_capable_model_name
OLLAMA_API_URL=https://ollama.com/v1
```

> 🔐 Never put your real API key in `.env.example`, and never commit `.env` to GitHub.

## Step 5 — Update `.gitignore`

```text
.env
venv/
__pycache__/
agent_memory.sqlite
agent_memory.sqlite-shm
agent_memory.sqlite-wal
workspace/
```

This keeps your API key, your local conversation database, and your practice files out of Git commits. (The `-shm` and `-wal` files are temporary helper files SQLite may create next to the database.)

---

# PART C — Build the Improved Agent

Replace the contents of `agent.py` with the complete code below. It is the **same architecture as Tutorial 1** (a model node, a tools node, and a loop between them), with memory and safety added. Read it once top to bottom; Part D explains every new piece.

## `agent.py`

```python
import os
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# ---------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("OLLAMA_API_KEY")
API_URL = os.getenv("OLLAMA_API_URL", "https://ollama.com/v1")
MODEL = os.getenv("OLLAMA_MODEL")

if not API_KEY or not MODEL:
    raise ValueError("Check OLLAMA_API_KEY and OLLAMA_MODEL in your .env file")

WORKSPACE = (BASE_DIR / "workspace").resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = str(BASE_DIR / "agent_memory.sqlite")
MAX_FILE_BYTES = 200_000

SYSTEM_PROMPT = (
    "You are a helpful assistant with two tools: read_file and write_file. "
    "Both tools only work inside a workspace folder, so use plain filenames "
    "such as 'notes.txt'. write_file cannot overwrite an existing file; if it "
    "reports that a file exists, tell the user and suggest a new filename."
)

llm = ChatOpenAI(
    base_url=API_URL,
    api_key=API_KEY,
    model=MODEL,
    temperature=0,
)


# ---------------------------------------------------------------
# 2. Safety helper
# ---------------------------------------------------------------
def safe_path(filename: str) -> Path:
    """Turn a filename into a full path, but only if it stays inside WORKSPACE."""
    path = (WORKSPACE / filename).resolve()
    if not path.is_relative_to(WORKSPACE):
        raise ValueError("Access denied: path is outside the workspace.")
    return path


# ---------------------------------------------------------------
# 3. Tools
# ---------------------------------------------------------------
@tool
def read_file(filename: str) -> str:
    """Read a UTF-8 text file from the workspace. Example filename: notes.txt"""
    try:
        path = safe_path(filename)

        if not path.is_file():
            return f"Error: '{filename}' does not exist in the workspace."

        if path.stat().st_size > MAX_FILE_BYTES:
            return f"Error: '{filename}' is larger than {MAX_FILE_BYTES} bytes."

        return path.read_text(encoding="utf-8")

    except UnicodeDecodeError:
        return f"Error: '{filename}' is not a UTF-8 text file."
    except Exception as error:
        return f"Error: {error}"


@tool
def write_file(filename: str, content: str) -> str:
    """Create a NEW text file in the workspace. Existing files are never overwritten."""
    try:
        path = safe_path(filename)

        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            return f"Error: content is larger than {MAX_FILE_BYTES} bytes."

        path.parent.mkdir(parents=True, exist_ok=True)

        # Mode "x" = create only. Raises FileExistsError if the file exists.
        with path.open("x", encoding="utf-8") as file:
            file.write(content)

        return f"Created '{filename}' in the workspace."

    except FileExistsError:
        return f"Error: '{filename}' already exists and will not be overwritten. Use a new filename."
    except Exception as error:
        return f"Error: {error}"


tools = [read_file, write_file]
llm_with_tools = llm.bind_tools(tools)


# ---------------------------------------------------------------
# 4. The graph (same shape as Tutorial 1)
# ---------------------------------------------------------------
def ask(state: MessagesState):
    """Call the model with the system prompt plus the saved conversation."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("ask", ask)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "ask")
builder.add_conditional_edges("ask", tools_condition)  # tool call? -> "tools", else finish
builder.add_edge("tools", "ask")                       # after a tool runs, go back to the model


# ---------------------------------------------------------------
# 5. Terminal chat loop
# ---------------------------------------------------------------
def chat(agent):
    thread_id = "main"

    print(f"Workspace: {WORKSPACE}")
    print("Commands: /new  |  /thread <id>  |  exit")

    while True:
        prompt = input(f"\n[{thread_id}] > ").strip()

        if not prompt:
            continue

        if prompt.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if prompt == "/new":
            thread_id = str(uuid4())
            print(f"Started a new conversation: {thread_id}")
            continue

        if prompt.startswith("/thread"):
            parts = prompt.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: /thread <thread_id>")
            else:
                thread_id = parts[1].strip()
                print(f"Switched to conversation: {thread_id}")
            continue

        config = {"configurable": {"thread_id": thread_id}}

        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
            )
            print(f"\nAI: {result['messages'][-1].content}")
        except Exception as error:
            print(f"\nSomething went wrong: {error}")


def main():
    # The "with" block keeps the database connection open while you chat
    # and closes it cleanly when the program ends.
    with SqliteSaver.from_conn_string(DATABASE_PATH) as checkpointer:
        agent = builder.compile(checkpointer=checkpointer)
        chat(agent)


if __name__ == "__main__":
    main()
```

> 💡 Notice there are **no classes** here. Everything is plain functions, which keeps the code easy to read while you are learning.

---

# PART D — Understand the New Code

You already know `@tool`, `bind_tools`, `StateGraph`, `ToolNode`, `ask`, and `tools_condition` from Tutorial 1. Let's focus on what is new.

## 1. `BASE_DIR` and `WORKSPACE`

```python
BASE_DIR = Path(__file__).resolve().parent
```

`__file__` is the path of `agent.py`. `.resolve().parent` turns that into the **folder containing the script**.

```python
WORKSPACE = (BASE_DIR / "workspace").resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)
```

This builds the workspace path and creates the folder if it doesn't exist yet (`exist_ok=True` means "don't complain if it's already there").

Why base everything on `BASE_DIR`? Because it doesn't matter which folder PowerShell is currently in. The workspace and database always end up next to `agent.py`.

## 2. The `safe_path()` function

This is one of the most important additions. Every file tool passes its filename through it.

Suppose the model asks for `hello.txt`:

```text
WORKSPACE / "hello.txt"  →  my-first-agent/workspace/hello.txt   ✅ allowed
```

Now suppose it asks for `../agent.py`. The `..` means "go up one folder":

```text
WORKSPACE / "../agent.py"  →  my-first-agent/agent.py   ❌ outside the workspace
```

Two steps catch this:

1. **`resolve()`** cleans up the path: it removes `..`, follows symbolic links, and gives the real final location.
2. **`is_relative_to(WORKSPACE)`** checks that the final location is still inside the workspace. If not, we raise a `ValueError`.

Absolute paths such as `C:\Windows\notepad.exe` are also rejected, because they resolve to a location outside the workspace.

The tool catches that `ValueError` and **returns it as text** so the model can read the error and explain it to you instead of crashing the program.

> `is_relative_to` requires Python 3.9 or newer.

## 3. Why file writing changed from `"w"` to `"x"`

Tutorial 1 used:

```python
with open(file_path, "w", encoding="utf-8") as file:
    file.write(content)
```

The problem is mode `"w"`. If the file already exists, opening it with `"w"` **empties it first**. One careless request and your data is gone.

Tutorial 2 uses:

```python
with path.open("x", encoding="utf-8") as file:
    file.write(content)
```

Mode `"x"` means *exclusive creation*: it creates a new file, but raises `FileExistsError` if the file is already there. The tool catches that error and tells the model to pick another filename.

For now, the agent must use a **new** filename to create another file. Safe editing with human approval comes in the next tutorial.

## 4. `MAX_FILE_BYTES`

```python
MAX_FILE_BYTES = 200_000
```

This limits how large a file the agent will read or write. Large files create huge prompts, cost more tokens, and may exceed the model's context window.

We check `path.stat().st_size` **before** reading, so a huge file is never loaded into memory. It is a basic size guard, not a full token-budget system.

> 🔒 Remember: when `read_file` returns text, that text is sent to Ollama Cloud as part of the model request. **Keep secrets and sensitive files out of the workspace.**

## 5. The system prompt

```python
messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
```

A **system prompt** gives the model standing instructions. We add it in front of the saved messages on every call, but we do **not** store it in the state. That way the checkpointer saves only the real conversation, and the system prompt never gets duplicated in the history.

## 6. `MessagesState` — why history can grow

`MessagesState` is a ready-made state with one field, `messages`. Its special behavior: when a node returns `{"messages": [new_message]}`, LangGraph **appends** the new message to the list instead of replacing it.

This is why `ask()` returns only the new response, and the history keeps growing naturally: your messages, the model's replies, tool calls, and tool results all end up in one list.

## 7. The most important change: `SqliteSaver`

```python
with SqliteSaver.from_conn_string(DATABASE_PATH) as checkpointer:
    agent = builder.compile(checkpointer=checkpointer)
```

In Tutorial 1 you compiled the graph with:

```python
agent = builder.compile()
```

Now we pass in a **checkpointer**. After each step, LangGraph saves the state to `agent_memory.sqlite`, and on the next call it can restore the history for the chosen conversation.

The `with` block (a *context manager*) keeps the database connection open while the chat loop runs and closes it automatically when the program exits.

## 8. Why every call needs a `thread_id`

```python
config = {"configurable": {"thread_id": thread_id}}

result = agent.invoke(
    {"messages": [HumanMessage(content=prompt)]},
    config=config,
)
```

Two things are happening here:

1. The first argument is **only the newest message**.
2. `config` tells LangGraph **which saved conversation to load** and continue.

Because `MessagesState` appends new messages and the checkpointer restores the saved ones for that `thread_id`, the model receives the whole conversation. That's what turns separate terminal inputs into one continuing chat.

> If you forget to pass `config`, a checkpointer-enabled graph raises an error, because it doesn't know which thread to use.

## 9. Why we added `/new` and `/thread`

```python
thread_id = str(uuid4())
```

`uuid4()` generates a random, unique ID, perfect for naming a new conversation.

- `/new` switches to a fresh `thread_id`. The old conversation is **not deleted**; it's still in the database.
- `/thread <id>` switches back to a saved conversation by its ID.

Example:

```text
[main] > /new
Started a new conversation: 3f2b8c1e-....

[3f2b8c1e-....] > My favorite programming language is Java.
```

You can exit, restart, and come back with:

```text
[main] > /thread 3f2b8c1e-....
```

Use the real ID printed by your program, not the example above.

> **Memory rule:** `main` is the default thread each time the program starts. To resume any other thread, you must enter its ID yourself.

---

# PART E — Run It!

Activate your virtual environment and run:

```powershell
python agent.py
```

You should see the workspace path, the list of commands, and a prompt like `[main] >`.

> ℹ️ `input()` reads **one line at a time**, so type each test below as a single line.

## Test 1 — Create a file

```text
[main] > Create a file named hello.txt containing the text: Hello from Tutorial 2!
```

**Expected:** the agent calls `write_file`, and `hello.txt` appears inside the `workspace` folder.

## Test 2 — Read the file

```text
[main] > Read hello.txt and tell me what it contains.
```

**Expected:** the agent calls `read_file` and reports the saved text.

## Test 3 — Check conversation memory

```text
[main] > My favorite programming language is Java.
[main] > Which programming language did I say I like?
```

**Expected:** the agent answers **Java**, because both messages belong to the same thread.

## Test 4 — Check persistence across restarts

1. Tell the agent your favorite programming language.
2. Type `exit`.
3. Run `python agent.py` again.
4. Ask which language you mentioned.

**Expected:** the agent remembers, because the default thread is `main` and its history was saved in `agent_memory.sqlite`.

## Test 5 — Start a separate conversation

```text
[main] > /new
```

Tell this new thread a *different* fact, then ask about it. Then ask about the Java fact: the new thread should **not** know it. Switch back with `/thread <id>`, using the ID of your old thread (`main` for the first one).

**Expected:** each thread has its own separate history.

## Test 6 — Try to escape the workspace

```text
[main] > Use read_file to read ../agent.py
```

**Expected:** the tool rejects the path and your Python program is **not** read.

## Test 7 — Check overwrite protection

Create `hello.txt` first, then ask the agent to create it again with different content.

**Expected:** the second attempt returns an error saying the file already exists. Open `workspace/hello.txt` yourself to confirm the original text is unchanged.

> The exact wording of replies depends on your model, so always verify against the **actual files** in your `workspace` folder.

---

# PART F — Troubleshooting

| Problem | What to check |
| --- | --- |
| `ModuleNotFoundError: langgraph.checkpoint.sqlite` | Activate your virtual environment and run `pip install langgraph-checkpoint-sqlite`. |
| The model forgets previous messages | Make sure every call uses the same `thread_id` and that you didn't switch with `/new`. |
| Memory disappears after restarting | Make sure `agent_memory.sqlite` is still in your project folder and you aren't deleting it. Remember: only `main` is resumed automatically. |
| `401 Unauthorized` or model errors | Recheck the API key, base URL, and model name that worked in Tutorial 1. |
| The model never calls a tool | Your model may not support tool calling. Try a tool-capable model. |
| A file request is rejected | The path must resolve **inside** the `workspace` folder. |
| The agent can't write to an existing file | This is intentional. Choose a new filename. |
| One thread keeps failing with a tool-call error | If a run was interrupted mid-tool-call, that thread's saved history can be left incomplete. Start a fresh one with `/new`. |
| Long conversations get slower or fail | Saved history grows with every turn. Start a new thread for new topics. |

---

# PART G — Important Limitations

This is a much better foundation, but it is **not production-ready**. Know its limits:

1. **Saved conversation ≠ permanent knowledge.** The checkpointer stores the *message history* of each thread. It does not turn useful facts into a searchable knowledge base shared by all conversations.

2. **History keeps growing.** Every turn adds messages, including tool calls and tool outputs. Eventually you will need to trim or summarize long conversations.

3. **Workspace limits don't make everything safe.** They protect only these two file tools. Any tool you add later needs its own permission checks. Also, because the model runs in the cloud, **anything the agent reads may leave your computer**.

4. **There is no human approval yet.** The agent *rejects* overwrites instead of asking you. A better design pauses before risky actions, shows you what's about to happen, and continues only after you approve. That is exactly what Tutorial 3 builds.

---

# PART H — What's Next: Tutorial 3

## Tutorial 3 — Human-in-the-Loop Approval

Right now your agent either does something or refuses outright. Real agents often need a third option: **stop and ask you first**.

In Tutorial 3 you will learn how to:

- Pause a running graph with LangGraph's `interrupt()`
- Show the human exactly what the agent wants to do (for example, "edit `notes.txt`")
- Resume the same run with the human's decision using `Command(resume=...)`
- Let the agent edit existing files, but **only after you approve**

And here's why Tutorial 2 matters: pausing and resuming a graph needs saved state and a `thread_id`. The checkpointer and thread system you just built are the foundation for it.

## 🎯 Your learning target

After Tutorial 1, you could connect an LLM to tools. After Tutorial 2, you can manage three things every dependable agent needs: **state, persistence, and permissions**.

Build these upgrades into the **same project** instead of starting over each time. By Tutorial 3, you will have an agent that remembers its conversation, stays inside its workspace, and asks for approval before doing anything sensitive.

---

⭐ If this tutorial helped you, consider starring the repository, and happy building!
