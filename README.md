# 📝 Blog Writing Agent

An AI-powered, multi-stage blog writing pipeline built with **LangGraph**, **Groq LLMs**, and **Tavily Search**. Give it a topic, and it autonomously researches, plans, writes, and assembles a complete, publication-ready technical blog post in Markdown.

---

## ✨ Features

- **Intelligent Routing** — Automatically classifies topics as `closed_book`, `hybrid`, or `open_book` to decide if web research is needed before writing.
- **Live Web Research** — Uses [Tavily Search](https://tavily.com/) to gather fresh, relevant evidence with recency filtering and deduplication.
- **Structured Planning** — An orchestrator node produces a detailed outline with goals, bullet points, word targets, and metadata for each section.
- **Parallel Section Writing** — Worker nodes write individual sections concurrently via LangGraph's fan-out/fan-in pattern, keeping each section focused and on-topic.
- **Grounded & Cited Output** — Research-backed claims are automatically cited with source URLs; the system enforces strict grounding policies per mode.
- **Markdown Export** — The reducer assembles all sections in order, adds the title, and saves the final blog as a `.md` file in the `outputs/` directory.

---

## 🏗️ Architecture

The agent is structured as a **LangGraph StateGraph** with the following nodes:

```
START → Router → [Research] → Orchestrator → Worker(s) → Reducer → END
```

| Node             | Role                                                                 |
| ---------------- | -------------------------------------------------------------------- |
| **Router**       | Decides whether the topic needs web research and selects the mode.   |
| **Research**     | Runs Tavily searches, synthesizes evidence, filters by recency.      |
| **Orchestrator** | Generates a structured `Plan` with 2–5 sections (tasks).             |
| **Worker**       | Writes one section of the blog using the plan and evidence.          |
| **Reducer**      | Merges all sections in order, adds the title, and saves to disk.     |

### Routing Modes

| Mode          | Research? | Description                                                      |
| ------------- | --------- | ---------------------------------------------------------------- |
| `closed_book` | No        | Evergreen topics (concepts, fundamentals) — no external sources. |
| `hybrid`      | Yes       | Mostly evergreen but needs fresh examples, tools, or models.     |
| `open_book`   | Yes       | Volatile/time-sensitive topics (news roundups, rankings, etc.).  |

---

## 📂 Project Structure

```
Blog Agent/
├── blog-agent.ipynb    # Main notebook — contains the full pipeline
├── outputs/            # Generated blog posts (Markdown files)
├── .env                # API keys (GROQ_API_KEY, TAVILY_API_KEY)
├── .gitignore
├── myenv/              # Python virtual environment
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- A [Groq](https://console.groq.com/) API key
- A [Tavily](https://tavily.com/) API key

### 1. Clone the Repository

```bash
git clone https://github.com/Qhmad-art/blog-written-agent.git
cd blog-written-agent
```

### 2. Create & Activate a Virtual Environment

```bash
python -m venv myenv

# Windows
myenv\Scripts\activate

# macOS / Linux
source myenv/bin/activate
```

### 3. Install Dependencies

```bash
pip install langchain-groq langchain-community langgraph tavily-python python-dotenv pydantic
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 5. Run the Notebook

Open `blog-agent.ipynb` in Jupyter Notebook or VS Code and run all cells. To generate a blog post, invoke the graph:

```python
result = app.invoke({
    "topic": "application of machine learning",
    "as_of": "2026-09-17"
})
```

The generated blog will be saved as a Markdown file in the `outputs/` directory.

---

## ⚙️ Configuration

| Parameter | Type   | Description                                            |
| --------- | ------ | ------------------------------------------------------ |
| `topic`   | `str`  | The subject of the blog post.                          |
| `as_of`   | `str`  | ISO date (`YYYY-MM-DD`) used for recency calculations. |

The **LLM model** is configured inside the notebook:

```python
llm = ChatGroq(
    model="openai/gpt-oss-safeguard-20b",
    temperature=0,
)
```

You can swap to any model available on [Groq](https://console.groq.com/docs/models).

---

## 📄 Sample Output

The agent has generated blogs such as:

- *Demystifying Machine Learning: A Beginner's Guide*
- *Mastering Machine Learning: A Practical Guide for Developers*
- *From Model to Production: A Practical Guide to Deploying Machine Learning Services*

Each output is a fully structured Markdown document with headings, tables, code snippets, and cited sources where applicable.

---

## 🧩 Key Technologies

| Technology                                                           | Purpose                          |
| -------------------------------------------------------------------- | -------------------------------- |
| [LangGraph](https://github.com/langchain-ai/langgraph)              | Stateful, graph-based agent orchestration |
| [LangChain](https://www.langchain.com/)                             | LLM abstraction & prompt management       |
| [Groq](https://groq.com/)                                           | Ultra-fast LLM inference                  |
| [Tavily](https://tavily.com/)                                       | AI-optimized web search API               |
| [Pydantic](https://docs.pydantic.dev/)                              | Structured output schemas                 |

---

## 📜 License

This project is open source. Feel free to use, modify, and distribute.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.
