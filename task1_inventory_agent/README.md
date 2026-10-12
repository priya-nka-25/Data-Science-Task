# Task 1: Autonomous Inventory Intelligence Agent

## 📌 Problem Statement
You are provided with an Excel spreadsheet containing inventory records (`Inventory-Records-Sample-Data.xlsx`). The objective is to build an autonomous agent that answers natural language questions about the inventory data. The agent must:
1. **Write and execute code** (Python/Pandas) dynamically in a secure sandboxed environment to query and analyze the data.
2. **Look up definitions or domain context** (e.g., Inventory Turnover Ratio, Safety Stock, Reorder Point, Holding Cost) using an integrated search tool.
3. **Summarise findings in plain English** with actionable business takeaways.
4. **Provide an interactive chat interface** where users can chat with the dataset.

---

## 🏗️ Solution Workflow & Architecture

```
User Query (e.g. "What products have high turnover and what does that mean?")
                      │
                      ▼
             ┌──────────────────┐
             │  ReAct Agent /   │◄─── Schema Summary & Column Mapping
             │ Semantic Planner │
             └────────┬─────────┘
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
┌──────────────────┐     ┌──────────────────────┐
│  Python REPL     │     │ Context Search Tool  │
│  Execution       │     │ (Supply Chain KB)    │
│  (Sandboxed)     │     │                      │
└────────┬─────────┘     └──────────┬───────────┘
         │                          │
         │ (DataFrame Result)       │ (Operational Definition)
         └────────────┬─────────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ Plain-English    │
             │ Summarizer       │
             └────────┬─────────┘
                      │
                      ▼
         Streamlit UI / CLI Output
  (Summary + Plotly Chart + Code + Definition)
```

### Key Modules:
- `tools.py`:
  - `ExcelDataStore`: Cleans header rows, normalizes column names, and validates data types.
  - `PythonReplTool`: Sandboxed environment providing `df`, `pd`, and `np` with auto-repair retry logic.
  - `ContextSearchTool`: Knowledge base covering key inventory metrics (Turnover, Safety Stock, EOQ, Stockout Risk, GMROI, etc.).
- `agent.py`:
  - `InventoryAgent`: Coordinates query decomposition, code planning, execution, and synthesis. Supports online LLMs (Gemini / OpenAI) and offline deterministic planning.
- `app.py`:
  - Streamlit chat UI featuring conversational memory, data preview metrics, interactive Plotly charts, code inspection, and definition cards.
- `cli_agent.py`:
  - Command-line runner for headless batch testing.

---

## 🚀 How to Run

### Option 1: Interactive Streamlit Web Interface
Run the web application:
```bash
streamlit run task1_inventory_agent/app.py
```
Open your browser at `http://localhost:8501`.

### Option 2: Command-Line Interface (CLI Demo)
Run the automated test suite across key inventory scenarios:
```bash
python task1_inventory_agent/cli_agent.py
```

### Configuration (Optional)
To use online LLM mode:
- Set environment variable: `set GEMINI_API_KEY="your_api_key"` or `set OPENAI_API_KEY="your_api_key"`
- Or input your key directly into the Streamlit sidebar.
*Note: If no API key is provided, the agent runs in offline mode using its built-in semantic planner.*

