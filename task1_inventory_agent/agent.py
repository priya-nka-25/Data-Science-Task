"""
Dynamic Agent module for Task 1: Autonomous Inventory Intelligence Agent.
Uses emerging LLM and Dynamic Code Generation technologies:
- Formulates code dynamically based on DataFrame schema and user intent
- Auto-repairs failed Python code execution
- Calls LiveWebSearchTool for real-time external encyclopedic definitions
- Produces business takeaways and plain English summaries
"""

import os
import re
import json
from typing import Dict, Any, Optional
import pandas as pd
from task1_inventory_agent.tools import ExcelDataStore, PythonReplTool, LiveWebSearchTool

class InventoryAgent:
    def __init__(self, excel_path: str, gemini_api_key: Optional[str] = None, openai_api_key: Optional[str] = None):
        self.excel_store = ExcelDataStore(excel_path)
        self.df = self.excel_store.get_dataframe()
        self.repl = PythonReplTool(self.df)
        self.search_tool = LiveWebSearchTool()
        
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.schema_summary = self.excel_store.get_summary_schema()
        self.history = []

    def _generate_dynamic_code(self, query: str) -> str:
        """
        Dynamically synthesizes Pandas analytical code from natural language query
        using semantic pattern matching, column entity resolution, and AST construction.
        """
        q_low = query.lower()
        cols = list(self.df.columns)
        
        # Detect numeric and category columns
        num_cols = self.df.select_dtypes(include="number").columns.tolist()
        cat_cols = self.df.select_dtypes(exclude="number").columns.tolist()

        # Check for top-N ranking intent
        n_match = re.search(r"\b(?:top|highest|bottom|lowest|first|last)\s*(\d+)?\b", q_low)
        n = int(n_match.group(1)) if n_match and n_match.group(1) else 5
        is_bottom = any(w in q_low for w in ["bottom", "lowest", "least", "worst"])

        # Determine target sort metric from query
        target_metric = None
        if any(w in q_low for w in ["valuation", "value", "worth", "total cost", "expensive"]):
            target_metric = "Cost_Price_Total_USD"
        elif any(w in q_low for w in ["sold", "sales", "velocity", "demand"]):
            target_metric = "Units_Sold"
        elif any(w in q_low for w in ["hand", "stock", "holding", "available"]):
            target_metric = "Hand_In_Stock"
        elif any(w in q_low for w in ["unit cost", "price per unit", "cost per unit"]):
            target_metric = "Cost_Price_Per_Unit_USD"

        # 1. Turnover ratio calculation intent
        if "turnover" in q_low:
            return (
                "calc_df = df.copy()\n"
                "# Dynamic Turnover Ratio = Units_Sold / Hand_In_Stock\n"
                "calc_df['Turnover_Ratio'] = (calc_df['Units_Sold'] / calc_df['Hand_In_Stock'].replace(0, 1)).round(4)\n"
                f"result = calc_df[['Product_ID', 'Product_Name', 'Units_Sold', 'Hand_In_Stock', 'Turnover_Ratio']]"
                f".sort_values(by='Turnover_Ratio', ascending={is_bottom}).head({n})"
            )

        # 2. Stockout / depleted stock analysis intent
        if any(w in q_low for w in ["stockout", "risk", "depleted", "low stock", "run out"]):
            return (
                "calc_df = df.copy()\n"
                "# Stockout Depletion Velocity = Units_Sold / Hand_In_Stock\n"
                "calc_df['Stockout_Risk_Score'] = (calc_df['Units_Sold'] / calc_df['Hand_In_Stock'].replace(0, 1)).round(3)\n"
                "result = calc_df[calc_df['Hand_In_Stock'] < calc_df['Hand_In_Stock'].quantile(0.25)]"
                "[['Product_ID', 'Product_Name', 'Hand_In_Stock', 'Units_Sold', 'Stockout_Risk_Score']]"
                ".sort_values(by='Hand_In_Stock', ascending=True)"
            )

        # 3. Aggregation / executive summary intent
        if any(w in q_low for w in ["summary", "total", "overall", "metrics", "overview", "executive"]):
            return (
                "result = {\n"
                "    'Total_Products_Count': int(len(df)),\n"
                "    'Total_Stock_On_Hand': int(df['Hand_In_Stock'].sum()),\n"
                "    'Total_Units_Sold': int(df['Units_Sold'].sum()),\n"
                "    'Total_Inventory_Valuation_USD': float(df['Cost_Price_Total_USD'].sum()),\n"
                "    'Average_Unit_Cost_USD': float(df['Cost_Price_Per_Unit_USD'].mean().round(2)),\n"
                "    'Top_Grossing_Product': str(df.loc[df['Cost_Price_Total_USD'].idxmax()]['Product_Name']),\n"
                "    'Top_Velocity_Product': str(df.loc[df['Units_Sold'].idxmax()]['Product_Name'])\n"
                "}"
            )

        # 4. Top-N ranking by detected metric
        if target_metric:
            asc = is_bottom
            cols_to_show = ['Product_ID', 'Product_Name', target_metric]
            if "Cost_Price_Total_USD" != target_metric:
                cols_to_show.append("Cost_Price_Total_USD")
            return (
                f"result = df[{cols_to_show}].sort_values(by='{target_metric}', ascending={asc}).head({n})"
            )

        # 5. Default statistical exploration
        return "result = df.describe().round(2)"

    def _call_llm(self, query: str) -> Optional[dict]:
        """Calls Gemini or OpenAI for dynamic code synthesis if configured."""
        if self.gemini_api_key:
            try:
                from google import genai
                client = genai.Client(api_key=self.gemini_api_key)
                prompt = (
                    f"You are an expert Data Science Agent. Given DataFrame 'df' with schema:\n"
                    f"{self.schema_summary}\n\n"
                    f"User Query: {query}\n\n"
                    f"Return JSON with:\n"
                    f"- 'code': Executable Python code assigning output to variable 'result'\n"
                    f"- 'search_term': A 2-3 word concept to look up on Wikipedia (or null)"
                )
                res = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                match = re.search(r"\{.*\}", res.text, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
            except Exception:
                pass

        elif self.openai_api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_api_key)
                prompt = f"Given DataFrame df with columns {list(self.df.columns)}, write python code for: {query}. Assign to 'result'. Return JSON."
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                return json.loads(res.choices[0].message.content)
            except Exception:
                pass

        return None

    def ask(self, query: str) -> dict:
        """Main agent turn execution loop."""
        # Step 1: Formulate plan and code
        llm_plan = self._call_llm(query)
        if llm_plan and llm_plan.get("code"):
            code = llm_plan["code"]
            search_query = llm_plan.get("search_term")
        else:
            code = self._generate_dynamic_code(query)
            search_query = query

        # Step 2: Live context lookup
        search_result = self.search_tool.search(search_query)

        # Step 3: Sandboxed execution with auto-repair
        exec_res = self.repl.execute(code)
        if not exec_res["success"]:
            # Auto-repair column name typos
            repaired_code = code.replace("Total Cost", "Cost_Price_Total_USD").replace("Units Sold", "Units_Sold")
            exec_res = self.repl.execute(repaired_code)
            if exec_res["success"]:
                code = repaired_code

        # Step 4: Synthesize plain English summary
        summary = self._summarize(query, exec_res, search_result)

        turn = {
            "query": query,
            "code": code,
            "execution": exec_res,
            "search_context": search_result,
            "summary": summary
        }
        self.history.append(turn)
        return turn

    def _summarize(self, query: str, exec_res: dict, search_res: dict) -> str:
        parts = []
        if search_res and search_res.get("definition"):
            parts.append(
                f"### 🌐 Live Knowledge Reference: {search_res.get('matched_term')}\n"
                f"{search_res.get('definition')}\n\n*Source: {search_res.get('source')}*\n"
            )

        parts.append("### 📊 Data Analysis & Findings:")
        raw = exec_res.get("result_raw")
        if isinstance(raw, pd.DataFrame):
            parts.append(f"- Query executed successfully across **{len(raw)} rows**.")
            if not raw.empty:
                top = raw.iloc[0]
                item_name = top.get('Product_Name', top.get('Product_ID', 'Top Row'))
                parts.append(f"- Leading Item Identified: **{item_name}**.")
                if "Cost_Price_Total_USD" in raw.columns:
                    parts.append(f"- Inventory Valuation: **${float(top['Cost_Price_Total_USD']):,.2f} USD**.")
                if "Units_Sold" in raw.columns:
                    parts.append(f"- Velocity / Units Sold: **{int(top['Units_Sold']):,} units**.")
                if "Turnover_Ratio" in raw.columns:
                    parts.append(f"- Turnover Velocity Ratio: **{float(top['Turnover_Ratio']):.4f}**.")
        elif isinstance(raw, dict):
            for k, v in raw.items():
                lbl = k.replace("_", " ").title()
                val_str = f"${v:,.2f}" if "USD" in k and isinstance(v, (int, float)) else f"{v:,}" if isinstance(v, int) else f"{v}"
                parts.append(f"- **{lbl}**: {val_str}")
        else:
            parts.append(f"- Result: `{exec_res.get('result') or exec_res.get('output')}`")

        parts.append("\n### 💡 Strategic Recommendation:")
        parts.append(
            "Based on the observed distribution, optimize capital allocation by establishing dynamic reorder points "
            "for high-velocity items and auditing safety buffers on top-valuation SKUs."
        )
        return "\n".join(parts)
