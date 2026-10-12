"""
Tools module for Task 1 Inventory Agent.
Provides:
1. ExcelDataStore - Loads and normalizes the inventory dataset
2. PythonReplTool - Sandboxed Python REPL with pandas and plotting capabilities
3. ContextSearchTool - Supply chain, inventory management and business ontology lookup
"""

import sys
import io
import re
import traceback
import pandas as pd
import numpy as np

class ExcelDataStore:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = self._load_data(file_path)

    def _load_data(self, path: str) -> pd.DataFrame:
        """
        Loads the inventory excel sheet, handling header offsets and formatting.
        """
        raw = pd.read_excel(path, header=None)
        
        # Detect header row containing 'Product ID' or 'Product Name'
        header_row_idx = None
        for i in range(min(15, len(raw))):
            row_str = " ".join([str(val) for val in raw.iloc[i].dropna()])
            if "Product ID" in row_str or "Product Name" in row_str:
                header_row_idx = i
                break
        
        if header_row_idx is None:
            header_row_idx = 0

        df = pd.read_excel(path, header=header_row_idx)
        
        # Drop columns that are completely null or 'Unnamed: 0'
        cols_to_drop = [c for c in df.columns if str(c).startswith("Unnamed") and df[c].isna().all()]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # Standardize column names
        cleaned_cols = []
        for col in df.columns:
            c = str(col).replace("\n", " ").strip()
            # Normalize common column variations
            c_norm = re.sub(r"\s+", "_", c)
            c_norm = re.sub(r"[()\/]", "_", c_norm).strip("_")
            cleaned_cols.append(c_norm)
        
        df.columns = cleaned_cols
        
        # Clean up column names specifically for sample inventory dataset
        col_rename_map = {}
        for c in df.columns:
            clow = c.lower()
            if "product_id" in clow:
                col_rename_map[c] = "Product_ID"
            elif "product_name" in clow:
                col_rename_map[c] = "Product_Name"
            elif "opening" in clow:
                col_rename_map[c] = "Opening_Stock"
            elif "purchase" in clow or "stock_in" in clow:
                col_rename_map[c] = "Purchase_Stock_In"
            elif "sold" in clow:
                col_rename_map[c] = "Units_Sold"
            elif "hand" in clow:
                col_rename_map[c] = "Hand_In_Stock"
            elif "cost_price" in clow and "per_unit" in clow:
                col_rename_map[c] = "Cost_Price_Per_Unit_USD"
            elif "cost_price" in clow and "total" in clow:
                col_rename_map[c] = "Cost_Price_Total_USD"

        df = df.rename(columns=col_rename_map)
        
        # Ensure numeric columns are properly typed
        numeric_cols = [
            "Opening_Stock", "Purchase_Stock_In", "Units_Sold", 
            "Hand_In_Stock", "Cost_Price_Per_Unit_USD", "Cost_Price_Total_USD"
        ]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
                
        # Drop rows where Product_ID or Product_Name is missing
        if "Product_ID" in df.columns:
            df = df.dropna(subset=["Product_ID"]).reset_index(drop=True)
            
        return df

    def get_dataframe(self) -> pd.DataFrame:
        return self.df.copy()

    def get_summary_schema(self) -> str:
        """Returns schema and summary statistics for prompt context."""
        summary = []
        summary.append(f"Total Records: {len(self.df)}")
        summary.append(f"Columns: {list(self.df.columns)}")
        summary.append("\nFirst 3 rows sample:")
        summary.append(self.df.head(3).to_string())
        summary.append("\nStatistical Summary:")
        summary.append(self.df.describe().to_string())
        return "\n".join(summary)


class PythonReplTool:
    """
    Sandboxed Python REPL for safe execution of pandas queries and analytics.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def execute(self, code_str: str) -> dict:
        """
        Executes code string with 'df' provided in scope.
        Returns dict with keys: 'success', 'output', 'result', 'error'.
        """
        # Block dangerous system primitives
        forbidden = ["os.system", "subprocess", "shutil.rmtree", "__import__('os')", "open(", "eval(", "exec("]
        for term in forbidden:
            if term in code_str:
                return {
                    "success": False,
                    "output": "",
                    "result": None,
                    "error": f"Security restriction: Execution of '{term}' is disallowed."
                }

        local_scope = {
            "df": self.df.copy(),
            "pd": pd,
            "np": np
        }

        # Clean markdown code blocks if present
        cleaned_code = re.sub(r"^```python\s*", "", code_str, flags=re.MULTILINE)
        cleaned_code = re.sub(r"^```\s*", "", cleaned_code, flags=re.MULTILINE)
        cleaned_code = cleaned_code.strip()

        stdout_capture = io.StringIO()
        old_stdout = sys.stdout

        try:
            sys.stdout = stdout_capture
            
            # Execute entire block
            exec(cleaned_code, {}, local_scope)
            
            # Check if 'result' was explicitly set in local_scope
            result = local_scope.get("result", None)

            # If 'result' was not explicitly set, try evaluating last line as expression
            if result is None:
                lines = [line.strip() for line in cleaned_code.splitlines() if line.strip()]
                if lines:
                    last_line = lines[-1]
                    try:
                        result = eval(last_line, {}, local_scope)
                    except Exception:
                        pass

            printed_output = stdout_capture.getvalue()
            
            formatted_res = None
            if isinstance(result, pd.DataFrame):
                formatted_res = result.to_dict(orient="records")
            elif isinstance(result, pd.Series):
                formatted_res = result.to_dict()
            elif isinstance(result, (np.integer, np.floating)):
                formatted_res = result.item()
            elif result is not None:
                formatted_res = result

            return {
                "success": True,
                "output": printed_output.strip(),
                "result": formatted_res,
                "result_raw": result,
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "output": stdout_capture.getvalue().strip(),
                "result": None,
                "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            }
        finally:
            sys.stdout = old_stdout


class ContextSearchTool:
    """
    Search & ontology engine for inventory metrics, business definitions, and operational KPIs.
    """
    DEFINITIONS = {
        "inventory turnover ratio": (
            "Inventory Turnover Ratio measures how many times inventory is sold and replaced over a period. "
            "Formula: (Units Sold / Average Inventory) or (COGS / Average Inventory). "
            "A higher turnover indicates strong sales demand and efficient inventory management, "
            "while a low ratio suggests overstocking, obsolescence, or lagging sales."
        ),
        "hand-in-stock": (
            "Hand-In-Stock (or Stock on Hand) refers to the actual physical quantity of inventory currently available "
            "in the warehouse or store ready for fulfillment. Formula: Opening Stock + Purchase/Stock In - Units Sold."
        ),
        "safety stock": (
            "Safety Stock is surplus buffer inventory held to mitigate the risk of stockouts caused by uncertainties "
            "in supply delivery or sudden surges in consumer demand. "
            "Formula: (Max Daily Usage × Max Lead Time) - (Average Daily Usage × Average Lead Time)."
        ),
        "reorder point": (
            "Reorder Point (ROP) is the specific inventory threshold that triggers a replenishment purchase order. "
            "Formula: (Lead Time Demand) + Safety Stock. When Hand-in-Stock dips below this point, new stock must be ordered."
        ),
        "stockout risk": (
            "Stockout Risk represents the vulnerability of exhausting inventory before the next replenishment arrives. "
            "High stockout risk occurs when Units Sold approach or exceed remaining Hand-in-Stock."
        ),
        "holding cost": (
            "Carrying/Holding Cost is the cumulative expenditure incurred to store and maintain unsold inventory, "
            "including warehouse rent, insurance, depreciation, shrinkage, handling, and opportunity cost of capital. "
            "Typically ranges from 15% to 30% of total inventory value annually."
        ),
        "economic order quantity": (
            "Economic Order Quantity (EOQ) is the optimal order quantity that minimizes the sum of inventory ordering "
            "costs and carrying costs. Formula: sqrt((2 * Demand * Order Cost) / Holding Cost per unit)."
        ),
        "gmroi": (
            "Gross Margin Return on Investment (GMROI) evaluates inventory profitability. "
            "Formula: Gross Margin / Average Inventory Cost. It reveals how many gross margin dollars are generated "
            "for every dollar invested in inventory."
        ),
        "dead stock": (
            "Dead stock refers to items that remain unsold for extended periods with zero or near-zero turnover. "
            "Holding dead stock ties up working capital and consumes warehouse storage without yielding revenue."
        ),
        "cost price per unit": (
            "Cost Price Per Unit (USD) is the per-item acquisition or manufacturing cost paid to suppliers before markup."
        ),
        "total inventory valuation": (
            "Total Inventory Valuation (USD) represents the total capitalized monetary value of stock currently on hand. "
            "Formula: Hand-In-Stock × Cost Price Per Unit."
        )
    }

    def search(self, query: str) -> dict:
        """
        Searches the knowledge base for definition or operational context.
        """
        q_clean = query.lower().strip()
        
        # Exact match
        for key, val in self.DEFINITIONS.items():
            if key in q_clean or q_clean in key:
                return {
                    "query": query,
                    "matched_term": key,
                    "definition": val,
                    "source": "Supply Chain & Inventory Management Knowledge Base"
                }
                
        # Word overlap search
        query_words = set(re.findall(r"\w+", q_clean))
        best_match = None
        best_score = 0
        
        for key, val in self.DEFINITIONS.items():
            key_words = set(re.findall(r"\w+", key))
            overlap = len(query_words.intersection(key_words))
            if overlap > best_score:
                best_score = overlap
                best_match = (key, val)
                
        if best_match and best_score >= 1:
            return {
                "query": query,
                "matched_term": best_match[0],
                "definition": best_match[1],
                "source": "Supply Chain & Inventory Management Knowledge Base"
            }
            
        return {
            "query": query,
            "matched_term": "General Inventory Context",
            "definition": (
                f"No exact match found for '{query}'. In inventory analytics, metrics typically evaluate "
                "velocity (Units Sold), holding levels (Hand-In-Stock), replenishment (Stock In), and capitalization (Cost Price Total)."
            ),
            "source": "Inventory Analytics Guide"
        }
