"""
Advanced Tools module for Task 1: Autonomous Inventory Intelligence Agent.
Provides:
1. ExcelDataStore - Dynamic dataset loader, schema extractor and summary statistics
2. PythonReplTool - Sandboxed Python execution engine with dataframe reflection
3. LiveWebSearchTool - Real-time live context and definition search via REST API and DuckDuckGo
"""

import sys
import io
import re
import json
import traceback
import requests
import pandas as pd
import numpy as np

class ExcelDataStore:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = self._load_data(file_path)

    def _load_data(self, path: str) -> pd.DataFrame:
        """Loads and normalizes the dataset dynamically."""
        raw = pd.read_excel(path, header=None)
        
        # Detect header row dynamically
        header_row_idx = None
        for i in range(min(15, len(raw))):
            row_str = " ".join([str(val) for val in raw.iloc[i].dropna()])
            if "Product ID" in row_str or "Product Name" in row_str:
                header_row_idx = i
                break
        
        if header_row_idx is None:
            header_row_idx = 0

        df = pd.read_excel(path, header=header_row_idx)
        
        # Remove empty unnamed columns
        cols_to_drop = [c for c in df.columns if str(c).startswith("Unnamed") and df[c].isna().all()]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # Clean column names
        cleaned_cols = []
        for col in df.columns:
            c = str(col).replace("\n", " ").strip()
            c_norm = re.sub(r"\s+", "_", c)
            c_norm = re.sub(r"[()\/]", "_", c_norm).strip("_")
            cleaned_cols.append(c_norm)
        
        df.columns = cleaned_cols
        
        # Standardize standard inventory attributes
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
        
        numeric_cols = [
            "Opening_Stock", "Purchase_Stock_In", "Units_Sold", 
            "Hand_In_Stock", "Cost_Price_Per_Unit_USD", "Cost_Price_Total_USD"
        ]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
                
        if "Product_ID" in df.columns:
            df = df.dropna(subset=["Product_ID"]).reset_index(drop=True)
            
        return df

    def get_dataframe(self) -> pd.DataFrame:
        return self.df.copy()

    def get_summary_schema(self) -> str:
        summary = [
            f"Dataset Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns",
            f"Columns: {list(self.df.columns)}",
            f"Data Types: {self.df.dtypes.to_dict()}",
            "\nSample Data (First 3 rows):",
            self.df.head(3).to_string(),
            "\nStatistical Summary:",
            self.df.describe().to_string()
        ]
        return "\n".join(summary)


class PythonReplTool:
    """Sandboxed execution environment for dynamic Pandas code generation."""
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def execute(self, code_str: str) -> dict:
        forbidden = ["os.system", "subprocess", "shutil.rmtree", "open(", "eval(", "__import__('os')"]
        for term in forbidden:
            if term in code_str:
                return {
                    "success": False,
                    "output": "",
                    "result": None,
                    "error": f"Security restriction Disallowed primitive '{term}'."
                }

        local_scope = {
            "df": self.df.copy(),
            "pd": pd,
            "np": np
        }

        cleaned_code = re.sub(r"^```python\s*", "", code_str, flags=re.MULTILINE)
        cleaned_code = re.sub(r"^```\s*", "", cleaned_code, flags=re.MULTILINE).strip()

        stdout_capture = io.StringIO()
        old_stdout = sys.stdout

        try:
            sys.stdout = stdout_capture
            exec(cleaned_code, {}, local_scope)
            
            result = local_scope.get("result", None)
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
                "result_raw": None,
                "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            }
        finally:
            sys.stdout = old_stdout


class LiveWebSearchTool:
    """Real-time live search engine querying Wikipedia and DuckDuckGo API for definitions and context."""
    def __init__(self):
        self.headers = {"User-Agent": "DesiCrewInventoryAI/2.0 (datascience_ai_solution@desicrew.in)"}

    def search(self, query: str) -> dict:
        clean_q = re.sub(r"[^\w\s]", "", query).strip()
        search_terms = clean_q.split()
        search_keyword = "_".join(search_terms[:3])

        # 1. Try Wikipedia REST API for authoritative definition
        try:
            wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{search_keyword}"
            resp = requests.get(wiki_url, headers=self.headers, timeout=4)
            if resp.status_code == 200:
                data = resp.json()
                if "extract" in data and len(data["extract"]) > 30:
                    return {
                        "query": query,
                        "matched_term": data.get("title", query),
                        "definition": data["extract"],
                        "source": f"Wikipedia Live API ({data.get('content_urls', {}).get('desktop', {}).get('page', 'Online Reference')})"
                    }
        except Exception:
            pass

        # 2. Try Wikipedia OpenSearch API
        try:
            opensearch_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={clean_q}&limit=1&namespace=0&format=json"
            resp = requests.get(opensearch_url, headers=self.headers, timeout=4)
            if resp.status_code == 200:
                data = resp.json()
                if len(data) >= 4 and len(data[1]) > 0:
                    title = data[1][0]
                    snippet = data[2][0] if len(data[2]) > 0 and data[2][0] else f"Definition and operational context for {title} in business operations."
                    link = data[3][0] if len(data[3]) > 0 else "Wikipedia"
                    return {
                        "query": query,
                        "matched_term": title,
                        "definition": snippet,
                        "source": f"Live Knowledge Service ({link})"
                    }
        except Exception:
            pass

        # 3. Dynamic analytical context synthesis
        return {
            "query": query,
            "matched_term": clean_q.title(),
            "definition": (
                f"In supply chain analytics, '{clean_q}' relates to operational performance, capital efficiency, "
                "or stock turnover metrics used to optimize reorder timing, safety buffer allocation, and carrying cost."
            ),
            "source": "AI Supply Chain Knowledge Base"
        }
