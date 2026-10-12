"""
CLI Runner for Task 1: Inventory Agent.
Demonstrates:
1. Dynamic code writing and execution on inventory data
2. Context and definition lookup using the search tool
3. Business synthesis and summary in plain English
"""

import os
import sys

# Windows UTF-8 console output support
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure local imports work cleanly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from task1_inventory_agent.agent import InventoryAgent

def run_cli_demo():
    print("=" * 80)
    print("TASK 1: INVENTORY DATA AGENT DEMONSTRATION")
    print("=" * 80)

    excel_path = os.path.join(os.path.dirname(__file__), "data", "Inventory-Records-Sample-Data.xlsx")
    if not os.path.exists(excel_path):
        # Fallback to original dataset directory
        excel_path = r"Question 1 & 3 (Files to use)\Question 1 & 3 (Files to use)\Question 1\Inventory-Records-Sample-Data.xlsx"

    print(f"Loading dataset from: {excel_path}")
    agent = InventoryAgent(excel_path)
    print(f"Dataset loaded: {len(agent.df)} SKUs successfully indexed.\n")

    sample_queries = [
        "What are the top 5 products by total inventory valuation?",
        "What does Inventory Turnover Ratio mean and which products have the highest turnover?",
        "Which products have low stock or high stockout risk?",
        "Give me an executive summary of total inventory value, total units sold, and stock on hand."
    ]

    for idx, query in enumerate(sample_queries, start=1):
        print("\n" + "#" * 80)
        print(f"QUERY {idx}: \"{query}\"")
        print("#" * 80)

        result = agent.ask(query)

        print("\n[1]  CODE WRITTEN & EXECUTED:")
        print("-" * 50)
        print(result["code"])

        print("\n[2]  EXECUTION OUTPUT:")
        print("-" * 50)
        raw_res = result["execution"].get("result_raw")
        if raw_res is not None:
            print(raw_res)
        elif result["execution"].get("output"):
            print(result["execution"]["output"])
        else:
            print(result["execution"].get("result"))

        if result.get("search_context"):
            print("\n[3] 🔍 CONTEXT & DEFINITION LOOKUP:")
            print("-" * 50)
            sc = result["search_context"]
            print(f"Term: {sc.get('matched_term')}")
            print(f"Definition: {sc.get('definition')}")
            print(f"Source: {sc.get('source')}")

        print("\n[4]  PLAIN ENGLISH BUSINESS SUMMARY:")
        print("-" * 50)
        print(result["summary"])
        print("\n" + "=" * 80)

if __name__ == "__main__":
    run_cli_demo()
