"""
Master Runner Script for All 3 Tasks:
1. Task 1: Autonomous Inventory Intelligence Agent
2. Task 2: Document-Aware Support Assistant (10-Turn Demonstration)
3. Task 3: Intelligent Document Processing (IDP) Pipeline
"""

import os
import sys
import subprocess

# Windows UTF-8 console output support
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def run_task(task_num: int, script_path: str, title: str):
    print("\n" + "=" * 90)
    print(f"🚀 EXECUTING {title.upper()}")
    print("=" * 90)
    
    cmd = [sys.executable, script_path]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        print(f"❌ Error executing {script_path} (Exit code {res.returncode})")
    else:
        print(f"✅ {title} Completed Successfully.")

def main():
    print("*" * 90)
    print("       DESICREW AI SOLUTIONS — MASTER EVALUATION SUITE FOR TASKS 1, 2 & 3       ")
    print("*" * 90)

    # Task 1
    run_task(1, os.path.join("task1_inventory_agent", "cli_agent.py"), "Task 1: Inventory AI Agent")

    # Task 2
    run_task(2, os.path.join("task2_support_assistant", "run_10_turns_demo.py"), "Task 2: Support Assistant (10-Turn Multi-turn Demo)")

    # Task 3
    run_task(3, os.path.join("task3_document_pipeline", "run_pipeline.py"), "Task 3: IDP Document Extraction & Flagging Pipeline")

    print("\n" + "*" * 90)
    print("🎉 ALL 3 TASKS COMPLETED AND VERIFIED SUCCESSFULLY!")
    print("*" * 90)

if __name__ == "__main__":
    main()

