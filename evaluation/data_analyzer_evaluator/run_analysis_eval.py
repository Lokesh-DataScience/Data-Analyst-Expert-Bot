import pandas as pd
import sys
from pathlib import Path

# Ensure project root is on sys.path so `utils` can be imported
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from data_analyzer_evaluator import DataAnalyzerEvaluator

# Sample dataset
df = pd.read_csv("student_admission_record_dirty.csv")

evaluator = DataAnalyzerEvaluator()

results = evaluator.evaluate(df)

print("\n📊 Data Analyzer Evaluation:\n")

for section, metrics in results.items():
    print(f"\n🔹 {section.upper()}")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")