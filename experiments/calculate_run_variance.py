import json
import re
import csv
import statistics
from pathlib import Path
from collections import defaultdict

# Regex to parse batch files: batch_eval_{model}_{YYYYMMDD}_{HHMMSS}.json
pattern = re.compile(r"batch_eval_(.*?)_(\d{8})_(\d{6})\.json")

# Structure: by_day[date][model][timestamp] = {"passed": x, "total": y, "time": z}
runs_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"passed": 0, "total": 0, "time": 0.0})))

print("Parsing evaluation files...")
for file in Path("results").rglob("batch_eval_*.json"):
    match = pattern.search(file.name)
    if not match:
        continue
    
    model, date, time = match.groups()
    
    try:
        data = json.loads(file.read_text())
        results = data.get("results", {})
        
        # Determine strategy if easily accessible in log
        for res in results.values() if isinstance(results, dict) else results:
            if isinstance(res, dict):
                runs_data[date][model][time]["total"] += 1
                if res.get("validation_passed", False):
                     runs_data[date][model][time]["passed"] += 1
                runs_data[date][model][time]["time"] += res.get("execution_time", 0.0)
    except Exception as e:
        pass

output_file = Path("results/daily_variance_report.csv")
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "Date", "Model", "Total Runs", 
        "Avg Pass Rate (%)", "Variance Pass Rate", "StdDev Pass Rate (%)", 
        "Avg Time per Run (s)", "Variance Time", "StdDev Time (s)"
    ])
    
    for date in sorted(runs_data.keys()):
        for model in sorted(runs_data[date].keys()):
            runs = runs_data[date][model]
            
            pass_rates = []
            times = []
            
            # Aggregate domains for each run (timestamp)
            for time_stamp, metrics in runs.items():
                if metrics["total"] > 0:
                    pass_rates.append((metrics["passed"] / metrics["total"]) * 100)
                times.append(metrics["time"])
            
            if not pass_rates:
                continue
                
            n_runs = len(pass_rates)
            
            mean_pass = sum(pass_rates) / n_runs
            mean_time = sum(times) / n_runs
            
            # Calculate variance and stddev (uses sample variance, needs 2+ items)
            if n_runs > 1:
                var_pass = statistics.variance(pass_rates)
                stdev_pass = statistics.stdev(pass_rates)
                var_time = statistics.variance(times)
                stdev_time = statistics.stdev(times)
            else:
                var_pass = 0.0
                stdev_pass = 0.0
                var_time = 0.0
                stdev_time = 0.0
                
            writer.writerow([
                date, model, n_runs,
                f"{mean_pass:.2f}", f"{var_pass:.2f}", f"{stdev_pass:.2f}",
                f"{mean_time:.2f}", f"{var_time:.2f}", f"{stdev_time:.2f}"
            ])

print(f"Report generated: {output_file}")
