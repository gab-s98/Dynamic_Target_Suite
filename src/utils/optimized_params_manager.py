import json
import os
import datetime
import pandas as pd

HISTORY_FILE = "optimized_params_history.json"
LAST_RUN_FILE = "last_optimized_params.json"

def get_storage_path(filename):
    """Returns absolute path in working directory."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, filename)

def save_optimization_run(model_type, objective_mode, v_x, penalty, tir_file,
                          initial_params: dict, optimized_params: dict,
                          kpis: dict = None) -> dict:
    """
    Saves an optimization run record to JSON history and updates last_optimized_params.json.
    """
    hist_path = get_storage_path(HISTORY_FILE)
    history = load_history()
    
    run_id = len(history) + 1
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    run_record = {
        "run_id": run_id,
        "timestamp": timestamp,
        "model_type": model_type,
        "objective_mode": objective_mode,
        "v_x": float(v_x),
        "penalty_factor": float(penalty),
        "tir_file": os.path.basename(tir_file) if tir_file else "default.tir",
        "initial_params": {k: float(v) for k, v in initial_params.items() if isinstance(v, (int, float))},
        "optimized_params": {k: float(v) for k, v in optimized_params.items() if isinstance(v, (int, float))},
        "kpis": {k: float(v) for k, v in (kpis or {}).items() if isinstance(v, (int, float))}
    }
    
    history.append(run_record)
    
    try:
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save optimization history: {e}")
        
    last_path = get_storage_path(LAST_RUN_FILE)
    try:
        with open(last_path, "w", encoding="utf-8") as f:
            json.dump(run_record, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save last optimization run: {e}")
        
    return run_record

def load_history() -> list:
    """Loads all past optimization runs."""
    hist_path = get_storage_path(HISTORY_FILE)
    if not os.path.exists(hist_path):
        return []
    try:
        with open(hist_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []

def get_last_run() -> dict:
    """Retrieves the most recent optimization run."""
    last_path = get_storage_path(LAST_RUN_FILE)
    if os.path.exists(last_path):
        try:
            with open(last_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    history = load_history()
    return history[-1] if history else {}

def export_history_to_csv(csv_filepath: str) -> bool:
    """Flattens optimization history and exports to CSV."""
    history = load_history()
    if not history:
        return False
    
    flat_rows = []
    for item in history:
        row = {
            "Run ID": item.get("run_id"),
            "Timestamp": item.get("timestamp"),
            "Model": item.get("model_type"),
            "Objective": item.get("objective_mode"),
            "Target Speed [m/s]": item.get("v_x"),
            "Penalty": item.get("penalty_factor"),
            "Tire": item.get("tir_file")
        }
        for k, v in item.get("optimized_params", {}).items():
            row[f"Opt_{k}"] = v
        for k, v in item.get("kpis", {}).items():
            row[f"KPI_{k}"] = v
        flat_rows.append(row)
        
    df = pd.DataFrame(flat_rows)
    df.to_csv(csv_filepath, index=False)
    return True
