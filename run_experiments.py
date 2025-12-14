import subprocess
import os
import sys
import time
import config

def run_script(script_name, log_file_path):
    print(f"\n[Pipeline] Starting {script_name}...")
    start_time = time.time()
    
    with open(log_file_path, "w") as log_file:
        process = subprocess.Popen(
            [sys.executable, script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            
        process.wait()
        
    duration = time.time() - start_time
    if process.returncode == 0:
        print(f"[Pipeline] Finished {script_name} in {duration:.2f}s.")
    else:
        print(f"[Pipeline] Error in {script_name}. Return code: {process.returncode}")
        sys.exit(1)

def main():
    # Use path from config
    results_dir = config.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"--- Starting Full Training Pipeline (Results: {results_dir}) ---")
    
    experiments = [
        ("train_baseline.py", "log_1_baseline.txt"),
        #("train_finetune.py", "log_2_finetune.txt"),
        ("train_simclr.py", "log_3_simclr_pretrain.txt"),
        ("train_linear_eval.py", "log_4_simclr_eval.txt"),
        ("final_evaluation.py", "log_5_final_report.txt")
    ]

    for script, log_name in experiments:
        if not os.path.exists(script):
            print(f"Error: Script {script} not found!")
            sys.exit(1)
            
        log_path = os.path.join(results_dir, log_name)
        run_script(script, log_path)

    print("\n[Pipeline] All tasks completed.")

if __name__ == "__main__":
    main()