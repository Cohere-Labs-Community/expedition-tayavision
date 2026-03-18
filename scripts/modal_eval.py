import sys
import os
import modal

app = modal.App("tayavision-eval")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_file("pyproject.toml", remote_path="/root/project/pyproject.toml", copy=True)
    # Install everything directly into the system path
    .run_commands(
        "pip install --upgrade pip",
        "cd /root/project && pip install . vllm ray 'transformers>=4.46.0' lm-eval"
    )
    .add_local_dir("evaluation", remote_path="/root/project/evaluation", copy=True)
)

# Persistent volume for results in the cloud
results_volume = modal.Volume.from_name("tayavision-results", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100",
    volumes={"/results": results_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600 * 4,
)
def run_evaluation(task: str, model_name: str, batch_size: str = "auto", log_samples: bool = False, apply_chat_template: bool = False, limit: int = None):
    # Setup project environment inside the container
    sys.path.insert(0, "/root/project")
    os.chdir("/root/project")
    
    # Define a unique output directory on the volume to avoid mount conflicts
    # and to organize results by task as requested
    cloud_output_dir = f"/results/modal_{task}"
    os.makedirs(cloud_output_dir, exist_ok=True)
    
    # We call your existing evaluation script as if it were running natively
    from evaluation.run_eval import main
    
    # Pass arguments to the argparse in run_eval.py
    sys.argv = [
        "run_eval.py",
        "--task", task,
        "--model-name", model_name,
        "--backend", "vllm", 
        "--batch-size", batch_size,
        "--output-dir", cloud_output_dir
    ]
    
    if log_samples:
        sys.argv.append("--log-samples")
    
    if apply_chat_template:
        sys.argv.append("--apply-chat-template")
        
    if limit:
        sys.argv.extend(["--limit", str(limit)])
    
    print(f"Starting evaluation for task: {task} using model: {model_name}...")
    main()
    
    # Read generated results to send them back locally (recursive)
    results_data = {}
    for root, dirs, files in os.walk(cloud_output_dir):
        for filename in files:
            if filename.endswith(".json") or filename.endswith(".jsonl"):
                abs_path = os.path.join(root, filename)
                # Keep the 'modal_{task}' part in the relative path for local organization
                rel_path = os.path.relpath(abs_path, "/results")
                with open(abs_path, "r") as f:
                    results_data[rel_path] = f.read()
    
    results_volume.commit()
    return results_data

# 3. Running our function locally and remotely
# The @app.local_entrypoint defines the starting point when we run `modal run scripts/modal_eval.py`
@app.local_entrypoint()
def main(task: str, model_name: str = "CohereLabs/tiny-aya-base", batch_size: str = "auto", log_samples: bool = False, apply_chat_template: bool = False, limit: int = None):
    
    # We trigger the remote call that runs in the cloud with .remote()
    print("Initializing cloud GPU...")
    results_dict = run_evaluation.remote(
        task=task,
        model_name=model_name,
        batch_size=batch_size,
        log_samples=log_samples,
        apply_chat_template=apply_chat_template,
        limit=limit
    )
    
    # Write the results back to the local results directory
    local_base_dir = "evaluation/results"
    os.makedirs(local_base_dir, exist_ok=True)
    
    for rel_path, content in results_dict.items():
        local_path = os.path.join(local_base_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w") as f:
            f.write(content)
        print(f"Synced result: {local_path}")
    
    print("\nEvaluation complete. Results are stored in evaluation/results/")
