import os
import sys
import threading
import time
import uuid

import modal

app = modal.App("tayavision-eval-checkpoint")
MODEL_NAME = "TrishanuDas/tayavision-instruct-665k"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_file("pyproject.toml", remote_path="/root/project/pyproject.toml", copy=True)
    .run_commands(
        "pip install --upgrade pip",
        "cd /root/project && pip install . 'transformers>=4.46.0' lm-eval"
    )
    .add_local_dir("evaluation", remote_path="/root/project/evaluation", copy=True)
    .add_local_dir("models", remote_path="/root/project/models", copy=True)
    .add_local_dir("config", remote_path="/root/project/config", copy=True)
    .add_local_dir("src", remote_path="/root/project/src", copy=True)
    .add_local_dir("pipeline", remote_path="/root/project/pipeline", copy=True)
    .add_local_dir("scripts", remote_path="/root/project/scripts", copy=True)
)

# Persistent volume for results and checkpoints
results_volume = modal.Volume.from_name("tayavision-results", create_if_missing=True)

def _periodic_commit(volume: modal.Volume, stop_event: threading.Event, interval: int = 300):
    """Background thread: commit volume every `interval` seconds so checkpoints survive crashes."""
    while not stop_event.wait(timeout=interval):
        try:
            volume.commit()
        except Exception:
            pass

@app.function(
    image=image,
    gpu="A100",
    volumes={"/results": results_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600 * 4,
    retries=modal.Retries(max_retries=1),
)
def run_evaluation(
    task: str,
    run_id: str,
    batch_size: str = "2",
    log_samples: bool = False,
    apply_chat_template: bool = True,
    limit: int = None,
):
    sys.path.insert(0, "/root/project")
    os.chdir("/root/project")

    # Per-run directory: /results/runs/{run_id}/{task}/
    run_dir = f"/results/runs/{run_id}/{task}"
    os.makedirs(run_dir, exist_ok=True)

    # Point the backend at the checkpoint dir for resume support
    os.environ["TAYA_CHECKPOINT_DIR"] = run_dir

    import evaluation.tiny_aya_vision_lm_eval  # noqa: F401 — registers tiny-aya-vision backend

    # Background thread commits the volume every 5 min so checkpoints survive crashes
    stop_event = threading.Event()
    commit_thread = threading.Thread(
        target=_periodic_commit, args=(results_volume, stop_event), daemon=True
    )
    commit_thread.start()

    try:
        from evaluation.run_eval import main

        sys.argv = [
            "run_eval.py",
            "--task", task,
            "--model-name", MODEL_NAME,
            "--backend", "tiny-aya-vision",
            "--batch-size", batch_size,
            "--output-dir", run_dir,
        ]
        if log_samples:
            sys.argv.append("--log-samples")
        if apply_chat_template:
            sys.argv.append("--apply-chat-template")
        if limit:
            sys.argv.extend(["--limit", str(limit)])

        main()
    finally:
        stop_event.set()

    # Read results and return to local machine
    results_data = {}
    for root, dirs, files in os.walk(run_dir):
        for filename in files:
            if filename.endswith(".json") or filename.endswith(".jsonl"):
                abs_path = os.path.join(root, filename)
                rel_path = os.path.relpath(abs_path, "/results")
                with open(abs_path) as f:
                    results_data[rel_path] = f.read()

    results_volume.commit()
    return results_data


# Usage:
#   modal run scripts/modal_eval_checkpoint_en.py --task cvqa
#   modal run scripts/modal_eval_checkpoint_en.py --task cvqa_en
@app.local_entrypoint()
def main(
    task: str = "cvqa",
    run_id: str = "",
    batch_size: str = "2",
    log_samples: bool = False,
    apply_chat_template: bool = True,
    limit: int = None,
):
    # Generate a new run ID or resume an existing one
    if not run_id:
        run_id = str(uuid.uuid4())[:8]
        print(f"Starting new run: {run_id}")
    else:
        print(f"Resuming run: {run_id}")

    tasks = ["cvqa", "cvqa_en"] if task == "all" else [task]
    futures = [
        run_evaluation.remote(
            task=t,
            run_id=run_id,
            batch_size=batch_size,
            log_samples=log_samples,
            apply_chat_template=apply_chat_template,
            limit=limit,
        )
        for t in tasks
    ]

    local_base_dir = "evaluation/results"
    os.makedirs(local_base_dir, exist_ok=True)

    for results_dict in futures:
        for rel_path, content in results_dict.items():
            local_path = os.path.join(local_base_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w") as f:
                f.write(content)
            print(f"Synced result: {local_path}")

    print(f"\nEvaluation complete. Run ID: {run_id}")
    print("Results stored in evaluation/results/")
