import modal
import subprocess
from pathlib import Path

CONFIG = "wordle.toml"
NUM_GPUS = 2

app = modal.App("vf-wordle")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git")
    .pip_install("uv")
    .uv_sync(extras=["verifiers[rl]"])
    .uv_pip_install("wordle", extra_index_url="https://hub.primeintellect.ai/will/simple/")
    .add_local_file("wordle.toml", f"/root/{CONFIG}")
)

@app.function(
    image=image,
    gpu=f"H100:{NUM_GPUS}",
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=60 * 60 * 24,
)
def train():
    from verifiers.scripts.rl import load_toml, build_vllm_command

    data = load_toml(Path(f"/root/{CONFIG}"))
    infer_gpus = data["inference"]["gpus"]
    total_gpus = infer_gpus + data["trainer"]["gpus"]
    
    inference_gpu_str = "CUDA_VISIBLE_DEVICES=" + ",".join(str(i) for i in range(infer_gpus))
    trainer_gpu_str = "CUDA_VISIBLE_DEVICES=" + ",".join(str(i) for i in range(infer_gpus, total_gpus))
    
    vllm_cmd = build_vllm_command(data["model"], data["inference"], inference_gpu_str)
    train_cmd = " ".join([trainer_gpu_str, "uv run", "vf-train", "@", f"/root/{CONFIG}"])

    subprocess.Popen(vllm_cmd, shell=True)
    subprocess.run(train_cmd, shell=True)