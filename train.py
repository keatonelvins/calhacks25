import modal
import argparse
import subprocess
from pathlib import Path
from verifiers.scripts.rl import load_toml, build_vllm_command, build_train_command

app = modal.App("vf")
artifacts_volume = modal.Volume.from_name("vf-artifacts", create_if_missing=True)
envs = [p.name for p in Path("environments").glob("*")]

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git")
    .pip_install("uv")
    .uv_sync(extras=["rl"])
    .env({"WANDB_PROJECT": "calhacks25"})
    .add_local_dir("configs", "/root/configs")
)
if len(envs) > 0:
    print(f"Adding {envs} to image")
    image = image.add_local_python_source(*envs)

def build_gpu_prefix(start: int, end: int):
    return "CUDA_VISIBLE_DEVICES=" + ",".join(str(i) for i in range(start, end))

def train(config: str, infer_gpus: int, total_gpus: int):
    data = load_toml(Path(f"/root/{config}"))
    
    vllm_cmd = build_vllm_command(data["model"], data["inference"], build_gpu_prefix(0, infer_gpus))
    train_cmd = build_train_command(data["env"]["id"], f"/root/{config}", build_gpu_prefix(infer_gpus, total_gpus))
    
    subprocess.Popen(vllm_cmd, shell=True)
    subprocess.run(train_cmd, shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the TOML file (must be in configs/).")
    parser.add_argument("--detach", action="store_true", help="Keep the Modal runner alive even if the terminal closes.")
    args = parser.parse_args()
    config, detach = args.config, args.detach

    data = load_toml(Path(config))
    infer_gpus = data["inference"]["gpus"]
    total_gpus = infer_gpus + data["trainer"]["gpus"]

    runner = app.function(
        image=image,
        gpu=f"H100:{total_gpus}",
        timeout=60 * 60 * 24,
        volumes={"/artifacts": artifacts_volume},
        secrets=[modal.Secret.from_name("wandb-secret")],
    )(train)
    with modal.enable_output(), app.run(detach=detach):
        runner.remote(config, infer_gpus, total_gpus)
