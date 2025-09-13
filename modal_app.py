# modal_app.py
import sys
import modal

app = modal.App("keyu-inference")

# Persistent volume for weights
volume = modal.Volume.from_name("fitdit-weights", create_if_missing=True)

# Base image with deps (no model weights baked in)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")  # OpenCV deps
    .pip_install_from_requirements("requirements.txt")
    .pip_install("huggingface_hub")
    .add_local_dir(".", remote_path="/root/FitDiT")  # your repo code
)

# One-time weight sync task
@app.function(volumes={"/root/FitDiT/models": volume}, timeout=600, image=image)
def sync_weights():
    """Download Hugging Face weights into the persistent volume."""
    from huggingface_hub import snapshot_download

    print("⬇️ Downloading FitDiT weights to volume...")
    snapshot_download(
        repo_id="BoyuanJiang/FitDiT",
        local_dir="/root/FitDiT/models",
        repo_type="model",
        resume_download=True,
    )
    print("✅ Weights are now cached in modal.Volume 'fitdit-weights'")


# Inference server
@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=600,
    volumes={"/root/FitDiT/models": volume},  # mount weights here
    # min_instances=1,  # keep a container warm
)

@modal.asgi_app()
def fastapi_app():
    """Expose the FastAPI app with pre-mounted weights."""
    sys.path.append("/root/FitDiT")
    from gradio_sd3_api import app as inference_app
    return inference_app
