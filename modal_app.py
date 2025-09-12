import sys
import modal
from huggingface_hub import snapshot_download

# -------------------------------------------------------------------
# App name
# -------------------------------------------------------------------
app = modal.App("keyu-inference")

# -------------------------------------------------------------------
# Step 1: Function to download Hugging Face model at build time
# -------------------------------------------------------------------
def download_models():
    print("⬇️ Downloading FitDiT weights from Hugging Face...")
    snapshot_download(
        repo_id="BoyuanJiang/FitDiT",
        local_dir="/root/FitDiT/models",
        repo_type="model"
    )
    print("✅ Weights downloaded to /root/FitDiT/models")

# -------------------------------------------------------------------
# Step 2: Base image with dependencies + model download
# -------------------------------------------------------------------
base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")  # OpenCV deps
    .pip_install_from_requirements("requirements.txt")
    .pip_install("huggingface_hub")
    .run_function(download_models)          # download model into container
    .add_local_dir(".", remote_path="/root/FitDiT")
)

# -------------------------------------------------------------------
# Step 3: Preload FitDiT model once and store in snapshot
# -------------------------------------------------------------------
@app.function(
    image=base_image,
    gpu="A100-40GB",
    timeout=600,
    min_containers=1,   # keep at least 1 container warm
)
def load_and_snapshot():
    sys.path.append("/root/FitDiT")
    from gradio_sd3_api import FitDiTGenerator, app as inference_app
    print("🚀 Preloading FitDiT model into snapshot...")
    gen = FitDiTGenerator("/root/FitDiT/models", device="cuda:0", with_fp16=True)
    inference_app.state.generator = gen
    print("✅ Generator stored in inference_app.state")
    return inference_app

# -------------------------------------------------------------------
# Step 4: FastAPI app that reuses preloaded generator
# -------------------------------------------------------------------
@app.function(
    image=base_image,
    gpu="A100-40GB",
    timeout=600,
    min_containers=1,   # keep at least 1 container warm
)
@modal.asgi_app()
def fastapi_app():
    sys.path.append("/root/FitDiT")
    from gradio_sd3_api import app as inference_app
    return inference_app
