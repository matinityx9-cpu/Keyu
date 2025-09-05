# modal_app.py
import sys
import modal
from huggingface_hub import snapshot_download

app = modal.App("igor-inference")

def download_models():
    print("⬇️ Downloading FitDiT weights from Hugging Face...")
    snapshot_download(
        repo_id="BoyuanJiang/FitDiT",
        local_dir="/root/FitDiT/models",
        repo_type="model"
    )
    print("✅ Weights downloaded to /root/FitDiT/models")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")  # OpenCV dependencies
    .pip_install_from_requirements("requirements.txt")
    .pip_install("huggingface_hub")
    .run_function(download_models)  # <--- runs at build time
    .add_local_dir(".", remote_path="/root/FitDiT")
)

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=600,
)
@modal.asgi_app()
def fastapi_app():
    sys.path.append("/root/FitDiT")
    from gradio_sd3_api import app as inference_app
    return inference_app
