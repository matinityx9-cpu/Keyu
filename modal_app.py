# modal_app.py
import sys
import modal

app = modal.App("fitdit-inference")

# Build the container image with requirements and mount local code
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path="/root/FitDiT")   # ✅ NEW API
)

# Expose the FastAPI app with GPU support
@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=600,
)
@modal.asgi_app()
def fastapi_app():
    sys.path.append("/root/FitDiT")  # so imports like gradio_sd3_api work
    from gradio_sd3_api import app as inference_app
    return inference_app
