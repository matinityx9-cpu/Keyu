# modal_app.py
import modal

# Import your existing FastAPI app
from gradio_sd3_api import app as inference_app

# Create a Modal app (name it anything you like)
modal_app = modal.App("fitdit-inference")

# Build the container image with all dependencies
# It reads your requirements.txt so you don’t need to hardcode every package
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements.txt")
)

# Expose the FastAPI app with GPU support
@modal_app.function(
    image=image,
    gpu="A100-40GB",
    timeout=600  # allow long inference requests (seconds)
)
@modal.asgi_app()
def fastapi_app():
    return inference_app
