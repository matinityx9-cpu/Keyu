# modal_app.py
import modal

app = modal.App("fitdit-inference")

# Build the container image with requirements
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements.txt")
)

# Expose the FastAPI app with GPU support
@app.function(   # <- was modal_app.function
    image=image,
    gpu="A100-40GB",
    timeout=600,
)
@modal.asgi_app()
def fastapi_app():
    # Import here, inside the function, AFTER requirements are installed
    from gradio_sd3_api import app as inference_app
    return inference_app
