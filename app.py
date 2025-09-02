import modal

# Name your Modal app
app = modal.App("fitdit")

# Define an image (runtime environment) with your requirements
image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
)

# Allocate a GPU (A100 recommended, T4 will likely OOM)
@app.function(image=image, gpu="A100", timeout=600)
def run_gradio():
    import subprocess
    subprocess.run([
        "python", "gradio_sd3.py",
        "--model_path", "local_model_dir",
        "--fp16"
    ])
