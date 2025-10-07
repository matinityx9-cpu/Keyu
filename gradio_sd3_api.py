# gradio_sd3_api_fixed.py
# Full file with async upload helper and awaited calls in /tryon endpoint.

import argparse
import os
import math
import tempfile
import shutil
import io
import traceback
from typing import Tuple

from preprocess.humanparsing.run_parsing import Parsing
from preprocess.dwpose import DWposeDetector
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
import torch
import torch.nn as nn
from src.pose_guider import PoseGuider
from PIL import Image
from src.utils_mask import get_mask_location
import numpy as np
from src.pipeline_stable_diffusion_3_tryon import StableDiffusion3TryOnPipeline
from src.transformer_sd3_garm import SD3Transformer2DModel as SD3Transformer2DModel_Garm
from src.transformer_sd3_vton import SD3Transformer2DModel as SD3Transformer2DModel_Vton
import cv2
import random

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response, JSONResponse
import uvicorn

# --------------------------- Keyu core (kept from original) ---------------------------
example_path = os.path.join(os.path.dirname(__file__), 'examples')


class KeyuGenerator:
    def __init__(self, model_root, offload=False, aggressive_offload=False, device="cuda:0", with_fp16=False):
        weight_dtype = torch.float16 if with_fp16 else torch.bfloat16
        transformer_garm = SD3Transformer2DModel_Garm.from_pretrained(os.path.join(model_root, "transformer_garm"), torch_dtype=weight_dtype)
        transformer_vton = SD3Transformer2DModel_Vton.from_pretrained(os.path.join(model_root, "transformer_vton"), torch_dtype=weight_dtype)
        pose_guider =  PoseGuider(conditioning_embedding_channels=1536, conditioning_channels=3, block_out_channels=(32, 64, 256, 512))
        pose_guider.load_state_dict(torch.load(os.path.join(model_root, "pose_guider", "diffusion_pytorch_model.bin")))
        image_encoder_large = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=weight_dtype)
        image_encoder_bigG = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", torch_dtype=weight_dtype)
        pose_guider.to(device=device, dtype=weight_dtype)
        image_encoder_large.to(device=device)
        image_encoder_bigG.to(device=device)
        self.pipeline = StableDiffusion3TryOnPipeline.from_pretrained(model_root, torch_dtype=weight_dtype, transformer_garm=transformer_garm, transformer_vton=transformer_vton, pose_guider=pose_guider, image_encoder_large=image_encoder_large, image_encoder_bigG=image_encoder_bigG)
        self.pipeline.to(device)
        if offload:
            self.pipeline.enable_model_cpu_offload()
            self.dwprocessor = DWposeDetector(model_root=model_root, device='cpu')
            self.parsing_model = Parsing(model_root=model_root, device='cpu')
        elif aggressive_offload:
            self.pipeline.enable_sequential_cpu_offload()
            self.dwprocessor = DWposeDetector(model_root=model_root, device='cpu')
            self.parsing_model = Parsing(model_root=model_root, device='cpu')
        else:
            self.pipeline.to(device)
            self.dwprocessor = DWposeDetector(model_root=model_root, device=device)
            self.parsing_model = Parsing(model_root=model_root, device=device)
        
    def generate_mask(self, vton_img, category, offset_top, offset_bottom, offset_left, offset_right):
        with torch.inference_mode():
            vton_img = Image.open(vton_img).convert("RGB")  # <-- force 3 channels
            vton_img_det = resize_image(vton_img)
            pose_image, keypoints, _, candidate = self.dwprocessor(np.array(vton_img_det)[:, :, ::-1])

            candidate[candidate<0]=0
            candidate = candidate[0]

            candidate[:, 0]*=vton_img_det.width
            candidate[:, 1]*=vton_img_det.height

            pose_image = pose_image[:,:,::-1] #rgb
            pose_image = Image.fromarray(pose_image)
            model_parse, _ = self.parsing_model(vton_img_det)

            mask, mask_gray = get_mask_location(category, model_parse, \
                                        candidate, model_parse.width, model_parse.height, \
                                        offset_top, offset_bottom, offset_left, offset_right)
            mask = mask.resize(vton_img.size)
            mask_gray = mask_gray.resize(vton_img.size)
            mask = mask.convert("L")
            mask_gray = mask_gray.convert("L")
            masked_vton_img = Image.composite(mask_gray, vton_img, mask)

            im = {}
            im['background'] = np.array(vton_img.convert("RGBA"))
            im['layers'] = [np.concatenate((np.array(mask_gray.convert("RGB")), np.array(mask)[:,:,np.newaxis]),axis=2)]
            im['composite'] = np.array(masked_vton_img.convert("RGBA"))
            
            return im, pose_image
        
    # --- background removal ---

    

    def process(
            self,
            vton_img,
            garm_img,
            pre_mask,
            pose_image,
            n_steps,
            image_scale,
            seed,
            num_images_per_prompt,
            resolution,
            ):
        
        assert resolution in ["768x1024", "1152x1536", "1536x2048"]
        new_width, new_height = map(int, resolution.split("x"))
        
        with torch.inference_mode():
            # --- load inputs ---
            garm_img = Image.open(garm_img).convert("RGB")
            vton_img = Image.open(vton_img).convert("RGB")
            
            # --- background removal on garment ---
            from rembg import remove
            garm_img = remove(garm_img)       # RGBA
            garm_img = garm_img.convert("RGB")  # flatten transparency
            
            # --- resize & pad ---
            model_image_size = vton_img.size
            garm_img, _, _ = pad_and_resize(garm_img, new_width=new_width, new_height=new_height)
            vton_img, pad_w, pad_h = pad_and_resize(vton_img, new_width=new_width, new_height=new_height)
            
            mask = pre_mask["layers"][0][:,:,3]
            mask = Image.fromarray(mask)
            mask, _, _ = pad_and_resize(mask, new_width=new_width, new_height=new_height, pad_color=(0,0,0))
            mask = mask.convert("L")
            
            pose_image, _, _ = pad_and_resize(
                pose_image, new_width=new_width, new_height=new_height, pad_color=(0,0,0)
                )
            
            # --- run inference ---
            if seed == -1:
                seed = random.randint(0, 2147483647)
                
            res = self.pipeline(
                height=new_height,
                width=new_width,
                guidance_scale=image_scale,
                num_inference_steps=n_steps,
                generator=torch.Generator("cpu").manual_seed(seed),
                cloth_image=garm_img,
                model_image=vton_img,
                mask=mask,
                pose_image=pose_image,
                num_images_per_prompt=num_images_per_prompt,
            ).images
            
            # --- unpad results back to original person size ---
            for idx in range(len(res)):
                res[idx] = unpad_and_resize(res[idx], pad_w, pad_h, model_image_size[0], model_image_size[1])
            
            return res




def pad_and_resize(im, new_width=768, new_height=1024, pad_color=(255, 255, 255), mode=Image.LANCZOS):
    old_width, old_height = im.size
    
    ratio_w = new_width / old_width
    ratio_h = new_height / old_height
    if ratio_w < ratio_h:
        new_size = (new_width, round(old_height * ratio_w))
    else:
        new_size = (round(old_width * ratio_h), new_height)
    
    im_resized = im.resize(new_size, mode)

    pad_w = math.ceil((new_width - im_resized.width) / 2)
    pad_h = math.ceil((new_height - im_resized.height) / 2)

    new_im = Image.new('RGB', (new_width, new_height), pad_color)
    
    new_im.paste(im_resized, (pad_w, pad_h))

    return new_im, pad_w, pad_h


def unpad_and_resize(padded_im, pad_w, pad_h, original_width, original_height):
    width, height = padded_im.size
    
    left = pad_w
    top = pad_h
    right = width - pad_w
    bottom = height - pad_h
    
    cropped_im = padded_im.crop((left, top, right, bottom))

    resized_im = cropped_im.resize((original_width, original_height), Image.LANCZOS)

    return resized_im


def resize_image(img, target_size=768):
    width, height = img.size
    
    if width < height:
        scale = target_size / width
    else:
        scale = target_size / height
    
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    
    return resized_img

# --------------------------- API wrapper and hardcoded params ---------------------------

# Hardcoded defaults requested:
TRYON_RESOLUTION = "1152x1536"
STEPS = 20
MASK_OFFSETS = (0, 0, 0, 0)  # top, bottom, left, right
GUIDANCE_SCALE = 2.0
SEED = -1
NUM_IMAGES = 1
RUN_MASKS = True
RUN_TRYON = True

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# existing code...
app = FastAPI(title="Keyu Try-on API")

# Allow all origins (for testing — you can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # or restrict to ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Modal startup hook ---
@app.on_event("startup")
async def load_model():
    model_path = "/root/keyu/models"   # 🔧 adjust this to where your weights are stored
    print("Loading keyu model for Modal deployment...")
    generator = KeyuGenerator(model_path, device="cuda:0", with_fp16=True)
    app.state.generator = generator
    print("✅ Model loaded and stored in app.state.generator")


# Health
@app.get("/")
async def root():
    return {"status": "ok", "service": "Keyu Try-on API"}


# ---------- Async helper that reads UploadFile bytes and writes to disk ----------
async def _save_upload_to_temp(upload: UploadFile, suffix: str) -> str:
    """
    Read the incoming UploadFile fully as bytes and write to a temporary file.
    Return the path to the temporary file.
    """
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    # Read all bytes from the UploadFile (async)
    data = await upload.read()

    # Write bytes to disk
    with open(path, "wb") as f:
        f.write(data)

    # Close the UploadFile
    try:
        await upload.close()
    except Exception:
        pass

    return path
# -------------------------------------------------------------------------------


from fastapi import Query

@app.post("/tryon")
async def tryon(
    person: UploadFile = File(...),
    garment: UploadFile = File(...),
    category: str = Form(...),
    format: str = Query("jpeg", regex="^(jpeg|png)$")  # <-- NEW
):
    """Run the Keyu try-on pipeline with hardcoded parameters.

    Request: multipart/form-data with fields:
      - person: image file (jpg/png)
      - garment: image file (jpg/png)
      - category: string ("Upper-body", "Lower-body", "Dresses")
      - format: string ("jpeg" or "png"), default = "jpeg"

    Response: image bytes
    """
    if category not in ["Upper-body", "Lower-body", "Dresses"]:
        raise HTTPException(status_code=400, detail="Invalid category. Choose one of Upper-body, Lower-body, Dresses")

    tmp_person = None
    tmp_garment = None
    try:
        tmp_person = await _save_upload_to_temp(person, suffix="_person.jpg")
        tmp_garment = await _save_upload_to_temp(garment, suffix="_garment.png")

        gen = app.state.generator
        pre_mask, pose_image = gen.generate_mask(tmp_person, category, *MASK_OFFSETS)

        res_images = gen.process(
            tmp_person,
            tmp_garment,
            pre_mask,
            pose_image,
            n_steps=STEPS,
            image_scale=GUIDANCE_SCALE,
            seed=SEED,
            num_images_per_prompt=NUM_IMAGES,
            resolution=TRYON_RESOLUTION,
        )

        if not res_images:
            raise HTTPException(status_code=500, detail="No image generated")

        out_img = res_images[0]
        buf = io.BytesIO()

        if format == "jpeg":   # <-- smaller & faster
            out_img.save(buf, format="JPEG", quality=90)
            media_type = "image/jpeg"
        else:                  # <-- fallback to PNG
            out_img.save(buf, format="PNG")
            media_type = "image/png"

        buf.seek(0)
        return Response(content=buf.getvalue(), media_type=media_type)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            if tmp_person and os.path.exists(tmp_person):
                os.remove(tmp_person)
            if tmp_garment and os.path.exists(tmp_garment):
                os.remove(tmp_garment)
        except Exception:
            pass



# --------------------------- Entrypoint ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keyu API runner")
    parser.add_argument("--model_path", type=str, required=True, help="The path of Keyu model.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--fp16", action="store_true", help="Load model with fp16, default is bf16")
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use.")
    parser.add_argument("--aggressive_offload", action="store_true", help="Offload model more aggressively to CPU when not in use.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the API")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind the API")

    args = parser.parse_args()

    # instantiate and store generator on app state
    print("Loading Keyu model (this can take a few minutes)...")
    generator = KeyuGenerator(args.model_path, offload=args.offload, aggressive_offload=args.aggressive_offload, device=args.device, with_fp16=args.fp16)
    app.state.generator = generator
    print("Model loaded. Starting API...")

    uvicorn.run(app, host=args.host, port=args.port)
    print("✅ FastAPI app loaded with /tryon route")
