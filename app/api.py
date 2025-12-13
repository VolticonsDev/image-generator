from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline
import tempfile
import os
from io import BytesIO

MODEL_ID = "runwayml/stable-diffusion-v1-5" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    print(f"Modelo Stable Diffusion cargado en: {DEVICE}")
except Exception as e:
    print(f"ERROR: Fallo al cargar SD. {e}")
    pipe = None


def generate_image_for_print(input_image: Image, style: str, animation_type: str):
    """
    Función de IA: Estiliza la imagen y devuelve el PNG final (sin animación).
    El argumento 'animation_type' se usa solo en el prompt para guiar el estilo.
    """
    if pipe is None:
        raise gr.Error("Modelo de IA no disponible. Revisa el hardware o los logs.")

    prompt = f"a high-quality animated sticker of a {style} style cartoon, {animation_type}, professional, digital art, no watermark"
    
    init_image = input_image.resize((512, 512)).convert("RGB")
    
    stylized_image = pipe(
        prompt=prompt,
        image=init_image,
        strength=0.75,
        guidance_scale=9.0,
        num_inference_steps=25
    ).images[0].convert("RGB") 

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        png_path = tmp_file.name 
    stylized_image.save(png_path, format='PNG')
    
    return png_path 

iface = gr.Interface(
    fn=generate_image_for_print, 
    inputs=[
        gr.Image(type="pil", label="Sube tu Foto"),
        gr.Dropdown(["Pop-Art", "Sketch", "Cartoon"], label="Estilo Deseado"),
        gr.Dropdown(["subtle_bounce", "head_turn", "text_pop"], label="Tipo de Diseño") 
    ],
    outputs=gr.File(label="PNG de Alta Calidad"), 
    title="Nano Banana Generator IA Engine (PNG)",
    description="Motor de IA para estilización de imágenes en alta resolución."
)

iface.launch()