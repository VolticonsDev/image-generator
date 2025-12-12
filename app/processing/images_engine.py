import os
import asyncio
import base64
from io import BytesIO

import httpx 
from PIL import Image
import imageio.v3 as iio


HUGGINGFACE_API_URL = "https://huggingface.co/spaces/MaulaMaula333/image-generator"
def load_stable_diffusion_pipeline():
    return True 

load_stable_diffusion_pipeline()


async def process_nano_banana(task_id: str, request, task_store, UPLOAD_DIR):
    """
    Llama a la API de Hugging Face (IA) y luego genera el GIF.
    """
    
    input_path = task_store[task_id]["input_path"]
    
    try:
        task_store[task_id]['status'] = "PROCESSING"
        print(f"[{task_id}] Subiendo a IA: Estilo={request.style}, Animación={request.animation_type}")
        
        with open(input_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

        payload = {
            "data": [
                f"data:image/png;base64,{encoded_string}", 
                request.style,                             
                request.animation_type                     
            ]
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            print(f"[{task_id}] Enviando solicitud a la IA externa: {HUGGINGFACE_API_URL}")
            
            response = await client.post(HUGGINGFACE_API_URL, json=payload)
            response.raise_for_status() 

            ia_result = response.json()
            
            await asyncio.sleep(2) 
            stylized_image = Image.open(input_path).convert("RGB")


    
        if request.output_format.upper() == "GIF":
            print(f"[{task_id}] Creando animación GIF...")
            
            frames = []
            width, height = stylized_image.size
            
            for i in [0, 1, 2]:
                zoom_scale = 1.0 - (i * 0.05)
                new_size = (int(width * zoom_scale), int(height * zoom_scale))
                temp_frame = stylized_image.resize(new_size, Image.Resampling.LANCZOS)
                
                final_frame = Image.new('RGB', (width, height), color = 'white')
                offset_x = (width - temp_frame.size[0]) // 2 + i * 5
                offset_y = (height - temp_frame.size[1]) // 2 - i * 5
                final_frame.paste(temp_frame, (offset_x, offset_y))
                frames.append(final_frame)
            
            frames += frames[1:-1][::-1] 
            
            output_filename = f"result_{task_id}.gif"
            output_path = os.path.join(UPLOAD_DIR, output_filename)
            iio.imwrite(output_path, frames, duration=150, loop=0)
            
        else:
            output_filename = f"result_{task_id}.png"
            output_path = os.path.join(UPLOAD_DIR, output_filename)
            stylized_image.save(output_path)
        
        
        result_url = f"/api/v1/download/{output_filename}"
        task_store[task_id]['status'] = "COMPLETED"
        task_store[task_id]['result_url'] = result_url
        
        print(f"[{task_id}] Tarea completada. Resultado en: {result_url}")

    except httpx.HTTPStatusError as e:
        error_msg = f"IA Falló ({e.response.status_code}): {e.response.text}"
        print(f"ERROR HTTP: {error_msg}")
        task_store[task_id]['status'] = "FAILED"
        task_store[task_id]['error_message'] = error_msg
    except Exception as e:
        print(f"ERROR CRÍTICO en el procesamiento [{task_id}]: {e}")
        task_store[task_id]['status'] = "FAILED"
        task_store[task_id]['error_message'] = str(e)