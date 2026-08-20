import os
import time
from openai import OpenAI
from PIL import Image
from io import BytesIO
import base64

client = OpenAI(api_key="")  # Add your API key here

input_dir = r"C:\Projetos\hand-bone\boneage-test-dataset\boneage-test-dataset"
output_dir = os.path.join(r"C:\Projetos\hand-bone", "output_cleaned_mult")
SIZE = (1024, 1024)
N_SYNTHETIC = 3  # Number of synthetic images to generate per original

os.makedirs(output_dir, exist_ok=True)

prompt = (
    "Enhance this pediatric hand X-ray by digitally removing non-anatomical artifacts or labels outside the hand region, "
    "while preserving all anatomical details and ensuring the radiograph remains realistic and diagnostically useful."
)

# Prepare mask once
mask = Image.new("RGBA", SIZE, (255, 255, 255, 255))
mask_bytes = BytesIO()
mask.save(mask_bytes, format="PNG")
mask_bytes.seek(0)
mask_bytes.name = "mask.png"

image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".png")]

for idx, filename in enumerate(image_files):
    # Subdirectory for each original image
    base_name = os.path.splitext(filename)[0]
    img_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(img_output_dir, exist_ok=True)
    
    # Check if already processed (all 3 exist)
    already_done = all(
        os.path.exists(os.path.join(img_output_dir, f"{base_name}_cleaned_{i+1}.png"))
        for i in range(N_SYNTHETIC)
    )
    if already_done:
        print(f"Já processado: {base_name}. Pulando...")
        continue

    try:
        print(f"\nProcessando imagem {idx+1}/{len(image_files)}: {filename}")

        img = Image.open(os.path.join(input_dir, filename)).convert("RGB").resize(SIZE)
        img_bytes = BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        img_bytes.name = "image.png"

        mask_bytes.seek(0)

        # Request 3 generations in a single call
        result = client.images.edit(
            model="gpt-image-1",
            image=img_bytes,
            mask=mask_bytes,
            prompt=prompt,
            n=N_SYNTHETIC,
            size="1024x1024"
        )

        # Save each generated image
        for i, data in enumerate(result.data):
            image_base64 = data.b64_json
            image_bytes = base64.b64decode(image_base64)
            output_filename = f"{base_name}_cleaned_{i+1}.png"
            output_path = os.path.join(img_output_dir, output_filename)
            with open(output_path, "wb") as f:
                f.write(image_bytes)
            print(f"  Imagem salva em: {output_path}")

    except Exception as e:
        print(f"Erro ao processar {filename}: {e}")

    # Optional: Adjust sleep time to avoid API rate limits
    time.sleep(15)

