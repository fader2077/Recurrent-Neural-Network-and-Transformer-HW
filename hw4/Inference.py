from PIL import Image
import requests

# Load an image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw)

# Prompt
prompt = "USER: <image>\nDescribe the main object in this image.\nASSISTANT:"

inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=50)
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(output)