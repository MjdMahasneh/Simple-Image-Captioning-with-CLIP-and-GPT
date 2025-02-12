from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import openai

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load an image
image_url = "https://example.com/path/to/your/image.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)

# Use CLIP to extract image features
inputs = processor(images=image, return_tensors="pt", padding=True)
image_features = model.get_image_features(**inputs)  # Shape: [1, 512]

# Convert image features to a text prompt (simplified)
# Here, we assume the feature vector is a "summary" of the image
image_prompt = "The image contains: " + " ".join([str(f) for f in image_features[0].tolist()])

# Use GPT-4 to generate a caption
openai.api_key = "your_openai_api_key"
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that generates captions for images."},
        {"role": "user", "content": image_prompt}
    ]
)

# Print the generated caption
caption = response['choices'][0]['message']['content']
print("Generated Caption:", caption)