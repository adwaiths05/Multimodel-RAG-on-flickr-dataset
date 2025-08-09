import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import argparse
import json
import os

# Load preprocessed dataset
def load_dataset(data_dir):
    captions_file = os.path.join(data_dir, "processed/captions.json")
    embeddings_file = os.path.join(data_dir, "processed/image_embeddings.npy")
    with open(captions_file, 'r') as f:
        captions_data = json.load(f)
    image_embeddings = np.load(embeddings_file)
    return captions_data, image_embeddings

# Initialize models
clip_model = SentenceTransformer('clip-ViT-B-32')
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Build FAISS index
def build_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Retrieve relevant image-text pairs
def retrieve(query, index, captions_data, image_embeddings, top_k=5):
    query_embedding = clip_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    retrieved = [captions_data[i] for i in indices[0]]
    return retrieved

# Generate caption for input image
def generate_caption(image_path, retrieved_data):
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt")
    
    # Augment with retrieved context
    context = " ".join([caption for item in retrieved_data for caption in item['captions']])
    inputs["text"] = f"Describe the image with focus on entities: {context}"
    
    outputs = blip_model.generate(**inputs, max_new_tokens=50)
    caption = blip_processor.decode(outputs[0], skip_special_tokens=True)
    return caption

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True, help="Path to input image")
    parser.add_argument("--query", default="describe entities", help="Text query for retrieval")
    parser.add_argument("--data_dir", default="data/flickr30k_entities", help="Directory with dataset")
    args = parser.parse_args()

    # Load dataset
    captions_data, image_embeddings = load_dataset(args.data_dir)
    
    # Build FAISS index
    index = build_index(image_embeddings)
    
    # Retrieve relevant data
    retrieved_data = retrieve(args.query, index, captions_data, image_embeddings)
    
    # Generate caption
    caption = generate_caption(args.image_path, retrieved_data)
    print(f"Generated Caption: {caption}")

if __name__ == "__main__":
    main()
