import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import glob

# Initialize CLIP model
clip_model = SentenceTransformer('clip-ViT-B-32')

def load_flickr30k_data(data_dir):
    # Load captions and annotations (simplified paths)
    captions_file = os.path.join(data_dir, "results_20130124.token")
    annotations_file = os.path.join(data_dir, "Annotations/annotations.json")
    
    # Read captions
    captions_data = []
    with open(captions_file, 'r') as f:
        for line in f:
            img_id, caption = line.strip().split('\t')
            captions_data.append({'image_id': img_id.split('#')[0], 'caption': caption})
    
    # Load annotations (bounding boxes and coreference chains)
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    return captions_data, annotations

def generate_embeddings(data_dir, captions_data):
    image_dir = os.path.join(data_dir, "flickr30k_images")
    image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
    
    image_embeddings = []
    processed_data = []
    
    for img_path in image_files:
        img_id = os.path.basename(img_path)
        # Filter captions for this image
        img_captions = [item for item in captions_data if item['image_id'] == img_id]
        
        if img_captions:
            # Generate image embedding
            img = Image.open(img_path).convert("RGB")
            img_embedding = clip_model.encode(img, convert_to_numpy=True)
            image_embeddings.append(img_embedding)
            
            # Store metadata
            processed_data.append({
                'image_id': img_id,
                'captions': [c['caption'] for c in img_captions],
                'image_path': img_path
            })
    
    return processed_data, np.array(image_embeddings)

def save_processed_data(data_dir, processed_data, image_embeddings):
    output_dir = os.path.join(data_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save captions and metadata
    with open(os.path.join(output_dir, "captions.json"), 'w') as f:
        json.dump(processed_data, f)
    
    # Save image embeddings
    np.save(os.path.join(output_dir, "image_embeddings.npy"), image_embeddings)

def main():
    data_dir = "data/flickr30k_entities"
    captions_data, annotations = load_flickr30k_data(data_dir)
    processed_data, image_embeddings = generate_embeddings(data_dir, captions_data)
    save_processed_data(data_dir, processed_data, image_embeddings)
    print("Preprocessing completed. Data saved in data/flickr30k_entities/processed/")

if __name__ == "__main__":
    main()
