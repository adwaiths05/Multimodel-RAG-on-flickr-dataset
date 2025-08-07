# Multimodel-RAG-on-flickr-dataset

## Overview

This project implements a multimodal Retrieval-Augmented Generation (RAG) model using the Flickr30k Entities dataset. It retrieves relevant image-text pairs based on a query and generates entity-grounded captions for a given image using a multimodal LLM.

## Features





- Retrieves image-text pairs using CLIP embeddings and FAISS indexing.



- Leverages 244k coreference chains and 276k bounding boxes for entity-specific context.



- Generates captions via a fine-tuned multimodal LLM (e.g., BLIP-2).

## Requirements





- Python 3.8+



- Libraries: torch, transformers, faiss-cpu, sentence-transformers, Pillow



- Flickr30k Entities dataset (images, captions, annotations)
