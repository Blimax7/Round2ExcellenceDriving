import os
import openai
import faiss
import numpy as np
from dotenv import load_dotenv
import pickle

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key from the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Path to the scraped data folder
SCRAPED_DATA_FOLDER = r'C:\Users\sohel\OneDrive\Desktop\AgenticRagChatbot\scraped_data'

# Path to store cached embeddings
EMBEDDINGS_CACHE_FILE = r'C:\Users\sohel\OneDrive\Desktop\AgenticRagChatbot\embeddings_cache.pkl'

def get_embedding(text, model="text-embedding-ada-002"):
    if isinstance(text, str) and len(text) > 0:
        response = openai.Embedding.create(input=[text], model=model)
        embedding = response['data'][0]['embedding']
        return embedding
    else:
        raise ValueError(f"Invalid input for embedding: {text}")

def store_multiple_embeddings_in_faiss(text_sections):
    if not text_sections:
        raise ValueError("No valid text sections to embed.")
        
    dimension = len(get_embedding(text_sections[0]))  # Get the dimension of the embeddings
    index = faiss.IndexFlatL2(dimension)
    
    embeddings = []
    for section in text_sections:
        try:
            embedding = get_embedding(section)
            embeddings.append(embedding)
        except ValueError as e:
            print(f"Skipping invalid section: {e}")
    
    if embeddings:
        embeddings_np = np.array(embeddings, dtype=np.float32)
        index.add(embeddings_np)
    else:
        raise ValueError("No valid embeddings generated.")
    
    return index, text_sections

def load_scraped_text_from_folder(folder_path):
    text_sections = []
    try:
        # Iterate through all files in the specified folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):  # Process only .txt files
                file_path = os.path.join(folder_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.readlines()
                        # Append non-empty lines to text_sections
                        text_sections.extend([line.strip() for line in content if line.strip()])
                except UnicodeDecodeError:
                    # Fallback to latin-1 encoding if utf-8 fails
                    with open(file_path, 'r', encoding='latin-1') as file:
                        content = file.readlines()
                        text_sections.extend([line.strip() for line in content if line.strip()])
    except Exception as e:
        print(f"Error loading text files from folder: {e}")
    
    return text_sections

def save_embeddings_to_cache(faiss_index, text_sections):
    with open(EMBEDDINGS_CACHE_FILE, 'wb') as f:
        pickle.dump((faiss_index, text_sections), f)

if __name__ == '__main__':
    # Load text from all files in the scraped data folder
    website_text_sections = load_scraped_text_from_folder(SCRAPED_DATA_FOLDER)

    if website_text_sections:
        try:
            # Store embeddings for multiple sections in FAISS and save them
            faiss_index, sections = store_multiple_embeddings_in_faiss(website_text_sections)
            save_embeddings_to_cache(faiss_index, sections)
            print("Embeddings created and cached successfully.")
        except ValueError as e:
            print(f"Error creating embeddings: {e}")
    else:
        print("Failed to process any text files.")