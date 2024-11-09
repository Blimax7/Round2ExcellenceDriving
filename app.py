from flask import Flask, render_template, request, jsonify, session  
import openai  
import os  
import faiss  
import numpy as np  
from dotenv import load_dotenv  
import pickle  
import uuid  
  
# Load environment variables from .env file  
load_dotenv()  
  
# Set your OpenAI API key from the environment variable  
openai.api_key = os.getenv("OPENAI_API_KEY")  
  
app = Flask(__name__)  
app.secret_key = os.getenv("SECRET_KEY")  # Set your secret key here  
app.config['SESSION_TYPE'] = 'filesystem'  
  
# Path to store cached embeddings  
EMBEDDINGS_CACHE_FILE = 'embeddings_cache.pkl'  
  
# Store user queries and their embeddings in a dictionary for caching  
embedding_cache = {}  
  
# Store user queries in a dictionary  
user_queries = {}  
  
def load_embeddings_from_cache():  
   if os.path.exists(EMBEDDINGS_CACHE_FILE):  
      with open(EMBEDDINGS_CACHE_FILE, 'rb') as f:  
        return pickle.load(f)  
   return None, None  
  
def get_embedding(text, model="text-embedding-ada-002"):  
   if not isinstance(text, str):  
      raise ValueError("Invalid input for embedding: input must be a string")  
   if len(text) == 0:  
      raise ValueError("Invalid input for embedding: input cannot be empty")  
  
   # Check if the embedding is already cached  
   if text in embedding_cache:  
      return embedding_cache[text]  
  
   # Use OpenAI's embedding library to get the embedding  
   response = openai.Embedding.create(input=text, model=model)  
   embedding = response['data'][0]['embedding']  
  
   # Cache the embedding for future use  
   embedding_cache[text] = embedding  
  
   return embedding  
  
@app.route('/')  
def index():  
   return render_template('index.html')  
  
@app.route('/ask', methods=['POST'])  
def ask():  
   data = request.get_json()  
   user_query = data.get('query')  
  
   if not user_query:  
      return jsonify({'response': "Invalid query: query cannot be empty"})  
  
   # Generate a unique user ID for the session  
   user_id = session.get('user_id')  
   if not user_id:  
      user_id = str(uuid.uuid4())  
      session['user_id'] = user_id  
  
   # Store the user query in the dictionary  
   if user_id not in user_queries:  
      user_queries[user_id] = []  
   user_queries[user_id].append(user_query)  
  
   # Check if the user is asking about their previous question  
   if user_query.lower() == "what was my previous question?":  
      previous_question = user_queries[user_id][-2] if len(user_queries[user_id]) > 1 else None  
      if previous_question:  
        return jsonify({'response': f"Your previous question was: {previous_question}"})  
      else:  
        return jsonify({'response': "I don't have any previous questions."})  
  
   # Load cached embeddings if available  
   faiss_index, sections = load_embeddings_from_cache()  
  
   if faiss_index is None or sections is None:  
      return jsonify({'response': "Embeddings not found. Please generate them first."})  
  
   # Get embedding for user query using the cache or generate new one  
   try:  
      query_embedding = get_embedding(user_query)  
  
      # Retrieve relevant content using FAISS  
      distances, indices = faiss_index.search(np.array([query_embedding], dtype=np.float32), k=3)  
  
      # Gather the most relevant sections based on FAISS index results  
      relevant_sections = [sections[i] for i in indices[0] if i != -1]  
  
      if relevant_sections:  
        relevant_content = ' '.join(relevant_sections)  
  
        # Use your custom prompt for generating a response  
        gpt4_response = generate_gpt4_response(relevant_content, user_query)  
  
        return jsonify({'response': gpt4_response})  
      else:  
        return jsonify({'response': "No relevant content found."})  
  
   except Exception as e:  
      return jsonify({'response': f"Error processing query: {e}"})  
  
def generate_gpt4_response(context, query):  
   if not context:  
      return "Invalid context: context cannot be empty"  
  
   prompt = f"""  
You are an AI assistant created by Excellence Driving to provide prompt and helpful customer service. Your knowledge is limited to information about Excellence Driving, its driving classes, licensing services, policies, and company details.  
  
Relevant context: {context}  
  
**Excellence Driving Services:**  
1. Driving Classes: Beginner, advanced defensive, specialized (seniors/teens), refresher courses.  
2. Licensing Services: Learner's permit, written test prep, behind-the-wheel prep, license renewal.  
  
**Excellence Driving Policies:**  
- 48-hour advance booking required  
- Late cancellation fee may apply  
- Valid learner's permit required for behind-the-wheel lessons  
- Payment due at booking  
  
When responding to user queries:  
1. Provide clear, concise, and helpful answers about Excellence Driving's services or policies.  
2. For specific driving techniques, offer general guidance and recommend discussing details with an instructor.  
3. For non-Excellence Driving topics, politely redirect to the local Department of Motor Vehicles.  
  
IMPORTANT: Keep your responses brief and to the point. Aim for no more than 2-3 short sentences unless absolutely necessary.  
  
User Query: {query}  
  
How may I assist you with Excellence Driving's services?  
"""  
  
   response = openai.ChatCompletion.create(  
      model="gpt-4o",  # Changed here from "gpt-4" to "gpt-4o"  
      messages=[  
        {"role": "system", "content": prompt},  
        {"role": "user", "content": query}  
      ]  
   )  
  
   return response['choices'][0]['message']['content']  
  
if __name__ == '__main__':  
   app.run(debug=True)