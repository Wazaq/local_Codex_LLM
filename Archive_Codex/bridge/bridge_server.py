from flask import Flask, request, jsonify
import requests
import json

# Configuration
BRIDGE_PORT = 8080
OLLAMA_URL = "http://localhost:11434/api/generate"
MEMORY_API_URL = "https://neural-nexus-palace.wazaqglim.workers.dev"

app = Flask(__name__)

def load_personality_data():
      # Use actual AIL semantic search instead
      payload = {
          "query": "Codex personality traits communication problem solving",
          "domains": ["Codex Personality Room"],
          "top_k": 10,
          "include_content": True
      }
      response = requests.post(f"{MEMORY_API_URL}/semantic-search", json=payload)
      if response.status_code == 200:
          return response.json()
      else:
          raise Exception(f"Failed to load personality data: {response.status_code}")

def fetch_memories():
    # Fetch conversation memories from the external memory system
    response = requests.get(f"{MEMORY_API_URL}/conversations")
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Failed to fetch memories")

@app.route('/send_message', methods=['POST'])
def send_message():
    try:
        # Load personality data
        personality_data = load_personality_data()

        # Fetch memories
        memories = fetch_memories()

        # Prepare the context for Qwen
        context = f"Personality Data: {json.dumps(personality_data)}\nMemories:\n"
        for memory in memories:
            context += f"{memory['timestamp']}: {memory['user']} - {memory['message']}\n"

        # Get user message from the request
        data = request.json
        user_message = data.get('message')

        # Prepare the full prompt to send to Qwen
        prompt = f"{context}\nUser: {user_message}\nQwen:"

        # Send the prompt to Qwen
        qwen_response = requests.post(OLLAMA_URL, json={"prompt": prompt})
        if qwen_response.status_code == 200:
            response_data = qwen_response.json()
            qwen_reply = response_data.get('response', '')

            # Log the user message and Qwen's reply to the external memory system
            log_message(user_message, "Brent")
            log_message(qwen_reply, "Codex")

            return jsonify({"response": qwen_reply})
        else:
            raise Exception("Failed to get response from Qwen")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def log_message(message, user):
    # Log the message to the external memory system
    payload = {
        "user": user,
        "message": message
    }
    requests.post(f"{MEMORY_API_URL}/log", json=payload)

if __name__ == '__main__':
    app.run(port=BRIDGE_PORT, debug=True)
