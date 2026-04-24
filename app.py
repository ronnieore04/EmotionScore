import os
import json
import re
from flask import Flask, request, jsonify, render_template
import anthropic

app = Flask(__name__)
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are an expert emotion analysis system. When given text, analyze it and return ONLY a JSON object with this exact structure — no explanation, no markdown, just raw JSON:

{
  "emotions": {
    "joy": 0.0,
    "sadness": 0.0,
    "anger": 0.0,
    "fear": 0.0,
    "surprise": 0.0,
    "disgust": 0.0,
    "neutral": 0.0
  },
  "dominant_emotion": "emotion_name",
  "sentiment": "positive" | "negative" | "neutral" | "mixed",
  "explanation": "One concise sentence explaining the emotional tone."
}

Rules:
- All emotion scores must be between 0.0 and 1.0
- Scores should reflect relative intensity; they do NOT need to sum to 1
- dominant_emotion must be the key with the highest score
- explanation must be under 25 words
- Return ONLY the JSON, nothing else"""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = (data or {}).get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400
    if len(text) > 2000:
        return jsonify({"error": "Text too long. Max 2000 characters."}), 400

    try:
        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Analyze this text:\n\n{text}"}],
        )

        raw = message.content[0].text.strip()

        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        result = json.loads(raw)
        return jsonify(result)

    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse emotion data. Please try again."}), 500
    except anthropic.APIError as e:
        return jsonify({"error": f"API error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
