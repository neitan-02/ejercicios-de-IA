from flask import Flask, request, jsonify
from flask_cors import CORS
import os   
import tempfile
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

app = Flask(__name__)
CORS(app)

# Cargar modelo una vez al iniciar
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

def audio_to_text(audio_path):
    try:
        waveform, sr = librosa.load(audio_path, sr=16000)
        input_audio = torch.tensor(waveform).unsqueeze(0)
        inputs = processor(input_audio.squeeze(0), sampling_rate=16000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(predicted_ids)[0].lower()
    except Exception as e:
        raise Exception(f"Error en audio_to_text: {str(e)}")

def parse_email_command(text):
    try:
        corrections = {
            "correr": "correo",
            "correoo": "correo",
            "imeil": "email",
            "corre": "correo",
            "correo a": "a",
            "correo asunto": "asunto"
        }
        
        text = text.lower().strip()
        for error, correction in corrections.items():
            text = text.replace(error, correction)
        
        recipients = []
        parts = [p.strip() for p in text.split("a ") if p.strip()]
        
        for part in parts:
            if ":" in part:
                name_part, message = part.split(":", 1)
                name_part = name_part.strip()
                message = message.strip()
            else:
                name_part = part
                message = ""
            
            email = ""
            name_lower = name_part.lower()
            if "laura" in name_lower:
                email = "al22221069@gmail.com"
            elif "enrique" in name_lower:
                email = "hernandez@gmail.com"
            elif "sofia" in name_lower:
                email = "al2221876@gmail.com"
            
            if email:
                recipients.append({
                    "name": name_part,
                    "email": email,
                    "message": message
                })
        
        return {
            "subject": "Entrega de proyecto",  # Asunto fijo
            "details": recipients
        }
    except Exception as e:
        raise Exception(f"Error en parse_email_command: {str(e)}")

@app.route('/process-audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    temp_path = os.path.join(tempfile.gettempdir(), audio_file.filename)
    audio_file.save(temp_path)
    
    try:
        text = audio_to_text(temp_path)
        result = parse_email_command(text)
        os.remove(temp_path)
        return jsonify(result)
    except Exception as e:
        os.remove(temp_path)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)