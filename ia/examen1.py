from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Configuración de prioridades
IMPORTANCE_WEIGHT = 0.6
URGENCY_WEIGHT = 0.3
DURATION_WEIGHT = 0.1

class PriorityModel:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def predict(self, importance, days_left, duration):
        """
        Prioridades basadas en:
        - Importancia (1-5): 5=alta, 4=alta, 3=media, 2=baja, 1=baja
        - Urgencia (días restantes): menos días = más urgente
        - Duración: tareas más cortas = mayor prioridad
        """
        # Normalizar los valores
        importance_norm = (importance - 1) / 4  # Escalar 1-5 a 0-1
        urgency_norm = 1 / max(1, days_left)    # Más urgente cuando días_left es pequeño
        duration_norm = 1 / max(0.5, duration)  # Priorizar tareas más cortas
        
        # Calcular puntuación combinada
        score = (IMPORTANCE_WEIGHT * importance_norm + 
                URGENCY_WEIGHT * urgency_norm + 
                DURATION_WEIGHT * duration_norm)
        
        # Asignar prioridad según reglas específicas
        if importance >= 4 and days_left <= 2:
            return "alta"
        elif importance >= 3 and days_left <= 5:
            return "alta"
        elif importance == 5:
            return "alta"
        elif importance == 4:
            return "alta" if score > 0.6 else "media"
        elif importance == 3:
            return "media"
        elif importance == 2:
            return "baja" if days_left > 3 else "media"
        else:  # importance == 1
            return "baja"

# Creamos una instancia del modelo
model = PriorityModel()

@app.route('/prioritize', methods=['POST'])
def prioritize():
    try:
        data = request.json
        
        # Validar entrada
        if not data or 'dueDate' not in data or 'importance' not in data or 'duration' not in data:
            return jsonify({'error': 'Datos incompletos'}), 400
        
        # Convertir y validar fecha
        try:
            if 'T' in data['dueDate']:
                due_date = datetime.strptime(data['dueDate'].split('T')[0], '%Y-%m-%d')
            else:
                due_date = datetime.strptime(data['dueDate'], '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': 'Formato de fecha inválido. Use YYYY-MM-DD'}), 400
            
        days_left = max(0, (due_date - datetime.now()).days)
        
        # Validar importancia (1-5)
        importance = min(max(1, int(data['importance'])), 5)
        
        # Validar duración (>0)
        duration = max(0.5, float(data['duration']))
        
        # Predecir prioridad con reglas mejoradas
        priority = model.predict(importance, days_left, duration)
        
        return jsonify({
            'priority': priority,
            'details': {
                'importance': importance,
                'days_left': days_left,
                'duration': duration,
                'score_components': {
                    'importance_weight': IMPORTANCE_WEIGHT,
                    'urgency_weight': URGENCY_WEIGHT,
                    'duration_weight': DURATION_WEIGHT
                }
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)