import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softmax
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# Datos de entrenamiento con reglas estrictas
X_train = torch.tensor([
    # Tiempos ≤ 10 minutos (5 estrellas si resuelto)
    [5.0, 1.0, 25.0, 0.0], [10.0, 1.0, 30.0, 1.0], [2.0, 1.0, 40.0, 2.0],
    [8.0, 1.0, 35.0, 0.0], [7.0, 1.0, 28.0, 1.0], [10.0, 1.0, 50.0, 2.0],
    # Tiempos 11-20 minutos (4 estrellas si resuelto)
    [15.0, 1.0, 45.0, 1.0], [18.0, 1.0, 32.0, 0.0], [12.0, 1.0, 38.0, 1.0],
    # Tiempos 21-40 minutos (3 estrellas si resuelto, 1-2 si no)
    [25.0, 1.0, 42.0, 2.0], [30.0, 1.0, 35.0, 0.0], [35.0, 0.0, 40.0, 1.0],
    # Tiempos 41-60 minutos (2 estrellas si resuelto, 1 si no)
    [45.0, 1.0, 30.0, 2.0], [50.0, 0.0, 45.0, 0.0], [60.0, 1.0, 28.0, 1.0],
    # Tiempos >60 minutos (1 estrella siempre)
    [70.0, 1.0, 35.0, 2.0], [90.0, 0.0, 40.0, 1.0], [120.0, 1.0, 50.0, 0.0],
    [200.0, 0.0, 30.0, 2.0], [85.0, 1.0, 45.0, 1.0]
], dtype=torch.float32)

# Asignación estricta según reglas de negocio
y_train = torch.tensor([
    4, 4, 4, 4, 4, 4,  # 5 estrellas (clase 4)
    3, 3, 3,           # 4 estrellas (clase 3)
    2, 2, 0,           # 3 estrellas (clase 2) o 1 (si no resuelto)
    1, 0, 1,           # 2 estrellas (clase 1) o 1 (si no resuelto)
    0, 0, 0, 0, 0      # 1 estrella siempre (clase 0)
], dtype=torch.long)

# Normalización robusta
median = X_train.median(dim=0).values
iqr = torch.quantile(X_train, 0.75, dim=0) - torch.quantile(X_train, 0.25, dim=0)
X_train_normalized = (X_train - median) / (iqr + 1e-8)

class StrictSatisfactionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_layer = nn.Linear(1, 16)  # Capa específica para tiempo
        self.other_layer = nn.Linear(3, 32)  # Capa para otras características
        self.combined = nn.Linear(48, 64)
        self.output = nn.Linear(64, 5)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        time = x[:, 0:1]  # Separar tiempo de espera
        other = x[:, 1:]   # Otras características
        
        # Procesar tiempo con mayor peso
        time_feat = torch.relu(self.time_layer(time * 2.0))  # Ponderación aumentada
        
        # Procesar otras características
        other_feat = torch.relu(self.other_layer(other))
        
        # Combinar
        combined = torch.cat([time_feat, other_feat], dim=1)
        combined = torch.relu(self.combined(combined))
        combined = self.dropout(combined)
        return self.output(combined)

model = StrictSatisfactionModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)

print("Entrenando modelo con reglas estrictas...")
for epoch in range(2500):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_normalized)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    if epoch % 250 == 0:
        with torch.no_grad():
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_train).float().mean()
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2%}')

def enforce_business_rules(wait_time, resolved, raw_pred):
    """Aplica reglas de negocio estrictas a la predicción"""
    if wait_time <= 10 and resolved == 1:
        return 4  # 5 estrellas (clase 4)
    elif wait_time > 60:
        return 0  # 1 estrella (clase 0)
    elif 11 <= wait_time <= 20 and resolved == 1:
        return 3  # 4 estrellas
    elif 21 <= wait_time <= 40:
        return 2 if resolved == 1 else min(raw_pred, 1)  # 3 estrellas o menos
    elif 41 <= wait_time <= 60:
        return 1 if resolved == 1 else 0  # 2 estrellas o 1
    return raw_pred

def predict_satisfaction():
    print("\n=== Predicción con Reglas Estrictas ===")
    try:
        wait_time = float(input("Tiempo de espera (minutos): "))
        resolved = int(input("¿Problema resuelto? (1=Sí, 0=No): "))
        age = int(input("Edad del cliente: "))
        service_type = int(input("Tipo de servicio (0-2): "))
        
        input_data = torch.tensor([[wait_time, resolved, age, service_type]], dtype=torch.float32)
        input_normalized = (input_data - median) / (iqr + 1e-8)
        
        with torch.no_grad():
            model.eval()
            output = model(input_normalized)
            prob = softmax(output, dim=1)
            _, raw_pred = torch.max(output.data, 1)
            raw_pred = raw_pred.item()
        
        # Aplicar reglas de negocio
        final_pred = enforce_business_rules(wait_time, resolved, raw_pred)
        satisfaction = final_pred + 1  # Convertir a escala 1-5
        
        print(f"\nPredicción: Nivel de satisfacción {satisfaction}/5")
        
        print("\nProbabilidades del modelo:")
        for i in range(5):
            print(f"{i+1} estrellas: {prob[0][i]:.2%}")
            
        # Mostrar advertencias si el modelo no coincide con reglas
        if raw_pred != final_pred:
            print(f"\nNOTA: El modelo predijo inicialmente {raw_pred + 1} estrellas")
            print("Se aplicaron reglas de negocio para ajustar la predicción")
            
    except ValueError:
        print("Error: Ingrese valores válidos")

# Casos de prueba
print("\n=== Pruebas del Sistema ===")
test_cases = [
    (5.0, 1, 30, 0),   # Debe ser 5 estrellas
    (10.0, 1, 35, 1),   # Debe ser 5 estrellas
    (15.0, 1, 40, 2),   # Debe ser 4 estrellas
    (30.0, 1, 45, 0),   # Debe ser 3 estrellas
    (25.0, 0, 30, 1),   # Debe ser 1-2 estrellas
    (50.0, 1, 50, 2),   # Debe ser 2 estrellas
    (70.0, 1, 35, 1),   # Debe ser 1 estrella
    (100.0, 0, 40, 2)   # Debe ser 1 estrella
]

for i, (wait, res, age, serv) in enumerate(test_cases):
    input_data = torch.tensor([[wait, res, age, serv]], dtype=torch.float32)
    input_normalized = (input_data - median) / (iqr + 1e-8)
    
    with torch.no_grad():
        model.eval()
        output = model(input_normalized)
        _, raw_pred = torch.max(output.data, 1)
        raw_pred = raw_pred.item()
    
    final_pred = enforce_business_rules(wait, res, raw_pred)
    
    print(f"\nCaso {i+1}: {wait} min, Resuelto: {res}, Edad: {age}, Tipo: {serv}")
    print(f"Predicción final: {final_pred + 1} estrellas (Modelo: {raw_pred + 1})")

# Interacción
while True:
    predict_satisfaction()
    resp = input("\n¿Predecir otro caso? (s/n): ").lower()
    if resp != 's':
        print("¡Gracias por usar el predictor con reglas estrictas!")
        break