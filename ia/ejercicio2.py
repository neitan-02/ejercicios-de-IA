import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# Los Datos de mi tremendo entrenamiento son: [tiempo_espera, problemas_resueltos, actitud_personal] -> satisfacción (0-10)
X_train = torch.tensor([
    [10.0, 2.0, 4.0],
    [5.0, 5.0, 5.0],
    [20.0, 1.0, 2.0]
], dtype=torch.float32)

y_train = torch.tensor([8.0, 10.0, 3.0], dtype=torch.float32)

# Definición del modelo
class SatisfactionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 32)
        self.layer2 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)
        x = torch.relu(self.layer2(x))
        return self.output(x)

# Instanciación, pérdida y optimizador
model = SatisfactionModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Normalización de características
mean = X_train.mean(dim=0)
std = X_train.std(dim=0)
X_train_normalized = (X_train - mean) / std

# Entrenamiento
print("Entrenando modelo...")
for epoch in range(3000):
    optimizer.zero_grad()
    outputs = model(X_train_normalized)
    loss = criterion(outputs.squeeze(), y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 300 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Función de predicción
def predict_satisfaction():
    print("\n=== Predicción de Satisfacción ===")
    try:
        wait_time = float(input("Tiempo de espera (minutos): "))
        solved = float(input("Problemas resueltos: "))
        attitude = float(input("Actitud del personal (1-5): "))
        
        # Normalización
        input_data = torch.tensor([[wait_time, solved, attitude]], dtype=torch.float32)
        input_normalized = (input_data - mean) / std
        
        with torch.no_grad():
            prediction = model(input_normalized).item()
            prediction = max(0, min(10, round(prediction)))
        
        print(f"\nPredicción: {prediction}/10 puntos de satisfacción")
    except ValueError:
        print("Error: Ingrese valores numéricos válidos")

# Ejemplos de prueba
print("\n=== Ejemplos de predicción ===")
test_cases = torch.tensor([
    [12.0, 3.0, 4.0],
    [7.0, 4.0, 5.0],
    [25.0, 2.0, 1.0]
], dtype=torch.float32)

test_normalized = (test_cases - mean) / std

with torch.no_grad():
    predictions = model(test_normalized).squeeze()

for i, (data, pred) in enumerate(zip(test_cases, predictions)):
    pred_rounded = max(0, min(10, round(pred.item())))
    print(f"Caso {i+1}: {data[0]:.0f} min, {data[1]:.0f} problemas, actitud {data[2]:.0f} -> {pred_rounded}/10")

# Interacción con usuario
while True:
    predict_satisfaction()
    if input("\n¿Predecir otro caso? (si/no): ").lower() != 'si':
        break