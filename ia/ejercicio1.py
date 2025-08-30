import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Configuración reproducible
torch.manual_seed(42)
np.random.seed(42)

# Mis datos de entrada son los sig: [ingresos, tarjetas] -> riesgo (0:bajo, 1:medio, 2:alto)
X_train = torch.tensor([
    [5000., 1.],  # bajo
    [8000., 2.],  # bajo
    [2000., 4.],  # medio
    [7000., 6.],  # alto
    [3000., 5.],  # alto (nuevo)
    [10000., 3.], # medio (nuevo)
    [1500., 7.],  # alto (nuevo)
    [9000., 1.],  # bajo (nuevo)
], dtype=torch.float32)

y_train = torch.tensor([0, 0, 1, 2, 2, 1, 2, 0], dtype=torch.long)

# Normalización de características
mean = X_train.mean(dim=0)
std = X_train.std(dim=0)
X_train = (X_train - mean) / std

class RiesgoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 16)
        self.layer2 = nn.Linear(16, 16)
        self.output = nn.Linear(16, 3)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)
        x = torch.relu(self.layer2(x))
        return self.output(x)

model = RiesgoModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Entrenamiento 
for epoch in range(5000):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

def predict_risk():
    print("\n=== Predicción de Riesgo Crediticio ===")
    try:
        income = float(input("Ingrese sus ingresos mensuales ($): "))
        cards = float(input("Ingresa sus números de tarjetas de crédito: "))
        
        # Normalización
        input_data = torch.tensor([[income, cards]], dtype=torch.float32)
        input_data = (input_data - mean) / std
        
        with torch.no_grad():
            output = model(input_data)
            _, predicted = torch.max(output.data, 1)
        
        risk_levels = ["BAJO (0)", "MEDIO (1)", "ALTO (2)"]
        print(f"\nResultado: Riesgo {risk_levels[predicted.item()]}")
        
        # Mostrar lógica de decisión
        prob = torch.softmax(output, dim=1)
        print(f"Probabilidades: Bajo: {prob[0][0].item():.2%}, Medio: {prob[0][1].item():.2%}, Alto: {prob[0][2].item():.2%}")
        
    except ValueError:
        print("Error: Ingrese valores numéricos válidos")

# Casos de prueba MEJORADOS
print("\n=== Casos de prueba mejorados ===")
test_cases = torch.tensor([
    [10000., 4.],  # Debería ser medio/alto
    [2000., 2.],   # Bajo/medio
    [5000., 6.],   # Alto
    [30000., 1.],  # Bajo
], dtype=torch.float32)

test_normalized = (test_cases - mean) / std

with torch.no_grad():
    outputs = model(test_normalized)
    probs = torch.softmax(outputs, dim=1)
    _, preds = torch.max(outputs.data, 1)

for i, (data, pred, prob) in enumerate(zip(test_cases, preds, probs)):
    print(f"\nCaso {i+1}: Ingresos ${data[0]:.0f}, {data[1]:.0f} tarjetas")
    print(f"Predicción: {['BAJO','MEDIO','ALTO'][pred.item()]}")
    print(f"Probabilidades: B:{prob[0].item():.2%} M:{prob[1].item():.2%} A:{prob[2].item():.2%}")

# Interacción
while True:
    predict_risk()
    if input("\n¿Predecir otro cliente? (si/no): ").lower() != 's':
        break