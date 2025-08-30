import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softmax

torch.manual_seed(42)

# Datos de entrenamiento mejorados
X_train = torch.tensor([
    [25.0, 70.0],   # Baja
    [60.0, 90.0],   # Media
    [80.0, 120.0],  # Alta
    [45.0, 80.0],   # Media
    [30.0, 100.0],  # Alta (nuevo caso)
    [50.0, 85.0],   # Media (nuevo)
    [35.0, 110.0],  # Alta (nuevo)
    [70.0, 95.0]    # Alta (nuevo)
], dtype=torch.float32)

y_train = torch.tensor([0, 1, 2, 1, 2, 1, 2, 2], dtype=torch.long)

# Normalización de características
mean = X_train.mean(dim=0)
std = X_train.std(dim=0)
X_train_normalized = (X_train - mean) / std

class PriorityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.layer2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 3)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = torch.relu(self.layer2(x))
        return self.output(x)

model = PriorityModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)

print("Entrenando modelo mejorado...")
for epoch in range(4000):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_normalized)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    if epoch % 400 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

def predict_priority():
    print("\n=== Clasificación de Prioridad Mejorada ===")
    try:
        age = int(input("Edad del paciente: "))
        heart_rate = int(input("Frecuencia cardíaca (lpm): "))
        
        # Normalización
        input_data = torch.tensor([[age, heart_rate]], dtype=torch.float32)
        input_normalized = (input_data - mean) / std
        
        with torch.no_grad():
            model.eval()
            output = model(input_normalized)
            prob = softmax(output, dim=1)
            _, predicted = torch.max(output.data, 1)
        
        priority_levels = ["BAJA (0)", "MEDIA (1)", "ALTA (2)"]
        print(f"\nPrioridad de atención: {priority_levels[predicted.item()]}")
        print(f"Probabilidades: Baja: {prob[0][0]:.2%}, Media: {prob[0][1]:.2%}, Alta: {prob[0][2]:.2%}")
        
        # Regla de emergencia adicional
        if heart_rate > 100 and predicted.item() < 2:
            print("\n¡ADVERTENCIA! Frecuencia cardíaca muy alta. Reconsiderar prioridad.")
    except ValueError:
        print("Error: Ingrese valores numéricos válidos")

# Casos de prueba críticos
print("\n=== Casos Críticos de Prueba ===")
critical_cases = torch.tensor([
    [35.0, 100.0],  # Debería ser ALTA
    [25.0, 110.0],  # ALTA (joven con taquicardia)
    [70.0, 85.0],   # MEDIA (edad avanzada pero FC normal)
    [40.0, 130.0]   # ALTA (taquicardia severa)
], dtype=torch.float32)

critical_normalized = (critical_cases - mean) / std

with torch.no_grad():
    model.eval()
    outputs = model(critical_normalized)
    probs = softmax(outputs, dim=1)
    _, preds = torch.max(outputs, 1)

for i, (data, pred, prob) in enumerate(zip(critical_cases, preds, probs)):
    print(f"\nCaso {i+1}: Edad {data[0]:.0f}, FC {data[1]:.0f}")
    print(f"Predicción: {['BAJA','MEDIA','ALTA'][pred.item()]}")
    print(f"Probabilidades: B:{prob[0]:.2%} M:{prob[1]:.2%} A:{prob[2]:.2%}")

# Interacción mejorada
while True:
    predict_priority()
    resp = input("\n¿Evaluar otro paciente? (si/no): ").lower()
    if resp != 'si' and resp != 's':
        print("Sistema de triaje cerrado. ¡Cuida tu salud!")
        break