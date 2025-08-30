import torch
import torch.nn as nn

X = torch.tensor([
    [35, 8, 3, 28],
    [30, 9, 4, 27],
    [28, 8, 5, 26],
    [32, 10, 3, 29],
    [36, 9, 3, 30],
    [40, 6, 2, 30],
    [38, 8, 2, 28],
    [42, 5, 1, 34],
    [48, 6, 1, 32],
    [37, 7, 2, 29],
    [50, 3, 0, 35],
    [60, 4, 1, 45],
    [55, 2, 0, 31],
    [62, 3, 0, 40],
    [58, 2, 1, 38]
], dtype=torch.float32)

X = X / torch.tensor([70.0, 10.0, 7.0, 100.0])

y = torch.tensor([
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2
], dtype=torch.long)

model = nn.Sequential(nn.Linear(4, 3))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10000):
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 500 == 0:
        print(f"Época {epoch+1}, Pérdida: {loss.item():.6f}")

ht = float(input("Horas trabajadas por semana: ")) / 70.0
sueno = float(input("Calidad del sueño (1–10): ")) / 10.0
act = float(input("Días de actividad física por semana: ")) / 7.0
edad = float(input("Edad: ")) / 100.0
entrada = torch.tensor([[ht, sueno, act, edad]])

salida = model(entrada)
clas = torch.argmax(salida, dim=1).item()
niveles = ["Bajo", "Medio", "Alto"]
print(f"Nivel de estrés estimado: {niveles[clas]}")