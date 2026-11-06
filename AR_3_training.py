import os
import time
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt



def clear_console(): 
    print("\n" + "="*80 + "\n" + " "*30 + "NOUVELLE EXÉCUTION\n" + "="*80 + "\n")

clear_console()    


SEED = 46
VAL_RATIO = 0.10
HIDDEN_SIZES = [128, 128, 128]
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 100
PATIENCE = 10

max_rings = 5
max_elements = 10
R = max_rings
N_MIN = 0
N_MAX = max_elements
K = N_MAX - N_MIN + 1

np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Appareil utilisé : {device}")



script_dir = os.path.dirname(os.path.abspath(__file__))

Minput_training = np.load(os.path.join(script_dir, 'Minput_training.npy'))
Moutput_training = np.load(os.path.join(script_dir, 'Moutput_training.npy'))

specs = Minput_training.T
architectures = Moutput_training.T.astype(int)
assert architectures.min() >= N_MIN and architectures.max() <= N_MAX, "nombre d'éléments hors de [N_MIN, N_MAX]"


specs_mean = specs.mean(axis=0)
specs_std = specs.std(axis=0)
specs_norm = (specs - specs_mean) / specs_std


nb = specs.shape[0]
rand_indices = np.random.permutation(nb)
nb_val = round(VAL_RATIO * nb)
val_indices = rand_indices[:nb_val]
fit_indices = rand_indices[nb_val:]

X_fit = torch.tensor(specs_norm[fit_indices], dtype=torch.float32)
C_fit = torch.tensor(architectures[fit_indices] - N_MIN, dtype=torch.long)
X_val = torch.tensor(specs_norm[val_indices], dtype=torch.float32).to(device)
C_val = torch.tensor(architectures[val_indices] - N_MIN, dtype=torch.long).to(device)

print(f"Split training : {nb} architectures -> {len(fit_indices)} pour l'apprentissage, "
      f"{len(val_indices)} pour la validation")
print(f"K = {K} classes (n de {N_MIN} à {N_MAX}) ; entrée du MLP : 4 + {R - 1}*{K} + {R} = {4 + (R - 1) * K + R}")



def entree_etape(specs_norm, prefixe, t):
    """Entrée du MLP pour l'anneau t+1 (t = 0..R-1), pour un lot de B architectures.
    specs_norm : (B, 4) spécifications standardisées
    prefixe    : (B, t) classes (n - N_MIN) des anneaux déjà choisis
    renvoie    : (B, 4 + (R-1)*K + R) = [spécifications | 4 cases de K valeurs | one-hot de l'étape]
    La case j contient le one-hot de l'anneau j+1 s'il est déjà choisi (j < t), des zéros sinon."""
    B = specs_norm.shape[0]
    cases = torch.zeros(B, (R - 1) * K, device=specs_norm.device)
    if t > 0:
        cases[:, :t * K] = F.one_hot(prefixe, K).reshape(B, t * K).float()
    etape = torch.zeros(B, R, device=specs_norm.device)
    etape[:, t] = 1.0
    return torch.cat([specs_norm, cases, etape], dim=1)


class MLP_AR(nn.Module):
    """MLP partagé par les 5 étapes : entrée 4 + 4K + 5 -> 3 couches cachées ReLU -> K logits.
    softmax(logits) = p(n_t | n_1..n_(t-1), spécifications)."""
    def __init__(self, input_size, hidden1, hidden2, hidden3, output_size):
        super(MLP_AR, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden3, output_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        return self.fc4(x)


def lignes_teacher_forcing(specs_norm, classes):
    """Teacher forcing : 5 lignes par architecture, avec les VRAIS anneaux précédents.
    specs_norm : (B, 4) ; classes : (B, R). Renvoie X (R*B, taille d'entrée) et Y (R*B,) ;
    les lignes de l'étape t sont X[t*B:(t+1)*B]."""
    X = torch.cat([entree_etape(specs_norm, classes[:, :t], t) for t in range(R)], dim=0)
    Y = torch.cat([classes[:, t] for t in range(R)], dim=0)
    return X, Y



dataset = TensorDataset(X_fit, C_fit)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

input_size = 4 + (R - 1) * K + R
model = MLP_AR(input_size, HIDDEN_SIZES[0], HIDDEN_SIZES[1], HIDDEN_SIZES[2], K).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

X_val_lignes, Y_val_lignes = lignes_teacher_forcing(X_val, C_val)



train_loss_list, val_loss_list, val_acc_list = [], [], []
best_val_loss = float('inf')
best_epoch = 0
best_state = None
epochs_sans_amelioration = 0

start_time = time.time()
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    total_lignes = 0
    for batch_specs, batch_classes in dataloader:
        batch_specs, batch_classes = batch_specs.to(device), batch_classes.to(device)
        X, Y = lignes_teacher_forcing(batch_specs, batch_classes)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(Y)
        total_lignes += len(Y)
    train_loss = total_loss / total_lignes

    model.eval()
    with torch.no_grad():
        logits_val = model(X_val_lignes)
        val_loss = criterion(logits_val, Y_val_lignes).item()
        bonnes = (logits_val.argmax(dim=1) == Y_val_lignes).float()
        val_acc = 100 * bonnes.reshape(R, -1).mean(dim=1).cpu().numpy()

    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss train: {train_loss:.4f}, Loss val: {val_loss:.4f}, "
          "Précision val par anneau (%): " + " ".join(f"{a:5.1f}" for a in val_acc))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        best_state = {nom: poids.detach().cpu().clone() for nom, poids in model.state_dict().items()}
        epochs_sans_amelioration = 0
    else:
        epochs_sans_amelioration += 1
        if epochs_sans_amelioration >= PATIENCE:
            print(f"Arrêt anticipé : pas d'amélioration de la loss de validation depuis {PATIENCE} epochs.")
            break

end_time = time.time()
print(f"\n=== Training finished in {end_time - start_time:.2f} seconds ===")
model.load_state_dict(best_state)
print(f"Meilleure epoch : {best_epoch} (loss val = {best_val_loss:.4f} ; précision val par anneau (%) : "
      + " ".join(f"{a:.1f}" for a in val_acc_list[best_epoch - 1]) + ")")



checkpoint = {
    'model_state_dict': best_state,
    'R': R,
    'K': K,
    'n_min': N_MIN,
    'n_max': N_MAX,
    'input_size': input_size,
    'hidden_sizes': list(HIDDEN_SIZES),
    'output_size': K,
    'specs_mean': specs_mean.tolist(),
    'specs_std': specs_std.tolist(),
    'std_G_SSL_HPBW': specs_std[:3].tolist(),
    'best_epoch': best_epoch,
    'best_val_loss': best_val_loss,
}
torch.save(checkpoint, os.path.join(script_dir, 'AR_model.pt'))

historique = np.vstack([np.arange(1, len(train_loss_list) + 1), train_loss_list, val_loss_list,
                        np.array(val_acc_list).T])
np.save(os.path.join(script_dir, 'AR_training_history.npy'), historique)
scipy.io.savemat(os.path.join(script_dir, 'AR_training_history.mat'), {'AR_training_history': historique})

plt.figure()
plt.plot(historique[0], train_loss_list, color='#2a78d6', linewidth=2, label='Training')
plt.plot(historique[0], val_loss_list, color='#eb6834', linewidth=2, label='Validation')
plt.axvline(best_epoch, color='#52514e', linestyle='--', linewidth=1, label=f'Meilleure epoch ({best_epoch})')
plt.xlabel("Epoch")
plt.ylabel("Loss (entropie croisée)")
plt.title("Courbe de perte (training / validation)")
plt.grid(alpha=0.3)
plt.legend()
plt.savefig(os.path.join(script_dir, 'AR_loss_curves.png'), dpi=150, bbox_inches='tight')

print(f"\n✅ AR_model.pt, AR_training_history (.npy et .mat) et AR_loss_curves.png ont été enregistrés "
      f"dans le dossier du script : {script_dir}")
plt.show()
