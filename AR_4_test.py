import os
import time
import numpy as np
import scipy.io
from scipy.signal import find_peaks
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def clear_console(): 
    print("\n" + "="*80 + "\n" + " "*30 + "NOUVELLE EXÉCUTION\n" + "="*80 + "\n")
clear_console()    


SEED = 46
INDICE_TEST = 10
N_EVAL = 1000
BEAM_WIDTH = 5
NB_TIRAGES = 50
TEMPERATURE = 1.0

DIAPO_SPECS = [28.94, -9.37, 74.95, 112.36]
DIAPO_ARCHI_FFNN = [6, 6, 6, 5, 5]
DIAPO_ERREURS_FFNN = [0.00, 0.21, 1.80]

carrierFreq = 2.45e9
c = 3e8
lambda_ = c / carrierFreq
r0 = 0.2 * lambda_
delta_r = 0.5 * lambda_
max_rings = 5

k = 2 * np.pi / lambda_
theta0 = 0.0
phi0 = 0
phi = 0
theta = np.linspace(0, 2 * np.pi, 1000)
theta_deg = np.rad2deg(theta)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Appareil utilisé : {device}")



script_dir = os.path.dirname(os.path.abspath(__file__))
Minput_test = np.load(os.path.join(script_dir, 'Minput_test.npy'))
Moutput_test = np.load(os.path.join(script_dir, 'Moutput_test.npy'))
Minput_complete = np.load(os.path.join(script_dir, 'Minput.npy'))



def calcul_AF_performance_metrics(elements_per_ring):
    rings = len(elements_per_ring)
    radii = r0 + delta_r * np.arange(max_rings)
    
    AF_az = np.zeros_like(theta, dtype=complex)
    
    for ring in range(rings):
        a = radii[ring]
        N = elements_per_ring[ring]
        if N == 0:
            continue
        phi_n = 2 * np.pi * np.arange(N) / N
        for n in range(N):
            phase = k * a * (np.sin(theta) * np.cos(phi - phi_n[n]) -
                             np.sin(theta0) * np.cos(phi0 - phi_n[n]))
            AF_az += np.exp(1j * phase)
    
    
    AF_norm_az = np.abs(AF_az) / (np.max(np.abs(AF_az)) + np.finfo(float).eps)
    AF_dB_az = 20 * np.log10(AF_norm_az + np.finfo(float).eps)
    AF_dB_az[AF_dB_az < -40] = -40
    theta_deg = np.rad2deg(theta)
    
    AF_abs_az = np.abs(AF_az)
    maxVal = np.max(AF_abs_az)
    maxVal_non_norm = 20 * np.log10(maxVal + np.finfo(float).eps)
    
    maxVal_dB = np.max(AF_dB_az)
    maxIdx = np.argmax(AF_dB_az)
    halfPower = maxVal_dB - 3
    AF_dB_ext = np.concatenate((AF_dB_az, AF_dB_az, AF_dB_az))
    theta_deg_ext = np.concatenate((theta_deg - 360, theta_deg, theta_deg + 360))
    maxIdx_ext = maxIdx + len(theta_deg)
    

    leftIdx_ext = np.where(AF_dB_ext[:maxIdx_ext] <= halfPower)[0][-1] if np.any(
        AF_dB_ext[:maxIdx_ext] <= halfPower) else None
    
    rightIdx_ext = np.where(AF_dB_ext[maxIdx_ext:] <= halfPower)[0]
    
        
    rightIdx_ext = rightIdx_ext[0] + maxIdx_ext if len(rightIdx_ext) > 0 else None
    
    if leftIdx_ext is None or rightIdx_ext is None:
        HPBW = 180
    else:
        HPBW = theta_deg_ext[rightIdx_ext] - theta_deg_ext[leftIdx_ext]

    responseLin = AF_norm_az
    peaks, _ = find_peaks(responseLin, distance=5)
    
    pk = responseLin[peaks]
    if len(pk) == 0:
        main_lobe_gain = maxVal_non_norm 
        true_SSL_gain = 0
    else: 
        sorted_idx = np.argsort(pk)[::-1]
        sorted_pk = pk[sorted_idx]
        sorted_angles = theta_deg[peaks][sorted_idx]

        threshold_dB = 1
        main_lobes_idx = np.where(
            20 * np.log10(sorted_pk + np.finfo(float).eps) >=
            20 * np.log10(sorted_pk[0] + np.finfo(float).eps) - threshold_dB)[0]
        side_lobe_idx = np.setdiff1d(np.arange(len(sorted_pk)), main_lobes_idx)
        
        main_lobe_gain = maxVal_non_norm
        
        true_SSL_gain = 20 * np.log10(sorted_pk[side_lobe_idx[0]] + np.finfo(float).eps) if len(
            side_lobe_idx) > 0 else 0
        
    return AF_dB_az, HPBW, maxVal_non_norm, true_SSL_gain

max_main_lobe_gain = np.max(Minput_complete[0, :])
max_true_SSL_gain  = np.max(np.abs(Minput_complete[1, :]))  
max_HPBW          = np.max(Minput_complete[2, :])

w1 = 0.33
w2 = 0.33
w3 = 0.34


def calculer_specs(architecture, theta0deg):
    """Renvoie G (dB), SSL (dB), HPBW (°) et le diagramme AF_dB_az (dB) de l'architecture
    [n1, ..., n5] dépointée vers theta0deg (°), calculés par la fonction du professeur.
    Sa fonction lit theta0 comme variable globale : on la fixe ici juste avant l'appel."""
    global theta0
    theta0 = np.deg2rad(theta0deg)
    AF_dB_az, HPBW, maxVal_non_norm, true_SSL_gain = calcul_AF_performance_metrics(
        np.asarray(architecture).astype(int))
    return maxVal_non_norm, true_SSL_gain, HPBW, AF_dB_az


def erreurs_prediction(gain_main_pred, SSL_pred, HPBW_pred, gain_main_ref, SSL_ref, HPBW_ref):
    """Lignes de calcul de l'erreur de test_model_FFNN.py, copiées telles quelles (seule l'indentation change).
    Accepte un échantillon (nombres) ou plusieurs échantillons (tableaux numpy)."""
    mae_HPBW = abs(HPBW_pred - HPBW_ref)
    mae_main_lobe_gain = abs(gain_main_pred - gain_main_ref)
    mae_true_SSL_gain = abs(SSL_pred - SSL_ref)

    err_rel_main_lobe_gain = 100*(abs(gain_main_pred - gain_main_ref) / (abs(max_main_lobe_gain) + 1e-8))
    err_rel_true_SSL_gain  = 100*(abs(SSL_pred - SSL_ref) / (abs(max_true_SSL_gain) + 1e-8))
    err_rel_HPBW           = 100*(abs(HPBW_pred - HPBW_ref) / (180 + 1e-8))

    error_pred_global = (    w1 * err_rel_main_lobe_gain +    w2 * err_rel_true_SSL_gain  +    w3 * err_rel_HPBW  )     
    return (mae_main_lobe_gain, mae_true_SSL_gain, mae_HPBW,
            err_rel_main_lobe_gain, err_rel_true_SSL_gain, err_rel_HPBW, error_pred_global)



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


checkpoint = torch.load(os.path.join(script_dir, 'AR_model.pt'), map_location=device)
R = checkpoint['R']
K = checkpoint['K']
N_MIN = checkpoint['n_min']
N_MAX = checkpoint['n_max']
specs_mean = np.array(checkpoint['specs_mean'])
specs_std = np.array(checkpoint['specs_std'])
std_G_SSL_HPBW = np.array(checkpoint['std_G_SSL_HPBW'])
hidden = checkpoint['hidden_sizes']
model = MLP_AR(checkpoint['input_size'], hidden[0], hidden[1], hidden[2], checkpoint['output_size']).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
assert R == max_rings, "le modèle et les constantes du dataset n'ont pas le même nombre d'anneaux"
print(f"Modèle chargé : K = {K} classes (n de {N_MIN} à {N_MAX}), meilleure epoch {checkpoint['best_epoch']}")

N_TOTAL_MIN = max(R * N_MIN, 1)
N_TOTAL_MAX = R * N_MAX

generateur = torch.Generator(device=device)
cache_simulations = {}



def nombre_total_depuis_G(G):
    """N = round(10**(G/20)), borné à [N_TOTAL_MIN, N_TOTAL_MAX]. G : nombre ou tableau.
    Renvoie (N borné, N avant bornage)."""
    N_reel = np.round(10 ** (np.asarray(G, dtype=float) / 20))
    return np.clip(N_reel, N_TOTAL_MIN, N_TOTAL_MAX).astype(int), N_reel


def masquer_logits(logits, somme, N, t):
    """Met à -inf les logits des valeurs v qui rendraient le total N impossible.
    À l'étape t (0..R-1), avec S = somme des anneaux déjà placés et L = R-1-t anneaux restant après
    celui-ci, v est autorisé si L*N_MIN <= N - S - v <= L*N_MAX. Au dernier anneau, seul v = N - S reste."""
    L = R - 1 - t
    v = torch.arange(N_MIN, N_MAX + 1, device=logits.device)
    reste = (N - somme)[:, None] - v[None, :]
    autorise = (reste >= L * N_MIN) & (reste <= L * N_MAX)
    return logits.masked_fill(~autorise, float('-inf'))


def decoder_greedy(specs_norm, N):
    """Greedy : argmax à chaque étape. specs_norm (B, 4), N (B,) -> architectures (B, R)."""
    B = specs_norm.shape[0]
    classes = torch.zeros(B, 0, dtype=torch.long, device=device)
    somme = torch.zeros(B, dtype=torch.long, device=device)
    for t in range(R):
        logits = masquer_logits(model(entree_etape(specs_norm, classes, t)), somme, N, t)
        c = logits.argmax(dim=1)
        classes = torch.cat([classes, c[:, None]], dim=1)
        somme = somme + c + N_MIN
    return (classes + N_MIN).cpu().numpy()


def decoder_beam(specs_norm, N):
    """Beam search (largeur BEAM_WIDTH) pour UNE spécification : specs_norm (1, 4), N (1,).
    Score d'un faisceau = somme des log-probabilités. Renvoie les faisceaux finaux (au plus BEAM_WIDTH, R)."""
    classes = torch.zeros(1, 0, dtype=torch.long, device=device)
    score = torch.zeros(1, device=device)
    for t in range(R):
        nb = classes.shape[0]
        somme = classes.sum(dim=1) + t * N_MIN
        logits = masquer_logits(model(entree_etape(specs_norm.expand(nb, -1), classes, t)), somme, N.expand(nb), t)
        score_total = (score[:, None] + torch.log_softmax(logits, dim=1)).reshape(-1)
        nb_garde = min(BEAM_WIDTH, int(torch.isfinite(score_total).sum()))
        score, idx = torch.topk(score_total, nb_garde)
        parent = torch.div(idx, K, rounding_mode='floor')
        classes = torch.cat([classes[parent], (idx % K)[:, None]], dim=1)
    return (classes + N_MIN).cpu().numpy()


def decoder_sampling(specs_norm, N):
    """Sampling : NB_TIRAGES architectures tirées selon softmax(logits / TEMPERATURE),
    pour UNE spécification : specs_norm (1, 4), N (1,). Renvoie (NB_TIRAGES, R)."""
    specs_rep = specs_norm.expand(NB_TIRAGES, -1)
    N_rep = N.expand(NB_TIRAGES)
    classes = torch.zeros(NB_TIRAGES, 0, dtype=torch.long, device=device)
    somme = torch.zeros(NB_TIRAGES, dtype=torch.long, device=device)
    for t in range(R):
        logits = masquer_logits(model(entree_etape(specs_rep, classes, t)), somme, N_rep, t)
        probas = torch.softmax(logits / TEMPERATURE, dim=1)
        c = torch.multinomial(probas, 1, generator=generateur)[:, 0]
        classes = torch.cat([classes, c[:, None]], dim=1)
        somme = somme + c + N_MIN
    return (classes + N_MIN).cpu().numpy()


def generer_candidats(G, SSL, HPBW, theta0deg, avertir=True):
    """Candidats des trois décodages masqués pour UN vecteur de spécifications.
    Renvoie ({'greedy': (1, R), 'beam': (<=BEAM_WIDTH, R), 'sampling': (NB_TIRAGES, R)}, N)."""
    N, N_reel = nombre_total_depuis_G(G)
    N = int(N)
    if avertir and N != N_reel:
        print(f"⚠️  round(10**(G/20)) = {N_reel:.0f} est hors de [{N_TOTAL_MIN}, {N_TOTAL_MAX}] : N est borné à {N}")
    specs_norm = (np.array([G, SSL, HPBW, theta0deg], dtype=float) - specs_mean) / specs_std
    specs_norm = torch.tensor(specs_norm, dtype=torch.float32, device=device)[None, :]
    N_t = torch.tensor([N], dtype=torch.long, device=device)
    generateur.manual_seed(SEED)
    with torch.no_grad():
        candidats = {'greedy': decoder_greedy(specs_norm, N_t),
                     'beam': decoder_beam(specs_norm, N_t),
                     'sampling': decoder_sampling(specs_norm, N_t)}
    return candidats, N


def specs_simulees(architecture, theta0deg):
    """(G, SSL, HPBW) de l'architecture ; chaque simulation n'est calculée qu'une fois."""
    cle = (tuple(int(n) for n in architecture), float(theta0deg))
    if cle not in cache_simulations:
        G, SSL, HPBW, _ = calculer_specs(architecture, theta0deg)
        cache_simulations[cle] = (G, SSL, HPBW)
    return cache_simulations[cle]


def verifier(candidats, G, SSL, HPBW, theta0deg):
    """Simule chaque candidat unique et garde celui qui minimise
    E = somme sur G, SSL, HPBW de |spec - cible| / std_train. Renvoie (architecture, specs, E)."""
    cible = np.array([G, SSL, HPBW])
    meilleur_E = np.inf
    for architecture in np.unique(candidats, axis=0):
        specs_obtenues = np.array(specs_simulees(architecture, theta0deg))
        E = np.sum(np.abs(specs_obtenues - cible) / std_G_SSL_HPBW)
        if E < meilleur_E:
            meilleure_architecture, meilleures_specs, meilleur_E = architecture, specs_obtenues, E
    return meilleure_architecture, meilleures_specs, meilleur_E


def predire_architecture(G, SSL, HPBW, theta0deg):
    """Prédit le nombre d'éléments de chaque anneau [n1, ..., n5] pour des spécifications cibles
    (G en dB, SSL en dB, HPBW en degrés, theta0deg en degrés), par générer puis vérifier :
    candidats greedy + beam search + sampling, simulés avec le code du professeur.
    Renvoie (architecture, specs obtenues [G, SSL, HPBW], E).
    Exemple : architecture, specs, E = predire_architecture(28.94, -9.37, 74.95, 112.36)"""
    candidats, _ = generer_candidats(G, SSL, HPBW, theta0deg)
    tous = np.vstack([candidats['greedy'], candidats['beam'], candidats['sampling']])
    return verifier(tous, G, SSL, HPBW, theta0deg)



NOMS_METRIQUES = ['MAE G (dB)', 'MAE SSL (dB)', 'MAE HPBW (°)',
                  'MSE G (dB²)', 'MSE SSL (dB²)', 'MSE HPBW (°²)',
                  'MAPE G (%)', 'MAPE SSL (%)', 'MAPE HPBW (%)',
                  'Erreur globale pondérée (%)',
                  'Bon N total (%)', 'Anneaux dans les bornes (%)', 'Architecture exacte (%)']


def calculer_metriques(Moutput_pred, Mspecs_pred, nb):
    """Métriques sur les nb premiers échantillons de test.
    Moutput_pred : (R, nb) architectures prédites ; Mspecs_pred : (3, nb) leurs G, SSL, HPBW re-simulés.
    MAE et MAPE viennent du calcul d'erreur du professeur (MAPE = moyenne de son erreur relative en %)."""
    cible = Minput_test[:3, :nb]
    reference = Moutput_test[:, :nb]
    (mae_G, mae_SSL, mae_HPBW, rel_G, rel_SSL, rel_HPBW, globale) = erreurs_prediction(
        Mspecs_pred[0], Mspecs_pred[1], Mspecs_pred[2], cible[0], cible[1], cible[2])
    mse = np.mean((Mspecs_pred - cible) ** 2, axis=1)
    return np.array([
        np.mean(mae_G), np.mean(mae_SSL), np.mean(mae_HPBW),
        mse[0], mse[1], mse[2],
        np.mean(rel_G), np.mean(rel_SSL), np.mean(rel_HPBW),
        np.mean(globale),
        100 * np.mean(Moutput_pred.sum(axis=0) == reference.sum(axis=0)),
        100 * np.mean(np.all((Moutput_pred >= N_MIN) & (Moutput_pred <= N_MAX), axis=0)),
        100 * np.mean(np.all(Moutput_pred == reference, axis=0)),
    ])


resume = []

def afficher(texte=""):
    """Affiche une ligne et la garde pour le résumé .txt."""
    print(texte)
    resume.append(texte)


debut_script = time.time()
afficher("Test du modèle autorégressif AR (AR_4_test.py)")
afficher(f"Modèle : AR_model.pt (meilleure epoch {checkpoint['best_epoch']}), K = {K} classes, n de {N_MIN} à {N_MAX}")
afficher(f"Décodages masqués : greedy, beam search (largeur {BEAM_WIDTH}), sampling ({NB_TIRAGES} tirages, "
         f"température {TEMPERATURE})")



afficher(f"\n=== Exemple détaillé : indice_test = {INDICE_TEST} ===")
gain_main_ref, SSL_ref, HPBW_ref, theta0deg = Minput_test[:, INDICE_TEST]
architecture_reference = Moutput_test[:, INDICE_TEST].astype(int)
architecture_predicted, specs_predites, E_predit = predire_architecture(gain_main_ref, SSL_ref, HPBW_ref, theta0deg)
candidats_exemple, _ = generer_candidats(gain_main_ref, SSL_ref, HPBW_ref, theta0deg, avertir=False)

G_ref_sim, SSL_ref_sim, HPBW_ref_sim, AF_ref_dB = calculer_specs(architecture_reference, theta0deg)
gain_main_pred, SSL_pred, HPBW_pred, AF_pred_dB = calculer_specs(architecture_predicted, theta0deg)
ecart_reference = np.max(np.abs(np.array([G_ref_sim - gain_main_ref, SSL_ref_sim - SSL_ref, HPBW_ref_sim - HPBW_ref])))

afficher(f"Architecture référence  : {architecture_reference}")
afficher(f"Architecture prédite    : {architecture_predicted}   (générer-vérifier, E = {E_predit:.3f})")
afficher(f"Architecture greedy     : {candidats_exemple['greedy'][0]}   (sans vérification)")
afficher(f"(Référence re-simulée avec le code du professeur : écart max avec Minput_test = {ecart_reference:.1e})")

afficher(f"\ntheta0deg                 : {theta0deg:.2f}°")

afficher(f"\n--- Performances Référence (spécifications cibles) ---")
afficher(f"Gain du lobe principal : {gain_main_ref:.2f} dB")
afficher(f"True Side Lobe Gain   : {SSL_ref:.2f} dB")
afficher(f"HPBW                  : {HPBW_ref:.2f}°")

afficher(f"\n--- Performances Prédiction ---")
afficher(f"Gain du lobe principal : {gain_main_pred:.2f} dB")
afficher(f"True Side Lobe Gain   : {SSL_pred:.2f} dB")
afficher(f"HPBW                  : {HPBW_pred:.2f}°")

(mae_main_lobe_gain, mae_true_SSL_gain, mae_HPBW, err_rel_main_lobe_gain, err_rel_true_SSL_gain, err_rel_HPBW,
 error_pred_global) = erreurs_prediction(gain_main_pred, SSL_pred, HPBW_pred, gain_main_ref, SSL_ref, HPBW_ref)

afficher("\n=== Erreurs absolues (MAE) entre prédiction et référence ===")
afficher(f"Erreur Gain Lobe Principal : {mae_main_lobe_gain:.2f} dB")
afficher(f"Erreur True SSL Gain      : {mae_true_SSL_gain:.2f} dB")
afficher(f"Erreur HPBW               : {mae_HPBW:.2f} degrés")

afficher("\n=== Erreurs relatives de prédiction (%) ===")
afficher(f"Erreur relative de prédiction Gain Lobe Principal : {err_rel_main_lobe_gain:.2f} %")
afficher(f"Erreur relative de prédiction True SSL Gain      : {err_rel_true_SSL_gain:.2f} %")
afficher(f"Erreur relative de prédiction HPBW               : {err_rel_HPBW:.2f} %")

afficher(f"\nErreur globale de prédiction pondérée : {error_pred_global:.2f} %")

plt.figure(figsize=(8, 6))
ax = plt.subplot(111, polar=True)
ax.plot(theta, AF_ref_dB, label='Référence', linewidth=2)
ax.plot(theta, AF_pred_dB, label='Prédiction', linewidth=2, linestyle='--')
ax.set_title("Diagramme de rayonnement (polaire)", fontsize=13)
ax.set_rlim([-40, 0])
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
plt.savefig(os.path.join(script_dir, 'AR_test_polar.png'), dpi=150, bbox_inches='tight')

plt.figure(figsize=(10, 5))
plt.plot(theta_deg, AF_ref_dB, label='Référence', linewidth=2)
plt.plot(theta_deg, AF_pred_dB, label='Prédiction', linewidth=2, linestyle='--')
plt.xlabel("Azimut (°)")
plt.ylabel("Gain (dB)")
plt.title("Diagramme de rayonnement - Azimut")
plt.xlim([0, 360])
plt.ylim([-40, 0])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'AR_test_azimuth.png'), dpi=150)


MARQUEURS = ['o', 's', '^', 'D', 'v']
COULEURS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']

def tracer_configuration(ax, architecture, titre):
    """Positions des éléments : anneau t de rayon r0 + (t-1)*delta_r, élément n à l'angle 2*pi*n/N
    (mêmes positions que dans calcul_AF_performance_metrics)."""
    radii = r0 + delta_r * np.arange(max_rings)
    cercle = np.linspace(0, 2 * np.pi, 200)
    for ring in range(max_rings):
        a = radii[ring] / lambda_
        ax.plot(a * np.cos(cercle), a * np.sin(cercle), color='#c3c2b7', linestyle='--', linewidth=1)
        N = architecture[ring]
        if N == 0:
            continue
        phi_n = 2 * np.pi * np.arange(N) / N
        ax.plot(a * np.cos(phi_n), a * np.sin(phi_n), linestyle='none', marker=MARQUEURS[ring], markersize=8,
                color=COULEURS[ring], label=f"Anneau {ring + 1} : {N} éléments")
    ax.set_aspect('equal')
    ax.set_title(f"{titre} : {architecture}")
    ax.set_xlabel("x / λ")
    ax.set_ylabel("y / λ")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)

fig, axes = plt.subplots(1, 2, figsize=(12, 7))
tracer_configuration(axes[0], architecture_reference, "Référence")
tracer_configuration(axes[1], architecture_predicted, "Prédiction")
fig.suptitle(f"Configuration du réseau (indice_test = {INDICE_TEST}, theta0 = {theta0deg:.2f}°)")
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'AR_test_array_configuration.png'), dpi=150, bbox_inches='tight')



def ligne_comparaison(nom, architecture, erreurs):
    return (f"{nom:36s}{str(np.array(architecture)):18s}{erreurs[0]:>7.2f} dB{erreurs[1]:>9.2f} dB"
            f"{erreurs[2]:>12.2f}°")

afficher("\n=== Exemple de la diapositive du professeur ===")
G_diapo, SSL_diapo, HPBW_diapo, theta0deg_diapo = DIAPO_SPECS
afficher(f"Spécifications cibles : G = {G_diapo:.2f} dB, SSL = {SSL_diapo:.2f} dB, HPBW = {HPBW_diapo:.2f}°, "
         f"theta0 = {theta0deg_diapo:.2f}°")
architecture_AR, specs_AR, E_AR = predire_architecture(G_diapo, SSL_diapo, HPBW_diapo, theta0deg_diapo)
G_FFNN, SSL_FFNN, HPBW_FFNN, _ = calculer_specs(DIAPO_ARCHI_FFNN, theta0deg_diapo)
cible_diapo = np.array([G_diapo, SSL_diapo, HPBW_diapo])
afficher(f"{'Méthode':36s}{'Architecture':18s}{'Erreur G':>10s}{'Erreur SSL':>12s}{'Erreur HPBW':>13s}")
afficher(ligne_comparaison("FFNN du professeur (diapositive)", DIAPO_ARCHI_FFNN, DIAPO_ERREURS_FFNN))
afficher(ligne_comparaison("FFNN re-simulé avec son code", DIAPO_ARCHI_FFNN,
                           np.abs(np.array([G_FFNN, SSL_FFNN, HPBW_FFNN]) - cible_diapo)))
afficher(ligne_comparaison("AR générer-vérifier", architecture_AR, np.abs(specs_AR - cible_diapo)))
afficher(f"Specs obtenues par l'AR : G = {specs_AR[0]:.2f} dB, SSL = {specs_AR[1]:.2f} dB, HPBW = {specs_AR[2]:.2f}°")
afficher("(Les cibles de la diapositive sont arrondies à 0.01 : les erreurs re-simulées du FFNN peuvent "
         "différer de 0.01.)")



nb_test = Minput_test.shape[1]
n_eval = min(N_EVAL, nb_test)
afficher(f"\n=== Évaluation sur le split test ({nb_test} échantillons) ===")

debut = time.time()
specs_ref = np.array([specs_simulees(Moutput_test[:, i], Minput_test[3, i]) for i in range(20)]).T
t_sim = (time.time() - debut) / 20
afficher(f"Contrôle : specs re-simulées des 20 premières architectures de référence - Minput_test : "
         f"écart max = {np.max(np.abs(specs_ref - Minput_test[:3, :20])):.1e}")
generer_candidats(*Minput_test[:, 0], avertir=False)
debut = time.time()
for i in range(3):
    generer_candidats(*Minput_test[:, i], avertir=False)
t_reseau = (time.time() - debut) / 3

estimation_greedy = nb_test * t_sim
estimation_gv = n_eval * (t_reseau + (BEAM_WIDTH + NB_TIRAGES) * t_sim)
print(f"Une simulation (code du professeur) : {1000 * t_sim:.2f} ms ; génération des candidats : "
      f"{1000 * t_reseau:.1f} ms par échantillon")
print(f"Temps estimé : greedy sur {nb_test} échantillons ~{estimation_greedy:.0f} s ; générer-vérifier sur "
      f"{n_eval} échantillons au plus ~{estimation_gv:.0f} s ({estimation_gv / 60:.1f} min), soit au plus "
      f"{1 + BEAM_WIDTH + NB_TIRAGES} candidats simulés par échantillon (doublons simulés une seule fois)")

N_test, N_test_reel = nombre_total_depuis_G(Minput_test[0, :])
afficher(f"N = round(10**(G/20)) borné pour {np.sum(N_test != N_test_reel)} échantillon(s) de test")
specs_norm_test = torch.tensor((Minput_test.T - specs_mean) / specs_std, dtype=torch.float32, device=device)
debut = time.time()
with torch.no_grad():
    Moutput_greedy = decoder_greedy(specs_norm_test, torch.tensor(N_test, device=device)).T
Mspecs_greedy = np.array([specs_simulees(Moutput_greedy[:, i], Minput_test[3, i]) for i in range(nb_test)]).T
print(f"Greedy : {nb_test} architectures générées et simulées en {time.time() - debut:.1f} s")
nb_architectures = nb_test
nb_violations = np.sum((Moutput_greedy.sum(axis=0) != N_test)
                       | np.any((Moutput_greedy < N_MIN) | (Moutput_greedy > N_MAX), axis=0))

modes = ['greedy', 'beam', 'sampling', 'gv']
Moutput_modes = {mode: np.zeros((R, n_eval), dtype=int) for mode in modes}
Mspecs_modes = {mode: np.zeros((3, n_eval)) for mode in modes}
nb_uniques = 0
pas = max(1, n_eval // 10)
debut = time.time()
for i in range(n_eval):
    G, SSL, HPBW, theta0deg_i = Minput_test[:, i]
    candidats, N = generer_candidats(G, SSL, HPBW, theta0deg_i, avertir=False)
    candidats['gv'] = np.vstack([candidats['greedy'], candidats['beam'], candidats['sampling']])
    nb_architectures += len(candidats['gv'])
    nb_violations += np.sum((candidats['gv'].sum(axis=1) != N)
                            | np.any((candidats['gv'] < N_MIN) | (candidats['gv'] > N_MAX), axis=1))
    nb_uniques += len(np.unique(candidats['gv'], axis=0))
    for mode in modes:
        architecture, specs_obtenues, _ = verifier(candidats[mode], G, SSL, HPBW, theta0deg_i)
        Moutput_modes[mode][:, i] = architecture
        Mspecs_modes[mode][:, i] = specs_obtenues
    if (i + 1) % pas == 0:
        ecoule = time.time() - debut
        print(f"Générer-vérifier : {i + 1}/{n_eval} échantillons ({ecoule:.0f} s, "
              f"reste ~{ecoule / (i + 1) * (n_eval - i - 1):.0f} s)")
temps_gv = time.time() - debut

afficher(f"Générer-vérifier : {nb_uniques / n_eval:.1f} candidats uniques simulés par échantillon en moyenne "
         f"({temps_gv:.0f} s pour {n_eval} échantillons)")
afficher(f"Règles du masque (somme = N, {N_MIN} <= n <= {N_MAX}) : {nb_violations} violation(s) sur "
         f"{nb_architectures} architectures générées")

methodes = [
    (f"Greedy, tout le test (n = {nb_test})", Moutput_greedy, Mspecs_greedy, nb_test),
    (f"Greedy (n = {n_eval})", Moutput_modes['greedy'], Mspecs_modes['greedy'], n_eval),
    (f"Beam {BEAM_WIDTH} + vérification (n = {n_eval})", Moutput_modes['beam'], Mspecs_modes['beam'], n_eval),
    (f"Sampling {NB_TIRAGES} + vérification (n = {n_eval})", Moutput_modes['sampling'], Mspecs_modes['sampling'], n_eval),
    (f"Générer-vérifier, 3 modes (n = {n_eval})", Moutput_modes['gv'], Mspecs_modes['gv'], n_eval),
]
noms_methodes = [nom for nom, _, _, _ in methodes]
metriques = np.array([calculer_metriques(Mo, Ms, nb) for _, Mo, Ms, nb in methodes])

afficher("\nMAPE = moyenne de l'erreur relative du professeur : |écart| / max(G) du dataset pour G, "
         "/ max(|SSL|) pour le SSL, / 180° pour le HPBW.")
afficher("Bon N total : même nombre total d'éléments que l'architecture de référence. "
         "Architecture exacte : indicatif seulement.")
for nom, m in zip(noms_methodes, metriques):
    afficher(f"\n--- {nom} ---")
    afficher(f"MAE  : G = {m[0]:.3f} dB | SSL = {m[1]:.3f} dB | HPBW = {m[2]:.3f}°")
    afficher(f"MSE  : G = {m[3]:.3f} dB² | SSL = {m[4]:.3f} dB² | HPBW = {m[5]:.3f} (°)²")
    afficher(f"MAPE : G = {m[6]:.2f} % | SSL = {m[7]:.2f} % | HPBW = {m[8]:.2f} %")
    afficher(f"Erreur globale de prédiction pondérée : {m[9]:.2f} %")
    afficher(f"Bon N total : {m[10]:.2f} % | Anneaux dans les bornes : {m[11]:.2f} % | "
             f"Architecture exacte : {m[12]:.2f} %")

afficher(f"\n{'Méthode':44s}{'MAE G':>8s}{'MAE SSL':>9s}{'MAE HPBW':>10s}{'Err. glob.':>12s}{'Exacte':>9s}")
for nom, m in zip(noms_methodes, metriques):
    afficher(f"{nom:44s}{m[0]:>8.3f}{m[1]:>9.3f}{m[2]:>10.3f}{m[9]:>10.2f} %{m[12]:>7.2f} %")



AR_test_predictions_greedy = np.vstack([Moutput_greedy, Mspecs_greedy])
AR_test_predictions_gv = np.vstack([Moutput_modes['gv'], Mspecs_modes['gv']])
AR_test_metrics = metriques

np.save(os.path.join(script_dir, 'AR_test_predictions_greedy.npy'), AR_test_predictions_greedy)
np.save(os.path.join(script_dir, 'AR_test_predictions_gv.npy'), AR_test_predictions_gv)
np.save(os.path.join(script_dir, 'AR_test_metrics.npy'), AR_test_metrics)
scipy.io.savemat(os.path.join(script_dir, 'AR_test_predictions_greedy.mat'),
                 {'AR_test_predictions_greedy': AR_test_predictions_greedy})
scipy.io.savemat(os.path.join(script_dir, 'AR_test_predictions_gv.mat'),
                 {'AR_test_predictions_gv': AR_test_predictions_gv})
scipy.io.savemat(os.path.join(script_dir, 'AR_test_metrics.mat'),
                 {'AR_test_metrics': AR_test_metrics,
                  'noms_methodes': np.array(noms_methodes, dtype=object),
                  'noms_metriques': np.array(NOMS_METRIQUES, dtype=object)})

afficher(f"\nColonnes de AR_test_metrics : {', '.join(NOMS_METRIQUES)}")
afficher(f"Temps total du test : {time.time() - debut_script:.0f} s")
with open(os.path.join(script_dir, 'AR_test_summary.txt'), 'w', encoding='utf-8') as fichier:
    fichier.write("\n".join(resume) + "\n")

print(f"\n✅ Prédictions, métriques (.npy et .mat), AR_test_summary.txt et figures PNG enregistrés "
      f"dans le dossier du script : {script_dir}")
plt.show()
