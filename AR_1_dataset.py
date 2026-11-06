import numpy as np
from scipy.signal import find_peaks
import scipy.io
import os
import time



def clear_console(): 
    print("\n" + "="*80 + "\n" + " "*30 + "NOUVELLE EXÉCUTION\n" + "="*80 + "\n")

clear_console()

script_dir = os.path.dirname(os.path.abspath(__file__))



nb_samples = 50000
max_rings = 5
max_elements = 10
theta0_max_deg = 180
carrierFreq = 2.45e9
c = 3e8
lambda_ = c / carrierFreq
k = 2 * np.pi / lambda_
r0 = 0.2 * lambda_
delta_r = 0.5 * lambda_
SEED = 46

theta0 = 0.0
phi0 = 0
phi = 0
theta = np.linspace(0, 2 * np.pi, 1000)
theta_deg = np.rad2deg(theta)



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


def calculer_specs(architecture, theta0deg):
    """Renvoie G (dB), SSL (dB), HPBW (°) et le diagramme AF_dB_az (dB) de l'architecture
    [n1, ..., n5] dépointée vers theta0deg (°), calculés par la fonction du professeur.
    Sa fonction lit theta0 comme variable globale : on la fixe ici juste avant l'appel."""
    global theta0
    theta0 = np.deg2rad(theta0deg)
    AF_dB_az, HPBW, maxVal_non_norm, true_SSL_gain = calcul_AF_performance_metrics(
        np.asarray(architecture).astype(int))
    return maxVal_non_norm, true_SSL_gain, HPBW, AF_dB_az



np.random.seed(SEED)
Moutput = np.zeros((max_rings, nb_samples))
Minput = np.zeros((4, nb_samples))

pas_affichage = max(1, nb_samples // 10)
start_time = time.time()

for sample_idx in range(nb_samples):
    while True:
        elements_per_ring = np.random.randint(0, max_elements + 1, size=max_rings)
        if np.sum(elements_per_ring > 0) > 0:
            break
    Moutput[:, sample_idx] = elements_per_ring

    theta0deg = np.random.uniform(0, theta0_max_deg)

    main_lobe_gain, true_SSL_gain, HPBW, _ = calculer_specs(elements_per_ring, theta0deg)

    Minput[0, sample_idx] = main_lobe_gain
    Minput[1, sample_idx] = true_SSL_gain
    Minput[2, sample_idx] = HPBW
    Minput[3, sample_idx] = theta0deg

    if (sample_idx + 1) % pas_affichage == 0:
        print(f"Échantillons générés : {sample_idx + 1}/{nb_samples} ({time.time() - start_time:.1f} s)")


print("\nCaractéristiques de la matrice Minput :")
print(f"- Shape (lignes, colonnes) : {Minput.shape}")
print("→ Premières colonnes (extraits) :")
print(np.round(Minput[:, :5], 4))  

print("\nCaractéristiques de la matrice Moutput :")
print(f"- Shape (lignes, colonnes) : {Moutput.shape}")
print("→ Premières colonnes (extraits) :")
print(Moutput[:, :5].astype(int))  



N_total = Moutput.sum(axis=0)
N_depuis_G = np.round(10 ** (Minput[0, :] / 20))
taux_N = 100 * np.mean(N_depuis_G == N_total)
print("\n=== Vérification 1 : nombre total d'éléments N déduit du gain G ===")
print(f"round(10**(G/20)) == nombre total d'éléments : {taux_N:.2f} % des échantillons "
      f"({np.sum(N_depuis_G != N_total)} écart(s) sur {nb_samples})")
print(f"Écart maximal |10**(G/20) - N| : {np.max(np.abs(10 ** (Minput[0, :] / 20) - N_total)):.4f} élément")
print(f"Exemple : 28 éléments -> 20*log10(28) = {20 * np.log10(28):.2f} dB")


anneaux_vides = (Moutput == 0)
interieur_vide = np.zeros(nb_samples, dtype=bool)
for ring in range(max_rings - 1):
    interieur_vide |= anneaux_vides[ring] & np.any(Moutput[ring + 1:, :] > 0, axis=0)
print("\n=== Vérification 2 : nombre d'éléments par anneau ===")
print(f"Générateur de dataset.py : np.random.randint(0, max_elements + 1) -> de 0 à {max_elements} éléments par anneau")
print(f"Valeurs observées dans Moutput : de {Moutput.min():.0f} à {Moutput.max():.0f}")
print(f"Anneaux vides : {100 * anneaux_vides.mean():.2f} % de tous les anneaux")
print("Anneau vide selon sa position (anneau 1 = intérieur) : "
      + ", ".join(f"anneau {ring + 1} : {100 * anneaux_vides[ring].mean():.1f} %" for ring in range(max_rings)))
print(f"Échantillons avec un anneau intérieur vide (suivi d'un anneau non vide) : {100 * interieur_vide.mean():.2f} %")
print(f"Architecture entièrement vide : rejetée par le générateur, donc N total de {N_total.min():.0f} à {N_total.max():.0f}")
print(f"=> Un anneau peut être vide, à n'importe quelle position : de 0 à {max_elements} éléments par anneau "
      f"({max_elements + 1} valeurs possibles), avec au moins 1 élément au total.")



scipy.io.savemat(os.path.join(script_dir, 'Minput.mat'), {'Minput': Minput})
scipy.io.savemat(os.path.join(script_dir, 'Moutput.mat'), {'Moutput': Moutput})

np.save(os.path.join(script_dir, 'Minput.npy'), Minput)
np.save(os.path.join(script_dir, 'Moutput.npy'), Moutput)


print(f"\n✅ Minput et Moutput ont été enregistrés (.mat et .npy) dans le dossier du script : {script_dir}")
print(f"Temps total : {time.time() - start_time:.1f} s")
