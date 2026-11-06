import numpy as np
import scipy.io
import os


def clear_console(): 
    print("\n" + "="*80 + "\n" + " "*30 + "NOUVELLE EXÉCUTION\n" + "="*80 + "\n")

clear_console()    


SEED = 46
TEST_RATIO = 0.10

np.random.seed(SEED)
script_dir = os.path.dirname(os.path.abspath(__file__))

file_path_input = os.path.join(script_dir, 'Minput.npy')
Minput = np.load(file_path_input)

file_path_output = os.path.join(script_dir, 'Moutput.npy')
Moutput = np.load(file_path_output)




total_cols = Minput.shape[1]
rand_indices = np.random.permutation(total_cols)

nb_test = round(TEST_RATIO * total_cols)
test_indices = rand_indices[:nb_test]
train_indices = rand_indices[nb_test:]

Minput_test = Minput[:, test_indices]
Minput_training = Minput[:, train_indices]
Moutput_test = Moutput[:, test_indices]
Moutput_training = Moutput[:, train_indices]

assert len(np.intersect1d(train_indices, test_indices)) == 0
assert len(train_indices) + len(test_indices) == total_cols
print(f"Répartition : {len(train_indices)} échantillons training, {len(test_indices)} échantillons test "
      f"({100 * TEST_RATIO:.0f} % test), sans échantillon commun.")



print("\nCaractéristiques de la matrice Minput_training :")
print(f"- Shape (lignes, colonnes) : {Minput_training.shape}")
print("→ Premières colonnes (extraits) :")
print(np.round(Minput_training[:, :5], 4))  

print("\nCaractéristiques de la matrice Moutput_training :")
print(f"- Shape (lignes, colonnes) : {Moutput_training.shape}")
print("→ Premières colonnes (extraits) :")
print(Moutput_training[:, :5].astype(int))  


print("\nCaractéristiques de la matrice Minput_test :")
print(f"- Shape (lignes, colonnes) : {Minput_test.shape}")
print("→ Premières colonnes (extraits) :")
print(np.round(Minput_test[:, :5], 4))  

print("\nCaractéristiques de la matrice Moutput_test :")
print(f"- Shape (lignes, colonnes) : {Moutput_test.shape}")
print("→ Premières colonnes (extraits) :")
print(Moutput_test[:, :5].astype(int))  



matrices = {
    'Minput_training': Minput_training,
    'Moutput_training': Moutput_training,
    'Minput_test': Minput_test,
    'Moutput_test': Moutput_test,
    'train_indices': train_indices.reshape(1, -1),
    'test_indices': test_indices.reshape(1, -1),
}
for nom, matrice in matrices.items():
    scipy.io.savemat(os.path.join(script_dir, nom + '.mat'), {nom: matrice})
    np.save(os.path.join(script_dir, nom + '.npy'), matrice)

print("\n✅ Minput_training, Moutput_training, Minput_test, Moutput_test, train_indices et test_indices "
      f"ont été enregistrés (.mat et .npy) dans le dossier du script : {script_dir}")
