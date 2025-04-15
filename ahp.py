
import numpy as np

#Methode qui calcule le poid des critères
def calculate_weights(comparison_matrix):
    norm_matrix = comparison_matrix / comparison_matrix.sum(axis=0) #Normalisation de la matrice
    weights = norm_matrix.mean(axis=1) #calcul du poids
    return weights

#Methode qui verifie la consistance de la matrice de comparaison
def consistency_check(comparison_matrix, weights):
    n = comparison_matrix.shape[0] # n le nombre de critère
    weighted_sum = comparison_matrix.dot(weights) # calcul de la somme des poids des critères
    lambda_max = weighted_sum / weights # calcul de lamda
    CI = (np.mean(lambda_max) - n) / (n - 1)  # calcule du consistency index
    RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
          7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    
    CR = CI / RI[n] if n in RI else CI # calcule du consistency Ration
    return CR

# methode to evaluate the overall weight of each alternative by its criteria weight.
def evaluate_alternatives(matrix, weights):
    scores = matrix.dot(weights)
    return scores


def main(comparison_matrix, alternatives_matrix):
    weights = calculate_weights(comparison_matrix)
    CR = consistency_check(comparison_matrix, weights)

    scores = evaluate_alternatives(alternatives_matrix, weights)
    
    return weights, CR, scores


###Initialisation des donneées

# Exemple de matrice de comparaison (M, S, C, P, B)
comparison_matrix = np.array([
    [1, 5, 3, 3, 7],     # M
    [1/5, 1, 1/3, 1/3, 5],   # S
    [1/3, 3, 1, 1, 5], # C
    [1/3, 3, 1, 1, 5], # P
    [1/7, 1/5, 1/5, 1/5, 1]  # B
])

# Définir les préférences et les alternatives
preferences = ["Mémoire", "Stockage", "Fréquence CPU", "Prix", "Marque"]
alternatives = [
    "Iphone 12", "Itel A56", "Tecno Camon 12", "Infinix Hot 10", 
    "Huawei P30", "Google Pixel 7", "Xiaomi Redmi Note 10", 
    "Samsung Galaxy S22", "Motorola Razr+", "Iphone X R", 
    "Samsung Galaxy Note 10"
]

# Créer une matrice avec NumPy
alternatives_matrix = np.zeros((len(alternatives), len(preferences)))

# Exemple d'assignation de valeurs (M, S, C, P, B)
alternatives_matrix[0] = [4, 64, 3.1, 250000, 3]  # Iphone 12
alternatives_matrix[1] = [1, 16, 1.3, 40000, 1]  # Itel A56
alternatives_matrix[2] = [4, 64, 2.0, 60000, 15]  # Tecno Camon 12
alternatives_matrix[3] = [4, 128, 2.0, 70000, 17]  # Infinix Hot 10
alternatives_matrix[4] = [6, 128, 2.6, 150000, 7]  # Huawei P30
alternatives_matrix[5] = [8, 128, 2.85, 250000, 9]  # Google Pixel 7
alternatives_matrix[6] = [6, 128, 2.2, 80000, 11]  # Xiaomi Redmi Note 10
alternatives_matrix[7] = [8, 256, 3.0, 300000, 21]  # Samsung Galaxy S22
alternatives_matrix[8] = [8, 256, 2.8, 350000, 13]  # Motorola Razr+
alternatives_matrix[9] = [3, 64, 2.5, 180000, 5]  # Iphone X R
alternatives_matrix[10] = [8, 256, 2.7, 220000,19 ] # Samsung Galaxy Note 10


### Exécuter le programme
weights, CR, scores = main(comparison_matrix, alternatives_matrix)

print("Poids des critères :")
for i, weight in enumerate(weights):
    print(f"Critère {i + 1}: {weight:.4f}")

print(f"Ratio de cohérence (CR): {CR:.4f}")
if CR < 0.1:
    print("La matrice de comparaison est cohérente.")
else:
    print("La matrice de comparaison n'est pas cohérente.")

print("\nScores des alternatives :")
for i, score in enumerate(scores):
    print(f"{alternatives[i]} : {score:.4f}")

# Déterminer la meilleure alternative
best_alternative_index = np.argmax(scores)
print(f"\nLa meilleure alternative est : {alternatives[best_alternative_index]} avec un score de {scores[best_alternative_index]:.4f}")