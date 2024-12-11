import sklearn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import csv
import matplotlib.pyplot as plt
import math

def content_csv(path):
    head=[]
    content=[]
    with open(path,newline='\n') as csvFile:
        #créer un objet lecteur csv
        lecteur_csv = csv.reader(csvFile, delimiter=',')
        #parcourir les lignes du fichier CSV
        i=0
        for ligne in lecteur_csv:
            if i==0:
                head.append(ligne)
            else:
                content.append(ligne)
            i=i+1
    return {"head":head, "data": np.array(content)}#on transforme le contenu en tableau numpy

def matrix_of_one(n):
    list=[]
    for i in range(n):
        list.append(1)
    return np.reshape(np.array(list), (1, -1))

def reduce(X,D):
    transpose_X=np.transpose(X)
    Z=transpose_X @ D @ np.transpose(matrix_of_one(X.shape[0]))
    M=np.transpose(matrix_of_one(X.shape[0]))@np.transpose(Z)
    return X-M

def usual_metrix(X):
    moyennes_colonnes=np.mean(X, axis=0)
    nouvelle_matrice=np.tile(moyennes_colonnes, (X.shape[0],1))
    Y=X-nouvelle_matrice
    inverse_result= np.sum(Y**2, axis=0)
    for i in range(len(inverse_result)):
        inverse_result[i]=inverse_result[i]
    return np.diag(1/inverse_result)

def usual_weight(X):
    D=np.identity(X.shape[0])
    return D/X.shape[0]

def convertir_en_reel(valeur):
    return float(valeur)

def completed_pca(X,M,D,path):
    #notre matrice a p colonnes et n lignes
    
    content = content_csv(path)["data"]
    head = content_csv(path)["head"]
    #on suppose ici que tous les champs de la dataset correspondent à des valeurs numériques
    
    #matrice centrée
    X_c=reduce(X,D)
    #calcul de la matrice S^2
    S_square=np.transpose(X_c)@D@X_c
    #calcul de la matrice S^2 M
    S_squareM=S_square@M
    
    valeurs_propres, vecteurs_propres = np.linalg.eig(S_squareM)
    #les valeurs propres et les vecteurs propres sont bien définies
    matrices_normes_colonnes=[]
    for j in range(S_squareM.shape[1]):
        #je recupère la j eme colonne
        C_j=np.transpose(S_squareM[:, j])
        #je calcule la norme de C_j
        N_C_j=S_squareM[:,j]@M@C_j
        matrices_normes_colonnes.append(N_C_j)
        
    indices_tri = np.argsort(-valeurs_propres, kind='mergesort')#on recupère les indices des éléments triés dans l'ordre décroissant
    
    V = S_squareM/matrices_normes_colonnes
    
    valeurs_propres=valeurs_propres[indices_tri]
    
    #calcul de la matrice des composantes principales
    
    C=X_c@M@V
    
    C_trie=C[:, indices_tri]
    
    #représentons le jeu de données
    C_trie_X=C_trie[:,0]
    C_trie_Y=C_trie[:,1]
    #créer une figure et un axe
    fig, ax = plt.subplots()
    
    #tracer le nuage de points
    content_cs = content_csv(path)["data"][:,-1]
    for j in range(len(C_trie_X)):
        if content_cs[j]=='0':
            ax.scatter(C_trie_X[j],C_trie_Y[j],c='red')
        elif content_cs[j]=='1':
            ax.scatter(C_trie_X[j],C_trie_Y[j],c='blue')
        elif content_cs[j]=='2':
            ax.scatter(C_trie_X[j],C_trie_Y[j],c='yellow')
        else:
            ax.scatter(C_trie_X[j],C_trie_Y[j],c='green')
    
    #afficher le graphique
    plt.title("Nuage des points des composantes principales")
    plt.xlabel("Composante principale 1")
    plt.ylabel("Composante pricipale 2")
    
    plt.show()
    
    
    
X=content_csv("iris.csv")["data"][:,:-1]
Y=np.vectorize(convertir_en_reel)(X)
T=reduce(Y,usual_weight(Y))
#completed_pca(T,usual_metrix(T),usual_weight(T),"iris.csv")
#completed_pca(np.transpose(T),usual_weight(T),usual_metrix(T),"iris.csv")