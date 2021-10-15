import numpy as np

epochs = 20
theta = 0
learnrate = 1

# Matriz de Input 4x3 (a ultima coluna eh o valor do bias)
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

# Matriz de Output 1x4
Y = np.array([0, 0, 1, 1])

# Matriz inicialmente nula de pesos 1x3 
W = np.zeros((1, 3))


def forward(X, W):
    output = np.dot(X, W.T)
    output = int(output) # Transforma em inteiro

    # processo sgn
    if output>theta: output = 1
    else: output = 0

    return output


def ajustar_peso(X, W, Y, output):
    valor_erro = Y - output
    erro_por_x = np.dot(valor_erro, X)
    W_novo = W + erro_por_x*learnrate
    return W_novo

# -------------------------------- TESTE --------------------------------
t = 3 # Linha de teste
"""
BACH: t = 0
BEETHOVEN: t = 1
EINSTEIN: t = 2
KEPLER: t = 3
"""

output_teste = forward(X[t,:], W)
Y[t] = int(Y[t])

if output_teste != Y[t]:
    W = ajustar_peso(X[t,:], W, Y[t], output_teste)


# -------------------------------- TREINO --------------------------------
i = 0 # Número de interações começa em 0
ordem = [0, 1, 2, 3]
ordem.remove(t) # Para não repetir a linha de treino

for _ in range(epochs):
    for s in ordem:
        Y[s] = int(Y[s])

        output_treino = forward(X[s,:], W) # Calculando o output de cada linha
        if output_treino != Y[s]:
            W = ajustar_peso(X[s,:], W, Y[s], output_treino)
    i = i+1 # Número de interação aumenta 1
    output_conj = np.dot(X, W.T)
    output_conj[output_conj>theta] = 1
    output_conj[output_conj<=theta] = 0
    if np.all(output_conj.T == Y): break
    
# -------------------------------- VERIFICAÇÃO --------------------------------
output_treino = forward(X[t,:], W) # Calculando o output de cada linha

if output_treino == Y[t]:
    print(f"\nSucesso no treinamento.")
else:
    print("\nO treinamento falhou.")
    print(f"Output de treino: {output_conj.T}")
    print(f"Output esperado: {Y}")

# -------------------------------- PRINTS --------------------------------
print(f"Número de interações: {i}")
#print(f"\nO valor escolhido para t foi {t}")
print(f"\nA ordem escolhida foi {ordem}")


#print(f"Pesos [W1, W2, Wb]: {W}")

print("\nQ.1 - O programa de treinamento funciona sempre, diferenciam-se apenas o número de interações e o vetor final de pesos.")
print("Q.2 - O número máximo de interações para corrigir os pesos é 3.\n")

