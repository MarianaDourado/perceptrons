import numpy as np

epochs = 30
theta = 0
learnrate = 1

# Matriz de Input (6x5)
X = np.array([[1, 0, 1, 1, 1], [0, 1, 0, 1, 1], [1, 0, 1, 0, 1], [0, 1, 1, 1, 1], [0, 0, 1, 1, 1], [0, 0, 0, 1, 1]])

# Matriz de Input da Questão 2
X2 = np.array([[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [0, 1, 0, 0, 1], [0, 1, 1, 1, 1], [1, 0, 0, 1, 1], [0, 0, 0, 0, 1]])

# Matriz de Output (1x6)
Y = np.array([1, 0, 1, 0, 1, 0])

# Matriz de pesos 
W = np.zeros((1, 5))

"""
1 = Gripe
0 = Resfriado
"""

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
t = 5 # Linha de teste (0 a 5)
"""
Gripe: t = par
Resfriado: t = ímpar
"""

output_teste = forward(X[t,:], W)
Y[t] = int(Y[t])

if output_teste != Y[t]:
    W = ajustar_peso(X[t,:], W, Y[t], output_teste)

# -------------------------------- TREINO --------------------------------
i = 0 # Número de interações começa em 0
ordem = [0, 1, 2, 3, 4, 5]
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
    #print(f"Output de treino no for: {output_conj.T}")
    if np.all(output_conj.T == Y): break
    

# -------------------------------- VERIFICAÇÃO --------------------------------
output_treino = forward(X[t,:], W) # Calculando o output de cada linha

if output_treino == Y[t]:
    print("\nSucesso no treinamento.")
else:
    print("\nO treinamento falhou.")
    print(f"Output de treino: {output_conj.T}")
    print(f"Output esperado: {Y}")

# -------------------------------- PRINTS --------------------------------
print(f"Número de interações: {i}")

#print(f"\nO valor escolhido para t foi {t}")
print(f"\nA ordem escolhida foi {ordem}")
#print(f"Pesos [W1, W2, W3, W4, Wb]: {W}")


# -------------------------------- RESPOSTAS --------------------------------
print("\nQ.1 - O número de interações varia entre 1 e 5, dependendo da linha escolhida para teste (t), e a da sequência seguida no treinamento.")

Y2 = np.dot(X2, W.T)
Y2[Y2>theta] = 1
Y2[Y2<=theta] = 0
print(f"Q.2 - O resultado para a tabela dada é: {Y2.T}, sendo 1 gripe, e 0 resfriado.")
print("Q.3 - Sim. O comportamento é baseado no ajuste ao meio em que é inserido, no caso, aos dados, configurando-se como uma característica da inteligência.\n")