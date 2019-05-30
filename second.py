import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn import *

# Declara a captura de vídeo e já coloca a imagem capturada em 10x10
#frame = cv2.imread('nome_da_imagem.extensão')
frame = cv2.imread('image.jpg')
frame1 = cv2.resize(frame, (10, 10))

# Lista as imagens
imgs= []
for pastaAtual, subPastas, arquivos in os.walk('images'):
    for arquivo in arquivos:
        imgs.append(os.path.join(pastaAtual, arquivo))

# Lê as imagens
r = 0
imgs_g = []
imgs_g.append(cv2.imread(imgs[0]))
while r < len(imgs):
    imgs_g.append(cv2.imread(imgs[r]))
    r += 1

# Coloca elas em 10px x 10px
r = 0
img = []
while r < len(imgs):
    img.append(cv2.resize(imgs_g[r], (10, 10)))
    r += 1

# Concatena os arrays em um só
X = np.array(img)

# Coloca index nos arrays
r = 0
y = []
while r < len(imgs):
    y.append(r)
    r += 1

# Declara y como array
y = np.array(y)

# Remodela y
Y = y.reshape(-1)

# Remodela X com o tamanho de y
X = X.reshape(len(y), -1)

# Cria o classificador 
classifier_linear = SVC(kernel='linear')

# Treinando o classificador com imagens e índices
classifier_linear.fit(X,Y)

# Previsão da categoria da imagem
pre = classifier_linear.predict(frame1.reshape(1,-1))

# Pontuação da previsão
score = classifier_linear.score(X,Y)

# Mostra a previsão
print('Resultado: {}'.format(pre))

# Mostra a pontuação da previsão
print('Precisão: {:.1f}%'.format(score * 100))

# Declara resultado como a imagem da previsão
result = imgs_g[pre[0]]

# Mostra a imagem baseado na previsão
cv2.imshow("result", result)
# Mostra a imagem testada
cv2.imshow('frame', frame)
# Espera pela tecla 0
cv2.waitKey(0)
