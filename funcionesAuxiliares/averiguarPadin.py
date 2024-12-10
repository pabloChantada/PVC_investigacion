tamano = (200, 300)

datos = CargaDatos('../../objects')
X, Y = construyeDataset(datos,numPixeles=1500, padin=10, pixeliza=1, balancear=True)



N = 5
n = 50
l=[0]*n
for i in range(n):
    print(i)
    acuracis = []
    for _ in range(N):
        print(_,end=' ')
        
        
        X_entrenamiento, Y_entrenamiento, X_test, Y_test = separaEntrenamientoTest(X,Y,razon=0.9)
        #X_entrenamiento, Y_entrenamiento, X_test, Y_test = separaEntrenamientoTest(X,Y,unoFuera=True)
        
        modelo_svm = svm.SVC(kernel='rbf')
        modelo_svm.fit(X_entrenamiento,Y_entrenamiento)
        Y_predicho = modelo_svm.predict(X_test)
        
        acuracis.append(accuracy_score(Y_test,Y_predicho))


    media = round(np.mean(acuracis),2)
    sigma = round(np.sqrt(np.var(acuracis)),2)
    mediana = round(np.median(acuracis),2)

    l[i]  = [media, sigma, mediana]

l = np.array([list(l[i]) for i in range(len(l))])

plt.plot(l[:,0],label='media acuracy')
plt.plot(l[:,1],label='sigma')
plt.plot(l[:,2],label='mediana')
plt.xlabel("grosor del recorte en pixeles")
plt.legend();