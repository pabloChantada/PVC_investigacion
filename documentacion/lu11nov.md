# Clase joaquim
## Primera parte - clasificación pixel a pixel.

Accuracy no vale. hay que usar *FScore*, *F1* *Precission*, *Reccal*. Además el *Dice* es propio de segmentación. Solo valen métricas de *test* o *cross-validation*. También hay que enseñar $\sigma$. En CV recomienda $K = 5$. Si desviación típica pequeña el problema no es dependiente de los datos. 

Segmentacion: Clasificacion pixel a pixel.

Analisis cuantitativo (arriba) y cualitativo (imégenes). 

Es un problema desvalanceado. (Problema: hay poco error porque hay muchos pixeles que salen bien.) 

Formas para solucionar:
 - _DownSampling_: quitar pixeles negros. Más rápido. Datos originales.
 - _OverSampling_: generaliza DataAugmentation. No hace falta generar aumentar el dataset, se generan los vectores de características. 

Hay familias de características. Debemos saber si unirlas vale la pena ($C_1 + C_2 + C_3 >_? C_i$). No es la idea perder tiempo entrenando. Con un modelo llega. Porque ese modelo? Mejor familia caracteristica? Aumentando dimensionalidad hay mejores resultados? Cualitivamente tiene pinta? Hicisteis post-procesado?

Conocimiento de dominio:
 - Unir carreteras: post-procesado

## Segunda parte - reconocimiento de patrones.

Se pueden descargar mas imágenes de internet. Es un problema multiclase, se puede tener una terceera clase: otros. (O  otros = {emu, flamenco}, jutificar decición.) Misma tabla que en parte 1. La gran diferencia es que en vez de clasificar pixeles se clasifican imágenes.  

Preguntas:
 - _Como representar?_ depende de la librería de Python.
 - _Necesito todos los pixeles?_ solo los que tienen interes -> máscaras.
 - _Para qué sirve las máscaras?_  simplifica la clasificación.

Recomendación: hacelo sin máscaras y poner la comparativa en la tabla.

Los vectores de las máscaras no miden las máscaras:
 - Truncar
 - Coger aleatoriamente pixeles y descartar el resto. (Estadísticamente converge.) No lo exige.
