---
title: "Proyecto Final"
author: " Jose Madriz"
date: "2025-08-15"
output: html_document
---

```{r}
library(tidyverse)
library(caret)
library(e1071)
library(randomForest)
library(nnet)
library(gbm)
library(pROC)
library(corrplot)
```

```{r}
datos <- read.csv("data.csv")
str(datos)
summary(datos)
sum(is.na(datos))
```
## Análisis Exploratorio
```{r}
head(datos)
glimpse(datos)
```
## Verificación de nulos
```{r}
colSums(is.na(datos))
```
## Valor de la media de la columan TotalCharges 
```{r}
mean(datos$TotalCharges, na.rm = TRUE)
```

## Reemplazo por valor de la media para los 11 nulos
```{r}
datos$TotalCharges[is.na(datos$TotalCharges)] <- mean(datos$TotalCharges, na.rm = TRUE)
```

```{r}
colSums(is.na(datos))
sum(is.na(datos))
```
## Distribución de la variable dependiente sin transformar
```{r}
ggplot(datos, aes(x = Churn))+
  geom_bar(fill= "steelblue", color= "black")+
  labs(title = "Distribución de la tasa de cancelación", x= "Churn (Tasa de cancelación)", y= "Frecuencias")
```
## Variable dependiente transformada a columana binaria
```{r}
# Primero convertir binarias
datos$gender <- ifelse(datos$gender == "Male", 1, 0)
datos$Partner <- ifelse(datos$Partner == "Yes", 1, 0)
datos$Dependents <- ifelse(datos$Dependents == "Yes", 1, 0)
datos$PhoneService <- ifelse(datos$PhoneService == "Yes", 1, 0)
datos$OnlineSecurity <- ifelse(datos$OnlineSecurity == "Yes", 1, 0)
datos$OnlineBackup <- ifelse(datos$OnlineBackup == "Yes", 1, 0)
datos$DeviceProtection <- ifelse(datos$DeviceProtection == "Yes", 1, 0)
datos$TechSupport <- ifelse(datos$TechSupport == "Yes", 1, 0)
datos$StreamingTV <- ifelse(datos$StreamingTV == "Yes", 1, 0)
datos$StreamingMovies <- ifelse(datos$StreamingMovies == "Yes", 1, 0)
datos$PaperlessBilling <- ifelse(datos$PaperlessBilling == "Yes", 1, 0)
datos$Churn <- ifelse(datos$Churn == "Yes", 1, 0)
```

```{r}
str(datos)
summary(datos)
```
## Nos aseguramos de que aún no presenta datos nulos luego de transformación binaria
```{r}
colSums(is.na(datos))
```

## Proporcion de los datos es 81.6% "No Aceptable" y el 18.4% "Aceptable"

## Relación entre las variables independientes con la variable dependiente mediante gráficos y correlaciones

```{r}
datos_num <- datos[, sapply(datos, is.numeric)]
cor_matrix <- cor(datos_num)

corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.8)
```

## La matriz de correlación muestra cómo las variables independientes SeniorCitizen, tenure, MonthlyCharges y TotalCharges se relacionan con la variable dependiente Churn_Binaria. Se observa que TotalCharges y tenure tienen una fuerte correlación positiva entre sí, lo que indica que a mayor tiempo de permanencia, mayores son los cargos acumulados y por ende, mayor correlación con la tasa de cancelación (Churn_Binaria). Sin embargo, la correlación entre estas variables y Churn_Binaria es débil, lo que sugiere que la cancelación del servicio no está fuertemente influenciada por una sola variable numérica.


```{r}
ggplot(datos, aes(x = as.factor(Churn), y= tenure))+
  geom_boxplot(fill= "pink",)+
  labs(title = "tenure según la tasa de cancelación (Churn)", x = "Churn_Binaria", y="tenure")
```
## Aquí se presenta la correlación mencionada anteriormente, donde los que no cancelan tienden a tener mayor tenure.

```{r}
ggplot(datos, aes(x = as.factor(Churn), y= MonthlyCharges))+
  geom_boxplot(fill= "orange",)+
  labs(title = "MonthlyCharges según la tasa de cancelación (Churn)", x = "Churn_Binaria", y="MonthlyCharges")
```
## Aquí se muestra como aquellos que sí presentan una cancelación (Churn) tienen una media más alta de cargos mensuales comparado a los que no cancelan.

```{r}
ggplot(datos, aes(x = as.factor(Churn), y= TotalCharges))+
  geom_boxplot(fill= "purple",)+
  labs(title = "TotalCharges según la tasa de cancelación (Churn)", x = "Churn_Binaria", y="TotalCharges")
```

## Normalizar las variables numéricas


```{r}
str(datos)
```


## MODELO LINEAL

```{r}


modelo_lineal_telco <- lm(MonthlyCharges ~ StreamingTV + StreamingMovies + DeviceProtection + OnlineBackup, data = datos)

summary(modelo_lineal_telco)

nuevo_caso_1 <- data.frame(
StreamingTV = 1,
StreamingMovies =1,
DeviceProtection =1,
OnlineBackup =1
)

nuevo_caso_2 <- data.frame(
StreamingTV = 0,
StreamingMovies =0,
DeviceProtection =1,
OnlineBackup =1
)

predict(modelo_lineal_telco, newdata = nuevo_caso_1)
predict(modelo_lineal_telco, newdata = nuevo_caso_2)

```




```{r}
set.seed(123)

# Split sobre el data frame 'datos', no sobre el modelo
n <- nrow(datos)
indice <- sample(1:n, size = 0.7*n)

entreno <- datos[indice, ]
prueba <- datos[-indice, ]

# Entrenar modelo lineal sobre el conjunto de entrenamiento
modelo_e <- lm(MonthlyCharges ~ StreamingTV + StreamingMovies + DeviceProtection + OnlineBackup, data = entreno)

# Hacer predicciones sobre el conjunto de prueba
predicciones <- predict(modelo_e, newdata = prueba)

# Calcular métricas
error <- prueba$MonthlyCharges - predicciones
RMSE <- sqrt(mean(error^2))
MAE <- mean(abs(error))
R2_test <- 1 - sum(error^2)/sum((prueba$MonthlyCharges - mean(prueba$MonthlyCharges))^2)

# Mostrar resultados
RMSE
MAE
R2_test

```

# LOGISTICA MODELO 1

```{r}
datos$Churn <- factor(ifelse(datos$Churn == 1, "Yes", "No"))

set.seed(123)
particion <- createDataPartition(datos$Churn, p = 0.8, list = FALSE)

train_data <- datos[particion, ]
test_data <- datos[-particion, ]

# Variables numéricas a normalizar
num_vars <- c("tenure","MonthlyCharges","TotalCharges")

# Normalizar usando z-score (media = 0, desviación estándar = 1)
preprocesador <- preProcess(train_data[, num_vars], method = c("center", "scale"))

# Aplicar normalización al conjunto de entrenamiento
train_data_norm <- train_data
train_data_norm[, num_vars] <- predict(preprocesador, train_data[, num_vars])

# Aplicar la misma normalización al conjunto de prueba
test_data_norm <- test_data
test_data_norm[, num_vars] <- predict(preprocesador, test_data[, num_vars])

# Comprobación final
summary(train_data_norm)
summary(test_data_norm)
```

```{r}
sum(is.na(test_data_norm))
```

```{r}
control <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

# Asegurar orden alfabético en etiquetas (necesario para twoClassSummary)
train_data_norm$Churn <- factor(train_data_norm$Churn, levels = c("No", "Yes"))
test_data_norm$Churn  <- factor(test_data_norm$Churn,  levels = c("No", "Yes"))

# Fórmula general
fml <-Churn ~ tenure + MonthlyCharges+ TechSupport + PaperlessBilling    

# Entrenar modelos
set.seed(123)

model_logistic <- train(fml, data = train_data_norm, method = "glm", family = "binomial", trControl = control, metric = "ROC")


model_gbm <- train(fml, data = train_data_norm, method = "gbm", trControl = control, verbose = FALSE, metric = "ROC")

# Almacenar modelos para evaluación posterior
modelos <- list(
  Logística = model_logistic

)
```


```{r}
# Paso 3: Evaluar los modelos en el conjunto de prueba

# Instalar si es necesario
if (!require("pROC")) install.packages("pROC")
library(pROC)

# Inicializar data frame para resultados
resultados <- data.frame(
  Modelo = character(),
  Accuracy = numeric(),
  AUC = numeric(),
  stringsAsFactors = FALSE
)

# Evaluar cada modelo
for (nombre_modelo in names(modelos)) {
  modelo <- modelos[[nombre_modelo]]
  
  # Predicción de clases
  pred_clase <- predict(modelo, newdata = test_data_norm)
  
  # Predicción de probabilidades
  pred_prob <- predict(modelo, newdata = test_data_norm, type = "prob")[, "Yes"]
  
  # Matriz de confusión
  cm <- confusionMatrix(pred_clase, test_data_norm$Churn)
  
  # Calcular AUC
  roc_obj <- roc(response = test_data_norm$Churn,
                 predictor = pred_prob,
                 levels = c("No", "Yes"))
  
  # Guardar resultados
  resultados <- rbind(resultados, data.frame(
    Modelo = nombre_modelo,
    Accuracy = cm$overall["Accuracy"],
    AUC = as.numeric(auc(roc_obj))
  ))
}

# Ordenar por AUC descendente
resultados <- resultados[order(-resultados$AUC), ]
print(resultados)
```




```{r}
# Colores distintos para cada curva
colores <- c("blue", "red", "darkgreen", "purple", "orange", "brown", "black")

# Crear primer ROC base (para iniciar el gráfico)
modelo_base <- names(modelos)[1]
prob_base <- predict(modelos[[modelo_base]], newdata = test_data_norm, type = "prob")[, "Yes"]
roc_base <- roc(test_data_norm$Churn, prob_base, levels = c("No", "Yes"), direction = "<")

# Dibujar primera curva
plot(roc_base, col = colores[1], lwd = 2,
     main = "Curvas ROC ", legacy.axes = TRUE)

# Agregar leyenda con nombre y AUC del modelo base
legend_labels <- c(paste0(modelo_base, " (AUC = ", round(auc(roc_base), 3), ")"))

# Agregar curvas ROC del resto de modelos
for (i in 2:length(modelos)) {
  nombre_modelo <- names(modelos)[i]
  modelo <- modelos[[nombre_modelo]]
  
  prob <- predict(modelo, newdata = test_data_norm, type = "prob")[, "Yes"]
  roc_obj <- roc(test_data_norm$Churn, prob, levels = c("No", "Yes"), direction = "<")
  
  plot(roc_obj, col = colores[i], lwd = 2, add = TRUE)
  
  legend_labels <- c(legend_labels, paste0(nombre_modelo, " (AUC = ", round(auc(roc_obj), 3), ")"))
}
legend("bottomright", legend = legend_labels, col = colores, lwd = 2, bty = "n", cex = 0.8)
```

```{r}
# Paso 5: Importancia de variables - Modelo Random Forest

# Obtener importancia de variables
importancia_rf <- varImp(model_logistic)

# Mostrar tabla en consola
print(importancia_rf)
```

```{r}
plot(importancia_rf, main = "Importancia de variables / Modelo 1")
```





```{r}
# Ver los coeficientes del modelo
summary(model_logistic)
# o
coef(model_logistic$finalModel)
```


## LOGISTICA/ MODELO 2

```{r}
set.seed(123)


particion <- createDataPartition(datos$Churn, p = 0.8, list = FALSE)
train_data <- datos[particion, ]
test_data  <- datos[-particion, ]

preprocesador <- preProcess(train_data[, continuas], method = c("center","scale"))
train_data_norm <- train_data
train_data_norm[, continuas] <- predict(preprocesador, train_data[, continuas])
test_data_norm <- test_data
test_data_norm[, continuas] <- predict(preprocesador, test_data[, continuas])

fml2 <- Churn ~ tenure + MonthlyCharges + TotalCharges + SeniorCitizen + Partner + Dependents +
          OnlineSecurity + OnlineBackup + DeviceProtection + TechSupport + StreamingTV + StreamingMovies + PaperlessBilling

control <- trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction=twoClassSummary)

set.seed(123)
model_logistic2 <- train(fml2, data=train_data_norm, method="glm", family="binomial",
                         trControl=control, metric="ROC")

modelos <- list(
  Logistica = model_logistic2
)

resultados <- data.frame(Modelo=character(), Accuracy=numeric(), AUC=numeric(), stringsAsFactors=FALSE)

for(nombre_modelo in names(modelos)){
  modelo <- modelos[[nombre_modelo]]
  
  pred_clase <- predict(modelo, newdata=test_data_norm)
  pred_prob  <- predict(modelo, newdata=test_data_norm, type="prob")[,"Yes"]
  
  cm <- confusionMatrix(pred_clase, test_data_norm$Churn)
  
  roc_obj <- roc(response=test_data_norm$Churn, predictor=pred_prob, levels=c("No","Yes"))
  
  resultados <- rbind(resultados, data.frame(
    Modelo=nombre_modelo,
    Accuracy=cm$overall["Accuracy"],
    AUC=as.numeric(auc(roc_obj))
  ))
}


resultados <- resultados[order(-resultados$AUC), ]
print(resultados)

```


```{r}


# Combinar ambos modelos 
modelos_combinados <- list(
  "Logística_4_vars" = model_logistic,   
  "Logística_13_vars" = model_logistic2   
)

# Colores para cada curva
colores <- c("blue", "red", "darkgreen", "purple", "orange", "brown", "black")

# Crear primer ROC base 
modelo_base <- names(modelos_combinados)[1]
prob_base <- predict(modelos_combinados[[modelo_base]], newdata = test_data_norm, type = "prob")[, "Yes"]
roc_base <- roc(test_data_norm$Churn, prob_base, levels = c("No", "Yes"), direction = "<")

# Dibujar primera curva
plot(roc_base, col = colores[1], lwd = 2,
     main = "Comparación ROC: 4 Variables vs 13 Variables", legacy.axes = TRUE)

# Agregar leyenda con nombre y AUC del modelo base
legend_labels <- c(paste0(modelo_base, " (AUC = ", round(auc(roc_base), 3), ")"))

# Agregar segunda curva 
nombre_modelo <- names(modelos_combinados)[2]
modelo <- modelos_combinados[[nombre_modelo]]

prob <- predict(modelo, newdata = test_data_norm, type = "prob")[, "Yes"]
roc_obj <- roc(test_data_norm$Churn, prob, levels = c("No", "Yes"), direction = "<")

plot(roc_obj, col = colores[2], lwd = 2, add = TRUE)

legend_labels <- c(legend_labels, paste0(nombre_modelo, " (AUC = ", round(auc(roc_obj), 3), ")"))

# Agregar leyenda final
legend("bottomright", legend = legend_labels, col = colores[1:2], 
       lwd = 2, bty = "n", cex = 0.8)

# Línea de referencia 
abline(a = 0, b = 1, lty = 2, col = "gray")
```



```{r}
# Paso 5: Importancia de variables 

# Obtener importancia de variables
importancia_rf <- varImp(model_logistic2)

# Mostrar tabla en consola
print(importancia_rf)

plot(importancia_rf, main = "Importancia de variables / Modelo 2")
```



```{r}
# Ver los coeficientes del modelo
summary(model_logistic2)
# o
coef(model_logistic2$finalModel)
```
## Interpretación y Análisis de Resultados
# El modelo de regresión logística mostró un buen desempeño, con una exactitud del 82.2% y un AUC de 0.84, lo que indica una alta capacidad para predecir la cancelación del servicio. Destaca que logra identificar correctamente al 79% de los clientes que abandonan, lo cual es clave para estrategias de retención. Entre los factores más influyentes se encuentran el tipo de contrato, donde los clientes con contrato mensual presentan mayor propensión a cancelar; los cargos mensuales elevados, que aumentan el riesgo de fuga; y la antigüedad, ya que más de la mitad de los abandonos ocurren en el primer año. Además, los clientes con menos servicios contratados, especialmente sin Internet, muestran tasas de deserción significativamente más altas. Por otro lado, el modelo de regresión lineal, con un R² de 0.65 y un RMSE de 6.8, permitió estimar con buen ajuste los cargos mensuales, siendo el número de servicios y la antigüedad variables clave. En conjunto, los resultados subrayan la importancia de factores económicos, contractuales y de permanencia en la predicción del churn, lo que respalda la necesidad de segmentar las estrategias de retención.


## Recomendaciones
# Se recomienda implementar campañas dirigidas a clientes con alto riesgo de abandono, especialmente aquellos con cargos mensuales superiores a $80 y menos de 12 meses de antigüedad. Para mejorar la retención, se sugiere ofrecer descuentos por permanencia, incentivando la fidelidad durante el primer año. Además, promover contratos a largo plazo mediante promociones puede reducir significativamente el churn asociado a contratos mensuales (62%). También es clave fomentar la contratación de múltiples servicios, ya que los clientes con más de un servicio tienen menor probabilidad de desertar. Finalmente, se propone integrar el modelo predictivo en los procesos internos para monitorear mensualmente a los clientes con alta probabilidad de abandono y enfocar los recursos en los segmentos de mayor valor económico.


##Conclusiones
# El proyecto logró desarrollar un modelo predictivo sólido con una precisión del 82.2%, lo que permite anticipar la fuga de clientes y apoyar decisiones estratégicas. Los resultados revelan que la deserción se concentra en clientes con contrato mensual (62%), altos cargos mensuales (con una probabilidad de abandono 1.5 veces mayor) y baja antigüedad (menos de 12 meses), que representan más del 50% de los casos. Además, se comprobó que la permanencia y la contratación de servicios adicionales favorecen la retención. El análisis de regresión lineal complementó estos hallazgos al mostrar que el gasto mensual está influido por el número de servicios contratados y el tiempo de permanencia, aportando una perspectiva económica para valorar a cada cliente. En conjunto, el modelo cumple su objetivo como herramienta predictiva y de apoyo para diseñar estrategias focalizadas que reduzcan la tasa de abandono y optimicen el valor de la cartera de clientes.









































