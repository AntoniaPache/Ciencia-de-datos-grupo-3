# Ciencia de datos grupo 3
TP final para la materia de Ciencia de datos de la facultad.

## 🚌 Dominio del Negocio: SUBE (Sistema Único de Boleto Electrónico)

SUBE es el sistema de pago electrónico para el transporte público en Argentina, implementado por el Ministerio de Transporte. Este sistema permite a los usuarios acceder a diversos medios de transporte público como colectivos, trenes y subtes mediante una única tarjeta, facilitando la movilidad urbana y la recolección de datos sobre el uso del transporte público.

### Modelo de Sistema
El sistema SUBE opera a través de diferentes componentes y servicios:
- Tarjetas SUBE personalizadas y anónimas
- Red de puntos de carga
- Sistema de validación en transportes
- Integración tarifaria entre diferentes medios

### Tipos de Transporte
El sistema SUBE abarca diversos medios de transporte:
1. **Transporte Automotor**
   - Colectivos urbanos
   - Servicios de media y larga distancia
   - Transportes especiales

2. **Transporte Ferroviario**
   - Trenes urbanos y suburbanos
   - Subterráneos
   - Premetro

3. **Transporte Fluvial**
   - Servicios de lanchas y catamaranes

## 📊 Fuentes de Datos
La Dirección Nacional de Desarrollo Tecnológico del Ministerio de Transporte proporciona datos abiertos sobre las transacciones SUBE a través del portal datos.gob.ar. Los datasets contienen información valiosa sobre:
- Cantidad de transacciones diarias
- Datos históricos desde 2020
- Actualización diaria de la información
- Distribución por tipo de transporte

## 🎯 Problemática y Propuesta de Valor

### Definición del Problema
El sistema SUBE enfrenta desafíos relacionados con la optimización del servicio de transporte público:
- Variaciones en la demanda de transporte
- Patrones de uso por temporada y eventos especiales
- Necesidad de planificación eficiente de recursos

### Propuesta de Valor
Mediante el análisis exploratorio de los datos históricos de transacciones SUBE, este proyecto propone:
- Identificar patrones de uso del transporte público
- Analizar tendencias temporales en la demanda
- Proporcionar insights para la mejora del servicio

**Valor comercial:**
- Optimización de la frecuencia del servicio
- Mejor planificación de recursos
- Mejora en la experiencia del usuario
- Toma de decisiones basada en datos

## 🔍 Hipótesis de Trabajo

**Hipótesis central:**
"Existen patrones temporales definidos en el uso del transporte público que pueden identificarse a través del análisis de las transacciones SUBE, permitiendo optimizar la planificación del servicio."

**Sub-hipótesis:**
1. Las transacciones siguen patrones estacionales identificables
2. Existen diferencias significativas en el uso entre días laborables y no laborables
3. Los eventos especiales impactan en el volumen de transacciones
4. La pandemia de COVID-19 generó cambios en los patrones de uso

## 👥 Audiencia
Este trabajo está dirigido a:

**Autoridades de Transporte:**
- Comprensión de patrones de uso
- Optimización de servicios
- Planificación de recursos
- Mejora de la experiencia del usuario

**Operadores de Transporte:**
- Ajuste de frecuencias y capacidades
- Optimización de recursos
- Planificación operativa
- Reducción de costos operativos

## 📌 Conclusiones y Limitaciones para Machine Learning Predictivo

Tras el análisis exploratorio realizado sobre los datos de transacciones SUBE, se pueden extraer los siguientes puntos clave en relación a las hipótesis planteadas:

- **Hipótesis central:** Se observan patrones temporales definidos en el uso del transporte público, como mayor demanda en días laborables y una fuerte concentración en el AMBA. Esto valida parcialmente la hipótesis central.
- **Sub-hipótesis 1:** Se identifican ciertos patrones estacionales, pero la variabilidad mensual es muy alta y no siempre predecible.
- **Sub-hipótesis 2:** Se confirma que existen diferencias significativas entre días laborables y no laborables (la demanda cae un 44,7% en fines de semana).
- **Sub-hipótesis 3:** Los datos muestran saltos y caídas abruptas que sugieren impacto de eventos especiales, pero estos no están explícitamente identificados en los datos.
- **Sub-hipótesis 4:** Se observan cambios notables en la demanda durante la pandemia de COVID-19, pero la recuperación y los cambios posteriores no siguen patrones simples.

### ¿Por qué no es posible hacer predicción confiable con Machine Learning?

Se intentó ajustar un modelo de predicción de demanda **mensual** utilizando Prophet (modelo de series temporales) sobre los datos agregados. Los resultados muestran que:

- El modelo no logra capturar la variabilidad real de la demanda mensual (R2 negativo, errores altos y residuos grandes).
- Los residuos muestran patrones no explicados y alta dispersión, lo que indica que hay factores externos y no modelados que afectan la demanda.
- Las variables disponibles (fecha, provincia, tipo de transporte) no son suficientes para explicar ni predecir la demanda futura con precisión.

**Justificación técnica (valores reales):**

```python
# INTENTO DE PREDICCIÓN MENSUAL Y JUSTIFICACIÓN
# ...
MAE mensual: 191,557,533
RMSE mensual: 210,799,260
R2 mensual: -8.917
# y_true: [307956052 280818381 336390154 349483764 365452900 333547187 351676034 373951328 368348487 377762029 361743062 328481442 270681840 270547148 329032176 347072568 340627745  79011877]
# y_pred: [5.33775349e+08 5.17820964e+08 5.79054564e+08 4.51395194e+08 4.45635453e+08 4.58419691e+08 4.62692174e+08 4.75728131e+08 4.90733711e+08 5.13646268e+08 5.43038234e+08 5.66193072e+08 4.65477004e+08 4.91333040e+08 6.13438675e+08 5.33994783e+08 5.34843736e+08 5.43399717e+08]
```

- **MAE mensual:** 191.557.533
- **RMSE mensual:** 210.799.260
- **R2 mensual:** -8.917 (negativo, lo que indica que el modelo es peor que predecir la media)

Además, a pesar de los distintos enfoques probados (predicción diaria, semanal, mensual, por provincia, por tipo de transporte, etc.), **ningún modelo predictivo logra resultados satisfactorios**. Las razones principales son:

- **Falta de variables explicativas**: Los datos solo contienen información agregada por fecha, provincia y tipo de transporte. No hay variables sobre eventos, clima, feriados, huelgas, cambios tarifarios, ni datos socioeconómicos, que son determinantes para la demanda.
- **Alta variabilidad y shocks externos**: Se observan saltos abruptos y caídas en la demanda que no pueden ser anticipados por el modelo, ya que no hay información sobre los factores que los causan (por ejemplo, pandemia, restricciones, eventos masivos, etc.).
- **Estacionalidad y patrones no estables**: Aunque hay cierta estacionalidad, los patrones cambian de un año a otro y no son predecibles solo con la fecha.
- **Datos agregados**: La agregación a nivel mensual, diario o incluso por provincia, oculta la dinámica real y la heterogeneidad del sistema de transporte.
- **Resultados de los modelos**: Los intentos de predicción (mensual, diaria, etc.) arrojan métricas muy malas (R2 negativo, errores altos), lo que indica que el modelo es peor que simplemente predecir la media histórica.

> **Conclusión general:**
> Con los datos disponibles, solo es posible hacer análisis descriptivo y exploratorio. Para cualquier análisis predictivo confiable se necesitarían variables adicionales y datos más detallados. Los modelos actuales no pueden anticipar cambios ni capturar la complejidad real del sistema de transporte.
