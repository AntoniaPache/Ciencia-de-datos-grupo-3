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
