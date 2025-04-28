# Ciencia-de-datos-grupo-3
TP final para la materia de Ciencia de datos de la facultad.

## 🚲 Dominio del Negocio: Bay Wheels

Bay Wheels, anteriormente conocido como Ford GoBike, es un sistema de bicicletas compartidas operado por Lyft en el Área de la Bahía de San Francisco. Este servicio proporciona una red extensa de estaciones de acoplamiento y miles de bicicletas distribuidas estratégicamente por toda la región, permitiendo a los usuarios realizar viajes urbanos de manera eficiente y ecológica.

### Modelo de Negocio
El modelo de negocio de Bay Wheels ofrece diversas opciones para satisfacer las necesidades de diferentes perfiles de usuarios, incluyendo:
- Viajes individuales
- Pases diarios
- Membresías anuales

### Tipos de Bicicletas
La flota de Bay Wheels consta de tres tipos principales:
1. **Bicicletas eléctricas de nueva generación**
   - Mayor autonomía de batería
   - Motores más potentes
   - Opciones adicionales de altura del asiento

2. **Bicicletas eléctricas originales**
   - Velocidades de hasta 20 MPH
   - Asistencia al pedaleo

3. **Bicicletas clásicas (no eléctricas)**
   - Diseñadas para facilitar el desplazamiento en entorno urbano

## 📊 Fuentes de Datos
Bay Wheels pone a disposición pública sus datos de viajes a través de un bucket de Amazon S3 (baywheels-data). Los datasets contienen información valiosa y anónima sobre cada viaje realizado, incluyendo:
- Duración del viaje en segundos
- Fecha y hora de inicio y finalización
- Identificación y localización de estaciones de origen y destino
- Tipo de usuario (suscriptor o cliente casual)
- Coordenadas geográficas de las estaciones

## 🎯 Problemática y Propuesta de Valor

### Definición del Problema
Bay Wheels enfrenta desafíos operativos significativos relacionados con la distribución eficiente de bicicletas entre estaciones:
- Estaciones que quedan sin bicicletas disponibles durante horas pico
- Estaciones que alcanzan su capacidad máxima
- Recursos de mantenimiento y redistribución no optimizados

### Propuesta de Valor
Mediante el análisis exploratorio de los datos históricos de viajes, este proyecto propone:
- Optimizar la redistribución de bicicletas entre estaciones
- Mejorar la planificación de mantenimiento preventivo
- Proporcionar recomendaciones estratégicas para la expansión o reubicación de estaciones

**Valor comercial:**
- Reducción de costos operativos
- Mayor satisfacción del cliente
- Mejor aprovechamiento de los recursos existentes
- Apoyo a la toma de decisiones estratégicas basadas en evidencia

## 🔍 Hipótesis de Trabajo

**Hipótesis central:**
"Existen patrones temporales y geográficos definidos en el uso de las bicicletas compartidas de Bay Wheels que pueden identificarse a partir del análisis exploratorio del dataset histórico de viajes, sin necesidad de modelos predictivos complejos."

**Sub-hipótesis:**
1. Los viajes siguen patrones predecibles para optimizar la distribución
2. Existen rutas preferenciales entre pares de estaciones
3. El tipo de bicicleta influye en los patrones de uso
4. Los patrones difieren entre usuarios con membresía regular y usuarios del programa Bikeshare for All

## 👥 Audiencia
Este trabajo está dirigido a:

**Gerencia Comercial:**
- Comprensión de patrones de uso
- Optimización de la oferta de servicios
- Mejora de la experiencia del usuario
- Generación de nuevas fuentes de ingresos

**Gerencia Técnica:**
- Implementación de estrategias de redistribución eficientes
- Optimización de mantenimiento
- Reducción de costos operativos
