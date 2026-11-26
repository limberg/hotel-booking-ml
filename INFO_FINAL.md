## Justificación del problema

El proyecto se centra en un problema de clasificación binaria, cuyo objetivo es predecir si una reserva hotelera será cancelada o no. Esta elección permite abordar una necesidad real de negocio: anticipar cancelaciones para mejorar la planificación de la ocupación, la gestión operativa y la toma de decisiones del hotel. El documento de la práctica especifica precisamente que se debe diseñar un sistema automático para entrenar, evaluar y comparar distintos modelos de clasificación binaria.

En el dataset, la variable objetivo es "is_canceled", donde 0 indica que la reserva no fue cancelada y 1 que sí lo fue. Esto convierte el problema en una tarea clara de predicción supervisada, donde el modelo aprende patrones históricos para estimar el riesgo de cancelación de nuevas reservas.

## Análisis exploratorio de datos (EDA)

El dataset contiene una estructura rica para el análisis, con variables relacionadas con la reserva, el cliente, el canal de distribución y el estado final de la reserva. En las primeras observaciones ya se aprecia la presencia de valores ausentes en columnas como company, agent, country y children, por lo que será necesario tratarlas antes del modelado.

Durante el EDA se identificaron varios aspectos relevantes:

La variable objetivo presenta desbalance moderado, con más reservas no canceladas que canceladas.

Existen filas duplicadas, por lo que será necesario revisar si corresponden a registros repetidos reales o a duplicados del fichero.

Algunas variables numéricas, como lead_time y adr, muestran valores extremos, por lo que conviene analizarlas con cuidado antes de decidir su tratamiento.

lead_time resulta especialmente interesante porque mide la antelación con la que se realiza una reserva.

Variables como hotel, market_segment, deposit_type y customer_type aportan información útil para comparar el comportamiento de las cancelaciones.

La comparación entre hoteles muestra diferencias de comportamiento, lo que refuerza la utilidad de estas variables para el modelado.

A nivel interpretativo, el EDA permite concluir que el problema está bien planteado para un enfoque predictivo, ya que existen variables potencialmente explicativas y una variable objetivo clara. Además, el análisis descriptivo justifica decisiones futuras de preprocesamiento, como el tratamiento de nulos, duplicados, outliers y codificación de variables categóricas.

## Conclusión

El objetivo del proyecto es construir un sistema capaz de estimar el riesgo de cancelación de una reserva hotelera a partir de sus características. El análisis exploratorio confirma que el dataset es adecuado para este propósito y que existen patrones que pueden aprovecharse en modelos de Machine Learning para comparar distintos algoritmos y seleccionar el más efectivo.