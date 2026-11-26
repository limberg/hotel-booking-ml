### Predicción de cancelaciones de reservas hoteleras mediante Machine Learning

## Problema de negocio

Los hoteles necesitan anticipar las cancelaciones de reservas para mejorar la planificación de su ocupación y optimizar la gestión de su disponibilidad.

## Objetivo general

Desarrollar un sistema automático de clasificación binaria capaz de estimar la probabilidad de cancelación de una reserva hotelera a partir de sus características, utilizando diferentes algoritmos de Machine Learning.

## Resultado esperado

Obtener un modelo que identifique las reservas con mayor riesgo de cancelación, proporcionando información que permita al hotel anticiparse y tomar mejores decisiones operativas y comerciales.

## Datos

El dataset utilizado contiene información relacionada con reservas hoteleras. Entre sus variables se incluyen el tipo de hotel, la antelación de la reserva, fechas de llegada, duración de la estancia, número de adultos, niños y bebés, tipo de comida, país de origen, segmento de mercado, canal de distribución, historial del cliente, tipo de habitación, tipo de depósito, agente, empresa, precio medio diario y estado final de la reserva. La variable objetivo es is_canceled, donde 0 indica que la reserva no fue cancelada y 1 que sí fue cancelada.

En las primeras observaciones del archivo también se aprecia la presencia de valores ausentes en columnas como company, agent, country y children, por lo que será necesario tratar estos casos durante el preprocesamiento.

## Variable objetivo

is_canceled: variable objetivo binaria.

0 = reserva no cancelada

1 = reserva cancelada

## Consideraciones

Algunas variables describen características de la reserva en el momento de la solicitud, mientras que otras aportan información del resultado final. Por ello, durante el modelado se deberá evitar el uso de variables que puedan provocar fuga de información (Data Leakage) y sesgar la predicción.