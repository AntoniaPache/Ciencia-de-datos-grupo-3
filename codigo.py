import pandas as pd
import folium
from IPython.display import display
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import seaborn as sns
# Carga de datos
df1 = pd.read_csv("https://raw.githubusercontent.com/AntoniaPache/Ciencia-de-datos-grupo-3/main/datasets-tpo/dat-ab-usos-2020.csv")
df2 = pd.read_csv("https://raw.githubusercontent.com/AntoniaPache/Ciencia-de-datos-grupo-3/main/datasets-tpo/dat-ab-usos-2021.csv")
df3 = pd.read_csv("https://raw.githubusercontent.com/AntoniaPache/Ciencia-de-datos-grupo-3/main/datasets-tpo/dat-ab-usos-2022.csv")
df4 = pd.read_csv("https://raw.githubusercontent.com/AntoniaPache/Ciencia-de-datos-grupo-3/main/datasets-tpo/dat-ab-usos-2023.csv")
df5 = pd.read_csv("https://raw.githubusercontent.com/AntoniaPache/Ciencia-de-datos-grupo-3/main/datasets-tpo/dat-ab-usos-2024.csv")
df6 = pd.read_csv("https://raw.githubusercontent.com/AntoniaPache/Ciencia-de-datos-grupo-3/main/datasets-tpo/dat-ab-usos-2025.csv")
df = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)
df['PROVINCIA'] = df['PROVINCIA'].replace('CIUDAD AUTÓNOMA DE BUENOS AIRES', 'C.A.B.A')


# Preparación de datos
df['DIA_TRANSPORTE'] = pd.to_datetime(df['DIA_TRANSPORTE'])
df['AÑO'] = df['DIA_TRANSPORTE'].dt.year
df['MES'] = df['DIA_TRANSPORTE'].dt.month
df['MES_NOMBRE'] = df['DIA_TRANSPORTE'].dt.month_name()
df['DIA_SEMANA'] = df['DIA_TRANSPORTE'].dt.day_name()

# Resumen general
print("==== RESUMEN GENERAL ====")
print(f"Período: {df['DIA_TRANSPORTE'].min().date()} a {df['DIA_TRANSPORTE'].max().date()}")
print(f"Total de registros: {len(df):,}")
print(f"Empresas únicas: {df['NOMBRE_EMPRESA'].nunique()}")
print(f"Líneas únicas: {df['LINEA'].nunique()}")
print(f"Provincias: {df['PROVINCIA'].nunique()}")
print(f"Municipios: {df['MUNICIPIO'].nunique()}")
print(f"Cantidad total transportada: {df['CANTIDAD'].sum():,}")

print("\nProvincias únicas:")
print(df['PROVINCIA'].unique())

# Valores nulos
print("\nValores nulos por columna:")
print(df.isnull().sum())

# Distribuciones categóricas
print("\nDistribución por tipo de transporte:")
print(df['TIPO_TRANSPORTE'].value_counts())

print("\nDistribución por AMBA:")
print(df['AMBA'].value_counts())

print("\nDistribución por provincia (top 10):")
print(df['PROVINCIA'].value_counts().head(10))

# Estadísticas de la variable numérica principal
print("\nEstadísticas de la columna CANTIDAD:")
print(df['CANTIDAD'].describe())

# MAPA INTERACTIVO 
# Coordenadas aproximadas de las provincias argentinas
provincias_coords = {
    'BUENOS AIRES': [-36.6769, -60.5588],
    'MENDOZA': [-32.8908, -68.8272],
    'SANTA FE': [-31.6333, -60.7000],
    'SAN JUAN': [-31.5375, -68.5364],
    'JUJUY': [-24.1858, -65.2995],
    'NEUQUÉN': [-38.9516, -68.0591],
    'CHUBUT': [-43.2493, -65.1814],
    'ENTRE RÍOS': [-32.0588, -59.2014],
    'RÍO NEGRO': [-40.8135, -63.0000],
    'CÓRDOBA': [-31.4201, -64.1888],
    'CORRIENTES': [-27.4806, -58.8341],
    'MISIONES': [-26.8754, -54.6568],
    'SALTA': [-24.7821, -65.4232],
    'TUCUMÁN': [-26.8083, -65.2176],
    'SANTIAGO DEL ESTERO': [-27.7824, -64.2642],
    'CATAMARCA': [-28.4696, -65.7852],
    'LA RIOJA': [-29.4331, -66.8563],
    'SAN LUIS': [-33.2949, -66.3361],
    'LA PAMPA': [-36.6167, -64.2833],
    'SANTA CRUZ': [-48.8647, -69.9618],
    'TIERRA DEL FUEGO': [-54.8019, -68.3030],
    'FORMOSA': [-26.1775, -58.1781],
    'C.A.B.A': [-34.6037, -58.3816] # Añadir C.A.B.A. a las coordenadas
}

# Crear mapa base centrado en Argentina
mapa = folium.Map(
    location=[-38.4161, -63.6167],  # Centro de Argentina
    zoom_start=5,
    tiles='OpenStreetMap'
)

# Procesar datos por provincia
provincia_stats = df.groupby('PROVINCIA').agg({
    'CANTIDAD': ['sum', 'mean', 'count'],
    'NOMBRE_EMPRESA': 'nunique',
    'TIPO_TRANSPORTE': lambda x: x.value_counts().to_dict()
}).round(2)

# Aplanar los nombres de columnas
provincia_stats.columns = ['total_cantidad', 'promedio_cantidad', 'num_registros', 'num_empresas', 'tipos_transporte']

# Agregar marcadores por provincia
for provincia in provincia_stats.index:
    if provincia in provincias_coords:
        coords = provincias_coords[provincia]
        stats = provincia_stats.loc[provincia]

        # Determinar el tamaño del marcador basado en el total transportado
        max_total = provincia_stats['total_cantidad'].max()
        radio = max(8, min(50, (stats['total_cantidad'] / max_total) * 40))

        # Crear información del popup
        tipos_transport_str = ""
        if isinstance(stats['tipos_transporte'], dict):
            for tipo, cantidad in stats['tipos_transporte'].items():
                tipos_transport_str += f"  • {tipo}: {cantidad:,}<br>"

        popup_html = f"""
        <div style="width: 200px;">
            <h4 style="margin-bottom: 10px; color: #2E86AB;">{provincia}</h4>
            <hr style="margin: 5px 0;">
            <b>Estadísticas Generales:</b><br>
            • Total transportado: <b>{stats['total_cantidad']:,.0f}</b><br>
            • Promedio diario: <b>{stats['promedio_cantidad']:,.0f}</b><br>
            • Número de registros: <b>{stats['num_registros']:,}</b><br>
            • Empresas activas: <b>{stats['num_empresas']}</b><br>
            <br>
            <b>Por tipo de transporte:</b><br>
            {tipos_transport_str}
        </div>
        """

        # Determinar color basado en el volumen de transporte
        if stats['total_cantidad'] > 50000000:  # Alto volumen
            color = 'red'
        elif stats['total_cantidad'] > 10000000:  # Volumen medio
            color = 'orange'
        else:  # Volumen bajo
            color = 'green'

        folium.CircleMarker(
            location=coords,
            radius=radio,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{provincia}: {stats['total_cantidad']:,.0f} pasajeros",
            color='darkblue',
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=2
        ).add_to(mapa)

# Agregar marcador especial para AMBA
amba_data = df[df['AMBA'] == 'SI']
folium.Marker(
    [-34.6118, -58.3960],  # Buenos Aires
    popup=f"""
    <div style="width: 200px;">
        <h4 style="color: #A23B72;">ÁREA METROPOLITANA</h4>
        <hr>
        <b>Registros AMBA:</b> {len(amba_data):,}<br>
        <b>Total transportado:</b> {amba_data['CANTIDAD'].sum():,}<br>
        <b>Promedio diario:</b> {amba_data['CANTIDAD'].mean():.0f}<br>
        <b>Empresas:</b> {amba_data['NOMBRE_EMPRESA'].nunique()}<br>
        <b>Líneas:</b> {amba_data['LINEA'].nunique()}
    </div>
    """,
    tooltip="Área Metropolitana de Buenos Aires",
    icon=folium.Icon(color='purple', icon='bus', prefix='fa')
).add_to(mapa)

# Agregar leyenda
leyenda_html = '''
<div style="position: fixed;
            top: 10px; right: 10px; width: 180px; height: 120px;
            z-index:9999;
            font-size:14px; padding: 10px">
<p><i class="fa fa-circle" style="color:red"></i> Alto volumen (>50M)</p>
<p><i class="fa fa-circle" style="color:orange"></i> Volumen medio (10-50M)</p>
<p><i class="fa fa-circle" style="color:green"></i> Volumen bajo (<10M)</p>
<p><i class="fa fa-bus" style="color:purple"></i> AMBA</p>
</div>
'''
mapa.get_root().html.add_child(folium.Element(leyenda_html))

plt.figure(figsize=(12, 6))
evolucion_mensual = df.groupby(['AÑO', 'MES'])['CANTIDAD'].sum().reset_index()
for año in evolucion_mensual['AÑO'].unique():
    data_año = evolucion_mensual[evolucion_mensual['AÑO'] == año]
    plt.plot(data_año['MES'], data_año['CANTIDAD'], marker='o', label=f'{año}', linewidth=2)

plt.title('Evolución Mensual del Transporte por Año', fontsize=14, fontweight='bold')
plt.xlabel('Mes')
plt.ylabel('Cantidad Transportada')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(range(1, 13))
plt.tight_layout()
plt.show()

print("\n==== ANÁLISIS GEOGRÁFICO ====")
print("Top 10 provincias por cantidad transportada:")
print(provincia_stats[['total_cantidad', 'num_registros', 'num_empresas']].sort_values('total_cantidad', ascending=False).head(10))

print("\nComparación AMBA vs Interior:")
amba_comparison = df.groupby('AMBA')['CANTIDAD'].agg(['sum', 'mean', 'count'])
print(amba_comparison)

# Mostrar el mapa interactivo
print("\nMAPA INTERACTIVO DE TRANSPORTE EN ARGENTINA")
print("=" * 50)
display(mapa)


# Configurar estilo para visualizaciones
plt.style.use('default')
sns.set_palette("husl")

# Preparar variables adicionales para análisis
df['DIA_SEMANA_NUM'] = df['DIA_TRANSPORTE'].dt.dayofweek
df['SEMANA_AÑO'] = df['DIA_TRANSPORTE'].dt.isocalendar().week
df['TRIMESTRE'] = df['DIA_TRANSPORTE'].dt.quarter
df['ES_FINDE'] = (df['DIA_SEMANA_NUM'] >= 5).astype(int)


print("\n📊 ANÁLISIS TEMPORAL Y GEOGRÁFICO")
print("-" * 50)

# Crear figura con 3 subplots en una fila
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('ANÁLISIS COMPLETO: PATRONES TEMPORALES Y GEOGRÁFICOS', fontsize=16, fontweight='bold')

# GRÁFICO 1: Promedio por día de la semana
dow_data = df.groupby('DIA_SEMANA_NUM')['CANTIDAD'].mean()
dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
bars = axes[0].bar(range(7), dow_data.values, color=sns.color_palette("viridis", 7))
axes[0].set_title('Demanda Promedio por Día de Semana', fontsize=12, fontweight='bold')
axes[0].set_xticks(range(7))
axes[0].set_xticklabels(dias)
axes[0].set_ylabel('Cantidad Promedio')

# Añadir valores en las barras
for bar, val in zip(bars, dow_data.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.01,
                  f'{val:,.0f}', ha='center', va='bottom', fontsize=9)

# GRÁFICO 2: Fin de semana vs días laborables
finde_comparison = df.groupby(['ES_FINDE', 'AÑO'])['CANTIDAD'].mean().reset_index()
finde_pivot = finde_comparison.pivot(index='AÑO', columns='ES_FINDE', values='CANTIDAD')
finde_pivot.columns = ['Laborables', 'Fin de Semana']
finde_pivot.plot(kind='bar', ax=axes[1], color=['skyblue', 'orange'])
axes[1].set_title('Promedio Laborables vs Fin de Semana', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Cantidad Promedio')
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend()


# GRÁFICO 3: AMBA vs Interior
amba_evolution = df.groupby(['AÑO', 'MES', 'AMBA'])['CANTIDAD'].sum().reset_index()
amba_pivot = amba_evolution.pivot_table(index=['AÑO', 'MES'], columns='AMBA', values='CANTIDAD', fill_value=0)

if 'SI' in amba_pivot.columns and 'NO' in amba_pivot.columns:
    amba_monthly = amba_pivot.groupby(level=['AÑO', 'MES']).sum()
    amba_monthly.index = pd.to_datetime(amba_monthly.index.map(lambda x: f"{x[0]}-{x[1]:02d}-01"))

    axes[2].plot(amba_monthly.index, amba_monthly['SI'], label='AMBA', linewidth=3, color='red', marker='o', markersize=3)
    axes[2].plot(amba_monthly.index, amba_monthly['NO'], label='Interior', linewidth=3, color='blue', marker='s', markersize=3)

    axes[2].set_title('Evolución: AMBA vs Interior', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Cantidad Mensual')
    axes[2].set_xlabel('Período')
    axes[2].legend()
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3)

    # Añadir estadísticas
    total_amba = amba_monthly['SI'].sum()
    total_interior = amba_monthly['NO'].sum()
    pct_amba = (total_amba / (total_amba + total_interior)) * 100

    axes[2].text(0.02, 0.98, f'AMBA: {pct_amba:.1f}%',
                transform=axes[2].transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.1))
    axes[2].text(0.02, 0.88, f'Interior: {100-pct_amba:.1f}%',
                transform=axes[2].transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='blue', alpha=0.1))

else:
    axes[2].text(0.5, 0.5, 'No hay datos suficientes\npara AMBA vs Interior',
                transform=axes[2].transAxes, ha='center', va='center', fontsize=12)
    axes[2].set_title('AMBA vs Interior', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# ========================================
# ESTADÍSTICAS RESUMIDAS
# ========================================
print(f"\n📈 RESUMEN DE INSIGHTS:")
print("-" * 30)

# Día de mayor demanda
dia_mayor = dias[dow_data.idxmax()]
print(f"• Día de mayor demanda: {dia_mayor} ({dow_data.max():,.0f} promedio)")

# Diferencia fin de semana vs laborables
if len(finde_pivot.columns) == 2:
    dif_finde = ((finde_pivot['Fin de Semana'].mean() / finde_pivot['Laborables'].mean() - 1) * 100)
    print(f"• Fin de semana vs laborables: {dif_finde:+.1f}% diferencia")

# AMBA vs Interior
if 'SI' in amba_pivot.columns and 'NO' in amba_pivot.columns:
    print(f"• Concentración AMBA: {pct_amba:.1f}% del total nacional")
    print(f"• Ratio AMBA/Interior: {total_amba/total_interior:.2f}")

# ========================================
# INTENTO DE PREDICCIÓN MENSUAL Y JUSTIFICACIÓN
# ========================================
print("\n==== INTENTO DE PREDICCIÓN MENSUAL CON ML ====")

try:
    # Agrupar por mes
    df_pred_mensual = df.groupby([df['DIA_TRANSPORTE'].dt.to_period('M')])['CANTIDAD'].sum().reset_index()
    df_pred_mensual['ds'] = df_pred_mensual['DIA_TRANSPORTE'].astype(str) + '-01'
    df_pred_mensual['ds'] = pd.to_datetime(df_pred_mensual['ds'])
    df_pred_mensual = df_pred_mensual.rename(columns={'CANTIDAD': 'y'})
    # Entrenamos hasta 2023 para predecir 2024
    train = df_pred_mensual[df_pred_mensual['ds'] < '2024-01-01']
    test = df_pred_mensual[df_pred_mensual['ds'] >= '2024-01-01']
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(train[['ds', 'y']])
    future = m.make_future_dataframe(periods=len(test), freq='MS')
    forecast = m.predict(future)
    forecast_test = forecast[forecast['ds'].isin(test['ds'])]
    y_true = test['y'].values
    y_pred = forecast_test['yhat'].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"MAE mensual: {mae:,.0f} | RMSE mensual: {rmse:,.0f} | R2 mensual: {r2:.3f}")
    # Visualización
    plt.figure(figsize=(10,5))
    plt.plot(test['ds'], y_true, label='Real', marker='o')
    plt.plot(test['ds'], y_pred, label='Predicción', marker='x')
    plt.title('Predicción de demanda mensual (2024)')
    plt.legend()
    plt.show()
    # Residuos
    residuos = y_true - y_pred
    plt.figure(figsize=(8,4))
    plt.bar(test['ds'].dt.strftime('%Y-%m'), residuos)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuos de la predicción mensual')
    plt.ylabel('Error')
    plt.show()
    print("\nComentario: El R2 bajo y los residuos muestran que el modelo no logra capturar la variabilidad real mensual. Esto se debe a shocks externos, eventos no modelados y alta variabilidad no explicada por las variables disponibles. Por eso, no es posible hacer predicción confiable de demanda mensual solo con estos datos agregados.")
    # Imprimir valores para el README
    print("Valores para README:")
    print(f"MAE mensual: {mae:,.0f}")
    print(f"RMSE mensual: {rmse:,.0f}")
    print(f"R2 mensual: {r2:.3f}")
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
except Exception as e:
    print(f"No se pudo ajustar un modelo predictivo mensual: {e}")
