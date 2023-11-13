import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA

# Load data
data = "data-streaming.csv"
df = pd.read_csv(data)

# Set up Streamlit dashboard layout
st.title("Dashboard Business Intelligence - Penonton Streaming")
st.sidebar.title("Filter")
selected_genre = st.sidebar.selectbox("Pilih Genre", df['Genre'].unique())

# 1. Distribusi Penonton di Dunia (Top 10 Negara)
st.header("Penonton Terbanyak di Dunia")
top_countries = df['Country'].value_counts().head(10).index
df_top_countries = df[df['Country'].isin(top_countries)]
fig_pie = px.pie(df_top_countries, names='Country', title='Distribusi Penonton di Dunia (10 Negara Teratas)')
st.plotly_chart(fig_pie)

# 2. Grafik Penonton Terbesar per Periode (Top 10 Negara)
st.header("10 Grafik Penonton Terbesar per Periode")
total_duration_by_country = df.groupby('Country')['Duration_Watched (minutes)'].sum()
top_10_countries_bar = total_duration_by_country.nlargest(10)
fig_bar = px.bar(top_10_countries_bar, x=top_10_countries_bar.index, y='Duration_Watched (minutes)',
                 labels={'Duration_Watched (minutes)': 'Durasi Menonton (menit)', 'x': 'Negara'},
                 color=top_10_countries_bar.index)
st.plotly_chart(fig_bar)

# 3. Durasi Menonton Genre Tersedikit di 10 Negara
st.header("10 Negara Penonton Tersedikit")
df_genre_selected = df[df['Genre'] == selected_genre]
total_duration_by_country_genre = df_genre_selected.groupby('Country')['Duration_Watched (minutes)'].sum()
bottom_10_countries_bar_genre = total_duration_by_country_genre.nsmallest(10)
df_bottom_10_sorted = df_genre_selected[df_genre_selected['Country'].isin(bottom_10_countries_bar_genre.index)]
df_bottom_10_sorted = df_bottom_10_sorted.sort_values(by='Duration_Watched (minutes)')
fig_bar_genre = px.bar(df_bottom_10_sorted, x='Country', y='Duration_Watched (minutes)', color='Country',
                        title=f'Durasi Menonton Genre {selected_genre} di 10 Negara Penonton Tersedikit',
                        labels={'Duration_Watched (minutes)': 'Durasi Menonton (menit)', 'x': 'Negara'},
                        category_orders={"Country": df_bottom_10_sorted.groupby('Country')['Duration_Watched (minutes)'].sum().sort_values().index})
st.plotly_chart(fig_bar_genre)

# Forecasting menggunakan ARIMA (statsmodels)
def arima_forecast(data, country):
    df_country = data[data['Country'] == country]
    df_country['Index'] = range(1, len(df_country) + 1)
    df_country.set_index('Index', inplace=True)
    
    # Membuat model ARIMA
    model = ARIMA(df_country['Duration_Watched (minutes)'], order=(5, 1, 0))
    results = model.fit()

    # Melakukan forecasting
    forecast_steps = 12 
    forecast = results.get_forecast(steps=forecast_steps)
    return forecast.predicted_mean

# Memilih negara untuk forecasting
selected_country_arima = st.sidebar.selectbox("Pilih Negara untuk Forecasting (ARIMA)", df['Country'].unique())
st.header(f"Forecasting Durasi Menonton untuk {selected_country_arima} (ARIMA)")

# Mendapatkan hasil forecasting menggunakan ARIMA
arima_result = arima_forecast(df, selected_country_arima)

# Menampilkan grafik hasil forecasting ARIMA
fig_arima = px.line(x=range(1, len(arima_result) + 1), y=arima_result.values,
                    labels={'y': 'Durasi Menonton (menit)', 'x': 'Periode'}, title='Forecast ARIMA')
st.plotly_chart(fig_arima)

# Kesimpulan
st.header("Kesimpulan Business Intelligence")

# Menjelaskan kesimpulan dari model ARIMA
st.markdown("""
Model ARIMA digunakan untuk melakukan forecasting durasi menonton untuk suatu negara yang dipilih. 
Beberapa kesimpulan yang dapat diambil dari model ARIMA ini meliputi:

1. **Trend Forecasting:** Model ARIMA memberikan perkiraan tren durasi menonton di masa depan berdasarkan data historis.

2. **Optimasi Konten:** Informasi dari model ARIMA dapat membantu platform streaming mengoptimalkan penawaran kontennya 
   berdasarkan genre atau jenis konten yang diprediksi diminati oleh penonton.

3. **Pengambilan Keputusan Strategis:** Hasil prediksi ARIMA dapat digunakan dalam pengambilan keputusan strategis terkait 
   alokasi sumber daya, perencanaan promosi, dan investasi dalam pengembangan konten baru.

4. **Analisis Performa Negara:** Model ARIMA memberikan wawasan khusus untuk setiap negara, membantu platform 
   streaming memahami bagaimana preferensi penonton berubah dari waktu ke waktu dan di berbagai lokasi.

5. **Evaluasi Efektivitas Strategi Bisnis:** Dengan membandingkan hasil prediksi ARIMA dengan data aktual, bisnis dapat 
   mengevaluasi efektivitas strategi bisnis mereka dan melakukan penyesuaian jika diperlukan.

Penting untuk dicatat bahwa keakuratan model ARIMA sangat tergantung pada karakteristik data dan parameter yang dipilih.
Perlu dilakukan evaluasi reguler terhadap performa model dan mungkin pertimbangan untuk pengembangan model yang lebih canggih 
jika diperlukan.
""")
