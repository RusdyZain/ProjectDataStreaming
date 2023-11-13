import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Business Intelligence", page_icon=":bar_chart:", layout="wide")

# Load data
data = "data-streaming.csv"
df = pd.read_csv(data)

# Set up Streamlit dashboard layout
st.title("Dashboard - Penonton Streaming")
st.sidebar.title("Filter")

# Sidebar - Pilih Genre
selected_genre = st.sidebar.selectbox("Pilih Genre", df['Genre'].unique())

# Sidebar - Pilih Negara untuk Forecasting (ARIMA)
selected_country_arima = st.sidebar.selectbox("Pilih Negara untuk Forecasting (ARIMA)", df['Country'].unique())

# Sidebar - Jumlah Negara yang Ingin Ditampilkan
num_countries_top = st.sidebar.number_input("Jumlah Negara Terbanyak", min_value=1, max_value=len(df['Country'].unique()), value=10)
num_countries_bottom = st.sidebar.number_input("Jumlah Negara Tersedikit", min_value=1, max_value=len(df['Country'].unique()), value=10)

# Sidebar - Pilihan Negara Terbanyak atau Terendah
selected_duration = st.sidebar.radio("Pilih Durasi", ["Terbanyak", "Terendah"])

# Function for ARIMA Forecasting
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

# Main Content
col1, col2 = st.columns(2)

# 1. Distribusi Penonton di Dunia (Top N Negara)
with col1:
    st.header(f"Penonton {selected_duration} di Dunia (Top {num_countries_top} Negara)")
    if selected_duration == "Terbanyak":
        top_countries = df['Country'].value_counts().head(num_countries_top).index
    else:
        top_countries = df['Country'].value_counts().tail(num_countries_top).index
    
    df_top_countries = df[df['Country'].isin(top_countries)]
    fig_pie = px.pie(df_top_countries, names='Country', title=f'Distribusi Penonton di Dunia ({selected_duration} {num_countries_top} Negara)')
    st.plotly_chart(fig_pie, use_container_width=True)

# 2. Grafik Penonton Terbesar per Periode (Top N Negara)
with col2:
    st.header(f"10 Grafik Penonton {selected_duration} per Periode (Top {num_countries_top} Negara)")
    total_duration_by_country = df.groupby('Country')['Duration_Watched (minutes)'].sum()
    if selected_duration == "Terbanyak":
        top_countries_bar = total_duration_by_country.nlargest(num_countries_top)
    else:
        top_countries_bar = total_duration_by_country.nsmallest(num_countries_top)
    
    fig_bar = px.bar(top_countries_bar, x=top_countries_bar.index, y='Duration_Watched (minutes)',
                     labels={'Duration_Watched (minutes)': 'Durasi Menonton (menit)', 'x': 'Negara'},
                     color=top_countries_bar.index)
    st.plotly_chart(fig_bar, use_container_width=True)

# 3. Durasi Menonton Genre Tersedikit di N Negara
with col1:
   st.header(f"{num_countries_bottom} Negara Penonton Tersedikit")
   df_genre_selected = df[df['Genre'] == selected_genre]
   total_duration_by_country_genre = df_genre_selected.groupby('Country')['Duration_Watched (minutes)'].sum()
   bottom_countries_bar_genre = total_duration_by_country_genre.nsmallest(num_countries_bottom)
   df_bottom_sorted = df_genre_selected[df_genre_selected['Country'].isin(bottom_countries_bar_genre.index)]
   df_bottom_sorted = df_bottom_sorted.sort_values(by='Duration_Watched (minutes)')
   fig_bar_genre = px.bar(df_bottom_sorted, x='Country', y='Duration_Watched (minutes)', color='Country',
                           title=f'Durasi Menonton Genre {selected_genre} di {num_countries_bottom} Negara Penonton Tersedikit',
                           labels={'Duration_Watched (minutes)': 'Durasi Menonton (menit)', 'x': 'Negara'},
                           category_orders={"Country": df_bottom_sorted.groupby('Country')['Duration_Watched (minutes)'].sum().sort_values().index})
   st.plotly_chart(fig_bar_genre, use_container_width=True)

with col2:
   # Sidebar
   st.sidebar.title("Filter")
   selected_country = st.sidebar.multiselect("Pilih Negara", df['Country'].unique())

   # Filter data based on selected country
   filtered_df = df[df['Country'].isin(selected_country)] if selected_country else df

   # Hierarchy Diagram
   if selected_country:
      st.header("Hierarchical Diagram - Penonton Streaming")
      fig = px.treemap(filtered_df, path=["Country", "Genre", "Device_Type", "Location", "Subscription_Status", "Playback_Quality"])
      fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
      st.plotly_chart(fig, use_container_width=True)


# Tabel Data
st.header("Tabel Data")
st.dataframe(filtered_df)

# Forecasting menggunakan ARIMA (statsmodels)
st.header(f"Forecasting Durasi Menonton untuk {selected_country_arima} (ARIMA)")
arima_result = arima_forecast(df, selected_country_arima)
fig_arima = px.line(x=range(1, len(arima_result) + 1), y=arima_result.values,
                    labels={'y': 'Durasi Menonton (menit)', 'x': 'Periode'}, title='Forecast ARIMA')
st.plotly_chart(fig_arima, use_container_width=True)

# Implementasi dashboard EDA yang telah diubah
# ...

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