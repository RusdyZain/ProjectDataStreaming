# Business Intelligence: Netflix Streaming Data

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/) 
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-orange)](https://pandas.pydata.org/) 
[![Flask](https://img.shields.io/badge/Flask-Web_Framework-green)](https://flask.palletsprojects.com/)
[![Plotly](https://img.shields.io/badge/Plotly-Data_Visualization-blue)](https://plotly.com/)

## Deskripsi Proyek

Proyek **Business Intelligence: Netflix Streaming Data** adalah aplikasi web yang menampilkan statistik dari data streaming Netflix berdasarkan negara dan genre. Tujuan dari proyek ini adalah untuk menganalisis tren menonton film atau serial di Netflix berdasarkan genre dan negara, sehingga membantu pengguna memahami pola konsumsi konten global.

Proyek ini menggunakan dataset Netflix yang memuat informasi seperti judul, genre, negara, dan tanggal rilis, serta dilengkapi dengan visualisasi data menggunakan grafik yang interaktif.

## Fitur Utama

- **Analisis Berdasarkan Negara**: Menampilkan statistik jumlah konten yang ditonton per negara.
- **Analisis Berdasarkan Genre**: Menampilkan statistik konten berdasarkan genre populer di setiap negara.
- **Visualisasi Data Interaktif**: Penggunaan Plotly untuk membuat grafik interaktif yang memudahkan pemahaman data.
- **Data Filtering**: Pengguna dapat menyaring data berdasarkan negara, genre, dan periode waktu.
- **Tampilan Web User-friendly**: Dibangun menggunakan Flask sebagai framework backend dengan antarmuka yang responsif.

## Teknologi yang Digunakan

- **Python**: Bahasa pemrograman utama untuk analisis data dan pengembangan backend.
- **Pandas**: Digunakan untuk membaca, memproses, dan menganalisis data.
- **Flask**: Framework web minimalis yang digunakan untuk membuat API dan menampilkan data dalam bentuk web.
- **Plotly**: Library untuk membuat visualisasi data interaktif.
- **HTML/CSS/JavaScript**: Untuk mendesain dan membuat antarmuka pengguna yang menarik dan mudah digunakan.

## Instalasi

Ikuti langkah-langkah berikut untuk menjalankan proyek ini secara lokal:

1. **Clone repository ini**:
   ```bash
   git clone https://github.com/RusdyZain/ProjectDataStreaming.git
   cd ProjectDataStreaming
   ```

2. **Buat virtual environment (opsional tapi direkomendasikan)**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # untuk MacOS/Linux
   venv\Scripts\activate      # untuk Windows
   ```

3. **Instal dependensi**:
   Jalankan perintah berikut untuk menginstal semua dependensi yang diperlukan:
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan aplikasi**:
   Setelah semua dependensi diinstal, jalankan server Flask:
   ```bash
   flask run
   ```

   Buka browser dan kunjungi `http://127.0.0.1:5000` untuk melihat aplikasi.

## Struktur Proyek

Berikut adalah struktur direktori utama dari proyek ini:
```
.
├── app.py                # File utama untuk menjalankan aplikasi Flask
├── templates/            # Folder berisi template HTML untuk rendering halaman web
│   └── index.html        # Halaman utama aplikasi
├── static/               # Folder untuk menyimpan file CSS, JavaScript, dan aset statis lainnya
│   └── styles.css        # File CSS untuk styling halaman web
├── data/
│   └── netflix_data.csv   # Dataset Netflix yang digunakan dalam analisis
├── analysis/
│   └── data_analysis.py   # Script untuk proses analisis dan pemrosesan data
├── visualizations/
│   └── plotly_charts.py   # Script untuk membuat visualisasi menggunakan Plotly
└── requirements.txt       # File berisi daftar dependensi yang diperlukan
```

## Dataset

Dataset yang digunakan dalam proyek ini adalah dataset Netflix yang berisi informasi terkait film dan serial yang ditayangkan di platform. Data ini mencakup:
- **Title**: Nama film atau serial.
- **Genre**: Genre utama dari konten.
- **Country**: Negara asal atau tempat konten tersebut paling populer.
- **Release Date**: Tanggal rilis konten di Netflix.

Dataset ini diproses menggunakan Pandas untuk menyaring data yang relevan dan diolah lebih lanjut untuk ditampilkan dalam bentuk visual.

## Visualisasi

Proyek ini menggunakan **Plotly** untuk membuat visualisasi interaktif yang mencakup:
- **Bar Chart**: Untuk menampilkan jumlah konten berdasarkan genre per negara.
- **Pie Chart**: Untuk menampilkan distribusi genre di Netflix pada negara tertentu.
- **Line Chart**: Untuk menunjukkan tren konten Netflix yang rilis berdasarkan tahun dan genre.

Visualisasi ini membantu pengguna memahami tren global dalam konsumsi konten Netflix berdasarkan genre dan negara.

## Cara Penggunaan

1. **Analisis Berdasarkan Negara**: Pengguna dapat memilih negara dari daftar untuk melihat statistik konten yang ditonton di negara tersebut.
2. **Analisis Berdasarkan Genre**: Pengguna dapat memilih genre untuk melihat negara dengan konsumsi konten tertinggi pada genre tersebut.
3. **Interaksi dengan Grafik**: Pengguna dapat berinteraksi dengan grafik (zoom in/out, hover untuk detail) untuk eksplorasi lebih mendalam.

## Pengembangan Lanjutan

- [ ] Menambahkan fitur login dan autentikasi pengguna.
- [ ] Menambahkan lebih banyak filter untuk mempermudah eksplorasi data.
- [ ] Menghubungkan dengan API Netflix untuk mendapatkan data terbaru secara real-time.
- [ ] Menambahkan analisis sentimen berdasarkan ulasan pengguna di Netflix.

## Kontribusi

Jika Anda ingin berkontribusi dalam proyek ini, silakan fork repository ini, buat branch baru, dan ajukan pull request.

1. Fork repository ini.
2. Buat branch baru (`git checkout -b fitur-anda`).
3. Commit perubahan (`git commit -m 'Menambahkan fitur baru'`).
4. Push branch ke GitHub (`git push origin fitur-anda`).
5. Ajukan pull request.
