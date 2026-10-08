# 🛡️ Eksekusi Hari 1: Pendefinisian Masalah Bisnis & Data Cleaning Pipeline

> **Status**: ✅ **SELESAI (COMPLETED)**  
> **Fokus**: Scaffolding Proyek, Business Problem Framing, Data Hygiene, & Reusable Pipeline  
> **Artefak Utama**:  
> - Modul Pipeline: [`src/data_cleaning.py`](file:///d:/linkedin/gimme-some-creds/src/data_cleaning.py)  
> - Notebook Portofolio: [`notebooks/01_data_cleaning.ipynb`](file:///d:/linkedin/gimme-some-creds/notebooks/01_data_cleaning.ipynb) (Tereksekusi penuh dengan visual & output)  
> - Dataset Bersih: [`data/processed/train_cleaned.csv`](file:///d:/linkedin/gimme-some-creds/data/processed/train_cleaned.csv) & [`data/processed/test_cleaned.csv`](file:///d:/linkedin/gimme-some-creds/data/processed/test_cleaned.csv)  
> - Dependensi: [`requirements.txt`](file:///d:/linkedin/gimme-some-creds/requirements.txt)

---

## 1. Ringkasan Eksekutif Masalah Bisnis (_Business Framing_)

Dalam industri perbankan dan *fintech lending*, risiko kredit memiliki struktur kerugian yang asimetris:
* **Margin Keuntungan Bunga**: 8% – 15% per tahun dari pinjaman lancar.
* **Kerugian Gagal Bayar**: Hingga 100% modal pokok pinjaman ditambah biaya operasional penagihan.
* **Implikasi Finansial**: Satu debitur yang macet dapat menghapus laba yang dihasilkan oleh **8 hingga 12 debitur lancar**.

### Asymmetric Loss Matrix:
1. **False Negative (Tipe II - Bahaya Utama)**:
   * Model memprediksi nasabah *Lancar (0)*, tetapi nasabah mengalami *Default (1)*.
   * Dampak: Kerugian modal langsung (*Direct Credit Loss*), beban pencadangan kerugian (CKPN/NPL), dan biaya *debt collector*.
2. **False Positive (Tipe I - Kerugian Potensial)**:
   * Model memprediksi nasabah *Macet (1)*, padahal sebenarnya nasabah *Lancar (0)*.
   * Dampak: Kehilangan potensi margin bunga (*Opportunity Loss*) dan nasabah berpindah ke kompetitor.

Oleh karena itu, target optimasi model difokuskan pada **Recall & ROC-AUC** untuk memaksimalkan penangkapan debitur berisiko pada data yang timpang (hanya 6.68% default).

---

## 2. Struktur Direktori Proyek yang Telah Dibangun

```
gimme-some-creds/
├── datasets/                              <-- Data mentah asli (tidak disentuh/read-only)
│   ├── Data Dictionary.xls
│   ├── cs-training.csv
│   ├── cs-test.csv
│   └── sampleEntry.csv
├── data/
│   └── processed/                         <-- Data hasil pembersihan siap modeling
│       ├── train_cleaned.csv              <-- 150,000 baris x 16 kolom (0 missing)
│       └── test_cleaned.csv               <-- 101,503 baris x 16 kolom (0 missing)
├── notebooks/
│   └── 01_data_cleaning.ipynb             <-- Notebook naratif lengkap dengan visualisasi output
├── src/
│   └── data_cleaning.py                   <-- Kelas CreditDataCleaner (Modular & Anti-Data-Leakage)
├── markdown/
│   ├── report.md                          <-- Master plan & audit dataset
│   └── day-1.md                           <-- Laporan eksekusi Hari 1 ini
└── requirements.txt                       <-- Daftar dependensi library
```

---

## 3. Tindakan Pembersihan Data & Transformasi Fitur

Transformasi dijalankan menggunakan kelas `CreditDataCleaner` dengan prinsip **anti-data leakage** (statistik seperti median dan persentil dipelajari murni dari data latih, lalu diaplikasikan ke data uji):

| Fitur / Masalah | Nilai Sebelum Pembersihan | Tindakan Perbaikan | Fitur Baru yang Direkayasa (_Engineered Flags_) |
| :--- | :--- | :--- | :--- |
| **`age = 0`** | 1 nasabah berumur 0 tahun (data entry error). | Diganti NaN, diimputasi median umur latih (52 tahun). | — |
| **Kode Biro 96 & 98** | 269 baris bernilai 96/98 pada 3 kolom keterlambatan. | Dipetakan ke flag indikator error, nilai dikembalikan ke 0 (modus keterlambatan). | `HasPastDueRecordError` (1 jika terdapat kode 96/98, 0 jika normal). |
| **`RevolvingUtilization`** | Nilai ekstrem hingga 50,708 (3,321 baris > 1.0). | Diberikan batas atas (*winsorized*) pada persentil ke-99. | `RevolvingUtilizationOverLimit` (1 jika utilisasi > 100%, 0 jika normal). |
| **`MonthlyIncome`** | 29,731 baris kosong (19.82% missing). | Diimputasi berdasarkan median pendapatan per kelompok umur (*cohort median*). | `MonthlyIncome_is_missing` (1 jika pendapatan tidak diisi, 0 jika diisi). |
| **`DebtRatio`** | Nilai ekstrem hingga 329,664 (akibat pendapatan 0). | Diberikan batas atas (*capped*) pada persentil ke-99. | `DebtRatio_is_extreme` (1 jika rasio di atas p99, 0 jika normal). |
| **`NumberOfDependents`** | 3,924 baris kosong (2.62% missing). | Diimputasi menggunakan nilai 0 (median & modus keluarga). | `NumberOfDependents_is_missing` (1 jika data tanggungan hilang, 0 jika diisi). |

---

## 4. Hasil Verifikasi Akhir Kualitas Data

* **Dimensi Data Latih Bersih**: 150.000 baris × 16 kolom.
* **Dimensi Data Uji Bersih**: 101.503 baris × 16 kolom.
* **Jumlah Missing Values**: **0 di seluruh kolom**.
* **Engineered Signal Preservation**: Penanda anomali tidak dibuang begitu saja, melainkan diawetkan dalam 5 kolom indikator biner untuk memberikan sinyal prediktif berharga bagi model Machine Learning di Hari 3.

---

## 5. Rencana Transisi ke Hari 2

Dengan tersedianya dataset bersih di `data/processed/train_cleaned.csv`, kita siap melangkah ke **Hari 2**:
* **Exploratory Data Analysis (EDA) Multivariat**: Visualisasi korelasi dan distribusi fitur.
* **Risk Cohort Analysis**: Analisis tingkat default per kelompok umur, kuartil pendapatan, rasio utilisasi kredit, dan perbandingan kredit beragun (KPR) vs tanpa agunan.
* **Ekstraksi 3–5 Insight Bisnis Strategis**: Merumuskan temuan data ke dalam bahasa eksekutif perbankan untuk materi portofolio LinkedIn & GitHub.
