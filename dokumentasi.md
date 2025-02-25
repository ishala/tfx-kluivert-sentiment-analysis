# Submission 1: Analisis Sentimen Pada Masuknya Patrick Kluivert Sebagai Pelatih Timnas Indonesia

Nama: Muhammad Faishal Ali Dhiaulhaq

Username dicoding: faishal_ali

| | Deskripsi |
| ----------- | ----------- |
| Dataset | Hasil Scrapping pada data komentar di Youtube dengan topik "Masuknya Patrick Kluivert Sebagai Pelatih Timnas Indonesia". |
| Masalah | Masuknya Patrick Kluivert sebagai pelatih Timnas Indonesia menggantikan Shin Tae Yong sebagai pelatih lama menjadi polemik yang cukup viral. Hal ini tentunya menuai komentar netizen mengenai tindakan PSSI ini. |
| Solusi machine learning | Masalah ini menarik untuk dijadikan *case* dalam bidang analisis sentimen. Dengan pemisahan polaritas seperti positif, netral, dan negatif. Kita dapat mengetahui bagaimana netizen berkomentar pada fenomena ini. |
| Metode pengolahan | Metode-metode pengolahan data yang dilakukan meliput *Data Cleaning*, *Data Preprocessing* (*Case Folding*, *Replace Slang Words*, *Stopword Removal*, dan lain-lain), EDA, *Modelling*, hingga *Evaluation*.  |
| Arsitektur model | Arsitektur model yang digunakan yaitu *Neural Network* dengan *Bidirectional* LSTM pada Tensorflow. Dengan layer seperti embedding, Dense, serta Dropout yang dibungkus di dalam Sequentials |
| Metrik evaluasi | Metrik evaluasi yang digunakan yaitu seperti *train* dan *validation accuracy*, serta *train* dan *validation loss*. Selain itu, juga menerapkan metrik lain seperti *precision*, *recall*, AUC, serta *categorical accuracy* |
| Performa model | Model LSTM dapat mencapai akurasi *training* hingga 100%, namun dengan akurasi *validation* berkisar di angka 79%. Yang menandakan terjadinya overfitting pada data training. Untuk *precision*, *recall*, serta AUC berkisar di angka 68%. Serta untuk *categorical accuracy* masih berkisar di angka 57%. Untuk selanjutnya, dapat mencoba metode lain atau menambahkan dataset agar tidak terjadi *data imbalance*. |
