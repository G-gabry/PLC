| hyperparameter | method | metric | n_files | default_value | metric_at_default | best_value | metric_at_best | gain_over_default | median_time_ms_at_default | median_time_ms_at_best |
|---|---|---|---|---|---|---|---|---|---|---|
| ar_order | LPC | SNR (dB) | 200 | 256 | 15.45 | 1024 | 16.57 | 1.117 | 131.9 | 456.8 |
| ar_order | LPC | MR-STFT dist. | 200 | 256 | 0.4652 | 2048 | 0.3907 | 0.07458 | 131.9 | 1313 |
| ar_order | LPC | PLCMOS | 200 | 256 | 2.394 | 1536 | 2.506 | 0.1122 | 131.9 | 881.8 |
| ar_order | LPC | ViSQOL MOS-LQO | 200 | 256 | 4.507 | 1024 | 4.529 | 0.02212 | 131.9 | 456.8 |
| ar_order | NN | SNR (dB) | 200 | 256 | 15.21 | 768 | 15.41 | 0.1971 | 2964 | 3513 |
| ar_order | NN | MR-STFT dist. | 200 | 256 | 0.36 | 768 | 0.3578 | 0.002156 | 2964 | 3513 |
| ar_order | NN | PLCMOS | 200 | 256 | 2.463 | 768 | 2.52 | 0.05646 | 2964 | 3513 |
| ar_order | NN | ViSQOL MOS-LQO | 200 | 256 | 4.497 | 384 | 4.507 | 0.01007 | 2964 | 2998 |
| context_length_packets | LPC | SNR (dB) | 200 | 8 | 15.45 | 8 | 15.45 | 0 | 136.8 | 136.8 |
| context_length_packets | LPC | MR-STFT dist. | 200 | 8 | 0.4652 | 16 | 0.4609 | 0.004307 | 136.8 | 145.8 |
| context_length_packets | LPC | PLCMOS | 200 | 8 | 2.394 | 16 | 2.429 | 0.03536 | 136.8 | 145.8 |
| context_length_packets | LPC | ViSQOL MOS-LQO | 200 | 8 | 4.507 | 32 | 4.511 | 0.004223 | 136.8 | 175.1 |
| context_length_packets | NN | SNR (dB) | 200 | 8 | 15.21 | 8 | 15.21 | 0 | 2943 | 2943 |
| context_length_packets | NN | MR-STFT dist. | 200 | 8 | 0.36 | 8 | 0.36 | 0 | 2943 | 2943 |
| context_length_packets | NN | PLCMOS | 200 | 8 | 2.464 | 8 | 2.464 | 0 | 2943 | 2943 |
| context_length_packets | NN | ViSQOL MOS-LQO | 200 | 8 | 4.497 | 8 | 4.497 | 0 | 2943 | 2943 |
| diagonal_load | LPC | SNR (dB) | 200 | 0.001 | 15.45 | 0.001 | 15.45 | 0 | 127.4 | 127.4 |
| diagonal_load | LPC | MR-STFT dist. | 200 | 0.001 | 0.4652 | 1e-06 | 0.4525 | 0.01277 | 127.4 | 145.5 |
| diagonal_load | LPC | PLCMOS | 200 | 0.001 | 2.395 | 1e-05 | 2.409 | 0.01399 | 127.4 | 132.5 |
| diagonal_load | LPC | ViSQOL MOS-LQO | 200 | 0.001 | 4.507 | 0.0001 | 4.508 | 0.001247 | 127.4 | 128.7 |
| diagonal_load | NN | SNR (dB) | 200 | 0.001 | 15.21 | 0.0001 | 15.22 | 0.001466 | 3454 | 3257 |
| diagonal_load | NN | MR-STFT dist. | 200 | 0.001 | 0.36 | 1e-05 | 0.3594 | 0.0006027 | 3454 | 3334 |
| diagonal_load | NN | PLCMOS | 200 | 0.001 | 2.462 | 0.0001 | 2.467 | 0.005013 | 3454 | 3257 |
| diagonal_load | NN | ViSQOL MOS-LQO | 200 | 0.001 | 4.497 | 1e-05 | 4.498 | 0.001254 | 3454 | 3334 |
| extra_dim | LPC | SNR (dB) | 200 | 256 | 15.45 | 16 | 16.3 | 0.8539 | 130.2 | 143.5 |
| extra_dim | LPC | MR-STFT dist. | 200 | 256 | 0.4652 | 32 | 0.4152 | 0.05001 | 130.2 | 116.7 |
| extra_dim | LPC | PLCMOS | 200 | 256 | 2.408 | 384 | 2.414 | 0.006023 | 130.2 | 140.4 |
| extra_dim | LPC | ViSQOL MOS-LQO | 200 | 256 | 4.507 | 32 | 4.545 | 0.03782 | 130.2 | 116.7 |
| extra_dim | NN | SNR (dB) | 200 | 256 | 15.21 | 16 | 16.03 | 0.8185 | 3579 | 3777 |
| extra_dim | NN | MR-STFT dist. | 200 | 256 | 0.36 | 64 | 0.3393 | 0.02067 | 3579 | 3528 |
| extra_dim | NN | PLCMOS | 200 | 256 | 2.468 | 384 | 2.472 | 0.003994 | 3579 | 3468 |
| extra_dim | NN | ViSQOL MOS-LQO | 200 | 256 | 4.497 | 128 | 4.508 | 0.01103 | 3579 | 3632 |
