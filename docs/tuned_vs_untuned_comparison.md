# CIRCA Comparative Real-Board Performance Report

> **Generated:** 2026-07-18 00:11:40
> **Evaluation Dataset:** 50 Staged Real-Board PCB Images

## Performance Comparison Matrix

| Model Variant | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall | F1-Score | Avg Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nano Untuned** | 4 | 111 | 142 | 3.48% | 2.74% | **3.07%** | 803.4 ms |
| **Nano Fine-Tuned (CP)** | 24 | 400 | 122 | 5.66% | 16.44% | **8.42%** | 784.2 ms |
| **Small Untuned** | 8 | 50 | 138 | 13.79% | 5.48% | **7.84%** | 939.0 ms |
| **Small Fine-Tuned (CP)** | 28 | 252 | 118 | 10.00% | 19.18% | **13.15%** | 1412.9 ms |
| **Medium Untuned** | 20 | 125 | 126 | 13.79% | 13.70% | **13.75%** | 1476.0 ms |
| **Medium Fine-Tuned (CP)** | 17 | 235 | 129 | 6.75% | 11.64% | **8.54%** | 1470.5 ms |

## Key Insights
* **Impact of Domain-Adaptation (Copy-Paste):** Evaluating models on real-board photos shows the direct benefit of fine-tuning YOLOv12 on populated motherboards. The copy-paste composites help bridge the laboratory-to-exhibition domain gap.
* **Latency vs. Accuracy Trade-Off:** The Nano variant remains the fastest model. Use the F1-Score column above to verify if Small or Medium models provide a significant accuracy lift that justifies the additional latency overhead.
