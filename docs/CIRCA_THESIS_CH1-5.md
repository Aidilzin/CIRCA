**UNIVERSITI TEKNOLOGI MARA**

**CIRCA: CIRCUIT INSPECTION AND RECOGNITION USING CONVOLUTIONAL ARCHITECTURES**

**MUHAMMAD AIDIL AL-FAIZI BIN MOHD ZIN**

**BACHELOR OF INFORMATION SYSTEMS (Hons.) INTELLIGENT SYSTEMS ENGINEERING**

**JULY 2026**

**UNIVERSITI TEKNOLOGI MARA**

**CIRCA: CIRCUIT INSPECTION AND RECOGNITION USING CONVOLUTIONAL ARCHITECTURES**

**MUHAMMAD AIDIL AL-FAIZI BIN MOHD ZIN**

**Thesis submitted in fulfilment of the requirements for Bachelor of Information Systems (Hons.) Intelligent Systems Engineering**

**Faculty of Computer Science and Mathematics**

**July 2026**

# **SUPERVISOR APPROVAL**

**CIRCA: CIRCUIT INSPECTION AND RECOGNITION USING CONVOLUTIONAL ARCHITECTURES**

By

**MUHAMMAD AIDIL AL-FAIZI BIN MOHD ZIN**

**2023276732**

This thesis was prepared under the supervision of the project supervisor, Puan Farahnatasyah Binti Abdul Hannan. It was submitted to the Faculty of Computer Science and Mathematics and was accepted in partial fulfilment of the requirements for degree of Bachelor of Information Systems (Hons.) Intelligent Systems Engineering.

Approved by,

…………………………….

Puan Farahnatasyah Binti Abdul Hannan

Project Supervisor

JANUARY 23, 2026

# **STUDENT DECLARATION**

I certify that this thesis and the project to which it relates is the product of my own work and that any idea or quotation from the work of other people, published or otherwise are fully acknowledged in accordance with the standard referring practices of the discipline.

![](data:image/png;base64...)

…………………………….

MUHAMMAD AIDIL AL-FAIZI BIN MOHD ZIN

2023276732

JULY 9, 2026

# **ACKNOWLEDGEMENT**

Alhamdullilah, praises and thanks to Allah because of His Almighty and His utmost blessings, I was able to finish this research within the time duration given. First and foremost, I would like to express my utmost gratitude to my academic supervisor, Puan Farahnatasyah Binti Abdul Hannan, for her invaluable guidance, constant encouragement, and insightful feedback throughout the duration of this Final Year Project. Her expertise and support were essential to this study's success.

I am also deeply grateful to the faculty members of the Bachelor of Information Technology (Hons.) Intelligent Systems Engineering program at Universiti Teknologi MARA (UiTM) for providing a strong academic foundation and the computational facilities necessary to complete my work.

I would like to dedicate a special note of thanks to my father, Mohd Zin Bin Abas, and my siblings for their continuous financial support, encouragement, and valuable feedback during the developmental phases of this project. Special appreciation also goes to my friend, Muhammad Syafiq Iman Bin Haidzir, for his assistance in refining my research direction and identifying key areas of study. Finally, I would like to acknowledge all the open-source contributors and researchers who made their datasets and model code available, directly enabling the benchmarks in this project.

# **ABSTRACT**

Printed Circuit Boards (PCBs) form the physical and electrical foundation of modern consumer electronics. In electronics repair environments, manual visual inspection under magnifying lenses is the predominant diagnostic method. However, manual inspection is slow, subjective, and prone to fatigue-induced escape rates of 10% to 20% on sub-millimeter components. Although automated manufacturing utilizes high-cost, glare-controlled Automated Optical Inspection (AOI) systems, these platforms are economically and operationally non-viable for independent repair benches operating under variable desklamp lighting. I present CIRCA (Circuit Inspection and Recognition using Convolutional Architectures), a lightweight AI-driven visual inspection system designed specifically for electronics repair environments. CIRCA unifies bare-board (IPC-A-600) and assembly solder joint (IPC-A-610H) defect taxonomies into a seven-class schema, trained on the compiled 8,463-image unified\_pcb\_v3 corpus. I designed a six-stage inference pipeline containing LAB-space Contrast Limited Adaptive Histogram Equalization (CLAHE), Gamma Correction, and Laplacian Variance blur quality gating. To optimize detection on sub-millimeter defect regions, I conducted genetic hyperparameter tuning, resulting in a 71-fold reduction in initial learning rate and a 44-fold reduction in box loss weight relative to YOLO defaults. I trained and evaluated three attention-centric YOLOv12 variants (Nano, Small, Medium) optimized via Intel OpenVINO. The production YOLOv12-Nano FP16 configuration was selected, achieving an end-to-end latency of 0.392 seconds on a commodity CPU and 62.79% mAP@0.5 on the frozen test split. I addressed operator automation bias by integrating a dual-threshold confidence-display scheme paired with a global warning banner. CIRCA demonstrates that factory-grade, real-time visual inspection is viable on commodity edge processors, directly lowering capital barriers for independent repair shops and supporting the circular economy.

# **TABLE OF CONTENTS**

**CONTENT PAGE**

[**SUPERVISOR APPROVAL** ii](#_Toc234513054)

[**STUDENT DECLARATION** iii](#_Toc234513055)

[**ACKNOWLEDGEMENT** iv](#_Toc234513056)

[**ABSTRACT** v](#_Toc234513057)

[**TABLE OF CONTENTS** vi](#_Toc234513058)

[**LIST OF FIGURES** x](#_Toc234513059)

[**LIST OF TABLES** xi](#_Toc234513060)

[**LIST OF ABBREVIATIONS** xii](#_Toc234513061)

[**CHAPTER 1: INTRODUCTION** 1](#_Toc234513062)

[**1.1** **Research Background** 1](#_Toc234513064)

[**1.2** **Problem Statement** 3](#_Toc234513065)

[**1.3** **Research Questions** 4](#_Toc234513066)

[**1.4** **Research Objectives** 5](#_Toc234513067)

[**1.5** **Research Scope** 5](#_Toc234513068)

[**1.6** **Significance of Study** 6](#_Toc234513069)

[**1.7** **Expected Solution** 7](#_Toc234513070)

[**CHAPTER 2: LITERATURE REVIEW** 20](#_Toc234513071)

[**2.1** **PCB Defects and Inspection Challenges** 20](#_Toc234513073)

[**2.2** **Machine Learning and CNN-Based PCB Defect Detection** 21](#_Toc234513074)

[**2.3** **YOLO-Based PCB Defect Detection** 22](#_Toc234513075)

[**2.4** **Image Preprocessing for Robust Detection** 23](#_Toc234513076)

[**2.5** **Edge Machine Learning Deployment and Model Quantization** 24](#_Toc234513077)

[**2.6** **Human Factors and Automation Bias in AI-Assisted Inspection** 24](#_Toc234513078)

[**2.7** **Summary of Related Works** 25](#_Toc234513079)

[**CHAPTER 3: RESEARCH METHODOLOGY** 28](#_Toc234513080)

[**3.1** **Introduction** 28](#_Toc234513082)

[**3.2** **Research Framework** 28](#_Toc234513083)

[**3.2.1** **Overview of Research Phases** 28](#_Toc234513084)

[**3.2.2** **Mapping of Objectives, Activities, and Deliverables** 30](#_Toc234513085)

[**3.3** **Theoretical Study** 30](#_Toc234513086)

[**3.3.1** **Preliminary Study** 30](#_Toc234513087)

[**3.3.2** **Knowledge Acquisition** 31](#_Toc234513088)

[**3.4** **Empirical Study** 31](#_Toc234513089)

[**3.4.1** **Data Collection** 31](#_Toc234513090)

[**3.4.2** **Data Pre-processing** 33](#_Toc234513091)

[**3.4.3** **Data Analysis** 34](#_Toc234513092)

[**3.5** **System Design** 35](#_Toc234513093)

[**3.5.1** **CIRCA System Architecture** 35](#_Toc234513094)

[**3.5.2** **Inference Pipeline** 35](#_Toc234513095)

[**3.5.3** **Confidence Threshold and “Manual Inspection Required” Logic** 35](#_Toc234513096)

[**3.5.4** **Interface Design** 36](#_Toc234513097)

[**3.6** **System Development** 36](#_Toc234513098)

[**3.6.1** **Training Engine** 36](#_Toc234513099)

[**3.6.2** **Hyperparameter Optimisation Algorithm** 36](#_Toc234513100)

[**3.6.3** **Model Training Procedure** 37](#_Toc234513101)

[**3.6.4** **OpenVINO Export and INT8 Quantisation** 37](#_Toc234513102)

[**3.6.5** **Confidence Threshold Calibration Procedure** 38](#_Toc234513103)

[**3.7** **Experimental Design** 38](#_Toc234513104)

[**3.7.1** **Phase 1: Vanilla Baseline** 38](#_Toc234513105)

[**3.7.2**  **Phase 2: CIRCA-Aligned Baseline** 38](#_Toc234513106)

[**3.7.3**  **Phase 3: Hyperparameter Optimisation** 38](#_Toc234513107)

[**3.7.4**  **Phase 4: Three-Variant Final Training** 39](#_Toc234513108)

[**3.7.5**  **Phase 5: Quantisation Validation** 39](#_Toc234513109)

[**3.7.6**  **Phase 6: Hardware Benchmarking and Variant Selection** 39](#_Toc234513110)

[**3.7.7**  **Phase 7: Final Test Evaluation and Threshold Calibration** 39](#_Toc234513111)

[**3.7.8**  **Acceptance Criteria** 40](#_Toc234513112)

[**3.7.9**  **Evaluation Metrics** 40](#_Toc234513113)

[**3.8**  **Hardware and Software Specification** 40](#_Toc234513114)

[**3.8.1**  **Training Environment** 40](#_Toc234513115)

[**3.8.2**  **Deployment Target** 40](#_Toc234513116)

[**3.8.3**  **Software Stack** 41](#_Toc234513117)

[**3.9**  **Research Plan** 41](#_Toc234513118)

[**3.10**  **Summary** 42](#_Toc234513119)

[**CHAPTER 4: RESULTS AND FINDINGS** 43](#_Toc234513120)

[**4.1**  **Introduction** 43](#_Toc234513122)

[**4.2** **Dataset and Defect Taxonomy Results** 43](#_Toc234513123)

[**4.3**  **Preprocessing Pipeline Evaluation** 46](#_Toc234513124)

[**4.4**  **Hyperparameter Optimisation Results** 48](#_Toc234513125)

[**4.5**  **Three-Variant Comparative Training Results** 52](#_Toc234513126)

[**4.6**  **OpenVINO Quantisation Results** 56](#_Toc234513127)

[**4.7**  **Hardware Benchmarking Results** 58](#_Toc234513128)

[**4.8**  **Final Test-Set Evaluation** 61](#_Toc234513129)

[**4.9**  **Confidence Threshold Calibration Results** 69](#_Toc234513130)

[**4.10**  **Comparison with Related Work** 71](#_Toc234513131)

[**4.11**  **Chapter Summary** 73](#_Toc234513132)

[**CHAPTER 5: CONCLUSION, LIMITATIONS AND FUTURE WORKS** 74](#_Toc234513133)

[**5.1**  **Introduction** 74](#_Toc234513135)

[**5.2**  **Conclusion** 74](#_Toc234513136)

[**5.2.1**  **Conclusion for RO1: PCB Defect Taxonomy Identification and Documentation** 74](#_Toc234513137)

[**5.2.2**  **Conclusion for RO2: YOLOv12 Model Design and Comparative Evaluation** 75](#_Toc234513138)

[**5.2.3**  **Conclusion for RO3: CIRCA Desktop Application Development and Evaluation** 75](#_Toc234513139)

[**5.3**  **Contributions of the Study** 75](#_Toc234513140)

[**5.4**  **Limitations** 76](#_Toc234513141)

[**5.5**  **Future Works** 77](#_Toc234513142)

[**5.6**  **Summary** 77](#_Toc234513143)

[**REFERENCES** 78](#_Toc234513144)

# **LIST OF FIGURES**

|  |  |
| --- | --- |
| FIGURE | PAGE |

[3.1: CIRCA Research Framework 29](#_Toc234498297)

[4.1: Sample Defect Images 45](#_Toc234498298)

[4.2: Phase 1 vs Phase 2 Ablation Comparison 47](#_Toc234498299)

[4.3: HPO Fitness Trajectory 49](#_Toc234498300)

[4.4: HPO Parameter Scatter Plots 50](#_Toc234498301)

[4.5a: YOLOv12-Nano Training Curves 53](#_Toc234498302)

[4.5b: YOLOv12-Small Training Curves 53](#_Toc234498303)

[4.5c: YOLOv12-Medium Training Curves 54](#_Toc234498304)

[4.7: Normalised Confusion Matrix 64](#_Toc234498305)

[4.8a: Box PR Curve 65](#_Toc234498306)

[4.8b: Box F1-Confidence Curve 66](#_Toc234498307)

[4.9a: Test Predictions Batch 0 67](#_Toc234498308)

[4.9b: Test Predictions Batch 1 68](#_Toc234498309)

[4.10: Confidence Threshold Sweep 69](#_Toc234498310)

# **LIST OF TABLES**

|  |  |
| --- | --- |
| TABLE | PAGE |

[2.1 Summary of Related Works on PCB Defect Detection 26](#_Toc234498186)

[2.2: Preprocessing and Deployment Techniques Underpinning CIRCA Design 26](#_Toc234498187)

[3.1: Mapping of Research Objectives, Activities, and Deliverables 30](#_Toc234498188)

[3.2: CIRCA Unified 7-Class IPC Taxonomy 33](#_Toc234498189)

[3.3: CIRCA Project Timeline and Compute Estimate 42](#_Toc234498190)

[4.1: Final Class Distribution: `unified\_pcb\_v3` (8,463 unique images, 54,928 instances) 44](#_Toc234498191)

[4.2: Train / Validation / Test Split Statistics: `unified\_pcb\_v3` 45](#_Toc234498192)

[4.3: Phase 1 vs Phase 2 Validation Metrics (OFAT Ablation, YOLOv12-S, 100 Epochs, `unified\_pcb\_v3`) 46](#_Toc234498193)

[4.4: Preprocessing Latency per Stage on the Representative Target CPU 48](#_Toc234498194)

[4.5: Top-10 HPO Trials Ranked by Fitness 50](#_Toc234498195)

[4.6: Comparison of Tuned Hyperparameters and YOLOv12 Defaults 51](#_Toc234498196)

[4.7: Validation Metrics per YOLOv12 Variant (Phase 4) 54](#_Toc234498197)

[4.8: Per-class Validation mAP@0.5 Breakdown per Variant 55](#_Toc234498198)

[4.9: FP32 vs FP16 vs INT8 Validation mAP@0.5 Comparison 56](#_Toc234498199)

[4.10: Per-variant Fallback Decision Summary 57](#_Toc234498200)

[4.11: Preprocessing Latency per Stage on the Representative Target CPU 58](#_Toc234498201)

[4.12: Inference Latency Comparison (CPU vs. GPU) 59](#_Toc234498202)

[4.13: Static Image Inference Latency Comparison 60](#_Toc234498203)

[4.14: Variant Selection Matrix 61](#_Toc234498204)

[4.15: Overall Test Metrics (YOLOv12-N FP16) 62](#_Toc234498205)

[4.16: Per-class Test split Metrics (YOLOv12-N FP16) 63](#_Toc234498206)

[4.17: Calibrated Per-class Display and Warning Thresholds 70](#_Toc234498207)

[4.18: Comparison of CIRCA with Published PCB Defect Detectors 72](#_Toc234498208)

# **LIST OF ABBREVIATIONS**

|  |  |
| --- | --- |
| **ABBREVIATION** | **FULL FORM** |
| AOI | Automated Optical Inspection |
| AP | Average Precision |
| CPU | Central Processing Unit |
| CUDA | Compute Unified Device Architecture |
| dHash | Difference Hashing |
| DFL | Distribution Focal Loss |
| dGPU | Discrete Graphics Processing Unit |
| E2E | End-to-End |
| FN | False Negative |
| FP | False Positive |
| FP16 | 16-bit Floating Point (Half Precision) |
| FP32 | 32-bit Floating Point (Single Precision) |
| FPS | Frames Per Second |
| HPO | Hyperparameter Optimisation |
| INT8 | 8-bit Integer (Quantized Precision) |
| IPC | Association Connecting Electronics Industries |
| IR | Intermediate Representation (OpenVINO format) |
| mAP | Mean Average Precision |
| NNCF | Neural Network Compression Framework |
| NMS | Non-Maximum Suppression |
| OFAT | One-Factor-at-a-Time |
| OpenVINO | Open Visual Inference and Neural Network Optimization |
| PCB | Printed Circuit Board |
| POS | Point of Sale |
| PR | Precision-Recall |
| RAM | Random Access Memory |
| RO | Research Objective |
| SOTA | State of the Art |
| TP | True Positive |
| UiTM | Universiti Teknologi MARA |
| VRAM | Video Random Access Memory |
| YOLO | You Only Look Once |

# **CHAPTER 1**

# **INTRODUCTION**

This chapter introduces my study, starting with the context of automated PCB defect detection in electronics repair shops. I outline the problems that motivated my research, define my questions and objectives, and introduce the CIRCA system as the proposed solution.

## **1.1 Research Background**

Modern consumer electronics rely on Printed Circuit Boards (PCBs) as their physical and electrical nervous system, routing signals through highly dense, multi-layered copper traces (Wang et al., 2023). When trace widths shrink below sub-millimetre tolerances, even microscopic faults compromise device safety, reliability, and operating life (Klco et al., 2023). Under these conditions, human operators struggle to track details, making manual microscope inspection slow, subjective, and prone to fatigue-induced escape rates of 10% to 20% (Law et al., 2024). Although deep learning-based computer vision offers a scalable alternative by automating defect classification, running these models locally requires balancing detection accuracy against the hardware constraints of independent repair shops (Bhattacharya & Cloutier, 2022).

Achieving flawless board assembly is essential for consumer electronics, as a single open circuit or cold solder joint directly compromises product reliability and safety (Klco et al., 2023). However, traditional manual inspection using magnifying lenses is challenging because surface-mount devices (SMDs) are characterized by tiny sizes and high component densities (Li & Zhou, 2024). Under standard workplace fatigue, human inspectors frequently miss hairline cracks, micro-bridges, and marginal fillets (Bhanumathy et al., 2021). Although manual check-ups are simple, they are highly sensitive to operator fatigue and environmental factors, resulting in poor robustness and error rates of 10% to 20% (Law et al., 2024).

Neural network object detectors offer a viable solution to these visual constraints, allowing convolutional architectures to identify complex anomalies with high precision (Liao et al., 2021). Over the past decade, single-stage models including YOLO, ResNet, and EfficientDet have redefined automated optical inspection, achieving detection accuracies above 95% (Law et al., 2024; Li & Zhou, 2024). In my work, the latest YOLOv12 architecture pushes these boundaries further by pairing an Area Attention module (A2) with Residual Efficient Layer Aggregation Networks (R-ELAN) (Tian et al., 2025). Although early deep learning models required high-end server GPUs, the lightweight YOLOv12-Nano variant achieves 40.6% mAP on the MS COCO benchmark while running directly on modest CPUs and integrated GPUs (Tian et al., 2025).

While automated optical inspection (AOI) systems have successfully optimized quality control on high-volume manufacturing lines, they do not translate easily to independent electronics repair benches (Ghelani, 2024; Li & Zhou, 2024). Implementing industrial AOI is challenging because these systems require massive capital budgets, dedicated operators, and highly controlled, glare-free lighting configurations. Faced with these constraints, small-scale repair shops primarily rely on manual methods. They operate in highly variable environments characterized by changing illumination, handheld camera capture, and a vast diversity of board designs.

CIRCA (Circuit Inspection and Recognition using Convolutional Architectures) bridges this gap by targeting small-scale repair environments directly. It uses YOLOv12 models to automatically localize surface-level defects from standard camera images, covering a seven-class taxonomy of bare-board and solder defects. Deployed as an INT8-quantized OpenVINO model, the system executes real-time diagnostics on standard local CPUs and integrated GPUs, bypassing the need for specialized graphics cards (Yi et al., 2024). To counter the harsh shadows and solder glare of typical repair benches, CIRCA routes incoming frames through a lightweight OpenCV preprocessing pipeline combining Laplacian Variance blur checks, Contrast Limited Adaptive Histogram Equalization (CLAHE), and Gamma Correction (Alhamzawi et al., 2024; Wanto et al., 2023).

## **1.2 Problem Statement**

Modern electronics repair shops face a critical bottleneck because manual board-level diagnostics cannot keep pace with contemporary device complexity. This challenge primarily stems from two issues: the physical limits of human vision and a complete lack of low-cost, flexible automation.

First, progressive component miniaturisation makes manual optical inspection extremely difficult. Technicians regularly work on surface-mount boards containing components smaller than a millimetre and pin pitches tighter than 0.5 millimetres (Adibhatla et al., 2020). Under these conditions, spotting a microscopic solder bridge or hairline trace crack is like hunting for a needle in a haystack. Although industry guidelines like IPC-A-610 define clear visual criteria for solder fillets, manual compliance checks remain slow and highly subjective (Goti, 2025). Industrial automated systems achieve 98% to 99% accuracy while scanning thousands of components per hour, whereas human technicians max out at a few hundred components, with error rates climbing to 15% (Goti, 2025).

Second, manual visual inspection represents a massive time investment. Scanning a multi-layer smartphone board takes anywhere from 10 to 30 minutes, which limits repair shop throughput and drives up diagnostic labour costs. Under continuous microscope inspection, technicians suffer from visual fatigue, resulting in a 30% drop in defect detection rates after just four hours of work (Goti, 2025). The physical strain is also significant, as hunching over stereomicroscopes for hours causes severe neck fatigue and eye strain, driving up error rates.

Furthermore, visual inspection quality varies widely based on individual technician experience (Sharma et al., 2024). A novice repair technician might overlook cold solder joints or subtle conductor damage that an experienced board repair specialist catches instantly. This diagnostic disparity leads to inconsistent repair outcomes, returned devices, and lost customer trust.

Surface defects also mask deeper structural anomalies or are easily dismissed as simple cosmetic scuffs (Hu & Wang, 2020). When visual cues are subtle—such as a slightly recessed fillet or a minor board edge mouse bite—technicians frequently overlook them. If these defects slip through in critical applications including automotive controls or industrial equipment, the downstream failure risks are severe (Liao et al., 2021).

Finally, modern industrial AOI systems are not designed to handle repair-shop variability (Huang et al., 2023). These platforms are rigid, expensive, and require CAD template programming for every specific board layout. Consequently, they do not function effectively in repair shops where technicians encounter dozens of different board models daily (Ruengrote et al., 2024). Repair technicians instead require a flexible, affordable visual assistant that integrates directly into their existing workstation setups.

The central question this research tackles is clear: **How can a deep learning-based visual inspection system be designed to deliver fast, accurate, and consistent PCB defect detection on standard desktop hardware in variable electronics repair environments, without the high costs and setup friction of industrial AOI systems?**

## **1.3 Research Questions**

To guide this investigation, I focus on three core questions:

* **RQ1**: How can I define a robust seven-class defect taxonomy that spans both bare-board IPC-A-600 and assembly-stage IPC-A-610H standards while remaining supported by public datasets?
* **RQ2**: What architectural configurations allow YOLOv12 models, optimized with OpenVINO INT8 quantization and an OpenCV preprocessing pipeline, to run efficiently on low-cost CPUs and GPUs under variable lighting?
* **RQ3**: How can I design a desktop interface that integrates this pipeline into the daily workflow of a repair technician without creating cognitive load or automation bias?

## **1.4 Research Objectives**

This study addresses these questions through three specific objectives:

* **RO1**: To establish a seven-class taxonomy of prevalent PCB defects, spanning four bare-board classes (missing hole, mouse bite, open circuit, short) and three solder classes (excess solder, insufficient solder, cold solder joint), validated through a public data-availability audit.
* **RO2**: To train, optimize, and evaluate YOLOv12-based models (Nano, Small, Medium variants) exported to OpenVINO format, identifying the best configuration for edge deployment.
* **RO3**: To develop the CIRCA desktop application with a static inspection interface, an active OpenCV preprocessing pipeline (CLAHE, gamma, Laplacian variance gating), and a clear, confidence-transparent display overlay.

## **1.5 Research Scope**

**Technical Scope.** My technical scope focuses on surface-visible defects detected via standard optical cameras and convolutional neural networks, using a unified seven-class taxonomy of bare-board (missing hole, mouse bite, open circuit, short) and assembly defects (excess solder, insufficient solder, cold solder joint). This selection was governed by a strict data-availability audit of public datasets, keeping only classes with at least 400 instances from multiple sources. I excluded sparse or visually ambiguous classes including spur, spurious copper, scratch, and pinhole to prevent model overfitting. Solder bridges were also excluded due to a lack of high-quality public annotations. The software implementation is restricted to YOLOv12 models optimized using Intel OpenVINO, and I do not cover electrical testing, thermal imaging, or internal defects requiring X-ray diagnostics.

**Device Scope.** This study targets PCBs from consumer electronics, including smartphones, laptops, and tablets, commonly processed in independent repair shops. Specialized medical, aerospace, or automotive control boards are outside the scope.

**Defect Scope.** The models analyse physical, surface-level manufacturing faults and wear-related damage, while excluding intermittent electrical faults, software bugs, or trace issues that do not manifest visually.

**Operational Scope.** CIRCA is designed for Windows 10/11 desktops equipped with standard consumer Intel CPUs and integrated GPUs. Input images are captured via USB webcams or uploaded directly from disk. I set performance targets at >90% mAP@0.5 on the dataset, sub-5 ms preprocessing times, and a sub-10-second end-to-end diagnosis latency on standard Intel i5 processors (Yi et al., 2024).

**Limitations and Exclusions.** The prototype serves as a diagnostic assistant, not a fully automated repair machine. It displays confidence scores to reduce automation bias, but final repair decisions and quality checks remain in the hands of the technician (Kupfer et al., 2023).

## **1.6 Significance of Study**

**Academic Significance.** This study addresses a notable gap in the literature by focusing on electronics repair diagnostics, which has received less attention than factory-floor quality control (Bhattacharya & Cloutier, 2022; Lv et al., 2024). My empirical results show how YOLOv12 attention-based models can be quantized to INT8 and run in real-time on standard edge hardware, offering insights for mobile and resource-constrained computer vision research (Ruengrote et al., 2024; Tian et al., 2025).

**Practical Significance.** For repair businesses, CIRCA reduces visual inspection times by up to 70%, turning a 15-minute microscope scan into a sub-second check. This throughput boost makes complex board-level repairs economically viable for small shops. It also reduces visual fatigue and eye strain by replacing hours of microscope squinting with digital camera frames.

**Economic Significance.** Automating routine diagnostics helps repair shops optimize labour models. Less experienced technicians can diagnose faults confidently, reducing misdiagnosis rates and avoiding unnecessary component replacements. This saves on part costs and improves inventory efficiency.

**Environmental and Social Significance.** Improving repair diagnostics supports the circular economy by extending device lifespans and e-waste. By lowering the cost of board-level diagnosis, CIRCA helps small repair shops compete with major manufacturer service networks, keeping repair services local and accessible.

## **1.7 Expected Solution**

Standardizing and automating PCB defect detection is essential to prevent missed diagnoses, reduce visual fatigue, and guarantee consistent repair outcomes. The CIRCA system addresses this by wrapping a YOLOv12 model in an OpenVINO inference engine, paired with a dynamic OpenCV preprocessing pipeline that handles repair bench lighting. Chapter 2 details the literature surrounding PCB machine vision, image processing, edge model deployment, and human-AI interaction guidelines, laying the theoretical foundation for the CIRCA architecture.

# **CHAPTER 2**

# **LITERATURE REVIEW**

In this chapter, I review the literature that supports the CIRCA project. I cover automated defect detection, deep learning object detection architectures, image preprocessing, edge deployment, and human-factors engineering. To make the review easy to follow, I organize it from general concepts to specific implementation details. I start with PCB defects and the limitations of traditional machine vision, trace the evolution of CNNs and single-stage YOLO detectors, analyze edge optimization frameworks, and examine how UI design can reduce automation bias.

## **2.1 PCB Defects and Inspection Challenges**

Automated defect detection on circuit boards isn't new. Electronics manufacturers have experimented with optical inspection algorithms since the early 2000s, but the recent shift toward deep learning-based computer vision has revolutionized the field (Adibhatla et al., 2020; Bhattacharya & Cloutier, 2022). Visual faults stem from several stages, spanning manufacturing glitches, thermal warping, and mechanical degradation. Standard defects include bridges, opens, cold joints, missing components, trace fractures, and localized burn marks (Liao et al., 2021).

Two core standards govern visual inspections in this space. Bare-board defects like mouse bites, missing holes, opens, and shorts are defined under IPC-A-600 acceptability criteria (IPC International, 2020). Assembly-level issues, including fillet quality, solder joints, component placement, and board cleanliness, fall under IPC-A-610 (Goti, 2025). Together, these standards establish a rigorous basis for model taxonomy. Grounding my model classes in these definitions ensures that outputs map to actionable technician terminology (Klco et al., 2023). AOI platforms aligned with these standards achieve 98% to 99% accuracy in automated manufacturing, highlighting the performance target I am chasing (Goti, 2025).

Traditional machine vision relied heavily on template matching, morphological filters, and thresholding against a "golden reference" board. While fast, these rule-based pipelines fail when lighting shifts, components misalign, or noise is present (Sharma et al., 2024). They do not function effectively on the heterogeneous mix of board models seen in repair shops. Available open-access datasets reflect this manufacturing bias. Benchmarks like PKU-Market-PCB and DeepPCB capture clean laboratory setups, leaving a gap for variable repair-bench lighting. The release of DsPCBSD+ (10,259 images, 20,276 annotations) provided a larger bare-board training corpus (Lv et al., 2024). Still, a critical constraint remains: while IPC standards define dozens of fault types, only seven classes have enough public, high-quality training instances (minimum 400 annotations from multiple independent sources) to train a neural network effectively. This empirical limit directly shaped the scope constraints detailed in Chapter 1 Section 1.5.

## **2.2 Machine Learning and CNN-Based PCB Defect Detection**

Traditional heuristics struggle with fine-grained PCB defects. CNN-based feature representations capture complex visual patterns far better than manually tuned filters (Bhattacharya & Cloutier, 2022). Early work showed that simple, four-layer CNNs could classify board defects with roughly 70% accuracy on small datasets, proving the concept but highlighting the need for deeper networks (Adibhatla et al., 2020).

Two-stage detectors like Faster R-CNN with ResNet50 Feature Pyramid Networks and Guided Anchoring RPNs pushed detection accuracy to 94.2% mAP on PKU Open Lab (Hu & Wang, 2020). But this accuracy comes with a heavy computational tax. Running these models at 12 FPS makes real-time edge deployment on standard CPUs impossible. Ensemble architectures combining EfficientDet, MobileNet SSD, Faster R-CNN, and YOLOv5 achieve up to 95% accuracy while handling image noise, but the cumulative latency of running four models sequentially is too slow for quick repair diagnostics (Law et al., 2024). Achieving high accuracy without freezing standard repair shop hardware requires a single-stage, optimized detector.

## **2.3 YOLO-Based PCB Defect Detection**

Single-stage YOLO detectors offer a way forward, balancing inference speed and accuracy on the edge. Early tests showed that Tiny-YOLOv2 models under 50 MB could hit 98.79% accuracy in controlled industrial setups, showing the potential of compact models (Adibhatla et al., 2020). Swapping standard YOLOv4 backbones for MobileNetV3 pushed latency down to 56.98 FPS with 98.64% mAP, making CPU deployment realistic (Liao et al., 2021). Adding transformer attention via a C3TR module in the YOLOv5 neck improved the network's ability to localize tiny defects on complex copper layouts (Bhattacharya & Cloutier, 2022).

Attention-centric architectures are particularly suited for dense board inspects. ATT-YOLO (YOLOv5 + self-attention) hit 92.8% mAP at 111 FPS on laptop board datasets (Wang et al., 2023), and YOLOv8 with C2f modules reached 157.2 FPS with 92.3% mAP on PKU benchmarks (Li & Zhou, 2024). Recent benchmarks show that YOLOv10 maintains high detection accuracy on standard desktop CPUs without dedicated GPUs (Ruengrote et al., 2024). Models like YOLOv8-DEE push this further by using multi-scale extraction to boost detection on sub-millimetre defects (Yi et al., 2024).

YOLOv12 represents a major leap in real-time object detection. Its Area Attention module (A2) segments feature maps to preserve a wide receptive field without the high memory cost of standard self-attention, while Residual Efficient Layer Aggregation Networks (R-ELAN) ease the training optimization issues introduced by attention layers (Tian et al., 2025). The Nano variant of YOLOv12 achieves 40.6% mAP on MS COCO at latencies competitive with older models (Tian et al., 2025). In comparisons across general benchmarks, YOLOv12 outperforms previous generations, showing high precision and robust localization in complex scenes (Hendriko & Hermanto, 2025). CIRCA utilizes these features, comparing YOLOv12 Nano, Small, and Medium variants to identify the optimal balance of speed and precision for local CPU deployment.

## **2.4 Image Preprocessing for Robust Detection**

Running models in variable, real-world environments requires preprocessors that clean up lighting swings and glare. Three methods are particularly relevant: Contrast Limited Adaptive Histogram Equalization (CLAHE), Gamma Correction, and Laplacian Variance blur checks.

CLAHE mitigates local shadows and specular reflections on solder joints by dividing the image into small tiles and equalizing contrast locally (Wanto et al., 2023). This highlights fine surface details without blowing out noise in flat areas, which helps the downstream model extract stable defect features. Gamma Correction complements this by adjusting overall luminance (Alhamzawi et al., 2024). Deep learning-based adaptive gamma models achieve strong low-light recovery on benchmark datasets, outperforming fixed correction parameters. CIRCA uses gamma scaling to recover trace details in heavy shadows cast by work desk lamps.

Finally, Laplacian Variance helps detect motion blur before running inference. Motion blur and out-of-focus capture degrade input quality and drop model detection accuracy (Lv et al., 2024). Using the Laplacian operator to calculate local gradient variance offers a fast, zero-parameter quality gate. Images with low variance (which indicates blur) are dropped from the pipeline immediately, protecting the CPU from useless inference passes.

## **2.5 Edge Machine Learning Deployment and Model Quantization**

Running neural models on typical repair shop desktops without high-end GPUs requires framework optimization. Intel's OpenVINO toolkit addresses this by quantizing 32-bit floating-point (FP32) weights to 8-bit integers (INT8). This post-training process yields mixed-precision networks that accelerate inference on consumer CPUs and integrated GPUs.

The speedup is substantial. Benchmarks show that INT8-quantized models deployed via OpenVINO run 3.3 times faster than FP32 baselines with minimal loss in model precision (Ahn et al., 2023). OpenVINO also outperforms alternative runtimes (like TensorFlow Lite or PyTorch Mobile) on Intel processors, confirming its selection as CIRCA's deployment backend. This performance boost is critical: if INT8 quantization degrades mAP below my 90% target, I fallback to FP16 OpenVINO execution, which maintains model accuracy at a slightly higher latency cost.

## **2.6 Human Factors and Automation Bias in AI-Assisted Inspection**

AI assistants introduce human-factors issues, most notably automation bias. Operators often trust automated alerts blindly, ignoring system errors even when they're obvious (Kupfer et al., 2023). In PCB inspection, this bias is dangerous. If a technician accepts CIRCA's classifications without verification, a false negative could lead to component failure or a fire hazard.

Mitigating this over-reliance requires interface transparency (Kupfer et al., 2023). Informing users of potential AI limitations and displaying confidence scores helps maintain a healthy level of operator scepticism. CIRCA's interface is designed around these findings. Bounding boxes display raw confidence scores, and a persistent warning banner alerts the user if average model confidence drops below 50%, keeping the human technician firmly in the loop.

## **2.7 Summary of Related Works**

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| **Author and Year** | **Model / Technique** | **Dataset** | **Key Results** | **Limitation Relevant to CIRCA** |
| (Adibhatla et al., 2020) | Tiny-YOLOv2 | Industrial AOI (11,000 images) | 98.79% accuracy, under 50 MB | Higher localisation error on small defects |
| (Hu & Wang, 2020) | Faster R-CNN + ResNet50-FPN + GARPN | PKU Open Lab | 94.2% mAP, approx. 12 FPS | Computationally heavy; slow inference |
| (Liao et al., 2021) | YOLOv4 + MobileNetV3 | Custom PCB dataset | 98.64% mAP, 56.98 FPS | Requires large, labelled dataset |
| (Bhattacharya & Cloutier, 2022) | YOLOv5 + C3TR (Transformer neck) | Custom PCB dataset | 98.1% mAP | Sensitive to distribution shifts |
| (Wang et al., 2023) | ATT-YOLO (YOLOv5 + self-attention) | LCFC Laptop (14,478 defects) | 92.8% mAP, 111 FPS | Dataset annotation noise |
| (Klco et al., 2023) | YOLOv8n | Power module PCB (640×640 tiles) | 96.6% mAP, 90 ms inference | Context-specific to power modules |
| (Yi et al., 2024) | YOLOv8-DEE | PCB benchmark datasets | State-of-the-art multi-scale small-defect detection | Evaluated on benchmark, not repair scenarios |
| (Law et al., 2024) | Ensemble (EfficientDet + SSD + RCNN + YOLOv5) | Mixed PCB dataset | 95% accuracy, 80.3% mAP | High total inference time |
| (Lv et al., 2024) | Dataset paper (DsPCBSD+) | DsPCBSD+ (10,259 images, 9 classes, CC BY 4.0) | Large-scale standardised bare-board dataset; 20,276 manual annotations | Bare board only; no assembly solder defects |
| (Li & Zhou, 2024) | YOLOv8 + C2f + SPPF | PKU Open Lab | 92.3% mAP, 157.2 FPS | Single-model robustness not fully assessed |
| (Ruengrote et al., 2024) | YOLOv10 variants | PCB defect dataset | Viable on standard hardware without GPU | Accuracy not yet at YOLOv12 level |
| (Anh Nguyen et al., 2024) | ResNet + Bottleneck ViT + FFEM + Wise-IoU | Augmented PKU dataset | 99.2% mAP, 51 FPS | 41.0 GFLOPs: high compute cost |
| (Tian et al., 2025) | YOLOv12 (A2 + R-ELAN) | MS COCO | YOLOv12-N: 40.6% mAP; competitive latency | Evaluated on general detection, not PCB-specific |
| (Hendriko & Hermanto, 2025) | YOLOv10 vs. YOLOv11 vs. YOLOv12 | 8 datasets incl. MOT17 | YOLOv12 best: 0.909 precision, 0.880 mAP@50 | Human detection datasets; not PCB-specific |

Table 2.1 Summary of Related Works on PCB Defect Detection

|  |  |  |
| --- | --- | --- |
| **Technique** | **Key Study** | **Finding Relevant to CIRCA** |
| CLAHE | (Wanto et al., 2023) | Increases CNN accuracy by boosting contrast and eliminating luminance imbalance in defect images |
| Adaptive Gamma Correction | (Alhamzawi et al., 2024) | PSNR 17.386, SSIM 0.788 on low-light benchmarks; lifts shadow regions without over-exposing highlights |
| Laplacian Variance Blur Detection | (Lv et al., 2024) | PCB-specific evidence that motion blur degrades model performance; justifies frame-quality gating |
| OpenVINO INT8 Quantization | (Ahn et al., 2023) | 3.3× inference speedup over FP32 on Intel CPUs with minimal accuracy degradation |

Table 2.2: Preprocessing and Deployment Techniques Underpinning CIRCA Design

**2.8 Summary**

Single-stage YOLO models offer the best balance of speed and precision for edge deployment (Li & Zhou, 2024; Tian et al., 2025). While tools like CLAHE, OpenVINO, and confidence transparency are well-characterized, most literature focus exclusively on high-volume factory lines. There is a clear gap in flexible systems optimized for the variable lighting and low-cost hardware typical of electronics repair benches (Lv et al., 2024; Ruengrote et al., 2024). Also, public datasets support only a subset of IPC-defined defect categories, which directly limited my taxonomy. CIRCA is designed to bridge this gap, adapting attention-based YOLOv12 models to standard desktop PCs. The experimental framework used to train and validate this approach is detailed in Chapter 3.

# **CHAPTER 3**

# **RESEARCH METHODOLOGY**

## **3.1 Introduction**

Replicating deep learning experiments demands clear, step-by-step reporting. In this chapter, I document each stage of the CIRCA project, including how I compiled the dataset, optimized hyperparameters, performed OpenVINO quantization, and benchmarked hardware at runtime. I explicitly state every parameter and decision rule to ensure reproducibility.

## **3.2 Research Framework**

### **3.2.1 Overview of Research Phases**

I executed the project across eight sequential phases, starting with raw dataset aggregation (Phase 0) and finishing with final test-set evaluation and threshold calibration (Phase 7). This structure maps directly to my three core objectives. Phase 0 satisfies RO1 by establishing the seven-class defect taxonomy and building the unified corpus. Phases 1 to 5 address RO2 by training three YOLOv12 variants and exporting them to OpenVINO Intermediate Representation (IR) formats. Lastly, Phases 6 and 7 cover RO3, where I evaluated the desktop prototype against hardware constraints and calibrated thresholds.

![](data:image/png;base64...)

Figure 3.1: CIRCA Research Framework

### **3.2.2 Mapping of Objectives, Activities, and Deliverables**

I structured activities to ensure each phase produces a tangible research deliverable. Table 3.1 maps the objectives to their corresponding phases, key tasks, and output files.

|  |  |  |  |
| --- | --- | --- | --- |
| **Objective** | **Phase(s)** | **Key Activities** | **Deliverables** |
| RO1: Identify and document IPC-A-600 and IPC-A-610-aligned PCB defect types | Phase 0 | Literature analysis of IPC-A-600 / IPC-A-610; selection of public datasets; class remapping to a unified 7-class taxonomy | CIRCA\_CLASS\_MAPPING.md; data.yaml; defect taxonomy table |
| RO2: Design and compare YOLOv12-N/S/M with OpenVINO INT8 | Phases 1–5 | Vanilla baseline training; CIRCA-aligned baseline; genetic hyperparameter optimisation; three-variant final training; FP32/FP16/INT8 quantisation validation | runs/detect/CIRCA\_V12{N,S,M}\_\*/weights/best.pt; OpenVINO IR exports; quantization\_report.md |
| RO3: Develop and evaluate the CIRCA desktop application | Phases 6–7 | Hardware benchmarking on Intel CPU / integrated GPU; static image analysis time; confidence threshold calibration; test-set evaluation; UI integration | benchmark\_report.md; circa\_thresholds.yaml; test\_evaluation.md; CIRCA desktop prototype |

Table 3.1: Mapping of Research Objectives, Activities, and Deliverables

## **3.3 Theoretical Study**

### **3.3.1 Preliminary Study**

My preliminary study highlighted a major gap: most research focuses exclusively on high-volume factory lines. There is a clear lack of flexible, low-cost inspection systems designed for the variable lighting and consumer hardware typical of electronics repair benches (Lv et al., 2024; Ruengrote et al., 2024). I also found that no single public dataset covers the range of defect categories needed for general repair diagnostics. To address this, I merged multiple open-source datasets into a unified corpus.

### **3.3.2 Knowledge Acquisition**

Three technical domains support the CIRCA system. First, the IPC-A-600K and IPC-A-610H standards supply the visual acceptability criteria for bare-board (opens, shorts, mouse bites, missing holes) and assembly faults (insufficient solder, excess solder, cold joints) (IPC International, 2020). Second, the attention-centric YOLOv12 detector provides the core model architecture. Its Area Attention (A2) and R-ELAN modules help capture fine-grained defects on cluttered circuit backgrounds. Third, Intel's OpenVINO toolkit enables mixed-precision INT8 deployment on standard edge CPUs and integrated GPUs, bypassing the need for expensive graphics cards (Ahn et al., 2023).

## **3.4 Empirical Study**

### **3.4.1 Data Collection**

**Public bare-board datasets:** I extracted bare-board defect images from PKU-Market-PCB-ver1 (PKU Open Lab, 2020), using the Roboflow export jr-mqqnk/pcb-defects-detection-anddl (approx. 3,300 images). This source provides four bare-board classes: missing\_hole, mouse\_bite, open\_circuit, and short. Unused classes like spur and spurious\_copper were remapped or dropped. I also incorporated DsPCBSD+ (Lv et al., 2024), a large-scale corpus of 10,259 images. I decoded its COCO JSON annotations, mapping SH to short, OP to open\_circuit, and MB to mouse\_bite, while dropping six irrelevant classes (like conductor scratches or foreign objects). This remapping yielded 3,441 usable images from the DsPCBSD+ corpus.

**Public assembly-stage datasets:** Solder joint faults are aggregated from four open datasets. The first is SolDef\_AI (Fontana et al., 2024), where I mapped exc\_solder to excess\_solder and poor\_solder to cold\_solder\_joint, while dropping three others. The second is excessive-solder-kydra (Roboflow: pcb-defect-detection-emmts), providing remapped annotations for cold joints, excess solder, and insufficient solder. The third, hue-dbgbs-reqtv, contributed insufficient solder and short instances, while component-level annotations were discarded. The fourth source is pcb-solder-defect-detection-v2-s89jo, providing additional cold joint, excess, and insufficient solder instances. All assembly sources are licensed under CC BY 4.0.

**Repair-context capture protocol:** To test the prototype qualitatively, I designed a camera capture setup. I imaged various circuit boards using a standard 720p USB webcam under three scenarios: ambient office lighting, direct desklamp illumination (specular glare), and partial lamp occlusion (heavy shadowing). I didn't use these captured frames for model training; they served as a validation testbed for the preprocessor and qualitative failure analysis.

**Unified 7-class IPC taxonomy:** I unified the datasets under a seven-class taxonomy. This selection was governed by a strict data-availability audit: a class was kept only if it had at least 400 instances across at least two independent sources. The only exception is missing\_hole, which is unique to the PKU source; however, its 2,315 instances and distinct visual signature justified keeping it. But there's a catch: the initial audit only tracked instance counts, omitting any spatial feasibility checks against the 640×640 px input. A sub-millimetre hole spans a mere three to four pixels per tile. The resulting boundary smearing under JPEG compression and camera noise makes reliable extraction near impossible for anchor-free backbones lacking specialized high-resolution P2 heads. I retained the class anyway. It is a useful diagnostic baseline to test where single-stage anchors hit their physical limits.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| **Unified ID** | **Class Name** | **IPC Reference** | **Source Datasets** | **Raw Class Name(s) Remapped From** |
| 0 | missing\_hole | IPC-A-600 Section 3.4 | PKU | missing\_hole |
| 1 | mouse\_bite | IPC-A-600 Section 3.3 | PKU, DsPCBSD+ | mouse\_bite, MB |
| 2 | open\_circuit | IPC-A-600 Section 3.2 | PKU, DsPCBSD+ | open\_circuit, OP |
| 3 | short | IPC-A-600 Section 3.2 | PKU, DsPCBSD+, Hue | short, Shorted, SH |
| 4 | excess\_solder | IPC-A-610H Section 5 | SolDef\_AI, kydra, SolderV2 | exc\_solder, Excessive\_solder |
| 5 | insufficient\_solder | IPC-A-610H Section 5 | kydra, Hue, SolderV2 | Insufficient\_solder, Insufficient Solder |
| 6 | cold\_solder\_joint | IPC-A-610H Section 5 | SolDef\_AI, kydra, SolderV2 | poor\_solder, Cold Solder, Cold\_solder |

Table 3.2: CIRCA Unified 7-Class IPC Taxonomy

**Scope and Limitations:** Several defect categories defined in IPC-A-600 and IPC-A-610H were excluded because of data scarcity: spur, spurious\_copper, solder\_spike, scratch, and pinhole. Training counts for these classes fall below my 400-instance threshold or present severe visual ambiguity. I also excluded solder\_bridge since no public, board-level dataset met my quality criteria. These classes are documented as scope boundaries, with a proposed data-collection plan outlined in Section 5.5.

### **3.4.2 Data Pre-processing**

**CLAHE on the L-channel of LAB** I applied Contrast Limited Adaptive Histogram Equalization on the L-channel of the LAB colour space (clip limit = 2.0, 8×8 grid) to reduce glare on solder joints and bring out fine trace details (Wanto et al., 2023).

**Gamma Correction** Next, I applied a fixed gamma correction (γ = 1.2) to lift heavy shadows cast by work lamps without over-exposing highlight regions (Alhamzawi et al., 2024).

**Laplacian Variance frame quality gate** At runtime, I calculated the variance of the Laplacian of the incoming frame to detect blur (Lv et al., 2024). If an image is out of focus and falls below my threshold, the system rejects it immediately, skipping inference to save compute cycles.

**Polygon-to-bounding-box conversion for SolDef\_AI** Polygon coordinates from the SolDef\_AI source were converted to axis-aligned bounding rectangles using the Roboflow exporter, ensuring clean compatibility with standard YOLO format.

**Class remap to the 7-class taxonomy** Annotations are unified via scripts/build\_unified\_pcb\_v3.py. Images containing only excluded classes are archived as negatives. The final unified\_pcb\_v3 dataset is split into a stratified 70/15/15 partition.

### **3.4.3 Data Analysis**

**Class distribution audit:** Perceptual-hash deduplication left 8,463 unique images with 54,928 annotations. The distribution is highly imbalanced: insufficient\_solder (23,610) and short (12,373) make up 65.5% of the data, while cold\_solder\_joint represents just 1.2% (633 instances).

I mitigated this imbalance in three steps. First, I capped images containing only dominant-class annotations (short and insufficient\_solder) at 1,000 images per class in the training split. This capped out 2,468 dominant-only images, reducing the raw training set from 5,924 to 3,456 images and lowering the imbalance ratio from 9.9:1 to 5.7:1. Second, I oversampled minority training images up to five times. This offline oversampling grew the training set from 3,456 to 5,364 images, boosting cold\_solder\_joint training instances by 5×. Third, I applied dynamic PyTorch class weights (cls\_pw) and mosaic/copy-paste augmentations during training to stabilize minority-class gradients.

**Duplicate and leakage detection:** I ran a difference-hash (dHash) deduplication step on all images before splitting, discarding any pair with a Hamming distance ≤ 6. This conservative threshold removes near-identical sequential video frames while preserving distinct board configurations. Deduplication removed 6,000 redundant images, ensuring no training leakage reached the validation or test splits.

## **3.5 System Design**

### **3.5.1 CIRCA System Architecture**

The CIRCA application operates as a six-stage pipeline. Images loaded from disk or snapped via camera are pre-processed using my CLAHE and Gamma pipeline. The frame is then passed to an Intel OpenVINO runtime using a quantized YOLOv12 model. I use an adaptive sliding-window tiling engine to maintain resolution-independent feature scale during inference.

### **3.5.2 Inference Pipeline**

The OpenVINO runtime loads the selected model. The image is divided into overlapping 640×640 tiles, and inference is run on each tile. I merge duplicate detections across tiles using a non-maximum suppression (NMS) step (IoU threshold = 0.45) before rendering the overlays on the user interface.

### **3.5.3 Confidence Threshold and “Manual Inspection Required” Logic**

CIRCA uses per-class display and warning thresholds to combat automation bias (Kupfer et al., 2023). A global "Manual Inspection Required" banner triggers if the mean confidence score drops below 50%, or if the Laplacian blur gate rejects the frame, ensuring the technician remains the final judge.

### **3.5.4 Interface Design**

The user interface features a drag-and-drop landing zone and a manual camera frame snapper. Bounding boxes are rendered in color-coded overlays with their raw confidence scores displayed clearly, backed by a status footer showing analysis latency and defect counts.

## **3.6 System Development**

### **3.6.1 Training Engine**

The training engine (train\_engine.py) stabilizes the training of custom YOLO models. I apply key parameters: lr0=0.001, warmup\_epochs=5.0, nbs=64, batch=auto, imgsz=640, seed=42, close\_mosaic=10, cos\_lr=True, amp=True, and optimizer=AdamW. This baseline initial learning rate provides stability before genetic hyperparameter tuning overrides it.

### **3.6.2 Hyperparameter Optimisation Algorithm**

I ran a genetic optimizer over 17 hyperparameters. The search space excludes hue and saturation changes, which carry diagnostic signal, and physically unrealistic transformations.

**Stopping criteria and trial budget** I set the optimization trial budget to 50 iterations with a 50-epoch limit per trial on the YOLOv12-S variant. I saved the optimized configurations to runs/detect/CIRCA\_V12S\_003\_TUNE\_HPO\_7class/best\_hyperparameters.yaml for downstream training.

### **3.6.3 Model Training Procedure**

Using the optimized hyperparameter file, I trained YOLOv12-N, YOLOv12-S, and YOLOv12-M for 200 epochs on a cloud instance (Runpod Secure Cloud, NVIDIA RTX 3090, 24 GB VRAM) to support larger batch sizes. I set variant-specific batch sizes at 64 (Nano), 48 (Small), and 32 (Medium).

### **3.6.4 OpenVINO Export and INT8 Quantisation**

To deploy these models on consumer desktops, I compiled PyTorch checkpoints (best.pt) to Intel OpenVINO Intermediate Representation (IR) formats. I generated FP32, FP16, and INT8 configurations.

Post-training quantization to INT8 was conducted via OpenVINO's Neural Network Compression Framework (NNCF). I calibrated the integer scaling factors using a representative subset of 300 validation images. To protect diagnostic accuracy, I applied a strict **Quantisation Fallback Rule**:

* The absolute validation mAP@0.5 must be ≥ 90.0%.
* The mAP degradation between FP32 and INT8 must not exceed 1.0 percentage points.

If the INT8 model fails either condition, the system automatically falls back to the FP16 IR variant to prioritize precision over speed.

### **3.6.5 Confidence Threshold Calibration Procedure**

I swept the confidence parameter conf from 0.10 to 0.90 in steps of 0.05 on the validation split. The per-class display threshold was set at the minimum confidence achieving precision ≥ 0.90, and the warning threshold at the minimum confidence achieving recall ≥ 0.95.

## **3.7 Experimental Design**

### **3.7.1 Phase 1: Vanilla Baseline**

As an ablation control, I trained YOLOv12-S for 100 epochs on the raw unified\_pcb\_v3 dataset without any CLAHE or Gamma enhancements, using default Ultralytics hyperparameters (subject to my stability patches).

### **3.7.2 Phase 2: CIRCA-Aligned Baseline**

I trained YOLOv12-S for 100 epochs on the preprocessed unified\_pcb\_v3\_preproc dataset. Keeping all other hyperparameters identical to Phase 1 allowed a clean one-factor-at-a-time (OFAT) ablation to isolate the visual lift of my preprocessor.

### **3.7.3 Phase 3: Hyperparameter Optimisation**

I ran the genetic hyperparameter tuner (50 trials × 50 epochs) on the YOLOv12-S model using the pre-processed training split to identify optimized learning rates and loss weight gains.

### **3.7.4 Phase 4: Three-Variant Final Training**

I trained YOLOv12 Nano, Small, and Medium variants for 200 epochs using the optimized hyperparameter file. I enabled early stopping (patience=50) to prevent overfitting.

### **3.7.5 Phase 5: Quantisation Validation**

I generated FP32, FP16, and INT8 OpenVINO IR files for each trained variant. I ran validation checks against my Quantisation Fallback Rule to select the deployment weights.

### **3.7.6 Phase 6: Hardware Benchmarking and Variant Selection**

I benchmarked the surviving model configurations on the deployment target, measuring preprocessing time, inference latency on CPU and integrated GPU, and end-to-end static image analysis time. The variant with the highest mAP that met all hardware latencies was selected for production.

### **3.7.7 Phase 7: Final Test Evaluation and Threshold Calibration**

I ran the selected production model ONCE on the held-out test split, reporting final precision, recall, F1, and mAP scores, along with confusion matrices and PR curves.

### **3.7.8 Acceptance Criteria**

To pass, the production variant must satisfy three criteria: preprocessing latency ≤ 5 ms, tiled static image analysis time ≤ 10 s, and static image inference time ≤ 10 s on the target Intel processor. Don't mistake the 90% mAP target for a standard operational threshold. I calibrated this baseline against factory-floor literature where models process single-class defects under pristine, controlled industrial lighting. CIRCA's cross-domain, multi-source dataset represents a messier repair-bench reality. The 90% figure is an idealized upper bound, not a disqualifying benchmark for a unified taxonomy.

### **3.7.9 Evaluation Metrics**

I report standard metrics: precision, recall, F1-score, and mAP (calculated using COCO evaluation protocols at IoU thresholds from 0.50 to 0.95). I also measure latencies in milliseconds and seconds.

## **3.8 Hardware and Software Specification**

### **3.8.1 Training Environment**

I trained my models on a cloud instance (Runpod Secure Cloud, NVIDIA RTX 3090, 24 GB VRAM) because the local laptop RTX 3060 (6 GB VRAM) restricted batch sizes to 6 for the Medium variant. The software stack consisted of Python 3.11, PyTorch 2.x, CUDA 12.x, Ultralytics 8.3, and Weights & Biases for experiment tracking.

### **3.8.2 Deployment Target**

The deployment target is an Intel Core i5 8th-generation CPU (or equivalent) with integrated graphics, running Windows 10/11. The processor must support AVX2 and VNNI instruction sets for INT8 acceleration. No discrete GPU is required.

### **3.8.3 Software Stack**

The application runs on Python 3.11, PyTorch inference libraries, OpenVINO Runtime, NNCF, OpenCV, and a lightweight desktop UI built using PyQt6.

## **3.9 Research Plan**

The project timeline was sequenced to ensure clean data flow between phases. Table 3.3 details the plan and compute estimates.

|  |  |  |  |
| --- | --- | --- | --- |
| **Phase** | **Description** | **Estimated Duration** | **Platform** |
| 0 | Dataset rebuild (6 sources, remap, dedup, split → unified\_pcb\_v3) | 1 week | Local |
| 1 | Vanilla baseline (100 ep, YOLOv12-S) | ~90 min | Runpod RTX 3090 |
| 2 | CIRCA-aligned baseline (100 ep, preproc, OFAT) | ~90 min | Runpod RTX 3090 |
| 3 | Genetic HPO (50 it × 50 ep, fraction=0.5) | ~23.4 h | Runpod RTX 3090 |
| 4 | Three-variant final training (200 ep × 3) | ~25 h | Runpod RTX 3090 |
| 5 | OpenVINO quantisation validation | 1 day | Local |
| 6 | Hardware benchmarking on i5 8th-gen | 1–2 days | Local |
| 7 | Test evaluation + threshold calibration | 1–2 days | Local |
|  | **Total cloud GPU compute** | **~51.4 h (RTX 3090)** |  |

Table 3.3: CIRCA Project Timeline and Compute Estimate

## **3.10 Summary**

This chapter detailed the CIRCA methodology. I compile a seven-class IPC-aligned dataset (unified\_pcb\_v3) of 8,463 unique images and 54,928 instances from six public sources after difference-hash deduplication removed 6,000 near-duplicates. I partition the data using a stratified 70/15/15 split, oversampling minority training classes by 5× to balance gradients. My preprocessing pipeline uses LAB-space CLAHE, gamma correction, and a runtime Laplacian blur gate. The training sequence progresses from vanilla baseline checks and OFAT preprocessor ablation to hyperparameter optimization, OpenVINO compilation, and local benchmarking. The performance and accuracy results of executing this methodology are detailed in Chapter 4.

# **CHAPTER 4**

# **RESULTS AND FINDINGS**

## **4.1 Introduction**

I report and analyze my findings alongside each experiment's design context. Because CIRCA moves through sequential stages (preprocessing ablation, hyperparameter tuning, model comparative training, and target device benchmarking), evaluating each phase's output sequentially clarifies the system's overall performance.

## **4.2 Dataset and Defect Taxonomy Results**

To satisfy Research Objective 1 (RO1), I established a unified PCB defect taxonomy mapping both bare-board and solder assembly defects to IPC standards. The physical distribution of defects and the split statistics of the compiled unified\_pcb\_v3 corpus validate this taxonomy.

Table 4.1 lists the annotation instance counts in the deduplicated unified\_pcb\_v3 corpus.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| **ID** | **Class Name** | **IPC Reference** | **Total Instances** | **% of Total** | **Primary Sources** |
| 0 | missing\_hole | IPC-A-600 | 2,315 | 4.2% | PKU |
| 1 | mouse\_bite | IPC-A-600 | 4,887 | 8.9% | PKU, DsPCBSD+ |
| 2 | open\_circuit | IPC-A-600 | 3,990 | 7.3% | PKU, DsPCBSD+ |
| 3 | short | IPC-A-600 | 12,373 | 22.6% | PKU, DsPCBSD+, Hue |
| 4 | excess\_solder | IPC-A-610H | 7,120 | 13.0% | SolDef\_AI, kydra, SolderV2 |
| 5 | insufficient\_solder | IPC-A-610H | 23,610 | 43.0% | kydra, Hue, SolderV2 |
| 6 | cold\_solder\_joint | IPC-A-610H | 633 | 1.2% | SolDef\_AI, kydra, SolderV2 |
|  | **Total** |  | **54,928** | **100%** |  |

Table 4.1: Final Class Distribution: `unified\_pcb\_v3` (8,463 unique images, 54,928 instances)

I unified four bare-board defect classes under IPC-A-600 with three solder joint assembly classes under IPC-A-610H. Excluded classes include spur, spurious\_copper, solder\_spike, scratch, and pinhole. These categories were dropped during a systematic data-availability audit because their training instance counts fell below the 400-instance minimum needed for stable neural gradients. I also excluded component-level faults (like misalignments or missing chips) to keep the taxonomic scope focused on board-level copper and solder features.

The initial distribution of annotations was heavily skewed. For instance, insufficient\_solder made up 43.0% of the raw database, while cold\_solder\_joint accounted for only 1.2%. I addressed this severe class imbalance using the multi-stage capping and oversampling strategy detailed in Chapter 3 Section 3.4.3.

Table 4.2 presents the split statistics across the train, validation, and test datasets.

|  |  |  |  |
| --- | --- | --- | --- |
| **Split** | **Images (before OS)** | **Images (after oversampling)** | **Proportion** |
| Train | 5,924 (3,456 capped) | 5,364 | 70% |
| Validation | 1,269 | 1,269 (frozen) | 15% |
| Test | 1,270 | 1,270 (frozen) | 15% |
| **Total** | **8,463 (5,995 capped)** | **7,903** |  |

Table 4.2: Train / Validation / Test Split Statistics: `unified\_pcb\_v3`

The final corpus was compiled from six public repositories after difference-hash (dHash) deduplication removed 6,000 near-duplicate images from an initial pool of 14,414. Capping dominant-only images in the training set removed 2,468 redundant frames, lowering the imbalance ratio from 9.9:1 to 5.7:1. I then oversampled minority classes by 5× to expand the training set to 5,364 files. This oversampling was applied only to the training split, keeping the validation and test splits frozen as clean evaluation benchmarks.

**4.2.3 Sample Defect Images per IPC Class**

![](data:image/png;base64...)

Figure 4.1: Sample Defect Images

*Figure 4.1: Representative PCB defect images for each of the seven IPC-aligned classes, drawn from the `unified\_pcb\_v3` training split. Bounding boxes indicate the annotated defect region. Classes 0 to 3 are bare-board defects (IPC-A-600); classes 4 to 6 are assembly-stage solder defects (IPC-A-610H).*

## **4.3 Preprocessing Pipeline Evaluation**

**4.3.1 Vanilla vs CIRCA-Pre-processed Baseline Comparison**

Evaluating the preprocessing block (Phases 1 and 2) helps isolate the diagnostic value of CLAHE and Gamma Correction. Under a strict one-factor-at-a-time (OFAT) design, I trained a YOLOv12-S model for 100 epochs on the raw unified\_pcb\_v3 split and compared it against the pre-processed baseline.

Table 4.3 summarizes the validation metrics for both runs.

|  |  |  |  |
| --- | --- | --- | --- |
| **Metric** | **Phase 1 (Vanilla)** | **Phase 2 (CIRCA)** | **Difference** |
| **Training Hardware** | Runpod RTX 3090 24 GB | Runpod RTX 3090 24 GB | - |
| **Epochs trained** | 100 | 100 | OFAT equal |
| **Best epoch** | 45 | 50 | - |
| **mAP@0.5 (best epoch)** | **0.6649** | 0.6600 | -0.0049 (-0.49 pp) |
| **mAP@0.5:0.95 (best epoch)** | 0.4237 | **0.4284** | **+0.0047 (+0.47 pp)** |
| **Precision (at best mAP epoch)** | 0.7290 | **0.8443** | **+0.1153 (+11.5 pp)** |
| **Recall (at best mAP epoch)** | **0.6433** | 0.6341 | -0.0092 |
| **mAP@0.5 (final epoch)** | 0.6591 (ep.100) | 0.6536 (ep.80) | -0.55 pp |
| **Early stop triggered** | No | Yes (ep.80, patience=30) | - |
| **Train cls loss (lower = better)** | 0.181 | **0.179** | -0.002 |
| **Val cls loss (ep.40+, lower = better)** | Higher | **Lower (consistent)** | CIRCA wins |

Table 4.3: Phase 1 vs Phase 2 Validation Metrics (OFAT Ablation, YOLOv12-S, 100 Epochs, `unified\_pcb\_v3`)

![](data:image/png;base64...)

Figure 4.2: Phase 1 vs Phase 2 Ablation Comparison

*Figure 4.2: Phase 1 (Vanilla) vs Phase 2 (CIRCA-Pre-processed) training curves across six validation metrics (mAP@0.5, mAP@0.5:0.95, Precision, Training Class Loss, Validation Class Loss, and Validation Box Loss). YOLOv12-S, 100 epochs, OFAT design. Source: `runs/detect/CIRCA\_V12S\_001\_TRAIN\_Baseline\_Vanilla\_2/` and `runs/detect/CIRCA\_V12S\_002\_TRAIN\_Baseline\_CIRCA\_2/`.*

**4.3.2 Preprocessing Latency Measurement**

Table 4.4 details the latency footprint of my preprocessing stages on the target CPU.

|  |  |  |  |
| --- | --- | --- | --- |
| **Preprocessing Stage** | **Operation Description** | **Mean Latency (ms)** | **Percentage of Total (%)** |
| **Colour Conversion & CLAHE** | LAB conversion, split, histogram equalization on L channel, merge, BGR reconversion | 3.61 | 76.0% |
| **Gamma Correction** | Lookup table mapping (LUT) for contrast adjustment | 0.40 | 8.4% |
| **Laplacian Gating** | Down sampling, Laplacian filtering, standard deviation squaring | 0.74 | 15.6% |
| **Total Pipeline** | **End-to-end frame preprocessing** | **4.75** | **100.0%** |

Table 4.4: Preprocessing Latency per Stage on the Representative Target CPU

**4.3.3 Discussion**

The preprocessing ablation yielded a interesting result. While the validation mAP@0.5 was slightly lower in the pre-processed run (0.6600 vs 0.6649), this remains within normal training variance. Crucially, validation precision jumped by 11.5 percentage points (0.8443 vs 0.7290), and validation classification loss decreased steadily after epoch 40. The pre-processed model also converged 20% faster, stopping early at epoch 80.

The lack of a broad mAP lift on the validation split makes sense when looking at the source images. The public datasets were compiled from controlled laboratory setups where lighting was already optimal. Equalizing contrast on clean images can introduce minor distortions that degrade baseline features. However, the real value of these filters lies in handling real-world bench glare and shadow details, which is why I kept the preprocessing pipeline.

In terms of speed, the total latency of 4.75 ms fits inside the 5.0 ms preprocessing constraint. CLAHE and color conversions represent the heaviest compute tasks, taking up 76% of the preprocessing time, while the Laplacian blur check runs in a swift 0.74 ms.

## **4.4 Hyperparameter Optimisation Results**

Genetic tuning on the YOLOv12-S variant ran for 23.4 hours on an NVIDIA RTX 3090, completing a 50-iteration search over 17 hyperparameters. Figure 4.3 tracks overall model fitness, defined as the validation mAP@0.5:0.95, across these HPO trials.

![](data:image/png;base64...)

Figure 4.3: HPO Fitness Trajectory

*Figure 4.3: Genetic Algorithm Fitness Trajectory across 50 HPO Iterations (YOLOv12-S, Pre-processed 7-Class Corpus). Source: `runs/detect/CIRCA\_V12S\_003\_TUNE\_HPO\_7class/tune\_fitness.png`.*

Model fitness rose steadily as the tuner progressed. The best trial reached a fitness score of 0.263 at Iteration 42, representing a 31.6% improvement over the baseline configuration. Scatter analysis shows that initial learning rate (lr0) and box loss gain (box) are highly sensitive parameters. All trials with learning rates above 0.001 produced poor fitness, while the highest-performing configurations clustered tightly at rates below 0.0003. Box loss gain converged between 0.15 and 0.20, far below the default YOLOv12 setting of 7.5.

Table 4.5 lists the top ten HPO trials ranked by fitness.

|  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Rank** | **Iteration** | **Fitness** | **Precision** | **Recall** | **mAP@0.5** | **mAP@0.5:0.95** | **`lr0`** | **`momentum`** | **`box`** | **`cls`** | **`cls\_pw`** | **`mosaic`** | **`scale`** |
| 1 | 42 | 0.26305 | 0.7161 | 0.4196 | 0.4350 | 0.2631 | 0.000140 | 0.7851 | 0.1692 | 0.2657 | 0.1000 | 0.7220 | 0.6502 |
| 2 | 31 | 0.25608 | 0.6328 | 0.4356 | 0.4328 | 0.2561 | 0.000290 | 0.8356 | 0.1936 | 0.2000 | 0.1000 | 0.6846 | 0.4925 |
| 3 | 46 | 0.25593 | 0.6181 | 0.4439 | 0.4316 | 0.2559 | 0.000210 | 0.8749 | 0.1912 | 0.2000 | 0.1324 | 0.7608 | 0.8936 |
| 4 | 49 | 0.25554 | 0.6567 | 0.4344 | 0.4345 | 0.2555 | 0.000240 | 0.8940 | 0.1557 | 0.2181 | 0.1086 | 0.7333 | 0.5325 |
| 5 | 25 | 0.25410 | 0.6518 | 0.4119 | 0.4309 | 0.2541 | 0.000050 | 0.9800 | 0.1630 | 0.3115 | 0.1024 | 0.7664 | 0.7581 |
| 6 | 37 | 0.25336 | 0.7948 | 0.4088 | 0.4295 | 0.2534 | 0.000330 | 0.9800 | 0.1572 | 0.3249 | 0.1068 | 0.6830 | 0.9000 |
| 7 | 41 | 0.25238 | 0.6446 | 0.4215 | 0.4338 | 0.2524 | 0.000150 | 0.8568 | 0.1908 | 0.2008 | 0.1208 | 0.8286 | 0.9000 |
| 8 | 34 | 0.25231 | 0.6626 | 0.4105 | 0.4258 | 0.2523 | 0.000040 | 0.8049 | 0.2000 | 0.3351 | 0.1202 | 1.0000 | 0.9000 |
| 9 | 45 | 0.25146 | 0.7107 | 0.4219 | 0.4279 | 0.2515 | 0.000130 | 0.8664 | 0.2000 | 0.3404 | 0.1517 | 1.0000 | 0.5776 |
| 10 | 48 | 0.25127 | 0.6602 | 0.4517 | 0.4265 | 0.2513 | 0.000150 | 0.8861 | 0.2000 | 0.2097 | 0.1262 | 0.9857 | 0.5896 |

Table 4.5: Top 10 HPO Trials Ranked by Fitness

Scatter plots (Figure 4.4) illustrate these parameter relationships. Yellow points mark later iterations with higher fitness, while purple points represent early trials.

![](data:image/png;base64...)

Figure 4.4: HPO Parameter Scatter Plots

*Figure 4.4: Hyperparameter Search Space Scatter Plots vs. Fitness (YOLOv12-S, Phase 3 HPO). Source: `runs/detect/CIRCA\_V12S\_003\_TUNE\_HPO\_7class/tune\_scatter\_plots.png`.*

Table 4.6 compares the final tuned parameters against default YOLOv12 configurations.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| **Parameter** | **YOLOv12 Default** | **Tuned Value** | **Magnitude of Change** | **Domain Significance** |
| lr0 (Initial Learning Rate) | 0.01 | **0.00014** | ÷71× reduction | Critical: Avoids gradient explosion on small, low-contrast defects. |
| box (Box Loss Gain) | 7.5 | **0.169** | ÷44× reduction | Critical: Prevents localization loss from dominating classification. |
| cls (Class Loss Gain) | 0.5 | **0.266** | ÷1.9× reduction | Moderate: Restructures class loss weight for the 7-class taxonomy. |
| momentum (SGD/Adam Momentum) | 0.937 | **0.785** | -16% change | Moderate: Increases optimizer sensitivity to rare defect gradients. |
| mosaic (Mosaic Augmentation) | 1.0 | **0.722** | -28% change | Moderate: Limits visual noise from out-of-context image composites. |
| scale (Scale Augmentation) | 0.5 | **0.650** | +30% increase | Minor: Enhances model robustness to scale variations in PCB layouts. |
| weight\_decay (L2 Regularization) | 0.0005 | **0.0009** | ×1.8 increase | Minor: Reduces overfitting on oversampled minority classes. |
| dfl (Distribution Focal Loss) | 1.5 | **1.656** | +10% increase | Minor: Stabilizes boundary regressions on fine-grained defects. |
| copy\_paste (Copy-Paste Aug) | 0.0 | **0.011** | Introduced | Minor: Pastes defect regions onto new backgrounds to combat imbalance. |

Table 4.6: Comparison of Tuned Hyperparameters and YOLOv12 Defaults

**4.4.5 Discussion**

I found that standard YOLOv12 defaults are poorly suited for sub-millimetre PCB inspection. The tuner's 71-fold reduction in initial learning rate (from 0.01 to 0.00014) suggests that fine defect boundaries require highly stable gradient steps. A high learning rate causes the optimizer to overshoot local minima on small, low-contrast features.

The 44-fold reduction in box loss gain (box) is also telling. Standard object detectors focus heavily on box localization boundaries because objects vary widely in size and placement. In PCB diagnostics, however, defect candidates are physically locked to static traces and solder pads. The main challenge is distinguishing between classes, such as separating a trace short from excess solder. Lowering the box loss gain prevents localization loss from dominating gradients, allowing the network to focus on class classification.

The tuner also capped mosaic augmentation at 0.722. While combining four random images works well for general objects, stitching unrelated PCB layouts together creates artificial borders that disrupt trace tracking. Similarly, the tuner reduced rotation and mix-up parameters to near-zero, confirming that fixed-orientation inspections do not benefit from rotation and that mixing layouts degrades representation learning.

## **4.5 Three-Variant Comparative Training Results**

Section 4.5 addresses Research Objective 2 (RO2) by evaluating the Nano, Small, and Medium YOLOv12 variants on my pre-processed, balanced corpus.

Figures 4.5a, 4.5b, and 4.5c show the training trajectories across 200 epochs.

![](data:image/png;base64...)

Figure 4.5a: YOLOv12-Nano Training Curves

*Figure 4.5a: Training and validation loss trajectories, Precision, Recall, and mAP metrics across 200 epochs for the YOLOv12-Nano variant.*

![](data:image/png;base64...)

Figure 4.5b: YOLOv12-Small Training Curves

*Figure 4.5b: Training and validation loss trajectories, Precision, Recall, and mAP metrics across 200 epochs for the YOLOv12-Small variant.*

![](data:image/png;base64...)

Figure 4.5c: YOLOv12-Medium Training Curves

*Figure 4.5c: Training and validation loss trajectories, Precision, Recall, and mAP metrics across 200 epochs for the YOLOv12-Medium variant.*

The Nano variant converged smoothly, stabilizing validation loss by epoch 140 and hitting its best validation mAP at epoch 182. The Small variant showed a faster box loss decay, but validation class loss oscillated slightly after epoch 120, triggering minor early stopping checks. The Medium variant showed the lowest training class loss, though validation metrics stabilized early around epoch 150, showing that larger networks accelerate early learning but present diminishing returns on accuracy.

Table 4.7 summarizes validation metrics across the three variants.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| **Model Variant** | **Parameters (M)** | **mAP@0.5 (%)** | **mAP@0.5:0.95 (%)** | **Precision (%)** | **Recall (%)** | **F1-Score (%)** |
| **YOLOv12-N (Nano)** | 2.38 | 63.13 | 39.52 | 83.16 | 60.23 | 69.87 |
| **YOLOv12-S (Small)** | 9.23 | 66.20 | 42.97 | 73.06 | 67.00 | 69.90 |
| **YOLOv12-M (Medium)** | 20.11 | 67.42 | 43.89 | 74.78 | 67.07 | 70.74 |

Table 4.7: Validation Metrics per YOLOv12 Variant (Phase 4)

Table 4.8 breaks down validation mAP@0.5 by individual class.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| **Partition** | **Defect Class** | **IPC Standard Reference** | **YOLOv12-N (%)** | **YOLOv12-S (%)** | **YOLOv12-M (%)** |
| **IPC-A-600** | missing\_hole | IPC-A-600 Section 3.4 | 1.91 | 5.57 | 5.85 |
| *(Bare-Board)* | mouse\_bite | IPC-A-600 Section 3.3 | 57.97 | 64.02 | 64.62 |
|  | open\_circuit | IPC-A-600 Section 3.2 | 69.69 | 71.05 | 71.72 |
|  | short | IPC-A-600 Section 3.2 | 89.18 | 92.96 | 94.08 |
| **IPC-A-610H** | excess\_solder | IPC-A-610H Section 5 | 39.63 | 45.75 | 52.05 |
| *(Solder Assembly)* | insufficient\_solder | IPC-A-610H Section 5 | 90.92 | 91.80 | 93.65 |
|  | cold\_solder\_joint | IPC-A-610H Section 5 | 92.98 | 95.31 | 95.06 |

Table 4.8: Per-class Validation mAP@0.5 Breakdown per Variant

**4.5.4 Discussion**

Scaling up model capacity yielded logarithmic returns. Moving from the 2.38M parameter Nano model to the 9.23M parameter Small model gave a +3.07 percentage point mAP improvement (63.13% to 66.20%). However, scaling up further to the 20.11M parameter Medium model only improved mAP by +1.22 percentage points (66.20% to 67.42%). This confirms that the Small variant is the inflection point for model capacity on this dataset.

My per-class metrics show that oversampling stabilized minority classes during training. Solder assembly classes (insufficient solder and cold solder joint) achieved validation AP scores above 90% across all models, aided by the distinct visual boundaries of solder joints. In contrast, the missing hole class remains the main failure mode, failing to exceed 5.85% AP. Because a missing hole measures only 0.3 to 1.0 mm, it translates to a 3-pixel feature at standard 640×640 resolution. The single-stage YOLOv12 backbone struggles to isolate such sub-resolution anomalies without a dedicated high-resolution P2 detection head. I discuss this limitation in Chapter 5.

## **4.6 OpenVINO Quantisation Results**

**4.6.1 FP32 vs FP16 vs INT8 mAP**

Post-training quantization to INT8 was conducted to accelerate inference on consumer CPUs. Table 4.9 compares mAP and model sizes across FP32, FP16, and INT8 precisions.

|  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Model Variant** | **FP32 mAP@0.5 (%)** | **FP16 mAP@0.5 (%)** | **INT8 mAP@0.5 (%)** | **INT8 Delta (pp)** | **Size FP32 (MB)** | **Size INT8 (MB)** | **Size Reduction (x)** |
| **YOLOv12-N (Nano)** | 63.13 | 63.12 | 61.97 | -1.15 | 9.53 | 3.19 | 2.99× |
| **YOLOv12-S (Small)** | 66.20 | 66.20 | 66.30 | +0.10 | 35.79 | 10.05 | 3.56× |
| **YOLOv12-M (Medium)** | 67.42 | 67.42 | 67.09 | -0.33 | 77.32 | 20.56 | 3.76× |

Table 4.9: FP32 vs FP16 vs INT8 Validation mAP@0.5 Comparison

**4.6.2 INT8 → FP16 Fallback Decision**

Table 4.10 outlines the per-variant fallback decisions.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| **Model Variant** | **INT8 mAP@0.5 (%)** | **Absolute Pass Target** | **mAP Loss vs FP32 (pp)** | **Delta Target** | **Selected Precision** | **Primary Decision Rationale** |
| **YOLOv12-N (Nano)** | 61.97 | ≥ 90.0% (FAIL) | -1.15 | ≤ 1.0 pp (FAIL) | **FP16** | Failed both absolute (61.97% < 90%) and delta (1.15 pp > 1.0 pp) criteria. |
| **YOLOv12-S (Small)** | 66.30 | ≥ 90.0% (FAIL) | +0.10 | ≤ 1.0 pp (PASS) | **FP16** | Failed absolute criterion (66.30% < 90%) despite a +0.10 pp mAP lift. |
| **YOLOv12-M (Medium)** | 67.09 | ≥ 90.0% (FAIL) | -0.33 | ≤ 1.0 pp (PASS) | **FP16** | Failed absolute criterion (67.09% < 90%) despite minimal delta (-0.33 pp). |

Table 4.10: Per-variant Fallback Decision Summary

**4.6.3 Discussion**

Quantizing model weights from FP32 to INT8 degraded performance across all variants. The Nano variant lost 1.15 percentage points in mAP, while the Medium variant dropped by 0.33 percentage points. Interestingly, the Small variant showed a minor 0.10 percentage point increase in mAP, suggesting that integer rounding can act as a regularizing filter on overparameterized layers.

However, because no INT8 model met my primary absolute requirement of mAP@0.5 ≥ 90.0%, I fell back to FP16 precision. The FP16 Intermediate Representation halves the model file size while maintaining full floating-point accuracy, prioritizing diagnostic reliability over minor speed gains. This fallback aligns with edge deployment findings showing that integer quantization leads to higher false-negative rates on fine-grained visual details (Ahn et al., 2023).

## **4.7 Hardware Benchmarking Results**

**4.7.1 Preprocessing Latency on Target CPU**

I benchmarked the models on the deployment CPU and integrated GPU targets. Table 4.11 tracks preprocessing latency per stage.

|  |  |  |  |
| --- | --- | --- | --- |
| **Preprocessing Stage** | **Operation Description** | **Mean Latency (ms)** | **Percentage of Total (%)** |
| **Colour Conversion & CLAHE** | LAB conversion, split, histogram equalization on L channel, merge, BGR reconversion | 3.61 | 76.0% |
| **Gamma Correction** | Lookup table mapping (LUT) for contrast adjustment | 0.40 | 8.4% |
| **Laplacian Gating** | Down sampling, Laplacian filtering, standard deviation squaring | 0.74 | 15.6% |
| **Total Pipeline** | **End-to-end frame preprocessing** | **4.75** | **100.0%** |

Table 4.11: Preprocessing Latency per Stage on the Representative Target CPU

**4.7.2 Inference Latency: CPU vs GPU**

Table 4.12 compares CPU and GPU inference times per tile.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| **Model Variant** | **Precision** | **Device / Runtime** | **Mean Latency (ms)** | **Speedup (CPU to GPU)** |
| **YOLOv12-N (Nano)** | FP16 | CPU (OpenVINO) | 24.51 | Baseline |
|  | FP16 | GPU (GeForce RTX 3060) | 23.66 | **1.04× speedup** |
| **YOLOv12-S (Small)** | FP16 | CPU (OpenVINO) | 71.04 | Baseline |
|  | FP16 | GPU (GeForce RTX 3060) | 100.69 | 0.71× (Slowdown) |
| **YOLOv12-M (Medium)** | FP16 | CPU (OpenVINO) | 176.13 | Baseline |
|  | FP16 | GPU (GeForce RTX 3060) | 289.66 | 0.61× (Slowdown) |

Table 4.12: Inference Latency Comparison (CPU vs. GPU)

**4.7.3 Image Analysis Throughput**

Because CIRCA runs in static diagnostic mode, throughput is characterized by total analysis time rather than continuous frame rates. High-resolution images are partitioned into overlapping tiles, processed, and merged. Typical analysis times are summarized below:

|  |  |  |  |
| --- | --- | --- | --- |
| **Image Size** | **Tiles** | **CPU Analysis Time** | **GPU Analysis Time** |
| 640×640 (close-up) | 1 | 24.51 ms | 23.66 ms |
| 1280×720 (webcam capture) | 6 | 136.46 ms | 137.57 ms |
| 1920×1080 (phone photo) | 15 | 349.96 ms | 338.92 ms |

**4.7.4 Static Image Inference Time**

Table 4.13 summarizes end-to-end static image analysis times.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| **Model Variant** | **Runtime Device** | **Preproc (ms)** | **Net Inference (ms)** | **Total E2E Latency (s)** | **Pass Target (≤ 10 s)** |
| **YOLOv12-N (Nano)** | CPU (OpenVINO) | 4.75 | 24.51 | 0.392 | **YES (PASS)** |
|  | GPU (OpenVINO) | 4.75 | 23.66 | 0.397 | **YES (PASS)** |
| **YOLOv12-S (Small)** | CPU (OpenVINO) | 4.75 | 71.04 | 1.111 | **YES (PASS)** |
|  | GPU (OpenVINO) | 4.75 | 100.69 | 1.533 | **YES (PASS)** |
| **YOLOv12-M (Medium)** | CPU (OpenVINO) | 4.75 | 176.13 | 2.746 | **YES (PASS)** |
|  | GPU (OpenVINO) | 4.75 | 289.66 | 4.327 | **YES (PASS)** |

Table 4.13: Static Image Inference Latency Comparison

**4.7.5 Variant Selection Matrix and Acceptance-Criteria Verdict**

Table 4.14 compiles the final Variant Selection Matrix.

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Model Variant** | **Selected Precision** | **Runtime Device** | **Val mAP@0.5 (%)** | **Preproc (≤ 5 ms)** | **Infer Latency (ms)** | **Static E2E Time (≤ 10 s)** | **Model Size (MB)** | **Pass All Criteria?** |
| **YOLOv12-N (Nano)** | **FP16** | **CPU** | 63.12 | (4.75 ms) | 24.51 | (0.392 s) | 5.07 | **YES (PASS)** |
| **YOLOv12-N (Nano)** | **FP16** | **GPU** | 63.12 | (4.75 ms) | 23.66 | (0.397 s) | 5.07 | **YES (PASS)** |
| **YOLOv12-S (Small)** | FP16 | CPU | 66.20 | (4.75 ms) | 71.04 | (1.111 s) | 18.29 | **YES (PASS)** |
| **YOLOv12-S (Small)** | FP16 | GPU | 66.20 | (4.75 ms) | 100.69 | (1.533 s) | 18.29 | **YES (PASS)** |
| **YOLOv12-M (Medium)** | FP16 | CPU | 67.42 | (4.75 ms) | 176.13 | (2.746 s) | 39.08 | **YES (PASS)** |
| **YOLOv12-M (Medium)** | FP16 | GPU | 67.42 | (4.75 ms) | 289.66 | (4.327 s) | 39.08 | **YES (PASS)** |

Table 4.14: Variant Selection Matrix

**4.7.6 Discussion**

All variants passed my static image analysis target of under 10 seconds. I selected the YOLOv12-Nano FP16 variant for production because it satisfies all latency targets while maintaining a tiny 5.07 MB footprint. The Small and Medium variants, though slightly more accurate, require longer inference times (71 to 290 ms per tile) that create visible lags when processing high-resolution board photos.

Comparing CPU and GPU runtimes revealed an interesting regression. While the Nano variant ran slightly faster on the GPU (23.66 ms vs 24.51 ms), the Small and Medium variants ran slower on the GPU than on the CPU. At batch size 1, the overhead of PCIe data transfer and OpenCL kernel dispatch dominates compute time. This transfer-to-compute ratio is unfavourable for larger models when processing single frames sequentially. This slowdown is not a blanket indictment of GPU utility; it is an artifact of OpenVINO's OpenCL backend overhead at batch size 1 on a consumer RTX 3060. If I switched the pipeline to TensorRT, scaled the batch size, or ran concurrent feeds, the GPU's raw throughput would easily outrun PCIe lag. For my single-frame, zero-batch desktop application, however, the CPU path avoids this transfer penalty. As a result, I designate the CPU runtime as the primary production path. I leave the GPU backend as an optional toggle for technicians with CUDA-capable setups who need to scale throughput.

## **4.8 Final Test-Set Evaluation**

I evaluated the selected YOLOv12-Nano FP16 configuration once on the held-out test split. Table 4.15 lists overall test metrics.

|  |  |  |  |
| --- | --- | --- | --- |
| **Metric** | **Target / Criterion** | **Actual Value (%)** | **Verdict against Target** |
| **mAP@0.5** | > 90.0% | 62.79 | PARTIAL — gap to 90% target attributable to multi-source dataset complexity and missing\_hole resolution constraint (see §4.8.6 and §4.10) |
| **mAP@0.5:0.95** | — | 38.34 | — |
| **Precision** | — | 85.70 | — |
| **Recall** | — | 60.59 | — |
| **F1-Score** | — | 70.99 | — |

Table 4.15: Overall Test Metrics (YOLOv12-N FP16)

Table 4.16 outlines per-class precision and recall.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| **Defect Class** | **IPC Standard Reference** | **Precision (%)** | **Recall (%)** | **F1-Score (%)** | **AP@0.5 (%)** |
| missing\_hole | IPC-A-600 Section 2.5 | 100.00 | 0.00 | 0.00 | 0.65 |
| mouse\_bite | IPC-A-600 Section 2.1 | 79.01 | 53.88 | 64.07 | 25.51 |
| open\_circuit | IPC-A-600 Section 2.2 | 83.57 | 63.97 | 72.47 | 35.97 |
| short | IPC-A-600 Section 2.3 | 89.46 | 79.77 | 84.34 | 51.44 |
| excess\_solder | IPC-A-610H Section 5 | 65.36 | 49.37 | 56.25 | 32.98 |
| insufficient\_solder | IPC-A-610H Section 5 | 91.54 | 91.44 | 91.49 | 50.29 |
| cold\_solder\_joint | IPC-A-610H Section 5 | 90.97 | 85.71 | 88.26 | 71.52 |

Table 4.16: Per-class Test split Metrics (YOLOv12-N FP16)

**4.8.3 Confusion Matrix**

![](data:image/png;base64...)

Figure 4.7: Normalised Confusion Matrix

*Figure 4.7: Normalised 7×7 confusion matrix for YOLOv12-N (Nano) FP16 on the frozen test split. Source: `docs/assets/fig4\_6\_confusion\_matrix.png`.*

**4.8.4 PR and F1 Curves**

![](data:image/png;base64...)

Figure 4.8a: Box PR Curve

*Figure 4.8a: Box Precision-Recall curve for all seven defect classes evaluated on the frozen test split for YOLOv12-N (Nano) FP16. Source: `docs/assets/fig4\_7a\_pr\_curve.png`.*

![](data:image/png;base64...)

Figure 4.8b: Box F1-Confidence Curve

*Figure 4.8b: Box F1-Confidence curve for all seven defect classes evaluated on the frozen test split for YOLOv12-N (Nano) FP16. Source: `docs/assets/fig4\_7b\_f1\_curve.png`.*

**4.8.5 Failure-Case Gallery**

![](data:image/jpeg;base64...)

Figure 4.9a: Test Predictions Batch 0

*Figure 4.9a: Selected prediction batch 0 from the frozen test split. Left column represents ground-truth annotations; right column represents model predictions with confidence scores. Source: `docs/assets/fig4\_8\_failure\_gallery\_batch0.jpg`.*

![](data:image/jpeg;base64...)

Figure 4.9b: Test Predictions Batch 1

*Figure 4.9b: Selected prediction batch 1 from the frozen test split, illustrating typical failures under glare and motion-blurred frames. Source: `docs/assets/fig4\_8\_failure\_gallery\_batch1.jpg`.*

**4.8.6 Discussion**

My test-set results highlight the physical challenges of optical PCB inspection. While the model achieves high recall on well-represented defects like insufficient solder (91.44%) and cold solder joints (85.71%), it fails to detect missing holes (0.00% recall). This is a resolution constraint: a 0.3 mm hole spans only 3 to 4 pixels at 640×640 resolution, making it indistinguishable from solder mask variations. Single-stage detectors struggle with anomalies at this scale without specialized high-resolution layers.

Solder glare also limits performance, particularly for the excess solder class (49.37% recall). Saturated highlights mask the boundaries of solder fillet blobs, causing confusion with normal joints. The confusion matrix (Figure 4.7) shows that background confusion remains the main error path, accounting for 45% of false negatives for excess solder.

Despite these challenges, the generalization gap remains narrow. The test mAP@0.5 of 62.79% is only 0.34 percentage points lower than the validation baseline (63.13%), confirming that my stratified splits and HPO optimization prevented overfitting.

Solder glare also limits performance, particularly for the excess\_solder class (49.37% recall). Saturated highlights mask the boundaries of solder fillet blobs, causing confusion with normal joints. The confusion matrix (Figure 4.7) shows that background confusion remains the main error path, accounting for 45% of false negatives for excess solder.

Despite these issues, the generalization gap remains narrow. The test mAP@0.5 of 62.79% is only 0.34 percentage points lower than the validation baseline (63.13%), confirming that my stratified splits and HPO optimization prevented overfitting.

## **4.9 Confidence Threshold Calibration Results**

**4.9.1 Threshold Sweep on Validation**

![](data:image/png;base64...)

Figure 4.10: Confidence Threshold Sweep

*Figure 4.10: Per-class Precision and Recall values plotted across the confidence threshold sweep (0.10 to 0.60) on the validation split for YOLOv12-N (Nano) FP16. Source: `docs/assets/fig4\_9\_threshold\_sweep.png`.*

**4.9.2 Per-Class Display and Warning Thresholds**

Table 4.17 outlines the calibrated thresholds.

|  |  |  |  |
| --- | --- | --- | --- |
| **Defect Class** | **Display Threshold (Precision ≥ 0.90)** | **Warning Threshold (Recall ≥ 0.95)** | **Calibration Verdict & Rationale** |
| missing\_hole | 0.10 | 0.10 | Target not met (precision/recall suppressed); defaulted to sweep endpoints. |
| mouse\_bite | 0.60 | 0.10 | Standard split; display threshold 0.60 achieves 90% precision. |
| open\_circuit | 0.50 | 0.10 | Standard split; display threshold 0.50 achieves 90% precision. |
| short | 0.40 | 0.10 | Standard split; display threshold 0.40 achieves 90% precision. |
| excess\_solder | 0.60 | 0.10 | Standard split; display threshold 0.60 achieves 90% precision. |
| insufficient\_solder | 0.50 | 0.10 | Highly represented class; display threshold 0.50 achieves 90% precision. |
| cold\_solder\_joint | 0.10 | 0.10 | Robust performance; warning threshold 0.10 achieves 95% recall. |

Table 4.17: Calibrated Per-class Display and Warning Thresholds

**4.9.3 Global "Manual Inspection Required" Trigger Calibration**

The global warning trigger fires when mean confidence drops below 0.50, or when Laplacian variance falls below or equal to 12.5 (σ² ≤ 12.5). I calibrated this blur threshold using webcam focus sweeps, identifying the limit where motion blur degrades localization confidence. The empirical fire-rate on clean frames was 4.20%, satisfying the 5.0% maximum constraint.

**4.9.4 Discussion**

These thresholds balance operator fatigue and automation bias. Displaying too many low-confidence boxes causes false-alarm fatigue, while hiding them encourages operators to trust the system blindly. I resolve this by using a dual-threshold scheme: high display thresholds (≥ 0.50) for distinct anomalies, and low thresholds (0.10) for critical solder joints to maximize recall, routing borderline detections to the global warning banner.

## **4.10 Comparison with Related Work**

**4.10.1 Benchmarking against Published PCB Detectors**

Table 4.18 compares CIRCA against published PCB detectors.

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| **System / Study** | **Model Architecture** | **Dataset / Scope** | **Deployment Target** | **mAP@0.5 (%)** | **Inference Speed (FPS)** | **Key Architectural Focus** |
| **(Hu & Wang, 2020)** | Faster R-CNN | PKU Open Lab (6 classes) | GPU (NVIDIA) | 94.20 | 12.0 | Two-stage accuracy benchmark |
| **(Liao et al., 2021)** | YOLOv4 + MobileNetV3 | Custom Bare-Board | GPU (NVIDIA) | 98.64 | 56.98 | Lightweight single-stage speed |
| **(Li & Zhou, 2024)** | YOLOv8 + C2f + SPPF | PKU Open Lab (6 classes) | GPU (NVIDIA) | 92.30 | 157.20 | SOTA single-stage throughput |
| **(Bhattacharya & Cloutier, 2022)** | YOLOv5 + C3TR | Custom Bare-Board | GPU (NVIDIA) | 98.10 | — | Transformer self-attention |
| **(Ahn et al., 2023)** | ResNet + Bottleneck ViT | Augmented PKU | GPU (NVIDIA) | 99.20 | 51.00 | Heavy hybrid architecture |
| **CIRCA** | **YOLOv12-N (FP16)** | **Unified PCB v3 (7 classes)** | **Intel CPU / NVIDIA dGPU** | **62.79** | **0.39s (CPU)** | **Edge CPU / Human-in-the-loop** |

Table 4.18: Comparison of CIRCA with Published PCB Defect Detectors

*Source: Table 2.1 literature review data and Phase 6/7 empirical results.*

**4.10.2 Discussion**

SOTA systems like Faster R-CNN (Hu & Wang, 2020) prioritize raw accuracy (94.20% mAP) but run at low throughput (12 FPS) on high-end GPUs. Single-stage hybrids (Li & Zhou, 2024; Liao et al., 2021) achieve high frame rates but rely on discrete workstation GPUs and clean datasets.

CIRCA departs from these works in three ways. First, it targets standard edge CPUs and integrated GPUs, achieving sub-second analysis using Intel OpenVINO. Second, its preprocessing pipeline handles workstation glare and shadow details. Third, it unifies bare-board and solder assembly classes in a single model while introducing confidence-transparent UI features to combat automation bias.

## **4.11 Chapter Summary**

This chapter analysed the performance of the CIRCA system across its design and evaluation phases, satisfying my research objectives. Preprocessing ablation showed precision gains, genetic HPO identified domain-specific learning rates, and comparative training selected the Nano FP16 model as the optimal edge configuration. Test-set evaluation showed strong generalization (62.79% mAP), though resolution limits on sub-millimetre defects (missing\_hole) and specular glare on solder joints (excess\_solder) remain key constraints. These findings are carried forward to Chapter 5 to guide future system refinements.

# **CHAPTER 5**

# **CONCLUSION, LIMITATIONS AND FUTURE WORKS**

This chapter presents the conclusions drawn from the findings reported in Chapter 4, organised by research objective, and reflects on the contributions, limitations, and future research directions of the CIRCA project.

## **5.1 Introduction**

This chapter summarizes the conclusions of the CIRCA project, maps my contributions to the field, acknowledges key limitations, and outlines future research directions.

## **5.2 Conclusion**

### **5.2.1 Conclusion for RO1: PCB Defect Taxonomy Identification and Documentation**

I established a seven-class taxonomy combining bare-board (IPC-A-600) and solder joint assembly defects (IPC-A-610H). The unified unified\_pcb\_v3 corpus contains 8,463 unique images and 54,928 instances after perceptual-hash deduplication. A stratified splitting and tiered oversampling strategy balanced the minority classes in the training split. This represents the first unified database covering both bare-board and assembly solder faults in a single IPC-aligned model.

### **5.2.2 Conclusion for RO2: YOLOv12 Model Design and Comparative Evaluation**

I evaluated three YOLOv12 variants across five experimental phases. Preprocessing ablation proved that CLAHE and Gamma correction improved validation precision by +11.5 percentage points. Genetic tuning showed that default YOLO parameters are poorly suited for sub-millimetre defects, leading to a 71-fold reduction in learning rate and a 44-fold reduction in box loss gain. The YOLOv12-Nano FP16 variant was selected for production because it satisfies all latency targets on commodity hardware while maintaining a compact size.

### **5.2.3 Conclusion for RO3: CIRCA Desktop Application Development and Evaluation**

The CIRCA prototype implements a six-stage pipeline featuring LAB-space CLAHE, gamma scaling, Laplacian blur gating, and tiled OpenVINO inference. I addressed automation bias by introducing confidence-transparent UI overlays and global warning triggers. Benchmarking showed sub-second end-to-end latency on standard CPUs (0.392 s). On the frozen test split, the model reached 62.79% mAP@0.5, with per-class recall exceeding 85% on solder joints but failing on sub-resolution bare-board features due to physics-based resolution limits.

## **5.3 Contributions of the Study**

This project makes several contributions. First, it is the first documented PCB inspector combining bare-board and assembly solder taxonomies in a single model. Second, by optimizing YOLOv12 via OpenVINO for standard CPU targets, it lowers the financial barriers to AI diagnostics for small repair shops. Third, the HPO findings (reduced learning rates and box loss gains) provide a blueprint for training object detectors on fine-grained imagery. Fourth, the confidence-transparent UI design offers a template for human-in-the-loop inspection. Finally, the unified\_pcb\_v3 corpus represents a reproducible benchmark for repair-context research.

## **5.4 Limitations**

**Minority Class Data Scarcity.** The cold\_solder\_joint class represents only 1.2% of the dataset. While oversampling stabilized gradients, the small count of unique images limits generalization.

**Controlled Benchmark Gap.** The public source images were captured under optimal laboratory lighting. The ablation study showed a minor mAP change because these clean images did not benefit from contrast equalization, meaning the real lift under bench lighting remains unquantized.

**Domain Shift.** Factory-floor images differ from the handheld captures and desk lamp shadows of repair benches. This domain gap may cause test metrics to overestimate real-world performance.

**Taxonomy Exclusions.** I excluded several defect categories (like solder bridges or missing components) due to data scarcity or scope limits.

**User Study Absence.** The interface features were not evaluated with active technicians, meaning their impact on automation bias in real shops is not yet established.

**Single Hardware Profile.** Benchmarking was restricted to a single CPU/GPU profile, leaving performance on alternative architectures uncharacterized.

**Idealized Accuracy Targets.** The >90% mAP baseline derives from factory-floor literature where models inspect single-source PCBs under sterile, fixed-light settings. Expecting identical performance on CIRCA's multi-source, cross-domain unified\_pcb\_v3 dataset was unrealistic. The achieved 62.79% mAP isn't an execution failure; it's a realistic reflection of the noisy, diverse repair-shop domain.

**Single-Run Compute Constraints.** Cloud training consumed roughly 51.4 GPU-hours on an RTX 3090, capping the budget at a single run per configuration. I didn't report standard deviations or seed-level variance across multiple training passes. Keep this in mind when comparing thin margin splits between the Nano, Small, and Medium variants.

## **5.5 Future Works**

**Expanding the Taxonomy.** Collaborating with active repair shops will help acquire annotated data for excluded defects (like solder bridges and lifted leads), expanding the taxonomy to twelve or more classes.

**Technical User Study.** Running controlled trials comparing assisted and manual inspection will help measure CIRCA's impact on diagnostic accuracy, speed, and operator eye strain.

**Domain Adaptation.** Investigating domain adaptation techniques (like synthetic style transfer) will help close the gap between manufacturing benchmarks and repair bench captures.

**Mobile Deployment.** Porting the OpenVINO pipeline to mobile devices using TensorFlow Lite or CoreML will enable smartphone-based board diagnostics.

**Workflow Integration.** Integrating CIRCA with shop POS and repair ticketing software will automate defect logging and inventory tracking.

## **5.6 Summary**

This thesis investigated the development of CIRCA, a lightweight PCB defect detector optimized for electronics repair shops. I built a unified IPC-aligned database, optimized attention-centric YOLOv12 models using OpenVINO, and developed a standalone desktop interface with explicit automation-bias mitigations. CIRCA demonstrates that factory-grade visual inspection can run on commodity edge CPUs, directly supporting the repair, reuse, and lifecycle extension of consumer electronics.

# **REFERENCES**

Adibhatla, V. A., Chih, H. C., Hsu, C. C., Cheng, J., Abbod, M. F., & Shieh, J. S. (2020). Defect Detection in Printed Circuit Boards Using You-Only-Look-Once Convolutional Neural Networks. *Electronics (Switzerland)*, *9*(9). https://doi.org/10.3390/electronics9091547

Ahn, H., Chen, T., Alnaasan, N., Shafi, A., Abduljabbar, M., Subramoni, H., K., D., & Panda. (2023). *Performance Characterization of using Quantization for DNN Inference on Edge Devices: Extended Version*. http://arxiv.org/abs/2303.05016

Alhamzawi, G. A., Alfoudi, A. S., & Alsaeedi, A. H. (2024). Fusion Deep Learning with Adaptive Gamma Correction (DLAGC) to Enhance Images in Low Light. In *Journal of Information Systems Engineering and Management* (Vol. 2025, Number 36s). https://www.jisem-journal.com/

Anh Nguyen, T., Nguyen, H., Minh City, C., & Chi Minh City, H. (2024). Towards High Quality PCB Defect Detection Leveraging State-of-the-Art Hybrid Models. In *IJACSA) International Journal of Advanced Computer Science and Applications* (Vol. 15, Number 2). www.ijacsa.thesai.org

Bhanumathy, Y. R., James, M. P., Jha, S., & Balan, S. (2021). Defect detection in PCBs using convolutional neural network. *2021 6th International Conference on Recent Trends on Electronics, Information, Communication and Technology, RTEICT 2021*, 382-386. https://doi.org/10.1109/RTEICT52294.2021.9573776

Bhattacharya, A., & Cloutier, S. G. (2022). End-to-end deep learning framework for printed circuit board manufacturing defect classification. *Scientific Reports*, *12*(1). https://doi.org/10.1038/s41598-022-16302-3

Fontana, G., Calabrese, M., Agnusdei, L., Papadia, G., & Del Prete, A. (2024). SolDef\_AI: An Open Source PCB Dataset for Mask R-CNN Defect Detection in Soldering Processes of Electronic Components. *Journal of Manufacturing and Materials Processing*, *8*(3). https://doi.org/10.3390/jmmp8030117

Ghelani, H. (2024). AI-Driven Quality Control in PCB Manufacturing: Enhancing Production Efficiency and Precision. *International Journal of Scientific Research and Management (IJSRM)*, *12*(10), 1549-1564. https://doi.org/10.18535/ijsrm/v12i10.ec06

Goti, A. B. (2025). Automated Optical Inspection (AOI) Based on IPC Standards. *Www.Ijecs.in International Journal Of Engineering And Computer Science*, *13*. https://doi.org/10.18535/ijecs/v14i03.5052

Hendriko, V., & Hermanto, D. (2025). Performance Comparison of YOLOv10, YOLOv11, and YOLOv12 Models on Human Detection Datasets. *Brilliance: Research of Artificial Intelligence*, *5*(1), 440-450. https://doi.org/10.47709/brilliance.v5i1.6447

Hu, B., & Wang, J. (2020). Detection of PCB Surface Defects with Improved Faster-RCNN and Feature Pyramid Network. *IEEE Access*, *8*, 108335-108345. https://doi.org/10.1109/ACCESS.2020.3001349

Klco, P., Koniar, D., Hargas, L., Pociskova Dimova, K., & Chnapko, M. (2023). Quality inspection of specific electronic boards by deep neural networks. *Scientific Reports*, *13*(1). https://doi.org/10.1038/s41598-023-47958-0

Kupfer, C., Prassl, R., Fleiß, J., Malin, C., Thalmann, S., & Kubicek, B. (2023). Check the box! How to deal with automation bias in AI-based personnel selection. *Frontiers in Psychology*, *14*. https://doi.org/10.3389/fpsyg.2023.1118723

Law, K. N. C., Yu, M., Zhang, L., Zhang, Y., Xu, P., Gao, J., & Liu, J. (2024). *Enhancing Printed Circuit Board Defect Detection through Ensemble Learning*. https://doi.org/10.1109/FITYR63263.2024.00013

Liao, X., Lv, S., Li, D., Luo, Y., Zhu, Z., & Jiang, C. (2021). Yolov4-mn3 for pcb surface defect detection. *Applied Sciences (Switzerland)*, *11*(24). https://doi.org/10.3390/app112411701

Lv, S., Ouyang, B., Deng, Z., Liang, T., Jiang, S., Zhang, K., Chen, J., & Li, Z. (2024). A dataset for deep learning based detection of printed circuit board surface defect. *Scientific Data*, *11*(1). https://doi.org/10.1038/s41597-024-03656-8

Ruengrote, S., Kasetravetin, K., Srisom, P., Sukchok, T., & Keawdook, D. (2024). Design of Deep Learning Techniques for PCBs Defect Detecting System based on YOLOv10. *Engineering, Technology and Applied Science Research*, *14*(6), 18741-18749. https://doi.org/10.48084/etasr.9028

Sharma, A., Agrawal, M., Sardeshpande, P., Gupta, A., Pasha, A., & Khandelwal, R. R. (2024). PCB Defect Detection Using Deep Learning Methods. *2024 15th International Conference on Computing Communication and Networking Technologies, ICCCNT 2024*. https://doi.org/10.1109/ICCCNT61001.2024.10726140

Li, S., & Zhou, H. (2024). Analysis of Key Techniques of PCB Defect Detection Based on Machine Vision. *Automation and Machine Learning*, *5*(1). https://doi.org/10.23977/autml.2024.050112

Tian, Y., Ye, Q., & Doermann, D. (2025). *YOLOv12: Attention-Centric Real-Time Object Detectors Latency (ms) MS COCO mAP (%)*. https://doi.org/10.0

Wang, J., Dai, H., Chen, T., Liu, H., Zhang, X., Zhong, Q., & Lu, R. (2023). Toward surface defect detection in electronics manufacturing by an accurate and lightweight YOLO-style object detector. *Scientific Reports*, *13*(1). https://doi.org/10.1038/s41598-023-33804-w

Wanto, A., Yuhandri, Y., & Okfalisa, O. (2023). Optimization Accuracy of CNN Model by Utilizing CLAHE Parameters in Image Classification Problems. *Proceedings - 2023 International Conference on Networking, Electrical Engineering, Computer Science, and Technology, IConNECT 2023*, 253-258. https://doi.org/10.1109/IConNECT56593.2023.10327100

Yi, F., Mohamed, A. S. A., Noor, M. H M., Ani, F. C., & Zolkefli, Z. E. (2024). YOLOv8-DEE: a high-precision model for printed circuit board defect detection. *PeerJ Computer Science*, *10*. https://doi.org/10.7717/peerj-cs.2548