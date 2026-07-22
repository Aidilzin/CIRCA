UNIVERSITI TEKNOLOGI MARA
CIRCA: CIRCUIT INSPECTION AND
RECOGNITION USING
CONVOLUTIONAL ARCHITECTURES
MUHAMMAD AIDIL AL-FAIZI BIN MOHD ZIN
BACHELOR OF INFORMATION SYSTEMS
(Hons.) INTELLIGENT SYSTEMS
ENGINEERING
JULY 2026

UNIVERSITI TEKNOLOGI MARA
CIRCA: CIRCUIT INSPECTION AND
RECOGNITION USING
CONVOLUTIONAL
ARCHITECTURES
MUHAMMAD AIDIL AL-FAIZI BIN MOHD ZIN
Thesis submitted in fulfilment of the requirements
for Bachelor of Information Systems (Hons.)
Intelligent Systems Engineering
Faculty of Computer Science and Mathematics
July 2026

SUPERVISOR APPROVAL
CIRCA: CIRCUIT INSPECTION AND RECOGNITION USING
CONVOLUTIONAL ARCHITECTURES
By
MUHAMMAD AIDIL AL-FAIZI BIN MOHD ZIN
2023276732
This thesis was prepared under the supervision of the project supervisor, Puan
Farahnatasyah Binti Abdul Hannan. It was submitted to the Faculty of Computer
Science and Mathematics and was accepted in partial fulfilment of the requirements
for degree of Bachelor of Information Systems (Hons.) Intelligent Systems
Engineering.
Approved by,
ààààààààààà.
Puan Farahnatasyah Binti Abdul Hannan
Project Supervisor
JANUARY 23, 2026
ii

STUDENT DECLARATION
I certify that this thesis and the project to which it relates is the product of my own
work and that any idea or quotation from the work of other people, published or
otherwise are fully acknowledged in accordance with the standard referring practices
of the discipline.
ààààààààààà.
MUHAMMAD AIDIL AL-FAIZI BIN MOHD ZIN
2023276732
JULY 9, 2026
iii

ACKNOWLEDGEMENT
Alhamdullilah, praises and thanks to Allah because of His Almighty and His utmost
blessings, I was able to finish this research within the time duration given. First and
foremost, I would like to express my utmost gratitude to my academic supervisor, Puan
Farahnatasyah Binti Abdul Hannan, for her invaluable guidance, constant
encouragement, and insightful feedback throughout the duration of this Final Year
Project. Her expertise and support were crucial to the successful realization of this
study.
I am also deeply grateful to the faculty members of the Bachelor of Information
Technology (Hons.) Intelligent Systems Engineering program at Universiti Teknologi
MARA (UiTM) for providing a strong academic foundation and the computational
facilities necessary to complete my work.
I would like to dedicate a special note of thanks to my father, Mohd Zin Bin Abas, and
my siblings for their continuous financial support, encouragement, and valuable
feedback during the developmental phases of this project. Special appreciation also
goes to my friend, Muhammad Syafiq Iman Bin Haidzir, for his assistance in refining
my research direction and identifying key areas of study. Finally, I would like to
acknowledge all the open-source contributors and researchers who made their datasets
and model code available, directly enabling the benchmarks in this project.
iv

ABSTRACT
Printed Circuit Boards (PCBs) form the physical and electrical foundation of modern
consumer electronics. In electronics repair environments, manual visual inspection
under magnifying lenses is the predominant diagnostic method. However, manual
inspection is slow, subjective, and prone to fatigue-induced escape rates of 10% to
20% on sub-millimeter components. Although automated manufacturing utilizes high-
cost, glare-controlled Automated Optical Inspection (AOI) systems, these platforms
are economically and operationally non-viable for independent repair benches
operating under variable desklamp lighting. This study presents CIRCA (Circuit
Inspection and Recognition using Convolutional Architectures), a lightweight AI-
driven visual inspection system designed specifically for electronics repair
environments. CIRCA unifies bare-board (IPC-A-600) and assembly solder joint (IPC-
A-610H) defect taxonomies into a seven-class schema, trained on the compiled 8,463-
image unified_pcb_v3 corpus. We designed a six-stage inference pipeline containing
LAB-space Contrast Limited Adaptive Histogram Equalization (CLAHE), Gamma
Correction, and Laplacian Variance blur quality gating. To optimize detection on sub-
millimeter defect regions, genetic hyperparameter tuning was conducted, resulting in
a 71-fold reduction in initial learning rate and a 44-fold reduction in box loss weight
relative to YOLO defaults. We trained and evaluated three attention-centric YOLOv12
variants (Nano, Small, Medium) optimized via Intel OpenVINO. The production
YOLOv12-Nano FP16 configuration was selected, achieving an end-to-end latency of
0.392 seconds on a commodity CPU and 62.79% mAP@0.5 on the frozen test split.
We addressed operator automation bias by integrating a dual-threshold confidence-
display scheme paired with a global warning banner. CIRCA demonstrates that
factory-grade, real-time visual inspection is viable on commodity edge processors,
directly lowering capital barriers for independent repair shops and supporting the
circular economy.
v

TABLE OF CONTENTS
CONTENT PAGE
SUPERVISOR APPROVAL .......................................................................... ii
STUDENT DECLARATION ........................................................................ iii
ACKNOWLEDGEMENT ............................................................................ iv
ABSTRACT ................................................................................................. v
TABLE OF CONTENTS .............................................................................. vi
LIST OF FIGURES ..................................................................................... xi
LIST OF TABLES ...................................................................................... xii
LIST OF ABBREVIATIONS ...................................................................... xiii
CHAPTER 1: INTRODUCTION .................................................................. 1
1.1 Research Background ..................................................................... 1
1.2 Problem Statement ......................................................................... 2
1.3 Research Questions ........................................................................ 4
1.4 Research Objectives ....................................................................... 5
1.5 Research Scope .............................................................................. 5
1.6 Significance of Study ...................................................................... 6
1.7 Expected Solution .......................................................................... 7
CHAPTER 2: LITERATURE REVIEW .......................................................20
vi

2.1 PCB Defects and Inspection Challenges ......................................... 20
2.2 Machine Learning and CNN-Based PCB Defect Detection .............. 21
2.3 YOLO-Based PCB Defect Detection .............................................. 22
2.4 Image Preprocessing for Robust Detection ..................................... 23
2.5 Edge Machine Learning Deployment and Model Quantization ........ 25
2.6 Human Factors and Automation Bias in AI-Assisted Inspection ...... 25
2.7 Summary of Related Works .......................................................... 27
CHAPTER 3: RESEARCH METHODOLOGY ............................................30
3.1 Introduction ................................................................................ 30
3.2 Research Framework ................................................................... 30
3.2.1 Overview of Research Phases ................................................. 30
3.2.2 Mapping of Objectives, Activities, and Deliverables ................. 32
3.3 Theoretical Study ......................................................................... 32
3.3.1 Preliminary Study ................................................................. 32
3.3.2 Knowledge Acquisition .......................................................... 33
3.4 Empirical Study ........................................................................... 33
3.4.1 Data Collection ..................................................................... 33
3.4.2 Data Pre-processing ............................................................... 36
3.4.3 Data Analysis ........................................................................ 37
3.5 System Design .............................................................................. 37
3.5.1 CIRCA System Architecture .................................................. 37
3.5.2 Inference Pipeline .................................................................. 38
3.5.3 Confidence Threshold and ôManual Inspection Requiredö Logic
38
vii

3.5.4 Interface Design .................................................................... 38
3.6 System Development .................................................................... 39
3.6.1 Training Engine .................................................................... 39
3.6.2 Hyperparameter Optimisation Algorithm ............................... 39
3.6.3 Model Training Procedure ..................................................... 39
3.6.4 OpenVINO Export and INT8 Quantisation ............................. 40
3.6.5 Confidence Threshold Calibration Procedure .......................... 40
3.7 Experimental Design .................................................................... 40
3.7.1 Phase 1: Vanilla Baseline ........................................................ 41
3.7.2 Phase 2: CIRCA-Aligned Baseline .......................................... 41
3.7.3 Phase 3: Hyperparameter Optimisation .................................. 41
3.7.4 Phase 4: Three-Variant Final Training .................................... 41
3.7.5 Phase 5: Quantisation Validation ............................................ 41
3.7.6 Phase 6: Hardware Benchmarking and Variant Selection ......... 42
3.7.7 Phase 7: Final Test Evaluation and Threshold Calibration ....... 42
3.7.8 Acceptance Criteria ............................................................... 42
3.7.9 Evaluation Metrics ................................................................ 43
3.8 Hardware and Software Specification ............................................ 43
3.8.1 Training Environment ........................................................... 43
3.8.2 Deployment Target ................................................................ 43
3.8.3 Software Stack ...................................................................... 43
3.9 Research Plan .............................................................................. 44
3.10 Summary ................................................................................. 45
CHAPTER 4: RESULTS AND FINDINGS ...................................................46
viii

4.1 Introduction ................................................................................ 46
4.2 Dataset and Defect Taxonomy Results ............................................ 46
4.3 Preprocessing Pipeline Evaluation ................................................. 49
4.4 Hyperparameter Optimisation Results .......................................... 52
4.5 Three-Variant Comparative Training Results ................................. 56
4.6 OpenVINO Quantisation Results .................................................. 59
4.7 Hardware Benchmarking Results .................................................. 61
4.8 Final Test-Set Evaluation .............................................................. 65
4.9 Confidence Threshold Calibration Results ..................................... 73
4.10 Comparison with Related Work ................................................. 76
4.11 Chapter Summary........................................................................ 77
CHAPTER 5: CONCLUSION, LIMITATIONS AND FUTURE WORKS .......78
5.1 Introduction ................................................................................ 78
5.2 Conclusion ................................................................................... 78
5.2.1 Conclusion for RO1: PCB Defect Taxonomy Identification and
Documentation ................................................................................... 78
5.2.2 Conclusion for RO2: YOLOv12 Model Design and Comparative
Evaluation .......................................................................................... 79
5.2.3 Conclusion for RO3: CIRCA Desktop Application Development
and Evaluation.................................................................................... 79
5.3 Contributions of the Study ............................................................ 79
5.4 Limitations .................................................................................. 80
5.5 Future Works .............................................................................. 81
5.6 Summary ..................................................................................... 82
ix

REFERENCES ...........................................................................................83
x

LIST OF FIGURES
FIGURE PAGE
3.1: CIRCA Research Framework 31
4.1: Sample Defect Images 48
4.2: Phase 1 vs Phase 2 Ablation Comparison 50
4.3: HPO Fitness Trajectory 52
4.4: HPO Parameter Scatter Plots 54
4.5a: YOLOv12-Nano Training Curves 56
4.5b: YOLOv12-Small Training Curves 57
4.5c: YOLOv12-Medium Training Curves 57
4.7: Normalised Confusion Matrix 68
4.8a: Box PR Curve 69
4.8b: Box F1-Confidence Curve 70
4.9a: Test Predictions Batch 0 71
4.9b: Test Predictions Batch 1 72
4.10: Confidence Threshold Sweep 74
xi

LIST OF TABLES
TABLE PAGE
2.1 Summary of Related Works on PCB Defect Detection 28
2.2: Preprocessing and Deployment Techniques Underpinning CIRCA Design 28
3.1: Mapping of Research Objectives, Activities, and Deliverables 32
3.2: CIRCA Unified 7-Class IPC Taxonomy 35
3.3: CIRCA Project Timeline and Compute Estimate 44
4.1: Final Class Distribution: `unified_pcb_v3` (8,463 unique images, 54,928
instances) 47
4.2: Train / Validation / Test Split Statistics: `unified_pcb_v3` 48
4.3: Phase 1 vs Phase 2 Validation Metrics (OFAT Ablation, YOLOv12-S, 100
Epochs, `unified_pcb_v3`) 50
4.4: Preprocessing Latency per Stage on the Representative Target CPU 51
4.5: Top-10 HPO Trials Ranked by Fitness 53
4.6: Comparison of Tuned Hyperparameters and YOLOv12 Defaults 55
4.7: Validation Metrics per YOLOv12 Variant (Phase 4) 58
4.8: Per-class Validation mAP@0.5 Breakdown per Variant 58
4.9: FP32 vs FP16 vs INT8 Validation mAP@0.5 Comparison 60
4.10: Per-variant Fallback Decision Summary 60
4.11: Preprocessing Latency per Stage on the Representative Target CPU 62
4.12: Inference Latency Comparison (CPU vs. GPU) 62
4.13: Static Image Inference Latency Comparison 63
4.14: Variant Selection Matrix 64
4.15: Overall Test Metrics (YOLOv12-N FP16) 66
4.16: Per-class Test split Metrics (YOLOv12-N FP16) 67
4.17: Calibrated Per-class Display and Warning Thresholds 75
4.18: Comparison of CIRCA with Published PCB Defect Detectors 76
xii

LIST OF ABBREVIATIONS

| ABBREVIATION  | FULL FORM                                 |             |              |
| ------------- | ----------------------------------------- | ----------- | ------------ |
| AOI           | Automated Optical Inspection              |             |              |
| AP            | Average Precision                         |             |              |
| CPU           | Central Processing Unit                   |             |              |
| CUDA          | Compute Unified Device Architecture       |             |              |
| dHash         | Difference Hashing                        |             |              |
| DFL           | Distribution Focal Loss                   |             |              |
| dGPU          | Discrete Graphics Processing Unit         |             |              |
| E2E           | End-to-End                                |             |              |
| FN            | False Negative                            |             |              |
| FP            | False Positive                            |             |              |
| FP16          | 16-bit Floating Point (Half Precision)    |             |              |
| FP32          | 32-bit Floating Point (Single Precision)  |             |              |
| FPS           | Frames Per Second                         |             |              |
| HPO           | Hyperparameter Optimisation               |             |              |
| INT8          | 8-bit Integer (Quantized Precision)       |             |              |
| IPC           | Association                               | Connecting  | Electronics  |
Industries
| IR  | Intermediate  |     | Representation  |
| --- | ------------- | --- | --------------- |
(OpenVINO format)
| mAP   | Mean Average Precision  |          |              |
| ----- | ----------------------- | -------- | ------------ |
| NNCF  | Neural                  | Network  | Compression  |
Framework
| NMS       | Non-Maximum Suppression  |            |              |
| --------- | ------------------------ | ---------- | ------------ |
| OFAT      | One-Factor-at-a-Time     |            |              |
| OpenVINO  | Open  Visual             | Inference  | and  Neural  |
Network Optimization
| PCB  | Printed Circuit Board  |     |     |
| ---- | ---------------------- | --- | --- |
| POS  | Point of Sale          |     |     |
| PR   | Precision-Recall       |     |     |
xiii

RAM Random Access Memory
RO Research Objective
SOTA State of the Art
TP True Positive
UiTM Universiti Teknologi MARA
VRAM Video Random Access Memory
YOLO You Only Look Once
xiv

CHAPTER 1
INTRODUCTION
This chapter presents the background of the study, including the context of automated
PCB defect detection in electronics repair environments, the problem that motivated
this research, and an outline of the CIRCA system as the proposed solution.
1.1 Research Background
Printed Circuit Boards are the foundation of most modern-day consumer
electronics, providing the essential physical and electrical backbone for the
devicesÆ operation and enabling the power of high-density multi-layered traces
of sub-millimetre widths that transfer electrical signals (Wang et al., 2023).
Hence, minor PCB trace defects could have a significant impact on the overall
safety, reliability, and longevity of the finished product (Klco et al., 2023).
Human inspection, though, is unable to detect such flaws. Though previously
used, microscopic manual inspection suffers from low efficiency, potential for
human error, and up to 20% error rates due to shift-related fatigue (Law et al.,
2024). Computer vision-powered systems, on the other hand, are able to
automatically perform such PCB trace defect classifications, utilising deep
learning algorithms (Bhattacharya & Cloutier, 2022). The application of such
computer vision systems, in practice, requires a trade-off between the accuracy
of visual inspection and the performance of the algorithms used on the relatively
low computational capacity of the end-consumer devices.
The electronics assembly process is a high-precision art due to the large variety
of potential defects that render devices faulty or non-functional and the
1

considerable losses associated with such quality control failures, particularly in
the medical and automotive fields (Klco et al., 2023). It makes manual
microscopic inspection of solder connections and their reliability a key step in
the manufacturing process. Nevertheless, with the tendency to miniaturization,
and the increasing complexity of SMD components (Li & Zhou, 2024), the
ability to visually inspect the solder joints for cracks, bridges, or other issues
becomes more challenging for humans, especially under the effects of shift-
induced fatigue (Bhanumathy et al., 2021). Therefore, human inspection has a
relatively high error rate (10-20%) due to visual acuity and fatigue-related issues
(Law et al., 2024). Convolutional Neural Network (CNN)-based object detection
algorithms and single-stage successors, such as YOLO, ResNet, and
EfficientDet, have seen widespread use in industrial settings for optical PCB
inspections (Liao et al., 2021). It is primarily attributed to their high versatility
and relatively simple training process compared to other methods.
It should also be noted, however, that most current research tends to ignore the
limitations of low-end edge devices for computer vision applications. Thus,
while CNN-based systems offer state-of-the-art performance in object detection,
their performance characteristics suffer when deployed on low-power devices
due to the complexity of their architectures. On the other hand, edge devices
tend to be unable to utilise highly accurate, but computationally intensive
models (Liao et al., 2021). Therefore, there is a need for research into relatively
lightweight yet still relatively accurate computer vision models for automatic
PCB trace inspection. It should also be considered that, in the context of
consumer electronics, accuracy has to be prioritized over performance (Liao et
al., 2021).
1.2 Problem Statement
Modern electronics repair workshops are confronted by the challenge of
ensuring the quality of electrical appliancesÆ further functioning after their
partial disassembling and component level repair. The problem is aggravated by
the current tendency to miniaturize electronic devices that makes the manual
2

inspection of printed circuit boards (PCBs) ineffective and insufficient
(Adibhatla et al., 2020).
Such inspections require good eyesight, and visual acuity decreases significantly
when working with small-scale devices (Goti, 2025). Moreover, manual
inspection contradicts the requirement of speed since visual PCB inspections
take from 10 to 30 minutes to be done (Goti, 2025). In addition, this method is
inaccurate, since it is limited by existing industrial visual standards, namely IPC-
A-610 (Goti, 2025). Even though the standard defines the criteria solder fillet
visibility, IPC-A-610 visual inspection allows for relatively high manual
inspection error rates (between 10% and 15%) (Goti, 2025). Automated optical
inspection tools following IPC-A-610 standards demonstrate better results with
a 98%-99% inspection accuracy rate and a several-thousand-component per
hour processing speed (Goti, 2025). Visual inspection accuracy is also affected
by a workerÆs individual characteristics. For instance, the ability to detect minor
PCB defects is in direct connection with the years of labour experience, and the
inspection quality varies significantly from inspector to inspector (Sharma et al.,
2024). Another limitation is the inspectorÆs physical condition, namely such
prolonged visual inspection at a workstation leads to significant visual fatigue
and neck strain (Goti, 2025).
Furthermore, visual inspection results depend considerably on individual
characteristics, such as the inspectorÆs level of skill and expertise (Sharma et al.,
2024). For instance, relatively inexperienced repair technicians might fail to
identify cold solder joints and some other types of PCB defects that can be easily
detected by an experienced board-level repair specialist. This leads to the
development of low-quality PCB repair processes and, consequently, client
dissatisfaction and returns.
Some minor surface-level PCB defects could be masked or confused with other
PCB anomalies, which makes their identification and classification challenging
(Hu & Wang, 2020). Additionally, visual inspection has little value in case of the
suspicion of more involved structural or mechanical anomalies (Hu & Wang,
2020).
3

A visual inspection procedure of the PCB surface for minor anomalies is a
laborious process requiring a lot of time and efforts from the inspector (Sharma
et al., 2024). Such inspection time requirements limit the productivity of the
entire electronics repair process. During the prolonged observation of a PCB
under a magnifier, the visual acuity of the electronics repair specialist decreases,
which leads to higher chances of inspection errors (Goti, 2025).
Modern desktop automated optical inspection (AOI) systems possess a set of
limitations that prevent their usage in electronics repair workshops.
Contemporary AOI systems are expensive and usually designed for a specific
application, which reduces the flexibility of the electronics repair process
(Huang et al., 2023). AOI systems require additional configuration and
programming according to particular PCBs CAD data (Huang et al., 2023). Thus,
they are ill-suited for electronics repair workshops due to the necessity to
configure them for a wide variety of PCBs within the daily workload (Ruengrote
et al., 2024).
The key research question: How can a vision-based inspection system for rapid
and accurate PCB defect detection be designed that will operate on readily
available desktop PCs in the electronics repair workshops, thereby overcoming
the aforementioned limitations?
1.3 Research Questions
To guide us through the research, we pose three research questions:
ò RQ1: How can we devise a seven-class defect classification scheme that
was validated across both the bare-board IPC-A-600 and assembly-stage
IPC-A-610H standards while being supported by publicly available data
sets?
ò RQ2: What architectural modifications allow YOLOv12 object detectors,
when combined with OpenVINO INT8 optimization and an OpenCV
preprocessing pipeline, to maintain acceptable performance on both low-
cost CPUs and GPUs under varying illumination conditions?
4

ò RQ3: How can we design a desktop user interface that would let the above
pipeline operate as an integrated part of a repair technician's workflow
without inducing cognitive load or automation bias?
1.4 Research Objectives
This study tackles the posed questions by defining the following three
objectives:
ò RO1: Establish a 7-class taxonomy of common PCB defects including four
bare-board categories (missing hole, mouse bite, open circuit, short) and
three solder categories (excess, insufficient, cold joint) using publicly
available data.
ò RO2: Optimize and evaluate YOLOv12 object detectors (Nano, Small,
Medium variants) converted to OpenVINO IR format identifying the most
suitable model for edge deployment.
ò RO3: Develop the CIRCA desktop application featuring a static inspection
user interface, active OpenCV pre-processing pipeline (CLAHE, gamma,
Laplacian variance gating) and a confidence-threshold overlay visualization.
1.5 Research Scope
Technical Scope. We focus on surface-visible defects detected via standard
optical cameras and convolutional neural networks, using a unified seven-class
taxonomy of bare-board (missing hole, mouse bite, open circuit, short) and
assembly defects (excess solder, insufficient solder, cold solder joint). This
selection was driven by a data-availability audit of CC BY 4.0 public datasets.
Only classes with at least 400 instances from multiple sources were kept. Sparse
or visually ambiguous classes like spur, spurious copper, scratch, and pinhole
were excluded to prevent model overfitting. Solder bridges were also excluded
due to a lack of high-quality public annotations. The software implementation
is restricted to YOLOv12 models optimized using Intel OpenVINO. We don't
5

cover electrical testing, thermal imaging, or internal defects requiring X-ray
diagnostics.
Device Scope. This study targets PCBs from consumer electronicsùspecifically
smartphones, laptops, and tabletsùcommonly processed in independent repair
shops. Specialized medical, aerospace, or automotive control boards are outside
the scope.
Defect Scope. The models analyse physical, surface-level manufacturing faults
and wear-related damage. Intermittent electrical faults, software bugs, or trace
issues that don't manifest visually are excluded.
Operational Scope. CIRCA is designed for Windows 10/11 desktops equipped
with standard consumer Intel CPUs and integrated GPUs. Input images are
captured via USB webcams or uploaded directly from disk. Performance targets
are set at >90% mAP@0.5 on the dataset, sub-5 ms preprocessing times, and a
sub-10-second end-to-end diagnosis latency on standard Intel i5 processors (Yi
et al., 2024).
Limitations and Exclusions. The prototype is a diagnostic assistant, not a fully
automated repair machine. It provides confidence scores to reduce automation
bias, but final repair decisions and quality checks remain in the hands of the
technician (Kupfer et al., 2023).
1.6 Significance of Study
Technical Scope. The paper focuses on the computer vision task of detecting
surface-breaking defects in printed circuit boards (PCBs). Detection is
performed using standard optical cameras and convolutional neural networks.
The problem is classified into seven classes of bare-board defects (missing hole,
mouse bite, open circuit, short) and assembly defects (excess solder, insufficient
solder, cold solder joint). Classification was determined by the availability of
data from CC BY 4.0 repositories. Only the classes that had more than 400
instances across different data sets were retained for training while spares, spur,
spurious copper, scratch, and pinhole were excluded due to insufficient data for
6

reliable generalization. Additionally, solder bridge was excluded since there
were no public data sets that could support reliable training and validation of
neural networks. The software scope of the study is limited to YOLOv12
computer vision models optimized in Intel OpenVINO. Electrical tests and
internal PCB inspections with thermal cameras or X-ray machines are outside
the scope of this study.
Device Scope. The study focuses on consumer electronics. The target devices
are smartphones, laptops, and tablets that are repaired in independent repair
shops. Specialized medical, aviation, and automotive electronics are outside the
scope of this study.
Defect Scope. The study focuses on physical defects that can be seen or
measured on a PCB surface or inside the board. Intermittent electrical faults and
software faults are outside the scope of this study.
Operational Scope. CIRCA is designed for Windows 10/11 computers that have
standard Intel CPUs/GPUs. Images can be imported as a video stream through
a USB port or added as files. The expected performance includes more than 90%
mAP@0.5 on the given data set, less than 5 ms of image preprocessing, and less
than 10 seconds for the complete diagnostic cycle on a standard Intel i5
processor (Yi et al., 2024).
Limitations and Exclusions. The scope of this study does not include the
development of an autonomous robotic system. The findings should not be used
alone for PCB repair but should be employed as suggestions with the final
approval from a specialist (Kupfer et al., 2023).
1.7 Expected Solution
Standardizing and automatization of the process of PCB defect detection have
to be implemented to ensure diagnosis without mistakes, reduce eye strain, and
achieve high-quality repairs. The CIRCA system utilizes a YOLOv12 computer
vision model wrapped into the OpenVINO inference engine with a dynamic
7

OpenCV preprocessing pipeline for the repair bench lighting conditions.
Chapter 2 describes the literature on the topic of PCB machine vision, image
processing, edge model deployment, and human-AI collaboration principles that
were used as the basis for the CIRCA design.
8

CHAPTER 2
LITERATURE REVIEW
This chapter reviews the literature supporting the CIRCA project, covering automated
defect detection, deep learning object detection architectures, image preprocessing,
edge machine learning deployment, and human-factors engineering. We organize the
review from general concepts to specific implementation details. We start by discussing
PCB defects and the limitations of traditional machine vision, trace the evolution of
CNN and single-stage YOLO detectors, analyse edge optimization frameworks, and
examine how user interface design can mitigate operator automation bias.
2.1 PCB Defects and Inspection Challenges
Circuit PCAs defects cannot be detected automatically. Many electronic
manufacturers have explored computer vision algorithms for optical inspection
since the early 2000s (Adibhatla et al., 2020; Bhattacharya & Cloutier, 2022).
The root reasons for visual flaws range from production processes to thermal
stresses and mechanical damage. Bridges, opens, cold joints, missing
components, cracks, and burns are some of the most common (Liao et al., 2021).
There are two established norms for standardizing the criteria for visual
inspection (Goti, 2025). IPC-A-600 specifies the requirements for bare-board
defects, including ômouse bites,ö ôopens,ö missing holes, and ôshortsö (IPC
International, 2020). At the same time, IPC-A-610 defines assembly-related
issues, such as fillet, solder, placement, and cleanliness (Goti, 2025). Those
standards, in particular the definition of terms, provide the basis for developing
a concept taxonomy. Indeed, the model classes should be aligned with the terms
used by technicians to describe particular anomalies (Klco et al., 2023). AOI
20

systems following the standard have a 98-99 percent accuracy rate in automated
manufacturing (Goti, 2025). Thus, there is a strong motivation to aim for similar
performance.
The traditional approaches to machine vision relied on template-matching
procedures, morphological transformations, and thresholding compared to a
reference (ôgoldenö) board. However, such simplistic computer vision software
is not capable of dealing with variations in illumination, component layout, and
noise (Sharma et al., 2024). Additionally, template-matching is not compatible
with the diverse set of PCBs configurations utilized in repair shops. The
availability of those datasets is reflected in the open-access resources used for
the benchmarking. The majority of them originate from the production
environment, implying that the images do not include the variety of lighting
conditions found in repair benches. PKU-Market-PCB, for example, contains
3,726 defect-containing PCBs, 14,865 clean boards, and 1,928 fractured boards
(Lv et al., 2024). However, the recent DsPCBSD+ dataset with 10,259 and
20,276 image-label pairs offers a more considerable number of bare-board
samples (Lv et al., 2024). On the other hand, there is an issue of data scarcity
for particular classes. IPC standards list dozens of potential issues, but only
seven of them have enough instances (more than 400 labels from various
sources) to train an accurate neural network. This limitation partially explains
the choice of the classes described in Chapter 1, Section 1.5.
2.2 Machine Learning and CNN-Based PCB Defect Detection
Traditional heuristical approaches are inefficient in identifying specific circuit
board irregularities. The use of CNN-based feature representation is
significantly more effective compared to traditionally designed filters
(Bhattacharya & Cloutier, 2022). The first CNN research paper revealed that it
was possible to achieve approximately 70% classification accuracy on four-layer
CNNs using small-sized data sets, thus proving the ground-breaking nature of
the technology (Adibhatla et al., 2020).
21

The use of Two-Stage detectors with Faster R-CNN and ResNet50 Feature
Pyramid Networks and Guided Anchoring RPN has allowed researchers to
achieve 94.2% mAP on PKU Open Lab data sets (Hu & Wang, 2020). The
calculation speed of such networks is too slow for practical implementation, as
their calculation speed is only 12 FPS. Such a result is insufficient to run the
neural network in real-time on a conventional computer, thus rendering it
unviable for the majority of practical applications. The use of Ensemble
detectors such as EfficientDet, MobileNet-SSD, Faster R-CNN, and YOLOv5
allows achieving up to 95% accuracy in PCB inspection while being resistant to
image distortions (Law et al., 2024).
However, the use of Ensemble detection leads to a significant slowdown in the
inference process due to the simultaneous use of four models. While it may be
possible to achieve a high level of accuracy, the time it takes to perform the
calculation is too long to be useful in practice for rapid diagnosis of PCB defects.
In order to achieve sufficient efficiency, it is necessary to use a Single Stage
Detector with an optimization method.
2.3 YOLO-Based PCB Defect Detection
Single-stage YOLO detectors can be used to resolve the trade-off between the
speed of inference on the edge and its accuracy. Initial trials demonstrated that
Tiny-YOLOv2 models under 50 MB can achieve accuracy as high as 98.79% in
controlled industrial conditions, thus proving the effectiveness of reduced model
size (Adibhatla et al., 2020). Replacing the backbone architecture in YOLOv4
with MobileNetV3 slashed down the latency to 56.98 FPS with a still impressive
accuracy of 98.64% mAP, enabling its usage in CPU applications (Liao et al.,
2021). Meanwhile, the addition of transformer-style attention layers into the
YOLOv5 Neck via the C3TR module allowed the model to better localize low-
level defects in a copper plate (Bhattacharya & Cloutier, 2022).
Attention-based models demonstrate particular effectiveness in dense board
layouts. For example, ATT-YOLO, which is based on the YOLOv5 framework
22

with the addition of self-attention mechanisms, achieved accuracy as high as
92.8% mAP at 111 FPS on the laptop board dataset (Wang et al., 2023). In
another approach, YOLOv8 with C2f blocks demonstrated its effectiveness in
rapid inspection, achieving 157.2 FPS at 92.3% mAP on the PKU benchmark
(Li & Zhou, 2024). Recent developments allow YOLOv10 to maintain high
accuracy on standard central processing units (CPUs) without the need for
graphics processing units (GPUs) for intensive calculations (Ruengrote et al.,
2024). In turn, YOLOv8-DEE utilizes an extended set of multi-scale feature
extractors to enable the detection of sub-millimetre defects (Yi et al., 2024).
YOLOv12, which has been described recently, contains a novel Area Attention
module (A2) that reduces memory consumption compared to standard self-
attention mechanisms, while the Residual Efficient Layer Aggregation
Networks (R-ELAN) facilitate the gradient flow and reduce the complexity of
the model compared to previous YOLO iterations (Tian et al., 2025). Thus, the
Nano version of YOLOv12 achieved accuracy as high as 40.6% mAP on the MS
COCO dataset at competitive inference speeds with earlier versions of the
software (Tian et al., 2025). In a recent comparison of different vision models
(Hendriko & Hermanto, 2025), YOLOv12 demonstrated its effectiveness across
various benchmarks. Specifically, the model achieved strong results in complex
scenes with a high concentration of objects, demonstrating high accuracy and
precision. CIRCA scores utilize this property of YOLOv12 to compare different
versions of the model (Nano, Small, Medium) to select the version that
demonstrates the best performance-to-cost ratio in a given local CPU
environment.
2.4 Image Preprocessing for Robust Detection
To run models in variable and unpredictable conditions, preprocessors were
created that suppress the effects of changing lighting and glare. Three algorithms
were used to accomplish this task: Contrast Limited Adaptive Histogram
Equalization, Gamma Correction, and Laplacian Variance blur check. CLAHE
was used to reduce shadows and specular highlights from solder jointsÆ surfaces
23

(Wanto et al., 2023). This algorithm was chosen due to its ability to divide an
image into small tiles and perform contrast equalization on each, accentuating
fine image detail while not oversaturating flat image regions, allowing for better
defect detection by the following model. In turn, Gamma Correction, a family
of nonlinear contrast corrections, was used to improve the luminance of
underexposed images (Alhamzawi et al., 2024). The deep learning-powered
adaptive gamma model demonstrated its effectiveness on standard test sets with
a high accuracy rate, outperforming fixed gamma correction parameters. CIRCA
implements it to restore details in shadowed images blocked by the work desk
lamp. Finally, Laplacian Variance was applied as a motion blur detection test.
Blurred or out-of-focus images result in significantly lower detection accuracy
(Lv et al., 2024). The algorithm calculates the Laplacian operatorÆs variance,
which, as a quick and parameter-free quality metric, can reject blurry images,
eliminating the need for subsequent CPU time.
24

2.5 Edge Machine Learning Deployment and Model Quantization
The use of neural models on everyday PCÆs without the most expensive and
powerful GPUÆs requires some optimisation of the framework. IntelÆs
OpenVINO toolkit allows for reducing the size of 32-bit floating-point (FP32)
weights of standard models by 4 times, thus accelerating their work on less
powerful computers. This is achieved by approximating the weights with 8-bit
integers (INT8) in the process of post-training optimization. As a result, an
optimised mixed-precision network is obtained that is able to perform object
detection tasks faster. According to the results of the research conducted by Ahn
et al., 2023, networks optimised in OpenVINO Toolkit were proven to be 3.3
times faster than their FP32 counterparts with only a slight decrease in model
accuracy. The choice of OpenVINO Toolkit was also justified by the
comparative analysis of various runtimes, which showed its superior
performance over alternatives such as TensorFlow Lite or PyTorch Mobile on
Intel Processors. In case of insufficient accuracy of the model with INT8
quantization, there is an option to use the FP16 precision of the OpenVINO
Toolkit, which will still provide acceleration but to a lesser extent than using
INT8 weights. This decision is appropriate since mAP of the model has to be
held above 90% during training.
2.6 Human Factors and Automation Bias in AI-Assisted Inspection
AI assistants create human-factors issues, the most significant of which is
automation bias. Operators tend to place blind trust in the decisions made by
automated systems, even when these recommendations are clearly wrong
(Kupfer et al., 2023). In the context of the inspection of printed circuit boards,
this means that the operator would trust the classifications made by CIRCA
without questioning, which could lead to the use of faulty components or a fire
hazard.
25

The only way to reduce the safety risks posed by automation bias is to make the
operators aware of the possibility of errors in the recommendations of the AI
assistant (Kupfer et al., 2023). This can be achieved by providing the operators
with the possibility to review the decisions of the AI and present their own
recommendations. The interface of CIRCA has been designed with this principle
in mind. Firstly, the confidence scores of the model are presented to the operator
in the form of percentages in the bounding boxes. Secondly, if the average
confidence score falls below fifty percent, a warning banner appears at the top
of the interface, reminding the operator to remain engaged.
26

2.7  Summary of Related Works

| Author and  | Model /    | Dataset  | Key Results  | Limitation   |
| ----------- | ---------- | -------- | ------------ | ------------ |
| Year        | Technique  |          |              | Relevant to  |
CIRCA
(Adibhatla et  Tiny-YOLOv2  Industrial AOI  98.79%  Higher localisation
al., 2020)
|     |     | (11,000  | accuracy, under  | error on small  |
| --- | --- | -------- | ---------------- | --------------- |
|     |     | images)  | 50 MB            | defects         |
(Hu & Wang,  Faster R-CNN  PKU Open Lab  94.2% mAP,  Computationally
| 2020)  |              |     | approx. 12 FPS  |              |
| ------ | ------------ | --- | --------------- | ------------ |
|        | + ResNet50-  |     |                 | heavy; slow  |
|        | FPN + GARPN  |     |                 | inference    |
(Liao et al.,  YOLOv4 +  Custom PCB  98.64% mAP,  Requires large,
| 2021)  | MobileNetV3  | dataset  | 56.98 FPS  | labelled dataset  |
| ------ | ------------ | -------- | ---------- | ----------------- |
98.1% mAP
| (Bhattacharya  | YOLOv5 +      | Custom PCB  |     | Sensitive to         |
| -------------- | ------------- | ----------- | --- | -------------------- |
| & Cloutier,    | C3TR          | dataset     |     | distribution shifts  |
| 2022)          | (Transformer  |             |     |                      |
neck)
(Wang et al.,  ATT-YOLO  LCFC Laptop  92.8% mAP,  Dataset annotation
| 2023)  | (YOLOv5 +        | (14,478   | 111 FPS  | noise  |
| ------ | ---------------- | --------- | -------- | ------ |
|        | self-attention)  | defects)  |          |        |
YOLOv8n
(Klco et al.,  Power module  96.6% mAP, 90  Context-specific to
| 2023)  |     |     | ms inference  | power modules  |
| ------ | --- | --- | ------------- | -------------- |
PCB (640╫640
tiles)
(Yi et al., 2024)  YOLOv8-DEE  PCB  State-of-the-art  Evaluated on
|     |     | benchmark  | multi-scale   | benchmark, not    |
| --- | --- | ---------- | ------------- | ----------------- |
|     |     | datasets   | small-defect  | repair scenarios  |
detection
(Law et al.,  Ensemble  Mixed PCB  95% accuracy,  High total inference
| 2024)  |     | dataset  | 80.3% mAP  | time  |
| ------ | --- | -------- | ---------- | ----- |
(EfficientDet +
SSD + RCNN
+ YOLOv5)
(Lv et al.,  Dataset paper  DsPCBSD+  Large-scale  Bare board only; no
2024)  (DsPCBSD+)  (10,259 images,  standardised  assembly solder
|     |     | 9 classes, CC  | bare-board       | defects  |
| --- | --- | -------------- | ---------------- | -------- |
|     |     | BY 4.0)        | dataset; 20,276  |          |
manual
annotations
(Li & Zhou,  YOLOv8 + C2f  PKU Open Lab  92.3% mAP,  Single-model
| 2024)  | + SPPF  |     | 157.2 FPS  |     |
| ------ | ------- | --- | ---------- | --- |
robustness not fully
assessed
(Ruengrote et  YOLOv10  PCB defect  Viable on  Accuracy not yet at
| al., 2024)  | variants  | dataset  | standard  | YOLOv12 level  |
| ----------- | --------- | -------- | --------- | -------------- |
27

hardware
without GPU
(Anh Nguyen ResNet + Augmented 99.2% mAP, 51 41.0 GFLOPs: high
et al., 2024) Bottleneck ViT PKU dataset FPS compute cost
+ FFEM +
Wise-IoU
(Tian et al., YOLOv12 (A2 MS COCO YOLOv12-N: Evaluated on
2025) + R-ELAN) 40.6% mAP; general detection,
competitive not PCB-specific
latency
(Hendriko & YOLOv10 vs. 8 datasets incl. YOLOv12 best: Human detection
Hermanto, YOLOv11 vs. MOT17 0.909 precision, datasets; not PCB-
2025) YOLOv12 0.880 specific
mAP@50
Table 2.1 Summary of Related Works on PCB Defect Detection
Technique Key Study Finding Relevant to CIRCA
CLAHE (Wanto et al., 2023) Increases CNN accuracy by boosting
contrast and eliminating luminance
imbalance in defect images
Adaptive Gamma (Alhamzawi et al., PSNR 17.386, SSIM 0.788 on low-light
Correction 2024) benchmarks; lifts shadow regions
without over-exposing highlights
Laplacian Variance Blur (Lv et al., 2024) PCB-specific evidence that motion blur
Detection degrades model performance; justifies
frame-quality gating
OpenVINO INT8 (Ahn et al., 2023) 3.3╫ inference speedup over FP32 on
Quantization Intel CPUs with minimal accuracy
degradation
Table 2.2: Preprocessing and Deployment Techniques Underpinning CIRCA Design
2.8 Summary
Single-stage YOLO models are superior in terms of a balance between speed
and accuracy in the context of edge deployment (Li & Zhou, 2024; Tian et al.,
2025). Although existing literature on computer vision and industrial inspection
28

provides abundant insights into the performance of CLAHE, OpenVINO
acceleration, and confidence transparency, the majority of the works tackle the
task in the realm of high-volume manufacturing with little attention paid to the
specific peculiarities of small-batch electronics production (Lv et al., 2024;
Ruengrote et al., 2024). Besides, current public data sets lack taxonomy
completeness in terms of IPC-specified categories of manufacturing defects,
which posed a challenge for our own taxonomy design; thus, CIRCA aims to fill
this gap. The chapter ahead describes the process of fine-tuning attention-based
YOLOv12 model on the standard PC workstation. Note that the experimental
framework supporting this endeavour is detailed in Chapter 3.
29

CHAPTER 3
RESEARCH METHODOLOGY
3.1 Introduction
Reproducing deep learning experiments calls for transparency in terms of step-
by-step procedure description. This chapter will present details of every stage of
the CIRCA project, including the data gathering process, model training and
hyperparameter optimization, OpenVINO quantization, and hardware
performance benchmarking at inference stage. For each step, all parameters,
constraints, and decision-making rules will be described in detail.
3.2 Research Framework
3.2.1 Overview of Research Phases
The project was organized into eight sequential stages, with the initial stage
(Phase 0) involving the collection of the raw data set and the final stage (Phase
7) comprising the test-set validation and confidence threshold selection. The
work relates to the three research objectives. Specifically, Phase 0 was
completed to meet the requirements of Research Objective 1, including the
seven-class defect data set and unified data set creation. Research Objectives 2
was met by implementing three YOLOv12 variations and optimizing their
OpenVINO Intermediate Representation (IR). Finally, Research Objective 3
30

was addressed by performing the prototype testing and analysing the hardware
acceptance criteria.
Figure 3.1: CIRCA Research Framework
31

3.2.2  Mapping of Objectives, Activities, and Deliverables

We structured activities to ensure each phase produces a tangible research
deliverable. Table 3.1 maps the objectives to their corresponding phases, key
tasks, and output files.
| Objective  | Phase(s)  | Key Activities  | Deliverables  |
| ---------- | --------- | --------------- | ------------- |
Phase 0
| RO1: Identify and   |     | Literature analysis of  | CIRCA_CLASS_MA     |
| ------------------- | --- | ----------------------- | ------------------ |
| document IPC-A-600  |     | IPC-A-600 / IPC-A-      | PPING.md;          |
| and IPC-A-610-      |     | 610; selection of       | data.yaml; defect  |
taxonomy table
| aligned PCB defect  |     | public datasets; class  |     |
| ------------------- | --- | ----------------------- | --- |
| types               |     | remapping to a          |     |
unified 7-class
taxonomy
Phases 1û5
| RO2: Design and  |     | Vanilla baseline   | runs/detect/CIRCA_ |
| ---------------- | --- | ------------------ | ------------------ |
| compare YOLOv12- |     | training; CIRCA-   | V12{N,S,M}_*/weig  |
| N/S/M with       |     | aligned baseline;  | hts/best.pt;       |
OpenVINO INT8
|     |     | genetic              | OpenVINO IR          |
| --- | --- | -------------------- | -------------------- |
|     |     | hyperparameter       | exports;             |
|     |     | optimisation; three- | quantization_report. |
md
variant final training;
FP32/FP16/INT8
quantisation
validation
Phases 6û7
| RO3: Develop and     |     | Hardware               | benchmark_report.m      |
| -------------------- | --- | ---------------------- | ----------------------- |
| evaluate the CIRCA   |     | benchmarking on        | d;                      |
| desktop application  |     | Intel CPU /            | circa_thresholds.yam    |
|                      |     | integrated GPU;        | l; test_evaluation.md;  |
|                      |     | static image analysis  | CIRCA desktop           |
|                      |     | time; confidence       | prototype               |
threshold calibration;
test-set evaluation;
UI integration
Table 3.1: Mapping of Research Objectives, Activities, and Deliverables

3.3  Theoretical Study

3.3.1  Preliminary Study

32

The initial survey revealed a significant research gap, primarily related to the
literatureÆs emphasis on controlled environments, such as production lines. The
majority of prior research concentrated on standardized and controlled settings,
leaving aside more diverse and less predictable environments, such as those
found in independent repair garages (Lv et al., 2024; Ruengrote et al., 2024).
Additionally, the survey identified the absence of a comprehensive public
database that includes a sufficient variety of problems to aid in the training of
generic diagnostic systems. This influenced our decision to combine several
available open-sourced databases into one corpus of information.
3.3.2 Knowledge Acquisition
Three technical areas are worth discussing when examining the details of the
CIRCA system. First, the IPC-A-600K and IPC-A-610H standards provide a
reference for developing visual acceptance criteria for bare-board defects (opens,
shorts, mouse bites, and missing holes) and assembly defects (insufficient solder,
excess solder, and cold joints) (IPC International, 2020). Second, the CIRCA
employs an attention-driven YOLOv12 detector as a model architecture that
allows for identifying fine-grained soldering mask defects on a noisy
background with the help of the Area Attention (A2) module and R-ELAN
modules. Third, IntelÆs OpenVINO toolkit allows for accelerating model
performance with mixed-precision integer (INT8) on standard CPUs and
integrated GPUs without the graphics card (Ahn et al., 2023).
3.4 Empirical Study
3.4.1 Data Collection
33

Public bare-board datasets: We extracted bare-board defect images from
PKU-Market-PCB-ver1 (PKU Open Lab, 2020) using the Roboflow export jr-
mqqnk/pcb-defects-detection-anddl (~3300 images). This source contains four
bare-board classes: missing_hole, mouse_bite, open_circuit, and short, with
spur and spurious_copper remapped or dropped. We further leveraged the large-
scale corpus of DsPCBSD+ (Lv et al., 2024) comprising 10259 images. We
parsed its JSON COCO annotations to obtain short (SH), open_circuit (OP), and
mouse_bite (MB) with six other classes discarded. Thus, we obtained 3441
images from DsPCBSD+ corpus.
Public assembly-stage datasets: We gathered solder joint defect images from
four public sources. From the first, SolDef_AI (Fontana et al., 2024), we
extracted exc_solder mapped to excess_solder and poor_solder mapped to
cold_solder_joint, discarding three other classes. The second source, excessive-
solder-kydra (ôpcb-defect-detection-emmtsö) provides remapped labels of cold
joints, excess solder, and insufficient solder. The third source, hue-dbgbs-reqtv,
yields insufficient solder and short classes; component-level annotations were
discarded. Finally, pcb-solder-defect-detection-v2-s89jo provides additional
cold joint, excess, and insufficient solder instances. All four sources are
distributed under CC BY 4.0.
Repair-context capture protocol: We captured images of various circuit
boards using a standard 720p USB webcam as a testbed for our qualitative
prototype evaluation. We captured images under different lighting conditions,
namely, standard office lighting, direct desklamp exposure (specular highlights),
and heavy shadowing due to the lamp position blockage. These images were not
used for training, as we only utilized them for prototyping and failure mode
analysis.
Unified 7-class IPC taxonomy: We applied a unified 7-class taxonomy to all
datasets. We selected classes based on availability; only classes appearing in at
least two of the available sources with at least 400 instances per class were
considered. Note that the PKU source does not contain any instances of
missing_hole, but we kept the class, as it appeared in other sources. On the other
hand, some classes only appeared in a single source and were thus excluded.
34

Finally, a class was discarded if the majority of instances were too small to be
reliably processed by an anchor-free backbone, especially the P2 layer which
processes 640╫640 images.
While a single pixel roughly covers 0.2 mm on the x-axis of the image, a sub-
millimeter hole spans only 3 or 4 pixels in either axis in the given resolution.
The boundary smoothing step introduced by JPEG compression, combined with
speckle noise due to lighting conditions, substantially affects such small objects.
In particular, holes of this size are ill-suited for conventional anchor-based or
anchor-free detectors which use lower levels of feature pyramid. Hence, the
class was kept for future reference: it serves as a good sanity check for single-
shot detectors which attempt to process extremely small objects.
| Unified ID  | Class Name  | IPC Reference  |           |            |
| ----------- | ----------- | -------------- | --------- | ---------- |
|             |             |                | Source    | Raw Class  |
|             |             |                | Datasets  | Name(s)    |
Remapped
From
|     | 0  missing_hole  | IPC-A-600  | PKU  | missing_hole  |
| --- | ---------------- | ---------- | ---- | ------------- |
Section 3.4
|     | 1  mouse_bite    |              |           | mouse_bite, MB  |
| --- | ---------------- | ------------ | --------- | --------------- |
|     |                  | IPC-A-600    | PKU,      |                 |
|     |                  | Section 3.3  | DsPCBSD+  |                 |
|     | 2  open_circuit  | IPC-A-600    | PKU,      | open_circuit,   |
|     |                  | Section 3.2  | DsPCBSD+  | OP              |
3  short
|     |     | IPC-A-600    | PKU,       | short, Shorted,  |
| --- | --- | ------------ | ---------- | ---------------- |
|     |     | Section 3.2  | DsPCBSD+,  | SH               |
Hue
|     | 4  excess_solder  | IPC-A-610H  | SolDef_AI,       | exc_solder,  |
| --- | ----------------- | ----------- | ---------------- | ------------ |
|     |                   | Section 5   | kydra, SolderV2  |              |
Excessive_solde
r
5  insufficient_sold IPC-A-610H  kydra, Hue,  Insufficient_sol
|     | er  | Section 5  | SolderV2  | der, Insufficient  |
| --- | --- | ---------- | --------- | ------------------ |
Solder
|     | 6  cold_solder_join | IPC-A-610H  | SolDef_AI,       | poor_solder,  |
| --- | ------------------- | ----------- | ---------------- | ------------- |
|     | t                   | Section 5   | kydra, SolderV2  | Cold Solder,  |
Cold_solder
Table 3.2: CIRCA Unified 7-Class IPC Taxonomy

35

Scope and Limitations: Several of the defect categories specified by IPC-A-
600 and IPC-A-610H standards were excluded from consideration due to
insufficient data: spur, spurious_copper, solder_spike, scratch, and pinhole. For
the former four classes, public training sets contain fewer than the required 400
instances, or have ambiguous visual appearance. Additionally, we did not find a
public board-level annotated dataset that could be used for solder_bridge. All of
these categories were reported as being outside of the projectÆs scope, while a
possible data collection campaign is proposed in Section 5.5.
3.4.2 Data Pre-processing
CLAHE on the LAB L-channel: We apply Contrast Limited Adaptive
Histogram Equalization on the L-channel of the LAB color space (clip limit =
2.0, 8╫8 grid) to neutralize specular reflections on solder fillets and improve
trace visibility (Wanto et al., 2023).
Gamma Correction: The image undergoes a fixed-gamma transformation (? =
1.2) to brighten shadowed areas caused by desk lamps while preserving
highlight regions (Alhamzawi et al., 2024).
Laplacian Variance frame quality gate: We calculate the variance of the
Laplacian of the incoming image at runtime (Lv et al., 2024) and discard it if it
falls below a certain threshold to avoid wasting resources on out-of-focus or
motion-blurred frames.
Polygon to bounding box conversion in SolDef_AI: The polygon coordinates
provided by the SolDef_AI source were converted to axis-aligned bounding
boxes using the Roboflow exporter to ensure that the converted labels follow
the standard YOLO format.
Class remap to 7-class taxonomy: The label images were unified using
scripts/build_unified_pcb_v3.py. In the process, images containing only objects
from the excluded categories were collected as negative examples. All images
36

were then split according to a 70/15/15 distribution to form the final
unified_pcb_v3 dataset.
3.4.3 Data Analysis
Class distribution audit: Perceptual-hash deduplication yielded 8,463 images
with 54,928 annotations. The distribution is heavily skewed toward
insufficient_solder (23,610) and short (12,373) classes, comprising 65.5% of all
annotations, with cold_solder_joint accounting for only 1.2% (633 annotations).
The imbalance was addressed in three ways. First, by limiting the number of
ôdominant class onlyö images (short and insufficient_solder) in the training split
to 1,000 per class. This reduced the initially overwhelming
ôshort/insufficient_solder vs. cold_solder_jointö ratio from 9.9:1 to 5.7:1,
trimming 2,468 dominant-only images from the original 5,924. Second,
applying offline minority oversampling x5 increased the ôcold_solder_jointö
count x5. This brought the training set size to 5,364 images, up from 3,456 after
the first step. Finally, the class weights (cls_pw) in PyTorch and mosaic/copy-
paste augmentation were used to prevent gradient explosions.
Duplicate and leakage control: Duplicate images were removed using a
difference-hash (dHash) deduplication utility before splitting the dataset. With a
conservative 6/64-bit Hamming distance threshold, the deduplication step
removed 6,000 images. It ensured that near duplicates would not skew the
validation/test splits. The image pairs with a high similarity score were typically
temporally close videos of the same board layout, captured from different angles.
3.5 System Design
3.5.1 CIRCA System Architecture
37

The CIRCA application is a six-stage pipeline. Images are read from disk or
acquired from a camera, and undergo initial processing with our CLAHE and
Gamma pipeline. The frame is then fed to an Intel OpenVINO runtime, which
runs a quantized YOLOv12 model. An adaptive sliding-window tiling engine
maintains resolution-independent feature scale during inference.
3.5.2 Inference Pipeline
The OpenVINO runtime loads the chosen model; then the image is split into
640╫640 overlapping tiles, over which the model is inferred. Detections are
merged from overlapping tiles using a non-maximum suppression (IoU
threshold = 0.45) and rendered on top of the image in the user interface.
3.5.3 Confidence Threshold and ôManual Inspection Requiredö
Logic
CIRCA employs per-class display and warning thresholds to tackle the issue of
automation bias (Kupfer et al., 2023). In the event that the mean confidence falls
below 50%, or the frame was rejected by the Laplacian blur gate, a global
ôManual Inspection Requiredö banner is raised, making sure that the technician
is always in control.
3.5.4 Interface Design
The UI gives a drag-and-drop landing zone plus a manual camera frame snapper.
Bounding boxes in color-coded overlays show their raw confidence scores, with
a status footer that shows analysis latency and defect counts.
38

3.6 System Development
3.6.1 Training Engine
The training engine (train_engine.py) is used to stabilize the process of training
custom YOLO models. For this, we select the following key parameters:
lr0=0.001, warmup_epochs=5.0, nbs=64, batch=auto, imgsz=640, seed=42,
close_mosaic=10, cos_lr=True, amp=True, and optimizer=AdamW. Note that
the first value refers to the initial learning rate, which will be further optimized
by genetic hyperparameter tuning in the next stages and thus will not be used in
the final model.
3.6.2 Hyperparameter Optimisation Algorithm
We performed a genetic optimizer over 17 hyperparameters. The search space
does not include hue/saturation shifts, which provide little diagnostic
information and physically implausible transformations.
Stopping criteria/trial budget: We used a trial budget of 50 iterations for the
genetic algorithm with a 50-epoch budget per trial on the YOLOv12-S variant.
We saved the resulting optimized hyperparameters configuration as
runs/detect/CIRCA_V12S_003_TUNE_HPO_7class/best_hyperparameters.ya
ml for subsequent training.
3.6.3 Model Training Procedure
Using the optimized hyperparameter file, we trained YOLOv12-N, YOLOv12-
S, and YOLOv12-M for 200 epochs on a cloud instance (Runpod Secure Cloud,
NVIDIA RTX 3090, 24 GB VRAM) to achieve larger batch sizes. We set
39

variant-specific batch sizes to 64, 48, and 32 for Nano, Small, and Medium,
respectively.
3.6.4 OpenVINO Export and INT8 Quantisation
To adapt these models for use with a consumer desktop, the PyTorch
checkpoints (.pt) were converted to the OpenVINO Intermediate Representation
(.xml/.bin) using the FP32, FP16, and INT8 precision settings. Post-training
quantization was applied using OpenVINOÆs Neural Network Compression
Framework (NNCF). The calibration dataset comprised 300 validation images,
and the integer scaling factors were determined. The following Quantisation
Fallback Rule was applied to protect against diagnostic accuracy loss:
ò The absolute value of the validation mAP@0.5 must not decrease by
more than 1 % (from the FP32 baseline) after quantization to INT8
ò If these criteria were not met, the lower precision INT8 IR file would be
replaced by the FP16 one.
3.6.5 Confidence Threshold Calibration Procedure
The confidence parameter conf was swept from 0.10 to 0.90 with a step of 0.05
on the validation split. For each class, the display threshold was selected as the
minimum confidence value corresponding to precision ? 0.90, and the warning
threshold was selected as the minimum confidence value corresponding to recall
? 0.95.
3.7 Experimental Design
40

3.7.1 Phase 1: Vanilla Baseline
As an ablation control, we train YOLOv12-S for 100 epochs on the raw (i.e., no
CLAHE or Gamma enhanced) unified_pcb_v3 set with default Ultralytics
hyperparameters (with our stability patches, of course).
3.7.2 Phase 2: CIRCA-Aligned Baseline
We trained YOLOv12-S for 100 epochs on the pre-processed
unified_pcb_v3_preproc dataset. By keeping all other hyperparameters, the
same as in Phase 1, we could perform the ablation study of our preprocessor in
one-factor-at-a-time manner.
3.7.3 Phase 3: Hyperparameter Optimisation
We performed the genetic hyperparameter tuner (50 trials ╫ 50 epochs) on the
YOLOv12-S model with a pre-processed training split to find out optimal
learning rates and the loss weight gains.
3.7.4 Phase 4: Three-Variant Final Training
We trained YOLOv12 Nano, Small, and Medium variants for 200 epochs using
the optimized hyperparameter file. We enabled early stopping (patience=50) to
prevent overfitting.
3.7.5 Phase 5: Quantisation Validation
41

We generated FP32, FP16, and INT8 OpenVINO IR files for each trained variant.
We ran validation checks against our Quantisation Fallback Rule to select the
deployment weights.
3.7.6 Phase 6: Hardware Benchmarking and Variant Selection
We evaluated the performance of the remaining model configurations on the
target deployment platform, measuring the time taken for preprocessing, CPU +
integrated GPU inference, and full end-to-end static image analysis. The version
with the highest mAP while still meeting all latency constraints was chosen for
deployment.
3.7.7 Phase 7: Final Test Evaluation and Threshold Calibration
We ran the selected production model ONCE on the held-out test split, reporting
final precision, recall, F1, and mAP scores, along with confusion matrices and
PR curves.
3.7.8 Acceptance Criteria
To pass, the production variant must have a preprocessing latency of ?5 ms, a
tiled static image analysis time of ?10 s, and a static image inference time of
?10 s on the target Intel processor. Do not interpret the 90% mAP threshold as
an operational target. We determined this baseline through literature review of
vision-based industrial inspection models trained on single-class anomalies in
controlled factory illumination. The CIRCA cross-domain, multi-source
benchmark represents a more challenging, real-world scenario than a pristine
factory floor. Take the 90% mAP threshold as an aspirational target, rather than
a hard floor, for a unified anomaly detection taxonomy.
42

3.7.9 Evaluation Metrics
We report common metrics: precision, recall, F1-score, and mAP (calculated
with COCO evaluation protocols where IoU thresholds range from 0.50 to 0.95).
We further report latencies in milliseconds or seconds.
3.8 Hardware and Software Specification
3.8.1 Training Environment
We trained our models on a cloud instance (Runpod Secure Cloud, NVIDIA
RTX 3090, 24 GB VRAM) due to the limitations of the local laptop RTX 3060
(6 GB VRAM) which only allowed for 6 batch size for the Medium model. The
software stack comprised Python 3.11, PyTorch 2.x, CUDA 12.x, Ultralytics 8.3,
and Weights & Biases for experiment tracking.
3.8.2 Deployment Target
What are the target system requirements? The target system is a intel core i5,
8th generation, processor plus integrated graphic card running windows 10/11.
Must have AVX2 and VNNI support for INT8 acceleration. A discrete graphics
card is not necessary.
3.8.3 Software Stack
43

The application runs on Python 3.11, PyTorch inference libraries, OpenVINO
Runtime, NNCF, OpenCV, and a lightweight desktop UI built using PyQt6.

3.9   Research Plan

The project timeline was sequenced to ensure clean data flow between phases.
Table 3.3 details the plan and compute estimates.
| Phase  | Description  | Estimated  | Platform  |
| ------ | ------------ | ---------- | --------- |
Duration
|     | 0   | 1 week  | Local  |
| --- | --- | ------- | ------ |
Dataset rebuild (6
sources, remap,
dedup, split ?
unified_pcb_v3)
|     | 1   | ~90 min  | Runpod RTX 3090  |
| --- | --- | -------- | ---------------- |
Vanilla baseline (100
ep, YOLOv12-S)
|     | 2  CIRCA-aligned  | ~90 min  | Runpod RTX 3090  |
| --- | ----------------- | -------- | ---------------- |
baseline (100 ep,
preproc, OFAT)
|     | 3  Genetic HPO (50 it ╫  | ~23.4 h  | Runpod RTX 3090  |
| --- | ------------------------ | -------- | ---------------- |
50 ep, fraction=0.5)
|     | 4  Three-variant final  | ~25 h  | Runpod RTX 3090  |
| --- | ----------------------- | ------ | ---------------- |
training (200 ep ╫ 3)
|     | 5  OpenVINO  | 1 day  | Local  |
| --- | ------------ | ------ | ------ |
quantisation
validation
|     | 6   | 1û2 days  | Local  |
| --- | --- | --------- | ------ |
Hardware
benchmarking on i5
8th-gen
|     | 7   | 1û2 days  | Local  |
| --- | --- | --------- | ------ |
Test evaluation +
threshold calibration
|     | Total cloud GPU  | ~51.4 h (RTX 3090)  |     |
| --- | ---------------- | ------------------- | --- |
compute
Table 3.3: CIRCA Project Timeline and Compute Estimate

44

3.10 Summary
This chapter describes the CIRCA methodology in detail. First, we formed a
seven-class IPC-sorted dataset (unified_pcb_v3) of 8,463 unique images and
54,928 instances in total after the difference-hash deduplication stage removed
6,000 near-duplicates. We split the dataset into a 70/15/15 down-sampled
stratified train/validation/test partitions with 5╫ oversampling of the minority
classes in the train partition. Next, we applied a pre-processing pipeline that used
LAB colour space CLAHE, gamma correction, and a runtime Laplacian blur
gate. We then followed an iterative development approach, starting from the
simplest possible setup of the task, evaluating the baseline performance, and
performing an OFAT (One Factor At A Time) experiment to select the most
impactful pre-processing block. We next proceeded to hyperparameter
optimization, model compilation with OpenVINO, and local performance
benchmarking. The quantitative evaluation results of the methodology are
described in detail in Chapter 4.
45

CHAPTER 4
RESULTS AND FINDINGS
4.1 Introduction
We report and analyse our experimental results together with their context in
each section. The CIRCA pipeline is comprised of sequential stages, namely,
preprocessing ablation analysis, genetic optimization, comparative model
training, and target device benchmarking; therefore, we discuss the results of
each stage together with the context of the changes made to the model.
4.2 Dataset and Defect Taxonomy Results
In order to address Research Objective 1 (RO1), a comprehensive list of defect
categories was established, unifying both bare-board and solder assembly
categories according to the IPC standard. The identified distribution of physical
defects and the split-statistics of the derived unified_pcb_v3 corpus are
presented below to verify RO1.
The number of annotated instances for each category from the deduplicated
corpus is demonstrated in Table 4.1.
ID Class Name IPC Total % of Primary
Reference Instances Total Sources
0 missing_hole IPC-A-600 2,315 4.2% PKU
1 mouse_bite IPC-A-600 4,887 8.9% PKU,
DsPCBSD+
46

2 open_circuit IPC-A-600 3,990 7.3% PKU,
DsPCBSD+
3 short IPC-A-600 12,373 22.6% PKU,
DsPCBSD+,
Hue
4 excess_solder IPC-A-610H 7,120 13.0% SolDef_AI,
kydra,
SolderV2
5 insufficient_solder IPC-A-610H 23,610 43.0% kydra, Hue,
SolderV2
6 cold_solder_joint IPC-A-610H 633 1.2% SolDef_AI,
kydra,
SolderV2
Total 54,928 100%
Table 4.1: Final Class Distribution: `unified_pcb_v3` (8,463 unique images, 54,928
instances)
We have grouped the bare-board defect classes into 4 classes according to the
IPC-A-600 standard and solder joint assembly classes û into 3 classes according
to the IPC-A-610H standard. The excluded categories include spur,
spurious_copper, solder_spike, scratch, and pinhole. These classes were
removed from consideration during the data-availability audit due to the
insufficient number of samples, below the threshold of 400 instances, necessary
for achieving gradient stability in neural networks. In addition, we have
excluded the classes describing the fault in components, such as misalignment
or missing, in order to narrow the taxonomy to the features found on the board
itself: copper and solder.
The initial distribution of annotations was highly uneven, with insufficient
solder constituting the majority (43%) and cold solder joints presenting a minor
fraction (1.2%). To address this imbalance, we applied the procedure of multi-
stage capping and oversampling described in detail in Chapter 3, Section 3.4.3.
The distribution of instances in the train/validation/test sets is presented in Table
4.2 below.
47

| Split       | Images (before        | Images (after   | Proportion  |
| ----------- | --------------------- | --------------- | ----------- |
|             | OS)                   | oversampling)   |             |
| Train       | 5,924 (3,456 capped)  | 5,364           | 70%         |
| Validation  | 1,269                 | 1,269 (frozen)  | 15%         |
| Test        | 1,270                 | 1,270 (frozen)  | 15%         |
| Total       | 8,463 (5,995          | 7,903           |             |
capped)
Table 4.2: Train / Validation / Test Split Statistics: `unified_pcb_v3`
The final corpus of images was formed from six publicly available databases
after removing 6,000 near-duplicates recognized by difference-hash (dHash).
3988782_91149.jpg Captions were removed as the initial step in training. The
problem of imbalanced classes was addressed by removing 2,468 dominant-only
images, thus reducing the ratio from 9.9:1 to 5.7:1. Minority classes were then
oversampled 5╫ resulting in a training set of 5,364 files. The validation and test
subsets were used as is, serving as a reliable benchmark for performance
assessment.

4.2.3  Sample Defect Images per IPC Class

Figure 4.1: Sample Defect Images
[Figure 4.1: Representative PCB defect images for each of the seven IPC-
aligned classes, drawn from the unified_pcb_v3 training split. Bounding boxes
indicate the annotated defect region. Classes 0-3 are bare-board defects (IPC-
A-600); classes 4-6 are assembly-stage solder defects (IPC-A-610H).]
48

4.3   Preprocessing Pipeline Evaluation

4.3.1  Vanilla vs CIRCA-Pre-processed Baseline Comparison

Analysis of the Preprocessing Block: Evaluation of the Diagnostic Value of
CLAHE and Gamma Correction
The comparison of the results obtained during evaluation of the preprocessing
block (Phases 1 and 2) allows us to isolate the diagnostic value of CLAHE and
Gamma  Correction  individual  contribution  in  the  model  performance.
Following the one-factor-at-a-time (OFAT) philosophy, a YOLOv12-S model
was trained for 100 epochs using the raw unified_pcb_v3 split as a reference
and compared to the pre-processed baseline.
The validation results are presented below in Table 4.3.
|                       | Metric  | Phase 1     |         | Phase 2     |         | Difference      |
| --------------------- | ------- | ----------- | ------- | ----------- | ------- | --------------- |
|                       |         | (Vanilla)   |         | (CIRCA)     |         |                 |
| Training Hardware     |         | Runpod RTX  |         | Runpod RTX  |         | -               |
|                       |         | 3090 24 GB  |         | 3090 24 GB  |         |                 |
| Epochs trained        |         |             | 100     |             | 100     | OFAT equal      |
| Best epoch            |         |             | 45      |             | 50      | -               |
| mAP@0.5 (best epoch)  |         |             | 0.6649  |             | 0.6600  | -0.0049 (-0.49  |
pp)
| mAP@0.5:0.95 (best epoch)  |     |     | 0.4237  |     | 0.4284  | +0.0047 (+0.47  |
| -------------------------- | --- | --- | ------- | --- | ------- | --------------- |
pp)
Precision (at best mAP epoch)  0.7290  0.8443  +0.1153 (+11.5
pp)
| Recall (at best mAP epoch)  |     |     | 0.6433  |     | 0.6341  | -0.0092  |
| --------------------------- | --- | --- | ------- | --- | ------- | -------- |
mAP@0.5 (final epoch)  0.6591 (ep.100)  0.6536 (ep.80)  -0.55 pp
| Early stop triggered  |     |     | No  | Yes (ep.80,  |     | -   |
| --------------------- | --- | --- | --- | ------------ | --- | --- |
patience=30)
49

Train cls loss (lower = better) 0.181 0.179 -0.002
Val cls loss (ep.40+, lower = better) Higher Lower CIRCA wins
(consistent)
Table 4.3: Phase 1 vs Phase 2 Validation Metrics (OFAT Ablation, YOLOv12-S, 100 Epochs,
`unified_pcb_v3`)
Figure 4.2: Phase 1 vs Phase 2 Ablation Comparison
[Figure 4.2: Phase 1 (Vanilla) vs. Phase 2 (CIRCA-Pre-processed) training
curves across six validation metrics (mAP@0.5, mAP@0.5:0.95, Precision,
Training Class Loss, Validation Class Loss, and Validation Box Loss).
YOLOv12-S, 100 epochs, OFAT design. Source:
runs/detect/CIRCA_V12S_001_TRAIN_Baseline_Vanilla_2/ and
runs/detect/CIRCA_V12S_002_TRAIN_Baseline_CIRCA_2/.]
4.3.2 Preprocessing Latency Measurement
Table 4.4 details the latency footprint of our preprocessing stages on the target
CPU.
50

Preprocessing Stage Operation Description Mean Percentage
Latency of Total
(ms) (%)
Colour Conversion & LAB conversion, split, histogram 3.61 76.0%
CLAHE equalization on L channel, merge, BGR
reconversion
Gamma Correction Lookup table mapping (LUT) for 0.40 8.4%
contrast adjustment
Laplacian Gating Down sampling, Laplacian filtering, 0.74 15.6%
standard deviation squaring
Total Pipeline End-to-end frame preprocessing 4.75 100.0%
Table 4.4: Preprocessing Latency per Stage on the Representative Target CPU
4.3.3 Discussion
The results of the pre-processing ablation test revealed a nuanced negative effect
with a substantial classification signal. The results show that the raw dataÆs
validation mAP@0.5 was only slightly lower than the baselineÆs 0.6649, at
0.6600. However, the pre-processed batchÆs precision was almost 11.5% higher
(0.8443 vs. 0.7290), and its loss score was steadily decreasing past epoch 40.
The pre-processed model was also converging at about 20% faster than the raw
data batch, finishing at epoch 80 instead of 100.
I think it is entirely reasonable that the mAP did not improve on the validation
batch as applying CLAHE and Gamma Correction to photos already taken under
optimal lighting conditions introduced some distortions that reduced the quality
of the data. The same effect can be seen in real-time when the filters are applied
to video streams during inference to remove shadows and glare from the
workstations. Therefore, I think it is reasonable to keep the pre-processing steps.
The latency of 4.75ms is well within the acceptable threshold of 5ms. The
CLAHE operation and colour conversion make up the entirety of the
preprocessing steps and consume 76% of the frame processing time. Meanwhile,
the Laplacian blur check takes only 0.74ms to execute.
51

4.4 Hyperparameter Optimisation Results
The genetic algorithm optimization of the YOLOv12-S model took 23.4 hours
to complete on the NVIDIA RTX 3090 graphics card. During the search process
that lasted for 50 iterations, the algorithm sampled 17 decision variables. Figure
4.3 below shows the overall model fitness or the mean average precision (mAP)
at 0.5:0.95 on the validation set across all iterations of the HPO process.
Figure 4.3: HPO Fitness Trajectory
[Figure 4.3: Genetic Algorithm Fitness Trajectory across 50 HPO Iterations
(YOLOv12-S, Pre-processed 7-Class Corpus). Source:
runs/detect/CIRCA_V12S_003_TUNE_HPO_7class/tune_fitness.png.]
The modelÆs fitness was gradually improved as the tuner progressed. Figure 4.6
below shows that the best trial achieved a fitness value of 0.263, which is 31.6%
higher than the initial solution at iteration 42. It can be seen from the scatter plot
that the initial learning rate (lr0) and box loss gain (box) had the most significant
impact on model performance. All the successful trials had a value lower than
0.001 in the lr0 hyperparameter, while the highest fitness trials were grouped
52

between 0.0001 and 0.0003. The box loss gain was between 0.15 and 0.2, while
the default setting in YOLOv12 was set to 7.5.
Table 4.5 below shows the ten best HPO trials based on the fitness metric.
|     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Rank Iterati Fitnes Precis Recall mAP mAP@ `lr0` `mom `box` `cls` `cls_ `mos `scale
|     |     |     |     |               |       |     |     |          |     |
| --- | --- | --- | --- | ------------- | ----- | --- | --- | -------- | --- |
| on  | s   | ion |     | @0.5 0.5:0.95 | entum |     |     | pw` aic` | `   |

`
|     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
1 42 0.2630 0.7161 0.4196 0.4350 0.2631 0.00014 0.7851 0.169 0.265 0.100 0.722 0.650
|     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | 5   |     |     |     | 0   |     | 2   | 7 0 | 0 2 |
2   31   0.2560 0.6328   0.4356   0.4328   0.2561   0.00029 0.8356   0.193 0.200 0.100 0.684 0.492
|     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | 8   |     |     |     | 0   |     | 6   | 0 0 | 6 5 |
3   46   0.2559 0.6181   0.4439   0.4316   0.2559   0.00021 0.8749   0.191 0.200 0.132 0.760 0.893
|     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | 3   |     |     |     | 0   |     | 2   | 0 4 | 8 6 |
|     |     |     |     |     |     |     |     |     |     |
4 49 0.2555 0.6567 0.4344 0.4345 0.2555 0.00024 0.8940 0.155 0.218 0.108 0.733 0.532
|     | 4   |     |     |     | 0   |     | 7   | 1   6   | 3   5   |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ------- |
|     |     |     |     |     |     |     |     |         |         |
5 25 0.2541 0.6518 0.4119 0.4309 0.2541 0.00005 0.9800 0.163 0.311 0.102 0.766 0.758
|     | 0   |     |     |     | 0   |     | 0   | 5   4   | 4   1   |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ------- |
|     |     |     |     |     |     |     |     |         |         |
6 37 0.2533 0.7948 0.4088 0.4295 0.2534 0.00033 0.9800 0.157 0.324 0.106 0.683 0.900
|     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | 6   |     |     |     | 0   |     | 2   | 9 8 | 0 0 |
|     |     |     |     |     |     |     |     |     |     |
7 41 0.2523 0.6446 0.4215 0.4338 0.2524 0.00015 0.8568 0.190 0.200 0.120 0.828 0.900
|     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | 8   |     |     |     | 0   |     | 8   | 8 8 | 6 0 |
|     |     |     |     |     |     |     |     |     |     |
8 34 0.2523 0.6626 0.4105 0.4258 0.2523 0.00004 0.8049 0.200 0.335 0.120 1.000 0.900
|     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | 1   |     |     |     | 0   |     | 0   | 1 2 | 0 0 |
9   45   0.2514 0.7107   0.4219   0.4279   0.2515   0.00013 0.8664   0.200 0.340 0.151 1.000 0.577
|     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | 6   |     |     |     | 0   |     | 0   | 4 7 | 0 6 |
10   48   0.2512 0.6602   0.4517   0.4265   0.2513   0.00015 0.8861   0.200 0.209 0.126 0.985 0.589
|     | 7   |     |     |     | 0   |     | 0   | 7   2   | 7   6   |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ------- |
Table 4.5: Top 10 HPO Trials Ranked by Fitness
Scatter plots (Figure 4.4) illustrate these parameter relationships. Yellow points
mark later iterations with higher fitness, while purple points represent early trials.
53

Figure 4.4: HPO Parameter Scatter Plots
[Figure 4.4: Hyperparameter Search Space Scatter Plots vs. Fitness (YOLOv12-
| S,  | Phase  |     | 3   | HPO).  | Source:  |
| --- | ------ | --- | --- | ------ | -------- |
runs/detect/CIRCA_V12S_003_TUNE_HPO_7class/tune_scatter_plots.png.]
Table  4.6  compares  the  final  tuned  parameters  against  default YOLOv12
configurations.
| Parameter  | YOLOv1     | Tuned  | Magnitude  | Domain Significance  |     |
| ---------- | ---------- | ------ | ---------- | -------------------- | --- |
|            | 2 Default  | Value  | of Change  |                      |     |
lr0 (Initial  0.01  0.00014  ≈71╫  Critical: Avoids gradient
| Learning Rate)  |     |     | reduction  | explosion on small, low-contrast  |     |
| --------------- | --- | --- | ---------- | --------------------------------- | --- |
defects.
|                | 7.5  | 0.169  |            |                                  |     |
| -------------- | ---- | ------ | ---------- | -------------------------------- | --- |
| box (Box Loss  |      |        | ≈44╫       | Critical: Prevents localization  |     |
| Gain)          |      |        | reduction  | loss from dominating             |     |
classification.
|                  | 0.5  | 0.266  |            |                                    |     |
| ---------------- | ---- | ------ | ---------- | ---------------------------------- | --- |
| cls (Class Loss  |      |        | ≈1.9╫      | Moderate: Restructures class loss  |     |
| Gain)            |      |        | reduction  | weight for the 7-class taxonomy.   |     |
54

| momentum   | 0.937  0.785  | -16%    | Moderate: Increases optimizer  |
| ---------- | ------------- | ------- | ------------------------------ |
| (SGD/Adam  |               | change  | sensitivity to rare defect     |
| Momentum)  |               |         | gradients.                     |
1.0  0.722
| mosaic (Mosaic  |     | -28%    | Moderate: Limits visual noise  |
| --------------- | --- | ------- | ------------------------------ |
| Augmentation)   |     | change  | from out-of-context image      |
composites.
0.5  0.650
| scale (Scale   |     | +30%      | Minor: Enhances model              |
| -------------- | --- | --------- | ---------------------------------- |
| Augmentation)  |     | increase  | robustness to scale variations in  |
PCB layouts.
weight_decay  0.0005  0.0009  ╫1.8  Minor: Reduces overfitting on
|     |     | increase  | oversampled minority classes.  |
| --- | --- | --------- | ------------------------------ |
(L2
Regularization)
| dfl            | 1.5  1.656  | +10%      | Minor: Stabilizes boundary   |
| -------------- | ----------- | --------- | ---------------------------- |
| (Distribution  |             | increase  | regressions on fine-grained  |
| Focal Loss)    |             |           | defects.                     |
copy_paste  0.0  0.011  Introduced  Minor: Pastes defect regions onto
| (Copy-Paste  |     |     | new backgrounds to combat  |
| ------------ | --- | --- | -------------------------- |
| Aug)         |     |     | imbalance.                 |
Table 4.6: Comparison of Tuned Hyperparameters and YOLOv12 Defaults

4.4.5  Discussion

Standard  YOLOv12  defaults  are  poorly  suited  for  sub-millimetre  PCB
inspection. The tuner's 71-fold reduction in initial learning rate (from 0.01 to
0.00014) indicates that fine-grained defect boundaries require highly stable
gradient steps. A high learning rate causes the optimizer to overshoot local
minima on small, low-contrast features.
The 44-fold reduction in box loss gain (box) is equally significant. General
object detectors prioritize box localization boundaries because target objects
vary widely in size and position. In PCB diagnostics, however, defect candidates
are  physically  constrained  to  static  traces  and  solder  pads.  Distinguishing
between classes (such as separating a trace short from excess_solder) is the
primary challenge. Lowering the box loss gain prevents localization loss from
dominating gradients, allowing the network to focus on class distinction.
55

Mosaic augmentation was limited to 0.722. While combining four random
images works well for general scenery, stitching unrelated PCB designs together
creates artificial borders that disrupt trace tracking. Similarly, the tuner drove
rotation and mix-up parameters to near-zero, confirming that fixed-orientation
inspections do not benefit from rotation and that mixing layouts degrades
representation learning.
4.5 Three-Variant Comparative Training Results
Section 4.5 demonstrates the results of Research Objective 2 (RO2) by testing
the Nano, Small, and Medium YOLOv12 variants on our own pre-processed
balanced corpus. The training progress for all three models over 200 epochs is
visualized in Figures 4.5a , 4.5b , and 4.5c respectively.
Figures 4.5a, 4.5b, and 4.5c show the training trajectories across 200 epochs.
Figure 4.5a: YOLOv12-Nano Training Curves
[Figure 4.5a: Training and validation loss trajectories, Precision, Recall, and
mAP metrics across 200 epochs for the YOLOv12-Nano variant.]
56

Figure 4.5b: YOLOv12-Small Training Curves
[Figure 4.5b: Training and validation loss trajectories, Precision, Recall, and
mAP metrics across 200 epochs for the YOLOv12-Small variant.]
Figure 4.5c: YOLOv12-Medium Training Curves
[Figure 4.5c: Training and validation loss trajectories, Precision, Recall, and
mAP metrics across 200 epochs for the YOLOv12-Medium variant.]
The Nano version converged steadily to a stable validation loss by the 140th
epoch, achieving the maximum of the validation mAP at epoch 182. On the other
hand, the Small version showed a faster decrease in the loss function of the
boxes, while the validation class loss slightly oscillated after the 120th epoch,
allowing for several false stopping conditions. Finally, the Medium version had
57

the minimum train class loss, but it was stable already at the 150th epoch, while
for better performance, it requires a larger network that learns faster but with
diminishing returns in terms of accuracy.
Table 4.7 represents the validation results for the three variants discussed above.
| Model Variant  |     | Parame | mAP@     | mAP@      |     | Precisio | Recall  | F1-Score  |
| -------------- | --- | ------ | -------- | --------- | --- | -------- | ------- | --------- |
|                |     | ters   | 0.5 (%)  | 0.5:0.95  |     | n (%)    | (%)     | (%)       |
|                |     | (M)    |          | (%)       |     |          |         |           |
| YOLOv12-N      |     | 2.38   | 63.13    | 39.52     |     | 83.16    | 60.23   | 69.87     |
(Nano)
| YOLOv12-S  |     | 9.23  | 66.20  | 42.97  |     | 73.06  | 67.00  | 69.90  |
| ---------- | --- | ----- | ------ | ------ | --- | ------ | ------ | ------ |
(Small)
| YOLOv12-M  |     | 20.11  | 67.42  | 43.89  |     | 74.78  | 67.07  | 70.74  |
| ---------- | --- | ------ | ------ | ------ | --- | ------ | ------ | ------ |
(Medium)
Table 4.7: Validation Metrics per YOLOv12 Variant (Phase 4)
Table 4.8 breaks down validation mAP@0.5 by individual class.
| Partition  | Defect Class  |     | IPC Standard  |     | YOLOv1   |       | YOLOv1   | YOLOv1   |
| ---------- | ------------- | --- | ------------- | --- | -------- | ----- | -------- | -------- |
|            |               |     | Reference     |     | 2-N (%)  |       | 2-S (%)  | 2-M (%)  |
| IPC-A-600  | missing_hole  |     |               |     |          | 1.91  | 5.57     | 5.85     |
IPC-A-600 Section
3.4
(Bare-Board)  mouse_bite  IPC-A-600 Section  57.97  64.02  64.62
3.3
|     | open_circuit  |     |                    |     |     | 69.69  | 71.05  | 71.72  |
| --- | ------------- | --- | ------------------ | --- | --- | ------ | ------ | ------ |
|     |               |     | IPC-A-600 Section  |     |     |        |        |        |
3.2
|     | short  |     | IPC-A-600 Section  |     |     | 89.18  | 92.96  | 94.08  |
| --- | ------ | --- | ------------------ | --- | --- | ------ | ------ | ------ |
3.2
| IPC-A-610H  | excess_solder  |     |     |     |     | 39.63  | 45.75  | 52.05  |
| ----------- | -------------- | --- | --- | --- | --- | ------ | ------ | ------ |
IPC-A-610H
Section 5
| (Solder    | insufficient_s |     | IPC-A-610H  |     |     | 90.92  | 91.80  | 93.65  |
| ---------- | -------------- | --- | ----------- | --- | --- | ------ | ------ | ------ |
| Assembly)  | older          |     | Section 5   |     |     |        |        |        |
|            |                |     |             |     |     | 92.98  | 95.31  | 95.06  |
|            | cold_solder_j  |     | IPC-A-610H  |     |     |        |        |        |
|            | oint           |     | Section 5   |     |     |        |        |        |
Table 4.8: Per-class Validation mAP@0.5 Breakdown per Variant

58

4.5.4 Discussion
The growth of the model capacity demonstrates logarithmic returns, as evident
in the progression from the 2.38M parameter Nano model to the 9.23M
parameter Small model, which improved the mAP by +3.07 Pts (from 63.13%
to 66.20%). However, increasing the capacity again to the 20.11M parameter
Medium model only provided +1.22 Pts (from 66.20% to 67.42%). It is
reasonable to select the Small variant as the best choice for classification due to
the diminishing returns of growing model capacity on this particular dataset.
The oversampling procedure allowed the minority classes to benefit
significantly from it, as the solder assembly classes (insufficient_solder and
cold_solder_joint) demonstrate AP scores of over 90% on all the models, as their
solder joints have clear visual boundaries. The missing_hole class is the most
challenging one as it never achieved an AP score above 5.85%. This is explained
by the fact that a missing hole is a defect of 0.3û1.0 mm, which corresponds to
only 3 pixels in the 640╫640 image, thus being too small to be consistently
detected by a single-stage YOLOv12 backbone, which will be addressed in the
P2 detector from chapter 5.
4.6 OpenVINO Quantisation Results
4.6.1 FP32 vs FP16 vs INT8 mAP
Post-training quantization was applied to convert the weights to INT8 in order
to make the model run faster on typical CPUs. Table 4.9 below compares the
mAP and the sizes of the models using FP32, FP16, and INT8.
Model FP32 FP16 INT8 INT8 Size Size Size
Variant mAP mAP mAP Delta FP32 INT8 Reductio
@0.5 @0.5 @0.5 (pp) (MB) (MB) n (x)
(%) (%) (%)
59

| YOLOv12-N  | 63.13  63.12  | 61.97  | -1.15  | 9.53  | 3.19  | 2.99╫  |
| ---------- | ------------- | ------ | ------ | ----- | ----- | ------ |
(Nano)
| YOLOv12-S  | 66.20  66.20  | 66.30  | +0.10  | 35.79  | 10.05  | 3.56╫  |
| ---------- | ------------- | ------ | ------ | ------ | ------ | ------ |
(Small)
| YOLOv12-M  | 67.42  67.42  | 67.09  | -0.33  | 77.32  | 20.56  | 3.76╫  |
| ---------- | ------------- | ------ | ------ | ------ | ------ | ------ |
(Medium)
Table 4.9: FP32 vs FP16 vs INT8 Validation mAP@0.5 Comparison
4.6.2  INT8 ? FP16 Fallback Decision

Table 4.10 outlines the per-variant fallback decisions.
| Model Variant  | INT8     | Absolut  | mAP      | Delta     | Selecte  | Primary    |
| -------------- | -------- | -------- | -------- | --------- | -------- | ---------- |
|                | mAP@     | e Pass   | Loss vs  | Target    | d        | Decision   |
|                | 0.5 (%)  | Target   | FP32     |           | Precisio | Rationale  |
|                |          |          | (pp)     |           | n        |            |
| YOLOv12-N      | 61.97    | ? 90.0%  | -1.15    | ? 1.0 pp  | FP16     | Failed     |
| (Nano)         |          | (FAIL)   |          | (FAIL)    |          |            |
both
absolute
(61.97% <
90%) and
delta (1.15
pp > 1.0
pp)
criteria.
| YOLOv12-S  | 66.30  | ? 90.0%  | +0.10  | ? 1.0 pp  | FP16  | Failed    |
| ---------- | ------ | -------- | ------ | --------- | ----- | --------- |
| (Small)    |        | (FAIL)   |        | (PASS)    |       | absolute  |
criterion
(66.30% <
90%)
despite a
+0.10 pp
mAP lift.
| YOLOv12-M  | 67.09  | ? 90.0%  | -0.33  | ? 1.0 pp  | FP16  | Failed  |
| ---------- | ------ | -------- | ------ | --------- | ----- | ------- |
| (Medium)   |        | (FAIL)   |        | (PASS)    |       |         |
absolute
criterion
(67.09% <
90%)
despite
minimal
delta (-
0.33 pp).
Table 4.10: Per-variant Fallback Decision Summary
60

4.6.3 Discussion
Downscaling the model weights from FP32 to INT8 resulted in performance
degradation for all 3 variants. The Nano variant had the highest drop in
performance by -1.15%, followed by the Medium one with a -0.33% change. In
contrast, the Small variant saw an increase in performance by +0.1%. Such an
outcome suggests that using the integer representation as a regularizer for
overparameterized networks is beneficial.
Nevertheless, as none of the quantized models met our baseline mAP@0.5
>=90.0%, we had to revert to the next best precision, namely FP16. The choice
of FP16 IR was justified by the significant storage footprint reduction (~x2) and
the ability to keep diagnostic accuracy at the required level, rather than any
computational benefits. It was also the right decision from the edge deployment
perspective, due to the increased risk of false-negative detections for quantized
vision models (Ahn et al., 2023).
4.7 Hardware Benchmarking Results
4.7.1 Preprocessing Latency on Target CPU
We benchmarked the models on the deployment CPU and integrated GPU
targets. Table 4.11 tracks preprocessing latency per stage.
Preprocessing Stage Operation Description Mean Percentage
Latency of Total
(ms) (%)
Colour Conversion & LAB conversion, split, histogram 3.61 76.0%
CLAHE equalization on L channel, merge, BGR
reconversion
61

Gamma Correction  Lookup table mapping (LUT) for  0.40  8.4%
contrast adjustment
Laplacian Gating  Down sampling, Laplacian filtering,  0.74  15.6%
standard deviation squaring
Total Pipeline  End-to-end frame preprocessing  4.75  100.0%
Table 4.11: Preprocessing Latency per Stage on the Representative Target CPU

4.7.2  Inference Latency: CPU vs GPU
Table 4.12 compares CPU and GPU inference times per tile.
| Model Variant  | Precision  |               |               |                 |
| -------------- | ---------- | ------------- | ------------- | --------------- |
|                |            | Device /      | Mean Latency  | Speedup (CPU    |
|                |            | Runtime       | (ms)          | to GPU)         |
| YOLOv12-N      | FP16       | CPU           | 24.51         | Baseline        |
| (Nano)         |            | (OpenVINO)    |               |                 |
|                | FP16       | GPU (GeForce  | 23.66         | 1.04╫ speedup   |
RTX 3060)
| YOLOv12-S  | FP16  | CPU           | 71.04   | Baseline     |
| ---------- | ----- | ------------- | ------- | ------------ |
| (Small)    |       | (OpenVINO)    |         |              |
|            | FP16  | GPU (GeForce  | 100.69  | 0.71╫        |
|            |       | RTX 3060)     |         | (Slowdown)   |
| YOLOv12-M  | FP16  | CPU           | 176.13  | Baseline     |
| (Medium)   |       | (OpenVINO)    |         |              |
|            | FP16  | GPU (GeForce  | 289.66  | 0.61╫        |
|            |       | RTX 3060)     |         | (Slowdown)   |
Table 4.12: Inference Latency Comparison (CPU vs. GPU)

4.7.3  Image Analysis Throughput

Because CIRCA is running in a static diagnostic mode, throughput is then
characterized by the total analysis time rather than a frame rate. The high-
62

resolution image is divided into tiles for parallel processing and then combined.
The approximate analysis times are shown below:
| Image Size          |     | Tiles  |     | CPU Analysis Time  | GPU Analysis Time  |     |
| ------------------- | --- | ------ | --- | ------------------ | ------------------ | --- |
| 640╫640 (close-up)  |     | 1      |     | 24.51 ms           | 23.66 ms           |     |
| 1280╫720 (webcam    |     | 6      |     | 136.46 ms          | 137.57 ms          |     |
capture)
| 1920╫1080 (phone  |     | 15  |     | 349.96 ms  | 338.92 ms  |     |
| ----------------- | --- | --- | --- | ---------- | ---------- | --- |
photo)

4.7.4  Static Image Inference Time

Table 4.13 summarizes end-to-end static image analysis times.
| Model    | Runtime  | Preproc  |       | Net        | Total E2E    | Pass Target  |
| -------- | -------- | -------- | ----- | ---------- | ------------ | ------------ |
| Variant  | Device   |          | (ms)  | Inference  | Latency (s)  | (? 10 s)     |
(ms)
| YOLOv12-  | CPU         |     | 4.75  | 24.51  | 0.392  | YES (PASS)  |
| --------- | ----------- | --- | ----- | ------ | ------ | ----------- |
| N (Nano)  | (OpenVINO)  |     |       |        |        |             |
|           | GPU         |     | 4.75  | 23.66  | 0.397  | YES (PASS)  |
(OpenVINO)
| YOLOv12-S  | CPU         |     | 4.75  | 71.04   | 1.111  | YES (PASS)  |
| ---------- | ----------- | --- | ----- | ------- | ------ | ----------- |
| (Small)    | (OpenVINO)  |     |       |         |        |             |
|            | GPU         |     | 4.75  | 100.69  | 1.533  | YES (PASS)  |
(OpenVINO)
| YOLOv12- | CPU         |     | 4.75  | 176.13  | 2.746  | YES (PASS)  |
| -------- | ----------- | --- | ----- | ------- | ------ | ----------- |
| M        | (OpenVINO)  |     |       |         |        |             |
(Medium)
|     |      |     | 4.75  | 289.66  | 4.327  | YES (PASS)  |
| --- | ---- | --- | ----- | ------- | ------ | ----------- |
|     | GPU  |     |       |         |        |             |
(OpenVINO)
Table 4.13: Static Image Inference Latency Comparison

63

4.7.5  Variant Selection Matrix and Acceptance-Criteria Verdict

Table 4.14 compiles the final Variant Selection Matrix.
| Model  | Select Runti    | Val   | Prepr    | Infer  | Static  | Model  | Pass   |
| ------ | --------------- | ----- | -------- | ------ | ------- | ------ | ------ |
| Varian | ed  me          | mAP   | oc (? 5  | Laten  | E2E     | Size   | All    |
| t      | Precisi Device  | @0.5  | ms)      | cy     | Time    | (MB)   | Criter |
|        | on              | (%)   |          | (ms)   | (? 10   |        | ia?    |
s)
| YOLO | FP16  CPU  | 63.12  | (4.75  | 24.51  | (0.392  | 5.07  | YES     |
| ---- | ---------- | ------ | ------ | ------ | ------- | ----- | ------- |
|      |            |        | ms)    |        | s)      |       | (PASS)  |
v12-N
(Nano)
| YOLO | FP16  GPU  | 63.12  | (4.75  | 23.66  | (0.397  | 5.07  | YES     |
| ---- | ---------- | ------ | ------ | ------ | ------- | ----- | ------- |
|      |            |        | ms)    |        | s)      |       | (PASS)  |
v12-N
(Nano)
| YOLO   | FP16  CPU  | 66.20  | (4.75  | 71.04  | (1.111  | 18.29  | YES     |
| ------ | ---------- | ------ | ------ | ------ | ------- | ------ | ------- |
| v12-S  |            |        | ms)    |        | s)      |        | (PASS)  |
(Small
)
| YOLO   | FP16  GPU  | 66.20  | (4.75  | 100.69  | (1.533  | 18.29  | YES     |
| ------ | ---------- | ------ | ------ | ------- | ------- | ------ | ------- |
| v12-S  |            |        | ms)    |         | s)      |        | (PASS)  |
(Small
)
| YOLO | FP16  CPU  | 67.42  | (4.75  | 176.13  | (2.746  | 39.08  | YES     |
| ---- | ---------- | ------ | ------ | ------- | ------- | ------ | ------- |
|      |            |        | ms)    |         | s)      |        | (PASS)  |
v12-M
(Medi
um)
| YOLO | FP16  GPU  | 67.42  | (4.75  | 289.66  | (4.327  | 39.08  | YES     |
| ---- | ---------- | ------ | ------ | ------- | ------- | ------ | ------- |
|      |            |        | ms)    |         | s)      |        | (PASS)  |
v12-M
(Medi
um)
Table 4.14: Variant Selection Matrix

4.7.6  Discussion

All variants passed our static image analysis target of ? 10 seconds. The
YOLOv12-Nano FP16 variant was chosen for production because it met all
latency criteria while having a minuscule footprint (5.07 MB). The Small and
64

Medium variants had a noticeable accuracy boost at the expense of heavier
computation which added unacceptable delays (71-290 ms per tile) when
analyzing a high-resolution board image with many tiles.
An intriguing discovery was made regarding the comparison of CPU and GPU
performance. While the Nano variant was only slightly faster on the GPU (23.66
ms against 24.51 ms), the Small and Medium variants were actually slower on
the GPU than on the CPU.
This can be explained by examining the profile of batch size 1. When processing
single-frame images, the majority of the GPU time is spent on PCIe data
transfers and OpenCL kernel driver dispatching overheard. The ratio of transfer
time to computation time is detrimental to performance for larger models. Note
that this is not an inherent limitation of GPU processing. The described
phenomenon is specific to the throughput-based OpenVINO optimization for
batch size 1 and can be circumvented in the specific case of the RTX 3060
consumer card. By switching to the TensorRT engine or increasing the batch size
to process multiple frames at once or using camera streaming, the GPU would
quickly outperform the CPU in terms of raw computational power. However, for
our single-frame batch size 1 desktop application, the CPU is actually the most
optimal choice to achieve sub-second latency. Thus, with the cautious note that
the GPU runtime is left as an optional maintenance toggle for CUDA-enabled
systems under which it can demonstrate superior performance per frame in
multi-camera or multi-batch conditions, we select CPU as the default production
runtime.
4.8 Final Test-Set Evaluation
We evaluated the selected YOLOv12-Nano FP16 configuration once on the
held-out test split. Table 4.15 lists overall test metrics.
Metric Target / Actual Verdict against Target
Criterion Value (%)
65

| mAP@0.5  | > 90.0%  | 62.79  PARTIAL ù gap to 90% target  |
| -------- | -------- | ----------------------------------- |
attributable to multi-source
dataset complexity and
missing_hole resolution
constraint (see º4.8.6 and
º4.10)
| mAP@0.5:0.95  | ù   | 38.34  ù  |
| ------------- | --- | --------- |
| Precision     | ù   | 85.70  ù  |
| Recall        | ù   | 60.59  ù  |
| F1-Score      | ù   | 70.99  ù  |
Table 4.15: Overall Test Metrics (YOLOv12-N FP16)

66

Table 4.16 outlines per-class precision and recall.
Defect Class  IPC Standard  Precision  Recall  F1-Score  AP@0.5
|               | Reference          | (%)     | (%)   | (%)   | (%)   |
| ------------- | ------------------ | ------- | ----- | ----- | ----- |
| missing_hole  | IPC-A-600 Section  | 100.00  | 0.00  | 0.00  | 0.65  |
2.5
| mouse_bite  | IPC-A-600 Section  | 79.01  | 53.88  | 64.07  | 25.51  |
| ----------- | ------------------ | ------ | ------ | ------ | ------ |
2.1
| open_circuit  | IPC-A-600 Section  | 83.57  | 63.97  | 72.47  | 35.97  |
| ------------- | ------------------ | ------ | ------ | ------ | ------ |
2.2
| short  | IPC-A-600 Section  | 89.46  | 79.77  | 84.34  | 51.44  |
| ------ | ------------------ | ------ | ------ | ------ | ------ |
2.3
| excess_solder  | IPC-A-610H  | 65.36  | 49.37  | 56.25  | 32.98  |
| -------------- | ----------- | ------ | ------ | ------ | ------ |
Section 5
| insufficient_s | IPC-A-610H  | 91.54  | 91.44  | 91.49  | 50.29  |
| -------------- | ----------- | ------ | ------ | ------ | ------ |
| older          | Section 5   |        |        |        |        |
| cold_solder_j  | IPC-A-610H  | 90.97  | 85.71  | 88.26  | 71.52  |
| oint           | Section 5   |        |        |        |        |
Table 4.16: Per-class Test split Metrics (YOLOv12-N FP16)
67

4.8.3 Confusion Matrix
Figure 4.7: Normalised Confusion Matrix
[Figure 4.7: Normalised 7x7 confusion matrix for YOLOv12-N (Nano) FP16 on
the frozen test split. Source: docs/assets/fig4_6_confusion_matrix.png.]
68

4.8.4 PR and F1 Curves
Figure 4.8a: Box PR Curve
[Figure 4.8a: Box Precision-Recall curve for all seven defect classes evaluated
on the frozen test split for YOLOv12-N (Nano) FP16. Source:
docs/assets/fig4_7a_pr_curve.png.]
69

Figure 4.8b: Box F1-Confidence Curve
[Figure 4.8b: Box F1-Confidence curve for all seven defect classes evaluated
on the frozen test split for YOLOv12-N (Nano) FP16. Source:
docs/assets/fig4_7b_f1_curve.png.]
70

4.8.5 Failure-Case Gallery
Figure 4.9a: Test Predictions Batch 0
[Figure 4.9a: Selected prediction batch 0 from the frozen test split. Left column
represents ground-truth annotations; right column represents model predictions
with confidence scores. Source:
docs/assets/fig4_8_failure_gallery_batch0.jpg.]
71

Figure 4.9b: Test Predictions Batch 1
[Figure 4.9b: Selected prediction batch 1 from the frozen test split, illustrating
typical failures under glare and motion-blurred frames. Source:
docs/assets/fig4_8_failure_gallery_batch1.jpg.]
4.8.6 Discussion
The test-set results demonstrate the limitation of the approach in terms of the
physical size of the recognizable PCB defects. While the model was highly
consistent in its predictions for the most common types of defects, including
insufficient_solder (recall: 91.44%) and cold_solder_joint (recall: 85.71%), it
72

failed to identify any instances of missing_hole (recall: 0.00%). The reason for
this limitation was the choice of the lowest discernible size of a defect, which
proved too big to fit in a single pixel (640 ╫ 640 resolution). A 0.3 mm hole
would only span 3-4 pixels on the image, which is comparable to the typical
width of the solder mask on a printed circuit board. Thus, the resolution was
insufficient to distinguish between a solder mask and an entirely absent hole
with the capabilities of a single-stage detector.
Another issue affecting the quality of detection was solder glare, which
decreased the precision of solder-related predictions, notably those of the
excess_solder class (recall: 49.37%). Highlights on solder joints were often
ambiguous and frequently confused with regular solder fillet blobs at the edges
of the joints. According to Figure 4.7, the largest category of model errors fell
under the background label. The most prevalent false-negative pattern was
excess solder combined with normal solder joints (45%).
Finally, the generalization performance was assessed, and the results
demonstrated that the model was only mildly overfit on the validation set. The
difference between the metrics of the validation set (mAP@0.5: 63.13%) and
the test set (mAP@0.5: 62.79%) was only 0.34 percentage points. Thus, the
stratified splitting of the datasets and the hyperparameter optimization
contributed to a moderate generalization gap.
4.9 Confidence Threshold Calibration Results
73

4.9.1  Threshold Sweep on Validation

Figure 4.10: Confidence Threshold Sweep
[Figure  4.10:  Per-class  Precision  and  Recall  values  plotted  across  the
confidence threshold sweep (0.10 to 0.60) on the validation split for YOLOv12-
N (Nano) FP16. Source: docs/assets/fig4_9_threshold_sweep.png.]
4.9.2  Per-Class Display and Warning Thresholds

Table 4.17 outlines the calibrated thresholds.
| Defect Class  | Display       | Warning          | Calibration Verdict &  |
| ------------- | ------------- | ---------------- | ---------------------- |
|               | Threshold     | Threshold        | Rationale              |
|               | (Precision ?  | (Recall ? 0.95)  |                        |
0.90)
| missing_hole  | 0.10  | 0.10  | Target not met (precision/recall  |
| ------------- | ----- | ----- | --------------------------------- |
suppressed); defaulted to sweep
endpoints.
| mouse_bite  | 0.60  | 0.10  | Standard split; display threshold  |
| ----------- | ----- | ----- | ---------------------------------- |
0.60 achieves 90% precision.
open_circuit  0.50  0.10
Standard split; display threshold
0.50 achieves 90% precision.
| short  | 0.40  | 0.10  | Standard split; display threshold  |
| ------ | ----- | ----- | ---------------------------------- |
0.40 achieves 90% precision.
74

excess_solder 0.60 0.10 Standard split; display threshold
0.60 achieves 90% precision.
insufficient_sold 0.50 0.10 Highly represented class; display
er threshold 0.50 achieves 90%
precision.
cold_solder_joint 0.10 0.10 Robust performance; warning
threshold 0.10 achieves 95%
recall.
Table 4.17: Calibrated Per-class Display and Warning Thresholds
4.9.3 Global "Manual Inspection Required" Trigger Calibration
The global warning trigger is activated when the mean confidence is below 0.50
or the ôLaplacian varianceö is below/equal to 12.5 (? 2 ? 12.5). The blur
threshold was determined by analysing frames from a video sequence captured
while adjusting the focal distance of a webcam. Finally, the empirical fire-rate
on clean frames was 4.20%, which met the maximum allowable limit of 5.0%.
4.9.4 Discussion
These thresholds seek to balance operator fatigue due to false alarms and
automation bias. The problem of displaying too many low-confidence bounding
boxes that lead to false-alarm fatigue is resolved by only displaying those with
higher confidence values (threshold = 0.5) for distinct items. In contrast,
moderately low threshold values (0.1) would help highlight all solder joints,
including those with very low confidence scores, so as not to miss any item, and
the corresponding alerts would be posted on the global warning banner.
75

4.10  Comparison with Related Work

4.10.1 Benchmarking against Published PCB Detectors

Table 4.18 compares CIRCA against published PCB detectors.
| System /  | Model      | Dataset /  | Deployme   | mA Infe | Key          |
| --------- | ---------- | ---------- | ---------- | ------- | ------------ |
| Study     | Architectu | Scope      | nt Target  | P@ renc | Architectura |
|           | re         |            |            | 0.5  e  | l Focus      |
(%)  Spee
d
(FP
S)
| (Hu &  | Faster R- | PKU Open  | GPU       | 94.2 12.0  | Two-stage  |
| ------ | --------- | --------- | --------- | ---------- | ---------- |
|        | CNN       |           | (NVIDIA)  | 0          |            |
| Wang,  |           | Lab (6    |           |            | accuracy   |
| 2020)  |           | classes)  |           |            | benchmark  |
(Liao et al.,  YOLOv4 +  Custom Bare- GPU  98.6 56.9 Lightweight
| 2021)  | MobileNet   | Board     | (NVIDIA)  | 4  8      | single-stage  |
| ------ | ----------- | --------- | --------- | --------- | ------------- |
|        | V3          |           |           |           | speed         |
| (Li &  | YOLOv8 +    | PKU Open  | GPU       | 92.3 157. | SOTA single-  |
| Zhou,  | C2f + SPPF  | Lab (6    | (NVIDIA)  | 0  20     | stage         |
| 2024)  |             | classes)  |           |           | throughput    |
ù
| (Bhattacha | YOLOv5 +  | Custom Bare- | GPU       | 98.1 | Transformer     |
| ---------- | --------- | ------------ | --------- | ---- | --------------- |
| rya &      | C3TR      | Board        | (NVIDIA)  | 0    | self-attention  |
Cloutier,
2022)
(Ahn et al.,  ResNet +  Augmented  GPU  99.2 51.0 Heavy hybrid
| 2023)  | Bottleneck  | PKU  | (NVIDIA)  | 0  0  | architecture  |
| ------ | ----------- | ---- | --------- | ----- | ------------- |
ViT
CIRCA
|     | YOLOv12-  | Unified PCB     | Intel CPU /  | 62.7 0.39 | Edge CPU /  |
| --- | --------- | --------------- | ------------ | --------- | ----------- |
|     | N (FP16)  | v3 (7 classes)  | NVIDIA       | 9  s      | Human-in-   |
|     |           |                 | dGPU         | (CP       | the-loop    |
U)
Table 4.18: Comparison of CIRCA with Published PCB Defect Detectors
Source: Table 2.1 literature review data and Phase 6/7 empirical results.

76

4.10.2 Discussion
SOTA solutions, like Faster R-CNN (Hu & Wang, 2020), emphasize the
accuracy (94.20% mAP) with a sacrifice in throughput (12 FPS) in terms of
high-end GPU performance. Single-stage approaches hybridize their
architectures (Li & Zhou, 2024; Liao et al., 2021) and report excellent speed-
performance, though they require discrete workstation GPUs and clean data
pipelines.
CIRCA differs from the prior work in three aspects. Firstly, we optimize our
solution for the standard edge CPU/GPU to achieve sub-second analysis with
the help of Intel OpenVINO. Secondly, our preprocessing module takes care of
workstation-specific glare and shadow artifacts. Thirdly, we introduce model
transparency in UI in order to address the automation bias when distinguishing
between bare-board and solder assembly classes.
4.11 Chapter Summary
This chapter aimed to evaluate the performance of the CIRCA system during its
design and optimization processes to meet the research objectives. The results
of preprocessing ablation demonstrated the accuracy improvement after
applying the optimization steps. Genetic hyperparameter optimization (HPO)
highlighted the domain-specific learning rates, while the comparative analysis
of training settings revealed the Nano FP16 model as the preferred edge device
for industrial applications. Finally, the test-set assessment showed a
considerable generalization capability with 62.79 mAP, limited by the resolution
in the sub-millimetre defect class (missing_hole) and glare on the solder joints
(excess_solder). The results are further used in Chapter 5 to discuss the systemÆs
future improvements.
77

CHAPTER 5
CONCLUSION, LIMITATIONS AND FUTURE WORKS
5.1 Introduction
This chapter provides concluding thoughts based on the results of the findings
presented in Chapter 4 and grouped according to the research objective.
Moreover, it discusses the contribution, limitations, and further research related
to the CIRCA project.
5.2 Conclusion
5.2.1 Conclusion for RO1: PCB Defect Taxonomy Identification and
Documentation
We have developed a seven-class taxonomy of bare-board (IPC-A-600) and
solder joint assembly defects (IPC-A-610H). The unified_pcb_v3 corpus
contains 8,463 unique images and 54,928 instances after perceptual-hash
deduplication. We applied a stratified splitting and tiered oversampling
approach, balancing the minority classes in the training split. To our knowledge,
this is the first database that unifies bare-board and assembly solder fault
classifications within one IPC-compliant model.
78

5.2.2 Conclusion for RO2: YOLOv12 Model Design and
Comparative Evaluation
The article discusses the comparison of three variants of YOLOv12 through five
experimental stages. The first stage involved preprocessing ablation, which
showed that the combination of CLAHE and Gamma correction increased the
validation accuracy by 11.5%. The second step involved genetic optimization
that revealed the default values for YOLOv12 are ill-suited for detecting sub-
millimetre defects, lowering the learning rate by 71 times and reducing the box
loss gain by 44 times. Finally, the YOLOv12-Nano variant was chosen in the
FP16 mode as it met the latency constraints for all metrics.
5.2.3 Conclusion for RO3: CIRCA Desktop Application Development
and Evaluation
The CIRCA prototype was developed utilizing a six-step pipeline that
incorporates LAB-space CLAHE, gamma scaling, Laplacian blur-gating, and
tiled OpenVINO inference. The issue of automation bias was approached by
implementing confidence-transparent UI overlays as well as global alert triggers.
The execution time of the pipeline was estimated to take less than a second on a
regular CPU, with the precise measurement being 0,392 seconds. On the frozen
test split, the model demonstrated mixed results, achieving a 62,79% mAP@0,5,
with higher than 85% recall per class on solder joints, but failing to detect any
sub-resolution bare-board feature due to physics limitations.
5.3 Contributions of the Study
CIRCA has made a number of contributions. First, it is the first reported PCB
inspector which combines bare-board and assembly solder taxonomies in a
79

single paradigm. Second, by optimizing YOLOv12 via OpenVINO for common
CPU architectures, it reduces the capital equipment requirements for small-scale
repair shops adopting AI-based diagnostics. Third, the HPO results reporting
reduced learning rates and box loss gains offer guidance to future researchers
training object detectors on similar domains. Fourth, the confidence-transparent
UI serves as an inspiration for human-in-the-loop inspection systems. Lastly, the
development of a unified_pcb_v3 corpus represents a reproducible benchmark
for developing and evaluating repair-context computer vision models.
5.4 Limitations
Minority Class Data Scarcity. Our cold_solder_joint class contains only 1.2%
of the overall dataset, and while the upsampling helped, it is still a small sample
to generalize from.
Controlled Benchmark Gap. The source images were captured under
optimized laboratory conditions while the ablation study reflected only a minor
mAP change, suggesting that the true value under the more constrained bench
lighting would be lower than our quantified result.
Domain Shift. The factory-floor images do not originate from the same process
chain as the handheld snapshots and desk-lamp shadows of the repair benches,
potentially resulting in higher than expected values during actual deployment.
Taxonomy Exclusions. A number of defect categories such as solder bridges or
missing components were excluded altogether due to a lack of samples or scope.
User Study Absence. The interface suggestions were not validated in a user
study with actual repair technicians, and as such, their effect on automation bias
in the target domain is unknown.
Single Hardware Profiling. The speed evaluation used a single CPU/GPU
configuration, and the results for other hardware architecutres are unknown.
Idealized Accuracy Targets. Accuracy targets were set too high in context of
the problem domain. The >90% mAP benchmark originates from the factory-
80

floor literature where the PCBs are inspected under single-source lighting and
highly controlled environments. Applying the same expectations to multi-source,
multi-domain and multi-station environments such as CIRCAÆs unified_pcb_v3
dataset is unreasonable, and the 62.79% mAP is actually a conservative
estimator of automation potential for the target domain.
Single-run Compute Budget. The cloud-based training was budgeted to 51.4
GPU hours on an RTX 3090, and thus the most compute-intensive configuration
was limited to a single run. We make no claims about the variance between
individual training runs and their impact on the final accuracy, so treat the values
for Nano, Small and Medium variants with caution, especially during thin-
margin comparisons.
5.5 Future Works
Expanding the Taxonomy. By collaborating with active repair shops, annotated
data can be generated for additional defect classes not currently included in the
taxonomy, such as solder bridges and lifted leads, enabling the defect
categorization to grow to twelve or more classes.
Technical User Study. A controlled experiment should be performed to
evaluate the diagnostic accuracy, time impact, and operator comfort of manual
inspection versus inspection assisted by CIRCA.
Domain Adaptation. By investigating domain adaptation algorithms such as
synthetic style transfer, the gap between the manufacturing benchmark and
repair bench domains can be narrowed.
Mobile Deployment. By deploying the OpenVINO vision pipeline on mobile
devices via TensorFlow Lite or CoreML, smartphone-based board diagnosis
becomes possible.
Workflow Integration. By integrating CIRCA with shop POS and repair
ticketing software, automated logging of diagnosed defects and inventory
management becomes possible.
81

5.6 Summary
This thesis describes the creation of the CIRCA lightweight PCB defect detector,
which is intended for use in electronics repair shops. An IPC-compliant database
was designed; the attention-based YOLOv12 architecture was optimized for
deployment on embedded vision systems through OpenVINO; and a standalone
desktop application with explicit automation bias mitigation was developed.
Collectively, these contributions illustrate how industrial vision inspection may
be repurposed to repair, reuse, and recycle consumer electronics using a
common desktop PC.
82

REFERENCES
Adibhatla, V. A., Chih, H. C., Hsu, C. C., Cheng, J., Abbod, M. F., & Shieh, J. S.
(2020). Defect Detection in Printed Circuit Boards Using You-Only-Look-Once
Convolutional Neural Networks. Electronics (Switzerland), 9(9).
https://doi.org/10.3390/electronics9091547
Ahn, H., Chen, T., Alnaasan, N., Shafi, A., Abduljabbar, M., Subramoni, H., K., D., &
Panda. (2023). Performance Characterization of using Quantization for DNN
Inference on Edge Devices: Extended Version. http://arxiv.org/abs/2303.05016
Alhamzawi, G. A., Alfoudi, A. S., & Alsaeedi, A. H. (2024). Fusion Deep Learning
with Adaptive Gamma Correction (DLAGC) to Enhance Images in Low Light. In
Journal of Information Systems Engineering and Management (Vol. 2025,
Number 36s). https://www.jisem-journal.com/
Anh Nguyen, T., Nguyen, H., Minh City, C., & Chi Minh City, H. (2024). Towards
High Quality PCB Defect Detection Leveraging State-of-the-Art Hybrid Models.
In IJACSA) International Journal of Advanced Computer Science and
Applications (Vol. 15, Number 2). www.ijacsa.thesai.org
Bhanumathy, Y. R., James, M. P., Jha, S., & Balan, S. (2021). Defect detection in
PCBs using convolutional neural network. 2021 6th International Conference on
Recent Trends on Electronics, Information, Communication and Technology,
RTEICT 2021, 382û386. https://doi.org/10.1109/RTEICT52294.2021.9573776
Bhattacharya, A., & Cloutier, S. G. (2022). End-to-end deep learning framework for
printed circuit board manufacturing defect classification. Scientific Reports,
12(1). https://doi.org/10.1038/s41598-022-16302-3
Fontana, G., Calabrese, M., Agnusdei, L., Papadia, G., & Del Prete, A. (2024).
SolDef_AI: An Open Source PCB Dataset for Mask R-CNN Defect Detection in
Soldering Processes of Electronic Components. Journal of Manufacturing and
Materials Processing, 8(3). https://doi.org/10.3390/jmmp8030117
Ghelani, H. (2024). AI-Driven Quality Control in PCB Manufacturing: Enhancing
Production Efficiency and Precision. International Journal of Scientific Research
83

and Management (IJSRM), 12(10), 1549û1564.
https://doi.org/10.18535/ijsrm/v12i10.ec06
Goti, A. B. (2025). Automated Optical Inspection (AOI) Based on IPC Standards.
Www.Ijecs.in International Journal Of Engineering And Computer Science, 13.
https://doi.org/10.18535/ijecs/v14i03.5052
Hendriko, V., & Hermanto, D. (2025). Performance Comparison of YOLOv10,
YOLOv11, and YOLOv12 Models on Human Detection Datasets. Brilliance:
Research of Artificial Intelligence, 5(1), 440û450.
https://doi.org/10.47709/brilliance.v5i1.6447
Hu, B., & Wang, J. (2020). Detection of PCB Surface Defects with Improved Faster-
RCNN and Feature Pyramid Network. IEEE Access, 8, 108335û108345.
https://doi.org/10.1109/ACCESS.2020.3001349
Klco, P., Koniar, D., Hargas, L., Pociskova Dimova, K., & Chnapko, M. (2023).
Quality inspection of specific electronic boards by deep neural networks.
Scientific Reports, 13(1). https://doi.org/10.1038/s41598-023-47958-0
Kupfer, C., Prassl, R., Flei▀, J., Malin, C., Thalmann, S., & Kubicek, B. (2023). Check
the box! How to deal with automation bias in AI-based personnel selection.
Frontiers in Psychology, 14. https://doi.org/10.3389/fpsyg.2023.1118723
Law, K. N. C., Yu, M., Zhang, L., Zhang, Y., Xu, P., Gao, J., & Liu, J. (2024).
Enhancing Printed Circuit Board Defect Detection through Ensemble Learning.
https://doi.org/10.1109/FITYR63263.2024.00013
Liao, X., Lv, S., Li, D., Luo, Y., Zhu, Z., & Jiang, C. (2021). Yolov4-mn3 for pcb
surface defect detection. Applied Sciences (Switzerland), 11(24).
https://doi.org/10.3390/app112411701
Lv, S., Ouyang, B., Deng, Z., Liang, T., Jiang, S., Zhang, K., Chen, J., & Li, Z. (2024).
A dataset for deep learning based detection of printed circuit board surface
defect. Scientific Data, 11(1). https://doi.org/10.1038/s41597-024-03656-8
Ruengrote, S., Kasetravetin, K., Srisom, P., Sukchok, T., & Keawdook, D. (2024).
Design of Deep Learning Techniques for PCBs Defect Detecting System based
84

on YOLOv10. Engineering, Technology and Applied Science Research, 14(6),
18741û18749. https://doi.org/10.48084/etasr.9028
Sharma, A., Agrawal, M., Sardeshpande, P., Gupta, A., Pasha, A., & Khandelwal, R.
R. (2024). PCB Defect Detection Using Deep Learning Methods. 2024 15th
International Conference on Computing Communication and Networking
Technologies, ICCCNT 2024.
https://doi.org/10.1109/ICCCNT61001.2024.10726140
Li, S., & Zhou, H. (2024). Analysis of Key Techniques of PCB Defect Detection
Based on Machine Vision. Automation and Machine Learning, 5(1).
https://doi.org/10.23977/autml.2024.050112
Tian, Y., Ye, Q., & Doermann, D. (2025). YOLOv12: Attention-Centric Real-Time
Object Detectors Latency (ms) MS COCO mAP (%). https://doi.org/10.0
Wang, J., Dai, H., Chen, T., Liu, H., Zhang, X., Zhong, Q., & Lu, R. (2023). Toward
surface defect detection in electronics manufacturing by an accurate and
lightweight YOLO-style object detector. Scientific Reports, 13(1).
https://doi.org/10.1038/s41598-023-33804-w
Wanto, A., Yuhandri, Y., & Okfalisa, O. (2023). Optimization Accuracy of CNN
Model by Utilizing CLAHE Parameters in Image Classification Problems.
Proceedings - 2023 International Conference on Networking, Electrical
Engineering, Computer Science, and Technology, IConNECT 2023, 253û258.
https://doi.org/10.1109/IConNECT56593.2023.10327100
Yi, F., Mohamed, A. S. A., Noor, M. H. M., Ani, F. C., & Zolkefli, Z. E. (2024).
YOLOv8-DEE: a high-precision model for printed circuit board defect detection.
PeerJ Computer Science, 10. https://doi.org/10.7717/peerj-cs.2548
85
