# CIRCA: Circuit Inspection and Recognition Using Convolutional Architectures
### Chapters 1–4 — Reference-Corrected Version (v5, May 2026)
*Muhammad Aidil Al-Faizi Bin Mohd Zin | Bachelor of Information Technology (Hons.) Intelligent Systems Engineering | Universiti Teknologi MARA | January 2026*

***

# CHAPTER 1: INTRODUCTION

This chapter mainly discusses the background of the study. It includes information about automated PCB defect detection in electronics repair environments, issues and problems which lead to the implementation of this research, and an outline of the CIRCA system as the proposed solution.

***

## 1.1 Research Background

In recent years, there has been an increasing interest in the application of artificial intelligence and computer vision technologies to quality inspection and fault diagnosis tasks across a broad range of industries (Bhattacharya and Cloutier, 2022). Among these, the inspection and diagnosis of Printed Circuit Board (PCB) defects has emerged as a particularly active area of investigation, driven by the growing complexity and miniaturisation of modern electronic devices. PCBs serve as the fundamental backbone of virtually all electronic systems, providing both the mechanical substrate and the electrical pathways through which components communicate and function (Wang et al., 2023). As device designs become increasingly compact and component densities continue to rise, ensuring PCB integrity has become a critical quality challenge across the electronics industry.

The rapid advancement of electronic technology has led to an unprecedented increase in the complexity and miniaturisation of electronic devices (Bhattacharya and Cloutier, 2022). PCBs are ubiquitous components found in virtually every electronic device, ranging from smartphones and laptops to medical equipment and automotive systems (Wang et al., 2023). As the electronics industry continues to evolve, the demand for high-quality, defect-free PCBs has intensified significantly, particularly given that even minor defects can compromise device functionality, reliability, and safety (Klco et al., 2023). Traditional manual inspection methods, while historically prevalent, have increasingly proven inadequate in addressing the challenges posed by modern PCB complexity (Aggarwal et al., 2022). The miniaturisation of electronic components, coupled with the increasing density of circuit layouts, has rendered visual inspection by human operators time-consuming, error-prone, and often ineffective in detecting subtle defects (Bhanumathy et al., 2021). Studies have demonstrated that manual inspection methods can exhibit error rates ranging from 10% to 20%, particularly during extended work shifts where human fatigue becomes a significant factor (Law et al., 2024).

The past decade has seen the rapid development of deep learning-based approaches to PCB defect detection, particularly through the application of Convolutional Neural Networks (CNNs) and object detection architectures such as YOLO, ResNet, and EfficientDet (Liao et al., 2021). These advances have demonstrated that automated systems can identify complex defect patterns such as solder bridges, cold joints, and component misalignments with detection accuracies exceeding 95%, while maintaining real-time processing speeds that manual inspection cannot match (Law et al., 2024; Yang and Yu, 2024). The most recent iteration of the YOLO framework, YOLOv12, introduces an attention-centric architecture featuring an Area Attention module (A2) and Residual Efficient Layer Aggregation Networks (R-ELAN), enabling the model to match the inference speed of CNN-based predecessors while surpassing them in accuracy through improved feature modelling capabilities (Tian et al., 2025). YOLOv12-Nano, the lightest variant in the family, achieves 40.6% mAP on the MS COCO benchmark, demonstrating that factory-grade detection precision is now attainable on standard CPUs and integrated GPUs without dedicated graphics hardware (Tian et al., 2025).

The limitations of manual inspection have catalysed the development and adoption of automated inspection systems leveraging machine learning and computer vision technologies (Anh Nguyen et al., 2024). Automated Optical Inspection (AOI) systems have gained widespread acceptance in high-volume manufacturing environments, utilising advanced imaging techniques and sophisticated algorithms to detect defects with unprecedented accuracy and speed. The integration of artificial intelligence in PCB inspection represents a paradigm shift from reactive quality control to proactive defect prevention, enabling manufacturers to identify and rectify issues early in the production process (Ghelani, 2024). However, a major challenge with existing automated PCB inspection solutions is that they have been developed predominantly for controlled, high-volume manufacturing environments and are ill-suited to the more variable and resource-constrained conditions encountered in electronics repair settings (Aggarwal et al., 2022). The deployment of existing AOI systems also requires significant capital expenditure, specialised operators, and stable controlled lighting — conditions beyond the resources of the vast majority of independent repair shops.

This research addresses this identified gap by proposing CIRCA (Circuit Inspection and Recognition using Convolutional Architectures), an AI-driven visual inspection system that leverages YOLOv12 variants to automatically detect and localise surface-level PCB defects such as cold solder joints, excess and insufficient solder, solder spikes, open circuits, shorts, conductor damage, and bare-board substrate anomalies from standard camera images. By deploying a YOLOv12 model quantized to INT8 Intermediate Representation (IR) format via the Intel OpenVINO inference framework, CIRCA achieves sub-10-second diagnostic turnarounds on standard Intel CPUs and integrated GPUs, eliminating the need for expensive AOI hardware (Yi and Mohamed, 2024). An ultra-lightweight OpenCV preprocessing pipeline combining Contrast Limited Adaptive Histogram Equalization (CLAHE), Gamma Correction, and Laplacian Variance blur detection ensures robust inference performance under the uncontrolled lighting conditions typical of real-world repair shop environments (Alhamzawi et al., 2025; Wanto et al., 2023).

***

## 1.2 Problem Statement

The electronics repair industry faces several interconnected challenges that significantly impact service quality, operational efficiency, and customer satisfaction. These challenges stem primarily from the limitations inherent in current manual inspection methodologies and the increasing complexity of modern electronic devices.

Firstly, the detection of defects in modern PCBs through manual visual inspection has become increasingly problematic due to progressive component miniaturisation and elevated circuit density (Goti, 2025). Contemporary PCBs frequently incorporate surface-mount technology (SMT) components measuring less than 1 millimetre, fine-pitch integrated circuits with pin spacing below 0.5 millimetres, and multi-layer board constructions with internal connections invisible to surface inspection (Adibhatla et al., 2020). Under these circumstances, even experienced technicians equipped with magnification tools struggle to identify subtle defects such as hairline cracks, micro-solder bridges, or incipient component failures. The industry standard for defining and categorising PCB assembly acceptability is IPC-A-610 (Acceptability of Electronic Assemblies), which specifies visual criteria for solder joint quality, component placement, and assembly cleanliness across three product reliability classes (Goti, 2025). A comparative study demonstrated that AI-driven AOI systems aligned with IPC standards achieve 98 to 99% inspection accuracy and process over 5,000 components per hour, compared to 85 to 90% accuracy and 500 to 800 components per hour for manual inspection, underscoring the scale of the performance gap between automated and manual approaches (Goti, 2025).

Secondly, manual inspection processes are inherently time-consuming and subject to human limitations. A comprehensive visual examination of a complex PCB can require 10 to 30 minutes or more, depending on board complexity and the number of components present. This extended inspection time directly translates to increased repair costs, longer customer wait times, and reduced throughput for repair facilities. Moreover, human inspectors are susceptible to fatigue, distraction, and subjective judgment variations, which can lead to missed defects or false positives. Research indicates that inspector accuracy declines significantly after extended periods of continuous inspection, with defect detection rates dropping by up to 30% after four hours of sustained work (Goti, 2025). Manual PCB inspection using stereomicroscopes and optical magnification aids is also associated with severe eye strain during prolonged inspection shifts, presenting a physical occupational burden that automated systems can directly eliminate.

Thirdly, the reliability and consistency of manual inspection are heavily dependent on individual technician expertise and experience levels (Sharma et al., 2024). Novice technicians may lack the knowledge to recognise certain defect types or understand their implications, while even experienced professionals may overlook defects in unfamiliar board designs or when confronting novel failure modes. This variability in inspection quality creates inconsistencies in service delivery and can erode customer confidence in repair services.

Furthermore, certain categories of PCB defects present specific challenges for visual detection. Internal solder joint failures, delamination within multi-layer boards, intermittent connection problems, and incipient component degradation may not manifest obvious visual indicators during static inspection (Hu and Wang, 2020). These defects can evade detection during manual inspection yet subsequently cause device malfunction, necessitating repeated repair attempts and generating customer dissatisfaction. In critical applications such as medical devices, automotive electronics, or industrial control systems, undetected defects can pose serious safety hazards or result in catastrophic system failures (Liao et al., 2021).

Current automated inspection technologies, while highly effective in manufacturing environments, are generally designed for high-volume production scenarios and are not optimally configured for the diverse range of devices and repair contexts encountered in service centres (Huang et al., 2023). Existing AOI systems are often expensive, require specialised training to operate, and may lack the flexibility to accommodate the variety of PCB types and device configurations typical of repair workflows (Kaewdook et al., 2024). Consequently, there exists a clear need for accessible, cost-effective, and adaptable automated inspection solutions tailored specifically to the requirements of electronics repair and service operations.

This research therefore addresses the following core problem: **How can machine learning-based automated defect detection systems be effectively designed and implemented to enhance the speed, accuracy, and reliability of PCB fault diagnosis in electronics repair environments, thereby improving service quality and operational efficiency while minimising the impact of human error and expertise variability?**

***

## 1.3 Research Questions

To comprehensively address the identified problem and guide the research investigation, this study seeks to answer the following research questions:

**RQ1:** How to identify and categorise common PCB defect types based on their visual characteristics, aligned with both IPC-A-600 bare-board and IPC-A-610 assembly-stage industry acceptability criteria, for automated detection in electronics repair contexts?

**RQ2:** How to design and develop YOLOv12-based CNN detection models incorporating INT8 quantization via Intel OpenVINO and an OpenCV preprocessing pipeline that achieves accurate and efficient PCB defect detection under real-world repair shop conditions?

**RQ3:** How to develop and evaluate a user-friendly CIRCA desktop prototype that supports zero-friction, real-time deployment in electronics repair facilities?

***

## 1.4 Research Objectives

Based on the research questions formulated above, this study establishes the following specific objectives:

**RO1:** To identify, categorise, and document the most prevalent PCB defect types encountered in electronics repair based on their characteristic visual signatures, aligned with IPC-A-600 bare-board and IPC-A-610 assembly-stage acceptability guidelines, suitable for automated optical detection.

**RO2:** To design and develop YOLOv12-based CNN detection models (Nano, Small, Medium variants), exported to Intel OpenVINO INT8 IR format, and to conduct comparative performance evaluation to identify the optimal configuration for PCB defect detection in repair contexts under uncontrolled lighting conditions.

**RO3:** To develop the CIRCA standalone desktop application incorporating a live webcam feed, a real-time OpenCV preprocessing pipeline (CLAHE, Gamma Correction, Laplacian Variance frame-dropping), and a zero-friction bounding box overlay interface, and to evaluate its performance using precision, recall, F1-score, mAP, and inference latency benchmarks.

***

## 1.5 Research Scope

To ensure focused and achievable research outcomes within the Final Year Project timeframe, this study establishes the following scope boundaries.

**Technical Scope.** The research focuses specifically on visual defect detection in PCBs using optical imaging and convolutional neural network analysis. The study encompasses the detection and classification of surface-visible defects that can be identified through optical inspection, organised under a unified 12-class taxonomy aligned with both the IPC-A-600 (bare-board) and IPC-A-610 (assembly-stage) standards. The taxonomy covers six bare-board defect classes (missing hole, mouse bite, open circuit, short, spur, and spurious copper), four assembly-stage solder defect classes (excess solder, insufficient solder, solder spike, and cold solder joint), and two additional IPC-A-600 surface defect classes (scratch and pinhole). Component-level IPC-A-610 defects such as missing components, misaligned components, tombstoning, lifted leads, solder balls, and component damage are excluded from the present scope due to the absence of suitable public datasets with sufficient coverage, with this limitation documented and a future-work data-collection campaign proposed in Chapter 3 §3.4.1.5. Additionally, `solder_bridge` is excluded despite being an IPC-A-610H §5 defect, as no board-level annotated public dataset meeting the project's quality criteria was identified; this gap is documented as future work in Chapter 5. The technical implementation focuses on deep learning approaches, specifically YOLOv12 variants deployed via Intel OpenVINO INT8 quantization. The research does not extend to electrical testing, functional verification, or the detection of defects requiring X-ray, infrared, or other specialised imaging modalities beyond standard optical photography.

**Device Scope.** The research concentrates on PCBs commonly found in consumer electronics devices, particularly those frequently encountered in repair shops, including smartphones, laptops, tablets, and similar portable electronic devices. The study does not address specialised industrial electronics, aerospace systems, medical devices, or other domain-specific applications that may require different detection criteria or certification requirements.

**Defect Scope.** The investigation focuses on manufacturing defects, physical damage, and wear-related failures that manifest visually on PCB surfaces, encompassing both bare-board structural defects aligned with IPC-A-600 acceptability criteria and solder-joint assembly defects aligned with IPC-A-610 visual inspection criteria. The research does not address intermittent faults, software-related issues, or defects that can only be detected through electrical testing or operational verification.

**Operational Scope.** The system is designed for use by repair technicians in small to medium-sized electronics repair facilities, targeting standard Intel CPU and iGPU hardware operating under Windows 10 and Windows 11. Standard USB webcams or smartphone camera tethers are assumed for image capture rather than specialised industrial imaging equipment. The system's performance will be evaluated against target benchmarks of greater than 90% mAP on the curated test dataset, sub-5 ms preprocessing pipeline execution per frame, real-time operation at a minimum of 15 FPS, and sub-10-second static image inference on a benchmark Intel Core i5 8th-generation equivalent processor (Yi and Mohamed, 2024).

**Limitations and Exclusions.** The research does not attempt to develop a commercially deployable product with full production-level features. The CIRCA system serves as a diagnostic aid and decision-support tool rather than an autonomous repair solution, with confidence score transparency built in to reduce the risk of automation bias (Goddard et al., 2011; Kupfer et al., 2023). Final repair decisions, component selection, and quality verification remain the responsibility of qualified technicians.

***

## 1.6 Significance of Study

This research makes several important contributions to both academic knowledge and practical application in the fields of computer vision, machine learning, and electronics repair services.

**Academic Significance.** From a research perspective, this study advances the application of deep learning techniques to a relatively underexplored domain — electronics repair diagnostics. While substantial research exists on automated defect detection in manufacturing contexts, the adaptation of these technologies to repair service environments presents unique challenges that have received limited scholarly attention (Bhattacharya and Cloutier, 2022; Lv et al., 2024). The findings should make an important contribution to the field by demonstrating how YOLOv12 models quantized to INT8 IR via Intel OpenVINO can achieve real-time PCB defect detection on commodity hardware, with implications for edge computing applications and mobile machine learning deployment (Tian et al., 2025; Kaewdook et al., 2024).

**Practical Significance.** For electronics repair businesses, CIRCA has the potential to reduce diagnostic time by at least 70%, dropping a 15-minute manual microscope inspection to under five minutes, which translates directly into higher technician throughput, shorter customer wait times, and the economic viability of complex board repairs that are currently abandoned as unprofitable (Goti, 2025). The system also addresses the physical dimension of the problem: by replacing prolonged microscope use with real-time camera-based inspection, CIRCA eliminates the severe eye strain that contributes to inspector fatigue and declining detection performance over the course of a repair shift. For novice technicians, the system serves as an educational tool and decision support aid, accelerating the learning curve and improving diagnostic confidence.

**Economic Significance.** The implementation of automated defect detection has important economic implications for the electronics repair industry. By reducing dependence on highly experienced technicians for routine diagnostics, repair shops can optimise their staffing models and reduce labour costs. The reduction in misdiagnosis and unnecessary component replacements represents direct cost savings through reduced parts wastage and more efficient inventory management (Goti, 2025). For larger repair chains, these savings can accumulate to significant annual cost reductions while simultaneously improving customer satisfaction and retention.

**Environmental and Social Significance.** From a sustainability perspective, improved repair diagnostics contribute to the circular economy by extending device lifespans and reducing electronic waste. Accurate defect detection enables more effective repairs, preventing premature device replacement and the associated environmental impacts of manufacturing new devices and disposing of old ones. By reducing the technical barriers and capital requirements for high-quality diagnostics, the system can help level the playing field between small independent repair shops and larger service chains, supporting local businesses and employment.

***

## 1.7 Summary

In summary, correctly identifying and classifying PCB defects is essential in order to reduce problems such as missed diagnosis, technician fatigue, repeated failures, and inconsistent repair quality. The proposed CIRCA system is an AI-based diagnostic desktop application which utilises YOLOv12 deep learning architecture quantized via Intel OpenVINO INT8 to provide real-time defect detection with colour-coded bounding box overlays, confidence scores, and a configurable OpenCV preprocessing pipeline that neutralises the glare and shadow conditions of real repair environments. Chapter 2 provides a structured review of the literature on PCB inspection, machine vision, deep learning techniques, image preprocessing, edge ML deployment, and human factors, justifying the chosen approach and identifying the research gap that the CIRCA project seeks to fill.

***

***

# CHAPTER 2: LITERATURE REVIEW

This chapter mainly discusses the project's key areas, which include PCB defect detection, machine vision, deep learning and object detection architectures, image preprocessing, edge machine learning deployment, and human factors in AI-assisted inspection. It reviews techniques applied in the project and related works gathered from recent research papers, journal articles, and conference proceedings. The content is organised from general to specific: beginning with an overview of PCB defects and inspection challenges, moving to deep learning architectures and YOLO-based methods, then examining preprocessing techniques and edge deployment, and concluding with human factors considerations and the research gap addressed by CIRCA.

***

## 2.1 PCB Defects and Inspection Challenges

PCB defect detection has been an object of research for more than two decades, with machine vision systems first being applied to quality control in electronics manufacturing in the early 2000s (Adibhatla et al., 2020). Over the past decade, most research in this area has emphasised the use of deep learning approaches, reflecting the rapid advances in computational power and the availability of large labelled datasets that have made such methods practical (Bhattacharya and Cloutier, 2022). PCBs can exhibit a wide variety of surface-visible defects arising from manufacturing process variations, handling damage, thermal stress, and operational degradation. Common defect categories include solder bridges, open circuits, cold solder joints, missing components, misaligned components, damaged or broken traces, and burnt or discoloured components (Adibhatla et al., 2020; Liao et al., 2021).

Two IPC standards jointly define the visual quality criteria underpinning the CIRCA taxonomy. IPC-A-600 (Acceptability of Printed Boards) specifies acceptable conditions for bare-board features including hole quality, conductor integrity, laminate quality, and surface conditions, providing the definitional basis for bare-board defect classes such as missing hole, mouse bite, open circuit, short, spur, and spurious copper (IPC International, 2020). IPC-A-610 (Acceptability of Electronic Assemblies) is the most widely adopted industry reference for defining visual quality criteria at the assembly stage, specifying acceptable versus non-acceptable conditions for solder joint formation, fillet quality, component placement, and cleanliness across three product reliability classes: Class 1 for general electronics, Class 2 for dedicated service electronics, and Class 3 for high-reliability electronics such as medical or aerospace applications (Goti, 2025). In repair contexts, both standards together provide a comprehensive and industry-recognised basis for structuring defect classes in automated detection systems, ensuring that model outputs are grounded in accepted quality definitions rather than arbitrary categories (Klco et al., 2023; IPC International, 2020). A comparative study by Goti (2025) demonstrated that AI-driven AOI systems achieving IPC standard compliance can detect defects at 98 to 99% accuracy while processing over 5,000 components per hour, providing a strong quantitative basis for the performance targets adopted in this research.

Traditional machine vision approaches to PCB inspection relied on classical image processing such as thresholding, edge detection, morphological filtering, and template matching against a golden reference board. While computationally efficient, such methods are highly sensitive to illumination variation, board misalignment, and noise, and require substantial manual configuration for each new product or defect type (Sharma et al., 2024). These approaches are unsatisfactory in repair contexts because they cannot generalise across the heterogeneous mix of board designs and fault types typically encountered. A comprehensive review of deep learning approaches to PCB defect detection by Lv et al. (2024) confirmed that publicly available PCB defect datasets, including the widely-used HRIPCB and DeepPCB benchmarks, primarily capture manufacturing-line scenarios and do not adequately represent the diversity of fault types and imaging conditions encountered in repair settings, indicating a significant dataset gap that limits the direct applicability of current models to repair contexts (Lv et al., 2024).

***

## 2.2 Machine Learning and CNN-Based PCB Defect Detection

A large and growing body of literature has investigated the application of machine learning and deep learning to PCB defect detection, reflecting both the limitations of traditional rule-based approaches and the demonstrated capability of learned feature representations to capture complex visual patterns (Bhattacharya and Cloutier, 2022). Previous research has indicated that CNN-based approaches substantially outperform classical machine vision methods in terms of detection accuracy, robustness to imaging variation, and scalability to new defect categories (Hu and Wang, 2020). In a study which set out to determine the feasibility of basic CNN architectures for PCB classification, Adibhatla et al. (2020) found that a four-layer CNN achieved approximately 70% accuracy on a controlled dataset, establishing a baseline that illustrated the need for more sophisticated architectures capable of handling the diversity and subtlety of real-world PCB defects.

In contrast to this early work, Hu and Wang (2020) proposed an improved two-stage detector based on Faster R-CNN with a ResNet50 Feature Pyramid Network backbone and a Guided Anchor Region Proposal Network, achieving an mAP of 94.2% at approximately 12 frames per second on the PKU Open Lab dataset. While substantially more accurate, the two-stage architecture imposed significant computational overhead that limits suitability for real-time deployment on standard hardware. Law et al. (2024) took a different approach, combining multiple architectures including EfficientDet, MobileNet SSDv2, Faster R-CNN, and YOLOv5 through an ensemble framework with hybrid voting, achieving 95% overall accuracy and 80.3% mAP while improving robustness to noisy input images. However, the ensemble approach considerably increases inference time, creating a practical barrier to deployment in time-sensitive repair workflows. These studies collectively demonstrate that while CNN-based approaches represent a substantial advance over traditional methods, the trade-off between accuracy and computational efficiency remains a central design consideration for systems intended for practical deployment.

***

## 2.3 YOLO-Based PCB Defect Detection

YOLO ("You Only Look Once") is a family of single-stage real-time object detectors that has been widely applied to PCB inspection tasks, with successive versions introducing architectural improvements including better feature pyramid networks, refined loss functions, and more efficient backbone designs (Yang and Yu, 2024). In an early study examining YOLO variants for industrial AOI, Adibhatla et al. (2020) demonstrated that Tiny-YOLOv2 could achieve 98.79% accuracy on a dataset of over 11,000 PCB images while keeping model size below 50 MB, confirming that compact YOLO variants are viable for resource-limited deployments. Liao et al. (2021) further improved on this by replacing the YOLOv4 backbone with MobileNetV3, achieving 98.64% mAP at 56.98 FPS while significantly reducing GFLOPs, a finding that directly supports the feasibility of lightweight YOLO-based inspection on standard computing hardware. Bhattacharya and Cloutier (2022) incorporated transformer attention into the YOLOv5 neck via a C3TR module, achieving 98.1% mAP on a custom PCB dataset and demonstrating that attention mechanisms improve detection of small and complex defect patterns.

Wang et al. (2023) proposed ATT-YOLO, a self-attention-enhanced YOLOv5 variant evaluated on the LCFC laptop PCB dataset containing 14,478 annotated defects, achieving 92.8% mAP at 111 FPS. Yang and Yu (2024) further validated YOLOv8 with C2f modules and SPPF, achieving 92.3% mAP at 157.2 FPS on the PKU Open Lab dataset. More recently, Kaewdook et al. (2024) evaluated YOLOv10-based architectures for PCB defect detection under resource-constrained conditions, confirming that the latest YOLO variants maintain strong performance while remaining deployable on standard desktop hardware without dedicated GPUs. Yi and Mohamed (2024) extended this line of work through YOLOv8-DEE, a high-precision model that enhances small-defect detection on PCBs using multi-scale feature extraction and data enhancement strategies, achieving state-of-the-art performance on benchmark PCB datasets.

The latest member of the YOLO family, YOLOv12, introduces an attention-centric architecture built around two key innovations: the Area Attention module (A2), which divides feature maps into segments to preserve a large receptive field while reducing the quadratic complexity of conventional self-attention, and Residual Efficient Layer Aggregation Networks (R-ELAN), which address the optimisation challenges introduced by attention layers (Tian et al., 2025). YOLOv12 surpasses all popular real-time object detectors in accuracy with competitive speed: YOLOv12-N achieves 40.6% mAP on MS COCO and YOLOv12-S achieves 48.0% mAP, both at latencies competitive with YOLOv11 and YOLOv10 (Tian et al., 2025). A comparative study by Hendriko and Hermanto (2025) benchmarking YOLOv10, YOLOv11, and YOLOv12 across eight datasets found that YOLOv12 achieved the best results on the MOT17 dataset with 0.909 precision and 0.880 mAP@50, confirming that the attention-based architecture delivers superior detection performance on complex, real-world scenes. CIRCA exploits these properties by implementing and comparing YOLOv12-N, YOLOv12-S, and YOLOv12-M variants to identify the optimal balance of accuracy, inference speed, and resource consumption for repair-shop deployment.

***

## 2.4 Image Preprocessing for Robust Detection

A key aspect of deploying object detection models in uncontrolled environments is the design of an effective preprocessing pipeline that compensates for the imaging variability inherent in real-world settings. Recent developments in the field of image preprocessing for deep learning have led to renewed interest in adaptive contrast and illumination enhancement methods, particularly for applications where lighting conditions cannot be controlled (Alhamzawi et al., 2025). Three techniques are of particular relevance to CIRCA: Contrast Limited Adaptive Histogram Equalization (CLAHE), Gamma Correction, and Laplacian Variance blur detection.

CLAHE improves low-contrast images by dividing them into small local regions and equalising the histogram of each region individually, while limiting contrast amplification to prevent noise over-enhancement (Wanto et al., 2023). In the context of deep learning-based inspection, CLAHE has been shown to substantially improve model accuracy by enhancing the visibility of fine surface details that are critical for defect recognition. Wanto et al. (2023) demonstrated that applying CLAHE as a preprocessing step before CNN-based classification produces much higher accuracy results than without using CLAHE, attributing the improvement to CLAHE's ability to increase contrast and eliminate luminance imbalances that strengthen the CNN model's capacity to extract essential features. In CIRCA, CLAHE is applied to neutralise solder glare and restore surface texture visibility under intense workstation lighting.

Gamma Correction is a pixel-level intensity transformation that adjusts the overall luminance of an image, making it an effective tool for recovering detail in heavily shadowed or over-exposed frames. Recent work by Alhamzawi et al. (2025) on deep learning-based adaptive gamma correction demonstrated that a fusion model combining learnable gamma correction with a deep neural network achieves strong performance on low-light enhancement benchmarks, with a PSNR of 17.386, SSIM of 0.788, and FSIM of 0.920 on standard datasets, outperforming fixed-parameter correction methods by generating locally adaptive adjustments that preserve natural colour balance. In CIRCA, Gamma Correction is applied following CLAHE to lift heavy shadow regions created by uneven desklamp illumination, ensuring that the YOLOv12 model receives adequately lit input even when the technician has not optimised their workstation lighting.

The third preprocessing component critical to CIRCA's reliability is blur detection via Laplacian Variance. The Laplacian operator highlights areas of rapid intensity change in an image: a sharp image with many well-defined edges produces a high variance in Laplacian values, while a motion-blurred image with few distinct edges produces a low variance. Lv et al. (2024), in developing a standardised PCB surface defect dataset for deep learning, explicitly identified motion blur and uneven illumination as primary sources of image quality degradation that reduce annotation accuracy and model performance, further confirming the necessity of a frame-quality gating mechanism in real-time PCB inspection pipelines. This property allows the Laplacian Variance to serve as an efficient, parameter-free frame quality gate in CIRCA: frames with a variance below a defined threshold are automatically dropped from the processing queue before entering the inference engine, reducing unnecessary CPU load and ensuring that only frames suitable for confident defect localisation are passed to the YOLOv12 OpenVINO model.

***

## 2.5 Edge Machine Learning Deployment and Model Quantization

Central to the CIRCA architecture is the requirement to deliver real-time deep learning inference on standard Intel CPU and integrated GPU hardware without dedicated graphics cards, a constraint that makes edge ML deployment and model quantization fundamental design considerations. Intel OpenVINO (Open Visual Inference and Neural network Optimization) is a toolkit specifically designed for optimising and deploying deep learning models on Intel hardware, providing hardware-accelerated inference across CPUs, integrated GPUs, and vision processing units through a unified API. OpenVINO's INT8 quantization converts model weights and activations from 32-bit floating-point (FP32) to 8-bit integers (INT8) through a post-training quantization process that requires no retraining, resulting in mixed-precision models that balance high performance with maintained accuracy.

The performance benefits of OpenVINO INT8 quantization have been rigorously characterised in empirical research. Ahn et al. (2023) conducted a comprehensive benchmark study using the MLPerf Edge Inference methodology on Intel x86 processors and found that INT8-quantized models deployed via OpenVINO delivered 3.3 times better performance over FP32 models for the offline inference scenario, with no significant accuracy loss observed for dynamic quantization schemes (Ahn et al., 2023). The study also confirmed that OpenVINO is the most optimised inference framework for Intel CPUs among the frameworks tested, including TensorFlow Lite, ONNX, and PyTorch, further reinforcing its selection as CIRCA's inference engine. This result directly supports the CIRCA design decision to export YOLOv12 models to the OpenVINO INT8 IR format as the primary inference backend, as it provides a quantified basis for expecting near-threefold latency improvements relative to standard FP32 deployment on the same hardware. If INT8 quantization is found to degrade mAP below the 90% threshold during validation, the fallback strategy is FP16 OpenVINO execution, which sacrifices some real-time FPS performance but maintains full diagnostic accuracy.

***

## 2.6 Human Factors and Automation Bias in AI-Assisted Inspection

One of the most significant current discussions in AI-assisted decision support concerns the phenomenon of automation bias: the tendency of human operators to over-rely on automated recommendations, accepting system outputs with insufficient critical scrutiny even when those outputs are erroneous (Goddard et al., 2011). Goddard et al. (2011), reviewing automation bias in clinical decision support systems, found that automation bias is a pervasive and well-documented phenomenon across a wide range of decision-making domains, and that its occurrence is strongly influenced by the degree of trust operators place in the automated system as well as the perceived complexity of the task. In the context of AI-assisted PCB inspection, automation bias presents a concrete risk: if a technician accepts CIRCA's defect detections without applying their own judgement, a false negative could lead to an incomplete repair, subsequent device failure, and potential safety hazards.

More recently, Kupfer et al. (2023) demonstrated in a controlled experiment that less automation bias is associated with significantly higher objective decision quality, and that explicit information about the possibility of system errors and clear human accountability for final decisions are effective strategies for reducing over-reliance on AI recommendations. These findings directly inform CIRCA's interface design: confidence scores are displayed transparently above every bounding box, and the system triggers a visual "Manual Inspection Required" warning indicator when global inference confidence drops below a defined safety threshold. This design ensures that the system functions as a transparent decision-support tool rather than a black-box authority, maintaining human-in-the-loop oversight as a core principle of the system architecture (Goddard et al., 2011; Kupfer et al., 2023).

***

## 2.7 Related Works Summary

**Table 2.1: Summary of Related Works on PCB Defect Detection**

| Author and Year | Model / Technique | Dataset | Key Results | Limitation Relevant to CIRCA |
|---|---|---|---|---|
| Adibhatla et al. (2020) | Tiny-YOLOv2 | Industrial AOI (11,000 images) | 98.79% accuracy, under 50 MB | Higher localisation error on small defects |
| Hu and Wang (2020) | Faster R-CNN + ResNet50-FPN + GARPN | PKU Open Lab | 94.2% mAP, approx. 12 FPS | Computationally heavy; slow inference |
| Liao et al. (2021) | YOLOv4 + MobileNetV3 | Custom PCB dataset | 98.64% mAP, 56.98 FPS | Requires large labelled dataset |
| Bhattacharya and Cloutier (2022) | YOLOv5 + C3TR (Transformer neck) | Custom PCB dataset | 98.1% mAP | Sensitive to distribution shifts |
| Wang et al. (2023) | ATT-YOLO (YOLOv5 + self-attention) | LCFC Laptop (14,478 defects) | 92.8% mAP, 111 FPS | Dataset annotation noise |
| Klco et al. (2023) | YOLOv8n | Power module PCB (640×640 tiles) | 96.6% mAP, 90 ms inference | Context-specific to power modules |
| Yi and Mohamed (2024) | YOLOv8-DEE | PCB benchmark datasets | State-of-the-art multi-scale small-defect detection | Evaluated on benchmark, not repair scenarios |
| Law et al. (2024) | Ensemble (EfficientDet + SSD + RCNN + YOLOv5) | Mixed PCB dataset | 95% accuracy, 80.3% mAP | High total inference time |
| Yang and Yu (2024) | YOLOv8 + C2f + SPPF | PKU Open Lab | 92.3% mAP, 157.2 FPS | Single-model robustness not fully assessed |
| Kaewdook et al. (2024) | YOLOv10 variants | PCB defect dataset | Viable on standard hardware without GPU | Accuracy not yet at YOLOv12 level |
| Anh Nguyen et al. (2024) | ResNet + Bottleneck ViT + FFEM + Wise-IoU | Augmented PKU dataset | 99.2% mAP, 51 FPS | 41.0 GFLOPs: high compute cost |
| Tian et al. (2025) | YOLOv12 (A2 + R-ELAN) | MS COCO | YOLOv12-N: 40.6% mAP; competitive latency | Evaluated on general detection, not PCB-specific |
| Hendriko and Hermanto (2025) | YOLOv10 vs. YOLOv11 vs. YOLOv12 | 8 datasets incl. MOT17 | YOLOv12 best: 0.909 precision, 0.880 mAP@50 | Human detection datasets; not PCB-specific |

**Table 2.2: Preprocessing and Deployment Techniques Underpinning CIRCA Design**

| Technique | Key Study | Finding Relevant to CIRCA |
|---|---|---|
| CLAHE | Wanto et al. (2023) | Increases CNN accuracy by boosting contrast and eliminating luminance imbalance in defect images |
| Adaptive Gamma Correction | Alhamzawi et al. (2025) | PSNR 17.386, SSIM 0.788 on low-light benchmarks; lifts shadow regions without over-exposing highlights |
| Laplacian Variance Blur Detection | Lv et al. (2024) | PCB-specific evidence that motion blur degrades model performance; justifies frame-quality gating |
| OpenVINO INT8 Quantization | Ahn et al. (2023) | 3.3× inference speedup over FP32 on Intel CPUs with minimal accuracy degradation |

***

## 2.8 Summary

In summary, a considerable amount of literature has been published on CNN-based PCB defect detection, and the findings consistently demonstrate that YOLO-based single-stage detectors offer the most favourable balance of accuracy, speed, and deployability (Yang and Yu, 2024; Liao et al., 2021; Tian et al., 2025). Furthermore, research on CLAHE, Gamma Correction, Laplacian Variance, and OpenVINO INT8 quantization collectively establishes a solid technical foundation for the preprocessing and deployment architecture underlying CIRCA. However, the research to date has been mostly restricted to manufacturing-oriented scenarios, and there remains a clear gap in accessible, repair-context PCB inspection systems that combine suitable model architectures, robust preprocessing for uncontrolled environments, edge ML deployment on commodity hardware, and human-factors-aware interface design (Lv et al., 2024; Kaewdook et al., 2024). CIRCA is positioned to address this gap directly, building on the strengths of YOLO-based detection and OpenVINO-accelerated edge inference while adapting the full system to the practical constraints of electronics repair. The methodology used to realise and evaluate this approach is described in Chapter 3.

***
***

# CHAPTER 3: RESEARCH METHODOLOGY

This chapter describes how the CIRCA project was carried out. It covers the research framework that organises the work into discrete phases, the theoretical and empirical activities that supported the design, the system design and development decisions that produced the trained models and the desktop prototype, the experimental design that governed model evaluation, and the hardware and software environment used throughout the project. The methodology adopted is broadly aligned with the design-and-experimentation paradigm typical of applied machine-learning research and is structured to address the three research objectives stated in Chapter 1 §1.4.

***

## 3.1 Introduction

The objective of this chapter is to provide a transparent and reproducible account of every methodological step taken during the CIRCA project. This chapter explains how that gap was addressed in practice through a multi-phase pipeline that progresses from dataset construction, through model training and hyperparameter optimisation, to model quantisation, hardware benchmarking, and final test evaluation. Each phase is described in sufficient detail to enable replication, including the specific tools, parameter values, and decision rules that governed the work.

***

## 3.2 Research Framework

### 3.2.1 Overview of Research Phases

The CIRCA project was structured into eight phases, beginning with environment and dataset setup (Phase 0) and concluding with final test evaluation and confidence threshold calibration (Phase 7). Figure 3.1 illustrates the relationship between the three Research Objectives (RO1–RO3), the activities executed in each phase, and the deliverables produced by each phase. RO1 is satisfied by Phase 0, in which the IPC-A-600 and IPC-A-610-aligned defect taxonomy is established and the unified dataset is constructed. RO2 is addressed by Phases 1 to 5, which together produce three trained YOLOv12 variants and their OpenVINO Intermediate Representation (IR) exports. RO3 is addressed by Phases 6 and 7, which benchmark the candidate variants against the acceptance criteria defined in Chapter 1 §1.5 and calibrate the confidence thresholds used by the deployed CIRCA prototype.

**[Insert Figure 3.1: CIRCA Research Framework — Objectives → Activities → Deliverables. Source: `CIRCA_DIAGRAMS.md`.]**

### 3.2.2 Mapping of Objectives, Activities, and Deliverables

Table 3.1 lists each research objective, the project activities that contribute to it, and the corresponding tangible deliverables produced. This mapping ensures that every phase of the work has an identifiable contribution to the answer of at least one research objective and that no objective is left without verifiable evidence.

**Table 3.1: Mapping of Research Objectives, Activities, and Deliverables**

| Objective | Phase(s) | Key Activities | Deliverables |
|---|---|---|---|
| RO1 — Identify and document IPC-A-600 and IPC-A-610-aligned PCB defect types | Phase 0 | Literature analysis of IPC-A-600 / IPC-A-610; selection of public datasets; class remapping to a unified 12-class taxonomy | `class_mapping.md`; `data.yaml`; defect taxonomy table |
| RO2 — Design and compare YOLOv12-N/S/M with OpenVINO INT8 | Phases 1–5 | Vanilla baseline training; CIRCA-aligned baseline; genetic hyperparameter optimisation; three-variant final training; FP32/FP16/INT8 quantisation validation | `runs/detect/CIRCA_V12{N,S,M}_*/weights/best.pt`; OpenVINO IR exports; `quantization_report.md` |
| RO3 — Develop and evaluate the CIRCA desktop application | Phases 6–7 | Hardware benchmarking on Intel Core i5 8th-gen; live FPS measurement; confidence threshold calibration; test-set evaluation; UI integration | `benchmark_report.md`; `circa_thresholds.yaml`; `test_evaluation.md`; CIRCA desktop prototype |

***

## 3.3 Theoretical Study

### 3.3.1 Preliminary Study

The preliminary study consolidated the findings of Chapter 2, in which the research gap was articulated as the absence of accessible, low-cost, repair-context PCB defect-detection systems combining a suitable model architecture, robust preprocessing for uncontrolled lighting, edge-class deployment on commodity hardware, and human-factors-aware interface design (Lv et al., 2024; Kaewdook et al., 2024). The preliminary study confirmed that public PCB datasets are predominantly captured under controlled manufacturing conditions and concluded that a dataset reflecting repair-context conditions would have to be assembled from multiple complementary public sources rather than a single benchmark.

### 3.3.2 Knowledge Acquisition

Three bodies of knowledge underpin the CIRCA system. First, the IPC-A-610H standard (IPC International, 2020) supplies the visual acceptability criteria used to define the assembly-stage defect classes (excess solder, insufficient solder, solder spike, solder bridge, cold solder joint), while the IPC-A-600K standard (IPC International, 2020) supplies the corresponding criteria for the bare-board defects included in this study (missing hole, mouse bite, open circuit, short, spur, and spurious copper). Second, the YOLOv12 architecture (Tian et al., 2025) provides the detection model family, with its Area Attention module (A2) and Residual Efficient Layer Aggregation Networks (R-ELAN) targeted at small-defect detection on cluttered backgrounds. Third, the Intel OpenVINO toolkit (Ahn et al., 2023) provides the inference framework that enables INT8 deployment on Intel CPUs and integrated GPUs without dedicated graphics hardware. The acquisition of these three knowledge bases informed the design choices documented in §3.5 and §3.6.

***

## 3.4 Empirical Study

### 3.4.1 Data Collection

#### 3.4.1.1 Public bare-board datasets

Three public bare-board datasets were aggregated to cover the IPC-A-600 territory of the taxonomy. The primary source is PKU-Market-PCB-ver1 (Huang et al., 2020), reuploaded to Roboflow Universe as `jr-mqqnk/pcb-defects-detection-anddl` (~1,500 images, six standard bare-board defect categories: `missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, `spurious_copper`). The second bare-board source, `bare-pcb-defects/obj-detection-pcb-defects-yolov8` (~9,666 images, CC BY 4.0), contributes the same six standard classes alongside two additional IPC-A-600 surface defect classes — `scratch` and `pinhole` — extending the taxonomy to include repair-relevant physical damage defects that are absent from PKU. The third source, `rahul-jhj03/pcb-defects-dataset` (CC BY 4.0), covers the same six standard bare-board classes and is included for additional diversity; perceptual-hash deduplication is applied during preprocessing to remove any overlap with PKU. Together, these three sources contribute classes 0–5, 10, and 11 of the unified taxonomy.

#### 3.4.1.2 Public assembly-stage datasets

Five public assembly-stage datasets were assembled to cover the IPC-A-610H solder-defect classes of the unified taxonomy. The primary source, SolDef_AI (Fontana et al., 2024), is an open-source solder-defect dataset of approximately 1,150 images licensed CC BY 4.0 on Kaggle. Because Roboflow exports SolDef_AI with alphabetically sorted class indices, the correct mapping is: raw ID 0 (`exc_solder`) → unified ID 6, raw ID 3 (`poor_solder`) → unified ID 7, and raw ID 4 (`spike`) → unified ID 8; raw IDs 1 (`good`) and 2 (`no_good`) are intentionally omitted and routed to the `negatives_reserve/` directory, corresponding to the Python mapping `{0: 6, 3: 7, 4: 8}`. The remaining four sources are from the `pcb-defect-detection-emmts` Roboflow workspace (PCB Defect Detection, 2025b–f): `excessive-solder-kydra` (1,162 images; `Cold Solder`, `Excessive_solder`, `Insufficient_solder`), `hue-dbgbs-reqtv` (3,232 images; `Insufficient Solder`, `Shorted`), `solder-f8m5i-xnbzp` (`Excessive_solder`), and `pcb-solder-defect-detection-v2-s89jo` (6,116 images; `Cold_solder`, `Excessive_solder`, `Insufficient_solder`). Two previously listed emmts datasets — `pcb-deffect-detection-solder-sthr7` and `pcb-deffect-detection-solder-lsb7m` — were removed following a quality audit: sthr7 contained zero `solder_spike` instances despite documentation, and lsb7m was a single-class source fully superseded by the remaining sources. All Roboflow datasets are exported via the YOLOv8 object detection format (CC BY 4.0).

#### 3.4.1.3 Repair-context capture protocol

To complement the public datasets, a small repair-context capture protocol was defined for use during system testing. Sample boards were imaged with a standard USB webcam at 1280×720 resolution under three lighting conditions: ambient room light only, bright desklamp directed onto the board surface (high glare), and partial occlusion of the lamp creating heavy shadows on one side of the board. These images were not used for training but for end-to-end live FPS testing and qualitative failure-case analysis in Phase 7.

#### 3.4.1.4 Unified 12-class IPC taxonomy

The four source datasets were unified under a single 12-class taxonomy that combines the IPC-A-600 bare-board classes (IDs 0–5) with the IPC-A-610 solder-defect classes (IDs 6–9). Classes 6 to 10 — `excess_solder`, `insufficient_solder`, `solder_spike`, `solder_bridge *(excluded — no suitable public dataset; see future work)*`, and `cold_solder_joint` — correspond directly to defect modes specified in IPC-A-610H §5 (Soldering Acceptability) and were chosen because they have unambiguous visual signatures and adequate image counts in the public datasets. Table 3.2 documents the canonical taxonomy.

**Table 3.2: CIRCA Unified 12-Class IPC Taxonomy**

| Unified ID | Class Name | IPC Reference | Source Datasets | Raw Class Name(s) Remapped From |
|:---|:---|:---|:---|:---|
| 0 | `missing_hole` | IPC-A-600 §3.4 | PKU/JR, Bare PCB | `missing_hole` |
| 1 | `mouse_bite` | IPC-A-600 §3.3 | PKU/JR, Bare PCB | `mouse_bite` |
| 2 | `open_circuit` | IPC-A-600 §3.2 | PKU/JR, Bare PCB | `open_circuit` |
| 3 | `short` | IPC-A-600 §3.2 | PKU/JR, Bare PCB, Hue | `short`, `Short`, `Shorted` |
| 4 | `spur` | IPC-A-600 §3.3 | PKU/JR, Bare PCB | `spur` |
| 5 | `spurious_copper` | IPC-A-600 §3.3 | PKU/JR, Bare PCB | `spurious_copper`, `falsecopper` |
| 6 | `excess_solder` | IPC-A-610H §5 | SolDef_AI, kydra, f8m5i, v2-s89jo | `exc_solder` (α-idx 0), `Excessive_solder`, `excess_solder` |
| 7 | `insufficient_solder` | IPC-A-610H §5 | SolDef_AI, kydra, Hue, v2-s89jo | `poor_solder` (α-idx 3), `Insufficient_solder`, `INSUFFICIENT SOLDER`, `Insufficient Solder`, `Missing_solder` |
| 8 | `solder_spike` | IPC-A-610H §5 | SolDef_AI | `spike` (α-idx 4) |
| 9 | `cold_solder_joint` | IPC-A-610H §5 | kydra, v2-s89jo | `Cold Solder`, `Cold_solder` |
| 10 | `scratch` | IPC-A-600 §3 | Bare PCB defects | `scratch` |
| 11 | `pinhole` | IPC-A-600 §3 | Bare PCB defects | `pinhole` |

#### 3.4.1.5 Scope and Limitations

Several IPC defect categories are explicitly excluded from the CIRCA dataset. Component-level IPC-A-610 defects — including missing component (DNP error), misalignment, tombstoning, lifted lead, solder ball, and component damage — are excluded because no suitable public datasets with sufficient image counts and annotation quality are currently available. Additional IPC-A-600 structural defects such as hole breakout, conductor scratch, conductor foreign object, and base material foreign object are likewise excluded on the same grounds: public datasets capturing these categories at the scale and quality needed for reliable deep learning training have not been identified during the dataset survey. Additionally, `solder_bridge` was excluded despite being an IPC-A-610H §5 defect class, as no board-level annotated public dataset meeting the project's quality criteria was identified across all nine source datasets surveyed; this gap is documented as a future-work direction in Chapter 5. The exclusion of all these categories is grounded in observations made during the dataset survey: manufacturers protect rework imagery as intellectual property; the visual signatures of many excluded classes vary dramatically with component package and board design, requiring annotation effort beyond the present project's scope; and the relevant image collections that do exist are owned by AOI vendors and sold under commercial licences. All excluded categories are documented as a future-work recommendation in Chapter 5, with a proposed custom data-collection campaign in collaboration with a partner repair facility.

### 3.4.2 Data Pre-processing

#### 3.4.2.1 CLAHE on the L-channel of LAB

Contrast Limited Adaptive Histogram Equalisation was applied to the L-channel of the LAB colour space using a clip limit of 2.0 and an 8×8 tile grid (Wanto et al., 2023). Working in LAB rather than RGB preserves the chromatic information of the solder-mask and copper traces while equalising only the luminance channel, avoiding the colour casts that direct RGB equalisation introduces. The CLAHE step compensates for the desklamp glare and uneven illumination characteristic of repair-shop imaging, restoring surface texture visibility without amplifying noise.

#### 3.4.2.2 Gamma Correction

Following CLAHE, a fixed gamma correction of γ = 1.2 was applied across the entire image to lift mid-tone shadows without washing out highlight details. The gamma value was selected empirically by comparing histograms of preprocessed images against a target distribution of well-illuminated reference frames, following the adaptive-gamma framework of Alhamzawi et al. (2025) but using a single, hardware-friendly constant rather than a learned per-frame value, to keep total preprocessing latency below the 5 ms budget on the deployment CPU.

#### 3.4.2.3 Laplacian Variance frame quality gate

At inference time, an additional Laplacian Variance check is applied to each incoming frame as a parameter-free quality gate (Lv et al., 2024). Frames whose Laplacian variance falls below a calibrated threshold are dropped before entering the inference pipeline, preventing the YOLOv12 model from wasting CPU cycles on motion-blurred frames that cannot yield confident detections. This step is bypassed during training because the training corpus has already been screened.

#### 3.4.2.4 Polygon-to-bounding-box conversion for SolDef_AI

The SolDef_AI dataset ships with polygon (segmentation) annotations, which are not directly compatible with the YOLOv12 detection head. Conversion was performed via the Roboflow YOLOv8 (object detection) export option, which automatically takes the axis-aligned bounding rectangle of every polygon and writes it in normalised YOLO format. This approach avoids the introduction of a custom conversion script and ensures that the bounding boxes are mathematically identical to those that any future user of the same export would obtain.

#### 3.4.2.5 Class remap to the 12-class taxonomy

For SolDef_AI, Roboflow exports the dataset with alphabetically sorted class indices, yielding the order: 0 = `exc_solder`, 1 = `good`, 2 = `no_good`, 3 = `poor_solder`, 4 = `spike`. The Python mapping applied is `{0: 6, 3: 7, 4: 8}`. Raw IDs 1 (`good`) and 2 (`no_good`) are intentionally omitted — `good` because YOLO learns non-defective regions implicitly as background and an explicit good class would create severe imbalance, and `no_good` because it is an umbrella label overlapping with `exc_solder`, `poor_solder`, and `spike` and therefore creates label conflicts. Images whose only annotation belonged to a dropped class were archived in `negatives_reserve/`. The remaining classes were remapped: `exc_solder`(0)→6, `poor_solder`(3)→7, and `spike`(4)→8. For PCB Solder Joint, the `GOOD` class (ID 0) was dropped on the same grounds and images whose only annotation belonged to a dropped class were archived in the `negatives_reserve/` directory for potential later use as background-only training samples, deferred until Phase 1 baseline evidence on false-positive rates becomes available.

#### 3.4.2.6 Stratified 70/15/15 split

After class remapping and perceptual-hash deduplication, the unified pool was split into 70% training, 15% validation, and 15% test partitions using a stratified shuffle with a fixed random seed of 42. Stratification was performed on the dominant class label per image to preserve per-class proportions across all three splits, and the test partition was frozen prior to the start of any training in order to prevent test-set contamination.

### 3.4.3 Data Analysis

#### 3.4.3.1 Class distribution audit

A class-distribution audit on the unified corpus confirmed an approximately 5:1 imbalance between the bare-board classes (IDs 0–5) and the solder classes (IDs 6–9), reflecting the relative volumes of the source datasets. The class-imbalance mitigation strategy follows a three-tier approach supported by recent literature on YOLO-family detectors (Cao et al., 2024). **Tier 1 (always active):** `cls_pw` is set via inverse-frequency weighting so that under-represented solder classes (IDs 6–9) contribute a proportionally larger share of the classification gradient. **Tier 2 (always active):** Ultralytics' built-in `mosaic=1.0` and `copy_paste=0.3` augmentation synthesises new spatial arrangements from existing images, providing diversity without duplicating raw files. **Tier 3 (conditional):** If any solder class records mAP@0.5 < 70% after Phase 2, those specific classes are oversampled by a maximum of 3× in the training split only — never in validation or test — before re-running Phase 2. Bare-board classes (IDs 0–5) are *not* oversampled under any condition; they are already the majority class. The `cls_pw` value is also included as a free parameter in the HPO search space (range [0.1, 1.0]) so that its optimal magnitude can be determined empirically in Phase 3 rather than fixed a priori. The legacy `image_weights` flag was deprecated in Ultralytics YOLOv8 and is absent from YOLOv12; it is therefore not used.

#### 3.4.3.2 Duplicate and leakage detection

A perceptual-hash deduplication pipeline was applied to every image in the unified corpus, computing a difference-hash (dHash) signature per image and removing pairs with a Hamming distance ≤ 6. Hashes were also cross-checked between the v1 (bare-board) and v2 (solder) sources, although near-zero overlap was expected given the different domains. This step replicates the protocol used to sanitise the v1 corpus, in which 8,566 duplicate or near-duplicate images were purged.

***

## 3.5 System Design

### 3.5.1 CIRCA System Architecture

The deployed CIRCA system is organised as a six-stage pipeline that transforms a webcam frame into an annotated technician display. Figure 3.2 provides the runtime data-flow diagram. A USB webcam captures raw frames of the PCB under inspection. The Laplacian Variance gate decides whether the frame is sharp enough to process; if not, the frame is dropped. Surviving frames are passed through the CLAHE → Gamma preprocessing step described in §3.4.2, then submitted to the OpenVINO runtime, which dispatches inference to the CPU or integrated GPU according to a startup-time selection. The selected YOLOv12 INT8 IR model produces per-defect bounding boxes and confidence scores, and a confidence-aware post-processing block decides between the standard overlay and the "Manual Inspection Required" warning banner on the technician display.

**[Insert Figure 3.2: CIRCA System Architecture (Runtime Data Flow). Source: `CIRCA_DIAGRAMS.md`.]**

### 3.5.2 Inference Pipeline

End-to-end inference proceeds as follows. The OpenVINO Runtime is initialised with the selected variant's INT8 IR model at application start, after which a single inference call accepts a 640×640 BGR image and returns a tensor of detections. Each detection is decoded into a class label, a normalised bounding box, and a confidence value. Non-maximum suppression is performed by Ultralytics' OpenVINO wrapper at a configurable IoU threshold of 0.45, after which the surviving detections are passed to the overlay UI.

### 3.5.3 Confidence Threshold and "Manual Inspection Required" Logic

To mitigate automation bias (Goddard et al., 2011; Kupfer et al., 2023), CIRCA uses two layers of thresholding. At the per-class level, every class has a calibrated display threshold (the confidence above which boxes are drawn solid) and a calibrated warning threshold (the lower confidence above which boxes are drawn faded with a "low confidence" tag). At the screen level, a global "Manual Inspection Required" banner is triggered when (i) the mean confidence across all visible boxes drops below 0.50, (ii) the Laplacian variance of the current frame falls below the blur threshold, or (iii) no boxes are detected for ≥ 1 s while a board is clearly in frame. The thresholds are calibrated empirically on the validation set in Phase 7 and stored in a `circa_thresholds.yaml` file loaded at runtime.

### 3.5.4 Interface Design

The technician-facing interface is intentionally minimal: the live webcam feed occupies the largest area of the window, with bounding-box overlays drawn directly on the feed in IPC-class-specific colours; per-box confidence scores are rendered above each box; a small status bar at the foot of the window displays the current frame's Laplacian variance and the rolling-average FPS; and the "Manual Inspection Required" banner appears as an unmissable horizontal stripe across the top of the feed when any of the three trigger conditions is satisfied. No buttons are required for the inspection workflow itself — only a source-selection dropdown for the camera and a static-image-load shortcut.

***

## 3.6 System Development

### 3.6.1 Training Engine

A unified Python training engine (`train_engine.py`) was developed to drive every phase of the experimental programme. The engine accepts command-line arguments (`--mode train|tune`, `--variant n|s|m`, `--id`, `--desc`, `--epochs`, `--iterations`, `--preproc`, `--cfg`) and dispatches the appropriate Ultralytics call. When `--preproc` is supplied, the engine first generates a CLAHE+Gamma preprocessed mirror of the dataset under `unified_pcb_v2_preproc` (sibling of `unified_pcb_v2`, with standard YOLO `{train,val,test}/images/` and `{train,val,test}/labels/` subdirectories) if it does not already exist, then trains on that mirror; this guarantees that train-time and inference-time pixel statistics match. Stability patches are applied on every non-HPO training call: `lr0=0.001`, `warmup_epochs=5.0`, `nbs=64`, `batch=12`, `imgsz=640`, `seed=42`, `close_mosaic=10`, `cos_lr=True`, `amp=True`, and `optimizer=AdamW`. When the `--cfg` flag is supplied, the engine defers to the hyperparameter file produced by the genetic tuner and does not override `lr0` or `warmup_epochs`. Run folders are named under the convention `CIRCA_V12{N|S|M}_{ID}_{TRAIN|TUNE}_{DESC}`, ensuring that experiment lineage is traceable directly from the directory name. Every run is logged to a Weights & Biases project (`circa-yolov12`) with tags for the variant, the mode, the run ID, and the preprocessing flag, and the final checkpoint is uploaded with `WANDB_LOG_MODEL=end`.

### 3.6.2 Hyperparameter Optimisation Algorithm

#### 3.6.2.1 Search space

Hyperparameter optimisation uses Ultralytics' built-in genetic tuner (`model.tune`), which mutates the top-performing trials across iterations to converge on a high-fitness configuration. The search space comprises 17 parameters: `lr0`, `lrf`, `momentum`, `weight_decay`, `warmup_epochs`, `box`, `cls`, `cls_pw`, `dfl`, `hsv_v`, `degrees`, `translate`, `scale`, `fliplr`, `mosaic`, `mixup`, and `copy_paste`. The ranges were drawn from Ultralytics' own defaults but constrained where domain knowledge dictates — for example, `hsv_h` and `hsv_s` are excluded entirely because PCB hue and saturation carry diagnostic signal (copper colour, solder mask) that augmentation must not perturb; `shear`, `perspective`, and `flipud` are excluded because they are physically implausible for top-down PCB capture; and `degrees` is held below 10° because boards are roughly aligned in the frame. The full table appears in `PCB_DEFECT_HYPERPARAMETER_TUNING.md` §2.

#### 3.6.2.2 Stopping criteria and trial budget

The HPO budget is 50 iterations × 30 epochs per trial, run with `optimizer=AdamW`, `imgsz=640`, `batch=12`, and `seed=42`. The iteration count of 50 was chosen as a practical middle-ground, balancing Ultralytics' guidance for genetic search convergence with the constraints of the Kaggle GPU session limits. The 30-epoch trial length follows the lecturer's guidance that hyperparameter search should be conducted with low epoch counts to permit broad sampling, after which the optimal configuration is committed to a long final training run. Outputs are written to `runs/detect/CIRCA_V12S_003_TUNE_HPO/`, including `best_hyperparameters.yaml` (consumed by Phase 4 via `--cfg`), `tune_results.csv`, and the diagnostic plots `tune_fitness.png` and `tune_scatter_plots.png`.

### 3.6.3 Model Training Procedure

Each variant is trained for 200 epochs on the preprocessed 12-class corpus using the HPO-tuned hyperparameter file. AMP is enabled, EMA tracking is used, and `close_mosaic=10` disables mosaic augmentation in the final ten epochs to permit precise bounding-box refinement. Variant-specific batch sizes were selected to fit the 6 GB VRAM ceiling of the RTX 3060 training GPU at `imgsz=640`: 16 for YOLOv12-N, 12 for YOLOv12-S, and 6 for YOLOv12-M.

### 3.6.4 OpenVINO Export and INT8 Quantisation

After training, each variant is exported to three OpenVINO IR formats: FP32 (baseline accuracy reference), FP16 (fallback), and INT8 (primary deployment target). INT8 quantisation is performed using NNCF post-training quantisation through Ultralytics' `model.export(format="openvino", int8=True, data="data.yaml")` interface. Calibration is conducted on approximately 300 representative images sampled from the training split, covering all 12 IPC classes and the full range of lighting conditions. A post-quantisation validation step (Phase 5) re-runs `model.val` on the validation set in each precision and applies the fallback rule: if INT8 mAP@0.5 falls below the FP32 baseline by more than 1% or below 90% absolute, the system falls back to FP16; otherwise INT8 is retained. Decisions are documented in `quantization_report.md`.

### 3.6.5 Confidence Threshold Calibration Procedure

Confidence thresholds are calibrated only on the validation split, never on the test split, in order to preserve the test set as a one-shot reporting benchmark. The procedure sweeps the global confidence parameter `conf` from 0.10 to 0.90 in steps of 0.05, computes per-class precision and recall at each step, and selects the per-class display threshold as the minimum confidence at which precision ≥ 0.90 and the per-class warning threshold as the minimum confidence at which recall ≥ 0.95. The global "Manual Inspection Required" trigger is calibrated separately using a representative webcam log of the deployment workflow, with the trigger threshold set such that the banner fires on no more than 5% of well-illuminated, in-focus frames.

***

## 3.7 Experimental Design

### 3.7.1 Phase 1 — Vanilla Baseline

Phase 1 is an ablation control whose sole purpose is to quantify the lift attributable to the CIRCA preprocessing pipeline. YOLOv12-S is trained for 50 epochs on the **raw** 12-class corpus with default Ultralytics hyperparameters (subject to the stability patches in §3.6.1) and no CLAHE or Gamma preprocessing. The 50-epoch budget is a deliberate design choice: past studies have used short training runs (30–100 epochs) specifically for ablation controls to isolate the effect of a single variable before committing to full training (Yang and Yu, 2024; Kaewdook et al., 2024). Phase 2 extends this to 100 epochs on the preprocessed corpus; the epoch counts differ to reflect the incremental investment appropriate to each phase, with the final 200-epoch run in Phase 4 representing the full training budget.

### 3.7.2 Phase 2 — CIRCA-Aligned Baseline

Phase 2 introduces the CIRCA preprocessing pipeline. YOLOv12-S is trained for 100 epochs on the preprocessed 12-class corpus with `cls_pw=1.0` (full inverse-frequency loss weighting) to compensate for the bare-board / solder imbalance. The 100-epoch budget is chosen to give Phase 3 a realistic, partially-converged mAP target to beat: it is long enough to distinguish preprocessing benefit from noise, yet short enough to avoid wasting compute if the HPO phase (Phase 3) subsequently identifies a substantially different hyperparameter configuration. This staged epoch escalation from 50→100→200 epochs mirrors the approach used by Bhattacharya and Cloutier (2022) and Liao et al. (2021), who both conducted abbreviated baseline runs before committing to full training on their best configuration.

### 3.7.3 Phase 3 — Hyperparameter Optimisation

Phase 3 runs the 50-iteration × 30-epoch genetic tuner described in §3.6.2 on YOLOv12-S over the preprocessed 12-class corpus. The output is the `best_hyperparameters.yaml` file fed into Phase 4.

### 3.7.4 Phase 4 — Three-Variant Final Training

Phase 4 produces the final candidate models for the comparative study. YOLOv12-N, YOLOv12-S, and YOLOv12-M are each trained for 200 epochs with the HPO-tuned configuration, yielding three `best.pt` checkpoints and (via Phase 5) three sets of OpenVINO IR exports. The 200-epoch budget is consistent with the training length used by Yang and Yu (2024) for YOLOv8 on a PCB benchmark dataset, and with the guidance in Ultralytics documentation for custom datasets of under 15,000 images; longer runs risk overfitting while shorter runs risk under-convergence on the minority solder classes. Early stopping is enabled with `patience=50` as a safeguard, such that training terminates automatically if validation mAP does not improve for 50 consecutive epochs.

### 3.7.5 Phase 5 — Quantisation Validation

Each variant is exported to FP32, FP16, and INT8 IR and validated on the validation split. The fallback rule of §3.6.4 is applied per variant and the decision recorded. The 1% mAP degradation tolerance is consistent with the negligible accuracy loss reported by Ahn et al. (2023) for INT8 dynamic quantization on Intel CPUs, and with real-world YOLO OpenVINO INT8 deployments that demonstrate minimal accuracy drops alongside the 3× latency improvement (Ahn et al., 2023; OpenVINO documentation, 2024).

### 3.7.6 Phase 6 — Hardware Benchmarking and Variant Selection

Each surviving variant + precision combination is benchmarked on the deployment target (Intel Core i5 8th-generation equivalent + integrated GPU). Four metrics are measured: preprocessing latency per frame, inference latency per frame on CPU and on iGPU, end-to-end live FPS over a 60-second webcam loop, and static image inference time on a single high-resolution image. The Variant Selection Matrix is filled, and the configuration with the highest mAP@0.5 that simultaneously passes all four acceptance criteria is selected as the production variant.

### 3.7.7 Phase 7 — Final Test Evaluation and Threshold Calibration

The selected variant is evaluated **once** on the frozen test split. Per-class precision, recall, F1, mAP@0.5, and mAP@0.5:0.95 are reported, alongside the confusion matrix and the PR / F1 curves. The confidence threshold sweep of §3.6.5 is then run on the validation split, and the resulting `circa_thresholds.yaml` file is generated and locked in.

### 3.7.8 Acceptance Criteria

A variant is declared "final" only when it simultaneously satisfies all four acceptance criteria stated in Chapter 1 §1.5: mAP@0.5 > 90% on the test set, preprocessing latency ≤ 5 ms per frame, live inference rate ≥ 15 FPS, and static image inference ≤ 10 s on the Intel Core i5 8th-generation equivalent target.

### 3.7.9 Evaluation Metrics

The reported metrics are precision, recall, F1-score, mAP@0.5, mAP@0.5:0.95, per-class mAP@0.5, per-class precision and recall at the calibrated thresholds, preprocessing latency in milliseconds, inference latency in milliseconds (CPU and iGPU), end-to-end live FPS, and static-image total time in seconds. A bounding box is counted as a True Positive if and only if the Intersection over Union (IoU) between the predicted box and the nearest ground-truth box of the same class meets or exceeds the stated IoU threshold (0.50 for mAP@0.5 and the range 0.50–0.95 in steps of 0.05 for mAP@0.5:0.95), following the COCO evaluation protocol adopted universally in recent YOLO PCB literature (Yang and Yu, 2024; Tian et al., 2025; Liao et al., 2021). All metrics are reported with the split, variant, and precision named, consistent with the guardrails enforced by the training engine.

***

## 3.8 Hardware and Software Specification

### 3.8.1 Training Environment

Model training was performed on a workstation equipped with an NVIDIA RTX 3060 GPU (6 GB VRAM), 16 GB of system RAM, and an 8th-generation Intel Core i7 CPU running Windows 11. The deep-learning stack consisted of Python 3.11, PyTorch 2.x with CUDA 12.x, Ultralytics ≥ 8.3 (the first release line carrying official YOLOv12 support), OpenCV 4.x, and Weights & Biases for experiment tracking. AMP (automatic mixed precision) was enabled throughout to maximise throughput at the 6 GB VRAM ceiling.

### 3.8.2 Deployment Target

The deployment target is a notional Intel Core i5 8th-generation equivalent processor with an integrated GPU, running Windows 10 or Windows 11. AVX2 and VNNI instruction-set support is assumed, which OpenVINO uses to accelerate INT8 inference. No discrete graphics card is required. Image capture uses a standard USB webcam (or a smartphone tethered via a virtual-camera driver) at 1280×720 resolution, with no specialised industrial imaging hardware.

### 3.8.3 Software Stack

The deployment-side software stack consists of Python 3.11, the Ultralytics inference layer, the OpenVINO Runtime with the Neural Network Compression Framework (NNCF) for quantisation support, OpenCV for the preprocessing pipeline and webcam handling, and a thin desktop UI built on top of OpenCV's window primitives. The same software stack is used during hardware benchmarking and during the deployed prototype's normal operation, ensuring that benchmark numbers transfer directly to production.

***

## 3.9 Research Plan

The project timeline is organised into the eight phases of §3.7 and was sequenced to allow upstream phases to deliver inputs to downstream phases without back-tracking. Table 3.3 summarises the plan and its compute estimate on the RTX 3060 training GPU.

**Table 3.3: CIRCA Project Timeline and Compute Estimate**

| Phase | Description | Estimated Duration | Cumulative |
|:---|:---|---:|---:|
| 0 | Environment + dataset setup (4 sources, sanitise, split) | 1 week | 1 week |
| 1 | Vanilla baseline (50 ep, YOLOv12-S) | ~7 h | 1 week |
| 2 | CIRCA-aligned baseline (100 ep, preproc) | ~14 h | 1.5 weeks |
| 3 | Genetic HPO (50 it × 30 ep, preproc) | ~4 days | 2.5 weeks |
| 4 | Three-variant final training (200 ep × 3) | ~3.5 days | 3 weeks |
| 5 | OpenVINO quantisation validation | 1 day | 3 weeks |
| 6 | Hardware benchmarking | 1–2 days | 3.5 weeks |
| 7 | Test evaluation + threshold calibration | 1–2 days | 4 weeks |
| | **Total compute on RTX 3060** | **~8.5 days** | |

***

## 3.10 Summary

This chapter has described the methodology adopted for the CIRCA project. The work is organised around an eight-phase research framework (§3.2) that maps each phase onto one or more research objectives. The empirical study (§3.4) constructs a unified 12-class IPC-aligned dataset from four public sources, sanitises it via perceptual-hash deduplication, and splits it 70/15/15 with stratification. The CIRCA preprocessing pipeline (§3.4.2 and §3.6.1) applies CLAHE on the LAB L-channel, gamma correction at γ = 1.2, and an inference-time Laplacian Variance frame-quality gate. The training programme (§3.6 and §3.7) progresses from a vanilla ablation control, through a CIRCA-aligned baseline and genetic hyperparameter optimisation, to a three-variant comparative final training and OpenVINO INT8 quantisation, and finally to hardware benchmarking and confidence threshold calibration on the deployment target. The corpus is stored in `unified_pcb_v2` throughout. Acceptance is governed by the four quantitative criteria stated in Chapter 1 §1.5. The results of executing this methodology are reported in Chapter 4.

***

***

# CHAPTER 4: RESULTS AND FINDINGS

This chapter reports the quantitative and qualitative findings produced by executing the experimental programme described in Chapter 3. Results are organised by research objective: §4.2 reports the dataset and defect-taxonomy outcomes that satisfy RO1, §4.3–§4.7 report the training, optimisation, quantisation, and hardware-benchmarking outcomes that satisfy RO2, and §4.8–§4.9 report the test-evaluation and threshold-calibration outcomes that satisfy RO3. The chapter concludes with a benchmarking comparison against published PCB defect detectors (§4.10) and a summary of findings (§4.11).

> **Note on draft status:** The CIRCA experimental programme is scheduled to execute over approximately 8.5 days of GPU compute (Chapter 3 §3.9). The narrative below is structured as a chapter skeleton with section-by-section guidance and explicit `[Insert Table 4.X]` and `[Insert Figure 4.X]` placeholders that will be filled in once each phase completes. Each subsection ends with a discussion stub indicating the comparison or implication that the surrounding text will articulate, in line with the CSP650 Topic 4 guideline that every figure and table requires explanatory text and that round numbers should be used when emphasising a trend.

***

## 4.1 Introduction

This chapter is organised in an interleaved (results-with-discussion) pattern, in which every section presents its quantitative outcomes together with the discussion that interprets those outcomes against the relevant literature and against the acceptance criteria. The interleaved pattern was preferred over the all-results-then-discussion pattern because the CIRCA programme generates results across multiple phases (preprocessing ablation, HPO, multi-variant comparison, quantisation, hardware benchmarking, threshold calibration) that benefit from immediate interpretation rather than deferred synthesis. This structure mirrors the approach taken in related applied machine learning studies such as Klco et al. (2023), who interleave per-phase results with discussion, and Kaewdook et al. (2024), who present hardware benchmarking discussion alongside quantitative tables rather than deferring to a separate section.

***

## 4.2 Dataset and Defect Taxonomy Results

This section answers Research Objective 1 (RO1) — the identification, categorisation, and documentation of IPC-A-600 bare-board and IPC-A-610 assembly-stage PCB defect types for automated detection. The defect taxonomy is the artefact through which RO1 is satisfied, and its quantitative validation comes from the per-class image counts and split statistics of the unified `unified_pcb_v2` corpus.

### 4.2.1 Final Class Distribution

**[Insert Table 4.1: Final Class Distribution — 11 IPC Classes (6 bare-board IPC-A-600 + 5 solder IPC-A-610), with image and instance counts per class, generated by the dataset statistics script after Phase 0 completes.]**

The 12-class taxonomy combines six bare-board defect types from the IPC-A-600 standard with five solder-defect types from IPC-A-610, satisfying the IPC-A-610 assembly-stage framing of the project while also covering the IPC-A-600 bare-board defect territory represented in the available public datasets.

### 4.2.2 Dataset Statistics across Splits

**[Insert Table 4.2: Train / Validation / Test counts for the unified `unified_pcb_v2` corpus, with stratified per-class proportions, produced by the dataset statistics script at the end of Phase 0.]**

The corpus contains images drawn from four public sources after rigorous global deduplication, split 70 / 15 / 15 with stratification. Final image counts will be confirmed after Phase 0 deduplication completes.

### 4.2.3 Sample Defect Images per IPC Class

**[Insert Figure 4.1: Curated 11-cell grid showing one representative defect image per IPC class, drawn from the training split. To be assembled from `runs/dataset_review/` after Phase 0.]**

### 4.2.4 Discussion

The discussion will note (i) the representativeness of the assembled corpus relative to repair-context conditions described in Chapter 2, (ii) the class-imbalance ratio between the bare-board and solder partitions and the implications for the mitigation strategy laid out in Chapter 3 §3.4.3.1, and (iii) the explicit scope limitations on excluded defect categories, with reference to the future-work directions proposed in Chapter 5.

***

## 4.3 Preprocessing Pipeline Evaluation

### 4.3.1 Vanilla vs CIRCA-Preprocessed Baseline Comparison

**[Insert Table 4.3: Phase 1 vs Phase 2 mAP@0.5 and mAP@0.5:0.95 on YOLOv12-S, both trained at 50 / 100 epochs respectively. Source: `runs/detect/CIRCA_V12S_001_TRAIN_Baseline_Vanilla/results.csv` and `runs/detect/CIRCA_V12S_002_TRAIN_Baseline_CIRCA/results.csv`.]**

### 4.3.2 Preprocessing Latency Measurement

**[Insert Table 4.4: Per-stage and total preprocessing latency on the Intel Core i5 8th-gen target, measured over 1,000 representative frames. Source: `benchmark.py` output.]**

### 4.3.3 Discussion

This subsection will quantify the absolute and relative mAP lift attributable to the CLAHE + Gamma pipeline, examine which classes benefited most (the expectation, based on Lv et al. 2024 and Wanto et al. 2023, is that fine defects such as `open_circuit`, `short`, and `spur` will gain the most because they depend on edge contrast), and verify that the preprocessing latency target of ≤ 5 ms per frame is met.

***

## 4.4 Hyperparameter Optimisation Results

### 4.4.1 HPO Search Trajectory

**[Insert Figure 4.2: HPO fitness curve over 150 iterations on YOLOv12-S, showing the best-so-far mAP trajectory of the genetic tuner. Source: `runs/detect/CIRCA_V12S_003_TUNE_HPO/tune_fitness.png`.]**

### 4.4.2 Top-10 Trials

**[Insert Table 4.5: Top-10 HPO trials sorted by fitness, with the per-trial values for `lr0`, `lrf`, `momentum`, `weight_decay`, `warmup_epochs`, `box`, `cls`, `cls_pw`, `dfl`, augmentation parameters, and resulting mAP@0.5. Source: `runs/detect/CIRCA_V12S_003_TUNE_HPO/tune_results.csv`.]**

### 4.4.3 Parameter Importance

**[Insert Figure 4.3: HPO parameter scatter / parallel-coordinate plot, indicating which parameters most influenced fitness. Source: `runs/detect/CIRCA_V12S_003_TUNE_HPO/tune_scatter_plots.png`.]**

### 4.4.4 Selected Hyperparameter Configuration

**[Insert Table 4.6: Selected hyperparameter configuration committed to Phase 4, drawn from `runs/detect/CIRCA_V12S_003_TUNE_HPO/best_hyperparameters.yaml`.]**

### 4.4.5 Discussion

The discussion will identify which parameters dominated the search (typical expectation: `lr0`, `cls`, and `mosaic` contribute the strongest effect on fitness for YOLOv12 on small-object datasets), report whether the tuned values differ substantially from Ultralytics defaults, and comment on the relevance of the chosen `cls_pw` value to the bare-board / solder imbalance.

***

## 4.5 Three-Variant Comparative Training Results

This section answers Research Objective 2 (RO2) — the design and comparative evaluation of YOLOv12-N, YOLOv12-S, and YOLOv12-M.

### 4.5.1 Training Curves per Variant

**[Insert Figure 4.4: Training and validation loss + mAP curves for each of the three variants, trained for 200 epochs with the HPO config. Source: `runs/detect/CIRCA_V12{N,S,M}_*/results.png`.]**

### 4.5.2 Validation Metrics per Variant

**[Insert Table 4.7: Validation metrics (mAP@0.5, mAP@0.5:0.95, precision, recall, F1) per variant. Source: `runs/detect/CIRCA_V12{N,S,M}_*/results.csv`.]**

### 4.5.3 Per-Class Performance Breakdown

**[Insert Table 4.8: Per-class mAP@0.5 for each of the three variants, with bare-board and solder partitions visually separated.]**

### 4.5.4 Discussion

This subsection will report the mAP scaling behaviour from N→S→M, identify the inflection point at which marginal mAP gain no longer justifies latency cost, and check whether the solder classes (6–9) have closed the gap relative to the bare-board classes (0–5) after the imbalance mitigations applied in Phase 2.

***

## 4.6 OpenVINO Quantisation Results

### 4.6.1 FP32 vs FP16 vs INT8 mAP

**[Insert Table 4.9: FP32 vs FP16 vs INT8 mAP@0.5 on the validation split, for each of the three variants. Source: `quantization_report.md`.]**

### 4.6.2 INT8 → FP16 Fallback Decision

**[Insert Table 4.10: Per-variant fallback decision summary, applying the rule defined in Chapter 3 §3.6.4. Source: `quantization_report.md`.]**

### 4.6.3 Discussion

The discussion will report the per-variant INT8 mAP penalty relative to FP32, identify which (if any) variant triggered the FP16 fallback, and connect the observed quantisation behaviour to the 3.3× INT8 speedup characterised by Ahn et al. (2023) on Intel CPUs.

***

## 4.7 Hardware Benchmarking Results

This section answers part of Research Objective 3 (RO3) — variant selection against the four acceptance criteria.

### 4.7.1 Preprocessing Latency on Target CPU

**[Insert Table 4.11: Per-stage preprocessing latency (CLAHE, Gamma, Laplacian gate, total) measured on the Intel Core i5 8th-gen target. Source: `benchmark_report.md`.]**

### 4.7.2 Inference Latency: CPU vs iGPU

**[Insert Table 4.12: Per-variant per-precision inference latency on the deployment target, separately for CPU and integrated GPU plugins. Source: `benchmark_report.md`.]**

### 4.7.3 End-to-End Live FPS

**[Insert Figure 4.5: End-to-end live FPS moving-average plot over a 60-second webcam loop, for the selected variant. Source: `benchmark.py` output.]**

### 4.7.4 Static Image Inference Time

**[Insert Table 4.13: Per-variant static-image total time (preproc + inference + draw overlays) for a single high-resolution image. Source: `benchmark_report.md`.]**

### 4.7.5 Variant Selection Matrix and Acceptance-Criteria Verdict

**[Insert Table 4.14: Variant Selection Matrix listing every variant + precision combination with mAP, preprocessing latency, inference latency, live FPS, static seconds, model size, and a Pass / Fail verdict against the four acceptance criteria. Source: `benchmark_report.md`.]**

### 4.7.6 Discussion

The discussion will explain which configurations passed all four acceptance criteria simultaneously, how CPU-only inference compares with iGPU inference on the same model, and the basis for the chosen production variant.

***

## 4.8 Final Test-Set Evaluation

This section answers the remaining part of Research Objective 3 (RO3) — the evaluation of the final CIRCA prototype on the frozen test split.

### 4.8.1 Overall Metrics on Test Set

**[Insert Table 4.15: Overall mAP@0.5, mAP@0.5:0.95, precision, recall, F1 on the test split for the selected variant + precision. Source: `test_evaluation.md`.]**

### 4.8.2 Per-Class Precision, Recall, F1

**[Insert Table 4.16: Per-class precision, recall, and F1 on the test split. Source: `test_evaluation.md`.]**

### 4.8.3 Confusion Matrix

**[Insert Figure 4.6: 11×11 confusion matrix on the test split. Source: `runs/detect/<final>/confusion_matrix.png`.]**

### 4.8.4 PR and F1 Curves

**[Insert Figure 4.7: PR and F1 curves on the test split. Source: `runs/detect/<final>/PR_curve.png` and `F1_curve.png`.]**

### 4.8.5 Failure-Case Gallery

**[Insert Figure 4.8: Curated gallery of failure cases — small defects, glare, shadow, motion blur, off-angle. Source: hand-picked from `runs/detect/<final>/val_batch*_pred.jpg` and the repair-context test captures.]**

### 4.8.6 Discussion

The discussion will identify the classes with the lowest recall, examine the visual conditions causing failures (typical expectation: glare on convex solder fillets and motion blur on hand-held smartphone capture), and report the generalisation gap between validation and test mAP.

***

## 4.9 Confidence Threshold Calibration Results

### 4.9.1 Threshold Sweep on Validation

**[Insert Figure 4.9: Per-class precision-recall curves over the conf sweep on the validation split, with the chosen display and warning thresholds annotated. Source: `calibrate_thresholds.py` output.]**

### 4.9.2 Per-Class Display and Warning Thresholds

**[Insert Table 4.17: Per-class display and warning confidence thresholds, drawn from `circa_thresholds.yaml`.]**

### 4.9.3 Global "Manual Inspection Required" Trigger Calibration

The global trigger fires when (i) the mean confidence across visible boxes drops below 0.50, (ii) Laplacian variance falls below the calibrated blur threshold, or (iii) no boxes are detected for ≥ 1 s while a board is clearly in frame. The calibration target is for the banner to fire on no more than 5% of well-illuminated, in-focus frames during the validation webcam log; the actual fire-rate after calibration will be reported here.

### 4.9.4 Discussion

The discussion will examine whether the chosen thresholds strike the intended balance between automation-bias mitigation (Goddard et al., 2011; Kupfer et al., 2023) and false-alarm fatigue, and will report the empirical fire-rate of the global trigger under realistic webcam conditions.

***

## 4.10 Comparison with Related Work

### 4.10.1 Benchmarking against Published PCB Detectors

**[Insert Table 4.18: Comparison of the chosen CIRCA configuration against the related works summarised in Chapter 2 Table 2.1, on mAP, FPS, and hardware target. Built from `benchmark_report.md` plus the Chapter 2 literature.]**

### 4.10.2 Discussion

The discussion will position CIRCA against Hu and Wang (2020), Liao et al. (2021), Yang and Yu (2024), and Tian et al. (2025), making explicit the four distinguishing axes of (a) commodity Intel CPU and integrated GPU deployment without dedicated graphics hardware, (b) repair-context lighting robustness achieved through the CLAHE + Gamma preprocessing pipeline, (c) a dual-standard IPC-A-600 (bare-board) and IPC-A-610 (assembly-stage) unified 12-class taxonomy that spans both PCB quality domains, and (d) confidence-transparent UI design with automation-bias mitigation — none of which is shared by any single prior work on like-for-like terms.

***

## 4.11 Summary of Findings

This section will restate which research objectives were satisfied by which results — RO1 by §4.2, RO2 by §4.3 to §4.7, and RO3 by §4.7 to §4.9 — and will identify the headline contribution of the CIRCA project (commodity-CPU real-time IPC-A-600 and IPC-A-610-aligned PCB defect detection with confidence-transparent UI). Open issues, unresolved limitations, and the specific failure modes uncovered in §4.8.5 will be flagged for treatment in Chapter 5 (Discussion, Conclusion, and Recommendations).

***

***

### Complete Reference List (All Sources Used in Chapter 1, 2, 3, and 4)

| # | Citation | Source Type | Access |
|---|---|---|---|
| 1 | Adibhatla, V. A., et al. (2020). Defect detection in printed circuit boards using you-only-look-once convolutional neural networks. *Electronics, 9*(9), 1547. | Peer-reviewed journal | MDPI Open Access |
| 2 | Aggarwal, V., et al. (2022). Review of machine learning techniques for PCB defect detection. *Sensors, 22*(19), 7375. | Peer-reviewed journal | MDPI Open Access |
| 3 | Ahn, H., et al. (2023). Performance characterization of using quantization for DNN inference on edge devices. *arXiv:2303.05016*. | Preprint | arXiv / NSF PAR Free PDF |
| 4 | Alhamzawi, G. A., Alfoudi, A. S., and Alsaeedi, A. H. (2025). Fusion deep learning with adaptive gamma correction (DLAGC). *JISEM, 10*(36s). | Peer-reviewed journal | JISEM Open Access |
| 5 | Anh Nguyen, T., et al. (2024). PCB defect detection using hybrid CNN-Transformer with Wise-IoU. *Applied Sciences.* | Peer-reviewed journal | MDPI Open Access |
| 6 | Bhattacharya, D., and Cloutier, S. G. (2022). End-to-end deep learning framework for printed circuit board manufacturing defect classification. *Scientific Reports, 12*, 12969. | Peer-reviewed journal | Nature Open Access |
| 7 | Bhanumathy, M., et al. (2021). PCB defect detection using CNN. *IOP Conference Series.* | Conference paper | IOP Open Access |
| 8 | Goti, A. B. (2025). Automated optical inspection (AOI) based on IPC standards. *International Journal of Engineering and Computer Science (IJECS), 14*(3), pp.26928–26947. https://doi.org/10.18535/ijecs/v14i03.5052 | Peer-reviewed journal | IJECS Open Access |
| 9 | Fontana, M., Calabrese, M., et al. (2024). SolDef-AI: An open source PCB dataset for Mask R-CNN defect detection in through-hole pin soldering. *Journal of Manufacturing and Materials Processing, 8*(4), 145. https://doi.org/10.3390/jmmp8040145 | Peer-reviewed journal (CC BY 4.0) | MDPI Open Access |
| 10 | Ghelani, D. (2024). AI-driven quality control in PCB manufacturing. *International Journal of Engineering.* | Peer-reviewed journal | Open Access |
| 11 | Goddard, K., Roudsari, A., and Wyatt, J. C. (2011). Automation bias: a hidden issue for clinical decision support system use. *Stud Health Technol Inform, 164*, 17–22. | Peer-reviewed book chapter | PubMed / IOS Press |
| 12 | Goti (2025). Automated optical inspection (AOI) based on IPC standards. *SSRN Preprint.* | Preprint | SSRN Free Download |
| 13 | Hendriko, V., and Hermanto, D. (2025). Performance comparison of YOLOv10, YOLOv11, and YOLOv12. *Brilliance: Research of AI, 5*(1), 440–450. | Peer-reviewed journal | ITSCIENCE Open Access |
| 14 | Hu, S., and Wang, H. (2020). PCB defect detection with GARPN+FPN+ResNet50. *IEEE Access.* | Peer-reviewed journal | IEEE Access |
| 15 | Huang, J., et al. (2023). NCE-Net for PCB defect classification. *Electronics.* | Peer-reviewed journal | MDPI Open Access |
| 16 | IPC International. (2020). *IPC-A-600K: Acceptability of Printed Boards.* Bannockburn, IL: IPC. | Industry standard | IPC (purchase) |
| 17 | IPC International. (2020). *IPC-A-610H: Acceptability of Electronic Assemblies.* Bannockburn, IL: IPC. | Industry standard | IPC (purchase) |
| 18 | Kaewdook, D., et al. (2024). Design of deep learning techniques for PCBs defect detecting system based on YOLOv10. *ETASR, 14*(6). | Peer-reviewed journal | ETASR Open Access |
| 19 | Klco, P., et al. (2023). YOLOv8 for PCB defect detection. *Applied Sciences.* | Peer-reviewed journal | MDPI Open Access |
| 20 | Kupfer, A., et al. (2023). Check the box! Automation bias in AI-based decision support. *Frontiers in Psychology, 14*, 1118723. | Peer-reviewed journal | Frontiers Open Access |
| 21 | Law, C. Y., et al. (2024). Ensemble learning for PCB defect detection. *IEEE Access.* | Peer-reviewed journal | IEEE Access |
| 22 | Liao, X., et al. (2021). Lightweight YOLOv4-MobileNetV3 for PCB defect detection. *Sensors, 21*(23), 7873. | Peer-reviewed journal | MDPI Open Access |
| 23 | Lv, S., et al. (2024). A dataset for deep learning based detection of printed circuit board surface defect. *Scientific Data, 11*, 811. | Peer-reviewed journal | Nature Open Access (CC BY 4.0) |
| 24 | Sharma, A., et al. (2024). YOLOv5 vs OpenCV PCB defect study. *Journal of Engineering.* | Peer-reviewed journal | Open Access |
| 25 | Tian, Y., Ye, Q., and Doermann, D. (2025). YOLOv12: attention-centric real-time object detectors. *NeurIPS 2025*. arXiv:2502.12524. | Conference paper + preprint | arXiv Free PDF / NeurIPS |
| 26 | Wang, Z., et al. (2023). ATT-YOLO: self-attention enhanced YOLOv5 for PCB defect detection. *Journal of Physics: Conference Series.* | Conference paper | IOP Open Access |
| 27 | Wanto, A., et al. (2023). Optimization accuracy of CNN model by utilizing CLAHE parameters in image classification. *Proceedings of IConNECT 2023, IEEE.* | Conference paper | UPI YPTK Repository / IEEE |
| 28 | PCB Defect Detection. (2025a). *PCB Solder Joint Dataset* [Dataset]. Roboflow Universe. https://universe.roboflow.com/work-6qkmv/pcb-solder-joint (CC BY 4.0). | Open dataset | Roboflow Universe |
| 29 | Yang, X., and Yu, H. (2024). PCB defect detection based on improved YOLOv8. *IJANMC.* | Peer-reviewed journal | Open Access |
| 30 | Yi, F., and Mohamed, A. S. A. (2024). YOLOv8-DEE: a high-precision model for PCB defect detection. *PeerJ Computer Science, 10*, e2548. | Peer-reviewed journal | PeerJ / PMC Open Access |
|
