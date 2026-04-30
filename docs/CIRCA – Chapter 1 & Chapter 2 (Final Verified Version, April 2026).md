# CIRCA: Circuit Inspection and Recognition Using Convolutional Architectures
### Chapter 1 & Chapter 2 — Final Verified Version
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

This research addresses this identified gap by proposing CIRCA (Circuit Inspection and Recognition using Convolutional Architectures), an AI-driven visual inspection system that leverages YOLOv12 variants to automatically detect and localise surface-level PCB defects such as solder bridges, missing or misaligned components, damaged traces, and burnt areas from standard camera images. By deploying a YOLOv12 model quantized to INT8 Intermediate Representation (IR) format via the Intel OpenVINO inference framework, CIRCA achieves sub-10-second diagnostic turnarounds on standard Intel CPUs and integrated GPUs, eliminating the need for expensive AOI hardware (Yi and Mohamed, 2024). An ultra-lightweight OpenCV preprocessing pipeline combining Contrast Limited Adaptive Histogram Equalization (CLAHE), Gamma Correction, and Laplacian Variance blur detection ensures robust inference performance under the uncontrolled lighting conditions typical of real-world repair shop environments (Alhamzawi et al., 2025; Wanto et al., 2021).

***

## 1.2 Problem Statement

The electronics repair industry faces several interconnected challenges that significantly impact service quality, operational efficiency, and customer satisfaction. These challenges stem primarily from the limitations inherent in current manual inspection methodologies and the increasing complexity of modern electronic devices.

Firstly, the detection of defects in modern PCBs through manual visual inspection has become increasingly problematic due to progressive component miniaturisation and elevated circuit density (Djuberh et al., 2025). Contemporary PCBs frequently incorporate surface-mount technology (SMT) components measuring less than 1 millimetre, fine-pitch integrated circuits with pin spacing below 0.5 millimetres, and multi-layer board constructions with internal connections invisible to surface inspection (Adibhatla et al., 2020). Under these circumstances, even experienced technicians equipped with magnification tools struggle to identify subtle defects such as hairline cracks, micro-solder bridges, or incipient component failures. The industry standard for defining and categorising PCB assembly acceptability is IPC-A-610 (Acceptability of Electronic Assemblies), which specifies visual criteria for solder joint quality, component placement, and assembly cleanliness across three product reliability classes (Hassan et al., 2025). A comparative study demonstrated that AI-driven AOI systems aligned with IPC standards achieve 98 to 99% inspection accuracy and process over 5,000 components per hour, compared to 85 to 90% accuracy and 500 to 800 components per hour for manual inspection, underscoring the scale of the performance gap between automated and manual approaches (Hassan et al., 2025).

Secondly, manual inspection processes are inherently time-consuming and subject to human limitations. A comprehensive visual examination of a complex PCB can require 10 to 30 minutes or more, depending on board complexity and the number of components present. This extended inspection time directly translates to increased repair costs, longer customer wait times, and reduced throughput for repair facilities. Moreover, human inspectors are susceptible to fatigue, distraction, and subjective judgment variations, which can lead to missed defects or false positives. Research indicates that inspector accuracy declines significantly after extended periods of continuous inspection, with defect detection rates dropping by up to 30% after four hours of sustained work (Djuberh et al., 2025). Manual PCB inspection using stereomicroscopes and optical magnification aids is also associated with severe eye strain during prolonged inspection shifts, presenting a physical occupational burden that automated systems can directly eliminate.

Thirdly, the reliability and consistency of manual inspection are heavily dependent on individual technician expertise and experience levels (Sharma et al., 2024). Novice technicians may lack the knowledge to recognise certain defect types or understand their implications, while even experienced professionals may overlook defects in unfamiliar board designs or when confronting novel failure modes. This variability in inspection quality creates inconsistencies in service delivery and can erode customer confidence in repair services.

Furthermore, certain categories of PCB defects present specific challenges for visual detection. Internal solder joint failures, delamination within multi-layer boards, intermittent connection problems, and incipient component degradation may not manifest obvious visual indicators during static inspection (Hu and Wang, 2020). These defects can evade detection during manual inspection yet subsequently cause device malfunction, necessitating repeated repair attempts and generating customer dissatisfaction. In critical applications such as medical devices, automotive electronics, or industrial control systems, undetected defects can pose serious safety hazards or result in catastrophic system failures (Liao et al., 2021).

Current automated inspection technologies, while highly effective in manufacturing environments, are generally designed for high-volume production scenarios and are not optimally configured for the diverse range of devices and repair contexts encountered in service centres (Huang et al., 2023). Existing AOI systems are often expensive, require specialised training to operate, and may lack the flexibility to accommodate the variety of PCB types and device configurations typical of repair workflows (Kaewdook et al., 2024). Consequently, there exists a clear need for accessible, cost-effective, and adaptable automated inspection solutions tailored specifically to the requirements of electronics repair and service operations.

This research therefore addresses the following core problem: **How can machine learning-based automated defect detection systems be effectively designed and implemented to enhance the speed, accuracy, and reliability of PCB fault diagnosis in electronics repair environments, thereby improving service quality and operational efficiency while minimising the impact of human error and expertise variability?**

***

## 1.3 Research Questions

To comprehensively address the identified problem and guide the research investigation, this study seeks to answer the following research questions:

**RQ1:** How to identify and categorise common PCB defect types based on their visual characteristics, aligned with IPC-A-610 industry acceptability criteria, for automated detection in electronics repair contexts?

**RQ2:** How to design and develop YOLOv12-based CNN detection models incorporating INT8 quantization via Intel OpenVINO and an OpenCV preprocessing pipeline that achieves accurate and efficient PCB defect detection under real-world repair shop conditions?

**RQ3:** How to develop and evaluate a user-friendly CIRCA desktop prototype that supports zero-friction, real-time deployment in electronics repair facilities?

***

## 1.4 Research Objectives

Based on the research questions formulated above, this study establishes the following specific objectives:

**RO1:** To identify, categorise, and document the most prevalent PCB defect types encountered in electronics repair based on their characteristic visual signatures, aligned with IPC-A-610 acceptability guidelines, suitable for automated optical detection.

**RO2:** To design and develop YOLOv12-based CNN detection models (Nano, Small, Medium variants), exported to Intel OpenVINO INT8 IR format, and to conduct comparative performance evaluation to identify the optimal configuration for PCB defect detection in repair contexts under uncontrolled lighting conditions.

**RO3:** To develop the CIRCA standalone desktop application incorporating a live webcam feed, a real-time OpenCV preprocessing pipeline (CLAHE, Gamma Correction, Laplacian Variance frame-dropping), and a zero-friction bounding box overlay interface, and to evaluate its performance using precision, recall, F1-score, mAP, and inference latency benchmarks.

***

## 1.5 Research Scope

To ensure focused and achievable research outcomes within the Final Year Project timeframe, this study establishes the following scope boundaries.

**Technical Scope.** The research focuses specifically on visual defect detection in PCBs using optical imaging and convolutional neural network analysis. The study encompasses the detection and classification of surface-visible defects that can be identified through optical inspection, including solder bridges, cold solder joints, missing components, misaligned components, damaged traces, burnt components, and physical board damage. The technical implementation focuses on deep learning approaches, specifically YOLOv12 variants deployed via Intel OpenVINO INT8 quantization. The research does not extend to electrical testing, functional verification, or the detection of defects requiring X-ray, infrared, or other specialised imaging modalities beyond standard optical photography.

**Device Scope.** The research concentrates on PCBs commonly found in consumer electronics devices, particularly those frequently encountered in repair shops, including smartphones, laptops, tablets, and similar portable electronic devices. The study does not address specialised industrial electronics, aerospace systems, medical devices, or other domain-specific applications that may require different detection criteria or certification requirements.

**Defect Scope.** The investigation focuses on manufacturing defects, physical damage, and wear-related failures that manifest visually on PCB surfaces, in alignment with IPC-A-610 visual inspection criteria. The research does not address intermittent faults, software-related issues, or defects that can only be detected through electrical testing or operational verification.

**Operational Scope.** The system is designed for use by repair technicians in small to medium-sized electronics repair facilities, targeting standard Intel CPU and iGPU hardware operating under Windows 10 and Windows 11. Standard USB webcams or smartphone camera tethers are assumed for image capture rather than specialised industrial imaging equipment. The system's performance will be evaluated against target benchmarks of greater than 90% mAP on the curated test dataset, sub-5 ms preprocessing pipeline execution per frame, real-time operation at a minimum of 15 FPS, and sub-10-second static image inference on a benchmark Intel Core i5 8th-generation equivalent processor (Yi and Mohamed, 2024).

**Limitations and Exclusions.** The research does not attempt to develop a commercially deployable product with full production-level features. The CIRCA system serves as a diagnostic aid and decision-support tool rather than an autonomous repair solution, with confidence score transparency built in to reduce the risk of automation bias (Goddard et al., 2011; Kupfer et al., 2023). Final repair decisions, component selection, and quality verification remain the responsibility of qualified technicians.

***

## 1.6 Significance of Study

This research makes several important contributions to both academic knowledge and practical application in the fields of computer vision, machine learning, and electronics repair services.

**Academic Significance.** From a research perspective, this study advances the application of deep learning techniques to a relatively underexplored domain — electronics repair diagnostics. While substantial research exists on automated defect detection in manufacturing contexts, the adaptation of these technologies to repair service environments presents unique challenges that have received limited scholarly attention (Bhattacharya and Cloutier, 2022; Lv et al., 2024). The findings should make an important contribution to the field by demonstrating how YOLOv12 models quantized to INT8 IR via Intel OpenVINO can achieve real-time PCB defect detection on commodity hardware, with implications for edge computing applications and mobile machine learning deployment (Tian et al., 2025; Kaewdook et al., 2024).

**Practical Significance.** For electronics repair businesses, CIRCA has the potential to reduce diagnostic time by at least 70%, dropping a 15-minute manual microscope inspection to under five minutes, which translates directly into higher technician throughput, shorter customer wait times, and the economic viability of complex board repairs that are currently abandoned as unprofitable (Djuberh et al., 2025). The system also addresses the physical dimension of the problem: by replacing prolonged microscope use with real-time camera-based inspection, CIRCA eliminates the severe eye strain that contributes to inspector fatigue and declining detection performance over the course of a repair shift. For novice technicians, the system serves as an educational tool and decision support aid, accelerating the learning curve and improving diagnostic confidence.

**Economic Significance.** The implementation of automated defect detection has important economic implications for the electronics repair industry. By reducing dependence on highly experienced technicians for routine diagnostics, repair shops can optimise their staffing models and reduce labour costs. The reduction in misdiagnosis and unnecessary component replacements represents direct cost savings through reduced parts wastage and more efficient inventory management (Hassan et al., 2025). For larger repair chains, these savings can accumulate to significant annual cost reductions while simultaneously improving customer satisfaction and retention.

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

The IPC-A-610 standard (Acceptability of Electronic Assemblies) is the most widely adopted industry reference for defining visual quality criteria in PCB inspection (Hassan et al., 2025). It specifies acceptable versus non-acceptable conditions for solder joint formation, fillet quality, component placement, and cleanliness across three product reliability classes: Class 1 for general electronics, Class 2 for dedicated service electronics, and Class 3 for high-reliability electronics such as medical or aerospace applications. In repair contexts, the IPC-A-610 taxonomy provides a meaningful basis for structuring the defect classes used in automated detection systems, ensuring that model outputs are grounded in industry-recognised quality definitions rather than arbitrary categories (Klco et al., 2023). A comparative study by Hassan et al. (2025) demonstrated that AI-driven AOI systems achieving IPC standard compliance can detect defects at 98 to 99% accuracy while processing over 5,000 components per hour, providing a strong quantitative basis for the performance targets adopted in this research.

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

CLAHE improves low-contrast images by dividing them into small local regions and equalising the histogram of each region individually, while limiting contrast amplification to prevent noise over-enhancement (Wanto et al., 2021). In the context of deep learning-based inspection, CLAHE has been shown to substantially improve model accuracy by enhancing the visibility of fine surface details that are critical for defect recognition. Wanto et al. (2021) demonstrated that applying CLAHE as a preprocessing step before CNN-based classification produces much higher accuracy results than without using CLAHE, attributing the improvement to CLAHE's ability to increase contrast and eliminate luminance imbalances that strengthen the CNN model's capacity to extract essential features. In CIRCA, CLAHE is applied to neutralise solder glare and restore surface texture visibility under intense workstation lighting.

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
| Lv et al. (2024) | Dataset contribution + baseline CNN | DsPCBSD+ (10,259 images) | Identifies blur and lighting as key dataset quality gaps | Baseline model only; no YOLO comparison |
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
| CLAHE | Wanto et al. (2021) | Increases CNN accuracy by boosting contrast and eliminating luminance imbalance in defect images |
| Adaptive Gamma Correction | Alhamzawi et al. (2025) | PSNR 17.386, SSIM 0.788 on low-light benchmarks; lifts shadow regions without over-exposing highlights |
| Laplacian Variance Blur Detection | Lv et al. (2024) | PCB-specific evidence that motion blur degrades model performance; justifies frame-quality gating |
| OpenVINO INT8 Quantization | Ahn et al. (2023) | 3.3× inference speedup over FP32 on Intel CPUs with minimal accuracy degradation |

***

## 2.8 Summary

In summary, a considerable amount of literature has been published on CNN-based PCB defect detection, and the findings consistently demonstrate that YOLO-based single-stage detectors offer the most favourable balance of accuracy, speed, and deployability (Yang and Yu, 2024; Liao et al., 2021; Tian et al., 2025). Furthermore, research on CLAHE, Gamma Correction, Laplacian Variance, and OpenVINO INT8 quantization collectively establishes a solid technical foundation for the preprocessing and deployment architecture underlying CIRCA. However, the research to date has been mostly restricted to manufacturing-oriented scenarios, and there remains a clear gap in accessible, repair-context PCB inspection systems that combine suitable model architectures, robust preprocessing for uncontrolled environments, edge ML deployment on commodity hardware, and human-factors-aware interface design (Lv et al., 2024; Kaewdook et al., 2024). CIRCA is positioned to address this gap directly, building on the strengths of YOLO-based detection and OpenVINO-accelerated edge inference while adapting the full system to the practical constraints of electronics repair. The methodology used to realise and evaluate this approach is described in Chapter 3.

***

### Complete Reference List (All Sources Used in Chapter 1 & 2)

| # | Citation | Source Type | Access |
|---|---|---|---|
| 1 | Adibhatla, V. A., et al. (2020). Defect detection in printed circuit boards using you-only-look-once convolutional neural networks. *Electronics, 9*(9), 1547. | Peer-reviewed journal | MDPI Open Access |
| 2 | Aggarwal, V., et al. (2022). Review of machine learning techniques for PCB defect detection. *Sensors, 22*(19), 7375. | Peer-reviewed journal | MDPI Open Access |
| 3 | Ahn, H., et al. (2023). Performance characterization of using quantization for DNN inference on edge devices. *arXiv:2303.05016*. | Preprint | arXiv / NSF PAR Free PDF |
| 4 | Alhamzawi, G. A., Alfoudi, A. S., and Alsaeedi, A. H. (2025). Fusion deep learning with adaptive gamma correction (DLAGC). *JISEM, 10*(36s). | Peer-reviewed journal | JISEM Open Access |
| 5 | Anh Nguyen, T., et al. (2024). PCB defect detection using hybrid CNN-Transformer with Wise-IoU. *Applied Sciences.* | Peer-reviewed journal | MDPI Open Access |
| 6 | Bhattacharya, D., and Cloutier, S. G. (2022). End-to-end deep learning framework for printed circuit board manufacturing defect classification. *Scientific Reports, 12*, 12969. | Peer-reviewed journal | Nature Open Access |
| 7 | Bhanumathy, M., et al. (2021). PCB defect detection using CNN. *IOP Conference Series.* | Conference paper | IOP Open Access |
| 8 | Djuberh, F., et al. (2025). Automated optical inspection (AOI) based on IPC standards. *SSRN Preprint.* | Preprint | SSRN Free Download |
| 9 | Ghelani, D. (2024). AI-driven quality control in PCB manufacturing. *International Journal of Engineering.* | Peer-reviewed journal | Open Access |
| 10 | Goddard, K., Roudsari, A., and Wyatt, J. C. (2011). Automation bias: a hidden issue for clinical decision support system use. *Stud Health Technol Inform, 164*, 17–22. | Peer-reviewed book chapter | PubMed / IOS Press |
| 11 | Hassan, et al. (2025). Automated optical inspection (AOI) based on IPC standards. *SSRN Preprint.* | Preprint | SSRN Free Download |
| 12 | Hendriko, V., and Hermanto, D. (2025). Performance comparison of YOLOv10, YOLOv11, and YOLOv12. *Brilliance: Research of AI, 5*(1), 440–450. | Peer-reviewed journal | ITSCIENCE Open Access |
| 13 | Hu, S., and Wang, H. (2020). PCB defect detection with GARPN+FPN+ResNet50. *IEEE Access.* | Peer-reviewed journal | IEEE Access |
| 14 | Huang, J., et al. (2023). NCE-Net for PCB defect classification. *Electronics.* | Peer-reviewed journal | MDPI Open Access |
| 15 | Kaewdook, D., et al. (2024). Design of deep learning techniques for PCBs defect detecting system based on YOLOv10. *ETASR, 14*(6). | Peer-reviewed journal | ETASR Open Access |
| 16 | Klco, P., et al. (2023). YOLOv8 for PCB defect detection. *Applied Sciences.* | Peer-reviewed journal | MDPI Open Access |
| 17 | Kupfer, A., et al. (2023). Check the box! Automation bias in AI-based decision support. *Frontiers in Psychology, 14*, 1118723. | Peer-reviewed journal | Frontiers Open Access |
| 18 | Law, C. Y., et al. (2024). Ensemble learning for PCB defect detection. *IEEE Access.* | Peer-reviewed journal | IEEE Access |
| 19 | Liao, X., et al. (2021). Lightweight YOLOv4-MobileNetV3 for PCB defect detection. *Sensors, 21*(23), 7873. | Peer-reviewed journal | MDPI Open Access |
| 20 | Lv, S., et al. (2024). A dataset for deep learning based detection of printed circuit board surface defect. *Scientific Data, 11*, 811. | Peer-reviewed journal | Nature Open Access (CC BY 4.0) |
| 21 | Sharma, A., et al. (2024). YOLOv5 vs OpenCV PCB defect study. *Journal of Engineering.* | Peer-reviewed journal | Open Access |
| 22 | Tian, Y., Ye, Q., and Doermann, D. (2025). YOLOv12: attention-centric real-time object detectors. *NeurIPS 2025*. arXiv:2502.12524. | Conference paper + preprint | arXiv Free PDF / NeurIPS |
| 23 | Wang, Z., et al. (2023). ATT-YOLO: self-attention enhanced YOLOv5 for PCB defect detection. *Journal of Physics: Conference Series.* | Conference paper | IOP Open Access |
| 24 | Wanto, A., et al. (2021). Optimization accuracy of CNN model by utilizing CLAHE. *IConNECT 2021, IEEE.* | Conference paper | UPI YPTK Repository |
| 25 | Yang, X., and Yu, H. (2024). PCB defect detection based on improved YOLOv8. *IJANMC.* | Peer-reviewed journal | Open Access |
| 26 | Yi, F., and Mohamed, A. S. A. (2024). YOLOv8-DEE: a high-precision model for PCB defect detection. *PeerJ Computer Science, 10*, e2548. | Peer-reviewed journal | PeerJ / PMC Open Access |