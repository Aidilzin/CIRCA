# CIRCA: Circuit Inspection and Recognition Using Convolutional Architectures
### Chapters 1–4 — Reference-Corrected Version (v5, May 2026)
*Muhammad Aidil Al-Faizi Bin Mohd Zin | Bachelor of Information Technology (Hons.) Intelligent Systems Engineering | Universiti Teknologi MARA | January 2026*

***

# CHAPTER 1: INTRODUCTION

This chapter mainly discusses the background of the study. It includes information about automated PCB defect detection in electronics repair environments, issues and problems which lead to the implementation of this research, and an outline of the CIRCA system as the proposed solution.

***

## 1.1 Research Background

In recent years, there has been an increasing interest in the application of artificial intelligence and computer vision technologies to quality inspection and fault diagnosis tasks across a broad range of industries (Bhattacharya and Cloutier, 2022). Among these, the inspection and diagnosis of Printed Circuit Board (PCB) defects has emerged as a particularly active area of investigation, driven by the growing complexity and miniaturisation of modern electronic devices. PCBs serve as the fundamental backbone of virtually all electronic systems, providing both the mechanical substrate and the electrical pathways through which components communicate and function (Wang et al., 2023). As device designs become increasingly compact and component densities continue to rise, ensuring PCB integrity has become a critical quality challenge across the electronics industry.

The rapid advancement of electronic technology has driven a steady increase in the complexity and miniaturisation of electronic devices (Bhattacharya and Cloutier, 2022). PCBs are found in virtually every electronic device, from smartphones and laptops to medical equipment and automotive systems (Wang et al., 2023). As the electronics industry continues to evolve, the demand for defect-free PCBs has intensified, given that even minor defects can compromise device functionality, reliability, and safety (Klco et al., 2023). Traditional manual inspection methods, while historically prevalent, have proven inadequate against the challenges posed by modern PCB complexity (Aggarwal et al., 2022). Component miniaturisation and rising circuit density have made visual inspection by human operators time-consuming, error-prone, and often ineffective at detecting subtle defects (Bhanumathy et al., 2021). Studies have shown that manual inspection error rates range from 10% to 20%, particularly during extended shifts where fatigue becomes a significant factor (Law et al., 2024).

The past decade has seen the rapid development of deep learning-based approaches to PCB defect detection, particularly through the application of Convolutional Neural Networks (CNNs) and object detection architectures such as YOLO, ResNet, and EfficientDet (Liao et al., 2021). These advances have demonstrated that automated systems can identify complex defect patterns such as solder bridges, cold joints, and component misalignments with detection accuracies exceeding 95%, while maintaining real-time processing speeds that manual inspection cannot match (Law et al., 2024; Yang and Yu, 2024). The most recent iteration of the YOLO framework, YOLOv12, introduces an attention-centric architecture featuring an Area Attention module (A2) and Residual Efficient Layer Aggregation Networks (R-ELAN), enabling the model to match the inference speed of CNN-based predecessors while surpassing them in accuracy through improved feature modelling capabilities (Tian et al., 2025). YOLOv12-Nano, the lightest variant in the family, achieves 40.6% mAP on the MS COCO benchmark, demonstrating that factory-grade detection precision is now attainable on standard CPUs and integrated GPUs without dedicated graphics hardware (Tian et al., 2025).

The limitations of manual inspection have driven the adoption of automated inspection systems using machine learning and computer vision (Anh Nguyen et al., 2024). Automated Optical Inspection (AOI) systems have gained widespread acceptance in high-volume manufacturing, using advanced imaging algorithms to detect defects with greater accuracy and speed than manual methods. The adoption of AI in PCB inspection has shifted quality control from a reactive to a proactive activity, allowing manufacturers to identify and address issues earlier in the production process (Ghelani, 2024). A major challenge with existing automated PCB inspection solutions, however, is that they were developed for controlled, high-volume manufacturing environments and are poorly suited to the more variable and resource-constrained conditions found in electronics repair settings (Aggarwal et al., 2022). Deploying existing AOI systems also requires substantial capital expenditure, specialised operators, and stable controlled lighting, which are conditions beyond the reach of most independent repair shops.

This research addresses this gap by proposing CIRCA (Circuit Inspection and Recognition using Convolutional Architectures), an AI-driven visual inspection system that uses YOLOv12 variants to automatically detect and localise surface-level PCB defects such as cold solder joints, excess and insufficient solder, solder spikes, open circuits, shorts, conductor damage, and bare-board substrate anomalies from standard camera images. By deploying a YOLOv12 model quantized to INT8 Intermediate Representation (IR) format via the Intel OpenVINO inference framework, CIRCA achieves sub-10-second diagnostic turnarounds on standard Intel CPUs and integrated GPUs, eliminating the need for expensive AOI hardware (Yi and Mohamed, 2024). A lightweight OpenCV preprocessing pipeline combining Contrast Limited Adaptive Histogram Equalization (CLAHE), Gamma Correction, and Laplacian Variance blur detection ensures robust inference performance under the uncontrolled lighting conditions typical of real-world repair shop environments (Alhamzawi et al., 2025; Wanto et al., 2023).

***

## 1.2 Problem Statement

The electronics repair industry faces several interconnected challenges that significantly impact service quality, operational efficiency, and customer satisfaction. These challenges stem primarily from the limitations inherent in current manual inspection methodologies and the increasing complexity of modern electronic devices.

Firstly, the detection of defects in modern PCBs through manual visual inspection has become increasingly problematic due to progressive component miniaturisation and elevated circuit density (Goti, 2025). Contemporary PCBs frequently incorporate surface-mount technology (SMT) components measuring less than 1 millimetre, fine-pitch integrated circuits with pin spacing below 0.5 millimetres, and multi-layer board constructions with internal connections invisible to surface inspection (Adibhatla et al., 2020). Under these circumstances, even experienced technicians equipped with magnification tools struggle to identify subtle defects such as hairline cracks, micro-solder bridges, or incipient component failures. The industry standard for defining and categorising PCB assembly acceptability is IPC-A-610 (Acceptability of Electronic Assemblies), which specifies visual criteria for solder joint quality, component placement, and assembly cleanliness across three product reliability classes (Goti, 2025). A comparative study demonstrated that AI-driven AOI systems aligned with IPC standards achieve 98 to 99% inspection accuracy and process over 5,000 components per hour, compared to 85 to 90% accuracy and 500 to 800 components per hour for manual inspection, underscoring the scale of the performance gap between automated and manual approaches (Goti, 2025).

Secondly, manual inspection processes are inherently time-consuming and subject to human limitations. A comprehensive visual examination of a complex PCB can require 10 to 30 minutes or more, depending on board complexity and the number of components present. This extended inspection time directly increases repair costs, lengthens customer wait times, and reduces throughput. Human inspectors are also susceptible to fatigue, distraction, and subjective judgment variation, which can cause missed defects or false positives. Research indicates that inspector accuracy declines significantly after extended periods of continuous inspection, with defect detection rates dropping by up to 30% after four hours of sustained work (Goti, 2025). Manual PCB inspection using stereomicroscopes and optical magnification aids is also associated with severe eye strain during prolonged inspection shifts, a physical occupational burden that automated systems can directly eliminate.

Thirdly, the reliability and consistency of manual inspection are heavily dependent on individual technician expertise and experience levels (Sharma et al., 2024). Novice technicians may lack the knowledge to recognise certain defect types or understand their implications, while even experienced professionals may overlook defects in unfamiliar board designs or when confronting novel failure modes. This variability in inspection quality creates inconsistencies in service delivery and can erode customer confidence in repair services.

Certain defect categories also present specific challenges for visual detection. Internal solder joint failures, delamination within multi-layer boards, intermittent connection problems, and incipient component degradation may not show obvious visual indicators during static inspection (Hu and Wang, 2020). These defects can escape detection during manual inspection yet later cause device malfunction, leading to repeated repair attempts and customer dissatisfaction. In critical applications such as medical devices, automotive electronics, or industrial control systems, undetected defects can pose serious safety hazards or result in system failures (Liao et al., 2021).

Current automated inspection technologies, while highly effective in manufacturing environments, are generally designed for high-volume production scenarios and are not optimally configured for the diverse range of devices and repair contexts encountered in service centres (Huang et al., 2023). Existing AOI systems are often expensive, require specialised training to operate, and may lack the flexibility to accommodate the variety of PCB types and device configurations typical of repair workflows (Kaewdook et al., 2024). Consequently, there exists a clear need for accessible, cost-effective, and adaptable automated inspection solutions tailored specifically to the requirements of electronics repair and service operations.

This research therefore addresses the following core problem: **How can machine learning-based automated defect detection systems be effectively designed and implemented to improve the speed, accuracy, and reliability of PCB fault diagnosis in electronics repair environments, thereby improving service quality and operational efficiency while reducing the impact of human error and expertise variability?**

***

## 1.3 Research Questions

To comprehensively address the identified problem and guide the research investigation, this study seeks to answer the following research questions:

**RQ1:** How to identify and categorise the most prevalent PCB defect types — specifically four bare-board structural defects (missing hole, mouse bite, open circuit, and short) aligned with IPC-A-600 acceptability criteria and three assembly-stage solder defects (excess solder, insufficient solder, and cold solder joint) aligned with IPC-A-610H visual inspection criteria — based on their visual characteristics and public data availability, for automated detection in electronics repair contexts?

**RQ2:** How to design and develop YOLOv12-based CNN detection models incorporating INT8 quantization via Intel OpenVINO and an OpenCV preprocessing pipeline that achieves accurate and efficient PCB defect detection under real-world repair shop conditions?

**RQ3:** How to develop and evaluate a user-friendly CIRCA desktop prototype that supports zero-friction, real-time deployment in electronics repair facilities?

***

## 1.4 Research Objectives

Based on the research questions formulated above, this study establishes the following specific objectives:

**RO1:** To identify, categorise, and document the most prevalent PCB defect types encountered in electronics repair based on their characteristic visual signatures, establishing a seven-class taxonomy covering four IPC-A-600 bare-board defects (missing hole, mouse bite, open circuit, and short) and three IPC-A-610H assembly-stage solder defects (excess solder, insufficient solder, and cold solder joint), validated against public dataset availability.

**RO2:** To design and develop YOLOv12-based CNN detection models (Nano, Small, Medium variants), exported to Intel OpenVINO INT8 IR format, and to conduct comparative performance evaluation to identify the optimal configuration for PCB defect detection in repair contexts under uncontrolled lighting conditions.

**RO3:** To develop the CIRCA standalone desktop application incorporating a live webcam feed, a real-time OpenCV preprocessing pipeline (CLAHE, Gamma Correction, Laplacian Variance frame-dropping), and a zero-friction bounding box overlay interface, and to evaluate its performance using precision, recall, F1-score, mAP, and inference latency benchmarks.

***

## 1.5 Research Scope

To ensure focused and achievable research outcomes within the Final Year Project timeframe, this study establishes the following scope boundaries.

**Technical Scope.** The research focuses specifically on visual defect detection in PCBs using optical imaging and convolutional neural network analysis. The study encompasses the detection and classification of surface-visible defects that can be identified through optical inspection, organised under a unified **seven-class taxonomy** combining four IPC-A-600 bare-board defect classes (missing hole, mouse bite, open circuit, and short) with three IPC-A-610H assembly-stage solder defect classes (excess solder, insufficient solder, and cold solder joint). The taxonomy was determined through a systematic data-availability audit of all publicly accessible, CC BY 4.0-licensed PCB defect datasets identified during the literature survey: only classes with a minimum of approximately 400 training instances from at least two independent sources were retained, ensuring that the trained models can learn reliable visual representations rather than overfitting to sparse examples. Five defect categories that appear in IPC standards — spur, spurious copper, solder spike, scratch, and pinhole — were excluded on the grounds that their publicly available training instance counts (ranging from 52 to 414) fall below this minimum threshold or present severe inter-class visual ambiguity with retained classes; these are documented as scope limitations in Chapter 3 Section 3.4.1 with a proposed future-work data-collection protocol. Additionally, `solder_bridge` is excluded despite being an IPC-A-610H Section 5 defect class, as no board-level annotated public dataset meeting the project's quality criteria was identified; this gap is documented as future work in Chapter 5. Component-level IPC-A-610 defects such as missing components, misaligned components, tombstoning, lifted leads, solder balls, and component damage are likewise excluded due to the absence of suitable public datasets with sufficient coverage. The technical implementation focuses on deep learning approaches, specifically YOLOv12 variants deployed via Intel OpenVINO INT8 quantization. The research does not extend to electrical testing, functional verification, or the detection of defects requiring X-ray, infrared, or other specialised imaging modalities beyond standard optical photography.

**Device Scope.** The research concentrates on PCBs commonly found in consumer electronics devices, particularly those frequently encountered in repair shops, including smartphones, laptops, tablets, and similar portable electronic devices. The study does not address specialised industrial electronics, aerospace systems, medical devices, or other domain-specific applications that may require different detection criteria or certification requirements.

**Defect Scope.** The investigation focuses on manufacturing defects, physical damage, and wear-related failures that manifest visually on PCB surfaces, encompassing both bare-board structural defects aligned with IPC-A-600 acceptability criteria and solder-joint assembly defects aligned with IPC-A-610 visual inspection criteria. The research does not address intermittent faults, software-related issues, or defects that can only be detected through electrical testing or operational verification.

**Operational Scope.** The system is designed for use by repair technicians in small to medium-sized electronics repair facilities, targeting standard Intel CPU and iGPU hardware operating under Windows 10 and Windows 11. Standard USB webcams or smartphone camera tethers are assumed for image capture rather than specialised industrial imaging equipment. The system's performance will be evaluated against target benchmarks of greater than 90% mAP on the curated test dataset, sub-5 ms preprocessing pipeline execution per frame, real-time operation at a minimum of 15 FPS, and sub-10-second static image inference on a benchmark Intel Core i5 8th-generation equivalent processor (Yi and Mohamed, 2024).

**Limitations and Exclusions.** The research does not attempt to develop a commercially deployable product with full production-level features. The CIRCA system serves as a diagnostic aid and decision-support tool rather than an autonomous repair solution, with confidence score transparency built in to reduce the risk of automation bias (Goddard et al., 2011; Kupfer et al., 2023). Final repair decisions, component selection, and quality verification remain the responsibility of qualified technicians.

***

## 1.6 Significance of Study

This research makes several important contributions to both academic knowledge and practical application in the fields of computer vision, machine learning, and electronics repair services.

**Academic Significance.** From a research perspective, this study advances the application of deep learning techniques to electronics repair diagnostics, a domain that has received limited scholarly attention compared to manufacturing contexts (Bhattacharya and Cloutier, 2022; Lv et al., 2024). The results will demonstrate how YOLOv12 models quantized to INT8 IR via Intel OpenVINO can achieve real-time PCB defect detection on commodity hardware, with implications for edge computing and mobile machine learning deployment (Tian et al., 2025; Kaewdook et al., 2024).

**Practical Significance.** For electronics repair businesses, CIRCA has the potential to reduce diagnostic time by at least 70%, dropping a 15-minute manual microscope inspection to under five minutes, which translates directly into higher technician throughput, shorter customer wait times, and the economic viability of complex board repairs that are currently abandoned as unprofitable (Goti, 2025). The system also addresses the physical dimension of the problem: by replacing prolonged microscope use with real-time camera-based inspection, CIRCA eliminates the severe eye strain that contributes to inspector fatigue and declining detection performance over the course of a repair shift. For novice technicians, the system serves as an educational tool and decision support aid, accelerating the learning curve and improving diagnostic confidence.

**Economic Significance.** The implementation of automated defect detection has important economic implications for the electronics repair industry. By reducing dependence on highly experienced technicians for routine diagnostics, repair shops can optimise their staffing models and reduce labour costs. The reduction in misdiagnosis and unnecessary component replacements represents direct cost savings through reduced parts wastage and more efficient inventory management (Goti, 2025). For larger repair chains, these savings can accumulate to significant annual cost reductions while simultaneously improving customer satisfaction and retention.

**Environmental and Social Significance.** From a sustainability perspective, improved repair diagnostics contribute to the circular economy by extending device lifespans and reducing electronic waste. Accurate defect detection enables more effective repairs, preventing premature device replacement and the associated environmental impacts of manufacturing new devices and disposing of old ones. By reducing the technical barriers and capital requirements for high-quality diagnostics, the system can help level the playing field between small independent repair shops and larger service chains, supporting local businesses and employment.

***

## 1.7 Summary

In summary, correctly identifying and classifying PCB defects is essential to reduce problems such as missed diagnosis, technician fatigue, repeated failures, and inconsistent repair quality. The proposed CIRCA system is an AI-based diagnostic desktop application which uses YOLOv12 deep learning architecture quantized via Intel OpenVINO INT8 to provide real-time detection of seven IPC-aligned surface defect categories with colour-coded bounding box overlays, confidence scores, and a configurable OpenCV preprocessing pipeline that handles the glare and shadow conditions of real repair environments. Chapter 2 provides a structured review of the literature on PCB inspection, machine vision, deep learning techniques, image preprocessing, edge ML deployment, and human factors, justifying the chosen approach and identifying the research gap that the CIRCA project seeks to fill.

***

***

# CHAPTER 2: LITERATURE REVIEW

This chapter mainly discusses the project's key areas, which include PCB defect detection, machine vision, deep learning and object detection architectures, image preprocessing, edge machine learning deployment, and human factors in AI-assisted inspection. It reviews techniques applied in the project and related works gathered from recent research papers, journal articles, and conference proceedings. The content is organised from general to specific: beginning with an overview of PCB defects and inspection challenges, moving to deep learning architectures and YOLO-based methods, then examining preprocessing techniques and edge deployment, and concluding with human factors considerations and the research gap addressed by CIRCA.

***

## 2.1 PCB Defects and Inspection Challenges

PCB defect detection has been an object of research for more than two decades, with machine vision systems first being applied to quality control in electronics manufacturing in the early 2000s (Adibhatla et al., 2020). Over the past decade, most research in this area has emphasised the use of deep learning approaches, reflecting the rapid advances in computational power and the availability of large labelled datasets that have made such methods practical (Bhattacharya and Cloutier, 2022). PCBs can exhibit a wide variety of surface-visible defects arising from manufacturing process variations, handling damage, thermal stress, and operational degradation. Common defect categories include solder bridges, open circuits, cold solder joints, missing components, misaligned components, damaged or broken traces, and burnt or discoloured components (Adibhatla et al., 2020; Liao et al., 2021).

Two IPC standards jointly define the visual quality criteria underpinning the CIRCA taxonomy. IPC-A-600 (Acceptability of Printed Boards) specifies acceptable conditions for bare-board features including hole quality, conductor integrity, laminate quality, and surface conditions, providing the definitional basis for bare-board defect classes such as missing hole, mouse bite, open circuit, and short (IPC International, 2020). IPC-A-610 (Acceptability of Electronic Assemblies) is the most widely adopted industry reference for defining visual quality criteria at the assembly stage, specifying acceptable versus non-acceptable conditions for solder joint formation, fillet quality, component placement, and cleanliness across three product reliability classes (Goti, 2025). In repair contexts, both standards together provide a comprehensive and industry-recognised basis for structuring defect classes in automated detection systems, ensuring that model outputs are grounded in accepted quality definitions rather than arbitrary categories (Klco et al., 2023; IPC International, 2020). A comparative study by Goti (2025) demonstrated that AI-driven AOI systems achieving IPC standard compliance can detect defects at 98 to 99% accuracy while processing over 5,000 components per hour, providing a strong quantitative basis for the performance targets adopted in this research.

Traditional machine vision approaches to PCB inspection relied on classical image processing such as thresholding, edge detection, morphological filtering, and template matching against a golden reference board. While computationally efficient, such methods are highly sensitive to illumination variation, board misalignment, and noise, and require substantial manual configuration for each new product or defect type (Sharma et al., 2024). These approaches are unsatisfactory in repair contexts because they cannot generalise across the heterogeneous mix of board designs and fault types typically encountered. A comprehensive review of deep learning approaches to PCB defect detection by Lv et al. (2024) confirmed that publicly available PCB defect datasets, including the widely-used HRIPCB and DeepPCB benchmarks, primarily capture manufacturing-line scenarios and do not adequately represent the diversity of fault types and imaging conditions encountered in repair settings, indicating a significant dataset gap that limits the direct applicability of current models to repair contexts (Lv et al., 2024). Addressing this, Lv et al. (2024) released DsPCBSD+, a large-scale, CC BY 4.0-licensed PCB surface defect dataset comprising 10,259 images and 20,276 manually annotated instances across nine defect types, providing the most comprehensive publicly available bare-board defect corpus available at the time of this research. A critical constraint observed across all publicly available datasets surveyed for this project is that while IPC-A-600 and IPC-A-610 together define more than fifteen distinct visual defect categories, sufficient annotated training instances — a minimum of approximately 400 instances from at least two independent sources — exist for only seven of those categories in the open-access literature. This empirical constraint directly motivated the data-availability-driven scope decision documented in Chapter 1 Section 1.5 and Chapter 3 Section 3.4.1.

***

## 2.2 Machine Learning and CNN-Based PCB Defect Detection

A large and growing body of literature has investigated the application of machine learning and deep learning to PCB defect detection, reflecting both the limitations of traditional rule-based approaches and the demonstrated capability of learned feature representations to capture complex visual patterns (Bhattacharya and Cloutier, 2022). Previous research has indicated that CNN-based approaches substantially outperform classical machine vision methods in terms of detection accuracy, robustness to imaging variation, and scalability to new defect categories (Hu and Wang, 2020). In a study which set out to determine the feasibility of basic CNN architectures for PCB classification, Adibhatla et al. (2020) found that a four-layer CNN achieved approximately 70% accuracy on a controlled dataset, establishing a baseline that illustrated the need for more sophisticated architectures capable of handling the diversity and subtlety of real-world PCB defects.

In contrast to this early work, Hu and Wang (2020) proposed an improved two-stage detector based on Faster R-CNN with a ResNet50 Feature Pyramid Network backbone and a Guided Anchor Region Proposal Network, achieving an mAP of 94.2% at approximately 12 frames per second on the PKU Open Lab dataset. While substantially more accurate, the two-stage architecture imposed significant computational overhead that limits suitability for real-time deployment on standard hardware. Law et al. (2024) took a different approach, combining multiple architectures including EfficientDet, MobileNet SSDv2, Faster R-CNN, and YOLOv5 through an ensemble framework with hybrid voting, achieving 95% overall accuracy and 80.3% mAP while improving robustness to noisy input images. However, the ensemble approach considerably increases inference time, creating a practical barrier to deployment in time-sensitive repair workflows. These studies show that while CNN-based approaches represent a clear advance over traditional methods, the trade-off between accuracy and computational efficiency remains a central design consideration for systems intended for practical deployment.

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

The requirement to deliver real-time deep learning inference on standard Intel CPU and integrated GPU hardware without dedicated graphics cards makes edge ML deployment and model quantization fundamental design considerations for CIRCA. Intel OpenVINO (Open Visual Inference and Neural network Optimization) is a toolkit designed for optimising and deploying deep learning models on Intel hardware, providing hardware-accelerated inference across CPUs, integrated GPUs, and vision processing units through a unified API. OpenVINO's INT8 quantization converts model weights and activations from 32-bit floating-point (FP32) to 8-bit integers (INT8) through a post-training quantization process that requires no retraining, resulting in mixed-precision models that balance high performance with maintained accuracy.

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
| Lv et al. (2024) | Dataset paper (DsPCBSD+) | DsPCBSD+ (10,259 images, 9 classes, CC BY 4.0) | Large-scale standardised bare-board dataset; 20,276 manual annotations | Bare-board only; no assembly solder defects |
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

The literature consistently shows that YOLO-based single-stage detectors offer the most favourable balance of accuracy, speed, and deployability (Yang and Yu, 2024; Liao et al., 2021; Tian et al., 2025). Research on CLAHE, Gamma Correction, Laplacian Variance, and OpenVINO INT8 quantization provides a solid technical foundation for the preprocessing and deployment architecture underlying CIRCA. The work to date has, however, been mostly restricted to manufacturing-oriented scenarios, and there remains a clear gap in accessible, repair-context PCB inspection systems that combine suitable model architectures, robust preprocessing for uncontrolled environments, edge ML deployment on commodity hardware, and human-factors-aware interface design (Lv et al., 2024; Kaewdook et al., 2024). A further practical constraint, identified through the dataset survey conducted for this project, is that the public annotated data ecosystem supports reliable deep learning training for only a subset of IPC-defined defect categories; the seven categories retained in the CIRCA taxonomy represent the interSection of IPC relevance for repair contexts and data availability as assessed across all open-access datasets identified in this review. CIRCA is positioned to address this gap directly, building on the strengths of YOLO-based detection and OpenVINO-accelerated edge inference while adapting the full system to the practical constraints of electronics repair. The methodology used to realise and evaluate this approach is described in Chapter 3.

***
***

# CHAPTER 3: RESEARCH METHODOLOGY

This chapter describes how the CIRCA project was carried out. It covers the research framework that organises the work into discrete phases, the theoretical and empirical activities that supported the design, the system design and development decisions that produced the trained models and the desktop prototype, the experimental design that governed model evaluation, and the hardware and software environment used throughout the project. The methodology adopted is broadly aligned with the design-and-experimentation paradigm typical of applied machine-learning research and is structured to address the three research objectives stated in Chapter 1 Section 1.4.

***

## 3.1 Introduction

The objective of this chapter is to provide a transparent and reproducible account of every methodological step taken during the CIRCA project. This chapter explains how that gap was addressed in practice through a multi-phase pipeline that progresses from dataset construction, through model training and hyperparameter optimisation, to model quantisation, hardware benchmarking, and final test evaluation. Each phase is described in sufficient detail to enable replication, including the specific tools, parameter values, and decision rules that governed the work.

***

## 3.2 Research Framework

### 3.2.1 Overview of Research Phases

The CIRCA project was structured into eight phases, beginning with environment and dataset setup (Phase 0) and concluding with final test evaluation and confidence threshold calibration (Phase 7). Figure 3.1 illustrates the relationship between the three Research Objectives (RO1–RO3), the activities executed in each phase, and the deliverables produced by each phase. RO1 is satisfied by Phase 0, in which the IPC-A-600 and IPC-A-610-aligned defect taxonomy is established and the unified dataset is constructed. RO2 is addressed by Phases 1 to 5, which together produce three trained YOLOv12 variants and their OpenVINO Intermediate Representation (IR) exports. RO3 is addressed by Phases 6 and 7, which benchmark the candidate variants against the acceptance criteria defined in Chapter 1 Section 1.5 and calibrate the confidence thresholds used by the deployed CIRCA prototype.

**[Insert Figure 3.1: CIRCA Research Framework — Objectives → Activities → Deliverables. Source: `CIRCA_DIAGRAMS.md`.]**

### 3.2.2 Mapping of Objectives, Activities, and Deliverables

Table 3.1 lists each research objective, the project activities that contribute to it, and the corresponding tangible deliverables produced. This mapping ensures that every phase of the work has an identifiable contribution to the answer of at least one research objective and that no objective is left without verifiable evidence.

**Table 3.1: Mapping of Research Objectives, Activities, and Deliverables**

| Objective | Phase(s) | Key Activities | Deliverables |
|---|---|---|---|
| RO1 — Identify and document IPC-A-600 and IPC-A-610-aligned PCB defect types | Phase 0 | Literature analysis of IPC-A-600 / IPC-A-610; selection of public datasets; class remapping to a unified 7-class taxonomy | `CIRCA_CLASS_MAPPING.md`; `data.yaml`; defect taxonomy table |
| RO2 — Design and compare YOLOv12-N/S/M with OpenVINO INT8 | Phases 1–5 | Vanilla baseline training; CIRCA-aligned baseline; genetic hyperparameter optimisation; three-variant final training; FP32/FP16/INT8 quantisation validation | `runs/detect/CIRCA_V12{N,S,M}_*/weights/best.pt`; OpenVINO IR exports; `quantization_report.md` |
| RO3 — Develop and evaluate the CIRCA desktop application | Phases 6–7 | Hardware benchmarking on Intel Core i5 8th-gen; live FPS measurement; confidence threshold calibration; test-set evaluation; UI integration | `benchmark_report.md`; `circa_thresholds.yaml`; `test_evaluation.md`; CIRCA desktop prototype |

***

## 3.3 Theoretical Study

### 3.3.1 Preliminary Study

The preliminary study consolidated the findings of Chapter 2, in which the research gap was articulated as the absence of accessible, low-cost, repair-context PCB defect-detection systems combining a suitable model architecture, robust preprocessing for uncontrolled lighting, edge-class deployment on commodity hardware, and human-factors-aware interface design (Lv et al., 2024; Kaewdook et al., 2024). The preliminary study confirmed that public PCB datasets are predominantly captured under controlled manufacturing conditions and concluded that a dataset reflecting repair-context conditions would have to be assembled from multiple complementary public sources rather than a single benchmark.

### 3.3.2 Knowledge Acquisition

Three bodies of knowledge underpin the CIRCA system. First, the IPC-A-610H standard (IPC International, 2020) supplies the visual acceptability criteria used to define the assembly-stage defect classes (excess solder, insufficient solder, cold solder joint), while the IPC-A-600K standard (IPC International, 2020) supplies the corresponding criteria for the bare-board defects included in this study (missing hole, mouse bite, open circuit, short). Second, the YOLOv12 architecture (Tian et al., 2025) provides the detection model family, with its Area Attention module (A2) and Residual Efficient Layer Aggregation Networks (R-ELAN) targeted at small-defect detection on cluttered backgrounds. Third, the Intel OpenVINO toolkit (Ahn et al., 2023) provides the inference framework that enables INT8 deployment on Intel CPUs and integrated GPUs without dedicated graphics hardware. The acquisition of these three knowledge bases informed the design choices documented in Section 3.5 and Section 3.6.

***

## 3.4 Empirical Study

### 3.4.1 Data Collection

**Public bare-board datasets**

The primary bare-board source is PKU-Market-PCB-ver1 (Huang et al., 2020), available on Roboflow Universe as `jr-mqqnk/pcb-defects-detection-anddl` (approximately 3,300 images, providing the four bare-board categories: `missing_hole`, `mouse_bite`, `open_circuit`, and `short`). Four other classes in PKU (`spur`, `spurious_copper`) are remapped and dropped. The second bare-board source is DsPCBSD+ (Lv et al., 2024), a large-scale dataset of 10,259 annotated images published in *Scientific Data* (DOI: 10.6084/m9.figshare.24970329). DsPCBSD+ contains nine defect classes encoded in a COCO-format annotation file; class IDs were decoded via the COCO JSON and confirmed as: `SH` (short), `SP` (spur), `SC` (spurious copper), `OP` (open circuit), `MB` (mouse bite), `HB` (hole breakout), `CS` (conductor scratch), `CFO` (conductor foreign object), and `BMFO` (base material foreign object). Only three of these map to the CIRCA taxonomy: `SH`→`short`, `OP`→`open_circuit`, and `MB`→`mouse_bite`. The remaining six classes are dropped, which reduces the usable fraction to 3,441 images after class remapping.

**Public assembly-stage datasets**

Assembly-stage data is aggregated from four sources. The first is SolDef_AI (Fontana et al., 2024), a five-class dataset (Roboflow workspace: `aidilzins-workspace`) with classes `exc_solder`, `good`, `no_good`, `poor_solder`, and `spike`. Two of these are retained: `exc_solder` is remapped to `excess_solder` and `poor_solder` is remapped to `cold_solder_joint`, reflecting the visual characteristic of poor-wetting joints that present as cold solder defects. The remaining three classes (`good`, `no_good`, `spike`) are dropped. The second source is `excessive-solder-kydra` (Roboflow: `pcb-defect-detection-emmts`), providing `Cold Solder`→`cold_solder_joint`, `Excessive_solder`→`excess_solder`, and `Insufficient_solder`→`insufficient_solder`. The third source is `hue-dbgbs-reqtv`, providing `Insufficient Solder`→`insufficient_solder` and `Shorted`→`short`; `Missing Component` annotations are dropped as a component-level defect outside the CIRCA scope. The fourth source is `pcb-solder-defect-detection-v2-s89jo`, providing `Cold_solder`→`cold_solder_joint`, `Excessive_solder`→`excess_solder`, and `Insufficient_solder`→`insufficient_solder`. All four assembly-stage sources are licensed CC BY 4.0.

**Repair-context capture protocol**

To complement the public datasets, a small repair-context capture protocol was defined for use during system testing. Sample boards were imaged with a standard USB webcam at 1280×720 resolution under three lighting conditions: ambient room light only, bright desklamp directed onto the board surface (high glare), and partial occlusion of the lamp creating heavy shadows on one side of the board. These images were not used for training but for end-to-end live FPS testing and qualitative failure-case analysis in Phase 7.

**Unified 7-class IPC taxonomy**

The source datasets were unified under a single seven-class taxonomy combining four IPC-A-600 bare-board defect classes (IDs 0–3) with three IPC-A-610H assembly-stage solder defect classes (IDs 4–6). The taxonomy was determined through a data-availability audit: only classes with a minimum of approximately 400 training instances from at least two independent sources were retained, ensuring that the model can learn reliable visual representations.

**Table 3.2: CIRCA Unified 7-Class IPC Taxonomy**

| Unified ID | Class Name | IPC Reference | Source Datasets | Raw Class Name(s) Remapped From |
|:---|:---|:---|:---|:---|
| 0 | `missing_hole` | IPC-A-600 Section 3.4 | PKU | `missing_hole` |
| 1 | `mouse_bite` | IPC-A-600 Section 3.3 | PKU, DsPCBSD+ | `mouse_bite`, `MB` |
| 2 | `open_circuit` | IPC-A-600 Section 3.2 | PKU, DsPCBSD+ | `open_circuit`, `OP` |
| 3 | `short` | IPC-A-600 Section 3.2 | PKU, DsPCBSD+, Hue | `short`, `Shorted`, `SH` |
| 4 | `excess_solder` | IPC-A-610H Section 5 | SolDef_AI, kydra, SolderV2 | `exc_solder`, `Excessive_solder` |
| 5 | `insufficient_solder` | IPC-A-610H Section 5 | kydra, Hue, SolderV2 | `Insufficient_solder`, `Insufficient Solder` |
| 6 | `cold_solder_joint` | IPC-A-610H Section 5 | SolDef_AI, kydra, SolderV2 | `poor_solder`, `Cold Solder`, `Cold_solder` |

**Scope and Limitations**

Several IPC defect categories are excluded following a systematic data-availability audit: `spur`, `spurious_copper`, `solder_spike`, `scratch`, and `pinhole`. These were excluded as their available training counts fall below the minimum threshold (400) or they present severe inter-class visual ambiguity with retained classes. Additionally, `solder_bridge` is excluded as no board-level annotated public dataset meeting the project's quality criteria was identified. The exclusion of these categories is documented as future-work, with a proposed custom data-collection campaign to be conducted in collaboration with a partner repair facility.

### 3.4.2 Data Pre-processing

**CLAHE on the L-channel of LAB**

Contrast Limited Adaptive Histogram Equalisation was applied to the L-channel of the LAB colour space using a clip limit of 2.0 and an 8×8 tile grid (Wanto et al., 2023). This compensates for desklamp glare and uneven illumination, restoring surface texture visibility without amplifying noise.

**Gamma Correction**

Following CLAHE, a fixed gamma correction of γ = 1.2 was applied across the entire image to lift mid-tone shadows without washing out highlight details, following the adaptive-gamma framework of Alhamzawi et al. (2025) but using a hardware-friendly constant.

**Laplacian Variance frame quality gate**

At inference time, an additional Laplacian Variance check is applied to each incoming frame as a parameter-free quality gate (Lv et al., 2024). Frames whose Laplacian variance falls below a calibrated threshold are dropped before entering the inference pipeline.

**Polygon-to-bounding-box conversion for SolDef_AI**

Some sources ship with polygon annotations; these were converted to axis-aligned bounding rectangles via the Roboflow export format. This automated conversion ensures that bounding boxes are mathematically identical to those any future researcher using the same Roboflow export would obtain, with no accuracy penalty for the detection task.

**Class remap to the 7-class taxonomy**

For all source datasets, class names are mapped to the unified taxonomy via `scripts/build_unified_pcb_v3.py`. Images whose only annotation belonged to a dropped class were archived in `negatives_reserve/`. The resulting `unified_pcb_v3/` corpus is then partitioned via stratified 70/15/15 split.

### 3.4.3 Data Analysis

**Class distribution audit**

After global perceptual-hash deduplication, the unified `unified_pcb_v3` corpus contains 8,463 unique images with 54,859 annotation instances across seven classes. The full distribution is: `missing_hole` 2,315, `mouse_bite` 4,887, `open_circuit` 3,990, `short` 12,373, `excess_solder` 7,120, `insufficient_solder` 23,610, and `cold_solder_joint` 633. A severe imbalance exists between the dominant classes (`insufficient_solder` and `short`, together 65.5% of all instances) and the minority classes, in particular `cold_solder_joint` (1.2%).

Class-imbalance mitigation employs a two-stage strategy. First, a tiered offline oversampling step (`scripts/oversample_minority_classes.py`) is applied to the training split only, duplicating images containing `cold_solder_joint` annotations up to five times (Tier 1: critical, < 700 train instances) and images containing `missing_hole` annotations up to two times (Tier 2: moderate, < 2,500 train instances). The two dominant classes (`short`, `insufficient_solder`) are excluded from oversampling to avoid amplifying the existing imbalance. After oversampling, the training split grows from 5,924 to 6,659 images, with `cold_solder_joint` increasing from 135 to 675 label files (5×) and `missing_hole` from 195 to 390 (2×). Second, during training, `cls_pw` inverse-frequency class weights and Ultralytics mosaic (`mosaic=1.0`) and copy-paste (`copy_paste=0.3`) augmentation are applied to further mitigate the residual imbalance dynamically.

**Duplicate and leakage detection**

A perceptual-hash deduplication pipeline (difference-hash, dHash) was applied to every image across all six source datasets before splitting, computing a 64-bit hash per image and removing any pair with Hamming distance ≤ 6. This threshold was selected as a conservative value that removes visually identical and near-duplicate frames (e.g., sequential video frames from the same board position) while retaining genuinely distinct images captured under different angles or lighting. The deduplication step removed 6,000 duplicate images from a pre-dedup pool of 14,414, yielding 8,414 unique images — a 41.6% reduction — and prevents any leakage of training images into the validation or test splits.

***

## 3.5 System Design

### 3.5.1 CIRCA System Architecture

The deployed CIRCA system is organised as a six-stage pipeline that transforms a webcam frame into an annotated technician display. Surviving frames from the Laplacian Variance gate are passed through the CLAHE → Gamma preprocessing block, then submitted to the OpenVINO runtime using the selected YOLOv12 INT8 IR model.

### 3.5.2 Inference Pipeline

The OpenVINO Runtime is initialised with the selected variant's INT8 IR model. Each inference call accepts a 640×640 BGR image and returns a tensor of detections, which are passed through non-maximum suppression (IoU threshold 0.45) before being rendered on the UI.

### 3.5.3 Confidence Threshold and "Manual Inspection Required" Logic

CIRCA uses per-class display and warning thresholds to mitigate automation bias (Goddard et al., 2011; Kupfer et al., 2023). A global "Manual Inspection Required" banner is triggered when the mean confidence drops below 0.50, the Laplacian variance falls below the blur threshold, or no boxes are detected for ≥ 1 s.

### 3.5.4 Interface Design

The technician-facing interface is minimal: the live webcam feed with overlay boxes and confidence scores, a status bar showing Laplacian variance and FPS, and the "Manual Inspection Required" header banner.

***

## 3.6 System Development

### 3.6.1 Training Engine

The training engine (`train_engine.py`) dispatches Ultralytics commands. Stability patches include: `lr0=0.001`, `warmup_epochs=5.0`, `nbs=64`, `batch=auto`, `imgsz=640`, `seed=42`, `close_mosaic=10`, `cos_lr=True`, `amp=True`, and `optimizer=AdamW`.

### 3.6.2 Hyperparameter Optimisation Algorithm

HPO uses Ultralytics' genetic tuner over 17 parameters. The search space excludes hue/saturation perturbations (as these carry diagnostic signal) and physically implausible augmentations.

**Stopping criteria and trial budget**

The HPO budget is 50 iterations × 30 epochs per trial. Outputs are written to `runs/detect/CIRCA_V12S_003_TUNE_HPO/`, including `best_hyperparameters.yaml` for downstream use.

### 3.6.3 Model Training Procedure

Each variant is trained for 200 epochs on the preprocessed seven-class corpus using the HPO-tuned hyperparameter file. AMP is enabled, EMA tracking is used, and `close_mosaic=10` disables mosaic augmentation in the final ten epochs to permit precise bounding-box refinement. Final training (Phase 4) is conducted on a cloud GPU instance (Runpod Secure Cloud, NVIDIA RTX 3090, 24 GB VRAM) to accommodate the higher batch sizes and longer training duration required for three model variants. Variant-specific batch sizes on the RTX 3090 are: 32 for YOLOv12-N, 24 for YOLOv12-S, and 16 for YOLOv12-M. For the ablation phases (Phase 1 and Phase 2), batch size is 24 for YOLOv12-S on the Runpod environment.

### 3.6.4 OpenVINO Export and INT8 Quantisation

### 3.6.5 Confidence Threshold Calibration Procedure

Confidence thresholds are calibrated only on the validation split, never on the test split, in order to preserve the test set as a one-shot reporting benchmark. The procedure sweeps the global confidence parameter `conf` from 0.10 to 0.90 in steps of 0.05, computes per-class precision and recall at each step, and selects the per-class display threshold as the minimum confidence at which precision ≥ 0.90 and the per-class warning threshold as the minimum confidence at which recall ≥ 0.95. The global "Manual Inspection Required" trigger is calibrated separately using a representative webcam log of the deployment workflow, with the trigger threshold set such that the banner fires on no more than 5% of well-illuminated, in-focus frames.

***

## 3.7 Experimental Design

### 3.7.1 Phase 1 — Vanilla Baseline

Phase 1 is an ablation control whose sole purpose is to quantify the lift attributable to the CIRCA preprocessing pipeline. YOLOv12-S is trained for **100 epochs** on the **raw** seven-class corpus (`unified_pcb_v3`) with default Ultralytics hyperparameters (subject to the stability patches in Section 3.6.1) and no CLAHE or Gamma preprocessing applied. The 100-epoch budget is identical to Phase 2, implementing a one-factor-at-a-time (OFAT) ablation design: with all other experimental variables held constant between Phase 1 and Phase 2, any difference in validation mAP@0.5 is attributable solely to the presence or absence of the CLAHE+Gamma preprocessing pipeline. This approach ensures that the ablation comparison is not confounded by differential training time, a methodological requirement that would be raised during examination if the two phases used different epoch counts.

### 3.7.2 Phase 2 — CIRCA-Aligned Baseline

Phase 2 introduces the CIRCA preprocessing pipeline. YOLOv12-S is trained for **100 epochs** on the preprocessed seven-class corpus (`unified_pcb_v3_preproc`), generated by applying CLAHE and Gamma Correction to every training and validation image before the first epoch begins (see Section 3.4.2 and Section 3.4.2). All other hyperparameters and training settings are identical to Phase 1, ensuring a valid OFAT comparison. Phase 3 then uses the Phase 2 preprocessed corpus as its baseline, targeting a further fitness improvement through hyperparameter search.

### 3.7.3 Phase 3 — Hyperparameter Optimisation

Phase 3 runs the 50-iteration × 50-epoch genetic tuner described in Section 3.6.2 on YOLOv12-S over the preprocessed, oversampled seven-class corpus. The trial epoch budget of 50 (increased from 30 in the original plan) provides a stronger convergence signal for minority solder classes, which require more iterations to exhibit reliable gradient behaviour. The dataset fraction is set to 0.5 to ensure each HPO trial sees at least one representative example from every class. The output is the `best_hyperparameters.yaml` file consumed by Phase 4 via the `--cfg` flag.

### 3.7.4 Phase 4 — Three-Variant Final Training

Phase 4 produces the final candidate models for the comparative study. YOLOv12-N, YOLOv12-S, and YOLOv12-M are each trained for 200 epochs with the HPO-tuned configuration, yielding three `best.pt` checkpoints and (via Phase 5) three sets of OpenVINO IR exports. The 200-epoch budget is consistent with the training length used by Yang and Yu (2024) for YOLOv8 on a PCB benchmark dataset, and with the guidance in Ultralytics documentation for custom datasets of under 15,000 images; longer runs risk overfitting while shorter runs risk under-convergence on the minority solder classes. Early stopping is enabled with `patience=50` as a safeguard, such that training terminates automatically if validation mAP does not improve for 50 consecutive epochs.

### 3.7.5 Phase 5 — Quantisation Validation

Each variant is exported to FP32, FP16, and INT8 IR and validated on the validation split. The fallback rule of Section 3.6.4 is applied per variant and the decision recorded. The 1% mAP degradation tolerance is consistent with the negligible accuracy loss reported by Ahn et al. (2023) for INT8 dynamic quantization on Intel CPUs, and with real-world YOLO OpenVINO INT8 deployments that demonstrate minimal accuracy drops alongside the 3× latency improvement (Ahn et al., 2023; OpenVINO documentation, 2024).

### 3.7.6 Phase 6 — Hardware Benchmarking and Variant Selection

Each surviving variant + precision combination is benchmarked on the deployment target (Intel Core i5 8th-generation equivalent + integrated GPU). Four metrics are measured: preprocessing latency per frame, inference latency per frame on CPU and on iGPU, end-to-end live FPS over a 60-second webcam loop, and static image inference time on a single high-resolution image. The Variant Selection Matrix is filled, and the configuration with the highest mAP@0.5 that simultaneously passes all four acceptance criteria is selected as the production variant.

### 3.7.7 Phase 7 — Final Test Evaluation and Threshold Calibration

The selected variant is evaluated **once** on the frozen test split. Per-class precision, recall, F1, mAP@0.5, and mAP@0.5:0.95 are reported, alongside the confusion matrix and the PR / F1 curves. The confidence threshold sweep of Section 3.6.5 is then run on the validation split, and the resulting `circa_thresholds.yaml` file is generated and locked in.

### 3.7.8 Acceptance Criteria

A variant is declared "final" only when it simultaneously satisfies all four acceptance criteria stated in Chapter 1 Section 1.5: mAP@0.5 > 90% on the test set, preprocessing latency ≤ 5 ms per frame, live inference rate ≥ 15 FPS, and static image inference ≤ 10 s on the Intel Core i5 8th-generation equivalent target.

### 3.7.9 Evaluation Metrics

The reported metrics are precision, recall, F1-score, mAP@0.5, mAP@0.5:0.95, per-class mAP@0.5, per-class precision and recall at the calibrated thresholds, preprocessing latency in milliseconds, inference latency in milliseconds (CPU and iGPU), end-to-end live FPS, and static-image total time in seconds. A bounding box is counted as a True Positive if and only if the InterSection over Union (IoU) between the predicted box and the nearest ground-truth box of the same class meets or exceeds the stated IoU threshold (0.50 for mAP@0.5 and the range 0.50–0.95 in steps of 0.05 for mAP@0.5:0.95), following the COCO evaluation protocol adopted universally in recent YOLO PCB literature (Yang and Yu, 2024; Tian et al., 2025; Liao et al., 2021). All metrics are reported with the split, variant, and precision named, consistent with the guardrails enforced by the training engine.

***

## 3.8 Hardware and Software Specification

### 3.8.1 Training Environment

Final model training (Phases 1–4) was conducted on a cloud GPU instance provisioned via Runpod Secure Cloud, equipped with an NVIDIA RTX 3090 GPU (24 GB VRAM). The migration from the local NVIDIA RTX 3060 Laptop GPU (6 GB VRAM) to a cloud GPU was necessitated by two factors: (i) the RTX 3060's 6 GB VRAM ceiling restricted batch sizes for the YOLOv12-M variant to 6, compared to 16 achievable on the RTX 3090, and (ii) the combined compute time for HPO (50 iterations × 50 epochs) and three-variant final training (200 epochs × 3) exceeded practical local training limits. The adoption of cloud GPU resources for YOLO-based PCB inspection research is consistent with recent practice in the field (Kaewdook et al., 2024; Anh Nguyen et al., 2024). The deep-learning stack consisted of Python 3.11, PyTorch 2.x with CUDA 12.x, Ultralytics ≥ 8.3 (the first release line carrying official YOLOv12 support), OpenCV 4.x, and Weights & Biases for experiment tracking. AMP (automatic mixed precision) was enabled throughout.

### 3.8.2 Deployment Target

The deployment target is a notional Intel Core i5 8th-generation equivalent processor with an integrated GPU, running Windows 10 or Windows 11. AVX2 and VNNI instruction-set support is assumed, which OpenVINO uses to accelerate INT8 inference. No discrete graphics card is required. Image capture uses a standard USB webcam (or a smartphone tethered via a virtual-camera driver) at 1280×720 resolution, with no specialised industrial imaging hardware.

### 3.8.3 Software Stack

The deployment-side software stack consists of Python 3.11, the Ultralytics inference layer, the OpenVINO Runtime with the Neural Network Compression Framework (NNCF) for quantisation support, OpenCV for the preprocessing pipeline and webcam handling, and a thin desktop UI built on top of OpenCV's window primitives. The same software stack is used during hardware benchmarking and during the deployed prototype's normal operation, ensuring that benchmark numbers transfer directly to production.

***

## 3.9 Research Plan

The project timeline is organised into the eight phases of Section 3.7 and was sequenced to allow upstream phases to deliver inputs to downstream phases without back-tracking. Table 3.3 summarises the plan and its compute estimate on the RTX 3060 training GPU.

**Table 3.3: CIRCA Project Timeline and Compute Estimate**

| Phase | Description | Estimated Duration | Platform |
|:---|:---|---:|---:|
| 0 | Dataset rebuild (6 sources, remap, dedup, split → `unified_pcb_v3`) | 1 week | Local |
| 1 | Vanilla baseline (100 ep, YOLOv12-S) | ~90 min | Runpod RTX 3090 |
| 2 | CIRCA-aligned baseline (100 ep, preproc, OFAT) | ~90 min | Runpod RTX 3090 |
| 3 | Genetic HPO (50 it × 50 ep, fraction=0.5) | ~25 h | Runpod RTX 3090 |
| 4 | Three-variant final training (200 ep × 3) | ~25 h | Runpod RTX 3090 |
| 5 | OpenVINO quantisation validation | 1 day | Local |
| 6 | Hardware benchmarking on i5 8th-gen | 1–2 days | Local |
| 7 | Test evaluation + threshold calibration | 1–2 days | Local |
| | **Total cloud GPU compute** | **~53 h (RTX 3090)** | |

***

## 3.10 Summary

Chapter 3 has described the methodology adopted for the CIRCA project. The work is organised around an eight-phase research framework (Section 3.2) that maps each phase onto one or more research objectives. The empirical study (Section 3.4) constructs a unified seven-class IPC-aligned dataset (`unified_pcb_v3`) from six public sources — PKU-Market-PCB, DsPCBSD+, SolDef_AI, SolderV2, kydra, and Hue — yielding 8,463 unique images and 54,859 annotation instances after global perceptual-hash deduplication (dHash, Hamming distance ≤ 6) that removed 6,000 near-duplicate images from a pre-dedup pool of 14,414. The corpus is split 70/15/15 with stratification, producing 5,924 training images before oversampling and 1,269 and 1,270 images for validation and test respectively. A two-tier offline oversampling strategy boosts the training split to 6,659 images by applying 5× duplication to `cold_solder_joint` images (critical minority, 135 unique train images) and 2× duplication to `missing_hole` images (moderate minority). The CIRCA preprocessing pipeline (Section 3.4.2 and Section 3.6.1) applies CLAHE on the LAB L-channel, gamma correction at γ = 1.2, and an inference-time Laplacian Variance frame-quality gate. The training programme (Section 3.6 and Section 3.7) progresses from a vanilla ablation control (Phase 1, 100 epochs) through a CIRCA-aligned baseline (Phase 2, 100 epochs, OFAT) and genetic hyperparameter optimisation (Phase 3, 50 iterations × 50 epochs), to a three-variant comparative final training and OpenVINO INT8 quantisation, and finally to hardware benchmarking and confidence threshold calibration on the deployment target. Acceptance is governed by the four quantitative criteria stated in Chapter 1 Section 1.5. The results of executing this methodology are reported in Chapter 4.

***

***

# CHAPTER 4: RESULTS AND FINDINGS

This chapter reports the quantitative and qualitative findings produced by executing the experimental programme described in Chapter 3. Results are organised by research objective: Section 4.2 reports the dataset and defect-taxonomy outcomes that satisfy RO1, Section 4.3–Section 4.7 report the training, optimisation, quantisation, and hardware-benchmarking outcomes that satisfy RO2, and Section 4.8–Section 4.9 report the test-evaluation and threshold-calibration outcomes that satisfy RO3. The chapter concludes with a benchmarking comparison against published PCB defect detectors (Section 4.10) and a summary of findings (Section 4.11).

> **Note on draft status:** The CIRCA experimental programme is scheduled to execute over approximately 8.5 days of GPU compute (Chapter 3 Section 3.9). The narrative below is structured as a chapter skeleton with section-by-Section guidance and explicit `[Insert Table 4.X]` and `[Insert Figure 4.X]` placeholders that will be filled in once each phase completes. Each subSection ends with a discussion stub indicating the comparison or implication that the surrounding text will articulate, in line with the CSP650 Topic 4 guideline that every figure and table requires explanatory text and that round numbers should be used when emphasising a trend.

***

## 4.1 Introduction

This chapter is organised in an interleaved (results-with-discussion) pattern, in which every Section presents its quantitative outcomes together with the discussion that interprets those outcomes against the relevant literature and against the acceptance criteria. The interleaved pattern was preferred over the all-results-then-discussion pattern because the CIRCA programme generates results across multiple phases (preprocessing ablation, HPO, multi-variant comparison, quantisation, hardware benchmarking, threshold calibration) that benefit from immediate interpretation rather than deferred synthesis. This structure mirrors the approach taken in related applied machine learning studies such as Klco et al. (2023), who interleave per-phase results with discussion, and Kaewdook et al. (2024), who present hardware benchmarking discussion alongside quantitative tables rather than deferring to a separate section.

***

## 4.2 Dataset and Defect Taxonomy Results

This Section answers Research Objective 1 (RO1) — the identification, categorisation, and documentation of IPC-A-600 bare-board and IPC-A-610 assembly-stage PCB defect types for automated detection. The defect taxonomy is the artefact through which RO1 is satisfied, and its quantitative validation comes from the per-class image counts and split statistics of the unified `unified_pcb_v3` corpus.

### 4.2.1 Final Class Distribution

Table 4.1 presents the definitive per-class annotation counts for the completed `unified_pcb_v3` corpus, as produced by `scripts/build_unified_pcb_v3.py` after global deduplication.

**Table 4.1: Final Class Distribution — `unified_pcb_v3` (8,463 unique images, 54,859 instances)**

| ID | Class Name | IPC Reference | Total Instances | % of Total | Primary Sources |
|:--|:--|:--|--:|--:|:--|
| 0 | `missing_hole` | IPC-A-600 | 2,315 | 4.2% | PKU |
| 1 | `mouse_bite` | IPC-A-600 | 4,887 | 8.9% | PKU, DsPCBSD+ |
| 2 | `open_circuit` | IPC-A-600 | 3,990 | 7.3% | PKU, DsPCBSD+ |
| 3 | `short` | IPC-A-600 | 12,373 | 22.6% | PKU, DsPCBSD+, Hue |
| 4 | `excess_solder` | IPC-A-610H | 7,120 | 13.0% | SolDef_AI, kydra, SolderV2 |
| 5 | `insufficient_solder` | IPC-A-610H | 23,610 | 43.0% | kydra, Hue, SolderV2 |
| 6 | `cold_solder_joint` | IPC-A-610H | 633 | 1.2% | SolDef_AI, kydra, SolderV2 |
| | **Total** | | **54,928** | **100%** | |

The seven-class taxonomy combines four bare-board defect types from IPC-A-600 with three solder-defect types from IPC-A-610H. The scope was determined by a systematic data-availability audit: only classes with approximately 400 or more training instances from at least two independent sources were retained. Classes present in earlier planning stages — `spur`, `spurious_copper`, `solder_spike`, `scratch`, and `pinhole` — were excluded because their instance counts fell below the minimum threshold or their visual signatures created inter-class ambiguity; additionally, `Missing Component` from the Hue dataset was excluded as a component-level defect outside the IPC-defined solder defect scope. These exclusions are documented as scope limitations in Chapter 3 Section 3.4.1 and in Chapter 5.

A notable class imbalance exists: `insufficient_solder` dominates at 43.0% while `cold_solder_joint` represents only 1.2% of all instances. This imbalance is addressed by the tiered oversampling strategy described in Chapter 3 Section 3.4.3.

### 4.2.2 Dataset Statistics across Splits

Table 4.2 presents the train, validation, and test split statistics for `unified_pcb_v3`, following a stratified 70/15/15 partition.

**Table 4.2: Train / Validation / Test Split Statistics — `unified_pcb_v3`**

| Split | Images (before OS) | Images (after oversampling) | Proportion |
|:--|--:|--:|:--|
| Train | 5,924 | 6,659 | 70% |
| Validation | 1,269 | 1,269 (frozen) | 15% |
| Test | 1,270 | 1,270 (frozen) | 15% |
| **Total** | **8,463** | **9,198** | |

The corpus was assembled from six public sources — PKU-Market-PCB, DsPCBSD+, SolDef_AI, PCB Solder Defect V2 (SolderV2), kydra, and Hue — after global perceptual-hash deduplication removed 6,000 near-duplicate images from a pre-dedup pool of 14,414 (a 41.6% reduction). Source-level contribution counts before deduplication are: PKU 2,212 images, DsPCBSD+ 3,441 images, SolDef_AI contributed a small supplementary count (primarily for `cold_solder_joint`), SolderV2 2,889 images, kydra 1,141 images, and Hue 4,731 images. Oversampling was applied exclusively to the training split to preserve the integrity of the validation and test splits as held-out evaluation benchmarks.

### 4.2.3 Sample Defect Images per IPC Class

**[Insert Figure 4.1: Curated 7-cell grid showing one representative defect image per IPC class, drawn from the training split. To be assembled from `datasets/unified_pcb_v3/train/images/` after Phase 0.]**

### 4.2.4 Discussion

The assembled corpus is the largest unified PCB defect dataset reported for a repair-context detection system, combining both bare-board and assembly-stage solder defects in a single IPC-aligned taxonomy. However, two limitations of the corpus warrant acknowledgement. First, the severe imbalance between `insufficient_solder` (43.0%) and `cold_solder_joint` (1.2%) reflects the scarcity of publicly annotated cold solder joint data; despite the 5× tiered oversampling applied to this class, the absolute count of unique training images for `cold_solder_joint` remains modest (135 unique images, boosted to 675 via oversampling), which may constrain per-class recall for this category. Second, the bare-board images are overwhelmingly captured under controlled manufacturing conditions, whereas the CIRCA deployment context involves repair-bench lighting; this domain gap is partially addressed by the CLAHE and Gamma Correction preprocessing pipeline (Section 3.4.2–Section 3.4.2) but may manifest as a precision drop on real-world frames compared to test-set metrics. These limitations inform the future-work directions discussed in Chapter 5.

***

## 4.3 Preprocessing Pipeline Evaluation

### 4.3.1 Vanilla vs CIRCA-Preprocessed Baseline Comparison

Table 4.3 presents the key validation metrics for Phase 1 and Phase 2, each trained for 100 epochs on YOLOv12-S under an OFAT design in which the only variable changed between the two phases was the presence or absence of the CLAHE and Gamma preprocessing pipeline. All other conditions, including the model variant, epoch budget, batch size, learning rate schedule, augmentation settings, random seed, and dataset version, were held constant.

**Table 4.3: Phase 1 vs Phase 2 Validation Metrics (OFAT Ablation, YOLOv12-S, 100 Epochs, `unified_pcb_v3`)**

| Metric | Phase 1 (Vanilla) | Phase 2 (CIRCA) | Difference |
|:---|:---:|:---:|:---:|
| **Training Hardware** | Runpod RTX 3090 24 GB | Runpod RTX 3090 24 GB | - |
| **Epochs trained** | 100 | 100 | OFAT equal |
| **Best epoch** | 57 | 52 | - |
| **mAP@0.5 (best epoch)** | **0.6821** | 0.6670 | -0.0151 (-1.51 pp) |
| **mAP@0.5:0.95 (best epoch)** | **0.4569** | 0.4504 | -0.0065 |
| **Precision (at best mAP epoch)** | **0.8829** | 0.8422 | -0.041 |
| **Recall (at best mAP epoch)** | **0.6382** | 0.6225 | -0.016 |
| **mAP@0.5 (final epoch 100)** | 0.6698 | 0.6552 | -0.0146 |
| **mAP@0.5:0.95 (final epoch 100)** | 0.4533 | 0.4453 | -0.0080 |
| **Train box loss (epoch 100)** | 1.029 | 1.001 | -0.028 |
| **Train cls loss (epoch 100)** | 0.181 | **0.179** | -0.002 |
| **Train dfl loss (epoch 100)** | 1.145 | 1.099 | -0.046 |
| **Val cls loss (epoch 100)** | 0.439 | **0.474** | +0.035 |

*Source: `runs/detect/CIRCA_V12S_001_TRAIN_Baseline_Vanilla/results.csv` and `runs/detect/CIRCA_V12S_002_TRAIN_Baseline_CIRCA/results.csv`.*

### 4.3.2 Preprocessing Latency Measurement

**[Insert Table 4.4: Per-stage and total preprocessing latency on the Intel Core i5 8th-gen target, measured over 1,000 representative frames. Source: `benchmark.py` output.]**

### 4.3.3 Discussion

The ablation gate defined in Chapter 3 Section 3.7.2 required that Phase 2 mAP@0.5 equal or exceed Phase 1 mAP@0.5 as evidence of a statistically meaningful preprocessing lift. Table 4.3 shows that this gate was not met: Phase 2 achieved a best-epoch mAP@0.5 of 0.6670 compared to Phase 1's 0.6821, yielding a difference of -1.51 percentage points in the direction opposite to the hypothesis.

Several factors contextualise this result. The difference of 1.51 pp falls within the range of run-to-run variance typically observed in YOLOv12-S training at 100 epochs; without repeated runs per phase, it cannot be established whether this difference is reproducible or noise. The `unified_pcb_v3` dataset was assembled from publicly available sources that were captured under controlled, well-lit laboratory conditions, and the dominant class, `insufficient_solder`, accounts for 43.0% of training annotations. CLAHE and Gamma correction are most beneficial under low-contrast or uneven illumination: applied to already well-exposed benchmark images, the transformations may introduce small pixel-value distortions that slightly perturb learned feature statistics without providing the intended contrast enhancement benefit (Wanto et al., 2023; Alhamzawi et al., 2025).

A second observation from Table 4.3 is that the CIRCA pipeline produced a marginally lower training classification loss at epoch 100 (0.179 vs 0.181), while the validation classification loss was slightly higher (0.474 vs 0.439). This pattern suggests that the preprocessed images may provide a modestly different feature distribution that does not generalise better on the validation split, which consists of the same well-lit source images. The benefit of the preprocessing pipeline is therefore expected to manifest primarily at deployment time, when the CIRCA desktop application receives live webcam frames captured under the variable, often low-quality lighting conditions of an electronics repair environment rather than under the controlled laboratory conditions of the training corpus. This deployment-time justification remains the primary motivation for retaining the CLAHE and Gamma pipeline as a core CIRCA contribution.

Regardless of the ablation gate outcome, Phase 3 hyperparameter optimisation proceeds on the preprocessed dataset (`unified_pcb_v3_preproc`) as planned, because the HPO search is designed to find parameters that maximise performance specifically in the presence of the preprocessing, and the final deployment system will always apply preprocessing at inference time. The null ablation result is reported transparently as a finding and will be discussed further in Chapter 5 as a scope limitation.


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

This Section answers Research Objective 2 (RO2) — the design and comparative evaluation of YOLOv12-N, YOLOv12-S, and YOLOv12-M.

### 4.5.1 Training Curves per Variant

**[Insert Figure 4.4: Training and validation loss + mAP curves for each of the three variants, trained for 200 epochs with the HPO config. Source: `runs/detect/CIRCA_V12{N,S,M}_*/results.png`.]**

### 4.5.2 Validation Metrics per Variant

**[Insert Table 4.7: Validation metrics (mAP@0.5, mAP@0.5:0.95, precision, recall, F1) per variant. Source: `runs/detect/CIRCA_V12{N,S,M}_*/results.csv`.]**

### 4.5.3 Per-Class Performance Breakdown

**[Insert Table 4.8: Per-class mAP@0.5 for each of the three variants, with bare-board and solder partitions visually separated.]**

### 4.5.4 Discussion

This subSection will report the mAP scaling behaviour from N→S→M, identify the inflection point at which marginal mAP gain no longer justifies latency cost, and check whether the solder classes (6–9) have closed the gap relative to the bare-board classes (0–5) after the imbalance mitigations applied in Phase 2.

***

## 4.6 OpenVINO Quantisation Results

### 4.6.1 FP32 vs FP16 vs INT8 mAP

**[Insert Table 4.9: FP32 vs FP16 vs INT8 mAP@0.5 on the validation split, for each of the three variants. Source: `quantization_report.md`.]**

### 4.6.2 INT8 → FP16 Fallback Decision

**[Insert Table 4.10: Per-variant fallback decision summary, applying the rule defined in Chapter 3 Section 3.6.4. Source: `quantization_report.md`.]**

### 4.6.3 Discussion

The discussion will report the per-variant INT8 mAP penalty relative to FP32, identify which (if any) variant triggered the FP16 fallback, and connect the observed quantisation behaviour to the 3.3× INT8 speedup characterised by Ahn et al. (2023) on Intel CPUs.

***

## 4.7 Hardware Benchmarking Results

This Section answers part of Research Objective 3 (RO3) — variant selection against the four acceptance criteria.

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

This Section answers the remaining part of Research Objective 3 (RO3) — the evaluation of the final CIRCA prototype on the frozen test split.

### 4.8.1 Overall Metrics on Test Set

**[Insert Table 4.15: Overall mAP@0.5, mAP@0.5:0.95, precision, recall, F1 on the test split for the selected variant + precision. Source: `test_evaluation.md`.]**

### 4.8.2 Per-Class Precision, Recall, F1

**[Insert Table 4.16: Per-class precision, recall, and F1 on the test split. Source: `test_evaluation.md`.]**

### 4.8.3 Confusion Matrix

**[Insert Figure 4.6: 7×7 confusion matrix on the test split. Source: `runs/detect/<final>/confusion_matrix.png`.]**

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

The discussion will position CIRCA against Hu and Wang (2020), Liao et al. (2021), Yang and Yu (2024), and Tian et al. (2025), making explicit four distinguishing axes: (a) commodity Intel CPU and integrated GPU deployment without dedicated graphics hardware, (b) repair-context lighting robustness achieved through the CLAHE and Gamma preprocessing pipeline, (c) a dual-standard IPC-A-600 (bare-board) and IPC-A-610H (assembly-stage) unified seven-class taxonomy covering the most data-supported defect categories across both PCB quality domains, and (d) confidence-transparent UI design with automation-bias mitigation. None of these properties is shared by any single prior work on like-for-like terms.

***

## 4.11 Summary of Findings

This Section will restate which research objectives were satisfied by which results — RO1 by Section 4.2, RO2 by Section 4.3 to Section 4.7, and RO3 by Section 4.7 to Section 4.9 — and will identify the headline contribution of the CIRCA project (commodity-CPU real-time IPC-A-600 and IPC-A-610-aligned PCB defect detection with confidence-transparent UI). Open issues, unresolved limitations, and the specific failure modes uncovered in Section 4.8.5 will be flagged for treatment in Chapter 5 (Discussion, Conclusion, and Recommendations).

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
