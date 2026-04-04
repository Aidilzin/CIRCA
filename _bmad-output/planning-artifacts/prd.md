---
stepsCompleted:
  - step-01-init
  - step-02-discovery
  - step-02b-vision
  - step-02c-executive-summary
  - step-03-success
  - step-04-journeys
  - step-05-domain
  - step-06-innovation
  - step-07-project-type
  - step-08-scoping
  - step-09-functional
  - step-10-nonfunctional
  - step-11-polish
inputDocuments:
  - "docs/FYP_Proposal_(Chap 1-3).pdf"
  - "bmad_custom/module.yaml"
documentCounts:
  productBriefs: 0
  research: 0
  brainstorming: 0
  projectDocs: 2
classification:
  projectType: desktop_app
  architectureRequirements: "Edge ML Inference via Intel OpenVINO (YOLOv12 explicitly exported to INT8 IR format targeting sub-10s diagnostics on standard CPUs/iGPUs)"
  domain: process_control / scientific
  complexity: high
  projectContext: brownfield
workflowType: 'prd'
---

# Product Requirements Document - CIRCA

**Author:** Aidil
**Date:** 2026-04-01

## Executive Summary

CIRCA is a standalone desktop application designed to completely eliminate the diagnostic bottleneck and severe eye strain associated with manual printed circuit board (PCB) inspection. Targeting repair technicians working under strict time constraints, CIRCA replaces slow microscope checks with real-time, AI-driven visual inspection. By achieving sub-10 second diagnostic turnarounds, the system restores profitability to complex microscopic board repairs. The application integrates seamlessly into existing independent repair shop environments by utilizing standard webcams and laptop hardware, immediately removing the barrier of entry posed by bulky, prohibitively expensive Automated Optical Inspection (AOI) machines.

### What Makes This Special

CIRCA differentiates itself through its Live Video Feed UI combined with highly optimized Edge ML architecture. Rather than relying on a disjointed "take photo, upload, wait" workflow, technicians simply slide the PCB under a standard camera mount to see dynamically tracking, color-coded bounding boxes instantly identify critical defects (solder bridges, missing/misaligned components, damaged traces, and burnt areas). This factory-grade precision (>90% mAP) is democratized for consumer hardware via an Intel OpenVINO backend, utilizing YOLOv12 models quantized to INT8 IR format to achieve real-time performance on standard standalone CPUs/iGPUs without requiring dedicated graphics cards.

## Project Classification

- **Project Type:** Desktop Application (Standalone monolithic client)
- **Domain:** Process Control / Scientific (Electronics Manufacturing QA)
- **Complexity:** High (Real-time Live Video Feed UI, Intel OpenVINO Hardware Abstraction, YOLOv12 INT8 Quantization constraints)
- **Project Context:** Brownfield (FYP YOLOv12 prototyping baseline)

## Success Criteria

### User Success
- **Time Reduction:** Technicians reduce PCB inspection time by at least 70% compared to manual microscopic inspection (dropping a 15-minute inspection to under 5 minutes).
- **Physical Relief:** Complete elimination of severe eye strain associated with prolonged manual microscope use.
- **Zero-Friction Workflow:** No "Capture" or wait logic required; the live feed immediately snaps bounding boxes onto defects in real-time.

### Business Success
- **Repair Profitability:** Increases the volume of boards a single technician can diagnose per day, turning previously unprofitable complex repairs into high-margin jobs.
- **Capital Expenditure (CapEx) Savings:** Runs entirely on the shop's existing laptops and webcams, requiring zero expenditure for proprietary AOI hardware.
- **Thesis Completion:** Successful submission and defense of the FYP, proving the practical viability of edge AI in electronics QA.

### Technical Success
- **Model Accuracy:** YOLOv12 model achieves **>90% mAP** across the strict defect taxonomy (solder bridges, missing components, misaligned components, damages/burnt areas).
- **Image Preprocessing Resiliency:** An ultra-lightweight OpenCV pipeline executing in <5ms before OpenVINO inference handles uncontrolled environments. Mandates CLAHE to neutralize solder glare, Gamma Correction to lift heavy shadows, and Laplacian Variance blur detection to automatically drop moving frames, saving CPU cycles.
- **Inference Speed:** Real-time live video feed processing, or sub-10 second inference turnaround for high-resolution static captures.
- **Edge ML Execution:** Successful deployment of the YOLOv12 model via **Intel OpenVINO INT8 IR format**, running natively on standard Intel CPUs/iGPUs without crashing or requiring discrete graphics cards.

### Measurable Outcomes
- >90% mAP on the curated test dataset.
- <5ms OpenCV preprocessing pipeline execution time per frame.
- Real-time bounding box tracking or <10s processing time per board.
- 70% reduction in average diagnostic time per PCB.

## Project Scoping & Phased Development

### MVP Strategy & Philosophy
- **MVP Approach:** "Prove the Engine." The goal of the MVP is to mathematically and practically prove that AI-driven PCB defect detection can be democratized. If the core live-feed OpenVINO inference works under the <10s target, the product is a success. Non-essential features (e.g., exporting PDFs) do not contribute to the academic thesis and are deferred.
- **Resource Requirements:** 1 Solo Developer (FYP Student) acting as ML Engineer, Desktop App Developer, and QA Tester.

### MVP Feature Set (Phase 1)
**Core User Journeys Supported:**
- Technician successfully inspecting a board under standard/ideal lighting.
- Technician compensating for extreme glare/shadows via the settings panel.

**Must-Have Capabilities:**
- YOLOv12 model successfully quantized to Intel OpenVINO INT8 IR format.
- Live USB webcam/camera feed integration.
- Ultra-lightweight OpenCV preprocessing pipeline (CLAHE, Gamma Correction, Laplacian Variance frame-dropping).
- Zero-click User Interface with real-time color-coded bounding box overlays and Confidence Scores.
- Configuration sidebar to manually adjust OpenCV environmental parameters.

### Post-MVP Features
**Phase 2 (Growth):**
- Local PDF Report generation mapping defect locations for customer ticketing.
- Defect logging history to track common failure points across specific device models.
- Support for 4K external USB microscopes.

**Phase 3 (Expansion):**
- Integration with reverse-engineered CAD/BoardView schematics to cross-reference exactly *which* component is missing based on its board coordinates.
- Motorized stage integration for automatic, hands-free scanning of large motherboards.

### Risk Mitigation Strategy
- **Technical Risks (Latency & Hardware Overhead):** High risk of the OpenCV pipeline or inference engine maxing out an older CPU. **Mitigation:** Rely strictly on Laplacian Variance to aggressively drop blurred frames. If INT8 quantization degrades mAP below 90%, fallback to FP16 OpenVINO execution.
- **User/Environmental Risks (Changing Shop Conditions):** High risk of the model failing when a technician changes desklamps. **Mitigation:** Exposing the OpenCV settings to the user directly, allowing them to visually "tune" out the glare rather than failing the model.
- **Resource Risks (Solo Developer Burnout):** Running out of time before the FYP deadline. **Mitigation:** Strict enforcement of the "Offline-Only Windows-Only" boundary. Zero time spent writing backend APIs, authentication protocols, or Linux driver compatibility.

## User Journeys

### 1. Leo the Repair Technician (Primary - Happy Path)
- **The Scene:** Leo is staring down a stack of 15 unresponsive motherboards that need diagnosis by the end of his shift. He suspects a microscopic solder bridge on a power management IC but dreads the 15-minute hunch over his optical microscope.
- **The Action:** Instead of the microscope, Leo slides the motherboard under his desk-mounted webcam and launches CIRCA. He slowly pans the board under the lens. 
- **The Climax:** In real-time, the ultra-lightweight OpenCV pipeline neutralizes the harsh LED glare from his desklamp. The YOLOv12 OpenVINO model instantly snaps a red bounding box onto the solder bridge with a 94% confidence score. 
- **The Resolution:** Leo spots the exact failure point in less than 10 seconds. He applies flux, wipes the bridge, and resolves the repair profitably. He ends his shift without the usual crushing eye strain.

### 2. Leo the Repair Technician (Primary - Edge Case & Recovery)
- **The Scene:** Leo is inspecting a heavily water-damaged and burnt logic board. The carbon scoring creates deep, heavy shadows, and he's moving the board quickly and erratically in frustration.
- **The Action:** As he aggressively slides the board under the camera, motion blur would normally drown an AI model in false positives or crash the queue.
- **The Climax:** CIRCA's Laplacian Variance detects the aggressive motion blur and smartly drops those specific frames, saving CPU cycles. The moment he pauses, the Gamma Correction blasts the shadows, allowing the AI to detect the burnt traces instantly despite the terrible contrast.
- **The Resolution:** The system remains fluid and responsive (under 10s latency). Leo correctly scopes the jumper wire repairs instead of incorrectly assuming the board is a total loss.

### 3. Sarah the FYP Admin / Shop QA Lead (Admin/Settings Journey)
- **The Scene:** Sarah oversees the repair shop and realizes that after upgrading the shop's overhead shoplights to bright LEDs, CIRCA's bounding boxes are getting slightly less accurate due to extreme new glare.
- **The Action:** She opens CIRCA's Configuration tab to access the OpenCV preprocessing parameters.
- **The Climax:** She tweaks the CLAHE clip limit and Gamma Correction value sliders. A live-preview window immediately shows the glare neutralizing, confirming the new baseline.
- **The Resolution:** She saves the profile. The YOLOv12 model's >90% mAP is restored for all technicians in the shop without requiring any model retraining.

## Domain-Specific Requirements

### Compliance & Regulatory
- **Quality Standards Alignment:** The defect taxonomy (solder bridges, misaligned components, etc.) must loosely align with recognized electronics manufacturing acceptability standards (e.g., IPC-A-610 guidelines) to ensure industry-relevant diagnostics.
- **Thesis/Academic Compliance:** The system's YOLOv12 benchmarking methodology must be rigorously documented to satisfy FYP academic validation and reproducibility requirements.

### Technical Constraints
- **Hardware Endurance (Thermal Management):** The desktop application must be capable of operating continuously during a standard 8-hour repair shift without causing the host laptop to thermally throttle. Throttling would immediately destroy the <5ms OpenCV and sub-10s OpenVINO latency targets.
- **Environmental Reliability:** The AI UI and OpenCV pipeline must remain robust against dirty optical lenses, flux smoke, and intense, shifting workstation lighting.

### Integration Requirements
- **Local Network Isolation:** The system must function entirely offline. This protects proprietary repair shop methodologies/data and entirely eliminates network latency as a bottleneck.
- **Scope Containment (No Cloud APIs):** Strict avoidance of external ticketing systems or cloud API integrations to prevent FYP scope creep. The only export feature allowed is a simple local PDF report generation (Post-MVP Growth Feature).
- **Hardware Agnosticism:** Must interface seamlessly with standard OS-level UVC (USB Video Class) camera drivers across a wide variety of cheap webcams or smartphone tether apps.

### Risk Mitigations
- **False Negative Handling:** Visual "Confidence Scores" must be transparently displayed on all bounding boxes. If inference confidence drops below a defined threshold, the UI must clearly indicate "Manual Inspection Required" rather than failing silently.

## Innovation & Novel Patterns

### Detected Innovation Areas
- **Edge ML Democratization:** Running state-of-the-art YOLOv12 object detection on standard retail CPU/iGPU hardware using Intel OpenVINO INT8 quantization, directly replacing the need for prohibitively expensive, proprietary AOI hardware in small repair shops.
- **Dynamic Preprocessing Gating:** Using a sub-5ms OpenCV pipeline (CLAHE, Gamma Correction) coupled with Laplacian Variance to actively *gate and drop* frames containing physical motion blur, drastically reducing CPU load during rapid physical board manipulation by the technician.

### Market Context & Competitive Landscape
Traditional PCB inspection in independent repair shops is entirely manual (stereomicroscopes), which causes extreme eye strain and massive diagnostic bottlenecks. Conversely, factory-level inspection uses Automated Optical Inspection (AOI) machines which are physically massive, prohibitively expensive, and require highly controlled physical lighting environments out of reach for independent shops. CIRCA bridges this market gap by making AOI-level accuracy accessible on standard laptops in unconstrained physical lighting environments.

### Validation Approach
- **mAP Benchmarking:** Validating the OpenVINO INT8 quantized model against the original YOLOv12 baseline to ensure that structural quantization hasn't degraded accuracy below the >90% mAP threshold.
- **Latency Testing:** Rigorously profiling the OpenCV preprocessing pipeline to mathematically prove execution takes <5ms per frame on mid-range Intel silicon.

### Risk Mitigation
- **Quantization Degradation:** If INT8 quantization degrades accuracy significantly during validation, the fallback is executing in FP16 inference, which will sacrifice some real-time FPS but maintain absolute diagnostic accuracy.
- **Over-Reliance / Automation Bias:** The risk of a technician blindly trusting a false negative. This is mitigated by enforcing transparent Confidence Scores embedded directly above bounding boxes to prompt manual double-checks when necessary.

## Desktop-Specific Requirements

### Project-Type Overview
CIRCA operates as a standalone Windows client application designed for immediate, zero-friction deployment on repair shop floors. The architecture radically prioritizes self-contained local processing over network integrations, guaranteeing maximum stability of the hardware-accelerated OpenVINO inference engine.

### Platform Support
- **Target OS:** Strictly Windows 10 and Windows 11 (64-bit).
- **Rationale:** Cross-platform support is explicitly out of scope for the FYP. Limiting the system to Windows aligns perfectly with the target demographic of independent repair shops, who overwhelmingly depend on Windows workstations for proprietary schematic/boardview software.
- **Hardware Targeting:** Simplifies driver stability by exclusively targeting the Intel OpenVINO driver stack for Windows CPUs/iGPUs.

### Update Strategy & Offline Capabilities
- **100% Offline Distribution:** The product operates entirely without network dependencies. 
- **Prohibited Auto-Updaters:** Auto-update mechanisms are explicitly out of scope. Integrating cloud hosting, recurring network requests, or remote manifest checks injects severe scope creep and violates the core offline-only constraint.
- **Version Control:** Future YOLO model weight improvements or interface patches will merely be distributed asynchronously as fresh standalone installers.

### System Integration & Packaging
- **Dependency Isolation:** The entire application stack (Python runtime environment, heavy OpenCV binaries, Intel OpenVINO inference dependencies, and the UI frontend) will be bundled entirely using PyInstaller.
- **Deployment UX:** The final technical deliverable will be a single `.exe` file, enabling rapid drag-and-drop deployment on the shop floor without requiring technicians or evaluators to manually install Python or configure runtime environments.

## Functional Requirements

### Video Acquisition & Preprocessing
- **FR1:** The System can ingest a live video stream from standard locally-connected UVC cameras.
- **FR2:** The Technician can select the active video source from a list of available UVC devices via the user interface.
- **FR3:** The System can apply dynamic contrast enhancement (e.g., CLAHE) to incoming video frames.
- **FR4:** The System can apply shadow-lifting algorithms (e.g., Gamma Correction) to incoming video frames.
- **FR5:** The System can calculate image sharpness and intentionally drop blurry frames from the processing queue based on variance thresholds.

### Inference & Defect Detection
- **FR6:** The System can execute object detection inference using INT8 quantized model weights.
- **FR7:** The System can identify and locate solder bridges within a video frame.
- **FR8:** The System can identify and locate missing electronic components within a video frame.
- **FR9:** The System can identify and locate misaligned electronic components within a video frame.
- **FR10:** The System can identify and locate damaged traces or burnt areas within a video frame.
- **FR11:** The System can calculate a numerical confidence score for every detected defect.

### Live Diagnostic Interface
- **FR12:** The Technician can view the live, pre-processed video feed in real-time.
- **FR13:** The Technician can observe visual bounding boxes overlaid precisely on detected defects within the live feed.
- **FR14:** The Technician can observe the corresponding confidence score rendered alongside each bounding box.
- **FR15:** The Technician can observe a distinct visual warning indicator if global inference confidence drops below a defined safety threshold.

### System Configuration
- **FR16:** The Technician can manually adjust the contrast enhancement parameters via the user interface.
- **FR17:** The Technician can manually adjust the shadow-lifting parameters via the user interface.
- **FR18:** The Technician can manually adjust the blur-dropping sensitivity threshold via the user interface.
- **FR19:** The Technician can preview the immediate effects of environment parameter changes directly on the live video feed.

## Non-Functional Requirements

### Performance & Architecture
- **NFR1 (Preprocessing Latency):** The complete OpenCV image preprocessing pipeline (CLAHE, Gamma Correction, Variance calculation) must execute in **under 5 milliseconds** per frame on a benchmark Intel Core i5 (8th-gen or equivalent) processor.
- **NFR2 (Inference Latency):** The OpenVINO YOLOv12 INT8 inference engine must return bounding boxes and confidence scores in **under 10 seconds** when processing a static, high-resolution (1080p) frame, and operate fluidly on scaled continuous streams.
- **NFR3 (Live Feed Fluidity):** The live video UI must reliably render at a minimum of **15 Frames Per Second (FPS)** to prevent visual stuttering while the technician physically manipulates the motherboard under the camera.
- **NFR4 (Queue Resilience):** The system must automatically trigger the Laplacian Variance frame-dropping logic if the processing queue depth exceeds an accumulated latency of **150ms**, instantly shedding load to protect CPU cycles.
- **NFR5 (Thread Decoupling):** The main GUI thread (rendering the live video feed and UI controls) must be completely isolated from the OpenVINO inference and OpenCV preprocessing loops using asynchronous background workers (e.g., PyQt/PySide QThread or equivalent). The application UI must remain 100% responsive and never freeze or hang, even if an inference latency spike occurs.

### Environmental Reliability (Hardware & Deployment)
- **NFR6 (Thermal Endurance):** The application must be capable of running actively for a standard **8-hour repair shift** without causing the host laptop to thermally throttle below its CPU base clock speed.
- **NFR7 (Lighting Resilience):** The OpenVINO model + OpenCV preprocessing stack must mathematically maintain **>90% Mean Average Precision (mAP)** against the test dataset even when ambient lighting intensity is artificially varied by ±30%.
- **NFR8 (Deployment Size):** The final standalone PyInstaller executable (which bundles the Python runtime, OpenCV libraries, OpenVINO inference dependencies, and model weights) must not exceed a total file size of **2GB** to ensure rapid deployment via standard USB thumb drives across shop floors.
