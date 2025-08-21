# The Latest Daily Papers - Date: 2025-08-21
## Highlight Papers
### **[WISE-FUSE: Efficient Whole Slide Image Encoding via Coarse-to-Fine Patch Selection with VLM and LLM Knowledge Fusion](http://arxiv.org/abs/2508.14537v1)**
- **Summary**: Here's a summary and rigorous critique of the paper "WISE-FUSE: Efficient Whole Slide Image Encoding via Coarse-to-Fine Patch Selection with VLM and LLM Knowledge Fusion."

**Summary:**

The paper introduces WISE-FUSE, a novel framework designed to efficiently encode whole slide images (WSIs) in computational pathology. The core idea is to leverage vision-language models (VLMs) and large language models (LLMs) to selectively process diagnostically relevant regions, thereby significantly reducing computational costs. WISE-FUSE employs a coarse-to-fine patch selection strategy: it first computes similarity scores between low-resolution patches and class-specific textual descriptions using knowledge distillation via Cross-Scale Visual Prompt Tuning (CS-VPT), preserving fine-grained features.  Based on these scores, informative regions are selected, eliminating irrelevant patches. High-resolution patches corresponding to the selected regions are then encoded and fused with textual embeddings derived from LLMs to reinforce diagnostic context.  Experiments demonstrate that WISE-FUSE reduces WSI encoding time considerably while maintaining or improving diagnostic performance.

**Critical Evaluation:**

*   **Novelty:** The paper introduces several novel aspects:
    *   **Coarse-to-Fine Selection with VLM/LLM:** Using VLMs and LLMs for guiding patch selection in WSIs in a coarse-to-fine manner is a good idea.
    *   **Cross-Scale Visual Prompt Tuning (CS-VPT):**  Distilling knowledge from high-resolution patches into low-resolution ones using VPT is a useful technique. This helps preserve fine-grained information for efficient selection.
    *   **Knowledge Fusion with LLM:** Integrating LLM-derived morphological cues to enrich patch embeddings is a valuable contribution, particularly for compensating for information loss during patch selection.
*   **Significance:**
    *   **Efficiency:** The paper convincingly demonstrates a significant reduction in processing time compared to exhaustive patch processing methods. This is a crucial factor for practical deployment in clinical settings.
    *   **Performance:**  The fact that WISE-FUSE achieves performance comparable to or surpassing full-resolution baselines is significant.  It shows that selective processing, guided by VLMs and LLMs, can be highly effective.
    *   **Scalability:** The proposed framework is model-agnostic and shows the generalizability of the approach on several datasets.
*   **Strengths:**
    *   The paper clearly articulates the problem of computational burden in WSI analysis.
    *   The WISE-FUSE framework is well-designed and explained, with clear motivations for each component (CS-VPT, knowledge fusion).
    *   The experimental results are comprehensive, covering multiple datasets, tasks, and vision-language backbones.  The ablation study provides valuable insights into the contribution of each component.
    *   The writing is clear and well-organized.
*   **Weaknesses:**
    *   The reliance on manually generated morphological descriptions, while insightful, introduces a potential dependency on human expertise.
    *   The framework is evaluated primarily on classification and survival prediction tasks. More diverse tasks could further demonstrate the generalizability of the approach.
    *   Zero-shot analysis in BACH dataset yields selection of normal tissue patches. Fine-tuning of existing VLMs could further improve the performance.

*   **Potential Influence:**

    *   This work could influence future research towards more efficient and scalable WSI analysis methods. The combined use of VLMs and LLMs for intelligent patch selection and encoding can be a generalizable strategy.
    *   The CS-VPT technique can be adapted for other tasks where knowledge distillation across scales is beneficial.
    *   The framework can be extended to incorporate other modalities of information, such as genomic data, further improving diagnostic accuracy.

**Overall Assessment:**

WISE-FUSE addresses a critical challenge in computational pathology—the efficient processing of gigapixel WSIs—by intelligently leveraging VLMs and LLMs. The method shows strong promise, with significant reductions in processing time and good diagnostic performance. The limitations, such as the reliance on morphological descriptions, do not overshadow the novelty and potential impact of the framework.

**Score: 8.5**

The high score reflects the paper's strong novelty, significance, and experimental validation. While there are some limitations related to the automation of description generation and task diversity, the paper presents a well-designed and thoroughly evaluated framework with significant potential to advance the field of computational pathology.

- **Score**: 8/10

### **[LeanGeo: Formalizing Competitional Geometry problems in Lean](http://arxiv.org/abs/2508.14644v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "LeanGeo: Formalizing Competitional Geometry Problems in Lean":

**Summary:**

The paper introduces LeanGeo, a formal system for formalizing and solving competition-level geometry problems within the Lean 4 theorem prover.  LeanGeo aims to address the gap in existing geometry solving systems by providing a unified framework that can integrate with other mathematical fields in Mathlib. It features a comprehensive library of high-level geometric theorems within Lean's foundational logic, enabling rigorous proof verification.  The authors also present LeanGeo-Bench, a formal geometry benchmark, comprising problems from the International Mathematical Olympiad (IMO) and other advanced sources, to evaluate the performance of Large Language Models (LLMs) on geometric reasoning. They provide baseline results using several LLMs and discuss the limitations and opportunities for future work.

**Critical Evaluation:**

*   **Novelty:** The paper's novelty lies in several aspects:
    *   **Formalization within Lean:**  The creation of a comprehensive geometry theorem library specifically within the Lean 4 environment is a significant contribution. While LeanEuclid exists, LeanGeo addresses competition-level problems and integrates better with Mathlib, which opens doors for proofs involving other mathematical domains.
    *   **LeanGeo-Bench:** The construction of a formalized geometry benchmark in Lean is valuable. This allows for standardized and rigorous evaluation of automated theorem provers and LLMs in a challenging domain.
    *   **Comprehensive Scope:**  The theorem library aims to cover a broad range of geometric topics, from foundational to IMO level.

*   **Significance:** The significance of the paper is multifold:
    *   **Bridge between Symbolic and Neural Reasoning:**  By creating a formal environment for geometry problems, the paper facilitates the development of neuro-symbolic reasoning systems that can leverage both the deductive capabilities of theorem provers and the pattern recognition abilities of neural networks.
    *   **Rigorous Verification:** Formalization ensures that geometry proofs are logically sound and complete, which is a major advantage over systems that rely on diagrammatic reasoning or unordered formal systems.
    *   **Benchmark for AI Reasoning:**  LeanGeo-Bench provides a rigorous testbed for evaluating AI models' mathematical reasoning abilities, specifically in geometry. This helps to identify the strengths and weaknesses of existing models and to guide future research.
    *   **Potential for Education:**  A formal system for geometry can be used as an educational tool to help students learn and understand geometric concepts and proofs.

*   **Strengths:**
    *   Clear problem definition and motivation.
    *   Well-structured Lean library with extensive definitions and theorems.
    *   Creation of a challenging and comprehensive benchmark.
    *   Experimentation with state-of-the-art LLMs on the benchmark.
    *   Opensourcing of the theorem library and benchmark.
    *   Demonstration of how LeanGeo can integrate with Mathlib.

*   **Weaknesses:**
    *   **Limited LLM Performance:** The baseline results for LLMs on LeanGeo-Bench are relatively low, indicating that there is still significant room for improvement in automated geometric reasoning. This also means the benchmark, while useful, is probably still too hard for existing LLMs to make massive progress on without significant architectural changes.
    *   **Reliance on External Solver for Soundness:** The dependency on CVC5 for proof certificates could be a source of potential unsoundness. An important direction for future work is to provide native Lean proofs and ensure end-to-end soundness.
    *   **Scalability Challenges:**  The reliance on general-purpose SMT solvers can be a bottleneck for complex problems. Integrating domain-specific proof automation techniques would improve the system's scalability.
    *   **Limited RL Experiments:** While RL experiments are promising, they appear to be preliminary. More detailed results and analysis of RL performance would be beneficial.

*   **Impact and Influence:** The paper has the potential to significantly influence the field of automated theorem proving and neuro-symbolic reasoning by:
    *   Promoting the development of more sophisticated AI models for geometric reasoning.
    *   Encouraging the use of formal methods for verifying mathematical proofs.
    *   Providing a valuable resource for researchers and educators interested in geometry and formal mathematics.

**Justification for Score:**

Considering the strengths and weaknesses, the novelty of LeanGeo, and its potential significance, I would assign a score of **8**. The creation of a formal geometry system within Lean, coupled with a dedicated benchmark, represents a valuable contribution that will likely spur further research. The primary limitation lies in the relatively low performance of current LLMs on the benchmark, but this also highlights the importance of the problem and the need for further advancements. The reliance on external solvers also detracts from the overall robustness of the system. The paper is certainly significant and valuable, and I would expect to see further improvements to the system and its downstream applications in the future.

Score: 8

- **Score**: 8/10

### **[MCP-Universe: Benchmarking Large Language Models with Real-World Model Context Protocol Servers](http://arxiv.org/abs/2508.14704v1)**
- **Summary**: Okay, I'll provide a concise summary and a critical evaluation, including a novelty/significance score with a rigorous rationale, as requested.

**Paper Summary**

The paper introduces MCP-Universe, a novel benchmark designed to evaluate Large Language Models (LLMs) interacting with real-world Model Context Protocol (MCP) servers.  MCP is a standard for connecting LLMs to external data sources and tools, but existing benchmarks are considered simplistic and don't adequately capture real-world challenges like long-horizon reasoning and large, unfamiliar tool spaces. MCP-Universe addresses this gap by providing a comprehensive benchmark with 6 core domains spanning 11 different real-world MCP servers, including Location Navigation, Repository Management, Financial Analysis, 3D Design, Browser Automation, and Web Searching. The benchmark incorporates execution-based evaluators to ensure rigorous evaluation, covering format compliance, content matching, and dynamic ground truth retrieval for temporally sensitive tasks. Experiments with leading LLMs like GPT-5, Grok-4, and Claude-4.0-Sonnet reveal significant performance limitations, highlighting long-context and unknown-tools challenges.  The benchmark is open-sourced with an extensible evaluation framework and UI support.

**Critical Evaluation**

*   **Novelty:** The paper's primary strength lies in its focus on MCP interaction.  While existing benchmarks address LLMs and tool use, MCP-Universe is specifically designed to evaluate LLMs in the context of this emerging standard. The use of *real-world* MCP servers is a significant improvement over simulated environments or simplistic datasets. The combination of 6 diverse domains (11 servers) is also a strength, providing a broader coverage of real-world applications than prior work. The introduction of dynamic evaluators that consider time-sensitive tasks is also a valuable addition.

*   **Significance:**  The benchmark's significance is high because MCP represents a potential paradigm shift in how LLMs interact with the world. By identifying the limitations of current LLMs in MCP environments, this paper provides valuable insights for future research directions. Uncovering challenges related to long context, unknown tools, and cross-domain inconsistencies is crucial for designing more effective and robust agent systems.  The open-source nature of the benchmark and the extensible framework will facilitate further research and development in this area.

*   **Strengths:**
    *   Real-world MCP servers
    *   Diverse domains and tasks
    *   Execution-based evaluators, including dynamic evaluators
    *   Open-source and extensible framework

*   **Weaknesses:**
    *   The paper could benefit from a more in-depth analysis of the error modes of different LLMs. What *specific* types of errors are observed in each domain and with each model?
    *   While the benchmark covers a range of domains, there may be other relevant domains that could be included in future work (e.g., scientific computing, content creation).
    *   The paper mentions the "style bias" of LLM judges, but could elaborate on the specific steps taken to mitigate biases in the execution-based evaluators.

*   **Potential Influence:** This paper has the potential to become a standard benchmark for evaluating LLMs in MCP environments. It will likely influence the design of new LLM agents and MCP servers, fostering innovation in this rapidly evolving ecosystem. The framework could also be used to evaluate and compare different agent architectures (e.g., ReAct vs. more sophisticated planning-based agents).

**Score Justification:**

Considering the strengths and weaknesses, I assign a score of **8**. The novelty is good due to the focus on the important MCP standard and the use of real-world servers. The significance is high because the paper directly addresses a critical gap in LLM evaluation and identifies key challenges for future research. While the paper could benefit from more detailed error analysis, the overall contribution is substantial and will likely have a significant impact on the field.

**Score: 8**

- **Score**: 8/10

### **[ShizhenGPT: Towards Multimodal LLMs for Traditional Chinese Medicine](http://arxiv.org/abs/2508.14706v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "ShizhenGPT: Towards Multimodal LLMs for Traditional Chinese Medicine":

**Summary:**

The paper introduces ShizhenGPT, a multimodal Large Language Model (LLM) specifically tailored for Traditional Chinese Medicine (TCM). The authors address two key challenges in applying LLMs to TCM: the scarcity of high-quality TCM data and the inherent multimodality of TCM diagnostics, which involves observation, listening, smelling, and pulse-taking. To overcome data scarcity, they curate the largest TCM dataset to date, encompassing text, images, audio, and physiological signals. ShizhenGPT is pre-trained and instruction-tuned to achieve deep TCM knowledge and multimodal reasoning. The model is evaluated on recent TCM qualification exams and a newly built visual benchmark for medicinal recognition and visual diagnosis. Experimental results demonstrate that ShizhenGPT outperforms comparable-scale LLMs, competes with larger proprietary models, and achieves state-of-the-art performance in TCM visual understanding. The authors release the dataset, models, and code publicly.

**Critical Evaluation:**

**Novelty:**  The paper presents a novel contribution by creating the *first* multimodal LLM specifically designed for TCM.  While other TCM-specific LLMs exist, they are largely text-based and trained on smaller datasets. The incorporation of visual, auditory, and physiological signal data is a significant step forward. The creation of a new dataset and benchmark specifically for TCM visual tasks also increases novelty. The multimodal diagnostic capabilities, integrating the four diagnostic methods, are a notable strength.

**Significance:**  TCM is a globally relevant medical system serving hundreds of millions of people. An LLM that can understand and reason about TCM has the potential to significantly impact clinical decision-making, medical education, and preservation of traditional medical knowledge. Demonstrating that LLMs can effectively learn from and reason with diverse sensory modalities is also of broader significance for the field of AI.

**Strengths:**

*   **Comprehensive Dataset:**  The curated TCM dataset is a major strength.  The sheer size and diversity of the data (text, images, audio, and physiological signals) address a key bottleneck in TCM-related AI research.
*   **Multimodal Integration:** The model effectively integrates data from various modalities, demonstrating unified perception across sensory modalities, moving towards a more holistic approach to TCM diagnosis.
*   **Strong Experimental Results:**  ShizhenGPT outperforms existing LLMs of comparable scale on TCM exams and visual understanding tasks. The results suggest a significant improvement in TCM-specific knowledge and reasoning capabilities. Competing against much larger and proprietary models is impressive.
*   **Public Release:**  The public release of the dataset, models, and code makes this work highly valuable to the research community, enabling further exploration and development in this area.
* **Strong Benchmarking:** The custom benchmarks specifically built for TCM, across visual, textual and signal domains ensures a more targeted evaluation of the LLM's specific capabilities.

**Weaknesses:**

*   **Limited Signal Data:** The authors acknowledge that high-quality signal data (smell, pulse) is limited. This likely constrains the performance in those modalities.
*   **Lack of Clinical Validation:** The lack of real-world clinical testing is a significant limitation. While expert evaluations are valuable, they cannot fully replicate the complexities of a clinical setting. This is a common limitation of research in this area, but it needs to be addressed in future work.
*   **Incomplete Modal Coverage:** Important modalities like tactile sensation, are missing, which limits the scope of TCM reasoning that can be supported.
*   **Reliance on Existing LLM Architectures:** The paper leverages existing LLM architectures (Qwen). While this is understandable, the innovation in the model architecture is somewhat limited.

**Potential Influence:**

This work has the potential to inspire further research in applying multimodal LLMs to specialized domains. The successful integration of diverse sensory modalities and the creation of TCM-specific datasets and benchmarks provide a valuable foundation for future studies.

**Justification of Score:**

This paper demonstrates a significant advancement by developing the first multimodal LLM tailored for TCM, successfully addressing data scarcity and integrating diverse sensory modalities. While the lack of clinical validation and limitations in signal data collection hold it back, the comprehensive dataset, strong experimental results, and public release of resources make this a valuable and influential contribution to the field. It bridges the gap between advanced AI techniques and traditional medical systems, opening the door for future research and potential applications in TCM. The novel benchmark datasets and the model itself contribute significantly to pushing the boundaries of what LLMs can do in specialized domains.

Score: 8

- **Score**: 8/10

### **[TransLLM: A Unified Multi-Task Foundation Framework for Urban Transportation via Learnable Prompting](http://arxiv.org/abs/2508.14782v1)**
- **Summary**: Here's a summary and critical evaluation of the TransLLM paper:

**Summary:**

The paper introduces TransLLM, a unified foundation framework designed to address the challenges of multiple urban transportation tasks like traffic forecasting, EV charging demand prediction, and taxi dispatch. It overcomes the limitations of existing task-specific models (data-hungry and lacking generalization) and standard LLMs (struggling with spatiotemporal data and numerical reasoning). TransLLM achieves this by combining a lightweight spatiotemporal encoder (dilated temporal convolutions and dual-adjacency graph attention networks) with a large language model (LLM) through learnable prompt composition. A key component is the instance-level prompt routing mechanism, trained using reinforcement learning, that adapts prompts to specific input characteristics, moving beyond fixed task templates. The framework encodes spatiotemporal patterns, dynamically composes personalized prompts, and projects the resulting representations to task-specific output layers for prediction. Experiments across seven datasets and three tasks demonstrate TransLLM's effectiveness in both supervised and zero-shot settings.

**Critical Evaluation:**

* **Novelty:**  The primary novelty of the paper lies in the integrated approach combining spatiotemporal encoders, a learnable, instance-specific prompt routing mechanism, and LLMs within a *unified* urban transportation framework. Previous work like UrbanGPT focused more on direct LLM integration with temporal encoders, while TransLLM adds the crucial element of *dynamic* prompt generation tailored to individual instances. The use of reinforcement learning for prompt personalization is also a strong point of novelty.  The combination of dilated convolutions and dual-adjacency graph attention networks, although individually known, is novel in *this particular application* and integrated system.

* **Significance:**  The significance stems from addressing a practical challenge: creating a generalizable model for diverse urban transportation problems.  Existing methods often require task-specific engineering and large amounts of labeled data. TransLLM offers a way to leverage the power of LLMs with significantly less task-specific tuning and potentially better generalization. The results show improvements over strong baselines, and the zero-shot capabilities are particularly promising.  The exploration of how different LLMs (Vicuna, LLaMA3) can be integrated is also useful.  The taxi dispatch application further expands beyond basic forecasting demonstrating the framework's versatility.

* **Strengths:**
    * **Unified Framework:** The paper provides a general framework instead of a task-specific architecture. This promotes knowledge transfer and reduces development effort for new transportation applications.
    * **Instance-Level Adaptability:**  The reinforcement learning-based prompt routing provides an adaptive mechanism. This addresses a major limitation of static task-specific prompts.
    * **Comprehensive Evaluation:** The paper includes extensive experiments on several datasets and tasks, comparing against strong baselines in both supervised and zero-shot settings. The ablation studies clearly illustrate the impact of each module.
    * **Strong Results:** TransLLM consistently outperforms other methods.
    * **Well-written and organized:** The paper is clear, concise, and easy to follow.

* **Weaknesses:**
    * **Computational Cost:** While LoRA is used, training and deploying LLMs remain computationally expensive, especially compared to smaller, task-specific models. The paper doesn't offer detailed analysis of training time or inference cost.
    * **Reliance on LLM Quality:** The framework's performance is inherently tied to the capabilities of the underlying LLM. The paper acknowledges the dependency on larger pretrained models and does not explore methods for overcoming any LLM's inherent limitations.
    * **Limited LLM Exploration:** Only a few LLMs (Vicuna, LLaMA3) are explored. Further investigation into other LLMs and how they interact with the rest of the architecture could enhance the research.  The exploration of generalist LLMs (GPT-40) is somewhat superficial.
    * **Black Box Nature:**  Despite the "interpretability" claims, understanding *why* the RL-based prompt router selects certain prompts for specific instances remains a challenge.  More analysis of the learned prompt policies could be beneficial.
    * **Dataset Scale for Training:** The paper mentions using a reduced training subset for TransLLM compared to smaller models, raising questions regarding scaling data and compute.

* **Potential Influence:**  This work has the potential to significantly influence the development of more generalizable and adaptable urban transportation models. It opens new avenues for leveraging LLMs and transfer learning to solve complex problems in smart cities. The framework can be extended to other related domains like logistics, supply chain management, and resource allocation.

**Justification for Score:**

TransLLM presents a solid contribution to the field by effectively combining multiple techniques to tackle the challenge of building a unified and generalizable model for urban transportation. The novelty lies in the RL-based prompt adaptation and the holistic framework. While there are weaknesses regarding computational cost and the inherent reliance on LLMs, the comprehensive evaluation and promising results justify a score reflecting significant advancement.  The results are also compelling, showing notable improvements over strong benchmarks.

Score: 8

- **Score**: 8/10

### **[TransLight: Image-Guided Customized Lighting Control with Generative Decoupling](http://arxiv.org/abs/2508.14814v1)**
- **Summary**: Here's a summary and critical evaluation of the provided paper:

**Summary:**

The paper introduces TransLight, a novel framework for transferring light effects from a reference image to a target image with a high degree of fidelity and user control. The key innovation is a "Generative Decoupling" approach that uses two fine-tuned diffusion models to separate image content from light effects. This allows the light effect to be extracted from the reference image and composited onto the target image. TransLight enables users to flexibly control the position, direction, and intensity of the transferred light effects.  The method involves generating a large-scale dataset of image-content-light triplets to train the model. Experimental results demonstrate the effectiveness of TransLight in achieving realistic and customized light effect transfers.

**Critical Evaluation:**

*   **Novelty:** The paper's primary novelty lies in its ability to disentangle light effects from content in real-world images with sufficient fidelity to enable transfer. While style transfer and image relighting are established areas, the specific problem of transferring *complex* light effects (like lens flares, shafts of sunlight) while preserving the original content has not been adequately addressed. The use of generative decoupling to extract and then re-composite light effects is a significant contribution.  The million-scale dataset of content-light triplets also appears to be a novel contribution that could potentially be useful for other tasks in illumination editing and analysis.

*   **Significance:** The significance stems from the potential to provide users with unprecedented control over illumination in images. Current methods either lack fine-grained control or struggle to maintain content integrity during editing.  TransLight's ability to transfer complex light effects from a reference image, while also allowing for geometric transformations, opens up new possibilities for lighting stylization and realism enhancement. The performance demonstrated and the new benchmark set are important for future research. The approach could be applied to photo editing tools, virtual content creation, or image enhancement pipelines.

*   **Strengths:**
    *   The Generative Decoupling strategy is a clever and effective approach to tackling the difficult problem of separating light from content.
    *   The creation of a large-scale content-light triplet dataset is a substantial effort and a valuable resource for the community.
    *   The flexible control over the transferred light effects (position, direction, intensity) is a key strength, providing user customization that other methods often lack.
    *   The quantitative results, especially the Light FID score, along with the visual examples, convincingly demonstrate the superiority of TransLight.
    *   Comprehensive ablation studies thoroughly analyze different aspects of model design choices.

*   **Weaknesses:**
    *   The reliance on a proprietary dataset limits reproducibility. Although the creation process is detailed, the lack of access to the data will hinder other researchers' ability to directly compare against TransLight.
    *   The method's performance likely depends on the accuracy of the light extraction model. While the paper demonstrates good results, further analysis of failure cases and limitations would strengthen the evaluation. The paper mentions that subtle or indistinct light effects can pose challenges. A more detailed discussion of these limitations is required.
    *   While the paper compares to existing techniques in the related works, a more detailed comparison of the compute requirements of each technique is required.

*   **Potential Influence:** The paper has the potential to significantly influence research in illumination editing, style transfer, and generative image manipulation. It sets a new benchmark for transferring light effects and introduces a novel approach to disentangling light and content. The dataset, if made available, could also stimulate further research in this area. The idea of generative decoupling could be applied to other areas within visual processing.

*   **Justification for Score:** Given the novelty of the approach, the significance of the problem being addressed, the strength of the experimental results, and the potential impact on the field, a score of 8 is justified. While the lack of a publicly available dataset and the method's limitations are concerns, the overall contribution is substantial and represents a significant step forward in the field of image relighting and editing.

Score: 8

- **Score**: 8/10

## Other Papers
### **[Ouroboros: Single-step Diffusion Models for Cycle-consistent Forward and Inverse Rendering](http://arxiv.org/abs/2508.14461v1)**
### **[In2x at WMT25 Translation Task](http://arxiv.org/abs/2508.14472v1)**
### **[Reasoning is about giving reasons](http://arxiv.org/abs/2508.14488v1)**
### **[Semantic Energy: Detecting LLM Hallucination Beyond Entropy](http://arxiv.org/abs/2508.14496v1)**
### **[SATURN: Autoregressive Image Generation Guided by Scene Graphs](http://arxiv.org/abs/2508.14502v1)**
### **[Preguss: It Analyzes, It Specifies, It Verifies](http://arxiv.org/abs/2508.14532v1)**
### **[WISE-FUSE: Efficient Whole Slide Image Encoding via Coarse-to-Fine Patch Selection with VLM and LLM Knowledge Fusion](http://arxiv.org/abs/2508.14537v1)**
### **[Towards LLM-generated explanations for Component-based Knowledge Graph Question Answering Systems](http://arxiv.org/abs/2508.14553v1)**
### **[Who Sees What? Structured Thought-Action Sequences for Epistemic Reasoning in LLMs](http://arxiv.org/abs/2508.14564v1)**
### **[Towards Skeletal and Signer Noise Reduction in Sign Language Production via Quaternion-Based Pose Encoding and Contrastive Learning](http://arxiv.org/abs/2508.14574v1)**
### **[FakeHunter: Multimodal Step-by-Step Reasoning for Explainable Video Forensics](http://arxiv.org/abs/2508.14581v1)**
### **[Can LLM Agents Solve Collaborative Tasks? A Study on Urgency-Aware Planning and Coordination](http://arxiv.org/abs/2508.14635v1)**
### **[LeanGeo: Formalizing Competitional Geometry problems in Lean](http://arxiv.org/abs/2508.14644v1)**
### **[Virtual Multiplex Staining for Histological Images using a Marker-wise Conditioned Diffusion Model](http://arxiv.org/abs/2508.14681v1)**
### **[Improving in-context learning with a better scoring function](http://arxiv.org/abs/2508.14685v1)**
### **[MCP-Universe: Benchmarking Large Language Models with Real-World Model Context Protocol Servers](http://arxiv.org/abs/2508.14704v1)**
### **[ShizhenGPT: Towards Multimodal LLMs for Traditional Chinese Medicine](http://arxiv.org/abs/2508.14706v1)**
### **[GSFix3D: Diffusion-Guided Repair of Novel Views in Gaussian Splatting](http://arxiv.org/abs/2508.14717v1)**
### **[Transplant Then Regenerate: A New Paradigm for Text Data Augmentation](http://arxiv.org/abs/2508.14723v1)**
### **[Assessing the Quality and Security of AI-Generated Code: A Quantitative Analysis](http://arxiv.org/abs/2508.14727v1)**
### **[Multiscale Video Transformers for Class Agnostic Segmentation in Autonomous Driving](http://arxiv.org/abs/2508.14729v1)**
### **[Evaluating Multilingual and Code-Switched Alignment in LLMs via Synthetic Natural Language Inference](http://arxiv.org/abs/2508.14735v1)**
### **[MissionHD: Data-Driven Refinement of Reasoning Graph Structure through Hyperdimensional Causal Path Encoding and Decoding](http://arxiv.org/abs/2508.14746v1)**
### **[Cross-Modality Controlled Molecule Generation with Diffusion Language Model](http://arxiv.org/abs/2508.14748v1)**
### **[PepThink-R1: LLM for Interpretable Cyclic Peptide Optimization with CoT SFT and Reinforcement Learning](http://arxiv.org/abs/2508.14765v1)**
### **[TransLLM: A Unified Multi-Task Foundation Framework for Urban Transportation via Learnable Prompting](http://arxiv.org/abs/2508.14782v1)**
### **[Tinker: Diffusion's Gift to 3D--Multi-View Consistent Editing From Sparse Inputs without Per-Scene Optimization](http://arxiv.org/abs/2508.14811v1)**
### **[TransLight: Image-Guided Customized Lighting Control with Generative Decoupling](http://arxiv.org/abs/2508.14814v1)**
### **[Evaluating Retrieval-Augmented Generation vs. Long-Context Input for Clinical Reasoning over EHRs](http://arxiv.org/abs/2508.14817v1)**
### **[Long Chain-of-Thought Reasoning Across Languages](http://arxiv.org/abs/2508.14828v1)**
### **[Universal and Transferable Adversarial Attack on Large Language Models Using Exponentiated Gradient Descent](http://arxiv.org/abs/2508.14853v1)**
### **[The Prompting Brain: Neurocognitive Markers of Expertise in Guiding Large Language Models](http://arxiv.org/abs/2508.14869v1)**
### **[Squeezed Diffusion Models](http://arxiv.org/abs/2508.14871v1)**
### **[Quantization Meets dLLMs: A Systematic Study of Post-training Quantization for Diffusion LLMs](http://arxiv.org/abs/2508.14896v1)**
