# The Latest Daily Papers - Date: 2025-05-12
## Highlight Papers
### **[Diffusion Model Quantization: A Review](http://arxiv.org/abs/2505.05215v1)**
- **Summary**: Okay, I'll provide a summary and a critical evaluation with a justified score for the paper "Diffusion Model Quantization: A Review."

**Summary:**

This paper presents a comprehensive review of recent advancements in diffusion model quantization, a crucial technique for efficiently deploying these computationally intensive generative models on resource-constrained devices. The authors categorize and analyze various quantization methods tailored for diffusion models, including those based on U-Net and Diffusion Transformer (DiT) architectures.  The review covers both post-training quantization (PTQ) and quantization-aware training (QAT) techniques, examining strategies like calibration sampling, dynamic activation, and error correction. A significant portion of the work involves benchmarking open-source solutions across different image generation tasks and offering qualitative analyses of quantization artifacts such as color bias and blurring. Finally, the authors outline future research directions for the quantization of generative models in practical applications. They provide extensive resources on a linked survey project webpage, including code and pre-trained model information.

**Critical Evaluation:**

*   **Novelty:** The primary contribution of this paper is as a *survey*. Surveys, by nature, do not present fundamentally *new* methods or theoretical results. However, their novelty lies in their *synthesis* of existing knowledge, *identification* of key trends, and *framing* of future research directions. This paper is, according to the authors, the first dedicated survey on diffusion model quantization, a rapidly evolving field.  The taxonomy presented is relatively well thought out and divides into relevant categories, like U-Net and transformer approaches, PTQ and QAT, and a categorization of solutions to the specific problems of the architectures. The compilation of papers, codes, pre-trained models, and comparison results on a survey website is valuable.

*   **Significance:** Diffusion models are at the forefront of generative AI. Quantizing them is essential for making them practical on edge devices and in other resource-limited scenarios. Therefore, a comprehensive survey is highly relevant and important. The paper brings together fragmented research, allowing researchers to grasp the landscape of existing quantization techniques for diffusion models quickly. By highlighting the key challenges and comparing different approaches, it will enable researchers to build upon the work. The benchmark and qualitative studies offer a practical perspective and provide insights into the trade-offs between different quantization methods.

*   **Strengths:**
    *   **Comprehensiveness:** The survey appears to cover a wide range of relevant papers, including the most recent advancements.
    *   **Clear Taxonomy:** The structured categorization of quantization techniques makes the review easy to navigate.
    *   **Practical Evaluation:** The benchmark experiments and qualitative analysis provide valuable insights into the performance and limitations of different methods.
    *   **Future Directions:** Identifying promising areas for future research helps to guide the field.
    *   **Resource provision:** Creating and making available the links to codes and models is very useful for other researches.

*   **Weaknesses:**
    *   **Limited Theoretical Depth in Novel Results:** As a survey, it doesn't present novel theoretical insights or introduce fundamentally new quantization techniques. While the organization and categorization are useful, they are somewhat derivative from previous work in quantization.

* **Justification of Score:**
A paper's main strength is how much future researchers are helped and influenced to move to the next advancement. Here, having all the papers, codes and pre-trained model ready for research is extremely helpful. Given that the diffusion model is a hot area, helping research will cause ripple effects. The paper achieves high level of comprehensiveness and clear organization. The limitations are mainly due to the inherent nature of review papers, but the authors do an excellent job of fulfilling the core purposes. The score reflects the paper's value in consolidating the field of diffusion model quantization and guiding further research.
Score: 8

- **Score**: 8/10

### **[TokLIP: Marry Visual Tokens to CLIP for Multimodal Comprehension and Generation](http://arxiv.org/abs/2505.05422v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "TokLIP: Marry Visual Tokens to CLIP for Multimodal Comprehension and Generation":

**Summary:**

The paper introduces TokLIP, a novel visual tokenizer designed to enhance multimodal comprehension and generation within a unified autoregressive framework. The core idea is to semanticize vector-quantized (VQ) tokens by integrating them with CLIP-level semantics. TokLIP uses a low-level discrete VQ tokenizer followed by a ViT-based token encoder to capture high-level continuous semantics. This approach disentangles the training objectives for comprehension and generation, allowing the use of advanced VQ tokenizers without task-specific quantization operations.  Experiments demonstrate improved data efficiency and performance on image representation, multimodal comprehension, and image generation tasks compared to existing approaches like VILA-U, QLIP and models like Emu3, while achieving comparable results to SynerGen-VL using significantly less training data.

**Critical Evaluation:**

*   **Novelty:** The paper's core novelty lies in its strategy of "semanticizing" low-level VQ tokens rather than discretizing high-level features (as done by VILA-U). While the individual components (VQ-VAE, ViT, CLIP) are not new, the specific combination and training methodology (CLIP distillation and contrastive loss on VQ tokens) to inject semantics into VQ representations represents a novel contribution. The idea of disentangling comprehension and generation through this specific architecture and training regime is a significant advance.

*   **Significance:**  The paper addresses a critical bottleneck in unified multimodal models: the lack of high-level semantics in standard VQ tokens, which hinders comprehension performance. TokLIP tackles this problem effectively, enabling end-to-end autoregressive training with standard VQ tokens. The improved data efficiency is a particularly significant benefit, potentially democratizing research in this computationally expensive area. The experimental results showcasing superior or comparable performance with less data suggest that TokLIP can serve as a good foundation for training effective visual representation with the power of CLIP.

*   **Strengths:**
    *   Clear problem definition and motivation.
    *   Well-explained architecture and training methodology.
    *   Comprehensive experimental evaluation across multiple tasks.
    *   Demonstrated data efficiency compared to previous approaches.
    *   The disentanglement of comprehension and generation training objectives is a key architectural decision.

*   **Weaknesses:**
    *   While the authors show comparison with recent methods, a few points of comparison could be strengthen. The paper compared the generation capability with a few baselines, but an ablation of the low-level feature fusion (removing VQGAN features in Table 7) may further justify the efficacy of the proposed token semanticization.
    *   While the paper highlights the flexibility of TokLIP to incorporate advanced tokenizers and CLIP models, the extent of performance gain with these advanced models is not empirically shown, leaving room for future work.
    *   The paper would benefit from more qualitative analysis of the generated samples.
    *   The experiments are primarily focused on image and language.  Exploring other modalities (e.g., audio, video) would further strengthen the generalizability of TokLIP.

*   **Potential Influence:** TokLIP has the potential to become a widely adopted visual tokenizer for unified multimodal models. Its data efficiency and ability to enhance comprehension open up new avenues for research and development in this field. The core concept of semanticizing low-level tokens could inspire other approaches for improving representation learning in multimodal settings. The study is also accessible to broader community as the code is available.

*Score: 8*

**Rationale:**

A score of 8 reflects the paper's strong novelty, significance, and experimental results. The core idea of semanticizing low-level VQ tokens is both novel and impactful, addressing a key limitation in existing multimodal models. The paper demonstrates significant data efficiency, making it a valuable contribution to the field. The comprehensive experiments across various tasks further support the effectiveness of TokLIP. The weaknesses, although present, don't fundamentally detract from the strength of the core contribution, and instead highlight future avenues for research.

- **Score**: 8/10

### **[Lost in OCR Translation? Vision-Based Approaches to Robust Document Retrieval](http://arxiv.org/abs/2505.05666v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper investigates the performance of vision-based Retrieval-Augmented Generation (RAG) versus OCR-based RAG pipelines for document retrieval under varying document quality conditions. The authors compare ColQwen2 (a vision-language model) against Llama 3.2 (90B) with Nougat OCR, and Llama 3.2 with the Qwen embedding model.  They introduce a new dataset, DocDeg, that features various levels of visual degradation. Beyond traditional retrieval metrics, the authors also introduce a semantic answer evaluation benchmark. The results suggest that OCR-based RAG, particularly when employing high-quality OCR like Llama 3.2, generalizes better to unseen documents, while VLM-based RAG excels in computational efficiency. They ultimately suggest trading computational efficiency over semantic accuracy.

**Critical Evaluation:**

*   **Novelty:**

    *   The paper offers a valuable comparative analysis of VLM-based RAG (specifically ColPali) and a modern OCR-based RAG pipeline.  The explicit focus on comparing against a highly capable OCR system like Llama 3.2 is a key strength, as many previous studies relied on outdated baselines.
    *   The creation of the DocDeg dataset addresses a gap in existing benchmarks, which often focus on clean documents.  The annotation of degradation levels is a meaningful addition.
    *   The introduction of a semantic answer evaluation benchmark adds to the rigor of the study, moving beyond simple retrieval accuracy to assess the downstream performance. This is a major strength.
    *   While the core concept of comparing OCR and VLM-based approaches isn't entirely novel, the comprehensive experimental design, the new dataset, and semantic evaluation do represent a significant advancement.

*   **Significance:**

    *   The paper's findings provide practical guidance for RAG practitioners, highlighting the trade-offs between computational efficiency and semantic accuracy. This guidance is particularly relevant in production environments where document quality varies widely.
    *   The demonstration that OCR-based RAG (with good OCR) can outperform VLM-based RAG in terms of semantic accuracy, even with larger models is significant. It challenges the assumption that VLMs are inherently more robust to visual noise.
    *   The study's analysis of computational efficiency, including embedding time, retrieval latency, and memory usage, is also highly valuable for practical deployment considerations.
    *   The focus on models readily deployable and comparable is significant in the sense that this paper addresses a realistic problem and offers realistic comparisons that are useful to RAG practitioners.

*   **Strengths:**

    *   Comprehensive experimental design with careful control over document quality levels.
    *   Use of a new, annotated dataset that addresses a critical gap in existing benchmarks.
    *   Introduction of a semantic answer evaluation benchmark that provides a more holistic assessment of RAG performance.
    *   Explicit consideration of computational efficiency, which is crucial for practical deployment.

*   **Weaknesses:**

    *   While the paper uses Llama 3.2 (90B) for OCR, it uses ColQwen2 (7B) which is a significantly smaller VLM to embed images. The authors explain that using Llama 3.2 to create patch embeddings and perform retrieval would create significant memory and latency issues. However, there is not enough explanation around this.
    *   The explanation around the need to add OCR to images retrieved by ColPali to make the evaluation possible is not robust enough.
    *   The qualitative analysis in the experiment is limited to a short statement around the question-answer pairs being of good quality. Further investigation could be done to confirm the results of the evaluation.

*   **Potential Influence:**

    *   This work could influence future research by encouraging the development of more robust evaluation methodologies for RAG systems, particularly those that consider document quality and semantic accuracy.
    *   The DocDeg dataset could become a valuable resource for the community, facilitating further research on document retrieval under realistic conditions.
    *   The paper's practical guidance could inform the design and deployment of RAG systems in production environments, leading to improved performance and usability.

**Overall:**

The paper provides a solid contribution to the field of document retrieval and RAG by providing a comparative analysis between a VLM-based approach and an OCR-based approach. The authors present strong data to confirm their findings. Overall, this paper presents a good argument for choosing OCR-based RAG with a high-quality OCR system in noisy or varying quality data.

**Score: 8**

- **Score**: 8/10

### **[InstanceGen: Image Generation with Instance-level Instructions](http://arxiv.org/abs/2505.05678v1)**
- **Summary**: Here's a summary and critical evaluation of the InstanceGen paper:

**Summary:**

The paper "InstanceGen: Image Generation with Instance-level Instructions" addresses the challenge of generating images from complex text prompts that specify multiple objects, their individual attributes, and their spatial relationships. The method combines image-based structural guidance from a pre-trained text-to-image model with instance-level instructions extracted using a Large Language Model (LLM).  The process involves: 1) generating an initial image, 2) segmenting the image into instances using attention maps and segmentation models, 3) using an LLM to assign object identities and attributes to each segment, and 4) refining the image using attention-based losses to enforce these assignments while maintaining visual quality. They also introduce a new benchmark, CompoundPrompts, to evaluate these capabilities, which is broken down into three levels of difficulty based on object counts, instance-level attributes, and spatial relations. The method is demonstrated to achieve state-of-the-art results, particularly on complex prompts.

**Critical Evaluation:**

*   **Novelty:** The core novelty lies in the *integration* of existing techniques (diffusion models, segmentation, LLMs) in a specific way to tackle a well-defined problem. While each component is not individually novel, the way they are combined to achieve instance-level control is a unique contribution. The idea of using the initial image generation not just as a starting point but to inform the *structure* of the layout before LLM assignment is clever. It leverages the inherent structural understanding of diffusion models. Also the self-correction mechanism in the assigning stage and a careful balance between attention guidance, and structure perseveration make it stand out.
*   **Significance:** The problem addressed is undeniably important. The ability to generate images that precisely adhere to complex, multi-object prompts is a critical step toward more controllable and useful image generation systems. This paper's success on complex prompt fidelity opens doors to more advanced applications requiring fine-grained image control. The CompoundPrompts dataset is also a valuable contribution, filling a gap in existing benchmarks for multi-object attribute control.
*   **Strengths:**
    *   Strong empirical results: The paper demonstrates significant improvements over existing methods on both the DrawBench and CompoundPrompts datasets. Qualitative results are also convincing.
    *   Well-defined approach: The steps are clearly outlined, and the rationale for each component is well-explained.
    *   Addresses a clear limitation: It tackles a specific failure mode of text-to-image models, which is precisely rendering complex multi-object scenes.
    *   Ablation study provides insight: It clearly showcases the contribution of each component in the model.
    *   The prompt engineering to make LLM produce assignments and the careful design for image generation are two key features of the paper.

*   **Weaknesses:**
    *   Dependency on LLM: The reliance on the LLM for instance assignment means the system's performance is directly tied to the capabilities and biases of the LLM. Errors in the LLM's reasoning propagate to the final image. Also, the cost of relying on LLM for each forward pass is high and might be a deterrent for large-scale use.
    *   Complexity: The method is relatively complex, involving several stages and components. This complexity can make it difficult to implement and optimize.
    *   Scalability: Although the paper is tested across many scenarios and datasets, it still lacks proof of generality. Also as prompts grow, this system might have challenges due to relying on LLM.
    *   Limited evaluation metrics: While VQA metrics capture some aspects of the prompt fidelity, it’s not a perfect measure. More human evaluations would strengthen the conclusions.
    *   Failure cases: The paper acknowledges limitations regarding depth perception in object placement (i.e. front vs. back) and certain limitations regarding combining different objects together for spatial layouts, and more complex counting of objects.
*   **Potential Influence:** The paper is likely to influence future research in controllable image generation.  The idea of leveraging an initial image generation's structure to inform layout is a promising direction. The CompoundPrompts dataset will serve as a valuable tool for evaluating and comparing future methods.

**Justification of Score:**

The paper demonstrates a novel and effective approach to a significant problem in text-to-image generation. It provides strong empirical results and a valuable new benchmark. While it relies on existing techniques and faces certain limitations, the creative integration of these techniques and the performance gains warrant a high score. This method addresses a key challenge, showcasing how a combination of structural guidance from diffusion models and instruction following from LLMs can improve prompt fidelity in complex scenarios.

Score: 8

- **Score**: 8/10

### **[From Bias To Improved Prompts: A Case Study of Bias Mitigation of Clone Detection Models](http://arxiv.org/abs/2505.05679v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

This paper investigates the use of large language models (LLMs) for clone code detection. The authors find that LLMs like PaLM perform well on this task, achieving high F1 scores. However, they also highlight the susceptibility of LLMs to prompt bias, where small changes in the input prompt can significantly affect performance.  The paper then delves into the reasons behind these fluctuations, identifies eight categories of prompt bias related to LLMs' misunderstanding of the clone code task, and proposes a framework to mitigate this bias by incorporating "prompt bias lessons." Their approach involves analyzing model errors, categorizing them, and adding lessons derived from these categories to improve prompt effectiveness. Experiments show that using these "prompt bias lessons" leads to a significant improvement in F1 scores, underscoring the impact and potential for leveraging model errors.

**Critical Evaluation:**

The paper offers a valuable contribution to the growing body of research on using LLMs in software engineering. Its novelty lies in several aspects:

*   **LLMs for Clone Detection:** While LLMs are being used in many SE tasks, clone detection hasn't been widely explored, and the paper bridges this gap. This alone is worth noting.
*   **Focus on Prompt Bias from an Error Perspective:** Rather than simply searching for better prompts, the authors systematically investigate *why* the models struggle, analyzing errors, and categorizing them.  This is a significant shift in understanding prompt bias.
*   **Bias Mitigation Framework:** The approach of crafting "prompt bias lessons" based on error categories is innovative. It provides a structured way to improve model performance by addressing underlying misunderstandings.

**Strengths:**

*   **Rigorous Methodology:** The paper uses clearly defined experimental setups, including model selection, datasets, and evaluation metrics. The ablation study helps to assess the impact of individual prompt bias lessons. The use of Cohen's kappa and statistical significance tests adds rigor.
*   **Identified Bias Categories:** The identified prompt bias categories are insightful and well-defined. They provide a useful taxonomy for understanding how LLMs can misinterpret clone detection tasks.
*   **Quantifiable Improvements:** The paper demonstrates significant performance improvements through the proposed bias mitigation framework. The empirical evidence is compelling.
*   **Clear Writing and Organization:** The paper is well-structured and easy to follow.

**Weaknesses:**

*   **Manual Error Analysis:** The process of identifying prompt bias categories and crafting lessons relies on manual analysis. This is labor-intensive and potentially subjective. Automating this process would be a valuable extension. While the high Cohen's kappa minimizes the risk of bias, it is not eliminated.
*   **Limited Scope of Datasets:** While the chosen datasets (PoolC and a constructed cross-language dataset) are appropriate, the findings may not generalize to all clone detection scenarios or programming languages.
*   **Generalizability of Prompt Bias Categories:** While the "prompt bias lessons" had success in this study, the categories identified are not necessarily exhaustive.

**Significance:**

The findings have practical implications for developers using LLMs in software engineering tasks like clone detection. The proposed framework provides a way to improve model accuracy by addressing prompt bias. It pushes beyond 'black box' use of LLMs by encouraging analyzing and correcting misunderstandings.

**Justification for Score:**

The paper is not revolutionary, but its rigorous methodology, novel approach to prompt bias, and quantifiable improvements justify a high score. The biggest weakness is the manual nature of the bias identification, which limits scalability and introduces potential subjectivity. The practical relevance is clear.

Score: 8

- **Score**: 8/10

### **[APOLLO: Automated LLM and Lean Collaboration for Advanced Formal Reasoning](http://arxiv.org/abs/2505.05758v1)**
- **Summary**: Here's a summary and critical evaluation of the provided paper, aiming for rigor and justification.

**Summary:**

The paper introduces APOLLO, a novel and fully automated pipeline for automated theorem proving (ATP) in the Lean formal verification system.  APOLLO aims to improve the efficiency and accuracy of Large Language Models (LLMs) in generating formally correct proofs.  It achieves this by combining the reasoning abilities of LLMs with the strict error detection capabilities of the Lean compiler.  The pipeline involves a modular agent-based approach: 1) Syntax Refiner fixes syntax errors in LLM outputs, 2) a "Sorrifier" inserts `sorry` placeholders to mark incomplete proof steps, 3) an Auto Solver attempts to solve the remaining goals automatically, and 4) recursive application of the LLM on each remaining `sorry` goal with a low top-K budget, and finally 5) Proof Assembler combines sub-proofs back into a complete proof. The framework is model-agnostic and allows for collaboration between LLMs, the Lean compiler, and automated solvers.  The paper demonstrates that APOLLO significantly improves the state-of-the-art accuracy on the miniF2F benchmark, especially for smaller LLMs (e.g. 7B parameter models), while simultaneously reducing the sample complexity (the number of LLM calls) compared to simply generating whole proofs repeatedly.  The authors highlight the benefits of compiler-guided repair in achieving these results.

**Critical Evaluation:**

* **Novelty:**  The central idea of using the Lean compiler as a guide to repair and refine LLM-generated proofs is a significant innovation.  While prior work exists on using feedback to repair proofs, Apollo innovatively integrates this feedback into a modular, agent-based pipeline, allowing for targeted LLM calls on specific sub-problems. The modular design and specific techniques within each module (e.g., how the "Sorrifier" handles compilation errors) contributes to the novelty. This approach is a departure from the standard "whole-proof generation" paradigm prevalent in the field. Furthermore, the system's reliance on a low top-K budget for LLM calls contrasts with the previous standard practice of high-sampling for proof attempts.

* **Significance:**  The demonstrated results on the miniF2F benchmark are compelling.  Achieving state-of-the-art accuracy with a significantly reduced sampling budget is a valuable contribution.  The fact that APOLLO unlocks performance in smaller, general-purpose models (e.g., OpenAI's 03-mini/04-mini) is also noteworthy, suggesting a path towards more efficient and accessible ATP. The paper's impact lies in offering a scalable approach to automated theorem proving by utilizing existing tools and formalisms (compiler and automated solvers). A system that makes ATP more efficient may also lead to more research and exploration in this field.

* **Strengths:**
    * **Clear and well-defined pipeline:** The modular architecture of APOLLO is well-explained and makes it easy to understand and potentially extend.
    * **Strong empirical results:** The performance improvements on miniF2F are significant and backed by thorough experimentation and comparisons to existing methods.  The analysis of proof lengths and the impact of the recursion depth parameter provide valuable insights.
    * **Model-agnostic design:** APOLLO is compatible with a variety of LLMs, increasing its generalizability and potential impact.
    * **Emphasis on efficiency:**  The reduction in sample complexity is a crucial consideration for the practical application of ATP.

* **Weaknesses:**
    * **Reliance on base model quality:** The paper acknowledges that APOLLO's effectiveness is dependent on the quality of the initial proof sketch generated by the LLM.  While the system can repair many errors, it may struggle with fundamentally flawed proof strategies. This dependence limits the applicability of APOLLO to base models with sufficient reasoning capabilities. Further investigation into the minimum quality of input needed may have been useful.
    * **Limited exploration of tree-search integration:**  The limitations section mentions that APOLLO is not integrated with tree-search methods. Expanding into these would potentially add value to their tool.
    * **miniF2F focus:**  The evaluation is largely confined to the miniF2F benchmark.  While this is a standard dataset, it would be beneficial to see APOLLO's performance on other ATP benchmarks or real-world mathematical problems.
    * **"Sampling budget" discussion is convoluted.** It could have been clearer that there were two distinct metrics that were being conflated -- one as the sampling budget for a wholistic approach, and the other as the average sampling per "proof" from Apollo

* **Potential Influence:** APOLLO has the potential to significantly influence the field of ATP. The compiler-guided repair paradigm may become a standard approach, and the modular design could inspire future research on more sophisticated agent-based ATP systems. Its emphasis on efficiency makes it more accessible. The system may promote further collaboration between AI researchers and mathematicians by providing a platform for leveraging formal verification tools.

**Justification of Score:**

The paper presents a novel and significant contribution to the field of ATP by introducing APOLLO. The system's architecture, empirical results, and potential impact are noteworthy. While it has some limitations (especially dependence on base-model quality), the strengths outweigh the weaknesses. APOLLO advances the field by demonstrating a more efficient and accurate approach to combining LLMs with formal verification tools. Therefore, it merits a high score.

Score: 8

- **Score**: 8/10

### **[AgentXploit: End-to-End Redteaming of Black-Box AI Agents](http://arxiv.org/abs/2505.05849v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "AGENTXPLOIT: End-to-End Redteaming of Black-Box AI Agents":

**Summary:**

The paper introduces AGENTXPLOIT, a novel black-box fuzzing framework designed to automatically discover and exploit indirect prompt injection vulnerabilities in LLM agents. The framework operates in a black-box manner, requiring no access to the agent's internal workings. AGENTXPLOIT begins with a high-quality seed corpus of attack instructions and iteratively refines these using Monte Carlo Tree Search (MCTS) to maximize the likelihood of uncovering weaknesses. The approach incorporates adaptive seed scoring based on attack coverage and custom mutators for LLM agent inputs. The authors evaluate AGENTXPLOIT on AgentDojo and VWA-adv benchmarks, demonstrating significant improvements over baseline attacks, achieving success rates of 71% and 70% respectively, against 03-mini and GPT-4o based agents. The framework also exhibits strong transferability across unseen tasks and LLMs and shows promising results against defenses. Finally, the attacks are tested on real-world scenarios.

**Critical Evaluation:**

*   **Novelty:** The paper makes a strong claim of novelty by presenting AGENTXPLOIT as the first generic indirect prompt injection assessment method against black-box LLM agents. While previous works have addressed prompt injection, they often rely on white-box access, hand-crafted attacks, or are designed for specific agent types. AGENTXPLOIT's black-box fuzzing approach, coupled with MCTS-based seed selection and adaptive scoring, represents a genuine advance in the field. The systematic approach of fuzzing combined with intelligent seed selection addresses the sparse feedback signal challenge unique to LLM agents in a creative way.
*   **Significance:** Indirect prompt injection is a critical security risk for LLM agents, potentially leading to serious consequences. AGENTXPLOIT's ability to automatically discover and exploit these vulnerabilities in a black-box setting is highly significant. It allows for systematic red-teaming and vulnerability assessment, even without detailed knowledge of the agent's architecture or internal LLM. The demonstrated success rates against various benchmarks and the transferability of the generated attacks further highlight the framework's practical importance. The real-world case study further validates the transferability of the attack.
*   **Strengths:**
    *   The framework addresses an important and timely problem in LLM agent security.
    *   The black-box approach enhances its applicability to real-world deployed agents.
    *   The MCTS-based seed selection and adaptive scoring mechanisms are well-motivated and contribute to the effectiveness of the fuzzing process.
    *   Extensive experimental results on multiple benchmarks, defense strategies, and real-world scenarios validate the framework's performance and transferability.
    *   The ablation study provides insights into the contribution of different components of the framework.

*   **Weaknesses:**
    *   The paper assumes a binary success/failure feedback mechanism, which is extremely sparse. While it presents an adaptive scoring and MCTS approach to counter this, it still requires careful setup and doesn't deal with cases when feedback is noisy, ambiguous or delayed.
    *   While the black-box nature is emphasized, the need for an initial seed corpus implies some knowledge or assumptions about the agent's behavior and potential vulnerabilities. The reliance on an initial corpus can skew results, and better details on corpus construction are needed.
    *   The paper acknowledges that the defenses are not infallible and that ongoing research is necessary. Furthermore, the complexity and potential cost of deploying AGENTXPLOIT in resource-constrained environments are not thoroughly explored.
    *   While the code is mentioned as publicly available, this wasn't explicitly confirmed and could be checked to ensure reproducibility of work.

*   **Potential Influence:** AGENTXPLOIT has the potential to significantly influence the field by providing a practical and automated approach to identifying and mitigating prompt injection vulnerabilities in LLM agents. It can be used by security researchers, developers, and organizations to proactively assess the security posture of their LLM-based systems. The framework's modular design and black-box nature make it adaptable to a wide range of agent architectures and tasks, promoting its widespread adoption. The work also sets a precedent for using fuzzing techniques in the context of LLM security, inspiring further research in this area.

**Overall Score:**

Score: 8

**Justification:**

AGENTXPLOIT represents a significant advancement in the field of LLM agent security by providing a practical and automated black-box fuzzing framework for detecting and exploiting indirect prompt injection vulnerabilities. The novelty lies in the systematic approach of intelligent seed generation and selection, which addresses the challenges of sparse feedback signals and complex agent architectures. The extensive experimental validation on multiple benchmarks, defense strategies, and real-world scenarios solidifies the significance of the work. While the reliance on an initial seed corpus and binary feedback mechanisms imposes some limitations, the overall impact of AGENTXPLOIT on the field is substantial, warranting a high score of 8. The availability of the code as promised, would improve the quality to a 9.

- **Score**: 8/10

### **[PICD: Versatile Perceptual Image Compression with Diffusion Rendering](http://arxiv.org/abs/2505.05853v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "PICD: Versatile Perceptual Image Compression with Diffusion Rendering":

**Summary:**

The paper introduces PICD, a novel perceptual image compression codec specifically designed to handle both screen content and natural images effectively.  Traditional perceptual image codecs often struggle with screen content, especially text, due to their focus on preserving overall visual fidelity rather than accurate text reproduction. PICD addresses this by encoding text information losslessly and rendering it with a compressed image using a conditional diffusion model. The paper presents a three-tiered conditioning approach: 1) fine-tuning the base diffusion model on text-related content (domain level), 2) an efficient adaptor conditioned on text, location, and compressed image (adaptor level), and 3) instance-wise guidance during decoding. The results show PICD surpasses existing perceptual codecs for both text accuracy and visual quality. It performs as a perceptual codec effectively on natural images without text conditions.

**Critical Evaluation:**

*   **Novelty:** The core novelty lies in the combination of lossless text encoding with diffusion rendering to address the specific challenge of compressing screen content while maintaining both visual quality and text fidelity. The three-tiered conditioning approach for the diffusion model is also a notable contribution. Existing approaches often focus on one aspect (text accuracy or perceptual quality), and the integrated approach of PICD appears effective. The use of a conditional diffusion model with location-specific text rendering within image compression is well-justified and novel.

*   **Significance:** The paper addresses a practical problem in image compression, where screen content, with its inherent text and sharp edges, often suffers from artifacts when compressed using standard perceptual codecs. A successful solution is valuable, especially with the increasing prevalence of screen sharing, remote work, and educational content.  The superior performance demonstrated by PICD offers a clear improvement over existing methods.

*   **Strengths:**

    *   **Comprehensive Approach:** The three-tiered conditioning strategy provides a structured and effective way to guide the diffusion model, addressing different aspects of the problem.
    *   **Strong Empirical Results:**  The paper presents thorough experimental results demonstrating the advantages of PICD on both screen and natural images using a variety of metrics.  The ablation studies clearly illustrate the contribution of each component.
    *   **Addresses a Real-World Problem:**  The paper addresses a pertinent issue in practical image compression, namely the degradation of text in screen content.
    *   **Clear and Well-Written:** The paper is easy to understand, and it clearly describes the method and its benefits.

*   **Weaknesses:**

    *   **Decoding Speed:** The decoding speed, especially with instance-level guidance, is a potential limitation. While the paper acknowledges this and suggests avenues for improvement, the current decoding time might be a barrier to real-time applications. Specifically, the paper states that the decoding speed increases by a factor of three compared to other diffusion codecs, which is a drawback.
    *   **Reliance on OCR:** The reliance on OCR for text extraction might lead to failures in certain scenarios, particularly with complex or distorted text. Though failure cases are presented, a more robust solution less dependent on OCR success would improve the overall reliability.
    *   **Limited Discussion of Computational Complexity:** The paper mentions a peak memory usage and FLOPS, but a more in-depth analysis of the computational complexity of the encoding and decoding processes, especially concerning the diffusion model, would enhance the evaluation.

*   **Potential Influence:** PICD could influence future research in perceptual image compression by highlighting the importance of considering specific content types, like screen content, and by demonstrating the effectiveness of combining lossless text encoding with diffusion rendering. The proposed three-tiered conditioning approach could also be adopted and adapted in other generative compression models.

*   **Justification of the Score:** The paper demonstrates a clear and effective solution to a relevant problem, providing compelling empirical results.  The method builds on established techniques but introduces a novel and well-engineered combination to achieve superior performance. While the decoding speed and reliance on OCR are limitations, the overall contribution warrants a high score. The clear superiority and the thorough investigation of the approach are compelling.

Score: 8

- **Score**: 8/10

### **[DiffLocks: Generating 3D Hair from a Single Image using Diffusion Models](http://arxiv.org/abs/2505.06166v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "DiffLocks: Generating 3D Hair from a Single Image using Diffusion Models":

**Summary:**

The paper presents a novel framework called DiffLocks for generating detailed 3D hair geometry from a single image.  The core contributions are threefold: (1) the creation of a large-scale synthetic 3D hair dataset (40K hairstyles) to overcome data scarcity; (2) a new scalp representation, based on a learned texture map storing latent codes for individual hair strands, to enable modeling of complex hairstyles and varying hair densities; and (3) a conditional diffusion transformer network that leverages DINOv2 image features to generate accurate 3D strands from a single RGB image. The framework directly decodes the strand latent codes into 3D hair strands *without* post-processing steps. This allows DiffLocks to reconstruct diverse and complex hairstyles, including curly afro-like hair and balding patterns, with a level of detail exceeding previous methods.

**Critical Evaluation:**

*   **Novelty:** The paper demonstrates significant novelty in several aspects.
    *   **Large-scale synthetic dataset:** While synthetic data is common in this area, the creation of a 40K diverse hairstyle dataset is a valuable contribution to the community. The authors address the critical data scarcity issue that has plagued previous approaches.
    *   **Scalp Texture Representation:** The idea of storing latent codes for individual strands in a texture map provides a powerful and flexible representation that avoids the limitations of low-dimensional intermediate representations like guide strands. This texture can also encode hair density and balding patterns.
    *   **Diffusion-based Strand Generation:** Leveraging a diffusion transformer to directly generate hair strands from image features and a scalp texture, bypassing intermediate steps like 2D orientation maps, is an elegant solution that enables richer detail and realism.
    *   **End-to-end training:** Unlike prior work, their approach requires no post-processing to add detail or realism because all of it is learned by the model.

*   **Significance:** The paper addresses a critical challenge in digital human creation – realistic 3D hair modeling. DiffLocks exhibits a notable advancement over existing single-image hair reconstruction techniques. By addressing the data bottleneck, the paper enables the creation of more detailed and diverse hair models, which could have a significant impact on character realism in games, media, and other applications. This is one of the first methods capable of handling male-pattern baldness, afro-like hairstyles, and other complex geometries.

*   **Strengths:**
    *   **Strong empirical results:**  The paper provides both qualitative and quantitative comparisons demonstrating DiffLocks' superiority over existing state-of-the-art methods. Ablation studies effectively highlight the importance of each component of the framework.
    *   **Clear and well-structured:**  The paper is well-written and clearly explains the technical details of the proposed method.  The figures are informative and help the reader understand the framework.
    *   **Addresses a key limitation:** The paper directly tackles the data scarcity issue that has historically limited the performance of hair reconstruction methods.
    *   **Generalization:** Although trained only on synthetic data, the use of a pretrained image backbone enables generalization to real-world images.

*   **Weaknesses:**
    *   **Synthetic Data Bias:** While the synthetic dataset is large and diverse, there is always a risk of the model overfitting to the characteristics of synthetic data. The results, while good, might not fully translate to arbitrary "in-the-wild" images with complex lighting and occlusions.
    *   **Limitations on hairstyle variety:** Although improved, certain hairstyles, such as braids and ponytails, are still absent from the data set.
    *   **Computational Cost:** While claiming to be faster in runtime performance than prior work, the time for dataset generation is still relatively high.
    *   **Error in Section 4.3**: There is an apparent error in Section 4.3 where the authors state that 3D orientation field is created, yet they do not claim this as a creation of their work.

*   **Potential Influence:** The paper has the potential to influence future research on 3D hair modeling in the following ways:
    *   **New benchmark dataset:** The released 3D synthetic hair dataset will likely become a valuable resource for the community, facilitating the development of new and improved methods.
    *   **Inspiration for new representations:**  The scalp texture representation could inspire new ways of parameterizing and generating complex 3D geometries.
    *   **Emphasis on data:**  The paper highlights the importance of high-quality and diverse training data for deep learning-based reconstruction methods.

**Justification for Score:**

Given the paper's novel contributions, significant performance improvements, and strong potential for future impact, a score of 8 is justified. While there is still room for improvement, particularly in addressing limitations related to training data and complexity, the paper represents a significant step forward in the field of 3D hair modeling. The creation of a large-scale dataset and the novel use of a diffusion transformer in this context make it a noteworthy and impactful contribution.

**Score: 8**

- **Score**: 8/10

### **[Turbo-ICL: In-Context Learning-Based Turbo Equalization](http://arxiv.org/abs/2505.06175v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

This paper introduces a novel framework called Turbo-ICL for soft-input soft-output channel equalization in MIMO systems.  It leverages in-context learning (ICL), drawing inspiration from large language models (LLMs), to learn the channel characteristics directly from pilot signals and decoder feedback, eliminating the need for explicit channel state information (CSI). The key innovation lies in a prompt augmentation technique that incorporates extrinsic information from the decoder into the ICL model, enabling iterative refinement of symbol estimates during turbo decoding.  Two model variants, based on Transformer and state-space architectures (SSMs), are presented and their performance is evaluated through simulations.  Results demonstrate that the ICL equalizers outperform conventional model-based approaches, particularly in scenarios with low-resolution quantization, even when the baselines have access to perfect CSI. The authors also highlight the strengths of Transformer-based models under limited training data and the computational efficiency of SSMs.

**Critical Evaluation:**

*   **Novelty:** The paper's primary novelty resides in the integration of ICL into the turbo equalization framework, specifically for *soft-input soft-output* equalization. While previous works have explored ICL for MIMO detection, they typically output hard decisions and don't integrate with turbo decoding loops.  The prompt augmentation technique, using decoder feedback to refine symbol estimates iteratively, is also a significant contribution. Prior works have used ICL for channel estimation/equalization, but this work is the first, to the best of my knowledge, to introduce and integrate ICL into a turbo equalization architecture by providing a clever prompt augmentation technique. It successfully converts existing single-pass ICL channel estimators into iterative channel estimators that are suitable for turbo-decoding.

*   **Significance:** The paper addresses a crucial limitation of traditional turbo equalizers which rely on accurate channel models, that may become limiting with hardware constraints.  The ICL approach offers a CSI-free alternative, adapting to channel conditions directly from pilot data. This has potential to simplify receiver design and improve performance in challenging scenarios with hardware impairments and nonlinearities. The demonstrations showing improved robustness against quantization are significant, given the increasing use of low-resolution ADCs in communication systems. The paper convincingly demonstrates advantages of ICL in scenarios where linear models break down.

*   **Strengths:**
    *   **Clear Problem Definition:** The paper clearly identifies the limitations of traditional turbo equalizers and motivates the need for adaptive, CSI-free solutions.
    *   **Technical Innovation:** The prompt augmentation technique is a clever way to incorporate decoder feedback into the ICL framework.
    *   **Thorough Evaluation:** The paper provides extensive simulation results comparing the proposed approach to strong baselines under different conditions (modulation orders, quantization levels, SNRs, code rates).
    *   **Analysis of Architectures:** The paper offers a good comparative analysis of the Transformer and SSM architectures for ICL equalization, discussing their trade-offs in terms of generalization, scalability, and computational efficiency.

*   **Weaknesses:**
    *   **Complexity:** While the paper discusses computational efficiency, the complexity of implementing Transformer or SSM-based ICL equalizers in real-time systems remains a concern.
    *   **Pre-training Data Dependency:** The performance of ICL-based methods heavily relies on the diversity and quality of pre-training data. More discussion on strategies for generating robust pre-training datasets would be beneficial.
    *   **Limited Practical Considerations:** The simulations assume a quasi-static fading channel.  The paper could benefit from experiments that further simulate more realistic channel models.

*   **Potential Impact:** The paper has the potential to significantly impact the design of future communication receivers, particularly in scenarios where channel estimation is difficult or unreliable, or where hardware limitations are present.  The ICL approach could enable more adaptive and robust communication systems, especially in dynamic environments. The work is likely to stimulate further research on ICL for wireless communication, specifically on methods for channel equalization.

*   **Score:** 8

**Rationale:**

The paper represents a significant advancement in the field of turbo equalization by successfully integrating in-context learning for the first time, paving the way for CSI-free, adaptive receivers. While it is the first work to address the turbo equalization with ICL, several works have applied the Transformer to channel estimation. This limits its novelty, so the score is adjusted to 8.

- **Score**: 8/10

## Other Papers
### **[Multi-agent Embodied AI: Advances and Future Directions](http://arxiv.org/abs/2505.05108v1)**
### **[Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders](http://arxiv.org/abs/2505.05111v1)**
### **[MDAA-Diff: CT-Guided Multi-Dose Adaptive Attention Diffusion Model for PET Denoising](http://arxiv.org/abs/2505.05112v1)**
### **[Enhancing Text2Cypher with Schema Filtering](http://arxiv.org/abs/2505.05118v1)**
### **[Text2Cypher: Data Pruning using Hard Example Selection](http://arxiv.org/abs/2505.05122v1)**
### **[Research on Anomaly Detection Methods Based on Diffusion Models](http://arxiv.org/abs/2505.05137v1)**
### **[Overcoming Dimensional Factorization Limits in Discrete Diffusion Models through Quantum Joint Distribution Learning](http://arxiv.org/abs/2505.05151v1)**
### **[FedTDP: A Privacy-Preserving and Unified Framework for Trajectory Data Preparation via Federated Learning](http://arxiv.org/abs/2505.05155v1)**
### **[MARK: Memory Augmented Refinement of Knowledge](http://arxiv.org/abs/2505.05177v1)**
### **[Stochastic Variational Propagation: Local, Scalable and Efficient Alternative to Backpropagation](http://arxiv.org/abs/2505.05181v1)**
### **[Revealing Weaknesses in Text Watermarking Through Self-Information Rewrite Attacks](http://arxiv.org/abs/2505.05190v1)**
### **[EAM: Enhancing Anything with Diffusion Transformers for Blind Super-Resolution](http://arxiv.org/abs/2505.05209v1)**
### **[Diffusion Model Quantization: A Review](http://arxiv.org/abs/2505.05215v1)**
### **[Normalize Everything: A Preconditioned Magnitude-Preserving Architecture for Diffusion-Based Speech Enhancement](http://arxiv.org/abs/2505.05216v1)**
### **[QualBench: Benchmarking Chinese LLMs with Localized Professional Qualifications for Vertical Domain Evaluation](http://arxiv.org/abs/2505.05225v1)**
### **[ChemRxivQuest: A Curated Chemistry Question-Answer Database Extracted from ChemRxiv Preprints](http://arxiv.org/abs/2505.05232v1)**
### **[Latte: Transfering LLMs` Latent-level Knowledge for Few-shot Tabular Learning](http://arxiv.org/abs/2505.05237v1)**
### **[T-T: Table Transformer for Tagging-based Aspect Sentiment Triplet Extraction](http://arxiv.org/abs/2505.05271v1)**
### **[Software Development Life Cycle Perspective: A Survey of Benchmarks for Code Large Language Models and Agents](http://arxiv.org/abs/2505.05283v2)**
### **[HEXGEN-TEXT2SQL: Optimizing LLM Inference Request Scheduling for Agentic Text-to-SQL Workflow](http://arxiv.org/abs/2505.05286v1)**
### **[Benchmarking Ophthalmology Foundation Models for Clinically Significant Age Macular Degeneration Detection](http://arxiv.org/abs/2505.05291v1)**
### **[Toward Reasonable Parrots: Why Large Language Models Should Argue with Us by Design](http://arxiv.org/abs/2505.05298v1)**
### **[ICon: In-Context Contribution for Automatic Data Selection](http://arxiv.org/abs/2505.05327v1)**
### **[Denoising Diffusion Probabilistic Models for Coastal Inundation Forecasting](http://arxiv.org/abs/2505.05381v1)**
### **[PillarMamba: Learning Local-Global Context for Roadside Point Cloud via Hybrid State Space Model](http://arxiv.org/abs/2505.05397v1)**
### **[Frame In, Frame Out: Do LLMs Generate More Biased News Headlines than Humans?](http://arxiv.org/abs/2505.05406v1)**
### **[Crosslingual Reasoning through Test-Time Scaling](http://arxiv.org/abs/2505.05408v1)**
### **[Hide & Seek: Transformer Symmetries Obscure Sharpness & Riemannian Geometry Finds It](http://arxiv.org/abs/2505.05409v1)**
### **[Reasoning Models Don't Always Say What They Think](http://arxiv.org/abs/2505.05410v1)**
### **[TokLIP: Marry Visual Tokens to CLIP for Multimodal Comprehension and Generation](http://arxiv.org/abs/2505.05422v1)**
### **[LiTransProQA: an LLM-based Literary Translation evaluation metric with Professional Question Answering](http://arxiv.org/abs/2505.05423v2)**
### **[Ultra-FineWeb: Efficient Data Filtering and Verification for High-Quality LLM Training Data](http://arxiv.org/abs/2505.05427v1)**
### **[EcoAgent: An Efficient Edge-Cloud Collaborative Multi-Agent Framework for Mobile Automation](http://arxiv.org/abs/2505.05440v2)**
### **[clem:todd: A Framework for the Systematic Benchmarking of LLM-Based Task-Oriented Dialogue System Realisations](http://arxiv.org/abs/2505.05445v1)**
### **[Conversational Process Model Redesign](http://arxiv.org/abs/2505.05453v1)**
### **[UKElectionNarratives: A Dataset of Misleading Narratives Surrounding Recent UK General Elections](http://arxiv.org/abs/2505.05459v1)**
### **[Bring Reason to Vision: Understanding Perception and Reasoning through Model Merging](http://arxiv.org/abs/2505.05464v1)**
### **[ComPO: Preference Alignment via Comparison Oracles](http://arxiv.org/abs/2505.05465v1)**
### **[Mogao: An Omni Foundation Model for Interleaved Multi-Modal Generation](http://arxiv.org/abs/2505.05472v1)**
### **[DiffusionSfM: Predicting Structure and Motion via Ray Origin and Endpoint Diffusion](http://arxiv.org/abs/2505.05473v1)**
### **[Prompt to Polyp: Clinically-Aware Medical Image Synthesis with Diffusion Models](http://arxiv.org/abs/2505.05573v1)**
### **[KG-HTC: Integrating Knowledge Graphs into LLMs for Effective Zero-shot Hierarchical Text Classification](http://arxiv.org/abs/2505.05583v1)**
### **[PRIMG : Efficient LLM-driven Test Generation Using Mutant Prioritization](http://arxiv.org/abs/2505.05584v1)**
### **[ReactDance: Progressive-Granular Representation for Long-Term Coherent Reactive Dance Generation](http://arxiv.org/abs/2505.05589v1)**
### **[Enhancing Large Language Models with Faster Code Preprocessing for Vulnerability Detection](http://arxiv.org/abs/2505.05600v1)**
### **[HiBayES: A Hierarchical Bayesian Modeling Framework for AI Evaluation Statistics](http://arxiv.org/abs/2505.05602v1)**
### **[scDrugMap: Benchmarking Large Foundation Models for Drug Response Prediction](http://arxiv.org/abs/2505.05612v1)**
### **[Leveraging Large Language Models for enzymatic reaction prediction and characterization](http://arxiv.org/abs/2505.05616v1)**
### **[LiteLMGuard: Seamless and Lightweight On-Device Prompt Filtering for Safeguarding Small Language Models against Quantization-induced Risks and Vulnerabilities](http://arxiv.org/abs/2505.05619v1)**
### **[A Preliminary Study for GPT-4o on Image Restoration](http://arxiv.org/abs/2505.05621v1)**
### **[Looking Beyond Language Priors: Enhancing Visual Comprehension and Attention in Multimodal Models](http://arxiv.org/abs/2505.05626v1)**
### **[Privacy-Preserving Transformers: SwiftKey's Differential Privacy Implementation](http://arxiv.org/abs/2505.05648v1)**
### **[Unsupervised Blind Speech Separation with a Diffusion Prior](http://arxiv.org/abs/2505.05657v1)**
### **[Not Like Us, Hunty: Measuring Perceptions and Behavioral Effects of Minoritized Anthropomorphic Cues in LLMs](http://arxiv.org/abs/2505.05660v1)**
### **[Adaptive Stress Testing Black-Box LLM Planners](http://arxiv.org/abs/2505.05665v1)**
### **[Lost in OCR Translation? Vision-Based Approaches to Robust Document Retrieval](http://arxiv.org/abs/2505.05666v1)**
### **[InstanceGen: Image Generation with Instance-level Instructions](http://arxiv.org/abs/2505.05678v1)**
### **[From Bias To Improved Prompts: A Case Study of Bias Mitigation of Clone Detection Models](http://arxiv.org/abs/2505.05679v1)**
### **[Fine-Tuning Video-Text Contrastive Model for Primate Behavior Retrieval from Unlabeled Raw Videos](http://arxiv.org/abs/2505.05681v1)**
### **[Assessing Robustness to Spurious Correlations in Post-Training Language Models](http://arxiv.org/abs/2505.05704v1)**
### **[LLM-Text Watermarking based on Lagrange Interpolation](http://arxiv.org/abs/2505.05712v1)**
### **[Semantic-Space-Intervened Diffusive Alignment for Visual Classification](http://arxiv.org/abs/2505.05721v1)**
### **[Towards Secure Semantic Transmission In the Era of GenAI: A Diffusion-based Framework](http://arxiv.org/abs/2505.05724v1)**
### **[Automated Learning of Semantic Embedding Representations for Diffusion Models](http://arxiv.org/abs/2505.05732v1)**
### **[Multimodal Integrated Knowledge Transfer to Large Language Models through Preference Optimization with Biomedical Applications](http://arxiv.org/abs/2505.05736v1)**
### **[Harnessing LLMs Explanations to Boost Surrogate Models in Tabular Data Classification](http://arxiv.org/abs/2505.05744v1)**
### **[Insertion Language Models: Sequence Generation with Arbitrary-Position Insertions](http://arxiv.org/abs/2505.05755v1)**
### **[Evolutionary thoughts: integration of large language models and evolutionary algorithms](http://arxiv.org/abs/2505.05756v1)**
### **[APOLLO: Automated LLM and Lean Collaboration for Advanced Formal Reasoning](http://arxiv.org/abs/2505.05758v1)**
### **[Multi-Agent Systems for Robotic Autonomy with LLMs](http://arxiv.org/abs/2505.05762v1)**
### **[Sparse Attention Remapping with Clustering for Efficient LLM Decoding on PIM](http://arxiv.org/abs/2505.05772v1)**
### **[A Day in Their Shoes: Using LLM-Based Perspective-Taking Interactive Fiction to Reduce Stigma Toward Dirty Work](http://arxiv.org/abs/2505.05786v1)**
### **[Demystifying Diffusion Policies: Action Memorization and Simple Lookup Table Alternatives](http://arxiv.org/abs/2505.05787v1)**
### **[What Is Next for LLMs? Next-Generation AI Computing Hardware Using Photonic Chips](http://arxiv.org/abs/2505.05794v1)**
### **[3D CAVLA: Leveraging Depth and 3D Context to Generalize Vision Language Action Models for Unseen Tasks](http://arxiv.org/abs/2505.05800v1)**
### **[Accelerating Diffusion Transformer via Increment-Calibrated Caching with Channel-Aware Singular Value Decomposition](http://arxiv.org/abs/2505.05829v1)**
### **[AgentXploit: End-to-End Redteaming of Black-Box AI Agents](http://arxiv.org/abs/2505.05849v1)**
### **[PICD: Versatile Perceptual Image Compression with Diffusion Rendering](http://arxiv.org/abs/2505.05853v1)**
### **[Evolutionary ecology of words](http://arxiv.org/abs/2505.05863v1)**
### **[A 3D pocket-aware and evolutionary conserved interaction guided diffusion model for molecular optimization](http://arxiv.org/abs/2505.05874v1)**
### **[CAPE: Context-Aware Prompt Perturbation Mechanism with Differential Privacy](http://arxiv.org/abs/2505.05922v1)**
### **[NeoQA: Evidence-based Question Answering with Generated News Events](http://arxiv.org/abs/2505.05949v1)**
### **[Task-Adapter++: Task-specific Adaptation with Order-aware Alignment for Few-shot Action Recognition](http://arxiv.org/abs/2505.06002v1)**
### **[ArtRAG: Retrieval-Augmented Generation with Structured Context for Visual Art Understanding](http://arxiv.org/abs/2505.06020v1)**
### **[Unilogit: Robust Machine Unlearning for LLMs Using Uniform-Target Self-Distillation](http://arxiv.org/abs/2505.06027v1)**
### **[Healthy LLMs? Benchmarking LLM Knowledge of UK Government Public Health Information](http://arxiv.org/abs/2505.06046v1)**
### **[Noise-Consistent Siamese-Diffusion for Medical Image Synthesis and Segmentation](http://arxiv.org/abs/2505.06068v1)**
### **[Assessing Tenstorrent's RISC-V MatMul Acceleration Capabilities](http://arxiv.org/abs/2505.06085v1)**
### **[UniSymNet: A Unified Symbolic Network Guided by Transformer](http://arxiv.org/abs/2505.06091v1)**
### **[LLMs Outperform Experts on Challenging Biology Benchmarks](http://arxiv.org/abs/2505.06108v1)**
### **[The Application of Deep Learning for Lymph Node Segmentation: A Systematic Review](http://arxiv.org/abs/2505.06118v1)**
### **[LLMs Get Lost In Multi-Turn Conversation](http://arxiv.org/abs/2505.06120v1)**
### **[Can Prompting LLMs Unlock Hate Speech Detection across Languages? A Zero-shot and Few-shot Study](http://arxiv.org/abs/2505.06149v1)**
### **[A Scaling Law for Token Efficiency in LLM Fine-Tuning Under Fixed Compute Budgets](http://arxiv.org/abs/2505.06150v1)**
### **[DiffLocks: Generating 3D Hair from a Single Image using Diffusion Models](http://arxiv.org/abs/2505.06166v1)**
### **[Turbo-ICL: In-Context Learning-Based Turbo Equalization](http://arxiv.org/abs/2505.06175v1)**
### **[A Large Language Model-Enhanced Q-learning for Capacitated Vehicle Routing Problem with Time Windows](http://arxiv.org/abs/2505.06178v1)**
### **[Brain Hematoma Marker Recognition Using Multitask Learning: SwinTransformer and Swin-Unet](http://arxiv.org/abs/2505.06185v1)**
### **[Topo-VM-UNetV2: Encoding Topology into Vision Mamba UNet for Polyp Segmentation](http://arxiv.org/abs/2505.06210v1)**
