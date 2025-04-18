# The Latest Daily Papers - Date: 2025-04-18
## Highlight Papers
### **[Selective Demonstration Retrieval for Improved Implicit Hate Speech Detection](http://arxiv.org/abs/2504.12082v1)**
- **Summary**: Here's a summary and critical evaluation of the provided paper:

**Summary:**

The paper addresses the challenge of detecting implicit hate speech, which is subtle and context-dependent, making it difficult for standard language models to identify accurately. The authors propose Adaptive Retrieval-based In-context Learning (ARIIHA), a novel framework that leverages in-context learning without fine-tuning. ARIIHA adaptively retrieves demonstrations (examples) to guide the model, prioritizing those with similar target groups or high similarity scores. This dynamic retrieval strategy is modulated by thresholds derived from similarity scores and the model's reliance on shortcut cues. Experimental results on the Implicit Hate Corpus (IHC) dataset show that ARIIHA outperforms existing state-of-the-art techniques, demonstrating enhanced detection accuracy and reduced over-sensitivity.

**Critical Evaluation:**

*   **Novelty:** The paper introduces a novel approach to in-context learning for hate speech detection. The adaptive retrieval mechanism, combining target group prioritization with similarity scores and shortcut cue detection, is a significant contribution. The idea of dynamically switching between retrieval strategies based on thresholds is a strong point, showing adaptability to the specific input. The paper is also the first to propose a retrieval-based in-context learning framework for implicit hate speech.

*   **Significance:** Implicit hate speech detection is a critical area of research. ARIIHA's ability to improve detection accuracy while reducing over-sensitivity is significant. Over-sensitivity, which leads to false positives, is a common problem in hate speech detection, and the paper's approach directly tackles this issue. The demonstration of the method's effectiveness on a challenging dataset like IHC further strengthens its significance. The ablations performed clearly show the impact of the proposed methods.

*   **Strengths:**
    *   The adaptive retrieval mechanism is well-designed and theoretically sound.
    *   The experimental results demonstrate clear improvements over existing methods.
    *   The ablation study provides valuable insights into the contribution of each component.
    *   The case study provides further explanation on how ARIIHA works in certain situations.
    *   The approach does not require fine-tuning, making it more accessible and scalable.

*   **Weaknesses:**

    *   While the thresholds seem crucial to the approach, it's unclear how robust these thresholds are. They are optimized on the dev set, but the generalization of these thresholds to unseen datasets isn't discussed in depth.
    *   The target group prediction stage introduces a potential point of failure. If the target group is incorrectly predicted, the subsequent retrieval process will be compromised. More analysis on the impact of incorrect target group predictions would be valuable.
    *   The paper references Qwen2.5-7B, but its reliance on the model for evaluation is concerning since Qwen2 is a relatively new LLM compared to, for example, Llama2 or Mistral.

*   **Potential Influence:** The paper has the potential to influence the direction of research in hate speech detection and in-context learning. The adaptive retrieval mechanism could be adopted and extended in other tasks where context and nuance are crucial. Furthermore, its method can be extended in the future to consider multimodal hate speech data, further increasing the significance of the method.

**Justification for Score:**

ARIIHA makes a strong contribution to the field of implicit hate speech detection. The adaptive retrieval method is novel and effective, addressing a critical challenge in the area. The weaknesses are primarily related to the generalizability of the thresholds and the potential dependence on the accurate target group prediction. Overall, the method has significant potential to improve hate speech detection systems.

**Score: 8**

- **Score**: 8/10

### **[Entropy-Guided Watermarking for LLMs: A Test-Time Framework for Robust and Traceable Text Generation](http://arxiv.org/abs/2504.12108v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces a novel watermarking scheme for Large Language Models (LLMs) aimed at improving both detectability and text quality.  The key innovation is the introduction of a cumulative watermark entropy threshold.  Below this threshold, the output remains unwatermarked.  Above the threshold, watermarking is applied using preceding tokens as a seed to generate the key. This approach is designed to be compatible with and generalize existing sampling functions. Experimental results on various LLMs indicate significant improvements over existing methods in terms of text quality (especially on long-answer tasks) while maintaining high detection accuracy, even under paraphrase attacks. The authors also offer theoretical analysis proving the indistinguishability and provide the source code.

**Critical Evaluation:**

* **Novelty:** The concept of using an entropy threshold to dynamically determine when to apply watermarking is a significant contribution.  Existing methods often apply watermarking uniformly, which can degrade the quality of already deterministic or low-entropy text.  By adaptively controlling the watermarking process based on entropy, the proposed method offers a more refined approach that strikes a better balance between detectability and text quality. Adapting the binary sampling and constructing a new mapping are also innovative.
* **Significance:** The paper addresses a critical challenge in LLM watermarking: the trade-off between detectability and text quality.  The experimental results show substantial improvements in text quality, particularly for long-answer tasks. This is important because LLMs are increasingly used for tasks that require generating coherent and informative text. The improved robustness against paraphrase attacks also enhances the practical utility of the watermarking scheme in real-world scenarios where adversaries might try to remove the watermark. Providing the source code increases the potential impact.
* **Strengths:**
    *   The entropy threshold is an elegant and effective solution to the detectability/quality trade-off.
    *   The experimental evaluation is comprehensive, covering multiple LLMs, datasets, and attack methods.
    *   The paper includes theoretical analysis to support the design choices and claims.
    *   The results demonstrate a clear improvement over existing methods, particularly in long-answer question answering scenarios.
    *   The approach appears to be easily adaptable to existing sampling functions.

* **Weaknesses:**
    *   While the evaluation covers several models, further testing on larger and more recent LLMs would strengthen the results.
    *   The performance gains appear more pronounced on long-answer tasks. A more detailed analysis of performance on shorter text generation tasks would be useful.
    * The detailed explanation of mapping functions might be too complex to read.
    *   While robustness against paraphrase attacks is addressed, exploring more advanced attack methods (e.g., those using LLMs themselves to rephrase the text) could provide a more stringent evaluation.
    * The contribution is built on top of pre-existing watermarking methods.

* **Potential Influence:** The paper has the potential to influence the design of future watermarking schemes for LLMs. The entropy threshold approach could become a standard technique for improving text quality while maintaining detectability. The code release will accelerate the adoption of the approach by other researchers and practitioners.

* **Score:** 8

**Rationale:**

The paper presents a novel and well-validated watermarking scheme that makes a significant contribution to the field.  The entropy threshold is a clever idea that addresses a crucial limitation of existing methods. The experimental results are compelling, and the theoretical analysis provides a solid foundation for the approach.  While there are some limitations (scope of models evaluated, more advanced attacks) and there are many pre-existing works in the field, the strengths significantly outweigh the weaknesses. I assign a score of 8, indicating a valuable and impactful contribution that is likely to influence future research and development in LLM watermarking.

- **Score**: 8/10

### **[Anti-Aesthetics: Protecting Facial Privacy against Customized Text-to-Image Synthesis](http://arxiv.org/abs/2504.12129v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper addresses the privacy and copyright risks associated with personalized text-to-image synthesis using diffusion models.  It proposes a Hierarchical Anti-Aesthetic (HAA) framework designed to degrade the generation quality of maliciously customized models, thereby protecting facial privacy. HAA consists of two branches: (1) Global Anti-Aesthetics, which degrades the overall aesthetic properties of the generated content, and (2) Local Anti-Aesthetics, which disrupts local facial identity by guiding adversarial perturbations. The framework aims to degrade image quality from a global and local perspective, making it harder to create recognizable and malicious personalized content.  The authors demonstrate the effectiveness of their method through extensive experiments and comparisons with existing state-of-the-art privacy protection techniques.

**Critical Evaluation:**

*   **Novelty:** The paper introduces an interesting and novel perspective by using anti-aesthetics as a defense mechanism against malicious use of customized diffusion models.  While adversarial attacks are not new in the general sense, applying them specifically through the lens of manipulating aesthetic properties for privacy protection is a distinct contribution. The hierarchical approach, combining global and local anti-aesthetic strategies, adds another layer of novelty.

*   **Significance:** The potential impact of this work is considerable.  As personalized content generation becomes more widespread, protecting individual privacy and copyright is crucial.  The paper's approach to degrading image quality strategically, rather than simply adding noise, could be more effective in preventing malicious use while minimizing the impact on legitimate applications. The empirical validation shows HAA significantly outperforms existing methods in removing facial identity.

*   **Strengths:**

    *   **Innovative Approach:** Using aesthetics as a defense strategy is a compelling idea. It leverages the understanding of human perception to disrupt malicious content generation.
    *   **Hierarchical Design:** The separation of global and local anti-aesthetic components allows for a more nuanced and targeted attack on the customized diffusion models.
    *   **Empirical Validation:** The paper presents a thorough experimental evaluation with multiple datasets, diffusion models, and comparison methods, demonstrating the effectiveness of HAA.
    *   **Robustness Testing:** The inclusion of black-box testing with different DM versions and robustness tests against various image perturbations strengthens the practical value of the method.
*   **Weaknesses:**

    *   **Reliance on Aesthetic Metrics:** The success of the method depends heavily on the effectiveness of the aesthetic reward models (RMg and RM1). These models are trained on specific datasets, and their performance might degrade if applied to images with different styles or content.
    *   **Computational Cost:** The iterative training process, involving both adversarial noise generation and surrogate model optimization, could be computationally expensive. The paper does not address its efficiency explicitly.
    *   **Potential for Circumvention:** While HAA demonstrates strong results, it's essential to recognize that adversarial defenses are often subject to adversarial attacks. Future research might explore methods to circumvent the HAA framework.
    *   **Subjectivity of Aesthetics:** Aesthetic perception can be subjective and influenced by various factors (cultural background, individual preferences). The current method assumes that certain aesthetic properties are universally favored, which could lead to biases or inconsistencies in the results.

*   **Potential Influence:**  This paper has the potential to influence the design of future privacy-preserving content generation techniques. By highlighting the importance of aesthetic cues, it opens up new avenues for developing more effective and robust defense mechanisms.

**Justification:**

The paper presents a novel and promising approach to protecting privacy in the context of personalized content generation. Its strength lies in its innovative use of anti-aesthetics and its hierarchical design.  The empirical validation is comprehensive and convincing, indicating a significant improvement over existing methods.  However, the reliance on aesthetic metrics and the potential for circumvention are important limitations to consider. While the work contributes a significant advancement in the field, the issues of computational costs and subjective aesthetic perception are areas requiring further research.

**Score: 8**

A score of 8 reflects the paper's significant novelty and impact, strong empirical validation, and potential influence on the field, balanced by its reliance on specific aesthetic metrics and the inherent challenges of adversarial defense.

- **Score**: 8/10

### **[MOS: Towards Effective Smart Contract Vulnerability Detection through Mixture-of-Experts Tuning of Large Language Models](http://arxiv.org/abs/2504.12234v1)**
- **Summary**: Here's a summary and critical evaluation of the provided paper:

**Summary:**

The paper "MOS: Towards Effective Smart Contract Vulnerability Detection through Mixture-of-Experts Tuning of Large Language Models" introduces a new framework, MOS, for detecting vulnerabilities in smart contracts. It addresses limitations of existing methods (program analysis, deep learning, and LLMs) by employing a Mixture-of-Experts (MOE) tuning approach using Large Language Models (LLMs). The framework includes:

1.  **Continual Pre-training:** A domain-enhanced initialization for the LLM using smart contract data.
2.  **MOE-Tuning Dataset:** A dataset created with LLM generation and expert verification for reliable vulnerability explanations.
3.  **Vulnerability-Aware Routing:** Directing code features to relevant expert networks.
4.  **Specialized MOE Network:** Multiple parallel expert networks specializing in specific vulnerability patterns.
5.  **Dual-Objective Loss Function:** Optimizing both vulnerability detection and explanation.

The paper presents experimental results demonstrating that MOS outperforms existing methods in F1 score and accuracy across various vulnerability types, while also generating high-quality vulnerability explanations. The evaluation process also involves a combined approach using both human and LLM evaluation for better performance.

**Critical Evaluation:**

*   **Novelty:** The paper presents a novel combination of techniques for smart contract vulnerability detection. The use of MOE tuning with LLMs in this context is a significant step forward. Although other approaches exist, MOS’s architecture and training methodology (dual-objective loss, vulnerability-aware routing) offer a distinctive advantage.
*   **Significance:** Smart contract security is a crucial issue in blockchain technology due to immutability and potential for significant financial losses. The proposed framework aims to improve the current security standards. The improvements in detection performance and the generation of explanations contribute valuable tools for developers and auditors. The framework's architecture makes it inherently adaptable to new or emerging threats.
*   **Strengths:**
    *   Strong empirical results: The experiments show significant improvements over baselines.
    *   Comprehensive evaluation: Multiple datasets and evaluation metrics are used.
    *   Explanation capability: The framework focuses on providing meaningful explanations, which is important for practical use.
    *   Well-defined architecture: The individual components of the MOS framework are clearly described.
*   **Weaknesses:**
    *   Computational requirements: The framework requires considerable computational resources for training. This can limit its accessibility. The implementation details about the distributed resources might be helpful to the interested readers.
    *   Prompt engineering sensitivity: Like all LLM-based approaches, the framework's performance might be sensitive to prompt engineering choices.
    *   Dataset bias: Potential bias in the dataset, including the MOE-tuning and evaluation datasets, is a concern. More details on the distribution of vulnerabilities would make the claims much more stronger.
    *   Complexity for adoption: The complexity in architecture may be a possible hinderence for quick adaptation by practitioners. It can also impact explainability and trust to the model, leading to slower real-world adoptions.
*   **Potential influence:** The paper has the potential to influence the field by providing a more effective and explainable approach to smart contract vulnerability detection. The MOE tuning approach and the focus on generating explanations can be adopted by other researchers. The dataset created as part of this work is also a valuable contribution.

**Justification for the score:**

The paper demonstrates a strong understanding of current problems and makes a substantial contribution towards improving the current security standards. The experimental results, architectural choices, and evaluation techniques are well-justified. However, some challenges include the high computational costs, the potential prompt sensitivity, and complexities for practitioners. These limitations prevent it from reaching an exceptionally high score.
Score: 8

- **Score**: 8/10

### **[DMM: Building a Versatile Image Generation Model via Distillation-Based Model Merging](http://arxiv.org/abs/2504.12364v1)**
- **Summary**: **Summary:** The paper presents a novel approach called DMM (Distillation-Based Model Merging) aimed at addressing the challenges posed by the proliferation of diverse text-to-image (T2I) generation models. These models, typically fine-tuned on specialized datasets, result in redundancy and increased storage costs. Traditional model merging techniques, which rely on static linear interpolation, fail to harness the distinct styles and features of various models effectively. The authors propose a style-promptable image generation pipeline that utilizes style vectors to control arbitrary-style image outputs. The DMM paradigm compresses multiple teacher models into a unified T2I model while rethinking the merging task with new goals and evaluation protocols. Experiments reveal that DMM can effectively consolidate knowledge from various models and produce controllable images across different styles. **Critical Evaluation:** The paper makes a significant contribution to the field of image generation by addressing the pressing issues of model redundancy and storage costs, which are critical in practical applications. The introduction of a style-promptable approach is particularly noteworthy, as it enhances the flexibility and versatility of T2I models, allowing users to generate images in varied styles, which has substantial implications for creative applications, content generation, and user-centered design. However, the paper could be strengthened by a more comprehensive evaluation of the proposed merging strategy against existing methods. While the authors present promising experimental results, the lack of a detailed comparative analysis limits the understanding of DMM's effectiveness relative to other contemporary techniques in the literature. Furthermore, the scalability of the approach to future models and the practicality of real-world applications could be further elaborated. Overall, the paper is innovative and presents a well-defined methodology that potentially advances the state of T2I generation models. The ideas introduced can influence both academic research and industrial applications in the field. Therefore, considering its contribution, applicability, and the need for further analysis, the paper deserves a **high score**. Score: 8
- **Score**: 8/10

### **[Collaborative Perception Datasets for Autonomous Driving: A Review](http://arxiv.org/abs/2504.12696v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper "Collaborative Perception Datasets for Autonomous Driving: A Review" presents a comprehensive review of collaborative perception (CP) datasets designed for autonomous driving (AD). It categorizes and compares existing datasets based on cooperation paradigms (V2V, V2I, V2X, I2I), data sources (simulation vs. real-world), scenarios (intersection, urban street, highway), sensor modalities (image, LiDAR, radar), and supported perception tasks (object detection, semantic segmentation, tracking). The review also analyzes dataset availability, quality, and impact, and discusses challenges and future directions, including dataset scalability, standardization, privacy, and the integration of large language models (LLMs). The authors also provide a continuously updated online repository.

**Critical Evaluation:**

The paper addresses a timely and important need in the autonomous driving research community. The proliferation of CP datasets, while beneficial, has also created a need for a systematic overview and comparison to guide researchers in selecting appropriate resources and standardizing evaluation. The strengths of the paper include:

*   **Comprehensiveness:**  The review covers a significant number of CP datasets and analyzes them across multiple dimensions. The taxonomy used is logical and allows for easy comparison. The inclusion of I2I datasets demonstrates a forward-thinking perspective, capturing emerging trends.
*   **Detailed Analysis:** The comparative analysis is thorough, examining not only the general characteristics of the datasets but also specific metrics and benchmark results.
*   **Practical Value:**  The identification of challenges and future directions provides actionable insights for the community, highlighting areas needing further research and development. The accompanying online repository significantly enhances the practical value.
*   **Clear Roadmap:** The paper’s tracing of the chronological evolution is helpful to contextualize the field.
*   **Figures:** The inclusion of figures 1, 2, 3, 4, 5, 6 and 7 provide a visual component to the document to easily digest information.
*   **Tables:** The inclusion of tables I, II, III, IV and V are very helpful in comparing information between datasets to extract information quickly.

However, some weaknesses exist:

*   **Depth of Technical Details:** While the paper categorizes and compares datasets, the technical details of specific algorithms used for benchmarking on these datasets could be expanded.
*   **Subjectivity in "High-Influence" Datasets:** The selection of "high-influence" datasets, while supported by citation analysis, inevitably involves some subjectivity. Justification for why certain datasets are considered more influential than others, beyond citation counts, could be more elaborated.
*   **Limited Discussion of Specific LLM Integration:** While the paper mentions the potential of LLMs, it lacks concrete examples of how LLMs can be practically integrated into CP datasets beyond the high-level suggestions of data augmentation and scene analysis. Specific architectures and methodologies could be briefly outlined.
*   **The review is limited in scope to research published prior to April 2025.
*   **Dataset accessibility information could change over time.

**Novelty and Significance:**

The primary novelty lies in the paper's comprehensive scope. While individual aspects (e.g., sensor fusion, specific tasks) may have been reviewed before, the holistic, multi-dimensional analysis of *collaborative perception datasets* specifically is a significant contribution. The paper offers a much-needed structured overview of a rapidly evolving field, facilitating resource discovery and promoting standardization. The online repository is a valuable contribution that enhances the paper's longevity. The paper consolidates key information and provides a well-structured overview, saving researchers considerable time and effort in navigating the landscape of CP datasets. It also highlights gaps and encourages further development.

**Justification for Score:**

Despite the minor weaknesses noted, the paper delivers a valuable contribution to the autonomous driving and collaborative perception research community. It provides a much-needed systematic overview and comparison of a rapidly evolving field. The comprehensiveness, practical value, and insights into future directions make it a significant resource for researchers and practitioners alike. The online repository further amplifies the impact of the work.

Score: 8

- **Score**: 8/10

### **[Enhancing the Geometric Problem-Solving Ability of Multimodal LLMs via Symbolic-Neural Integration](http://arxiv.org/abs/2504.12773v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces GeoGen, a novel pipeline to enhance the geometric problem-solving (GPS) ability of Multimodal Large Language Models (MLLMs). GeoGen tackles the challenge of limited high-quality, step-by-step solution data and hallucinations in MLLMs for GPS by automatically generating such data. It leverages a symbolic system to produce correct, step-wise reasoning paths from geometry diagrams, and then uses a Large Language Model (LLM) called GeoLogic to translate between natural language and formal symbolic representations. This allows for verifying MLLM outputs against geometric principles. The paper also introduces two datasets, GeoExpand (expanding upon existing datasets) and GeoSynth (generated from scratch).  Experiments demonstrate that MLLMs trained with the new datasets and using GeoLogic for verification achieve improved performance on standard GPS benchmarks.

**Critical Evaluation:**

*   **Novelty:**

    *   The idea of using symbolic systems to *generate* training data for MLLMs in the geometric reasoning domain is a significant step. Previous works have mostly focused on guiding or verifying through the symbolic system. The automatic data generation pipeline (GeoGen) is relatively novel.
    *   GeoLogic, as a translator between natural language and formal geometric language, is also novel and allows for efficient verification and interpretability.
    *   The generation of both GeoExpand and GeoSynth datasets contributes to the community by providing high-quality, large-scale multi-step reasoning data, addressing a critical bottleneck.

*   **Significance:**

    *   The paper addresses a core issue in applying MLLMs to geometry: the lack of suitable training data. The synthetic data generation approach effectively overcomes this limitation.
    *   The results demonstrate a clear improvement in performance on GPS benchmarks, suggesting that the proposed approach is effective in enhancing MLLM reasoning capabilities.
    *   The integration of symbolic reasoning and LLMs provides a pathway towards more reliable and interpretable AI systems for complex reasoning tasks.
    *   By creating high-quality synthetic data, the work also reduces the reliance on costly and time-consuming manual annotation.

*   **Strengths:**

    *   **Clear Problem Definition:** The paper identifies and articulates a well-defined problem (data scarcity and hallucinations) in the context of MLLMs for GPS.
    *   **Sound Methodology:** The proposed GeoGen pipeline is well-motivated and technically sound.  The use of symbolic reasoning for generating ground truth is a strong advantage.
    *   **Comprehensive Experiments:** The paper presents a thorough experimental evaluation, including ablation studies and comparisons with existing methods. The analysis of data composition is particularly insightful.
    *   **Performance Improvement:** The results convincingly demonstrate a consistent performance improvement over various MLLMs.
    *   **Reproducibility**: the code is released

*   **Weaknesses:**

    *   **Verification Limitation:** The inference strategy through GeoLogic and symbolical verification is simple. More powerful and interpretable methods are expected, such as using a symbol solver to provide a better answer given the question and image.
    *   **Evaluation against more SOTA solvers**:  While comparisons are made to a broad range of existing approaches, the gap between the GeoGen model and the latest purely symbolic or neural solvers is evident.

*   **Potential Influence:**

    *   The paper has the potential to influence future research on MLLMs for mathematical reasoning and other complex domains.
    *   The proposed data generation approach can be adapted to other domains where high-quality training data is scarce.
    *   The integration of symbolic reasoning and LLMs can lead to more reliable and interpretable AI systems.
    *   The framework of GeoGen and GeoLogic provide a solid foundation for future work on MLLMs for GPS.

**Justification for Score:**

The paper presents a novel and significant contribution to the field of MLLMs for geometric problem-solving. The automatic data generation and symbolic integration significantly improve the MLLM performance, and the paper is well-written and thoroughly evaluated. While limitations exist, mainly in the completeness of the symbolic verification and the gap against very recent symbolic solvers, the strengths outweigh the weaknesses.

Score: 8

- **Score**: 8/10

### **[Image-Editing Specialists: An RLAIF Approach for Diffusion Models](http://arxiv.org/abs/2504.12833v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Image-Editing Specialists: An RLAIF Approach for Diffusion Models":

**Summary:**

The paper introduces a novel approach to training specialized instruction-based image editing diffusion models. It tackles the challenges of preserving structural coherence and achieving semantic alignment with user prompts (both textual and visual). The key innovation is an online reinforcement learning framework using AI feedback (RLAIF), eliminating the need for extensive human annotations. The method fine-tunes a pre-trained diffusion model by optimizing a loss function that encourages both structural similarity to the input image and semantic alignment with the provided instructions (text and style images).  The paper demonstrates improved precision in complex edits, stronger alignment with instructions, and showcases its utility in robotics by enhancing the realism of simulated environments.

**Critical Evaluation:**

*   **Novelty:**  The paper's main novelty lies in its RLAIF-based training approach for instruction-guided image editing.  While RLAIF and diffusion models are individually established, their combination in this specific context, coupled with the customized loss function that explicitly balances structural preservation and semantic alignment, represents a significant contribution. The use of visual prompts alongside text instructions, is a powerful idea for localized edits. Also, the specific method for combining this with classifer-free guidance is novel. The application to sim-to-real image editing in robotics is also novel, demonstrating practical value beyond artistic applications.

*   **Significance:** The work addresses a crucial limitation in current image editing diffusion models: the difficulty in performing precise, localized edits that maintain structural integrity. The ability to leverage visual prompts along with text significantly enhances the user control and reduces the need for complex text prompts. This simplifies user interaction and increases the potential for practical applications, especially in domains like robotics and design where fine-grained control is essential. The improvement in sim-to-real transfer in robotics is a compelling demonstration of real-world significance. The relatively lightweight fine-tuning (few steps) also increases usability.

*   **Strengths:**

    *   **Effective RLAIF Framework:** The AI feedback-driven reinforcement learning approach appears to be effective in achieving the desired balance between structural preservation and semantic alignment, without reliance on expensive human annotation.
    *   **Visual Prompt Integration:** The incorporation of visual prompts, along with textual instructions, is a significant strength, allowing for nuanced stylistic edits and reducing the need for overly complex text prompts.
    *   **Practical Application:** The application to sim-to-real transfer in robotics is a strong demonstration of the method's practical utility. The gains achieved here are significant.
    *   **Clear Evaluation:** The paper presents a comprehensive set of quantitative and qualitative evaluations, including comparisons with state-of-the-art methods and ablation studies.

*   **Weaknesses:**

    *   **Reliance on InstructPix2Pix:** While building on InstructPix2Pix is a good starting point, it means some limitations may be inherited. The paper could provide more analysis of this, and where it diverges.
    *   **Limited Edit Types:** While the paper covers a good variety of edits, some limitations remain, as acknowledged in the conclusion. The method is likely better suited to texture/material changes than large structural modifications.
    *   **Reliance on Grounded SAM for Evaluation Metrics:** Using Grounded SAM to assess how the model makes edits on the prompt region can be good for quantification, but will bias the quantitative results, so is not ideal.
    *   **Hyperparameter Sensitivity:** The classifier-free guidance scale is a hyperparameter that requires tuning for each edit type. This limits the ease of use in production systems.

*   **Potential Influence:** This work has the potential to influence the development of more precise and controllable image editing tools.  The RLAIF approach could be adapted to other generative tasks where aligning with human preferences is crucial. The integration of visual prompts could become a standard feature in future image editing interfaces.

**Justification for Score:**

The paper presents a clearly written, technically sound, and well-evaluated approach to an important problem. The novelty of the RLAIF framework and the integration of visual prompts are significant contributions. The demonstration of practical utility in robotics is a compelling aspect. While the approach has some limitations, it represents a clear advancement over existing methods and has the potential to influence future research. I am assigning a score of 8.

Score: 8

- **Score**: 8/10

### **[A Virtual Machine for Arbitrary Low-Precision GPGPU Computation in LLM Serving](http://arxiv.org/abs/2504.12984v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces Tilus, a GPGPU virtual machine (VM) designed specifically to improve the efficiency of serving Large Language Models (LLMs) with low-precision computation. Tilus addresses the limitations of existing approaches that either support only power-of-two bit widths or suffer from suboptimal performance due to high-level GPU programming abstractions. Tilus features a thread-block-level programming model, a hierarchical memory space, a novel algebraic layout system, and native support for arbitrary low-precision data types (1-8 bits, including integers and floating-point). The VM programs are compiled into efficient GPU programs with automatic vectorization and instruction selection. Experiments show that Tilus outperforms state-of-the-art low-precision kernels, including those generated by Triton and Ladder, as well as hand-optimized kernels like QuantLLM and Marlin.

**Critical Evaluation:**

*   **Novelty:** The paper presents a genuinely novel architecture in the Tilus VM. The combination of:

    *   Explicit thread-block-level programming with exposed hierarchical memory,
    *   An algebraic layout system enabling flexible register tensor reinterpretation,
    *   Native support for arbitrary low-precision data types,

    Distinguishes it from existing compiler-based approaches like Triton and Ladder. The algebraic layout system, in particular, seems innovative. It allows fine-grained control over data placement and movement that existing approaches often abstract away.
*   **Significance:** Addressing the performance limitations of low-precision kernels is very significant for efficient LLM serving. The ability to support arbitrary bit widths (e.g., 5-7 bits) fills a gap in the current ecosystem and allows for better accuracy-efficiency trade-offs. The substantial performance improvements shown in the experiments over existing compilers and hand-optimized kernels are highly impactful. Furthermore, the paper's claim of reducing programming effort through higher-level VM programmability (while maintaining performance) makes it a significant contribution. The demonstration within the context of realistic LLMs using vLLM solidifies the significance.
*   **Strengths:**
    *   The algebraic layout system is a well-defined and compelling component, addressing limitations in previous systems for managing memory layouts.
    *   Comprehensive experimental results, comparing against a wide range of baselines on diverse LLM kernels.
    *   The paper clearly articulates the problems with existing approaches and provides a well-reasoned solution.
    *   The implementation details (compilation pipeline, etc.) are reasonably well-documented.
*   **Weaknesses:**
    *   While the performance benefits are well documented, the complexity of the VM might make it less accessible than more user-friendly tools like Triton.  There's a trade-off between control and ease-of-use. More details on the programming experience with the VM would have been beneficial.
    *   The reliance on hand-tuning (tile sizes etc.) in the virtual machine program might limit the generalizability, despite the abstract nature of the VM itself. It may be necessary to use an auto-tuning layer (such as in the frameworks they compete with), which could diminish the performance advantage.
    *   The paper, while providing the components of their DSL, lacks examples of how these pieces come together. Having more examples would have made it easier for the reader to understand the virtual machine better.
*   **Potential Influence:** The paper has the potential to significantly influence the development of low-precision LLM serving frameworks. It provides a blueprint for designing more flexible and efficient kernels, and it could inspire new compiler optimizations and hardware architectures.  The algebraic layout system could be particularly influential.

**Overall:**

Tilus represents a substantial advance in the field of low-precision LLM serving. The combination of architectural novelties and significant performance improvements makes it a valuable contribution. The weaknesses identified (e.g., usability, tuning requirements) are areas for future research and development, and do not diminish the fundamental value of the work. The experimental section is solid and comprehensive.

Score: 8

- **Score**: 8/10

### **[InstructRAG: Leveraging Retrieval-Augmented Generation on Instruction Graphs for LLM-Based Task Planning](http://arxiv.org/abs/2504.13032v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces InstructRAG, a novel retrieval-augmented generation (RAG) framework for LLM-based task planning. The core idea is to address the limitations of LLMs' knowledge for complex tasks by leveraging external databases grounded through RAG. The authors identify two key challenges in applying RAG to task planning: enlargeability (extending the database's coverage) and transferability (generalizing to new tasks). InstructRAG addresses these through:

1.  **Instruction Graph:** Organizes past successful instruction paths as a graph. Nodes represent similar instructions, and edges represent associated tasks.
2.  **RL-Agent:** Uses reinforcement learning to explore and identify candidate instruction paths within the graph, enhancing enlargeability.
3.  **ML-Agent:** Employs meta-learning to select the most relevant path from the RL-Agent's suggestions and integrates it into the LLM's prompt, improving transferability.

The RL-Agent and ML-Agent are trained end-to-end in a multi-agent meta-reinforcement learning framework. Experiments on HotpotQA, ALFWorld, Webshop, and ScienceWorld datasets demonstrate significant performance improvements compared to existing methods. The paper also highlights InstructRAG's ability to adapt rapidly to new tasks through few-shot learning.

**Critical Evaluation:**

*   **Novelty:** The paper's primary novelty lies in its systematic approach to addressing the limitations of RAG in task planning and framing the problem in terms of enlargeability and transferability. While RAG and meta-learning are not new, their integration within a multi-agent framework specifically designed for instruction graph exploration *is* a novel contribution. Building an instruction graph and using RL to actively traverse it is also a good idea.
*   **Significance:** The paper presents strong empirical results across several challenging datasets. The performance gains over baselines, particularly RAP, are substantial. The few-shot learning capabilities are also significant, demonstrating the framework's adaptability. The analysis of ablation studies offers valuable insights into the contribution of each component.
*   **Strengths:**
    *   **Clear Problem Definition:** The paper clearly identifies the limitations of existing task planning approaches and motivates the use of RAG.
    *   **Well-Defined Framework:** InstructRAG is a well-structured and modular framework.
    *   **Strong Empirical Evaluation:** Extensive experiments on multiple datasets and with different LLMs provide compelling evidence of InstructRAG's effectiveness.
    *   **Detailed Analysis:** The paper includes thorough ablation studies and parameter sensitivity analysis.
*   **Weaknesses:**
    *   **Complexity:** The framework is relatively complex, with multiple components and training stages. This complexity might make it challenging to implement and tune in practice.
    *   **Instruction Graph Construction:** The method relies on initially building a graph with successful instruction paths. Performance might degrade if this process is flawed or if the initial instructions are not robust. The performance is constrained on the type of LLM is used in generating instruction graph in the first place. The instruction graph size also has the limitation to the graph size, so may require a trade-off on runtime vs. performance.
    *   **Limited Generalization Beyond RAG:** The paper focuses almost exclusively on using RAG for task planning. While that's the goal, the core ideas might be applicable to other areas where LLMs struggle with knowledge limitations, so could discuss a brief highlight.
    *   **Reliance on Pre-existing Task Solvers:** The construction of the instruction graph relies on a pre-existing task solver. The quality of this solver greatly impacts the quality of the instruction graph, which then determines the effectiveness of the whole method.

*   **Potential Influence:** The paper has the potential to influence future research in LLM-based task planning by demonstrating the effectiveness of RAG and providing a structured approach for leveraging external knowledge. The framework can also serve as a foundation for developing more advanced RAG techniques for various applications.

The paper is a significant contribution, but its somewhat higher level of complexity and dependency on the quality of initial instruction data limit the score.
Score: 8

- **Score**: 8/10

### **[GraphAttack: Exploiting Representational Blindspots in LLM Safety Mechanisms](http://arxiv.org/abs/2504.13052v1)**
- **Summary**: Here's a summary and critical evaluation of the provided paper, "GraphAttack: Exploiting Representational Blindspots in LLM Safety Mechanisms":

**Summary:**

The paper introduces GraphAttack, a novel method for generating jailbreak prompts for Large Language Models (LLMs) by exploiting vulnerabilities in their safety mechanisms. GraphAttack uses graph-based semantic representations (Abstract Meaning Representation - AMR, Resource Description Framework - RDF, and JSON knowledge graphs) to encode malicious prompts and then applies semantic transformations to evade safety filters. A key aspect of the approach is instructing LLMs to generate code that realizes the intent described in the semantic graphs, which has proven surprisingly effective in bypassing safety measures. The paper empirically demonstrates that GraphAttack achieves significantly higher attack success rates compared to existing jailbreaking methods across multiple LLMs and datasets, highlighting the limitations of current safety alignment techniques that primarily focus on surface-level patterns. The paper also offers a detailed analysis of the types of semantic transformations that are most effective at bypassing these mechanisms, providing insights for improving LLM safety. The study explores the vulnerability when generating code versus natural language.

**Critical Evaluation:**

**Strengths:**

*   **Novel Approach:** GraphAttack presents a genuinely novel approach to jailbreaking LLMs by focusing on semantic representations and transformations rather than surface-level text manipulations. This is a significant departure from many existing jailbreaking techniques.
*   **Systematic Methodology:** The paper offers a systematic framework for exploring the semantic transformation space, which enables a more principled and comprehensive assessment of model vulnerabilities. This goes beyond ad-hoc prompt engineering.
*   **Exploiting a Key Vulnerability:** The insight that LLMs are more vulnerable to harmful content encoded in semantic representations, particularly when generating code, is a crucial contribution. It exposes a significant gap in current safety alignment techniques.
*   **Comprehensive Evaluation:** The paper provides a comprehensive empirical evaluation of GraphAttack across multiple LLMs, datasets, and evaluation metrics. This robust evaluation strengthens the validity of the findings. The comparison with strong baselines further highlights the effectiveness of the proposed approach.
*   **Actionable Insights:** The analysis of effective semantic transformations and the discussion of potential countermeasures provide actionable insights for improving LLM safety.
* The generation of code to bypass safety features when structured semantic knowledge graphs of intent are given highlights the difference in treating requests.

**Weaknesses:**

*   **Dependence on External Parsers/Generators:** The method relies on external semantic parsers (AMR, RDF) or LLMs to generate JSON knowledge graphs. The accuracy and reliability of these parsers can impact the overall performance of GraphAttack. The use of LLMs introduces non-determinism.
*   **Limited Scope of Semantic Transformations:** While the paper explores a range of semantic transformations, the scope could be expanded to include more sophisticated techniques or combinations of transformations.
*   **Generalizability beyond specific datasets/models:** The experiments are conducted on a specific set of LLMs and datasets. More analysis on how GraphAttack performs on new models and new datasets would increase the generalizability of the results.

**Novelty and Significance:**

The novelty of this paper is high. The graph-based semantic transformation approach is a significant departure from existing jailbreaking techniques. The paper effectively highlights the critical vulnerability where models process semantic representations and code more as technical challenges than harmful content, offering new directions for research. The significance is also high. The work provides insights and practical methodology that can be used for red-teaming, and more robust safety measures can be built by addressing these vulnerabilities. The potential impact of GraphAttack extends beyond simply demonstrating new jailbreaking techniques as the framework provides a base for understanding limitations in the approaches that are in place.

**Justification for the Score:**

Considering the strengths and weaknesses, the novelty and significance of the work, it merits a score of **8**. The approach is significantly innovative, well-validated, and offers valuable insights for improving LLM safety. While there are some limitations in terms of reliance on external parsers and the scope of transformations, the overall contribution is substantial and will likely influence future research in this area.

**Score: 8**

- **Score**: 8/10

### **[HiScene: Creating Hierarchical 3D Scenes with Isometric View Generation](http://arxiv.org/abs/2504.13072v1)**
- **Summary**: Here's a summary and critical evaluation of the HiScene paper:

**Summary:**

The paper introduces HiScene, a novel hierarchical framework for generating 3D scenes from text prompts, addressing the limitations of existing approaches that often lack object diversity, realism, and editing flexibility. HiScene leverages isometric views to treat entire scenes as hierarchical "objects," enabling the decomposition into manipulatable items. The framework incorporates a video-diffusion-based amodal completion technique to handle object occlusions and shadows, as well as a shape prior injection mechanism for spatial coherence. The result is a system that generates high-fidelity 3D scenes with compositional identities, natural object arrangements, and enhanced editing capabilities.

**Critical Evaluation:**

*   **Novelty:** The paper's novelty lies in its hierarchical scene representation using isometric views, effectively bridging 2D image generation and 3D object generation. The amodal completion using video diffusion to handle occlusions and shadow removal is another novel aspect. Finally, the shape prior injection addresses the important issue of spatial coherence in generated scenes. The use of native 3D generation models for refinement is a strength.

*   **Significance:** Generating realistic and editable 3D scenes from text is a key challenge in computer graphics and AI. HiScene addresses this challenge by improving the realism and composability of generated scenes, offering a more user-friendly and flexible approach compared to previous methods that often rely on predefined 3D layouts or suffer from limitations in object diversity and spatial understanding. The ability to edit and manipulate individual objects within the generated scene significantly increases the practical utility of the generated scenes.

*   **Strengths:**

    *   **Hierarchical Approach:** Effectively enables compositional scene generation.
    *   **Video-Diffusion-Based Amodal Completion:** Adequately handles occlusions and shadows, improving the completeness and realism of objects.
    *   **Spatial Alignment:** Shape prior injection ensures spatial coherence.
    *   **Strong results:** Both the qualitative and quantitative results presented in the paper are impressive.
    *   **The method doesn't rely on hand-crafted rules** as existing scene creation approaches do.

*   **Weaknesses:**
    *   **Computational cost:** The runtime evaluation shows a reasonably high runtime, around 12 minutes. It is also worth pointing out that GALA3D and DreamScene could offer lower times with further optimization.
    *   **Limited Textures**: As the authors mention themselves, the method produces scenes with baked lighting.

*   **Potential Influence:** HiScene can significantly influence the field by providing a robust and flexible framework for 3D scene generation. It could be used in a variety of applications such as: interactive scene editing, game development, content creation, robotics, and augmented reality. By enabling users to easily create and manipulate 3D scenes from text, HiScene has the potential to democratize 3D content creation and open up new possibilities for creative expression and design.

*   **Justification for Score:** The paper demonstrates strong novelty in its approach to hierarchical scene generation and tackles crucial problems related to spatial coherence and occlusion handling with compelling results. While there are limitations, the combination of technical innovations and impactful results warrants a high score.

Score: 8

- **Score**: 8/10

### **[Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo](http://arxiv.org/abs/2504.13139v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces a novel framework for controlled text generation from large language models (LLMs) using sequential Monte Carlo (SMC). The framework addresses the challenge of incorporating syntactic and semantic constraints into LLM generation by framing it as a probabilistic conditioning problem.  It uses SMC to approximate the intractable posterior distribution resulting from these constraints. The key features of the proposed approach include: (1) incorporating both efficient and expensive potential functions (representing different types of constraints) at different stages of the SMC process; (2) using weight correction to mitigate greediness from locally constrained decoding; (3) adaptively resampling particles to focus computation on promising sequences; and (4) emphasizing programmable inference that enables easy specialization for different tasks. The framework is evaluated across four challenging domains (Python code generation, text-to-SQL, goal inference, and molecule synthesis) and shows improved performance compared to baseline methods, including larger models and fine-tuned approaches. The authors also provide empirical evidence that their approach results in better approximations of the target posterior distribution.

**Critical Evaluation:**

*   **Novelty:** The core idea of using SMC for controlled LLM generation isn't entirely new, as the paper itself acknowledges prior work in this area (Lew et al., 2023). However, the specific *architecture* developed *specializing* SMC for semantic parsing, along with its unique combination of techniques, including efficient handling of expensive potentials and adaptive resampling, demonstrates substantial novelty. Furthermore, the emphasis on programmable inference sets it apart from approaches based on learning proposals or twist functions, which require costly domain-specific fine-tuning. The system integrates with a prior language model probabilistic programming language offering a distinct advantage.

*   **Significance:** The paper makes a strong case for the practical significance of the proposed method. The empirical results demonstrate that the SMC framework allows smaller, open-source language models to outperform significantly larger models (including closed-source ones) on challenging tasks. This is particularly impactful as it suggests a path towards more resource-efficient and accessible controlled generation. The demonstration that the performance gains are linked to better approximation of the posterior distribution provides further theoretical grounding. The evaluation spans diverse domains, which increases the generalizability and impact.

*   **Strengths:**

    *   **Strong Empirical Validation:** The paper offers comprehensive experiments and ablation studies across diverse domains, clearly demonstrating the efficacy of the proposed approach.
    *   **Principled Approach:** The framework is grounded in a probabilistic formulation, providing a clear and well-motivated approach to controlled generation.
    *   **Practical Impact:** The results show that the method can make smaller models competitive with larger ones, potentially democratizing access to powerful controlled generation capabilities.
    *   **Open Source Availability:** The availability of the code is a valuable contribution that allows the research community to further explore and build upon the work.
*   **Weaknesses:**

    *   **Computational Cost:** While the paper claims minimal computational overhead, the costs of expensive potentials are domain dependent.
    *   **Increased parameters:** As it builds on prior language models, it may result in parameter bloat and high inference cost and high memory access cost.
    *   **Complexity:** While SMC is presented as an alternative to black-box solutions, implementing and tuning SMC algorithms can still be complex. The "programmable inference" aspect might require substantial expertise.
    *   **Reliance on Grammars:** Although versatile and configurable, this means that if the grammar is not sufficiently descriptive or too restrictive, the overall system performance might be negatively impacted.

*   **Potential Influence:** The paper has the potential to influence several areas of research: (1) The development of more efficient and accessible controlled generation techniques for LLMs. (2) The exploration of probabilistic programming as a tool for building and controlling language models. (3) The design of more robust and adaptable semantic parsing systems. (4) Further investigations on the use of SMC in the context of language generation.

**Justification of Score:**

The paper presents a novel and significant extension of SMC to controlled LLM generation. While the underlying SMC concept is not brand-new, the specialized architecture, emphasis on programmable inference, and comprehensive empirical validation demonstrate a substantial contribution. The practical impact of enabling smaller models to outperform larger ones, coupled with the open-source code release, strengthens the paper's significance. However, the framework does have limitations related to the complexity of implementing SMC and the need for well-defined constraints. While building on prior works, the significant improvements across numerous domains warrants a score of 8.

**Score: 8**

- **Score**: 8/10

### **[Exploring Expert Failures Improves LLM Agent Tuning](http://arxiv.org/abs/2504.13145v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Exploring Expert Failures Improves LLM Agent Tuning":

**Summary:**

The paper addresses the challenge of training Large Language Model (LLM) agents in complex environments where standard Rejection Sampling Fine-Tuning (RFT) can become trapped in local optima due to a lack of diversity in successful expert trajectories. The authors observe that even failed expert trajectories often contain valuable guidance, such as good initial plans or crucial actions. To leverage this information, they propose Exploring Expert Failures (EEF), a method that simulates states within failed expert trajectories, identifies beneficial actions that lead to successful agent behavior from these states, and incorporates these actions into the fine-tuning process. Harmful actions that contribute to failure are explicitly excluded. The approach demonstrates improved performance on WebShop and SciWorld, achieving state-of-the-art results and surpassing the performance of RFT and even GPT-4.

**Critical Evaluation:**

* **Novelty:** The key novelty lies in the observation that failed expert trajectories, typically discarded in RFT, can actually be a valuable source of training data. The idea of mining these failures for beneficial actions and strategically incorporating them is a valuable contribution. This differentiates EEF from approaches like ETO and NAT, which treat entire failed trajectories uniformly.

* **Significance:** The significance stems from the practical improvements demonstrated in complex environments like WebShop and SciWorld. Addressing the problem of diminishing returns with RFT on harder subtasks is a major hurdle in LLM agent training. EEF offers a pathway to overcome this by efficiently leveraging available but previously underutilized data. Reaching state-of-the-art performance, especially in WebShop, is a compelling demonstration of impact. The finding that GPT-3.5 trajectories can enrich a GPT-4-trained agent (EEF GPT-3&4) has both efficiency and learning potential.

* **Strengths:**
    * **Clear Motivation:** The paper clearly articulates the problem with RFT and provides a well-reasoned rationale for exploring expert failures.
    * **Technical Soundness:** The EEF method is technically sound and well-defined, with Algorithm 1 providing a clear outline of the process. The approach taken is sensible, especially the focus on excluding potentially harmful actions.
    * **Strong Empirical Results:** The experiments provide convincing evidence of EEF's effectiveness, with significant improvements over strong baselines across multiple datasets. The ablation studies further validate the importance of navigation skills and the impact of different simulation budgets.
    * **Practical Implications:** The results suggest that EEF can lead to more efficient and effective LLM agent training, especially in challenging environments where data is scarce. The results with GPT-3.5 add to the practical value, as they offer a path to higher performance without relying solely on more expensive GPT-4 trajectories.
* **Weaknesses:**
    * **Parameter Sensitivity:** While the paper defines parameters M (simulations for expert state) and I (iterations of training), more discussion around the sensitivity of performance to these parameters could be beneficial. What's the performance when M=10 or 20?
    * **Scalability on extremely large Datasets:** Could EEF be computationally expensive when dealing with datasets of hundreds of thousands or millions of trajectories?
    * **Action Identification:** The selection of beneficial actions is based on simulation. The accuracy of the action selection is crucial for success. Although mentioned, there could be a larger emphasis on discussion regarding this.
    * **Limited Expert types used:** All studies were based on GPT3.5/GPT4 type models. It would be beneficial to test on Open Source Models to prove universality.
* **Potential Influence:** EEF has the potential to influence the field by shifting the focus from solely relying on successful demonstrations to actively learning from failures. The approach is relatively simple to implement and can be integrated into existing RFT-based training pipelines. It could inspire further research into techniques for extracting valuable information from imperfect data.

**Justification of Score:**

The paper presents a novel and significant contribution to the field of LLM agent training. The key insight – leveraging information from failed expert trajectories – is both intuitive and effective. The technical approach is sound, and the empirical results are compelling. While some aspects, such as parameter sensitivity, action identification, scalability and the types of experts used in tests could benefit from further exploration, the paper addresses a critical challenge and provides a practical solution. Overall, the paper represents a substantial advance in LLM agent training.

Score: 8

- **Score**: 8/10

## Other Papers
### **[Subitizing-Inspired_Large_Language_Models_for_Floorplanning](http://arxiv.org/abs/2504.12076v1)**
### **[Selective Demonstration Retrieval for Improved Implicit Hate Speech Detection](http://arxiv.org/abs/2504.12082v1)**
### **[Reasoning-Based AI for Startup Evaluation (R.A.I.S.E.): A Memory-Augmented, Multi-Step Decision Framework](http://arxiv.org/abs/2504.12090v1)**
### **[Gauging Overprecision in LLMs: An Empirical Study](http://arxiv.org/abs/2504.12098v1)**
### **[Generalized Visual Relation Detection with Diffusion Models](http://arxiv.org/abs/2504.12100v1)**
### **[Entropy-Guided Watermarking for LLMs: A Test-Time Framework for Robust and Traceable Text Generation](http://arxiv.org/abs/2504.12108v1)**
### **[A Diffusion-Based Framework for Terrain-Aware Remote Sensing Image Reconstruction](http://arxiv.org/abs/2504.12112v1)**
### **[Clarifying Ambiguities: on the Role of Ambiguity Types in Prompting Methods for Clarification Generation](http://arxiv.org/abs/2504.12113v1)**
### **[Anti-Aesthetics: Protecting Facial Privacy against Customized Text-to-Image Synthesis](http://arxiv.org/abs/2504.12129v1)**
### **[Multilingual Contextualization of Large Language Models for Document-Level Machine Translation](http://arxiv.org/abs/2504.12140v1)**
### **[Mapping Controversies Using Artificial Intelligence: An Analysis of the Hamas-Israel Conflict on YouTube](http://arxiv.org/abs/2504.12177v1)**
### **[Trusting CHATGPT: how minor tweaks in the prompts lead to major differences in sentiment classification](http://arxiv.org/abs/2504.12180v1)**
### **[SALAD: Improving Robustness and Generalization through Contrastive Learning with Structure-Aware and LLM-Driven Augmented Data](http://arxiv.org/abs/2504.12185v1)**
### **[What Do Large Language Models Know? Tacit Knowledge as a Potential Causal-Explanatory Structure](http://arxiv.org/abs/2504.12187v1)**
### **[d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning](http://arxiv.org/abs/2504.12216v1)**
### **[Coding-Prior Guided Diffusion Network for Video Deblurring](http://arxiv.org/abs/2504.12222v1)**
### **[Watermarking Needs Input Repetition Masking](http://arxiv.org/abs/2504.12229v1)**
### **[MOS: Towards Effective Smart Contract Vulnerability Detection through Mixture-of-Experts Tuning of Large Language Models](http://arxiv.org/abs/2504.12234v1)**
### **[Cobra: Efficient Line Art COlorization with BRoAder References](http://arxiv.org/abs/2504.12240v1)**
### **[SIDME: Self-supervised Image Demoiréing via Masked Encoder-Decoder Reconstruction](http://arxiv.org/abs/2504.12245v1)**
### **[Comparative Evaluation of Radiomics and Deep Learning Models for Disease Detection in Chest Radiography](http://arxiv.org/abs/2504.12249v1)**
### **[DMM: Building a Versatile Image Generation Model via Distillation-Based Model Merging](http://arxiv.org/abs/2504.12364v1)**
### **[Themisto: Jupyter-Based Runtime Benchmark](http://arxiv.org/abs/2504.12365v1)**
### **[InstantCharacter: Personalize Any Characters with a Scalable Diffusion Transformer Framework](http://arxiv.org/abs/2504.12395v1)**
### **[A Human-AI Comparative Analysis of Prompt Sensitivity in LLM-Based Relevance Judgment](http://arxiv.org/abs/2504.12408v1)**
### **[Diffusion Based Robust LiDAR Place Recognition](http://arxiv.org/abs/2504.12412v1)**
### **[Mitigating LLM Hallucinations with Knowledge Graphs: A Case Study](http://arxiv.org/abs/2504.12422v1)**
### **[Don't Just Translate, Agitate: Using Large Language Models as Devil's Advocates for AI Explanations](http://arxiv.org/abs/2504.12424v1)**
### **[PlanGlow: Personalized Study Planning with an Explainable and Controllable LLM-Driven System](http://arxiv.org/abs/2504.12452v1)**
### **[Geometric Generality of Transformer-Based Gröbner Basis Computation](http://arxiv.org/abs/2504.12465v1)**
### **[SLURG: Investigating the Feasibility of Generating Synthetic Online Fallacious Discourse](http://arxiv.org/abs/2504.12466v1)**
### **[Integrating Structural and Semantic Signals in Text-Attributed Graphs with BiGTex](http://arxiv.org/abs/2504.12474v1)**
### **[Accelerating Clinical NLP at Scale with a Hybrid Framework with Reduced GPU Demands: A Case Study in Dementia Identification](http://arxiv.org/abs/2504.12494v1)**
### **[Multimodal LLM Augmented Reasoning for Interpretable Visual Perception Analysis](http://arxiv.org/abs/2504.12511v1)**
### **[Evaluating the Diversity and Quality of LLM Generated Content](http://arxiv.org/abs/2504.12522v1)**
### **[Memorization vs. Reasoning: Updating LLMs with New Knowledge](http://arxiv.org/abs/2504.12523v1)**
### **[Generalization through variance: how noise shapes inductive biases in diffusion models](http://arxiv.org/abs/2504.12532v1)**
### **[Knowledge Acquisition on Mass-shooting Events via LLMs for AI-Driven Justice](http://arxiv.org/abs/2504.12545v1)**
### **[ELAB: Extensive LLM Alignment Benchmark in Persian Language](http://arxiv.org/abs/2504.12553v1)**
### **[Benchmarking LLM-based Relevance Judgment Methods](http://arxiv.org/abs/2504.12558v1)**
### **[CDF-RAG: Causal Dynamic Feedback for Adaptive Retrieval-Augmented Generation](http://arxiv.org/abs/2504.12560v1)**
### **[ZeroSumEval: Scaling LLM Evaluation with Inter-Model Competition](http://arxiv.org/abs/2504.12562v1)**
### **[Prompt-Driven and Training-Free Forgetting Approach and Dataset for Large Language Models](http://arxiv.org/abs/2504.12574v1)**
### **[Identifying and Mitigating the Influence of the Prior Distribution in Large Language Models](http://arxiv.org/abs/2504.12585v1)**
### **[Simplifying Graph Transformers](http://arxiv.org/abs/2504.12588v1)**
### **[GeoSense: Evaluating Identification and Application of Geometric Principles in Multimodal Reasoning](http://arxiv.org/abs/2504.12597v1)**
### **[Code Copycat Conundrum: Demystifying Repetition in LLM-based Code Generation](http://arxiv.org/abs/2504.12608v1)**
### **[Packing Input Frame Context in Next-Frame Prediction Models for Video Generation](http://arxiv.org/abs/2504.12626v1)**
### **[Towards Characterizing Subjectivity of Individuals through Modeling Value Conflicts and Trade-offs](http://arxiv.org/abs/2504.12633v1)**
### **[A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation](http://arxiv.org/abs/2504.12636v1)**
### **[Scaling Instruction-Tuned LLMs to Million-Token Contexts via Hierarchical Synthetic Data Generation](http://arxiv.org/abs/2504.12637v1)**
### **[Persona-judge: Personalized Alignment of Large Language Models via Token-level Self-judgment](http://arxiv.org/abs/2504.12663v1)**
### **[GRAIL: Gradient-Based Adaptive Unlearning for Privacy and Copyright in LLMs](http://arxiv.org/abs/2504.12681v1)**
### **[Data-efficient LLM Fine-tuning for Code Generation](http://arxiv.org/abs/2504.12687v1)**
### **[Why and How LLMs Hallucinate: Connecting the Dots with Subsequence Associations](http://arxiv.org/abs/2504.12691v1)**
### **[Collaborative Perception Datasets for Autonomous Driving: A Review](http://arxiv.org/abs/2504.12696v1)**
### **[SmartFreeEdit: Mask-Free Spatial-Aware Image Editing with Complex Instruction Understanding](http://arxiv.org/abs/2504.12704v1)**
### **[SimUSER: Simulating User Behavior with Large Language Models for Recommender System Evaluation](http://arxiv.org/abs/2504.12722v1)**
### **[Validating LLM-Generated Relevance Labels for Educational Resource Search](http://arxiv.org/abs/2504.12732v1)**
### **[Mask Image Watermarking](http://arxiv.org/abs/2504.12739v1)**
### **[Privacy Protection Against Personalized Text-to-Image Synthesis via Cross-image Consistency Constraints](http://arxiv.org/abs/2504.12747v1)**
### **[Trajectory Adaptation using Large Language Models](http://arxiv.org/abs/2504.12755v1)**
### **[GraphOmni: A Comprehensive and Extendable Benchmark Framework for Large Language Models on Graph-theoretic Tasks](http://arxiv.org/abs/2504.12764v1)**
### **[Enhancing the Geometric Problem-Solving Ability of Multimodal LLMs via Symbolic-Neural Integration](http://arxiv.org/abs/2504.12773v1)**
### **[EarthGPT-X: Enabling MLLMs to Flexibly and Comprehensively Understand Multi-Source Remote Sensing Imagery](http://arxiv.org/abs/2504.12795v1)**
### **[Assesing LLMs in Art Contexts: Critique Generation and Theory of Mind Evaluation](http://arxiv.org/abs/2504.12805v1)**
### **[Saliency-Aware Diffusion Reconstruction for Effective Invisible Watermark Removal](http://arxiv.org/abs/2504.12809v1)**
### **[Image-Editing Specialists: An RLAIF Approach for Diffusion Models](http://arxiv.org/abs/2504.12833v1)**
### **[DashChat: Interactive Authoring of Industrial Dashboard Design Prototypes through Conversation with LLM-Powered Agents](http://arxiv.org/abs/2504.12865v1)**
### **[EmoVoice: LLM-based Emotional Text-To-Speech Model with Freestyle Text Prompting](http://arxiv.org/abs/2504.12867v1)**
### **[Information Gain-Guided Causal Intervention for Autonomous Debiasing Large Language Models](http://arxiv.org/abs/2504.12898v1)**
### **[Benchmarking Multi-National Value Alignment for Large Language Models](http://arxiv.org/abs/2504.12911v1)**
### **[MAIN: Mutual Alignment Is Necessary for instruction tuning](http://arxiv.org/abs/2504.12913v1)**
### **[ConExion: Concept Extraction with Large Language Models](http://arxiv.org/abs/2504.12915v1)**
### **[Exact Learning Dynamics of In-Context Learning in Linear Transformers and Its Application to Non-Linear Transformers](http://arxiv.org/abs/2504.12916v1)**
### **[Explainable AI in Usable Privacy and Security: Challenges and Opportunities](http://arxiv.org/abs/2504.12931v1)**
### **[Customizing Emotional Support: How Do Individuals Construct and Interact With LLM-Powered Chatbots](http://arxiv.org/abs/2504.12943v1)**
### **[Are Retrials All You Need? Enhancing Large Language Model Reasoning Without Verbalized Feedback](http://arxiv.org/abs/2504.12951v1)**
### **[QLLM: Do We Really Need a Mixing Network for Credit Assignment in Multi-Agent Reinforcement Learning?](http://arxiv.org/abs/2504.12961v1)**
### **[Accommodate Knowledge Conflicts in Retrieval-augmented LLMs: Towards Reliable Response Generation in the Wild](http://arxiv.org/abs/2504.12982v1)**
### **[A Virtual Machine for Arbitrary Low-Precision GPGPU Computation in LLM Serving](http://arxiv.org/abs/2504.12984v1)**
### **[Chain-of-Thought Prompting for Out-of-Distribution Samples: A Latent-Variable Study](http://arxiv.org/abs/2504.12991v1)**
### **[SHA256 at SemEval-2025 Task 4: Selective Amnesia -- Constrained Unlearning for Large Language Models via Knowledge Isolation](http://arxiv.org/abs/2504.12996v1)**
### **[ChatEXAONEPath: An Expert-level Multimodal Large Language Model for Histopathology Using Whole Slide Images](http://arxiv.org/abs/2504.13023v1)**
### **[TTRD3: Texture Transfer Residual Denoising Dual Diffusion Model for Remote Sensing Image Super-Resolution](http://arxiv.org/abs/2504.13026v1)**
### **[InstructRAG: Leveraging Retrieval-Augmented Generation on Instruction Graphs for LLM-Based Task Planning](http://arxiv.org/abs/2504.13032v1)**
### **[How Large Language Models Are Changing MOOC Essay Answers: A Comparison of Pre- and Post-LLM Responses](http://arxiv.org/abs/2504.13038v1)**
### **[GraphAttack: Exploiting Representational Blindspots in LLM Safety Mechanisms](http://arxiv.org/abs/2504.13052v1)**
### **[Aspect-Based Summarization with Self-Aspect Retrieval Enhanced Generation](http://arxiv.org/abs/2504.13054v1)**
### **[RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins](http://arxiv.org/abs/2504.13059v1)**
### **[ArtistAuditor: Auditing Artist Style Pirate in Text-to-Image Generation Models](http://arxiv.org/abs/2504.13061v1)**
### **[Accuracy is Not Agreement: Expert-Aligned Evaluation of Crash Narrative Classification Models](http://arxiv.org/abs/2504.13068v1)**
### **[HiScene: Creating Hierarchical 3D Scenes with Isometric View Generation](http://arxiv.org/abs/2504.13072v1)**
### **[SkyReels-V2: Infinite-length Film Generative Model](http://arxiv.org/abs/2504.13074v1)**
### **[EventVAD: Training-Free Event-Aware Video Anomaly Detection](http://arxiv.org/abs/2504.13092v1)**
### **[RF-DETR Object Detection vs YOLOv12 : A Study of Transformer-based and CNN-based Architectures for Single-Class and Multi-Class Greenfruit Detection in Complex Orchard Environments Under Label Ambiguity](http://arxiv.org/abs/2504.13099v1)**
### **[UniEdit-Flow: Unleashing Inversion and Editing in the Era of Flow Models](http://arxiv.org/abs/2504.13109v1)**
### **[VistaDPO: Video Hierarchical Spatial-Temporal Direct Preference Optimization for Large Video Models](http://arxiv.org/abs/2504.13122v1)**
### **[Low-hallucination Synthetic Captions for Large-Scale Vision-Language Model Pre-training](http://arxiv.org/abs/2504.13123v1)**
### **[LLMs Meet Finance: Fine-Tuning Foundation Models for the Open FinLLM Leaderboard](http://arxiv.org/abs/2504.13125v1)**
### **[Energy-Based Reward Models for Robust Language Model Alignment](http://arxiv.org/abs/2504.13134v1)**
### **[Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo](http://arxiv.org/abs/2504.13139v1)**
### **[Exploring Expert Failures Improves LLM Agent Tuning](http://arxiv.org/abs/2504.13145v1)**
### **[Personalized Text-to-Image Generation with Auto-Regressive Models](http://arxiv.org/abs/2504.13162v1)**
### **[Sleep-time Compute: Beyond Inference Scaling at Test-time](http://arxiv.org/abs/2504.13171v1)**
### **[It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization](http://arxiv.org/abs/2504.13173v1)**
