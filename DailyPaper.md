# The Latest Daily Papers - Date: 2025-08-17
## Highlight Papers
### **[REFN: A Reinforcement-Learning-From-Network Framework against 1-day/n-day Exploitations](http://arxiv.org/abs/2508.10701v1)**
- **Summary**: Here's a summary and critical evaluation of the provided research paper:

**Summary:**

The paper introduces REFN, a novel framework that leverages Reinforcement Learning (RL) to train Large Language Models (LLMs) for generating network filters to prevent 1-day/n-day vulnerability exploitations on networked devices. It addresses limitations in existing defenses (host-based patching and network-based filtering) such as scalability, compatibility, and error-prone deployment. REFN employs RL driven by online network rewards (instead of Human Feedback) to ensure scalability and compatibility.  The system utilizes a unified deployment on edge security gateways and provides robustness through online validation using real network traffic. The paper also highlights three core challenges in training LLMs for exploit prevention and addresses them with Agentic-RAG-based Knowledge Distillation, an RL-from-VNF Pipeline, and Online Agentic Validation.  Evaluations across 22 families of exploits show effectiveness, efficiency, and scalability.

**Critical Evaluation:**

*   **Novelty:** The paper proposes a holistic framework for vulnerability mitigation that integrates several components.  The core novelty lies in the combination and orchestration of: 1) leveraging reinforcement learning with real-time network rewards for LLM training instead of human feedback, 2) utilizing VNFs for automated reward generation and validation, and 3) using Agentic-RAG knowledge distillation to inject vulnerability-specific knowledge into the LLM. While individual components like RAG, LLMs, and RL are not novel, their integration in the specific context of network security and automatic exploit prevention exhibits significant novelty. The idea of treating network filtering rule generation as a "language" to be learned via RL is innovative.
*   **Significance:** The significance of this work stems from the increasing threat posed by 1-day/n-day vulnerabilities and the inadequacy of existing mitigation techniques.  The proposed framework addresses practical challenges that hinder the adoption of automated vulnerability mitigation at scale.  The paper contributes a proof-of-concept demonstrating the viability of training LLMs for preventing massive-scale exploits.  The reported improvements in accuracy, efficiency, and scalability are compelling. The focus on Edge Security Gateways as a central point for unified deployment enhances practical relevance.
*   **Strengths:**
    *   **Problem Relevance:** Targets a critical and growing security challenge.
    *   **Holistic Approach:** Integrates diverse techniques for a comprehensive solution.
    *   **Practical Focus:**  Addresses real-world deployment challenges (scalability, compatibility, error reduction).
    *   **Evaluation:** Includes rigorous experiments demonstrating improved performance.
    *   **Clear Contribution:** The three novel designs are clearly defined and well motivated.
*   **Weaknesses:**
    *   **Trust Assumptions:** The framework relies on strong trust assumptions regarding the security of edge security gateways and cloud servers, which may not always hold.
    *   **Encrypted Traffic:** While the paper mentions handling encrypted traffic, the methods described are limited and might not be effective against sophisticated evasion techniques.
    *   **Limited Vulnerability Types:** The evaluation is conducted across a specific set of 22 families of exploits, and the generalizability of the results to other types of vulnerabilities needs further investigation.
    *   **Accessibility of Training Data:** The paper admits that the effectiveness of the LLM is affected by the limited training data (lack of contexts and traces). This is a real-world issue as the training data for vulnerabilities may often be incomplete or inaccurate, hindering the performance of REFN.
    *   **Limited Exploration of LLM-Generated Exploits**: The paper does not elaborate on REFN's potential in proactively identifying unknown vulnerabilities by exploring LLM-generated exploits.
*   **Potential Influence:** The paper could significantly influence the field by:
    *   Inspiring further research on LLM-based network security solutions.
    *   Driving the development of more sophisticated automated vulnerability mitigation techniques.
    *   Encouraging the adoption of edge security gateways as a unified deployment platform.
    *   Motivating the creation of larger and more diverse datasets for training LLMs in security domains.
    *   Paving the way for more proactive security measures that can anticipate and prevent future exploits.

**Justification of Score:**

Despite the identified weaknesses, the strengths of the paper in addressing a critical problem with a novel and well-engineered solution, alongside the compelling evaluation results, justify a relatively high score. The practical focus and emphasis on real-world deployment challenges elevate the significance of the work. While trust assumptions and limitations in handling encrypted traffic are valid concerns, they can be addressed in future research. The framework provides a significant advancement towards automated vulnerability mitigation at scale.

Score: 8.5

- **Score**: 8/10

### **[Exploiting Discriminative Codebook Prior for Autoregressive Image Generation](http://arxiv.org/abs/2508.10719v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper addresses a key limitation in autoregressive image generation systems that use discrete tokenizers. These systems typically tokenize images into sequences of indices using a codebook but don't fully exploit the similarity information contained within the codebook. The authors identify two main problems with the common practice of using k-means clustering to extract this similarity: Token Space Disparity (non-uniform density in the token feature space) and Centroid Distance Inaccuracy (inaccurate distance calculations using cluster centroids in high-dimensional space).

To address these issues, the authors propose a novel method called Discriminative Codebook Prior Extractor (DCPE). DCPE utilizes an agglomerative clustering strategy that iteratively merges the most similar clusters, addressing the token space disparity. It also replaces centroid-based distances with instance-based distance measures, eliminating centroid distance inaccuracy. The authors demonstrate that DCPE can be seamlessly integrated into existing codebook prior-based methods, leading to accelerated training, improved FID scores, and better Inception Scores.

**Critical Evaluation:**

**Novelty:**

The primary novelty of this paper lies in its identification of the Token Space Disparity and Centroid Distance Inaccuracy issues that affect the performance of codebook-based autoregressive image generation.  While the idea of using a codebook prior isn't entirely new (IAR and CTF exist), the *specific* issues that DCPE targets *are* a valuable contribution.  The agglomerative clustering combined with instance-based distance is also a novel solution in this context. The idea to exploit the intrinsic information available on the codebook in order to improve training is simple yet highly effective. The proposal of the DCPE leads to a better utilization of codebook prior, effectively accelerating the training and enhancing the final image quality.

**Significance:**

The significance of the work is twofold:
1.  **Improved Training Efficiency:** DCPE significantly accelerates the training of autoregressive models. The impressive speedup observed in LlamaGen-B training (42%-55%) showcases the practical benefits of the method.
2.  **Improved Image Quality:** By better utilizing the token similarity information, DCPE leads to improvements in image quality, as evidenced by the improved FID and IS scores. This suggests a more faithful representation of the image distribution.
Moreover, the results have a direct impact on the field. DCPE's ease of integration as a plug-and-play module with IAR and CTF suggests that the method has the potential to be widely adopted by other researchers and applied to a wider range of tasks.

**Strengths:**

*   **Clear Problem Definition:** The paper clearly articulates the limitations of existing k-means based methods.
*   **Well-Motivated Solution:** The proposed DCPE method is well-motivated by the identified issues.
*   **Comprehensive Experiments:**  The experimental setup is thorough, including comparisons with state-of-the-art methods, ablation studies, and parameter sensitivity analysis.  The experiments cover various model sizes (LlamaGen-B, L, XL), and demonstrate the generalizability. The experiments and the performance gains are very compelling.
*   **Plug-and-Play Nature:**  DCPE's seamless integration with existing methods enhances its practical applicability.

**Weaknesses:**

*   **Computational Complexity (addressed but worth noting):** The initial naive implementation of DCPE had high computational complexity. The paper mitigates this with an optimized implementation, but the original complexity could have been a deterrent to exploration.
*   **Limited Scope of Application:** While the paper demonstrates improvements on LlamaGen and its integration into IAR and CTF, more examples on other autoregressive image generation models would strengthen the generalization claims.
*   **Random Selection Strategy Limitations in Inference:** The reliance on random selection of tokens within clusters for decoding, while simple, is a potential limitation. More sophisticated decoding strategies that consider token probabilities or other factors could potentially further improve image quality. The authors address this in the Limitation and Future Work section.

**Overall:**

The paper is a strong contribution to the field of autoregressive image generation. It provides a novel and effective approach for utilizing codebook prior information, leading to improved training efficiency and image quality. The paper is well-written, clearly explains the proposed method, and provides convincing experimental results.  While some limitations exist, the strengths outweigh the weaknesses.

**Score: 8**

**Rationale:** The paper presents a significant and practical improvement to existing autoregressive image generation methods. The identification and mitigation of Token Space Disparity and Centroid Distance Inaccuracy is a valuable contribution, and the resulting performance gains are impressive. While it might lack revolutionary groundbreaking implications on the field, the impact is direct, easily integrated into existing architectures, and provides a solid boost in terms of training time and final results. The DCPE algorithm addresses a well-defined problem and offers a tangible improvement in a quickly evolving field.

- **Score**: 8/10

### **[Thinking Inside the Mask: In-Place Prompting in Diffusion LLMs](http://arxiv.org/abs/2508.10736v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces ICE (In-Place Chain-of-Thought Prompting with Early Exit), a novel framework designed to enhance reasoning capabilities and improve inference efficiency in diffusion large language models (dLLMs). Unlike traditional autoregressive LLMs that use prefix-based prompting and sequential generation, ICE leverages the bidirectional attention mechanisms and iterative refinement processes of dLLMs.  ICE integrates in-place prompts directly within masked token positions during iterative refinement and employs a confidence-aware early exit mechanism to reduce computational overhead. The paper demonstrates ICE's effectiveness through extensive experiments, showing accuracy improvements and significant speedups on various reasoning benchmarks like GSM8K, MATH, MMLU, and GPQA.

**Critical Evaluation:**

**Novelty:** The core idea of ICE – integrating prompts directly into the masked token positions within a dLLM framework – exhibits a significant degree of novelty. Traditional CoT prompting has been limited to autoregressive architectures. Transforming reasoning from a pre-processing step into an integrated part of the generation process in dLLMs is a genuinely novel contribution. The confidence-aware early exit mechanism is also a clever adaptation to the iterative nature of dLLMs, exploiting the observation that answers often stabilize before the reasoning process fully completes. This is more than a simple engineering tweak; it showcases an understanding of the internal dynamics of dLLMs and how to leverage them effectively.

**Significance:** The significance of this work is multifaceted:

*   **Improved Reasoning in dLLMs:** The primary contribution is demonstrably enhancing the reasoning capabilities of dLLMs, a model class that has been lagging in this domain compared to autoregressive LLMs despite offering other advantages. This improvement is evidenced by the significant accuracy gains on challenging datasets such as GSM8K.

*   **Efficiency Gains:** The early exit mechanism addresses a significant practical bottleneck for dLLMs: their computational cost. The impressive speedups achieved, while maintaining or even improving accuracy, make dLLMs a much more viable alternative for many applications.

*   **Architectural Alignment:** The paper makes a compelling case for architectural alignment - tailoring prompting strategies to the strengths of the underlying model architecture. It shows that dLLMs can be more effectively utilized when prompting is designed to take advantage of their bidirectional attention mechanisms and iterative refinement.

*   **Insights into dLLM Dynamics:** The analysis of confidence dynamics within dLLMs is also valuable. The observation that answer confidence stabilizes before complete reasoning offers insights into how these models process information and can inform future research on optimization strategies.

**Strengths:**

*   **Well-Defined Problem and Solution:** The paper clearly identifies the limitations of prefix-based prompting in dLLMs and provides a well-defined and theoretically sound solution (ICE).
*   **Comprehensive Experiments:** The experimental evaluation is rigorous and comprehensive, covering a variety of datasets and model architectures.  The ablation studies effectively demonstrate the contribution of each component of ICE. The latency-accuracy trade-off comparisons are particularly convincing.
*   **Reproducibility:** The authors promise to release their code, which is crucial for reproducibility and adoption by the research community.
*   **Clear Presentation:** The paper is well-written and easy to understand, even though the technical details are complex.

**Weaknesses:**

*   **Hyperparameter Sensitivity:** Although discussed, the sensitivity of the results to hyperparameter choices, particularly the confidence threshold, is a potential concern. While the ablation studies address this somewhat, further research could explore more robust and adaptive methods for setting these parameters.
*   **Limited Generalization Claims:** While the paper covers multiple datasets and two dLLM architectures, wider evaluations across more models and tasks would strengthen the generalizability of the findings.
*  **Potential for Overfitting:** The paper presents impressive results, however the specific design of the thinking templates might lead to overfitting to certain datasets or question formats. A more diverse range of templates and a more rigorous exploration of their impact could strengthen the findings.

**Potential Influence:**  ICE has the potential to significantly influence future research in dLLMs by:

*   **Motivating New Prompting Strategies:** Inspiring researchers to explore other in-place prompting techniques tailored to the specific architectures of different models.
*   **Driving Further Optimization:** Encouraging the development of more efficient inference methods for dLLMs.
*   **Informing Model Design:** Providing insights into the internal dynamics of dLLMs that can inform the design of future models.

**Score:** 8.5

**Justification:**  The paper presents a genuinely novel and significant contribution to the field of dLLMs. ICE addresses a critical gap in reasoning capabilities while simultaneously improving efficiency. The comprehensive experimental validation, combined with the architectural insights gained and the potential influence on future research, warrants a high score. The weaknesses related to hyperparameter sensitivity and broader generalization are relatively minor and don't detract significantly from the overall value of the work. The paper is likely to become a foundational piece in the development of more powerful and efficient dLLMs.

- **Score**: 8/10

### **[Video-BLADE: Block-Sparse Attention Meets Step Distillation for Efficient Video Generation](http://arxiv.org/abs/2508.10774v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper "VIDEO-BLADE: BLOCK-SPARSE ATTENTION MEETS STEP DISTILLATION FOR EFFICIENT VIDEO GENERATION" addresses the computational bottleneck in video diffusion transformers by proposing a novel framework called BLADE. BLADE combines Adaptive Block-Sparse Attention (ASA) with a sparsity-aware step distillation paradigm based on Trajectory Distribution Matching (TDM). ASA dynamically generates content-aware sparsity masks to focus computation on salient spatiotemporal features. The sparsity-aware step distillation integrates sparsity directly into the distillation process, leading to faster convergence. The paper demonstrates significant end-to-end inference acceleration on text-to-video models like CogVideoX-5B and Wan2.1-1.3B, while also improving video quality as measured by VBench-2.0 scores and human evaluations.  A key aspect of BLADE is its data-free nature, avoiding the need for expensive video datasets during distillation.

**Critical Evaluation:**

*   **Novelty:** The novelty lies in the synergistic combination of block-sparse attention and step distillation within a data-free joint training framework.  While sparse attention and step distillation are not individually new, the way BLADE integrates them is. Existing approaches either combine them in a training-free, suboptimal manner or require separate training stages, negating the benefits of data-free distillation. ASA also presents some novelty with its dynamic, content-aware block-sparse approach, offering an improvement over static sparsity patterns. The integration of global tokens with the ASA mechanism (ASA_GT) is also a novel contribution.

*   **Significance:** The significance stems from the substantial gains in inference efficiency achieved without sacrificing video quality. The reported 14.10x speedup on Wan2.1-1.3B and 8.89x speedup on CogVideoX-5B are impressive and could potentially lower the barrier to entry for real-world video generation applications. The improvement in video quality, despite the high sparsity, is also noteworthy, suggesting that the framework not only accelerates but also regularizes the model effectively. The ablation studies further support the importance of each component in BLADE. This work reduces the compute costs associated with video generation.

*   **Strengths:**

    *   **Strong Empirical Results:** The paper presents thorough experimental results on multiple models and benchmarks, demonstrating the effectiveness of BLADE.
    *   **Data-Free Approach:** The data-free distillation is a significant advantage, as it avoids the need for large, high-quality video datasets during the distillation process.
    *   **Improved Quality:** The fact that the approach improves perceptual quality, not just speed, is compelling.
    *   **Clear Methodology:** The paper clearly explains the technical details of ASA and the sparsity-aware distillation paradigm.

*   **Weaknesses:**

    *   **Limited to Diffusion Transformers:** The framework is specifically designed for diffusion transformers, which may limit its broader applicability. While this is a leading architecture in video generation, a more general approach would be even more impactful.
    *   **Kernel speedup is sublinear compared to E2E speedup** Attention is not the sole bottleneck anymore and can be futher optimized.

*   **Potential Impact:** BLADE has the potential to significantly influence the field of video generation by enabling more efficient and accessible video creation. The approach could also be adapted to other generative tasks where computational cost is a limiting factor. The attention map analysis showing how BLADE focuses on important regions is also valuable for understanding the inner workings of video diffusion models.

**Justification of Score:**

Considering the novelty, significance, strengths, and weaknesses of the paper, a score of 8 is warranted. The integration of block-sparse attention and step distillation is indeed novel and well-executed, leading to tangible benefits in terms of speed and quality. The data-free aspect is a definite strength. The limitation to diffusion transformers and potential improvements of the speed bottleneck (identified via the sublinear kernel performance) are notable, but not enough to drastically alter the positive overall assessment. The contribution represents a significant advancement within the specific domain of video diffusion models and has a good chance of impacting future research.

Score: 8

- **Score**: 8/10

### **[The Knowledge-Reasoning Dissociation: Fundamental Limitations of LLMs in Clinical Natural Language Inference](http://arxiv.org/abs/2508.10777v1)**
- **Summary**: Here's a summary and critical evaluation of the provided paper:

**Summary:**

The paper investigates the reasoning abilities of Large Language Models (LLMs) in the clinical domain. It introduces a new benchmark, the Clinical Trial Natural Language Inference (CTNLI) dataset, designed to probe specific reasoning skills: Causal Attribution, Compositional Grounding, Epistemic Verification, and Risk State Abstraction. A key feature of CTNLI is the inclusion of Ground Knowledge and Meta-Level Reasoning Verification (GKMRV) probes, which allow the authors to decouple failures in factual recall from failures in reasoning. The study evaluates six LLMs and finds that while the models perform well on GKMRV probes, indicating they possess the necessary knowledge, they perform poorly on the main reasoning tasks. The paper argues that this reveals a fundamental limitation: LLMs often lack the structured, composable internal representations required to reliably apply their knowledge for robust clinical reasoning. They further suggest that LLMs instead rely on shortcut heuristics, resulting in outputs that may appear plausible but lack structural soundness.

**Critical Evaluation:**

*   **Novelty:** The key novelty of this paper lies in its methodology for probing the reasoning capabilities of LLMs. The construction of the CTNLI benchmark with its accompanying GKMRV probes provides a rigorous and measurable way to isolate and identify specific failures in the clinical reasoning process. The idea of decoupling knowledge from its deployment is a valuable contribution, and well executed.
*   **Significance:** The findings of this paper have significant implications for the use of LLMs in high-stakes domains like healthcare. The observed dissociation between knowledge recall and reasoning ability raises serious concerns about the reliability of LLMs for clinical decision support. It highlights the need for further research into the development of more robust and interpretable reasoning mechanisms. The focus on clinical reasoning is both timely and important.
*   **Strengths:**
    *   **Well-defined Research Question:** The paper clearly articulates its research question: whether scaling LLMs alone can lead to structured, generalizable internal representations capable of robust clinical reasoning.
    *   **Rigorous Methodology:** The development of the CTNLI benchmark with GKMRV probes is a significant strength. The design enables systematic evaluation and diagnosis of reasoning failures.
    *   **Comprehensive Evaluation:** The study evaluates several LLMs with different prompting strategies, lending robustness to the findings.
    *   **Clear and Convincing Results:** The observed dissociation between GKMRV performance and main task performance is compelling evidence for the limitations of current LLMs.
    *   **Actionable Insights:** The paper offers valuable suggestions for future research, including the need for neuro-symbolic integration, representation disentanglement, and separation of reasoning from ground knowledge.
*   **Weaknesses:**
    *   **Dataset Size:** While well-designed, the relatively small size of the CTNLI benchmark (ten instances per task per reasoning family) could limit the statistical power of the findings. A larger dataset might provide even stronger support for the conclusions.
    *   **Model Coverage:** While diverse, the models tested are a specific snapshot. Future advancements in LLM architecture might shift the landscape and diminish some of the observations.
    *   **Limited Focus on Specific Heuristics:** While the paper discusses some observed heuristics, further analysis of these heuristics could provide more granular insights into the models' failure modes. Is more depth and focus required?

*   **Impact:** This paper is likely to be influential in shaping the discussion about the trustworthiness of LLMs in high-stakes domains. It provides a framework for evaluating and improving the reasoning capabilities of these models. The benchmark itself could become a valuable resource for the research community. The focus on clinical reasoning as a test case also sets a precedent for more rigorous analyses of LLMs in other specialized domains.

**Justification for Score:**

This is a strong paper that makes a significant contribution to the field. While the dataset is relatively small, its design and the resulting findings are convincing. The paper clearly identifies a critical limitation of current LLMs and offers practical directions for future research. The work is both methodologically rigorous and timely, focusing on an important application domain. While not revolutionary, the paper makes a well-reasoned and well-supported argument that should influence future research on LLMs.

Score: 8

- **Score**: 8/10

### **[SSRL: Self-Search Reinforcement Learning](http://arxiv.org/abs/2508.10874v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces Self-Search Reinforcement Learning (SSRL), a novel approach that leverages Large Language Models (LLMs) as simulators for agentic search tasks. SSRL aims to reduce the dependence on costly interactions with external search engines by utilizing the internal knowledge and reasoning capabilities of LLMs. The paper first quantifies LLMs' intrinsic search capability, termed "Self-Search," through structured prompting and repeated sampling.  Then, SSRL enhances LLMs' Self-Search through format-based and rule-based rewards, allowing models to iteratively refine their knowledge utilization internally. Experimental results demonstrate that SSRL-trained policy models provide a cost-effective and stable environment for search-driven RL training, facilitating sim-to-real transfer. The key findings suggest that LLMs possess significant world knowledge, SSRL can reduce hallucination, and SSRL-trained models integrate seamlessly with external search engines.

**Critical Evaluation:**

*   **Novelty:** The idea of using LLMs as internal simulators to reduce reliance on external search engines is relatively novel. Prior works have explored LLMs for tool use and world models, but SSRL provides a specific framework with reinforcement learning to optimize the internal search process. The introduction of Self-Search as a quantified metric for LLM internal knowledge is a strong contribution. The format-based reward is somewhat incremental, building on existing work on reward shaping for RL.

*   **Significance:** The potential impact of SSRL is significant. Reducing dependence on external APIs can make agentic RL more scalable and cost-effective. The sim-to-real transfer results suggest that skills learned in this simulated environment can generalize to real-world search scenarios, which is a crucial step towards building more robust and practical AI agents. Demonstrating that LLMs can internally refine their knowledge and reduce hallucination is also very important.

*   **Strengths:**

    *   **Clear problem statement:** The paper clearly defines the problem of costly interactions with external search engines in agentic RL.
    *   **Well-defined methodology:** SSRL is presented as a well-defined framework with clear steps for implementation.
    *   **Comprehensive evaluation:** The paper provides extensive experimental results across multiple benchmarks and model families. Ablation studies shed light on the importance of various components of SSRL.
    *   **Sim-to-real transfer:** The sim-to-real transfer results are a major strength, demonstrating the practical applicability of the approach.
    *   **Insightful Findings:** The analysis provides crucial insights on the scaling behavior, advantages, and disadvantages of internal LLM search and knowledge retrieval.

*   **Weaknesses:**

    *   **Incremental Reward:** The format-based reward, though contributing, might be considered incremental, building upon existing reward shaping techniques.
    *   **Hallucination not fully addressed:** While SSRL reduces hallucination, the paper doesn't completely eliminate it. More investigation is needed to address this issue fully. The limited dataset and exceptional difficulty of the excluded BrowseComp benchmark also raises questions regarding the true generality of SSRL.
    *   **Limited Scope on Knowledge Bias:** The model family comparison points out that certain bias might have occurred during the pretraining of current LLMs for certain domain tasks, such as in mathematical reasoning.

*   **Potential Influence:** SSRL has the potential to influence the field of agentic AI by enabling more scalable and cost-effective training of RL agents. It could inspire new research directions in leveraging LLMs as internal simulators and developing techniques for sim-to-real transfer. Furthermore, exploring a method to refine and elicit internal knowledge will be a very valuable asset to the open-source community.

**Justification for Score:**

The paper demonstrates a novel and significant approach (SSRL) to a crucial challenge in agentic RL (costly external interactions). The thorough experimentation and successful sim-to-real transfer results strengthen its validity and impact. While the reward is incremental and hallucinations are not fully resolved, the paper's potential for advancing the field and enabling more scalable RL justifies a good score.

Score: 8

- **Score**: 8/10

## Other Papers
### **[Chem3DLLM: 3D Multimodal Large Language Models for Chemistry](http://arxiv.org/abs/2508.10696v1)**
### **[REFN: A Reinforcement-Learning-From-Network Framework against 1-day/n-day Exploitations](http://arxiv.org/abs/2508.10701v1)**
### **[Probabilistic Forecasting Method for Offshore Wind Farm Cluster under Typhoon Conditions: a Score-Based Conditional Diffusion Model](http://arxiv.org/abs/2508.10705v1)**
### **[CountCluster: Training-Free Object Quantity Guidance with Cross-Attention Map Clustering for Text-to-Image Generation](http://arxiv.org/abs/2508.10710v1)**
### **[NextStep-1: Toward Autoregressive Image Generation with Continuous Tokens at Scale](http://arxiv.org/abs/2508.10711v1)**
### **[Exploiting Discriminative Codebook Prior for Autoregressive Image Generation](http://arxiv.org/abs/2508.10719v1)**
### **[EgoCross: Benchmarking Multimodal Large Language Models for Cross-Domain Egocentric Video Question Answering](http://arxiv.org/abs/2508.10729v1)**
### **[Thinking Inside the Mask: In-Place Prompting in Diffusion LLMs](http://arxiv.org/abs/2508.10736v1)**
### **[Natively Trainable Sparse Attention for Hierarchical Point Cloud Datasets](http://arxiv.org/abs/2508.10758v1)**
### **[Video-BLADE: Block-Sparse Attention Meets Step Distillation for Efficient Video Generation](http://arxiv.org/abs/2508.10774v1)**
### **[The Knowledge-Reasoning Dissociation: Fundamental Limitations of LLMs in Clinical Natural Language Inference](http://arxiv.org/abs/2508.10777v1)**
### **[Object Fidelity Diffusion for Remote Sensing Image Generation](http://arxiv.org/abs/2508.10801v1)**
### **[Memory-Augmented Transformers: A Systematic Review from Neuroscience Principles to Technical Solutions](http://arxiv.org/abs/2508.10824v1)**
### **[Reinforced Language Models for Sequential Decision Making](http://arxiv.org/abs/2508.10839v1)**
### **[Psyche-R1: Towards Reliable Psychological LLMs through Unified Empathy, Expertise, and Reasoning](http://arxiv.org/abs/2508.10848v1)**
### **[Performance of GPT-5 in Brain Tumor MRI Reasoning](http://arxiv.org/abs/2508.10865v1)**
### **[SSRL: Self-Search Reinforcement Learning](http://arxiv.org/abs/2508.10874v1)**
