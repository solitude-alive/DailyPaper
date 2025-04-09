# The Latest Daily Papers - Date: 2025-04-09
## Highlight Papers
### **[Do PhD-level LLMs Truly Grasp Elementary Addition? Probing Rule Learning vs. Memorization in Large Language Models](http://arxiv.org/abs/2504.05262v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper "Do PhD-level LLMs Truly Grasp Elementary Addition? Probing Rule Learning vs. Memorization in Large Language Models" investigates whether Large Language Models (LLMs) actually understand the fundamental principles of addition or if they simply memorize patterns from training data.  Instead of using complex mathematical benchmarks, the authors focus on elementary two-integer addition.  They probe two core properties: commutativity (A + B = B + A) and compositional generalization (using isomorphic symbolic mappings, e.g., 7 -> Y). The study reveals that while LLMs perform well on numerical addition, their performance significantly degrades when using symbolic mappings, and they frequently violate the commutativity property. Providing explicit addition rules also paradoxically degrades performance, suggesting a conflict between externally provided rules and internalized knowledge. The authors conclude that LLMs primarily rely on memorization and pattern matching rather than genuine rule-based reasoning for addition.

**Critical Evaluation:**

* **Novelty:** The paper's novelty lies in its focused and controlled approach to evaluating LLMs' understanding of a seemingly trivial task – elementary addition. Most prior work concentrates on complex mathematical reasoning, which can obscure fundamental issues. The use of symbolic mapping to disrupt pattern matching and reveal underlying weaknesses is a strong and original contribution. Probing commutativity is also a simple yet effective method to expose underlying issues in how LLMs perform addition.
* **Significance:**  The paper has significant implications for how we interpret LLM performance on mathematical tasks. It suggests that high scores on benchmarks might be misleading, reflecting pattern recognition rather than true understanding.  This challenges the assumption that scaling up model size alone will lead to genuine reasoning abilities.  The paper highlights the need for new evaluation methodologies that can distinguish between memorization and true algorithmic comprehension. Showing that explicitly providing rules degrades performance is a crucial finding that indicates a misalignment between model computation and mathematical cognition and therefore warrants more investigation.
* **Strengths:**
    * **Clear and Focused Research Question:** The paper clearly defines its research question and approach.
    * **Controlled Experiments:** The experimental design is well-controlled, isolating the effects of different factors (digit length, symbolic mapping, rule provision).
    * **Comprehensive Evaluation:** The study evaluates a range of state-of-the-art LLMs, strengthening the generality of its findings.
    * **Thoughtful Analysis:** The paper provides a thoughtful discussion of the implications of the results and suggests directions for future research.
    * **Reproducible Methodology:** The methodology appears well-defined and reproducible.
* **Weaknesses:**
    * **Limited Scope:** While focusing on addition is a strength, it also limits the generalizability of the findings to other mathematical operations. One could argue that addition, due to its repetitive nature, is particularly susceptible to pattern matching.
    * **Zero-Shot Setting Limitation:** While the zero-shot setting is useful for establishing baselines, it might not fully reflect the potential of LLMs when fine-tuned or prompted with relevant examples. It would strengthen the paper to explore few-shot performance with more emphasis on symbolic addition.
    * **Explanation of Negative Results:** While providing explicit rules reduces performance, the exact mechanism driving this phenomenon is not fully explained. Is it interference with learned patterns, a failure to properly parse the rules, or something else? More analysis in this area would be valuable.
    * **Over claiming true "understanding"**: The claim that LLMs can achieve a form of true "understanding" is possibly too high a bar. It might have been more effective to frame the results in light of what strategies are used in performing these tasks.
    * **Dataset generation for specific probes.** The probe design for the experiments seems hand crafted but could have included automated techniques.

* **Impact and Influence:** The paper is likely to influence future research on evaluating LLMs' mathematical capabilities. It could inspire the development of new benchmarks and evaluation methods that are more sensitive to rule-based reasoning. It also encourages researchers to consider alternative architectural designs that promote genuine understanding over pattern recognition. The work is already generating discussion within the LLM research community.

**Overall:**

This is a solid paper with a clear message, well-designed experiments, and significant implications for the field. While there are limitations, the strengths outweigh the weaknesses. It addresses a fundamental concern about LLM reasoning and provides valuable insights for future research.

Score: 8

- **Score**: 8/10

### **[Gaussian Mixture Flow Matching Models](http://arxiv.org/abs/2504.05304v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces Gaussian Mixture Flow Matching (GMFlow), a novel generative model that extends diffusion and flow matching techniques by modeling the denoising distribution as a Gaussian Mixture (GM) rather than a single Gaussian. This allows the model to capture more complex, multi-modal velocity distributions in the data.  GMFlow uses a KL divergence loss to train the GM parameters and develops GM-SDE/ODE solvers that leverage analytic denoising distributions for fast and precise few-step sampling. A novel probabilistic guidance scheme is proposed to address the over-saturation issues often encountered with classifier-free guidance (CFG). Experiments demonstrate GMFlow's superior performance compared to flow matching baselines, achieving competitive or state-of-the-art results on benchmarks like ImageNet with significantly fewer sampling steps.

**Critical Evaluation:**

* **Novelty:** The paper's primary novelty lies in the GM parameterization of the flow velocity field. This is a departure from previous work that assumed a unimodal Gaussian distribution. The probabilistic guidance and GM-specific SDE/ODE solvers are also novel components. The link that the framework generalises previous diffusion models is also interesting.

* **Significance:**
    * **Improved Sampling Efficiency:** Achieving high generation quality with significantly fewer sampling steps is a significant contribution, especially for practical applications where computational cost is a concern. The reported improvements in Precision scores with relatively few steps (e.g., 6 steps on ImageNet) are notable.
    * **Mitigation of Over-Saturation:** The probabilistic guidance scheme is a valuable contribution, as it addresses a common issue in CFG-based generation, leading to more visually appealing results with better color fidelity.
    * **Theoretical Foundation:** Providing a rigorous theoretical justification for the GMFlow approach, including the generalization of previous models and the derivation of GM-specific solvers, strengthens the paper's contribution.

* **Strengths:**
    * **Strong Empirical Results:** The experiments are well-designed and comprehensive, providing compelling evidence for GMFlow's superior performance. The use of both a toy dataset (checkerboard) and a challenging benchmark (ImageNet) is commendable.
    * **Clear and Well-Written:** The paper is generally well-written and explains the technical details of GMFlow clearly.
    * **Comprehensive Ablation Studies:** The paper includes ablation studies that analyze the impact of different design choices, such as the number of GM components and the use of sub-steps in the ODE solvers. These studies provide valuable insights into the model's behavior.

* **Weaknesses:**
    * **Pixel-Wise Factorization:** The pixel-wise factorization approach for high-dimensional data, while a practical solution, might limit the model's ability to capture long-range dependencies. The authors acknowledge this limitation and suggest it as a direction for future research.
    * **Complexity:** The addition of Gaussian mixtures increases the parameter count and computational complexity compared to standard diffusion. The trade-off between complexity and improved performance needs to be considered, especially when scaling to very high-resolution images or videos. A thorough runtime analysis might be helpful.
    * **Clarity on Probabilistic Guidance:** While innovative, the mechanics of the proposed probabilistic guidance, especially the derivation of the Gaussian mask and its rationale, could be explained more intuitively.

* **Potential Impact:** The paper has the potential to influence the development of more efficient and robust generative models. The GMFlow approach could be applied to various domains, including image synthesis, video generation, and audio modeling. The proposed probabilistic guidance scheme could also be adopted in other CFG-based models.

**Justification for Score:**

The paper presents a significant advancement in diffusion and flow matching models.  The GM parameterization is a non-trivial extension that demonstrably improves sampling efficiency and addresses the common over-saturation problem in CFG. The theoretical justification and the development of tailored solvers add to the paper's value.  While pixel-wise factorization limits its expressiveness for long-range dependencies and some components of the technique might require some additional clarification, the improvements are substantive and well supported. Therefore, this paper deserves a solid score reflecting the technical soundness, empirical validation, and potential for future research.

Score: 8

- **Score**: 8/10

### **[GraphRAFT: Retrieval Augmented Fine-Tuning for Knowledge Graphs on Graph Databases](http://arxiv.org/abs/2504.05478v1)**
- **Summary**: Here's a summary and critical evaluation of the GraphRAFT paper:

**Summary:**

The paper introduces GraphRAFT, a retrieval-augmented fine-tuning framework for question answering over knowledge graphs (KGs) stored in graph databases. GraphRAFT focuses on generating provably correct Cypher queries using a finetuned LLM to retrieve relevant subgraphs from the graph database.  A key contribution is the ability to work "off-the-shelf" with native graph databases, leveraging the database's query engine for efficient subgraph retrieval. The method involves finetuning an LLM to produce Cypher queries and employing a grounded constrained decoding strategy to ensure syntactical and semantic correctness. The retrieved subgraph is then used by a second finetuned LLM to reason about the answer. The experiments show that GraphRAFT outperforms existing methods on the STaRK-prime and STaRK-mag datasets, achieving significantly better results on standard metrics and demonstrating sample efficiency.

**Critical Evaluation:**

*   **Novelty:** The paper's primary novelty lies in its practical approach to GraphRAG by directly addressing the integration with existing graph databases via Cypher query generation. While other methods might use GraphRAG conceptually, they often remain abstract or require ad hoc retrieval processes, making them unsuitable for real-world deployments with graph DBs. The "grounded constrained decoding" is also a notable contribution, ensuring that generated Cypher queries are not only syntactically valid but also semantically meaningful concerning the specific KG schema.

*   **Significance:** The significance of GraphRAFT stems from bridging the gap between LLM-based reasoning and the structured, query-optimized environment of graph databases. Many real-world KGs are stored in such databases, and GraphRAFT enables a more direct and efficient way to leverage them for question answering.  The significant performance improvements on established benchmarks demonstrate the effectiveness of the approach, particularly highlighting sample efficiency which is essential in real-world application.

*   **Strengths:**

    *   **Practical Integration:** The method is explicitly designed to be compatible with native graph databases and their query engines (Cypher), unlike many other approaches.
    *   **Guaranteed Correctness:** The grounded constrained decoding ensures that the generated queries are executable and semantically valid, reducing errors and improving efficiency.
    *   **Modular Architecture:** The retrieve-and-reason framework is modular and extensible, allowing for easy integration of different components or improvements in the future.
    *   **Strong Empirical Results:**  The paper provides solid experimental results, demonstrating significant improvements over state-of-the-art methods on challenging benchmarks.
    *   **Sample Efficiency:** The method can achieve very good results when trained with a smaller amount of data.

*   **Weaknesses:**

    *   **Dependency on Graph Schema:** The approach relies on having access to the graph schema for the grounded constrained decoding. This might be a limitation when the schema is not readily available or is incomplete.
    *   **LLM Latency:** While the method uses the graph DB for retrieval which may be faster than alternative methods, there is likely a latency overhead associated with using LLMs to generate queries.
    *   **Limited Ablation Studies:**  While the results are strong, more ablation studies could further clarify the contribution of individual components (e.g., the grounded decoding strategy, the finetuning of LLM L2).
    *   **Dataset Specificity:** While the datasets are well-established, the performance might vary on other KG domains with different characteristics (e.g., significantly larger or differently structured KGs).

*   **Potential Influence:** GraphRAFT has the potential to significantly influence the field of GraphRAG by promoting a more practical and efficient approach that is well-suited for real-world applications. Its focus on using native graph databases, combined with its strong performance and sample efficiency, makes it a valuable contribution that can accelerate the development and deployment of KG-based question answering systems. The query generation method is also a key contribution and may be used in different areas beyond KBQA.

**Score: 8**

**Rationale:**

GraphRAFT represents a solid advancement in the field of GraphRAG. The combination of provably correct Cypher query generation, integration with native graph databases, and strong experimental results warrants a high score. While some weaknesses exist, the strengths outweigh them, making this a valuable contribution with considerable practical potential. A score of 8 reflects the paper's noteworthy novelty, significance, and its potential to influence future research and development in GraphRAG.

- **Score**: 8/10

### **[Evaluating the Generalization Capabilities of Large Language Models on Code Reasoning](http://arxiv.org/abs/2504.05518v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper "Evaluating the Generalization Capabilities of Large Language Models on Code Reasoning" investigates how well large language models (LLMs) generalize their code reasoning abilities to different kinds of programs. The authors introduce techniques to generate both in- and out-of-distribution programs, including code from domain-specific languages (DSLs), LLM-generated code, competitive programming solutions (LeetCode), and mutated versions of existing programs. They then evaluate LLM performance using two main experiments: execution prediction (predicting program output) and execution choice (selecting the program the LLM is most confident in and predicting its output). The paper conducts an extensive evaluation across various state-of-the-art LLMs, analyzing their generalization capabilities over time and across different program classes.  The key findings indicate a substantial increase in LLM generalization capability over the past year. Recent models perform almost perfectly, while earlier models seemed to rely on pattern matching. They also find that performance varies depending on the distribution of the programs and that mutation is an effective way to measure generalization. The paper also looks at the ability of LLMs to choose which version (original or mutated) to reason about.

**Critical Evaluation:**

*   **Novelty:** The paper makes several novel contributions. The techniques for generating out-of-distribution code via DSL sampling and mutation are valuable additions to the evaluation toolkit. The introduction of the "reversion" metric provides a new way to understand when models are likely relying on memorization rather than genuine reasoning. Systematically comparing performance on original vs. mutated code is also novel and insightful.
*   **Significance:** The findings have significant implications for understanding the limitations of LLMs on code-related tasks. Demonstrating that performance on standard benchmarks can be misleading due to data contamination and pattern matching is critical. The results highlight the need for more rigorous evaluation methodologies. The observed improvement in generalization with newer models points towards progress, but also underscores the importance of continuous monitoring and evaluation using appropriate benchmarks. The focus on code *reasoning* rather than just generation fills a gap in the literature.  Understanding what constitutes reasoning is itself a complex question, and the paper makes a valuable step forward.
*   **Strengths:** The paper has several strengths.  The experimental methodology is well-designed and thorough. The use of multiple LLMs (including both open- and closed-source models) provides a comprehensive view of the current state of the field. The analysis is rigorous and insightful, leading to clear conclusions about generalization vs. pattern matching. The paper is well-written and clearly communicates complex ideas. The use of both original and mutated code and choice experiments provides strong evidence for claimed abilities.
*   **Weaknesses:** The paper could benefit from deeper investigation into why mutation exposes weaknesses in some models. What specific kinds of mutations are most effective at differentiating between memorization and true reasoning? While line of code serves as a proxy for program complexity, better metrics (e.g., cyclomatic complexity, nesting depth) may provide more nuanced insights. Although diverse models were used, the experiments are limited to list processing and LeetCode problem. It remains an open question whether the conclusions generalize to larger and more complex software systems. The reliance on `pass@1` while justified limits assessment; a richer analysis based on partial success or failure modes could be valuable.
*   **Potential Influence:** The paper is likely to influence the field by highlighting the need for more robust evaluation methods. The proposed techniques for generating out-of-distribution code can be adopted by other researchers to create more challenging benchmarks. The focus on code reasoning will encourage the development of models that go beyond pattern matching.

**Justification for the Score:**

I assign a score of 8. The paper offers significant advancements to our understanding of LLM capabilities in code reasoning and provides a novel methodology for rigorous evaluation, and is well-executed. The main shortcomings are that the analysis of the specific impact of certain mutation operators could have been more detailed and there is no investigation regarding why the best models are indeed better. While the analysis is rigorous and detailed, there may also be limitations from the benchmark datasets used, since the list-processing and LeetCode are of modest complexity compared to realistic code.

**Score: 8**

- **Score**: 8/10

### **[Efficient Reinforcement Finetuning via Adaptive Curriculum Learning](http://arxiv.org/abs/2504.05520v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces ADARFT (Adaptive Curriculum Reinforcement Finetuning), a novel method to improve the efficiency and accuracy of reinforcement finetuning (RFT) for large language models (LLMs), especially in mathematical reasoning. ADARFT dynamically adjusts the difficulty of training problems based on the model's recent reward signals. This ensures the model is consistently challenged without being overwhelmed, avoiding wasted computation on trivial or unsolvable tasks. The method requires a lightweight extension to standard RFT algorithms like Proximal Policy Optimization (PPO) and doesn't modify the reward function or model architecture. Experiments on competition-level math datasets show that ADARFT reduces training steps and improves accuracy.

**Critical Evaluation:**

*   **Novelty:**  The core idea of using adaptive curriculum learning within the RFT framework is novel. While curriculum learning itself is a well-established concept, its application specifically to RFT for LLMs, especially in the context of structured reasoning problems like mathematics, appears to be a valuable contribution. The method's simplicity, requiring only a wrapper around existing RFT algorithms, enhances its practical appeal. The approach has some similarity with prior works like LIMR (Li et al., 2025) which also use a learning impact measurement. However, unlike LIMR that requires a full training run upfront and is model dependent, ADARFT adaptively selects the training problems based on current model performance, is significantly more lightweight, and does not require a costly upfront calculation.

*   **Significance:** The paper addresses a key challenge in RFT – its sample and computational inefficiency. The reported improvements in training efficiency (up to 2x speedup) and accuracy are significant and could make RFT more scalable for complex reasoning tasks. The fact that the method performs particularly well in imbalanced data regimes is also important, as real-world datasets often exhibit such imbalances. The experiments are fairly comprehensive, covering multiple data distributions, model sizes, and diverse math problems.

*   **Strengths:**

    *   Clear problem statement and well-motivated solution.
    *   Simple and easy-to-implement algorithm.
    *   Significant performance improvements in terms of efficiency and accuracy.
    *   Thorough experimental evaluation across multiple datasets and models.
    *   Addresses a practical challenge in the field of LLM finetuning.

*   **Weaknesses:**

    *   The hyperparameter tuning for the curriculum update mechanism (η, α, β) might require careful adjustment for different tasks and model architectures. While the paper provides some intuition, it would benefit from a more detailed analysis of hyperparameter sensitivity.
    *   The reliance on accurate difficulty estimation is a potential bottleneck. While the paper proposes a reasonable approach using a pre-trained model, the quality of difficulty scores directly affects the performance of ADARFT.
    *  The method primarily focuses on mathematical reasoning tasks. While this domain is relevant, it would be beneficial to demonstrate the effectiveness of ADARFT on other RFT tasks, such as code generation.

*   **Potential Impact:** The paper has the potential to influence the development of more efficient and effective RFT methods for LLMs. Its simplicity and strong empirical results make it a promising direction for future research. It is likely that other researchers will build upon this work by exploring different difficulty estimation techniques, adaptive curriculum schedules, and applications to broader range of tasks.

**Score: 8**

**Rationale:** ADARFT presents a novel and significant contribution by integrating adaptive curriculum learning into RFT. The reported improvements in efficiency and accuracy are substantial and the algorithm is relatively simple to implement. While the method has some limitations regarding hyperparameter tuning and difficulty estimation, its strengths outweigh its weaknesses, making it a valuable addition to the field. The potential for impact is high, as ADARFT offers a practical and scalable approach to improving RFT for complex reasoning tasks.

- **Score**: 8/10

### **[User Feedback Alignment for LLM-powered Exploration in Large-scale Recommendation Systems](http://arxiv.org/abs/2504.05522v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper addresses the challenge of balancing novelty and relevance in LLM-powered recommendation systems, specifically in the context of user interest exploration.  The core idea is to decouple novelty generation and preference alignment. The authors propose a novel approach involving two specialized LLMs: one focused on generating novel interest clusters, and another focused on aligning those clusters with user preferences based on aggregated implicit feedback (clicks, dwell time, etc.). The system uses inference-time scaling, generating multiple novel recommendations and then using the alignment model to select the best-of-n. The authors conduct live experiments on a commercial short-form video platform, demonstrating significant gains in user satisfaction (watch activity, active users) and exploration diversity compared to existing methods.

**Critical Evaluation:**

**Strengths:**

*   **Practical Problem and Application:** The paper tackles a crucial challenge in real-world recommendation systems: exploration vs. exploitation.  The live experiments on a large-scale platform validate the practical applicability of the approach.
*   **Novelty of the Approach:** The decomposed LLM architecture (novelty LLM + alignment LLM) is a compelling design choice.  Decoupling the objectives of novelty generation and preference alignment allows for independent optimization and avoids the catastrophic forgetting issues observed when trying to fine-tune a single LLM for both. Inference-time scaling with a separate alignment model provides a useful way to improve ranking of novel suggestions.
*   **Rigorous Evaluation:** The live A/B testing provides solid evidence for the effectiveness of the proposed method. The use of metrics like positive playback rate, completion rate, and unique engaged user-cluster pairs demonstrates the impact on both user satisfaction and exploration. The comparison to several production models is also a strong point.
*   **Clarity:** The paper is well-written and clearly explains the problem, the proposed solution, and the experimental setup. The limitations of previous approaches (e.g., RLHF) are also well articulated.

**Weaknesses:**

*   **Implicit Feedback Limitations:** While aggregating implicit feedback is a reasonable approach, it is inherently noisy and potentially biased. The paper mentions post-processing steps to address this, but a deeper analysis of the impact of noise and bias on the alignment model would be beneficial.
*   **Limited Generalizability Discussion:** The experiments are performed on a specific short-form video platform.  While the core ideas are general, a discussion of the potential challenges and adaptations required for applying the approach to other domains (e.g., e-commerce, music streaming) would strengthen the paper.
*   **Technical Detail Level:** While the paper does well in explaining the architecture and model training, the paper doesn't go into the technical details or explain how the various models are implemented, or the exact objective functions used.

**Significance:**

The paper offers a significant contribution to the field by demonstrating a practical and effective way to integrate LLMs into large-scale recommendation systems while addressing the important problem of balancing novelty and relevance. The decomposed architecture and inference-time scaling strategy are valuable insights that can be adapted and extended in future research. The live experiment results provide strong evidence for the real-world impact of the approach.

**Score:** 8

**Rationale:**

The paper presents a novel and well-evaluated approach to a significant problem in recommendation systems. The decomposed LLM architecture is a key contribution, and the live experiment results are compelling. However, the limitations related to implicit feedback and generalizability warrant a slight deduction. Also, a greater level of technical detail would have helped boost the score. While valuable, the paper builds upon existing hierarchical planning paradigms, slightly diminishing its novelty compared to truly groundbreaking work. Therefore, a score of 8 reflects the paper's solid contributions and practical impact while acknowledging its limitations.

- **Score**: 8/10

### **[DEL: Context-Aware Dynamic Exit Layer for Efficient Self-Speculative Decoding](http://arxiv.org/abs/2504.05598v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces DEL, a "plug-and-play" module designed to enhance the efficiency of speculative decoding (SD) in large language models (LLMs), specifically focusing on the LayerSkip method. DEL dynamically selects both the exit layer and speculation length during inference, adapting to the specific context of the sequence being generated. DEL achieves this by tracking token acceptance rates at different layers using cached hidden states, estimating a "Token-per-Layer" (TPL) metric to guide its choices. A dynamic draft-exiting mechanism is also employed, adjusting the speculation length based on confidence scores of individual draft tokens. Experiments demonstrate that DEL improves speed over auto-regressive decoding and outperforms existing SD techniques across various models, tasks, and model sizes.

**Critical Evaluation:**

**Novelty:**  The paper's novelty lies in its adaptive and context-aware approach to exit layer and speculation length selection within a self-speculative decoding framework. Prior works have primarily focused on static configurations or dynamic adjustments based on less granular information. Dynamically adjusting the exit layer is a key improvement. The use of cached hidden states to estimate acceptance rates and dynamically adjusting confidence threshold are also incremental but valuable innovations. The combination of these techniques within a plug-and-play module further adds to the novelty.

**Significance:**  The significance of DEL stems from its ability to improve the efficiency of LLM inference without sacrificing output quality.  The results indicate substantial speedups, suggesting DEL could have a practical impact on deploying LLMs, especially in resource-constrained environments or latency-sensitive applications. The plug-and-play nature of DEL makes it relatively easy to integrate into existing LayerSkip pipelines, increasing its potential for adoption.

**Strengths:**

*   **Adaptive Approach:**  The dynamic selection of exit layer and speculation length addresses a key limitation of static or coarsely tuned SD methods.
*   **Context-Awareness:**  The use of token acceptance rates and confidence scores allows DEL to adapt to the specific characteristics of the input sequence.
*   **Plug-and-Play Design:** The modularity facilitates integration with existing pipelines.
*   **Strong Empirical Results:**  The experiments demonstrate consistent speedups across a range of models and tasks.
*   **Thorough Analysis:** The ablation study provides insights into the contributions of different components of DEL.
*   **Lightweight implementation**: the paper demonstrates only a small amount of runtime and memory overhead, making the method promising in practical use case.

**Weaknesses:**

*   **Incremental Improvements:** While DEL combines multiple techniques, each individual component can be seen as an incremental advancement upon existing methods. However, the combination creates a strong method.
*   **Dependency on LayerSkip:** DEL is specifically designed to work with LayerSkip, limiting its direct applicability to other SD approaches that do not use early exit.
*   **Limited Exploration of Different TPL estimations**: While TPL is a reasonable estimate of decoding speed, the paper could discuss the possibility of the method working with different estimation methods.
*   **Heuristic-Based Design:**  The selection of exit layer and speculation length relies on heuristics derived from the TPL metric. A more formal optimization approach might lead to further improvements.

**Potential Influence:**

DEL has the potential to influence the direction of research in speculative decoding, particularly regarding adaptive and context-aware approaches. Its plug-and-play nature could facilitate its adoption in practical applications, potentially leading to wider deployment of more efficient LLM inference.

**Justification for Score:**

DEL represents a significant engineering contribution to self-speculative decoding, with a context-aware method that dynamically adjusts the exit layer to further improve decoding speed without sacrificing output quality. While its components are incremental improvements on existing techniques, the combination within a plug-and-play module and the demonstrated empirical gains justify a relatively high score. The dependency on LayerSkip and the heuristic-based design are factors that slightly reduce the rating.

Score: 8

- **Score**: 8/10

### **[ShadowCoT: Cognitive Hijacking for Stealthy Reasoning Backdoors in LLMs](http://arxiv.org/abs/2504.05605v1)**
- **Summary**: The paper "ShadowCoT: Cognitive Hijacking for Stealthy Reasoning Backdoors in LLMs" introduces a novel framework for backdoor attacks on large language models (LLMs) that utilize Chain-of-Thought (CoT) reasoning. Unlike previous backdoor attacks that focus on manipulating input tokens or prompts, ShadowCoT targets the internal reasoning mechanism of LLMs, aiming to hijack multi-step reasoning chains and generate logically coherent but adversarial outputs. The method involves a multi-stage injection pipeline, where attention pathways are selectively rewired and intermediate representations are perturbed based on internal reasoning states. Reinforcement learning and reasoning chain pollution are used to synthesize stealthy adversarial CoTs that are difficult to detect. Experimental results across various reasoning benchmarks and LLMs demonstrate the effectiveness of ShadowCoT in achieving high attack success rates while preserving benign performance.

**Critical Evaluation:**

The paper presents a significant and novel contribution to the field of LLM security by shifting the focus from surface-level manipulations to the deeper cognitive processes within the model. Targeting the reasoning chain itself, rather than just input or output tokens, is a key advancement.

**Strengths:**

*   **Novelty:** The core idea of cognitively aligned adversarial reasoning is innovative. The paper effectively argues that existing methods lack direct intervention into the model's reasoning dynamics, making ShadowCoT a unique approach. The method integrates attention head localization, multi-stage backdoor injection, and reasoning chain pollution effectively.
*   **Comprehensive Approach:** The framework is well-designed with a detailed methodology, including trigger design, attention head localization, adversarial chain construction, multi-stage backdoor injection, and reasoning chain pollution. Each step is carefully explained and justified.
*   **Experimental Validation:** Extensive experiments across diverse reasoning benchmarks (ProofNet, GSM8K, AQUA-RAT, StrategyQA) and LLM architectures (LLaMA-2-7B, Falcon-7B, Mistral-7B, DeepSeek-R1-Distill-Qwen-1.5B) provide solid evidence for the effectiveness and generalizability of the approach. The use of various metrics (ASR, HSR, PPL, AODR) adds to the credibility of the results.
*   **Stealthiness:** The analysis of the attack's stealthiness, including perplexity measurements and a comparative evaluation against existing defenses, is a significant contribution. The paper demonstrates that ShadowCoT can effectively evade existing detection mechanisms.
*   **Ablation Study:** The ablation study provides insights into the contribution of each component in ShadowCoT, particularly highlighting the importance of the RSC and CABA modules.
*   **Strong Ethical Considerations:** The authors explicitly acknowledge ethical considerations and emphasize the defensive nature of the research. They refrain from releasing potentially harmful artifacts like triggers or checkpoints.
*   **Clear Presentation:** The paper is well-written, clearly structured, and easy to follow. The figures and tables effectively illustrate the proposed method and experimental results.

**Weaknesses:**

*   **Complexity:** While the comprehensive approach is a strength, it also results in a fairly complex framework, which may make it more challenging to implement and adapt in practice.
*   **Limited Defense Strategies:** While the paper shows the attack's effectiveness against some existing defenses, it does not deeply explore potential novel defense strategies that might be specifically tailored to counter ShadowCoT's unique vulnerabilities.
*   **Limited Scalability Discussion:** While impressive, the computational resources required to implement the full ShadowCoT pipeline on even larger models might be substantial, which could be a barrier to broader adoption.

**Significance:**

The paper's significance lies in identifying a crucial vulnerability in CoT-enhanced LLMs and proposing a novel attack framework that directly targets the internal reasoning mechanisms. This work highlights the need for more sophisticated defense strategies that go beyond surface-level checks and can effectively model and monitor the deep semantics of adversarial reasoning paths. The discovery that stronger reasoning models are more susceptible to this attack vector due to more faithful adherence to corrupted logic also contributes to a better understanding of LLM vulnerabilities. The cross-task transferability analysis also raises concerns about potential generalization of backdoors.

**Justification for Score:**

The paper's significant novelty in attacking the reasoning chain itself, well-designed framework, extensive experimental validation, and careful analysis of stealthiness warrant a high score. The limitations discussed above prevent it from achieving the highest possible score. However, the potential impact on the field is significant, as it uncovers a critical vulnerability and challenges existing security paradigms.

Score: 8

- **Score**: 8/10

### **[Sugar-Coated Poison: Benign Generation Unlocks LLM Jailbreaking](http://arxiv.org/abs/2504.05652v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces a novel jailbreak attack method, "Sugar-Coated Poison" (SCP), against Large Language Models (LLMs). SCP exploits a discovered vulnerability termed "Defense Threshold Decay" (DTD), where LLMs become more susceptible to jailbreak attacks after generating a substantial amount of benign content. SCP works in two phases: first, it prompts the model to generate benign content; second, it uses adversarial reasoning to steer the model towards generating malicious content. The authors demonstrate SCP's effectiveness across various models and datasets, achieving state-of-the-art attack success rates. They also propose a defense strategy, POSD (Part-of-Speech Defense), to mitigate such attacks.

**Critical Evaluation:**

*   **Novelty:** The paper introduces a novel attack strategy (SCP) and a corresponding vulnerability (DTD) in LLMs. The core idea of leveraging benign content to weaken defenses before launching the jailbreak is innovative. The detailed analysis of attention mechanisms to understand and exploit DTD strengthens the novelty. The POSD defence also introduces a novel approach to mitigate jailbreak by focusing on sentence structure.

*   **Significance:**  The paper's findings are significant because they highlight a weakness in current LLM safety mechanisms. Discovering DTD reveals how the very process of generating seemingly harmless content can unintentionally pave the way for malicious outputs, a subtle and dangerous vulnerability. SCP's high success rates underscore the importance of addressing this vulnerability. Furthermore, the POSD defense provides a potential avenue for improving LLM robustness, although its effectiveness may vary across different models and scenarios.

*   **Strengths:**
    *   **Thorough analysis:** The paper provides a detailed analysis of the attention mechanisms in LLMs to explain the DTD vulnerability.
    *   **Strong empirical results:**  The experimental results demonstrate the effectiveness of SCP across various LLMs.
    *   **Comprehensive comparison:** The authors compare SCP with a diverse set of existing attack methods.
    *   **Defense Strategy:**  The authors not only identify a vulnerability but also propose a potential defense (POSD).
    *   **Clear presentation:** The paper is well-written and easy to understand.

*   **Weaknesses:**
    *   **Limited scope of defense:** The defense strategy (POSD) may not be universally effective against all types of jailbreak attacks and could potentially affect model generalization (although the paper claims this is minimal).
    *   **Potential bias in evaluation:** GPT-4 is used as the evaluator, but it is a black-box model, and there is a chance that its evaluation can be biased as a result.

*   **Impact:**  This paper has the potential to significantly influence the field of LLM security. By exposing the DTD vulnerability and introducing the SCP attack, it prompts further research on improving the robustness of LLMs against jailbreak attacks.

*   **Justification for the Score:**

I am assigning a score of **8**. The paper presents a novel and significant contribution to the field. The discovery of DTD and its exploitation via SCP is well-researched and rigorously demonstrated through experiments.  The paper clearly explains the underlying mechanisms and offers a potential solution, further enhancing its value. The only area for some critique is on the generality of the POSD defence and the black-box evaluation that could slightly alter the results. Considering the strengths of the paper in highlighting the fragility of LLM defences, and the significant advancement it makes in understanding LLM defences, a score of 8 is justified.

Score: 8

- **Score**: 8/10

### **[TARO: Timestep-Adaptive Representation Alignment with Onset-Aware Conditioning for Synchronized Video-to-Audio Synthesis](http://arxiv.org/abs/2504.05684v1)**
- **Summary**: Here's a summary and critical evaluation of the provided research paper:

**Summary:**

The paper introduces TARO (Timestep-Adaptive Representation Alignment with Onset-Aware Conditioning), a novel framework for synthesizing high-fidelity and temporally coherent audio from video. Built upon flow-based transformers, TARO features two main innovations: (1) Timestep-Adaptive Representation Alignment (TRA), which dynamically adjusts the alignment strength of latent representations based on the noise schedule, and (2) Onset-Aware Conditioning (OAC), which integrates onset cues to enhance synchronization with dynamic visual events. Experiments on VGGSound and Landscape datasets show that TARO outperforms prior methods in terms of audio quality and synchronization precision.

**Critical Evaluation:**

*   **Novelty:** The paper presents a combination of existing techniques (flow-based transformers, representation alignment, onset detection) but integrates them in a novel and effective way for video-to-audio synthesis. The adaptive nature of representation alignment (TRA) based on the noise schedule is a key innovation. OAC, while leveraging existing onset detection models, provides a more explicit mechanism for aligning audio with dynamic visual cues, addressing a limitation of previous approaches. The use of pretrained audio encoders for TRA is also a novel application in this domain.
*   **Significance:** Video-to-audio synthesis is a challenging problem with significant applications. By addressing the limitations of existing methods in balancing fidelity, synchronization, and efficiency, TARO makes a valuable contribution. The improved audio quality and synchronization precision, demonstrated through comprehensive experiments, represent a notable advancement.
*   **Strengths:**
    *   **Clear problem definition:** The paper clearly identifies the limitations of existing video-to-audio synthesis methods.
    *   **Novel approach:** TARO introduces innovative components (TRA and OAC) that effectively address these limitations.
    *   **Comprehensive experiments:** The paper presents extensive experimental results on multiple datasets, demonstrating the superiority of TARO over state-of-the-art methods using both objective and subjective metrics. Ablation studies clearly highlight the impact of each component.
    *   **Good results:** The results of the experiments demonstrates good performance, with improvements on a number of metrics.
    *   **Well-written and clear:** The paper is generally well-written and easy to understand. The figures and tables are informative and effectively illustrate the proposed approach.
*   **Weaknesses:**
    *   **Reliance on pre-trained models:** TARO relies on pre-trained models (visual encoder, audio encoder, onset detection model), which may limit its generalizability to datasets with different characteristics. The choice and quality of these pre-trained models directly impact the performance of TARO.
    *   **Zero-shot performance:** While the zero-shot evaluation on the Landscape dataset is promising, further analysis is needed to understand the limitations of TARO's generalization ability.
    *   **Computational cost:** While described as efficient the paper might benefit from a more rigorous discussion and comparison regarding the computational complexity of TARO and other models.
    *   **Modest parameter count:** The paper emphasizes the relatively small parameter size. While this is beneficial it is only modest in comparison to much larger networks.

**Justification for Score:**

TARO presents a novel and effective approach to video-to-audio synthesis, addressing key limitations of existing methods and achieving state-of-the-art performance on established benchmarks. The adaptive representation alignment and onset-aware conditioning are valuable contributions. While there is a reliance on pretrained models and some questions on the zero-shot performance and the limitations of parameter count may be considered, the extensive experiments and clear presentation justify a strong score. The paper has the potential to influence future research in this area and facilitate practical applications of video-to-audio synthesis.

**Score: 8**

- **Score**: 8/10

### **[SEVERE++: Evaluating Benchmark Sensitivity in Generalization of Video Representation Learning](http://arxiv.org/abs/2504.05706v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "SEVERE++: Evaluating Benchmark Sensitivity in Generalization of Video Representation Learning":

**Summary:**

The paper addresses the limitations of current video self-supervised learning (SSL) benchmarks, which often fail to adequately assess the generalization capabilities of learned representations. It introduces SEVERE++, an extended benchmark that evaluates video SSL models across four key downstream factors: domain shift, sample efficiency, action granularity, and task diversity. The authors benchmark 12 transformer-based methods (video-only and video-text) against 10 CNN-based methods, conducting over 1100 experiments across 8 datasets and 7 downstream tasks. The analysis reveals that transformer-based models, despite architectural advancements, remain sensitive to downstream conditions. No single method consistently generalizes across all factors, highlighting the need for more robust and transferable video models. The code will be made publicly available.

**Critical Evaluation:**

* **Novelty:** The paper's primary novelty lies in its comprehensive and systematic evaluation of modern video SSL methods across a wide range of downstream factors, building upon the authors' previous work.  The expansion to transformer-based methods, the inclusion of temporal action localization, and the detailed cross-model analysis significantly advance the understanding of generalization in video SSL. The identified sensitivities are *not entirely unexpected* given the known issues with generalization in machine learning; however, the *depth and breadth* of the experiments, along with the clear articulation of the specific factors, is valuable.

* **Significance:** The work is significant for several reasons:
    * **Highlighting Limitations:** It exposes the limitations of relying solely on standard action recognition benchmarks like Kinetics-400 and UCF-101. It shows how these benchmarks can be misleading in assessing the true generalization ability of video SSL models.
    * **Guiding Future Research:** The SEVERE++ benchmark provides a valuable tool for the community to evaluate and compare new video SSL approaches more effectively. It establishes a more rigorous and comprehensive evaluation protocol. The detailed analysis suggests specific directions for future research, such as developing methods more robust to domain shift, optimized for fine-grained action recognition, and effective with limited data.
    * **Comparing CNNs and Transformers in SSL:**  The comparison between CNN-based and transformer-based architectures provides valuable insights. The observation that CNNs can still outperform transformers in certain scenarios (e.g., fine-grained action recognition or few-shot learning) is particularly important. It challenges the uncritical adoption of transformers without fully considering their strengths and weaknesses in different contexts.

* **Strengths:**
    * **Comprehensive Evaluation:** The sheer scale of the experiments and the diversity of the datasets and tasks is a major strength.
    * **Well-Defined Factors:** The downstream factors are clearly defined and provide a useful framework for analyzing generalization.
    * **Clear Presentation:** The results are presented clearly and concisely, with insightful observations and interpretations.  The updated SEVERE++ benchmark is well-motivated and provides a practical tool for the research community.
    * **Reproducibility:** The authors have committed to releasing their code, which enhances the reproducibility and impact of the work.

* **Weaknesses:**
    * **Limited Architectural Study:**  While the paper compares CNNs and transformers, it uses a fixed R(2+1)D-18 backbone for CNNs and ViT-B for transformers.  A more detailed architectural study, including different transformer sizes or variations, would strengthen the findings.
    * **Dataset Bias:** Although the benchmark is diverse, all datasets still inherently contain some biases, and the definition of "domain shift" is relative to the chosen datasets. More diverse or carefully curated datasets could further improve the benchmark.
    * **Pre-training Dataset Differences:** The models for transfer learning use slightly different base dataset as the starting point of their models. While the work does an analysis across all datasets, ideally, all model parameters should come from the same pre-training source.
    * **Lack of Deeper Explanation:**  While the paper meticulously identifies performance differences across methods and settings, deeper mechanistic explanations for *why* specific models generalize better in certain scenarios are limited. Investigating feature representations and learning dynamics could further enhance the study.

* **Overall Impact:** The paper provides a significant service to the video SSL community by rigorously evaluating and highlighting the limitations of current benchmarks and models. The SEVERE++ benchmark is poised to become a valuable resource for future research.

**Justification of Score:**

Despite some limitations, the paper represents a significant contribution to the field of video self-supervised learning.  The comprehensive evaluation, the clear articulation of key challenges, and the introduction of a practical benchmark justify a high score. While the observed sensitivities aren't entirely surprising in isolation, the depth of analysis and the specific insights gained from the large-scale experiments are valuable. This makes the assessment a more thorough and rigorous analysis of generalization than other ad-hoc video self-supervised assessments.

Score: 8

- **Score**: 8/10

### **[DDT: Decoupled Diffusion Transformer](http://arxiv.org/abs/2504.05741v1)**
- **Summary**: Here's a summary and critical evaluation of the "DDT: Decoupled Diffusion Transformer" paper:

**Summary:**

The paper introduces a Decoupled Diffusion Transformer (DDT) architecture for image generation, aiming to address the optimization dilemma in existing diffusion transformers. The core idea is to decouple the encoding of low-frequency semantic information from the decoding of high-frequency details.  DDT achieves this using a dedicated condition encoder to extract semantic self-conditions and a separate velocity decoder.  The authors claim that DDT enables faster training convergence and improved performance compared to existing diffusion transformer models.  They further propose a statistical dynamic programming approach for sharing self-condition information between adjacent denoising steps to enhance inference speed.  The paper presents results on ImageNet 256x256 and 512x512 datasets, demonstrating state-of-the-art FID scores and improved training efficiency.

**Critical Evaluation:**

*   **Novelty:** The idea of decoupling the encoder and decoder in a diffusion transformer is relatively novel in the context of diffusion models. While encoder-decoder architectures are common in other areas of computer vision, their application to diffusion transformers to specifically address the optimization trade-off between semantic encoding and detail generation is a good contribution. The statistical dynamic programming approach for self-condition sharing is also a worthwhile addition that leads to faster convergence.

*   **Significance:** The reported results are significant. Achieving state-of-the-art FID scores on ImageNet, particularly with notably faster training, would have a substantial impact on the diffusion modeling community. The speedup in training is especially valuable, as it reduces the computational cost associated with large-scale diffusion model training. The demonstrated improvement could enable faster experimentation and development of new diffusion-based applications. Also, the speedup for inference makes diffusion models more practical in the real world.

*   **Strengths:**
    *   Clear problem definition: The paper clearly identifies the optimization dilemma in standard diffusion transformers.
    *   Well-motivated design: The decoupled architecture is logically motivated by the problem definition.
    *   Strong empirical results: The paper presents strong experimental results, demonstrating improved FID scores and training speed.
    *   Practical acceleration: The inference speedup via self-condition sharing is a valuable contribution.
    *   The ablation studies offer some limited insight to DDT's performance.

*   **Weaknesses:**
    *   Incremental Improvements: Several components such as, "Recent architectural improvements such as SwiGLU, ROPE, and RMSNorm have been ex-tensively validated in the research community", are incorporated from pre-existing methods. However, DDT appears to achieve a better result, so there's still good value.

*   **Justification for Score:**

The paper presents a clear, well-motivated approach to a relevant problem in diffusion modeling. The DDT architecture and the self-condition sharing strategy represent a valuable contribution. The most appealing aspect of the paper is that both training and inference are sped up. While the techniques they use build upon existing works, the results are highly persuasive.

Therefore, a score of 8 is justified as a solid, original, and valuable contribution to the field.

**Score: 8**

- **Score**: 8/10

### **[MDK12-Bench: A Multi-Discipline Benchmark for Evaluating Reasoning in Multimodal Large Language Models](http://arxiv.org/abs/2504.05782v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces MDK12-Bench, a new multi-disciplinary benchmark designed to evaluate the reasoning capabilities of Multimodal Large Language Models (MLLMs) using real-world K-12 examination questions. The benchmark covers six disciplines (math, physics, chemistry, biology, geography, and information science) and includes 140K instances with fine-grained knowledge point annotations, detailed answer explanations, and difficulty labels.  Furthermore, the paper presents a dynamic evaluation framework that mitigates data contamination issues by bootstrapping question forms, types, and image styles during evaluation. Extensive experiments on MDK12-Bench show the limitations of current MLLMs in multimodal reasoning and provide insights for developing next-generation models.

**Critical Evaluation:**

* **Novelty:** The paper presents a valuable contribution to the field by introducing a large-scale, multi-disciplinary K-12 benchmark for evaluating MLLMs. While existing benchmarks exist, MDK12-Bench distinguishes itself through its breadth of disciplines, detailed knowledge structure annotations, and the incorporation of a dynamic evaluation framework. The dynamic evaluation, while not entirely new in concept, is applied in a novel way within the context of a K-12 benchmark to address data contamination, which is a significant problem in the field.  The level of granularity in the annotations (instance-level knowledge points linked to a knowledge tree) is a strength that enables more fine-grained error analysis.

* **Significance:** The K-12 domain is well-suited for evaluating fundamental reasoning skills across a variety of subjects, mimicking a crucial stage in human cognitive development. By leveraging real-world examination questions, the benchmark assesses the model's ability to solve practical, knowledge-intensive problems. The inclusion of a dynamic evaluation framework enhances the benchmark's reliability and fairness, which are critical for accurately comparing different models.  The analysis of MLLM performance on the benchmark sheds light on the weaknesses of current models and highlights specific areas (e.g., contextual understanding, resistance to perturbation) that require further research. This benchmark can accelerate the improvement of multi-modal reasoning capabilities.

* **Strengths:**
    *   **Large Scale and Multi-Disciplinarity:**  The size and scope of MDK12-Bench make it a more challenging and comprehensive evaluation resource than many existing benchmarks.
    *   **Structured Knowledge Representation:** The detailed annotations and knowledge structure provide a strong foundation for in-depth analysis of model performance.
    *   **Dynamic Evaluation:** The bootstrapping approach helps mitigate data contamination and ensures the benchmark remains relevant as models evolve.
    *   **Clear Problem Definition:**  The focus on K-12 examination questions provides a concrete and well-defined problem space for evaluating reasoning skills.
    *   **Comprehensive Leaderboard:**  The extensive experiments and detailed analysis on various MLLMs provides valuable insights and a strong baseline for future research.

*   **Weaknesses:**
    * The description of the bootstrapping strategies could be improved with more concrete examples.
    * The reliance on GPT to judge the answer matching and adaptation has its limitations, as even advanced LLMs are not perfect judges of nuanced reasoning.
    *  The paper states that Gemini2 thinking excels in biology and chemistry, but the paper does not define the specifics that makes them perform well in these areas.
    * The improvement of models could have been enhanced by using the data generated from the study to create a model of their own that excelled in these tests.

* **Potential Influence:** MDK12-Bench has the potential to become a widely used benchmark for evaluating and advancing MLLMs in the field of AI. The benchmark's structure and the dynamic evaluation framework offer valuable tools for researchers to develop more robust and reliable models. By focusing on fundamental reasoning skills across a range of disciplines, MDK12-Bench can contribute to progress towards more general-purpose AI. The insights gained from using this benchmark can guide future research in areas such as knowledge representation, reasoning algorithms, and multi-modal integration.

**Score: 8**

**Justification:** MDK12-Bench is a significant contribution to the MLLM evaluation landscape. It offers a substantial improvement in terms of scale, coverage, annotation quality, and data contamination mitigation compared to many existing benchmarks. While some aspects of the methodology, especially the reliance on LLMs for answer judgment, could be further refined, the overall impact of the paper is substantial. The benchmark's potential to drive research and development in the field of MLLMs, as well as its innovative dynamic evaluation framework, warrant a high score.

- **Score**: 8/10

### **[PaMi-VDPO: Mitigating Video Hallucinations by Prompt-Aware Multi-Instance Video Preference Learning](http://arxiv.org/abs/2504.05810v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "PaMi-VDPO: Mitigating Video Hallucinations by Prompt-Aware Multi-Instance Video Preference Learning":

**Summary:**

The paper addresses the problem of video hallucinations in Video Multimodal Large Language Models (VLLMs). It proposes a new framework called Prompt-aware Multi-instance Learning Video DPO (PaMi-VDPO) to mitigate these hallucinations. Unlike existing Direct Preference Optimization (DPO) methods, PaMi-VDPO performs online preference learning by leveraging video augmentations to generate rejected samples while keeping responses fixed, thereby avoiding the need for manual preference annotations. The key innovation is to construct a candidate set of augmented clips and select the most prompt-aware and distinct one through a close-to-far selection strategy, preventing false rejections and improving alignment between video content and generated responses. The authors demonstrate that PaMi-VDPO improves performance on video hallucination benchmarks without additional parameters or architectural changes, while maintaining stable general performance.

**Critical Evaluation:**

**Strengths:**

*   **Novelty:** The paper presents a novel approach to address video hallucinations in VLLMs, differing from standard DPO which relies on human/LLM-annotated preferences. The online video augmentation and the prompt-aware multi-instance learning framework are innovative contributions. The core idea of selecting augmentations that are semantically similar yet prompt-aware is well reasoned.

*   **Technical Soundness:** The proposed PaMi-VDPO framework is technically sound.  The close-to-far selection strategy is well-defined, and the use of Jensen-Shannon divergence for measuring output divergence provides a justifiable approach.  The equations are clear and the implementation details are reasonably described.

*   **Empirical Validation:**  The paper includes comprehensive experiments and ablation studies to validate the effectiveness of PaMi-VDPO. The authors compare their method against state-of-the-art VLLMs and demonstrate significant performance improvements on hallucination benchmarks. The ablation studies systematically investigate the impact of different components, providing insights into the framework's behavior. The evaluation utilizes established benchmarks and metrics.

*   **Practical Significance:**  PaMi-VDPO is practical as it seamlessly integrates into existing VLLMs without requiring architectural modifications or pre-constructed preference data, making it efficient and scalable. The demonstrated improvements, particularly the surpassing of GPT-4o on a hallucination benchmark using only 10K SFT data, make the method quite relevant.

*   **Clear Presentation:** The paper is generally well-written and clearly structured. The problem is well-motivated, the proposed method is explained in detail, and the experimental results are presented clearly with insightful discussions. The figures are helpful in visualizing the key concepts.

**Weaknesses:**

*   **Limited Diversity of Augmentations:** While the augmentations explored cover basic temporal and visual aspects (cropping, shuffling, reversing etc.), the method might benefit from exploring more sophisticated or semantically richer augmentations. It is unclear if the proposed algorithm is limited by the specific set of augmentations studied.

*   **Dependency on LLM for "Prompt Awareness":** The "prompt awareness" hinges on the LLM's (i.e., LLAVA-OV-7B) understanding and responses to different augmentations. While JS divergence is used, relying on the LLM's output might inherit its own biases or hallucination tendencies, potentially affecting the selection of the best augmentation. This dependency, and its potential effects, could be further investigated.

*   **Potential for Overfitting:** The method focuses on the video segment for preference learning. This might potentially lead to overfitting, and therefore, further studies should be done on whether PaMi-VDPO would be able to generalize to unseen scenarios where the dataset characteristics deviate.

*   **Limited analysis into failure cases**: The qualitative analysis could have been improved by showing examples where the proposed method fails and explaining the potential reasons.

**Significance:**

This paper provides a valuable contribution to the field of video understanding and multimodal learning. The PaMi-VDPO framework offers a promising approach for mitigating video hallucinations and improving the reliability of VLLMs. The online preference learning strategy and prompt-aware multi-instance learning formulation have the potential to influence future research in this area.

**Justification for Score:**

I assign a score of **8**.  PaMi-VDPO presents a novel, technically sound, and empirically validated approach to addressing video hallucinations, which is a crucial problem in the field. The avoidance of manual/LLM annotation for creating preference data is a significant strength, improving efficiency and adaptability. While the reliance on an LLM for "prompt awareness" introduces a dependency and the limited diversity of augmentations are weaknesses, the overall contribution is substantial.  The method's ability to outperform GPT-4o (according to the paper's claims) on a key benchmark is particularly noteworthy and signifies that the algorithm addresses a key deficiency in current vision-language models. The contribution should drive future exploration in this area.

Score: 8

- **Score**: 8/10

### **[GOLLuM: Gaussian Process Optimized LLMs -- Reframing LLM Finetuning through Bayesian Optimization](http://arxiv.org/abs/2504.06265v1)**
- **Summary**: Here is a summary and critical evaluation of the paper "GOLLuM: Gaussian Process Optimized LLMs – Reframing LLM Finetuning through Bayesian Optimization."

**Summary:**

The paper introduces GOLLuM, a novel framework that integrates Large Language Models (LLMs) with Gaussian Processes (GPs) for sample-efficient optimization, particularly in chemistry. The key idea is to reframe LLM finetuning as Gaussian Process marginal likelihood optimization, using LLM-based deep kernels. This approach jointly optimizes the LLM and the GP, enabling the LLM's embeddings to adapt to the structure imposed by the GP kernel, leading to a contrastive learning effect. The paper demonstrates the effectiveness of GOLLuM on a variety of chemistry benchmarks, showing improvements in discovery rate, sample efficiency, and generalization compared to static LLM embeddings, domain-specific representations, and disjoint finetuning approaches. The authors also provide insights into the factors that enable successful high-dimensional Bayesian optimization.

**Critical Evaluation:**

*   **Novelty:** The paper presents a genuinely novel architecture.  Integrating LLMs and GPs in this way, using the GP marginal likelihood as a finetuning objective for the LLM, is a significant departure from existing approaches. While individual components like deep kernel learning or using LLMs for BO have been explored, their seamless integration and joint optimization for inducing contrastive structure is innovative. Furthermore, the explicit formalization demonstrating the latent-space reorganization occurring naturally through the training objective is novel.

*   **Significance:** The potential impact on sample-efficient optimization in fields like chemistry is significant. The ability to reduce the number of expensive experiments through better-calibrated uncertainty estimates is important. The findings concerning the importance of aligning the representation space with the GP's inductive bias offer valuable insights for designing effective Bayesian optimization strategies in high-dimensional spaces. The framework's general applicability, demonstrated by strong performance across a wide range of chemistry benchmarks and LLM architectures, boosts the significance. However, the focus is largely on chemistry; while the methodology is claimed to be general, more extensive evaluation in other domains would strengthen the claim.

*   **Strengths:**

    *   **Sound Methodology:**  The GOLLuM framework is well-defined, and the optimization process is clearly described.
    *   **Comprehensive Evaluation:**  The paper provides extensive empirical results across numerous benchmarks, LLM architectures, and hyperparameter settings. The use of multiple chemistry datasets and the comparison to several baselines support the claims.
    *   **Valuable Insights:** The analysis of representation factors and their influence on BO success is valuable. The identification of normalized smoothness as a key metric for assessing representation quality is interesting.
    *   **Clear Presentation:** The paper is well-written and structured. The visual aids (figures, tables, pseudocodes) assist in understanding the approach and results.
    *   **Code and Data Availability:** This enhances reproducibility and facilitates future research.

*   **Weaknesses:**

    *   **Limited Theoretical Depth:** While the empirical results are strong, the theoretical analysis of why GOLLuM works is somewhat limited. More rigorous theoretical justification for the contrastive learning effect, beyond the intuitive explanation, would be beneficial.

    *   **Domain Specificity:** The primary focus is on chemistry. Although the authors state the framework's general applicability, demonstrating it in other complex optimization domains (e.g., materials science, engineering design) would significantly strengthen the claim.

    *   **Computational Complexity:**  Finetuning LLMs, even with PEFT, can be computationally expensive. The paper doesn't address this practically. Also, the marginal likelihood computation and gradient backpropagation through the joint LLM-GP model can be complex. A more detailed discussion of the computational requirements and potential scaling challenges would be beneficial.

    *   **Reproducibility beyond chemistry-specialized baselines:** BO in chemistry is usually compared against standard GP with chemistry-specialized kernels like DRFP. Given the paper’s emphasis on LLM generalization outside chemistry-specialized representations, the experimental design could have also included comparisons against, for example, a BO baseline with molecular fingerprints in the property prediction experiments.

    *   **Lack of ablation of the architectural components:** while the paper conducts experiments on several PEFT setups and architectural differences, it would have been interesting to see what component enables the gains more (e.g. experiments that include and exclude LORA layers).

*   **Potential Influence:** The paper has the potential to influence research in several areas:

    *   **Bayesian Optimization:**  It provides a new approach to incorporate LLMs into BO, addressing the limitations of existing methods.
    *   **LLM Finetuning:** It reframes LLM finetuning as a GP marginal likelihood optimization problem, offering a new perspective on task adaptation.
    *   **Chemical Informatics:** It demonstrates the effectiveness of combining LLMs and GPs for sample-efficient chemical optimization.
    *   **Deep Kernel Learning:** It shows how LLMs can be used as deep kernels in GPs, opening up new possibilities for combining neural networks and probabilistic models.

**Justification for Score:**

Considering the novelty, significance, strengths, and weaknesses, I assign a score of 8.

The paper presents a genuinely novel and potentially influential framework. The strengths lie in its sound methodology, comprehensive evaluation, and valuable insights. The weaknesses are primarily the limited theoretical depth and the domain-centric focus. While more theoretical analysis and broader applicability would have boosted the score, the current contribution is still an exceptional advance that significantly extends the field.

Score: 8

- **Score**: 8/10

## Other Papers
### **[BRIDGES: Bridging Graph Modality and Large Language Models within EDA Tasks](http://arxiv.org/abs/2504.05180v1)**
### **[Concise Reasoning via Reinforcement Learning](http://arxiv.org/abs/2504.05185v1)**
### **[P2Mark: Plug-and-play Parameter-intrinsic Watermarking for Neural Speech Generation](http://arxiv.org/abs/2504.05197v1)**
### **[Quantum Program Linting with LLMs: Emerging Results from a Comparative Study](http://arxiv.org/abs/2504.05204v1)**
### **[Post-Training Language Models for Continual Relation Extraction](http://arxiv.org/abs/2504.05214v1)**
### **[Unleashing the Power of LLMs in Dense Retrieval with Query Likelihood Modeling](http://arxiv.org/abs/2504.05216v1)**
### **[Leveraging LLMs for Utility-Focused Annotation: Reducing Manual Effort for Retrieval and RAG](http://arxiv.org/abs/2504.05220v2)**
### **[LLM-based Automated Grading with Human-in-the-Loop](http://arxiv.org/abs/2504.05239v1)**
### **[Explaining Low Perception Model Competency with High-Competency Counterfactuals](http://arxiv.org/abs/2504.05254v1)**
### **[Learning to Reason Over Time: Timeline Self-Reflection for Improved Temporal Reasoning in Language Models](http://arxiv.org/abs/2504.05258v1)**
### **[Do PhD-level LLMs Truly Grasp Elementary Addition? Probing Rule Learning vs. Memorization in Large Language Models](http://arxiv.org/abs/2504.05262v1)**
### **[Enhancing LLM-Based Short Answer Grading with Retrieval-Augmented Generation](http://arxiv.org/abs/2504.05276v1)**
### **[The challenge of uncertainty quantification of large language models in medicine](http://arxiv.org/abs/2504.05278v1)**
### **[Truthful or Fabricated? Using Causal Attribution to Mitigate Reward Hacking in Explanations](http://arxiv.org/abs/2504.05294v1)**
### **[One-Minute Video Generation with Test-Time Training](http://arxiv.org/abs/2504.05298v1)**
### **[Dimension-Free Convergence of Diffusion Models for Approximate Gaussian Mixtures](http://arxiv.org/abs/2504.05300v1)**
### **[Gaussian Mixture Flow Matching Models](http://arxiv.org/abs/2504.05304v1)**
### **[EduPlanner: LLM-Based Multi-Agent Systems for Customized and Intelligent Instructional Design](http://arxiv.org/abs/2504.05370v1)**
### **[TRATSS: Transformer-Based Task Scheduling System for Autonomous Vehicles](http://arxiv.org/abs/2504.05407v1)**
### **[Less but Better: Parameter-Efficient Fine-Tuning of Large Language Models for Personality Detection](http://arxiv.org/abs/2504.05411v1)**
### **[EP-Diffuser: An Efficient Diffusion Model for Traffic Scene Generation and Prediction via Polynomial Representations](http://arxiv.org/abs/2504.05422v1)**
### **[Generative Adversarial Networks with Limited Data: A Survey and Benchmarking](http://arxiv.org/abs/2504.05456v1)**
### **[Studying Image Diffusion Features for Zero-Shot Video Object Segmentation](http://arxiv.org/abs/2504.05468v1)**
### **[GraphRAFT: Retrieval Augmented Fine-Tuning for Knowledge Graphs on Graph Databases](http://arxiv.org/abs/2504.05478v1)**
### **[REEF: Relevance-Aware and Efficient LLM Adapter for Video Understanding](http://arxiv.org/abs/2504.05491v1)**
### **[A Survey on Hypothesis Generation for Scientific Discovery in the Era of Large Language Models](http://arxiv.org/abs/2504.05496v1)**
### **[Prism: Dynamic and Flexible Benchmarking of LLMs Code Generation with Monte Carlo Tree Search](http://arxiv.org/abs/2504.05500v1)**
### **[Evaluating the Generalization Capabilities of Large Language Models on Code Reasoning](http://arxiv.org/abs/2504.05518v1)**
### **[Efficient Reinforcement Finetuning via Adaptive Curriculum Learning](http://arxiv.org/abs/2504.05520v1)**
### **[User Feedback Alignment for LLM-powered Exploration in Large-scale Recommendation Systems](http://arxiv.org/abs/2504.05522v1)**
### **[Pretraining Language Models for Diachronic Linguistic Change Discovery](http://arxiv.org/abs/2504.05523v1)**
### **[Bridging Industrial Expertise and XR with LLM-Powered Conversational Agents](http://arxiv.org/abs/2504.05527v1)**
### **[COIG-P: A High-Quality and Large-Scale Chinese Preference Dataset for Alignment with Human Values](http://arxiv.org/abs/2504.05535v1)**
### **[Caption Anything in Video: Fine-grained Object-centric Captioning via Spatiotemporal Multimodal Prompting](http://arxiv.org/abs/2504.05541v1)**
### **[SciSciGPT: Advancing Human-AI Collaboration in the Science of Science](http://arxiv.org/abs/2504.05559v1)**
### **[From Fairness to Truthfulness: Rethinking Data Valuation Design](http://arxiv.org/abs/2504.05563v1)**
### **[Can Large Language Models Match Tutoring System Adaptivity? A Benchmarking Study](http://arxiv.org/abs/2504.05570v1)**
### **[Knowledge-Instruct: Effective Continual Pre-training from Limited Data using Instructions](http://arxiv.org/abs/2504.05571v1)**
### **[Tuning-Free Image Editing with Fidelity and Editability via Unified Latent Diffusion Model](http://arxiv.org/abs/2504.05594v1)**
### **[DEL: Context-Aware Dynamic Exit Layer for Efficient Self-Speculative Decoding](http://arxiv.org/abs/2504.05598v1)**
### **[Skywork R1V: Pioneering Multimodal Reasoning with Chain-of-Thought](http://arxiv.org/abs/2504.05599v1)**
### **[On the Impact of Language Nuances on Sentiment Analysis with Large Language Models: Paraphrasing, Sarcasm, and Emojis](http://arxiv.org/abs/2504.05603v1)**
### **[ShadowCoT: Cognitive Hijacking for Stealthy Reasoning Backdoors in LLMs](http://arxiv.org/abs/2504.05605v1)**
### **[FactGuard: Leveraging Multi-Agent Systems to Generate Answerable and Unanswerable Questions for Enhanced Long-Context LLM Extraction](http://arxiv.org/abs/2504.05607v1)**
### **[Two Intermediate Translations Are Better Than One: Fine-tuning LLMs for Document-level Translation Refinement](http://arxiv.org/abs/2504.05614v1)**
### **[Model-Agnostic Policy Explanations with Large Language Models](http://arxiv.org/abs/2504.05625v1)**
### **[TAGC: Optimizing Gradient Communication in Distributed Transformer Training](http://arxiv.org/abs/2504.05638v1)**
### **[Leveraging Prompt-Tuning for Bengali Grammatical Error Explanation Using Large Language Models](http://arxiv.org/abs/2504.05642v1)**
### **[Sugar-Coated Poison: Benign Generation Unlocks LLM Jailbreaking](http://arxiv.org/abs/2504.05652v1)**
### **[Reconstruction-Free Anomaly Detection with Diffusion Models via Direct Latent Likelihood Evaluation](http://arxiv.org/abs/2504.05662v1)**
### **[VC-LLM: Automated Advertisement Video Creation from Raw Footage using Multi-modal LLMs](http://arxiv.org/abs/2504.05673v1)**
### **[Towards Smarter Hiring: Are Zero-Shot and Few-Shot Pre-trained LLMs Ready for HR Spoken Interview Transcript Analysis?](http://arxiv.org/abs/2504.05683v1)**
### **[TARO: Timestep-Adaptive Representation Alignment with Onset-Aware Conditioning for Synchronized Video-to-Audio Synthesis](http://arxiv.org/abs/2504.05684v1)**
### **[Separator Injection Attack: Uncovering Dialogue Biases in Large Language Models Caused by Role Separators](http://arxiv.org/abs/2504.05689v1)**
### **[StayLTC: A Cost-Effective Multimodal Framework for Hospital Length of Stay Forecasting](http://arxiv.org/abs/2504.05691v1)**
### **[STRIVE: A Think & Improve Approach with Iterative Refinement for Enhancing Question Quality Estimation](http://arxiv.org/abs/2504.05693v1)**
### **[Large Language Models Enhanced Hyperbolic Space Recommender Systems](http://arxiv.org/abs/2504.05694v1)**
### **[SEVERE++: Evaluating Benchmark Sensitivity in Generalization of Video Representation Learning](http://arxiv.org/abs/2504.05706v1)**
### **[Automated Archival Descriptions with Federated Intelligence of LLMs](http://arxiv.org/abs/2504.05711v1)**
### **[Single-Agent vs. Multi-Agent LLM Strategies for Automated Student Reflection Assessment](http://arxiv.org/abs/2504.05716v1)**
### **[QEMesh: Employing A Quadric Error Metrics-Based Representation for Mesh Generation](http://arxiv.org/abs/2504.05720v1)**
### **[Unified Generative Search and Recommendation](http://arxiv.org/abs/2504.05730v1)**
### **[Retrieval Augmented Generation with Collaborative Filtering for Personalized Text Generation](http://arxiv.org/abs/2504.05731v1)**
### **[LLM$\times$MapReduce-V2: Entropy-Driven Convolutional Test-Time Scaling for Generating Long-Form Articles from Extremely Long Resources](http://arxiv.org/abs/2504.05732v1)**
### **[Rank-Then-Score: Enhancing Large Language Models for Automated Essay Scoring](http://arxiv.org/abs/2504.05736v1)**
### **[DDT: Decoupled Diffusion Transformer](http://arxiv.org/abs/2504.05741v1)**
### **[SEA-LION: Southeast Asian Languages in One Network](http://arxiv.org/abs/2504.05747v1)**
### **[Transferable Mask Transformer: Cross-domain Semantic Segmentation with Region-adaptive Transferability Estimation](http://arxiv.org/abs/2504.05774v1)**
### **[MDK12-Bench: A Multi-Discipline Benchmark for Evaluating Reasoning in Multimodal Large Language Models](http://arxiv.org/abs/2504.05782v1)**
### **[How to Enable LLM with 3D Capacity? A Survey of Spatial Reasoning in LLM](http://arxiv.org/abs/2504.05786v1)**
### **[Storybooth: Training-free Multi-Subject Consistency for Improved Visual Storytelling](http://arxiv.org/abs/2504.05800v1)**
### **[StealthRank: LLM Ranking Manipulation via Stealthy Prompt Optimization](http://arxiv.org/abs/2504.05804v1)**
### **[PaMi-VDPO: Mitigating Video Hallucinations by Prompt-Aware Multi-Instance Video Preference Learning](http://arxiv.org/abs/2504.05810v1)**
### **[Right Question is Already Half the Answer: Fully Unsupervised LLM Reasoning Incentivization](http://arxiv.org/abs/2504.05812v1)**
### **[Parasite: A Steganography-based Backdoor Attack Framework for Diffusion Models](http://arxiv.org/abs/2504.05815v1)**
### **[Leveraging Robust Optimization for LLM Alignment under Distribution Shifts](http://arxiv.org/abs/2504.05831v1)**
### **[Mind the Trojan Horse: Image Prompt Adapter Enabling Scalable and Deceptive Jailbreaking](http://arxiv.org/abs/2504.05838v1)**
### **[PathGPT: Leveraging Large Language Models for Personalized Route Generation](http://arxiv.org/abs/2504.05846v1)**
### **[On the Importance of Conditioning for Privacy-Preserving Data Augmentation](http://arxiv.org/abs/2504.05849v1)**
### **[Physics-aware generative models for turbulent fluid flows through energy-consistent stochastic interpolants](http://arxiv.org/abs/2504.05852v1)**
### **[Enhancing Coreference Resolution with Pretrained Language Models: Bridging the Gap Between Syntax and Semantics](http://arxiv.org/abs/2504.05855v1)**
### **[Agent Guide: A Simple Agent Behavioral Watermarking Framework](http://arxiv.org/abs/2504.05871v1)**
### **[HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference](http://arxiv.org/abs/2504.05897v1)**
### **[Assessing Thai Dialect Performance in LLMs with Automatic Benchmarks and Human Evaluation](http://arxiv.org/abs/2504.05898v1)**
### **[PRIMEDrive-CoT: A Precognitive Chain-of-Thought Framework for Uncertainty-Aware Object Interaction in Driving Scene Scenario](http://arxiv.org/abs/2504.05908v1)**
### **[CKGAN: Training Generative Adversarial Networks Using Characteristic Kernel Integral Probability Metrics](http://arxiv.org/abs/2504.05945v1)**
### **[Unsupervised Location Mapping for Narrative Corpora](http://arxiv.org/abs/2504.05954v1)**
### **[Diffusion Based Ambiguous Image Segmentation](http://arxiv.org/abs/2504.05977v1)**
### **[An Empirical Study of GPT-4o Image Generation Capabilities](http://arxiv.org/abs/2504.05979v1)**
### **[NativQA Framework: Enabling LLMs with Native, Local, and Everyday Knowledge](http://arxiv.org/abs/2504.05995v1)**
### **[Optuna vs Code Llama: Are LLMs a New Paradigm for Hyperparameter Tuning?](http://arxiv.org/abs/2504.06006v1)**
### **[Llama-3-Nanda-10B-Chat: An Open Generative Large Language Model for Hindi](http://arxiv.org/abs/2504.06011v1)**
### **[CamContextI2V: Context-aware Controllable Video Generation](http://arxiv.org/abs/2504.06022v1)**
### **[OSDM-MReg: Multimodal Image Registration based One Step Diffusion Model](http://arxiv.org/abs/2504.06027v1)**
### **[Multi-Sense Embeddings for Language Models and Knowledge Distillation](http://arxiv.org/abs/2504.06036v1)**
### **[Explainable AI for building energy retrofitting under data scarcity](http://arxiv.org/abs/2504.06055v1)**
### **[Knowledge Graph Completion with Relation-Aware Anchor Enhancement](http://arxiv.org/abs/2504.06129v1)**
### **[QGen Studio: An Adaptive Question-Answer Generation, Training and Evaluation Platform](http://arxiv.org/abs/2504.06136v1)**
### **[A Training-Free Style-aligned Image Generation with Scale-wise Autoregressive Model](http://arxiv.org/abs/2504.06144v1)**
### **[V-MAGE: A Game Evaluation Framework for Assessing Visual-Centric Capabilities in Multimodal Large Language Models](http://arxiv.org/abs/2504.06148v1)**
### **[Navigating the Rabbit Hole: Emergent Biases in LLM-Generated Attack Narratives Targeting Mental Health Groups](http://arxiv.org/abs/2504.06160v1)**
### **[Assessing how hyperparameters impact Large Language Models' sarcasm detection performance](http://arxiv.org/abs/2504.06166v1)**
### **[TxGemma: Efficient and Agentic LLMs for Therapeutics](http://arxiv.org/abs/2504.06196v1)**
### **[From 128K to 4M: Efficient Training of Ultra-Long Context Large Language Models](http://arxiv.org/abs/2504.06214v1)**
### **[Encoder-Decoder Gemma: Improving the Quality-Efficiency Trade-Off via Adaptation](http://arxiv.org/abs/2504.06225v1)**
### **[LExT: Towards Evaluating Trustworthiness of Natural Language Explanations](http://arxiv.org/abs/2504.06227v1)**
### **[HiFlow: Training-free High-Resolution Image Generation with Flow-Aligned Guidance](http://arxiv.org/abs/2504.06232v1)**
### **[Transfer between Modalities with MetaQueries](http://arxiv.org/abs/2504.06256v1)**
### **[FEABench: Evaluating Language Models on Multiphysics Reasoning Ability](http://arxiv.org/abs/2504.06260v1)**
### **[Hogwild! Inference: Parallel LLM Generation via Concurrent Attention](http://arxiv.org/abs/2504.06261v1)**
### **[GOLLuM: Gaussian Process Optimized LLMs -- Reframing LLM Finetuning through Bayesian Optimization](http://arxiv.org/abs/2504.06265v1)**
