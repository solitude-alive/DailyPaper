# The Latest Daily Papers - Date: 2025-03-09
## Highlight Papers
### **[A Practical Memory Injection Attack against LLM Agents](http://arxiv.org/abs/2503.03704v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces MINJA, a novel memory injection attack targeting LLM agents.  MINJA allows an attacker to inject malicious records into an agent's memory bank by simply interacting with the agent through queries and observing the outputs.  The core idea involves crafting malicious records that, when retrieved later, steer the agent towards undesirable actions when processing a victim user's query.  This is achieved by designing bridging steps to link a benign query with malicious reasoning, using indication prompts to guide the agent's response generation, and employing a progressive shortening strategy to improve the retrieval probability of the malicious record. The authors evaluate MINJA on diverse agents across various tasks and demonstrate its effectiveness in compromising agent memory and inducing malicious reasoning.

**Critical Evaluation:**

*   **Novelty:** The paper's primary novelty lies in its realistic threat model. It moves away from the assumption of direct memory manipulation, which is a common limitation of previous poisoning attacks on LLM agents.  MINJA operates under the constraints that an attacker has only standard user access, posing a more practical and challenging scenario. The introduction of bridging steps, indication prompts, and progressive shortening strategy are also novel contributions.

*   **Significance:** The findings highlight a serious vulnerability in current LLM agent designs. The ability for any user to influence the agent's memory and, consequently, its behavior raises significant security concerns, especially in safety-critical applications such as autonomous driving or healthcare. The paper underscores the need for robust memory security mechanisms in LLM agents. The attack success rate is concerningly high (98.2% for injection, 76.8% for elicitation) further suggesting current safeguards are insufficient.

*   **Strengths:**

    *   **Practical Threat Model:** The paper addresses a realistic attack scenario, making it highly relevant.
    *   **Comprehensive Evaluation:** The experiments are conducted on a diverse set of agents, datasets, and tasks, which adds to the credibility of the results.
    *   **Well-Defined Methodology:** The attack steps are clearly described, making it reproducible.
    *   **Stealthiness:** The attack is designed to be stealthy, minimally impacting the overall utility of the agent which makes it difficult to be detected.

*   **Weaknesses:**

    *   **Limited Defense Discussion:** While the paper briefly mentions potential defenses (detection-based moderation), it does not delve deeply into concrete countermeasures or evaluate their effectiveness. The t-SNE visualization is interesting but doesn't provide specific directions for defense strategies.
    *   **Indication Prompt Design:** The design of indication prompts, while effective, might be task-specific. More discussion on how to generalize or automate prompt creation would be beneficial.
    *   **Scalability of Victim-Target Mapping:** For high-stakes scenarios where precise manipulation is crucial, the attacker needs to inject multiple malicious records. This could be computationally intensive and less practical on high-volume systems.

*   **Potential Influence:** The paper is likely to influence future research in LLM agent security.  It will prompt investigations into more effective memory sanitization techniques, anomaly detection methods, and robust input/output validation mechanisms. This will also prompt a more critical evaluation of existing LLM agent memory architectures, with likely design changes to enhance security.

*   **Score Justification:**

    The paper makes a significant contribution by addressing a practical and previously underexplored attack vector in LLM agents. The novelty of the approach, realistic threat model, and clear presentation of results warrant a high score. However, the limited discussion of defenses and potential scalability issues prevent it from achieving a truly exceptional rating.

Score: 8

- **Score**: 8/10

### **[Process-based Self-Rewarding Language Models](http://arxiv.org/abs/2503.03746v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Process-based Self-Rewarding Language Models":

**Summary:**

The paper addresses the limitations of existing Self-Rewarding Language Models (SRLMs), particularly their poor performance in mathematical reasoning tasks.  It proposes a novel "Process-based Self-Rewarding" (PSRLM) approach.  This involves introducing step-wise LLM-as-a-Judge (evaluating intermediate steps) and step-wise preference optimization into the self-rewarding paradigm.  The PSRLM allows LLMs to simultaneously perform complex reasoning and evaluate the individual steps, generating finer-grained and more accurate reward signals. Experiments on 7B and 72B models across various mathematical reasoning benchmarks demonstrate that PSRLM can effectively enhance LLMs' mathematical reasoning capabilities, even surpassing human performance.  The paper shows LLMs trained with PSRLM exhibit improved mathematical abilities and LLM-as-a-Judge capabilities, suggesting PSRLM has the potential to unlock superhuman intelligence.

**Critical Evaluation:**

*   **Novelty:** The core novelty lies in the strategic integration of step-wise reasoning evaluation within the self-rewarding framework.  While SRLMs exist, their application to complex mathematical reasoning has been problematic. This paper offers a specific, arguably elegant, solution to this issue.  The use of LLM-as-a-Judge for *individual reasoning steps*, rather than the final result, is a significant differentiator. The iterative refinement using the model's own judgments on reasoning steps makes it a more powerful and targeted approach to self-improvement.

*   **Significance:** The work has the potential to significantly impact the field of LLM research, particularly in areas requiring robust reasoning capabilities. The concept of PSRLM could enable the creation of more capable LLMs, especially in domains where high-quality, human-annotated training data is scarce or expensive. The findings highlight the potential for self-improvement mechanisms to surpass human-level performance in certain cognitive tasks. The detailed analysis of the model's behavior during the iterative training process (e.g., changes in step number and length) is valuable for understanding how self-rewarding mechanisms operate.

*   **Strengths:**
    *   **Clear Problem Statement and Motivation:** The paper clearly articulates the limitations of existing SRLMs and the need for a more tailored approach for mathematical reasoning.
    *   **Well-Defined Methodology:**  The PSRLM pipeline is presented in a structured and easy-to-understand manner. The steps, including initialization, data generation, and optimization, are clearly explained.
    *   **Comprehensive Experiments:** The paper uses a variety of benchmarks and model sizes to validate the effectiveness of PSRLM.  The experiments are detailed, and the results are presented clearly.
    *   **Interesting Analysis:** The analysis of the model's behavior during the training process, including the analysis of LLM-as-a-Judge consistency and the data distribution analysis, adds significant value.
    *   **Careful Initialization Data Generation**: High-quality filtering and generation of initial data are performed by GPT-01 for the EFT data.
    *   **Robustness** The paper also analyzes and shows the robust of the algorithm.

*   **Weaknesses:**
    *   **Computational Cost:** The PSRLM approach likely has a significant computational cost associated with the iterative training process and the step-wise evaluation. This could limit its widespread adoption. This is only implicit but should be stated explicitly.
    *   **Reliance on Initialization:** The success of PSRLM depends on the initial capabilities of the LLM used. There is a potential for the model to get stuck in local optima if the initial model is not sufficiently capable.
    *   **Limited Scope:** While the paper demonstrates the effectiveness of PSRLM for mathematical reasoning, its applicability to other complex reasoning domains remains to be explored.
    *   **Complexity:** The entire system pipeline is complex. It is difficult to reproduce all the results in different contexts.

*   **Potential Influence:** The PSRLM approach could be adopted in other areas of complex reasoning, such as scientific discovery, software development, and legal reasoning. It could also inspire the development of new self-improvement mechanisms for LLMs.

**Score:** 8

**Rationale:**

The paper presents a novel and well-executed approach to address a significant limitation of existing SRLMs. The proposed PSRLM method has shown a meaningful improvements in complex mathematical reasoning ability by LLMs. Although it does not introduce a new area of science, it presents a innovative technical approach that contributes to advance LLM development and solve a known problem. It's a clever, well-validated improvement with significant potential within a defined scope. While there are limitations regarding computational cost, reliance on good initialization, and scope of application, the strengths outweigh the weaknesses, making this a solid and impactful contribution to the field. The score reflects the paper's strong methodology, clear presentation, and potential to inspire future research, while also acknowledging the presence of some limitations that restrict it from achieving a perfect score.

- **Score**: 8/10

### **[RiskAgent: Autonomous Medical AI Copilot for Generalist Risk Prediction](http://arxiv.org/abs/2503.03802v1)**
- **Summary**: Here is a summary and critical evaluation of the provided research paper:

**Summary:**

The paper introduces RiskAgent, an autonomous medical AI copilot designed for generalist risk prediction across a wide range of medical scenarios.  RiskAgent employs a multi-agent Large Language Model (LLM) system consisting of a Decider, Executor, and Reviewer, collaborating with evidence-based clinical decision support tools (risk calculators).  The system first retrieves appropriate risk calculation tools from a database, extracts necessary parameters from patient data, executes the tools, and then provides risk predictions. A novel benchmark, MedRisk, is created to evaluate LLM risk prediction capabilities. Experimental results demonstrate that RiskAgent outperforms existing commercial and open-source LLMs, including GPT-4, in risk prediction accuracy. The paper also demonstrates the generalizability of RiskAgent to rare diseases, cancer risk assessments, and other diagnostic tasks, emphasizing its potential for resource-limited clinical applications.

**Critical Evaluation:**

* **Novelty:** The paper presents a novel architecture (RiskAgent) for integrating LLMs with existing clinical decision support tools. This is a significant departure from simply fine-tuning LLMs on medical data. The development of MedRisk, a comprehensive benchmark specifically for medical risk prediction, is another noteworthy contribution.  The specific multi-agent architecture is innovative, showing a clear understanding of the challenges of deploying LLMs in clinical settings. The approach of collaborating with and utilizing existing evidence-based medical tools shows a significant shift in how LLMs are used for medical tasks.
* **Significance:**  The potential impact of this work is substantial.  By leveraging existing clinical decision support tools, RiskAgent addresses the limitations of current LLMs in medicine, namely resource intensity, data requirements, privacy concerns, and the tendency to hallucinate.  The improved accuracy and ability to provide evidence-based answers could lead to more reliable clinical decision support, potentially improving patient outcomes. The generalizability across different diseases and task types adds to its broad applicability in a wide range of medical scenarios. The fact that the code, data, and models are open-sourced encourages further research and development in this area. It addresses the critical issue of trustworthy AI by providing evidence for clinicians to inspect the output recommendations.

* **Strengths:**
    * **Comprehensive Evaluation:** The paper includes a thorough evaluation against a variety of LLMs (both commercial and open-source), demonstrating the superiority of RiskAgent.
    * **Novel Architecture:** The multi-agent architecture and integration with existing medical tools are well-designed and effectively address the challenges of applying LLMs to complex medical tasks.
    * **Benchmark Creation:** MedRisk fills a gap in available benchmarks for evaluating LLMs in risk prediction.
    * **Generalizability:** Demonstrated strong performance in diverse areas, including rare diseases, cancer, and other diagnostic tasks.
    * **Resource Efficiency:** The paper emphasizes the reduction in computational resource requirements, increasing the feasibility of deployment in resource-constrained settings.
    * **Reproducibility:** Open-sourcing the code, data, and models enables reproducibility and further development.

* **Weaknesses:**
    * **Reliance on MDCalc:** The reliance on MDCalc as the source of risk calculators raises concerns about the completeness and potential biases of this specific resource. Expanding the tool library would strengthen the solution's robustness.
    * **Limited Exploration of Failure Modes:**  While the paper demonstrates overall superior performance, a deeper dive into the types of errors RiskAgent still makes would be beneficial. Analyzing the specific scenarios where it fails and comparing those to failures from other LLMs can provide valuable insights and inform future research directions.
    * **Potential for Bias:** Risk calculators used for generating the questions and answering might be inherently biased based on the populations they were originally developed from. The paper does not explicitly discuss potential mitigation strategies.
    * **Small Parameter RiskAgent Model:** While beneficial for resource constraints, there might still be improvements with larger models, that are not shown in the work.

* **Influence on the Field:** This paper has the potential to shift the paradigm of how LLMs are used in medicine, promoting a collaborative approach with existing tools rather than relying solely on fine-tuning large models. It opens avenues for research into more trustworthy, explainable, and accessible AI solutions for clinical decision support.

Considering the strengths and weaknesses, I assign the following score:

Score: 8.5

**Rationale:**

The paper makes a significant contribution to the field of medical AI by introducing a novel architecture and a comprehensive benchmark for risk prediction. It addresses critical challenges related to resource intensity, trustworthiness, and generalizability, demonstrating superior performance compared to existing state-of-the-art LLMs. While the reliance on MDCalc and limited exploration of failure modes are limitations, the overall impact of the work is substantial, with the potential to influence future research and development in this area and to change how LLMs are applied in clinical practice. The open-sourcing of code, data, and models further amplifies its potential impact.

- **Score**: 8/10

### **[A Little Depth Goes a Long Way: The Expressive Power of Log-Depth Transformers](http://arxiv.org/abs/2503.03961v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "A Little Depth Goes a Long Way: The Expressive Power of Log-Depth Transformers":

**Summary:**

The paper investigates the expressive power of transformers with depths that grow logarithmically with the input sequence length (log-depth transformers). It demonstrates theoretically that even highly uniform transformers with log-depth can express important problems like recognizing regular languages and solving graph connectivity, which are known to be beyond the capabilities of fixed-depth transformers. This highlights the significance of even minimal depth scaling. The authors also show that scaling depth logarithmically is more efficient (in terms of resource requirements) than scaling width or using chain-of-thought reasoning to achieve similar computational capabilities. Empirical validation is provided, suggesting that the theoretical depth requirements for regular language recognition align well with practical observations.

**Critical Evaluation:**

*   **Novelty:** The paper's core novelty lies in rigorously demonstrating the expressive power of *log-depth* transformers for problems known to be beyond *fixed-depth* transformers. While previous work identified the limitations of fixed-depth transformers, this paper provides a more nuanced understanding by showing that minimal, logarithmic depth scaling is sufficient to overcome these limitations.  The comparison to width scaling and chain-of-thought also adds value by contextualizing the efficiency gains of dynamic depth. While the specific problems tackled (regular language recognition, graph connectivity) are not new in the context of transformer analysis, the focus on log-depth scaling as a *minimal* resource increment makes the contribution distinctive.

*   **Significance:** The paper's significance stems from its clarification of the relationship between transformer depth and reasoning abilities, particularly in sequential reasoning tasks. The theoretical results on regular language recognition and graph connectivity directly address questions about what kinds of reasoning can be performed with increasing levels of model complexity and how much increase is actually needed. This has potential implications for designing more efficient and capable transformer architectures, as it suggests that focusing on even slightly dynamic depth may be a promising direction. Additionally, the input length bounds provided by fixed-depth corollaries helps quantify minimal model scales. The practical validation reinforces these theoretical findings, solidifying their relevance to real-world applications. It also offers a path toward a more principled approach to model design, moving away from purely empirical hyperparameter tuning.

*   **Strengths:**
    *   **Rigorous theoretical analysis:** The paper provides formal proofs and complexity-theoretic connections, which lend credibility to its claims.
    *   **Clear problem framing:** The research questions are well-defined and relevant to the transformer research community.
    *   **Strong theoretical results:** The proofs directly establish the computational power of log-depth transformers for chosen problems.
    *   **Empirical validation:** The experimental results support the theoretical findings, demonstrating that the theoretical depth scaling aligns with practical depth requirements.
    *   **Comparison to alternatives:** The paper effectively contrasts scaling depth with scaling width and chain-of-thought steps, which offers valuable insights.

*   **Weaknesses:**
    *   **Limited scope of empirical validation:** The empirical evaluation focuses primarily on the A5 state tracking task. Further experiments on other regular languages and different tasks would strengthen the conclusions. While A5 state tracking is canonical, generalization can always be questioned.
    *   **Relatively small set of problems:** While the selected problems are important, they are still specific examples. It is unclear how these insights apply more broadly to different types of reasoning.
    *   **The practicality of the log-depth scaling might be limited by very long sequences:** In some applications, sequences are very long, and logarithmic depth may already be a large number.

*   **Potential Influence:**
    *   This work could influence the design of transformers by promoting the use of dynamically scaled depth, or at least providing principled guidance for choosing depths for reasoning-intensive tasks.
    *   It can spur further research into analyzing the expressive power of different transformer architectures and scaling strategies.
    *   It motivates exploration of dynamic depth approaches as a more efficient alternative to scaling width or using chain-of-thought.

**Justification for Score:**

The paper makes a novel and well-supported contribution to our understanding of transformer expressivity. The results are theoretically sound, empirically validated, and offer practical implications for model design. While the scope of the empirical validation is somewhat limited, the core findings are significant enough to warrant a high score. It demonstrates a clear theoretical advantage of dynamic depth, supported by experimentation, and will likely influence future research in the field.

Score: 8

- **Score**: 8/10

### **[Robust Data Watermarking in Language Models by Injecting Fictitious Knowledge](http://arxiv.org/abs/2503.04036v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces a novel approach to data watermarking for language models (LLMs) by injecting fictitious knowledge into training data. Unlike previous methods that rely on specific token sequences or stylistic patterns (making them vulnerable to filtering and forgetting), this technique embeds plausible yet fictional facts about non-existent entities. The method constructs these watermarks by sampling from FrameNet to generate semantically plausible facts, then uses LLMs to create documents describing these facts. The authors demonstrate that these "fictitious knowledge" watermarks are effective at evading data preprocessing filters (including deduplication) and can be reliably memorized by LLMs. Furthermore, they show that watermark detection is possible even with API-only access through question-answering tasks, making it practical for closed-source models.  The paper evaluates the impact of various design choices (e.g., watermark length, attribute diversity) and demonstrates the robustness of the approach against post-training modifications like continual pretraining and instruction tuning.

**Critical Evaluation:**

*   **Novelty:** The primary novelty lies in the concept of using *fictitious knowledge* as watermarks. This contrasts with previous approaches focused on repetitive token sequences or stylistic biases.  This strategy addresses a significant weakness in existing methods: their vulnerability to detection and removal during data preprocessing, particularly by adversaries seeking to scrub copyrighted material. Using plausible but false information makes the watermarks blend in with the broader training data, making them harder to flag and remove lexically.
*   **Significance:** The significance stems from addressing the limitations of existing watermarking techniques in the context of increasingly sophisticated LLM pipelines. The ability to evade filters, resist forgetting after post-training, and enable verification even with API-only access makes this approach significantly more practical and robust for real-world deployment. The QA-based detection method is particularly valuable given the prevalence of closed-source LLMs. The insights on how factors like watermark length, attribute diversity, and injection strategies affect memorization contribute valuable knowledge to the field.
*   **Strengths:**

    *   **Robustness:** The paper demonstrates the robustness of the approach against several realistic challenges in LLM development, including data filtering, post-training modification, and closed-API scenarios.
    *   **Practicality:** The QA-based verification method makes the watermarking practical for real-world application, especially with closed-source LLMs.
    *   **Comprehensive Evaluation:** The authors conduct a thorough evaluation, analyzing the impact of various design choices and exploring the scalability of the approach.
    *   **Clear and Well-Written:** The paper is clearly written and well-organized, making it easy to follow the methodology and understand the results.

*   **Weaknesses:**

    *   **Proxy Evaluation:** The paper relies heavily on proxy settings (training smaller models or continually pretraining) for evaluation. While this is understandable given the computational cost of training large LLMs from scratch, it raises questions about how well the findings will generalize to the largest models and datasets. Though the continual pretraining section serves as a good proxy, it's not the same as training from scratch on a trillion-token dataset.
    *   **Limited Adversarial Evaluation:** Although it explores adversarial deduplication, a more thorough evaluation against diverse and sophisticated adversarial removal techniques would strengthen the paper.  For instance, could an adversary identify FrameNet-derived attributes and remove documents containing them?
    *   **Ethical Considerations (Acknowledged but Limited Discussion):** While the authors briefly address the ethical implications of injecting fictitious information, a more in-depth discussion of potential unintended consequences would be beneficial. For example, could the widespread adoption of this technique inadvertently introduce biases or misinformation into the knowledge base of LLMs?
    *   **Scalability of Generation:** Generating high-quality, diverse, and plausible fictitious knowledge at scale for very large datasets may prove to be a bottleneck. The paper doesn't fully address the potential challenges of scaling up the watermark generation process.

*   **Impact:** This paper is likely to have a significant impact on the field of data watermarking for LLMs. Its focus on robustness, practicality, and scalability addresses key limitations of previous work and makes it a valuable contribution to the development of effective copyright protection mechanisms for LLMs. It will likely spur further research into techniques for creating stealthier and more resilient watermarks.

**Score:** 8

**Justification:** The paper presents a novel and significant contribution to data watermarking for language models. The "fictitious knowledge" approach is innovative and addresses major shortcomings of existing methods. The comprehensive evaluation and practical verification method enhance the value of the work.  The weaknesses, primarily the reliance on proxy evaluation and limited adversarial testing, prevent it from achieving a higher score. It is highly likely to be influential in the area of watermarking and IP protection for LLMs.

- **Score**: 8/10

### **[Beyond Memorization: Evaluating the True Type Inference Capabilities of LLMs for Java Code Snippets](http://arxiv.org/abs/2503.04076v1)**
- **Summary**: Here's a summary and evaluation of the paper "Beyond Memorization: Evaluating the True Type Inference Capabilities of LLMs for Java Code Snippets":

**Summary:**

This paper investigates whether Large Language Models (LLMs) truly *understand* Java code snippets when performing type inference, or if they are simply *memorizing* information from their training data, leading to data leakage. The authors identify a potential data leakage issue with the commonly used StatType-SO benchmark because its contents have been publicly available for an extended period.  They conduct three experiments: (1) evaluating LLMs on a newly generated, unseen dataset called ThaliaType; (2) assessing LLM performance on semantically equivalent but syntactically transformed code snippets; and (3) using delta debugging to find the minimal code needed for LLMs to perform type inference. The results suggest that LLMs struggle with unseen data, are sensitive to syntactic variations, and can infer types with minimal code, even lacking crucial simple names, implying reliance on pattern matching instead of semantic understanding. The paper concludes that prior evaluations of LLMs on type inference might be overly optimistic due to data leakage. The authors emphasize the need for carefully designed and rigorously evaluated benchmarks for assessing LLM capabilities.

**Critical Evaluation:**

This paper addresses a crucial and timely concern regarding the evaluation of LLMs in software engineering tasks: data leakage. The identification of potential flaws in commonly used benchmarks like StatType-SO is a significant contribution.  The three-pronged approach of the study provides strong evidence supporting the authors' claim that LLMs might be performing sophisticated pattern matching rather than true semantic analysis in type inference.

**Strengths:**

*   **Addresses an important problem:**  The paper directly tackles the challenge of data leakage, a pervasive issue that can inflate LLM performance and mislead researchers.
*   **Rigorous methodology:** The use of a new, unseen dataset (ThaliaType) avoids data leakage and provides a more accurate assessment of LLMs' generalization capabilities. Semantic-preserving transformations and delta debugging are well-chosen techniques for evaluating understanding and identifying necessary code elements.
*   **Clear and well-supported results:** The experimental results consistently point to the limitations of LLMs in true semantic understanding. The comparison with SnR (a non-ML based tool) is effective.
*   **Practical implications:** The paper's findings highlight the need for caution when interpreting LLM performance and the importance of creating more robust and leak-proof benchmarks.
*   **Replicability:** The authors make their data and code publicly available, encouraging further research and verification.
*   **Clarity of presentation:** The paper is well-written and easy to understand.

**Weaknesses:**

*   **Limited Scope of Transformations:** While the semantic-preserving transformations are useful, the scope of transformations could be considered limited. There may be other types of semantic-preserving transformations that would further probe the LLMs' understanding.
*   **RQ3 limitation**: The study focused on GPT-40-mini due to cost, limiting the conclusions on this front. While this is understandable, it could be seen as a limitation.
*   **Limited number of libraries:** They used the same set of libraries as StatType-SO. While useful for comparisons, it still is limited in scope.

**Novelty and Significance:**

The paper is novel in its focused investigation of data leakage in the specific context of LLM-based type inference for Java code snippets. It's significant because it questions the validity of previous LLM evaluations and provides a methodology for more rigorous assessments.  The ThaliaType dataset is a valuable contribution to the community. The use of semantic preserving transformations and delta debugging adds depth to the analysis and uncovers nuanced aspects of LLM behavior.

**Potential Influence:**

The paper is likely to influence future research by encouraging researchers to:

*   Be more aware of data leakage issues.
*   Develop new, more challenging, and carefully designed benchmarks.
*   Utilize techniques like semantic-preserving transformations and delta debugging to evaluate LLM capabilities more effectively.
*   Rethink the current emphasis on solely relying on accuracy metrics and consider alternative evaluation methodologies.

**Justification for Score:**

This paper is an important contribution to the field. It provides a crucial critical analysis of LLM's capabilities and is backed by solid experiments. The identified limitations are valid, but they don't significantly detract from the overall impact. The methodological contributions (ThaliaType, the approach to using transformations and delta debugging) are substantial. Given this, I assign a score of 8. It doesn't reach a 9 or 10 because while the conclusions are strong, the number of applied transformation techniques is limited.

Score: 8

- **Score**: 8/10

### **[Chart-HQA: A Benchmark for Hypothetical Question Answering in Charts](http://arxiv.org/abs/2503.04095v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary**

The paper "Chart-HQA: A Benchmark for Hypothetical Question Answering in Charts" introduces a new benchmark for evaluating multimodal large language models (MLLMs) on their ability to reason about charts.  The key novelty is the "Hypothetical Question Answering" (HQA) task, which requires models to answer questions that involve counterfactual reasoning based on information presented in charts. This contrasts with existing chart QA benchmarks that primarily focus on factoid question answering. The authors argue that factoid QA can be susceptible to output biases in MLLMs, where models rely on parametric memory rather than actual chart understanding. To create the Chart-HQA benchmark, they develop a human-AI interactive data synthesis approach called HAI. HAI leverages the text-editing capabilities of LLMs combined with human expert knowledge to generate diverse and high-quality HQA data efficiently.  The paper evaluates 18 MLLMs on Chart-HQA, revealing significant generalization challenges and imbalanced reasoning performance across different answer types.

**Critical Evaluation**

*   **Novelty:** The paper introduces a valuable new perspective on evaluating MLLMs for chart understanding. The HQA task is well-motivated and addresses a significant limitation of existing benchmarks. The HAI data synthesis approach is also a noteworthy contribution, demonstrating a practical method for generating counterfactual reasoning datasets with limited human effort. The task construction is thoughtful, considering the structural limitations of different chart types when imposing assumptions.

*   **Significance:** The findings presented in the paper are significant. They demonstrate that current MLLMs, despite their success on factoid chart QA, struggle with more complex reasoning tasks like counterfactual inference. This highlights the need for further research into developing MLLMs that truly "understand" visual data rather than simply memorizing patterns. The benchmark provides a valuable resource for researchers to track progress in this area. The detailed analysis of MLLM performance across different answer types within the HQA task is also insightful, revealing potential weaknesses in symbolic reasoning capabilities.

*   **Strengths:**

    *   **Clear Problem Definition:** The paper clearly articulates the problem of output biases in MLLMs for chart QA.
    *   **Well-Designed Task:** The HQA task is logically sound and designed to assess true chart understanding.
    *   **Efficient Data Synthesis:** The HAI approach provides a practical method for creating a challenging dataset with reasonable cost and complexity.
    *   **Comprehensive Evaluation:** The paper evaluates a diverse set of MLLMs and provides insightful analysis of their performance.
    *   **Detailed Analysis:** The focus on analyzing performance based on answer types reveals nuanced model shortcomings.

*   **Weaknesses:**

    *   **Reliance on GPT-4:** The HAI approach depends heavily on GPT-4, which might raise concerns about potential biases or limitations inherent to that specific model. While human feedback is used, the underlying generations are still driven by GPT-4's understanding.

    *   **Limited Dataset Size (potentially):** The paper mentions using 900 instruction proposals and 2172 hypothetical questions which seems like a rather small dataset. While well-curated, increasing the scale could offer a more robust evaluation of MLLMs. It's unclear if the 2172 questions are derived from 947 *different* factoid QA pairings.

    *   **Generalizability of Findings (potential):** While the benchmark demonstrates challenges on *existing* MLLMs, it would be useful to investigate how *specific training strategies* designed to address such biases impact scores in this setting.  Do methods focused on grounding improve performance? Are smaller but well-curated training sets sufficient to overcome the challenge compared to larger, less carefully curated ones?

*   **Potential Influence:** The Chart-HQA benchmark has the potential to become a standard evaluation tool for MLLMs in the chart understanding domain. It could drive the development of more robust and reliable models that are less susceptible to output biases.

**Score: 8**

**Rationale:**

The paper presents a significant contribution to the field of multimodal learning. The HQA task is a well-defined and valuable addition to the evaluation landscape for MLLMs. The HAI data synthesis approach is practical and efficient. However, the relatively small dataset size and dependency on GPT-4 for data generation (potential introduction of bias) are minor limitations. The benchmark highlights an important problem and offers a valuable resource for future research, justifying a high score. There's a significant opportunity for future work to address the dependency on a single LLM for synthesis and scale the data.

- **Score**: 8/10

### **[Token-Efficient Long Video Understanding for Multimodal LLMs](http://arxiv.org/abs/2503.04130v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces STORM (Spatiotemporal Token Reduction for Multimodal LLMs), a novel architecture that aims to improve long video understanding in video-based multimodal large language models (Video-LLMs). The key idea is to incorporate a dedicated temporal encoder, specifically leveraging the Mamba State Space Model, between the image encoder and the LLM. This encoder integrates temporal information into the image tokens, generating enriched representations that preserve inter-frame dynamics across the entire video sequence.  STORM also employs token reduction strategies like test-time sampling and training-based temporal and spatial pooling to reduce computational costs on the LLM without losing crucial temporal information. The paper claims that STORM simultaneously reduces training and inference latency while improving performance, enabling efficient and robust video understanding over extended temporal contexts.  Evaluations show state-of-the-art results across long video understanding benchmarks (MLVU, LongVideoBench, VideoMME) while reducing computation costs and decoding latency for fixed numbers of input frames.

**Critical Evaluation:**

*   **Novelty:** The idea of incorporating a temporal encoder using Mamba SSM between the image encoder and the LLM is a significant step forward. It addresses a key limitation of existing Video-LLMs, which treat frames independently, lacking explicit temporal modeling. Using Mamba for temporal modeling is promising due to its linear complexity and potential for capturing long-range dependencies. The combination of this encoder with token reduction techniques (temporal and spatial pooling) to manage computational costs is also valuable.

*   **Significance:** The paper addresses a very important and challenging problem in the field: efficient long video understanding. Overcoming the computational bottlenecks associated with long videos is critical for real-world applications.  The reported improvements in performance and reductions in latency are compelling and demonstrate the practical benefits of the STORM architecture. The benchmarks used (MLVU, LongVideoBench, VideoMME, EgoSchema) are well-established and relevant for evaluating long-video understanding. Showing state-of-the-art performance while significantly reducing computation makes this quite significant.

*   **Strengths:**
    *   Clear problem definition and motivation.
    *   Well-defined architecture with a good explanation of the Mamba-based temporal encoder.
    *   Effective token reduction strategies to manage computational costs.
    *   Comprehensive experimental evaluation on multiple datasets.
    *   Significant performance improvements compared to existing methods.
    * Thorough ablation studies demonstrate the contribution of various components.
    * Addresses an important problem with significant practical implications.

*   **Weaknesses:**
    *   The paper does not provide as much detail as would be ideal about hyperparameter tuning and architecture choices for the Mamba SSM in the temporal encoder. More ablation studies focusing on specific choices regarding the Mamba layer could be informative.
    * The computational efficiency claims, while supported by reported numbers, could be strengthened by also including concrete energy usage data alongside latency measurements to provide a better assessment of its overall environmental impact.

*   **Potential Influence:** The paper has the potential to significantly influence the field of Video-LLMs. The STORM architecture provides a viable solution for efficient long video understanding. Other researchers might adopt or adapt this approach. The token reduction strategies presented could also become common practice in the field. In general, the ideas could influence the development of future generations of efficient and high-performing Video-LLMs. The success in integrating temporal modeling so early into the pipeline may lead to further research along these lines.

**Justification for Score:**

I am assigning a score of 8/10. The paper presents a novel and well-engineered architecture (STORM) that effectively addresses the challenge of efficient long video understanding in Video-LLMs. The use of Mamba SSM for temporal modeling, combined with carefully designed token reduction techniques, leads to substantial performance improvements and computational savings. While minor aspects related to Mamba architecture choices and more detailed energy usage data might be explored further, the paper's strengths outweigh its weaknesses. The paper is likely to have a notable and positive influence on the field.

**Score: 8**

- **Score**: 8/10

### **[How to Mitigate Overfitting in Weak-to-strong Generalization?](http://arxiv.org/abs/2503.04249v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper addresses the issue of overfitting in weak-to-strong generalization, a technique used to train AI models on tasks beyond human evaluation capabilities. It proposes a two-stage framework to improve both the quality of the supervision signals and the quality of the questions used for training.  The first stage filters training data based on the weak model's self-consistency to improve label quality. The second stage reuses discarded questions and leverages the previously finetuned strong model to generate answers and refine the training set. The authors demonstrate improvements in performance gap recovered (PGR) on mathematical reasoning benchmarks (GSM8k and MATH) using Llama 3 and Deepseek models.

**Critical Evaluation:**

*   **Novelty:** The core idea of addressing both supervision and question quality is a valuable contribution. While filtering based on confidence is not entirely novel, the two-stage framework, specifically the *reuse* of discarded questions with the *refined* strong model's supervision, adds a significant layer of novelty. The analysis of the trade-off between supervision quality and question quality/diversity is a key insight.
*   **Significance:** Superalignment is a critical problem in AI safety, and weak-to-strong generalization is a promising approach.  Addressing overfitting in this setting is highly relevant. The empirical results, showing substantial improvements in PGR, even reaching 100% on some benchmarks, suggest the framework's potential to advance the field. The paper identifies a fundamental problem in weak-to-strong generalization, analyzes the causes, and proposes a practical solution with supporting theoretical arguments.
*   **Strengths:**

    *   Well-defined problem statement and clear motivation.
    *   The proposed two-stage framework is easy to understand and implement.
    *   Comprehensive experimental evaluation with state-of-the-art models (Llama 3, Deepseek) and benchmarks (GSM8k, MATH).
    *   Detailed analysis of results, including the impact on question difficulty and diversity.
    *   Ablation studies to demonstrate the importance of each component of the framework.
    *   Connection to theoretical concepts (pseudolabel correction, coverage expansion).
*   **Weaknesses:**

    *   The computational overhead of the two-stage finetuning approach could be a limitation for large-scale applications and real-time scenarios.
    *   The need to tune the confidence thresholds for filtering might be a practical challenge, as the optimal thresholds can vary across different tasks and datasets. The lack of an automated method for threshold selection is a practical hurdle.
    *   The experiments are limited to mathematical reasoning tasks. The effectiveness of the framework on other types of tasks needs to be validated.
    *   While the theoretical analysis provides some justification, it could be strengthened with more rigorous theoretical bounds on the generalization error.

*   **Potential Influence:** The paper has the potential to significantly influence research on weak-to-strong generalization and superalignment. The proposed framework provides a practical and effective approach to address overfitting, which is a major obstacle in this field. The insights on the trade-off between supervision and question quality could inform the design of future training strategies.  The release of the code and models would further amplify its impact. However, the limitations regarding the computational cost and threshold tuning may limit immediate adoption in all settings.

**Score: 8**

**Rationale:**

The paper presents a novel and significant contribution to the field of weak-to-strong generalization. The two-stage framework effectively mitigates overfitting by addressing both supervision and question quality. The comprehensive experimental evaluation and detailed analysis provide strong evidence for the effectiveness of the approach. However, the computational cost and the need for threshold tuning are practical limitations that prevent a higher score. While the theoretical analysis adds some depth, it could be made more rigorous. Overall, the paper is a valuable contribution that has the potential to advance research on superalignment, though some practical challenges need to be addressed in future work.

- **Score**: 8/10

### **[RCRank: Multimodal Ranking of Root Causes of Slow Queries in Cloud Database Systems](http://arxiv.org/abs/2503.04252v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "RCRank: Multimodal Ranking of Root Causes of Slow Queries in Cloud Database Systems":

**Summary:**

The paper introduces RCRank, a novel framework designed to diagnose and rank the root causes of slow queries in cloud database systems.  RCRank leverages a multimodal approach, incorporating information from query statements (SQL), execution plans, execution logs, and key performance indicators (KPIs). It employs self-supervised pre-training to improve cross-modal alignment and task relevance, and uses root-cause-adaptive cross-transformers for adaptive feature fusion. Finally, the model is trained with an impact-aware objective to identify and rank root causes according to their potential for improving query performance.  The authors validate RCRank on both real and synthetic datasets, demonstrating its ability to outperform state-of-the-art methods in root cause identification and ranking.

**Critical Evaluation:**

*   **Novelty:**

    *   **Strengths:** The key novelty lies in its multimodal approach to *ranking* root causes based on *estimated impact* on performance, rather than merely *identifying* potential causes. The use of self-supervised pre-training to align different modalities and the adaptive fusion technique using cross-transformers specifically tailored to different root causes are also novel contributions. It addresses a significant gap in existing research, which mainly focuses on identifying root causes without prioritizing them according to their impact. This allows for a more targeted approach to slow query optimization, saving resources and time.

    *   **Weaknesses:** While the combination of multimodal data is not completely new in database diagnostics, the way RCRank integrates the data and ranks the impact on performance is a significant contribution. The individual components, such as the use of transformers or specific pre-training techniques, might have been explored in other contexts. However, the specific configuration and application within the context of slow query root cause ranking are novel.

*   **Significance:**

    *   **Strengths:** The paper addresses a highly relevant problem in cloud database management: slow query performance.  By prioritizing root causes based on their impact, RCRank has the potential to significantly improve the efficiency of slow query resolution, leading to better user experience and cost savings for cloud database users.  The experimental results, demonstrating the superiority of RCRank over existing methods, underscore its practical significance. The ability to handle both real and synthetic data demonstrates the robustness and generalizability of the proposed framework.

    *   **Weaknesses:** The practical deployment of RCRank could be challenging. The collection and processing of multimodal data (SQL, execution plans, logs, KPIs) at scale could introduce overhead.  Moreover, the effectiveness of the method depends on the quality of the root cause impact labels, which are obtained through rule-based and LLM-based revision methods. While the LLM approach can improve accuracy, there's a risk of errors and biases affecting the quality of those labels. The paper could benefit from a more thorough discussion of the limitations and practical considerations for deploying RCRank in a real-world cloud database environment, e.g., considerations about data privacy.

*   **Rigor:**

    *   **Strengths:** The experiments are well-designed and comprehensive, using multiple datasets (both real and synthetic) and a variety of evaluation metrics. The comparison with several strong baselines demonstrates the effectiveness of RCRank. Ablation studies provide valuable insights into the contribution of each component of the framework.

    *   **Weaknesses:** While the code and data are reportedly available, independent verification of the results would further strengthen the paper.

*   **Clarity:**

    *   The paper is generally well-written and organized, making it easy to understand the proposed framework and its components. The diagrams and tables are helpful in illustrating the concepts and results.

**Justification for the Score:**

Overall, the paper makes a significant contribution to the field of database management by introducing a novel and effective method for ranking root causes of slow queries in cloud database systems. The multimodal approach, combined with self-supervised pre-training and adaptive feature fusion, allows RCRank to achieve superior performance compared to existing methods. The experimental results are compelling and demonstrate the practical significance of the proposed framework. While there are some limitations regarding the deployment complexity and the potential for errors in impact labeling, the strengths of the paper outweigh its weaknesses. Given the novel approach and significant practical impact,

**Score: 8**

- **Score**: 8/10

### **[Lost in Literalism: How Supervised Training Shapes Translationese in LLMs](http://arxiv.org/abs/2503.04369v1)**
- **Summary**: Here's a summary and evaluation of the paper:

**Summary:**

The paper "Lost in Literalism: How Supervised Training Shapes Translationese in LLMs" investigates the phenomenon of translationese in large language model (LLM)-based machine translation systems. Translationese refers to unnatural translations characterized by overly literal rendering of source text, even when LLMs are pre-trained on vast corpora of natural language. The authors systematically evaluate translationese in LLM outputs, finding its prevalence and tracing its roots to biases introduced during supervised fine-tuning (SFT). They propose mitigation strategies, including polishing golden references and filtering unnatural training instances. Empirical evaluations demonstrate that these approaches reduce translationese and improve translation naturalness, as validated by human evaluations and automatic metrics.

**Critical Evaluation:**

*   **Strengths:**

    *   **Novel Problem Focus:** The paper addresses a critical but often overlooked aspect of LLM-based translation: translation style and naturalness.  While adequacy is often emphasized, naturalness is equally important for creating fluent and user-friendly translations.
    *   **Systematic Investigation:** The study provides a systematic and quantitative analysis of translationese in LLMs, using both human evaluation and automatic metrics. The methodology involving expert annotators to identify translationese spans adds rigor.
    *   **Root Cause Analysis:** Identifying SFT as a source of translationese bias is a valuable contribution. It shifts the focus from solely pre-training data to considering the impact of the training paradigm.
    *   **Effective Mitigation Strategies:**  The proposed mitigation strategies, particularly polishing golden references, demonstrate practical ways to improve LLM translation quality. The improvements are validated by both human and automatic evaluations.
    *   **Reproducibility:** The release of data and code enhances the reproducibility and accessibility of the research.
*   **Weaknesses:**

    *   **Limited Language Pairs:** The core evaluations are focused on English-Chinese and German-English, potentially limiting the generalizability to other language pairs, especially low-resource languages or languages with vastly different syntactic structures.
    *   **Reliance on Specific LLMs:** The study primarily utilizes specific LLMs (Llama-3.1-8B, Qwen-2.5-7B, and GPT models).  While these are representative, exploring a wider range of models, particularly those designed with different architectures or training objectives, would strengthen the conclusions.
    *   **Subjectivity of Human Evaluation:** While expert annotators are used, the identification of translationese can still be somewhat subjective. The paper could benefit from a more in-depth discussion of the criteria used for annotation and measures taken to ensure inter-annotator reliability beyond reporting aggregate agreement scores. The lack of example annotations in the main paper is an oversight.
    *   **Over-emphasis on Faithfulness?** The finding that "translation" prompts inherently promote more literal translations, with less emphasis on generating natural, expressive output, is interesting. However, the emphasis on *faithfulness* may not be the *cause* of translationese, but instead a product of *specific* design decisions and/or datasets used during SFT.

*   **Novelty and Significance:**

    *The paper is the first systematic study addressing translationese in LLMs.* Although prior works have looked at translationese in NMT and LLMs, this work uniquely identifies and addresses its causes in the context of LLM fine-tuning. It tackles the problem with concrete solutions that are experimentally validated.
    *The paper demonstrates the role of SFT in skewing the generative potential of otherwise expressive language models* towards a literal, less natural translationese.

**Justification of Score:**

The paper makes a significant contribution by highlighting the problem of translationese in LLM-based translation, tracing its roots to SFT, and proposing practical mitigation strategies. The methodology is sound, and the results are supported by both human and automatic evaluations. The analysis and insights are valuable for the machine translation community. The primary weakness is the limited language scope and reliance on a few specific LLMs. Therefore, a score of **8** is assigned.

**Score: 8**

- **Score**: 8/10

### **[Speculative MoE: Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling](http://arxiv.org/abs/2503.04398v1)**
- **Summary**: Here's a summary and critical evaluation of the "Speculative MoE" paper:

**Summary:**

The paper introduces Speculative MoE (s-MoE), a technique aimed at improving the efficiency of Mixture-of-Experts (MoE) inference.  s-MoE addresses the communication bottleneck inherent in Expert Parallelism (EP) by proactively predicting expert routing paths for tokens. This is achieved through two main mechanisms: speculative token shuffling (s-TS) and speculative expert grouping (s-EG).  s-TS pre-shuffles tokens to devices where their predicted experts reside, while s-EG pre-clusters experts likely to be activated by related tokens. The approach is integrated into both DeepSpeed-MoE and SGLang frameworks, demonstrating performance improvements across various MoE models, datasets, and hardware configurations (homogeneous and heterogeneous interconnects). The paper also includes several engineering optimizations such as fused kernels and deduplication.

**Critical Evaluation:**

*   **Novelty:**  The core idea of *speculatively* reducing communication in MoE inference is incremental. While previous works have explored expert prefetching/offloading and some forms of expert placement (e.g., ExFlow), the combination of s-TS and s-EG, along with their specific implementations within popular frameworks, and the use of a balanced co-clustering technique constitutes a significant engineering advance.  The use of probabilistic models for token-expert route prediction, and modeling both intra-layer and inter-layer affinity is valuable, even if it is based on a tabularized conditional probability approach. The concept of online speculative token shuffling and offline expert pre-grouping is a sound engineering decision.

*   **Significance:** The paper demonstrates practical improvements in MoE inference throughput under different latency constraints (TTFT, TPOT, p90-TBT).  The results showing gains on both fast homogeneous (NVLink) and slow heterogeneous (PCIe) interconnects are especially important as it makes the technique relevant to a broad range of deployment scenarios. The integration and benchmarking in both DeepSpeed-MoE and SGLang underscores the technique's generalizability. The analysis on the effectiveness of s-TS vs. s-EG helps in understanding the impact of different components and suggests potential for further optimization.

*   **Strengths:**
    *   Clear problem definition and well-motivated approach.
    *   Comprehensive experimental evaluation across different models, datasets, and hardware.
    *   Integration and validation in multiple popular frameworks.
    *   Detailed ablation studies and analysis of contributing factors.
    *   Practical system optimizations.

*   **Weaknesses:**
    *   The probabilistic models for predicting token-expert routes, although effective, are relatively simple.  More sophisticated prediction methods could potentially yield further gains.
    *   The description of some implementation details could be more thorough (e.g., the specifics of the co-clustering algorithm in Appendix B).
    *   While the paper claims to reduce EP communication volume losslessly, the accuracy of token-expert route prediction is less than perfect, implying potential for performance degradation in some extreme cases. This aspect could be further investigated.

*   **Impact:** The paper has a high potential to influence the field of MoE inference, especially as MoE models become more prevalent. The techniques presented are practical and readily deployable in existing frameworks.  It provides a valuable contribution towards democratizing LLM inference, particularly for setups with resource constraints.

**Justification for Score:**

The score reflects that while s-MoE is not a revolutionary breakthrough in MoE theory, it is a significant *practical* advancement. It takes existing principles and concepts (expert prefetching, token re-ordering) and combines them with careful engineering and system-level optimizations to achieve tangible performance benefits. The paper's strengths lie in its thorough evaluation, practical focus, and integration into established frameworks, making it immediately useful to researchers and practitioners working on MoE deployment. The weaknesses are mostly related to the relative simplicity of prediction models and less-than-perfect route predictions that need more detailed analysis. It builds nicely on ExFlow by incorporating inter-layer affinity and adding support for intra-layer affinity, leading to better pre-scheduling. Overall, the novelty, significance, and influence of this work point toward a strong contribution to the community.

Score: 8

- **Score**: 8/10

### **[SOLAR: Scalable Optimization of Large-scale Architecture for Reasoning](http://arxiv.org/abs/2503.04530v1)**
- **Summary**: Here's a summary and critical evaluation of the provided research paper:

**Summary:**

The paper introduces SOLAR (Scalable Optimization of Large-scale Architecture for Reasoning), a framework designed to dynamically optimize the reasoning topology of Large Language Models (LLMs).  It addresses the limitations of the standard Chain-of-Thought (CoT) approach by enabling LLMs to use tree-like or graph-like reasoning structures, adapting to the specific demands of different problems. The framework includes: (1) TAG (Topological-Annotation-Generation) for automated dataset creation and segmentation, (2) Topological-Scaling, a reward-driven system to align training and inference, and (3) M-TRM (Multi-Task Topological Reward Model) for autonomous topology selection and answer generation. The paper presents empirical results on MATH and GSM8K datasets, demonstrating accuracy improvements and reduced inference latency.

**Critical Evaluation:**

* **Novelty:** The paper offers significant novelty. While CoT and its extensions like ToT and GoT are established, SOLAR takes a step further by dynamically *learning* and *selecting* the most appropriate reasoning topology for a given problem. The automated annotation and synthetic dataset generation (TAG) is also a valuable contribution, addressing a key bottleneck in the field.  The idea of a multi-task reward model (M-TRM) to jointly optimize topology selection and answer generation is also novel. The proposed 'Topological Scaling' approach, bridging training and inference scaling, is also a fresh perspective.

* **Significance:** The paper's potential significance is considerable. The reported gains in accuracy on challenging datasets like MATH and GSM8K are impressive. The reduction in response length (and thus, inference latency) is practically important. More broadly, the SOLAR framework offers a path towards more adaptive and efficient reasoning in LLMs, which is crucial for their real-world deployment. The work also sets up a dynamic competition mechanism for reasoning topologies.

* **Strengths:**
    * **Clear Problem Definition:** The paper clearly identifies the limitations of current LLM reasoning approaches, specifically the reliance on CoT.
    * **Well-Defined Framework:** The SOLAR framework is well-defined with distinct components (TAG, Topological-Scaling, M-TRM).
    * **Empirical Validation:**  The paper provides strong empirical evidence to support its claims, with results on benchmark datasets. The ablation studies are particularly useful in isolating the contributions of each component.
    * **Automated Annotation:** The automated annotation scheme can be beneficial for other downstream applications in reasoning as well.
    * **Practical Implications:** The work has clear practical implications for improving the performance and efficiency of LLMs in real-world tasks.
    * **Tradeoff Analysis:** The discussion of the trade-offs between the different scaling strategies (Topological Tuning, Topological Rewarding, Hybrid Scaling) is valuable for practitioners.

* **Weaknesses:**
    * **Reliance on Existing Datasets:** The experiments are conducted on standard math and reasoning datasets. While these datasets are challenging, expanding the evaluation to other domains would strengthen the generalizability of the findings.
    * **Computational Cost:** While the paper mentions the increased computational cost of hybrid scaling, a more detailed analysis of the computational resources required for training the M-TRM and for inference would be beneficial. A comparison to other techniques with similar performance gains would be helpful.
    * **Limited Exploration of Topology Selection:** While the paper shows that different topologies are beneficial, a deeper investigation into *why* certain topologies are better suited for specific problem types would be valuable. What characteristics of a problem indicate that a tree-like or graph-like reasoning structure is necessary?
    * **Black Box nature of LLMs:** Many arguments are justified by empirical gains, however, there could be an improved explanation of the underlying cause for the phenomena observed.
    * **Synthetic Data Discussion:** While the discussion of synthetic data generation is thorough, the experiments using synthetic data are lacking. It would be beneficial to show the utility of synthetic data.

* **Overall Impact:** This paper will likely have a strong impact on the field.  It presents a novel and effective approach to improving LLM reasoning, and the automated annotation tool will likely be adopted in future research efforts. The ideas will spark new research into learning optimal reasoning topologies and will lead to more adaptive and efficient LLMs.

**Score: 8**

**Rationale:** The paper presents a novel, well-defined, and empirically validated framework for dynamically optimizing the reasoning topology of LLMs.  The findings are significant, showing substantial accuracy gains and reduced inference latency. While there is room for improvement in terms of computational cost analysis, generalizability to other domains, and understanding the underlying mechanisms of topology selection, the paper represents a significant advance in the field of LLM reasoning.

- **Score**: 8/10

### **[LLM-guided Plan and Retrieval: A Strategic Alignment for Interpretable User Satisfaction Estimation in Dialogue](http://arxiv.org/abs/2503.04675v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces PRAISE (Plan and Retrieval Alignment for Interpretable Satisfaction Estimation), a novel framework for estimating user satisfaction in dialogue systems. PRAISE addresses the limitations of existing USE methods by focusing on interpretability and efficiency. It leverages Large Language Models (LLMs) to generate natural language strategies for classifying user satisfaction, quantifies the relevance between utterances and strategies using a feature retriever, and classifies satisfaction levels using a score analyzer. The framework iteratively refines these strategies to optimize performance. A key advantage is that PRAISE only uses LLMs during training, enabling efficient inference without direct LLM calls during deployment. Experimental results demonstrate state-of-the-art performance on several benchmark datasets.

**Critical Evaluation:**

*   **Novelty:** The paper presents a genuinely novel approach to user satisfaction estimation by integrating LLMs for strategy generation and a retrieval mechanism for aligning utterances with these strategies. This strategy is more interpretable than "black box" LM approaches. The separation of the training phase (LLM usage) and inference phase (efficient model) is a significant contribution. The specific architecture combining a strategy planner, feature retriever, and score analyzer, tailored for USE, appears to be original.

*   **Significance:** The ability to estimate user satisfaction accurately and efficiently is crucial for improving dialogue systems. PRAISE offers several benefits:

    *   *Improved Performance:* The paper demonstrates SOTA performance on multiple USE benchmarks, indicating a real advancement.
    *   *Enhanced Interpretability:* The method provides utterance-level explanations, aiding in understanding user satisfaction and guiding system improvements. The example in Figure 4 showcasing the use of a strategy to explain a user’s DSAT (Dissatisfied) state is compelling.
    *   *Scalability:* Eliminating the need for LLMs during inference makes the approach more scalable and cost-effective compared to methods that rely on continuous LLM calls.

*   **Strengths:**

    *   *Clear Problem Definition:* The paper clearly defines the problem of user satisfaction estimation and highlights the shortcomings of current approaches.
    *   *Well-Defined Framework:* PRAISE is a well-structured framework with a logical flow and clear descriptions of each module.
    *   *Strong Experimental Results:* The experiments are comprehensive, covering multiple datasets and comparing PRAISE against several strong baselines, including models from the GPT family.
    *   *Ablation Studies:* The ablation study provides insight into the importance of different components of the framework, which helps to understand the overall effectiveness.
    *   *Scalability Analysis:* The paper demonstrates a clear advantage of PRAISE on speed and hardware requirements compared to other models.
    *   *Interpretability Showcase:* Demonstrating how PRAISE can improve interpretability of USE.

*   **Weaknesses:**

    *   *Dependency on LLM Knowledge:* The quality of the generated strategies is heavily dependent on the LLM's internal knowledge. While the paper acknowledges this limitation, it could be explored further. What happens when the LLM lacks domain-specific information or has biases?
    *   *Dataset Focus:* The exclusive focus on task-oriented dialogues is limiting. Exploring performance on open-domain dialogues would broaden the impact of the work. While justified, the limited scale of evaluation on more dialogue turns would improve the real world demonstration and usefulness of PRAISE.
    *   *Limited initial strategies:* While experimentation with initial strategies is done through ablations, the number of human defined strategies remains limited. The impact of the quality of initial stategies is unknown.

*   **Potential Influence:** PRAISE has the potential to influence the field by:

    *   *Providing a practical and scalable solution for user satisfaction estimation.*
    *   *Promoting the use of LLMs for generating interpretable dialogue strategies.*
    *   *Encouraging further research on developing explainable AI techniques for dialogue systems.*

*   **Justification for Score:** I am assigning a score of **8**. PRAISE presents a novel and significant contribution to user satisfaction estimation in dialogue systems by combining the strengths of LLMs with efficient retrieval and reasoning. The experimental results support the effectiveness of the framework, and the focus on interpretability and scalability are valuable features. While the dependence on LLM knowledge and limited domain scope are valid concerns, the paper presents a strong and well-executed approach that is likely to stimulate further research. The rigorous ablation studies and scalability analysis solidify this as a worthy contribution.

Score: 8

- **Score**: 8/10

### **[Quantifying the Reasoning Abilities of LLMs on Real-world Clinical Cases](http://arxiv.org/abs/2503.04691v1)**
- **Summary**: Here's a concise summary and critical evaluation of the paper, focusing on novelty, significance, and providing a score with a rigorous justification:

**Summary:**

The paper introduces MedR-Bench, a new benchmark designed to evaluate the reasoning capabilities of large language models (LLMs) in the medical domain. MedR-Bench distinguishes itself by focusing not just on the accuracy of final outputs (diagnoses, treatment plans) but also on the quality of the reasoning processes leading to those outputs. The benchmark comprises 1,453 structured patient cases derived from real-world case reports and spans 13 body systems and 10 specialty disorders, including both common and rare diseases.  A key contribution is the "Reasoning Evaluator," a novel agentic system that automatically assesses free-text reasoning responses based on efficiency, factuality, and completeness, leveraging web-scale medical resources. The authors evaluate five state-of-the-art reasoning LLMs using MedR-Bench, revealing strengths in simple diagnosis tasks and weaknesses in more complex areas like assessment recommendation and treatment planning.

**Critical Evaluation:**

*   **Novelty:** The paper demonstrates significant novelty. While medical LLM benchmarks exist, MedR-Bench uniquely emphasizes the evaluation of *reasoning processes* in addition to final outputs. The construction of the benchmark from real-world case reports (as opposed to synthetic or question-answering datasets) adds ecological validity. The "Reasoning Evaluator" is also a novel contribution, automating the complex task of assessing free-text reasoning quality in a scalable and objective manner.

*   **Significance:** The paper's significance lies in several areas. First, it addresses a critical gap in the evaluation of medical LLMs, moving beyond simple accuracy metrics to assess the *how* and *why* of their decision-making. This is crucial for building trust and reliability in clinical settings. Second, the benchmark provides a comprehensive and challenging testbed for LLMs, encompassing a diverse range of medical cases and clinical stages. Third, the detailed evaluation of existing LLMs provides valuable insights into their current capabilities and limitations, highlighting areas where further development is needed. The finding that open-source models are closing the gap is particularly significant for promoting accessibility and equity in healthcare AI. The limitations also point to valuable research areas for focus and advancement, particularly the need for greater completeness. However, the study uses cases only from a specific open-access source and might benefit from diversifying the data sources for future studies.

*   **Strengths:**
    *   Focus on reasoning processes, a crucial aspect often overlooked in existing benchmarks.
    *   Use of real-world case reports, increasing ecological validity.
    *   Novel "Reasoning Evaluator" for automated assessment of free-text reasoning.
    *   Comprehensive evaluation of multiple LLMs across different clinical stages.
    *   Emphasis on transparency and explainability in medical AI.

*   **Weaknesses:**
    *   The quality of the extracted "reasoning processes" from the case reports depends on the original case report authors' thoroughness, which might introduce bias.
    *   While the Reasoning Evaluator offers automation, it still relies on an LLM (GPT-40), potentially introducing another layer of bias or limitations in understanding medical nuance.
    *   The study primarily analyzes structured case reports, which might not fully capture the complexities and uncertainties of real-world clinical practice.
    *   While including a broad range of diseases, the depth of analysis for each specific condition may be limited due to dataset size.
    *   The dependence on web-based resources in the reasoning evaluation could make the results subject to the limitations of access to reliable, up-to-date, and trustworthy information.

**Justification of Score:**

I assign a score of **8**. The paper represents a substantial contribution to the field of medical AI. The emphasis on reasoning evaluation is timely and important, and the MedR-Bench dataset and Reasoning Evaluator tool are valuable resources for the community. The clear identification of LLM strengths and weaknesses, particularly in complex clinical stages, helps focus future research efforts. While the limitations related to data source dependency, potential evaluator bias, structured data format, and scale are valid, the paper's novelty and significance outweigh these drawbacks. The work has the potential to significantly influence the development and evaluation of medical LLMs, leading to more reliable and trustworthy AI-powered clinical tools.

**Score: 8**

- **Score**: 8/10

## Other Papers
### **[Developing and Utilizing a Large-Scale Cantonese Dataset for Multi-Tasking in Large Language Models](http://arxiv.org/abs/2503.03702v1)**
### **[A Practical Memory Injection Attack against LLM Agents](http://arxiv.org/abs/2503.03704v1)**
### **[Effective LLM Knowledge Learning via Model Generalization](http://arxiv.org/abs/2503.03705v1)**
### **[Rethinking Video Tokenization: A Conditioned Diffusion-based Approach](http://arxiv.org/abs/2503.03708v1)**
### **[Improving LLM Safety Alignment with Dual-Objective Optimization](http://arxiv.org/abs/2503.03710v1)**
### **[Towards Understanding Distilled Reasoning Models: A Representational Approach](http://arxiv.org/abs/2503.03730v1)**
### **[Process-based Self-Rewarding Language Models](http://arxiv.org/abs/2503.03746v1)**
### **[The MASK Benchmark: Disentangling Honesty From Accuracy in AI Systems](http://arxiv.org/abs/2503.03750v1)**
### **[RiskAgent: Autonomous Medical AI Copilot for Generalist Risk Prediction](http://arxiv.org/abs/2503.03802v1)**
### **[Vision-Language Models Struggle to Align Entities across Modalities](http://arxiv.org/abs/2503.03854v1)**
### **[LEWIS (LayEr WIse Sparsity) -- A Training Free Guided Model Merging Approach](http://arxiv.org/abs/2503.03874v1)**
### **[Pretrained LLMs as Real-Time Controllers for Robot Operated Serial Production Line](http://arxiv.org/abs/2503.03889v1)**
### **[On the Convergence of Adam-Type Algorithm for Bilevel Optimization under Unbounded Smoothness](http://arxiv.org/abs/2503.03908v1)**
### **[Safe LLM-Controlled Robots with Formal Guarantees via Reachability Analysis](http://arxiv.org/abs/2503.03911v1)**
### **[GuardDoor: Safeguarding Against Malicious Diffusion Editing via Protective Backdoors](http://arxiv.org/abs/2503.03944v1)**
### **[COARSE: Collaborative Pseudo-Labeling with Coarse Real Labels for Off-Road Semantic Segmentation](http://arxiv.org/abs/2503.03947v1)**
### **[Performance Comparison of Large Language Models on Advanced Calculus Problems](http://arxiv.org/abs/2503.03960v1)**
### **[A Little Depth Goes a Long Way: The Expressive Power of Log-Depth Transformers](http://arxiv.org/abs/2503.03961v1)**
### **[Generative Learning of Densities on Manifolds](http://arxiv.org/abs/2503.03963v1)**
### **[All-atom Diffusion Transformers: Unified generative modelling of molecules and materials](http://arxiv.org/abs/2503.03965v1)**
### **[Model Behavior Specification by Leveraging LLM Self-Playing and Self-Improving](http://arxiv.org/abs/2503.03967v1)**
### **[ReasonGraph: Visualisation of Reasoning Paths](http://arxiv.org/abs/2503.03979v1)**
### **[Image Data Augmentation for the TAIGA-IACT Experiment with Conditional Generative Adversarial Networks](http://arxiv.org/abs/2503.03982v1)**
### **[RetinalGPT: A Retinal Clinical Preference Conversational Assistant Powered by Large Vision-Language Models](http://arxiv.org/abs/2503.03987v1)**
### **[DSV-LFS: Unifying LLM-Driven Semantic Cues with Visual Features for Robust Few-Shot Segmentation](http://arxiv.org/abs/2503.04006v1)**
### **[Benchmarking Large Language Models on Multiple Tasks in Bioinformatics NLP with Prompting](http://arxiv.org/abs/2503.04013v1)**
### **[TextDoctor: Unified Document Image Inpainting via Patch Pyramid Diffusion Models](http://arxiv.org/abs/2503.04021v1)**
### **[Robust Data Watermarking in Language Models by Injecting Fictitious Knowledge](http://arxiv.org/abs/2503.04036v1)**
### **[Beyond Existance: Fulfill 3D Reconstructed Scenes with Pseudo Details](http://arxiv.org/abs/2503.04037v1)**
### **[Underlying Semantic Diffusion for Effective and Efficient In-Context Learning](http://arxiv.org/abs/2503.04050v1)**
### **[RA-DP: Rapid Adaptive Diffusion Policy for Training-Free High-frequency Robotics Replanning](http://arxiv.org/abs/2503.04051v1)**
### **[Uncovering inequalities in new knowledge learning by large language models across different languages](http://arxiv.org/abs/2503.04064v1)**
### **[FREAK: Frequency-modulated High-fidelity and Real-time Audio-driven Talking Portrait Synthesis](http://arxiv.org/abs/2503.04067v1)**
### **[Beyond Memorization: Evaluating the True Type Inference Capabilities of LLMs for Java Code Snippets](http://arxiv.org/abs/2503.04076v1)**
### **[PokéChamp: an Expert-level Minimax Language Agent](http://arxiv.org/abs/2503.04094v1)**
### **[Chart-HQA: A Benchmark for Hypothetical Question Answering in Charts](http://arxiv.org/abs/2503.04095v1)**
### **[Disparities in LLM Reasoning Accuracy and Explanations: A Case Study on African American English](http://arxiv.org/abs/2503.04099v1)**
### **[LLMs Can Generate a Better Answer by Aggregating Their Own Responses](http://arxiv.org/abs/2503.04104v1)**
### **[InterChat: Enhancing Generative Visual Analytics using Multimodal Interactions](http://arxiv.org/abs/2503.04110v1)**
### **[Simple Self Organizing Map with Visual Transformer](http://arxiv.org/abs/2503.04121v1)**
### **[Diff-Reg v2: Diffusion-Based Matching Matrix Estimation for Image Matching and 3D Registration](http://arxiv.org/abs/2503.04127v1)**
### **[Token-Efficient Long Video Understanding for Multimodal LLMs](http://arxiv.org/abs/2503.04130v1)**
### **[Biological Sequence with Language Model Prompting: A Survey](http://arxiv.org/abs/2503.04135v1)**
### **[Dynamic Benchmarking of Reasoning Capabilities in Code Large Language Models Under Data Contamination](http://arxiv.org/abs/2503.04149v1)**
### **[Ticktack : Long Span Temporal Alignment of Large Language Models Leveraging Sexagenary Cycle Time Expression](http://arxiv.org/abs/2503.04150v1)**
### **[KidneyTalk-open: No-code Deployment of a Private Large Language Model with Medical Documentation-Enhanced Knowledge Database for Kidney Disease](http://arxiv.org/abs/2503.04153v1)**
### **[Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation](http://arxiv.org/abs/2503.04162v1)**
### **[TIMER: Temporal Instruction Modeling and Evaluation for Longitudinal Clinical Records](http://arxiv.org/abs/2503.04176v1)**
### **[Measuring temporal effects of agent knowledge by date-controlled tool use](http://arxiv.org/abs/2503.04188v1)**
### **[MASTER: Multimodal Segmentation with Text Prompts](http://arxiv.org/abs/2503.04199v1)**
### **[Knowledge-Decoupled Synergetic Learning: An MLLM based Collaborative Approach to Few-shot Multimodal Dialogue Intention Recognition](http://arxiv.org/abs/2503.04201v1)**
### **[Energy-Guided Optimization for Personalized Image Editing with Pretrained Text-to-Image Diffusion Models](http://arxiv.org/abs/2503.04215v1)**
### **[FuseChat-3.0: Preference Optimization Meets Heterogeneous Model Fusion](http://arxiv.org/abs/2503.04222v1)**
### **[Synthetic Data is an Elegant GIFT for Continual Vision-Language Models](http://arxiv.org/abs/2503.04229v1)**
### **[SemaSK: Answering Semantics-aware Spatial Keyword Queries with Large Language Models](http://arxiv.org/abs/2503.04234v1)**
### **[DiffPO: Diffusion-styled Preference Optimization for Efficient Inference-Time Alignment of Large Language Models](http://arxiv.org/abs/2503.04240v1)**
### **[ThrowBench: Benchmarking LLMs by Predicting Runtime Exceptions](http://arxiv.org/abs/2503.04241v1)**
### **[How to Mitigate Overfitting in Weak-to-strong Generalization?](http://arxiv.org/abs/2503.04249v1)**
### **[RCRank: Multimodal Ranking of Root Causes of Slow Queries in Cloud Database Systems](http://arxiv.org/abs/2503.04252v1)**
### **[ADOR: A Design Exploration Framework for LLM Serving with Enhanced Latency and Throughput](http://arxiv.org/abs/2503.04253v1)**
### **[How to Move Your Dragon: Text-to-Motion Synthesis for Large-Vocabulary Objects](http://arxiv.org/abs/2503.04257v1)**
### **[Towards Autonomous Reinforcement Learning for Real-World Robotic Manipulation with Large Language Models](http://arxiv.org/abs/2503.04280v1)**
### **[How Do Hackathons Foster Creativity? Towards AI Collaborative Evaluation of Creativity at Scale](http://arxiv.org/abs/2503.04290v1)**
### **[MathMistake Checker: A Comprehensive Demonstration for Step-by-Step Math Problem Mistake Finding by Prompt-Guided LLMs](http://arxiv.org/abs/2503.04291v1)**
### **[Mapping AI Benchmark Data to Quantitative Risk Estimates Through Expert Elicitation](http://arxiv.org/abs/2503.04299v1)**
### **[Malware Detection at the Edge with Lightweight LLMs: A Performance Evaluation](http://arxiv.org/abs/2503.04302v1)**
### **[Solving Word-Sense Disambiguation and Word-Sense Induction with Dictionary Examples](http://arxiv.org/abs/2503.04328v1)**
### **[The Challenge of Identifying the Origin of Black-Box Large Language Models](http://arxiv.org/abs/2503.04332v1)**
### **[In-depth Analysis of Graph-based RAG in a Unified Framework](http://arxiv.org/abs/2503.04338v1)**
### **[LEDiT: Your Length-Extrapolatable Diffusion Transformer without Positional Encoding](http://arxiv.org/abs/2503.04344v1)**
### **[Large Language Models for Zero-shot Inference of Causal Structures in Biology](http://arxiv.org/abs/2503.04347v1)**
### **[Layer-Specific Scaling of Positional Encodings for Superior Long-Context Modeling](http://arxiv.org/abs/2503.04355v1)**
### **[Lost in Literalism: How Supervised Training Shapes Translationese in LLMs](http://arxiv.org/abs/2503.04369v1)**
### **[TRACT: Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning for LLM-as-a-Judge](http://arxiv.org/abs/2503.04381v1)**
### **[Shaping Shared Languages: Human and Large Language Models' Inductive Biases in Emergent Communication](http://arxiv.org/abs/2503.04395v1)**
### **[TableLoRA: Low-rank Adaptation on Table Structure Understanding for Large Language Models](http://arxiv.org/abs/2503.04396v1)**
### **[Speculative MoE: Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling](http://arxiv.org/abs/2503.04398v1)**
### **[Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search](http://arxiv.org/abs/2503.04412v1)**
### **[Can Large Language Models Predict Antimicrobial Resistance Gene?](http://arxiv.org/abs/2503.04413v1)**
### **[Learning Transformer-based World Models with Contrastive Predictive Coding](http://arxiv.org/abs/2503.04416v1)**
### **[AOLO: Analysis and Optimization For Low-Carbon Oriented Wireless Large Language Model Services](http://arxiv.org/abs/2503.04418v1)**
### **[Activation Space Interventions Can Be Transferred Between Large Language Models](http://arxiv.org/abs/2503.04429v1)**
### **[TPC: Cross-Temporal Prediction Connection for Vision-Language Model Hallucination Reduction](http://arxiv.org/abs/2503.04457v1)**
### **[Guiding LLMs to Generate High-Fidelity and High-Quality Counterfactual Explanations for Text Classification](http://arxiv.org/abs/2503.04463v1)**
### **[DAST: Difficulty-Adaptive Slow-Thinking for Large Reasoning Models](http://arxiv.org/abs/2503.04472v1)**
### **[Large Language Models in Bioinformatics: A Survey](http://arxiv.org/abs/2503.04490v1)**
### **[Multi-modal Summarization in Model-Based Engineering: Automotive Software Development Case Study](http://arxiv.org/abs/2503.04506v1)**
### **[SOLAR: Scalable Optimization of Large-scale Architecture for Reasoning](http://arxiv.org/abs/2503.04530v1)**
### **[Keeping Yourself is Important in Downstream Tuning Multimodal Large Language Model](http://arxiv.org/abs/2503.04543v1)**
### **[ViT-VS: On the Applicability of Pretrained Vision Transformer Features for Generalizable Visual Servoing](http://arxiv.org/abs/2503.04545v1)**
### **[Benchmarking Reasoning Robustness in Large Language Models](http://arxiv.org/abs/2503.04550v1)**
### **[Compositional Translation: A Novel LLM-based Approach for Low-resource Machine Translation](http://arxiv.org/abs/2503.04554v1)**
### **[HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization](http://arxiv.org/abs/2503.04598v1)**
### **[The Best of Both Worlds: Integrating Language Models and Diffusion Models for Video Generation](http://arxiv.org/abs/2503.04606v1)**
### **[Towards Data-Efficient Language Models: A Child-Inspired Approach to Language Learning](http://arxiv.org/abs/2503.04611v1)**
### **[START: Self-taught Reasoner with Tools](http://arxiv.org/abs/2503.04625v1)**
### **[Mark Your LLM: Detecting the Misuse of Open-Source Large Language Models via Watermarking](http://arxiv.org/abs/2503.04636v1)**
### **[Implicit Cross-Lingual Rewarding for Efficient Multilingual Preference Alignment](http://arxiv.org/abs/2503.04647v1)**
### **[LLM-guided Plan and Retrieval: A Strategic Alignment for Interpretable User Satisfaction Estimation in Dialogue](http://arxiv.org/abs/2503.04675v1)**
### **[Compositional World Knowledge leads to High Utility Synthetic data](http://arxiv.org/abs/2503.04687v1)**
### **[Quantifying the Reasoning Abilities of LLMs on Real-world Clinical Cases](http://arxiv.org/abs/2503.04691v1)**
### **[UIPE: Enhancing LLM Unlearning by Removing Knowledge Related to Forgetting Targets](http://arxiv.org/abs/2503.04693v1)**
### **[L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning](http://arxiv.org/abs/2503.04697v1)**
### **[Universality of Layer-Level Entropy-Weighted Quantization Beyond Model Architecture and Size](http://arxiv.org/abs/2503.04704v1)**
### **[Shifting Long-Context LLMs Research from Input to Output](http://arxiv.org/abs/2503.04723v1)**
### **[L$^2$M: Mutual Information Scaling Law for Long-Context Language Modeling](http://arxiv.org/abs/2503.04725v1)**
