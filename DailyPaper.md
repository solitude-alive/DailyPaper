# The Latest Daily Papers - Date: 2025-03-28
## Highlight Papers
### **[Can Large Language Models Predict Associations Among Human Attitudes?](http://arxiv.org/abs/2503.21011v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper investigates the ability of large language models (LLMs), specifically GPT-4o, to predict human attitudes and beliefs based on other attitudes, even when those attitudes are seemingly disparate and lack surface-level semantic similarity.  The authors created a novel dataset of human responses to diverse attitude statements and tested GPT-4o's ability to recreate pairwise correlations among attitudes and predict individual attitudes. They found that GPT-4o can predict attitudes beyond surface-level similarity, suggesting it captures aspects of the deeper, latent structure of human belief systems.  While semantic similarity improved prediction accuracy, the model's ability to make meaningful social inferences from dissimilar attitudes was a key finding.  The paper also explores the potential for LLMs to be used for persuasive purposes, raising ethical concerns.

**Critical Evaluation:**

**Novelty:** The paper's primary novelty lies in its shift of focus from predicting attitudes based on *similar* or closely related topics (as previous research has done) to exploring predictions between *dissimilar* attitudes. This addresses a crucial gap in the existing literature, which has often focused on surface-level semantic similarities to explain LLM performance.  The creation of a new dataset specifically designed to test this aspect is also a valuable contribution.  The paper also breaks new ground by using a novel method to evaluate correlations across attitudes with the OpenAI API. This methodology is novel and can have useful applications to future research. The examination of these factors and the methodology are what elevate the significance of this paper.

**Significance:**  The findings are significant for several reasons:

*   **Understanding LLM Reasoning:** The study provides insights into the inner workings of LLMs, suggesting they are capable of more than just pattern matching based on semantic similarity. They seem to be learning some underlying structure of human belief systems and applying that in their reasoning processes.
*   **Ethical Implications:**  The paper highlights the potential for LLMs to be used for manipulative purposes by understanding and predicting individual beliefs. This is a timely and important consideration given the increasing prevalence of LLMs in various applications.
*   **Applications in Social Science:** The finding that LLMs can capture correlational structures in human data has significant implications for social science research, potentially enabling the creation of more accurate simulations of human behavior.
*   **Practical Applications:** The findings can be used to improve recommendation systems, target advertising, and align chat assistants to the cultural perspectives of their users.

**Strengths:**

*   **Clear Research Question:**  The paper clearly defines the problem it's addressing (predicting attitudes across dissimilar topics).
*   **Robust Methodology:** The study employs a well-designed methodology, including a custom dataset, a validated research experiment, careful construction of prompts, and rigorous statistical analysis.
*   **Thoughtful Discussion:** The paper offers a balanced and nuanced discussion of the findings, acknowledging limitations and addressing ethical considerations.
*   **Strong Supporting Evidence:** The claims made in the paper are all strongly supported by clear and robust data.

**Weaknesses:**

*   **Limited Depth on "Latent Structure":** While the paper shows LLMs can predict across dissimilar attitudes, it doesn't delve deeply into *what* this underlying "latent structure" actually is. More qualitative analysis to identify what conceptual connections the LLM is making would strengthen the study.
*   **Complexity of Metric Presentation:** Some of the experimental metrics are difficult to follow for someone not deeply versed in LLM evaluation, and the structure of the results could be improved with some clarification.

**Overall:**

The paper presents a significant contribution to the field by moving beyond simply demonstrating that LLMs can predict attitudes to exploring *how* they do it, even when surface-level similarities are absent.  The ethical implications and potential applications in social science are also important. Despite minor weaknesses in the depth of explanation and presentation, the study is well-executed and insightful.
**Score: 8**

**Rationale:**  The score reflects the strong methodology, original and important research question, and insightful findings. The significance of the paper is somewhat tempered by the limitations noted above (specifically the limited deep dive into understanding the latent structure). This score also takes into consideration the high volume of research conducted in the field of LLMs as of late, which limits the influence that research may have unless highly significant.

- **Score**: 8/10

### **[Online Reasoning Video Segmentation with Just-in-Time Digital Twins](http://arxiv.org/abs/2503.21056v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces a novel framework for online reasoning video segmentation (RS) called "Online Reasoning Video Segmentation with Just-in-Time Digital Twins." This framework aims to address the limitations of existing RS approaches that heavily rely on multimodal Large Language Models (LLMs). It disentangles perception and reasoning by employing an LLM planner to construct low-level scene representations (digital twins) from high-level video using specialist vision models. This approach, termed "just-in-time," selectively requests information from these models only when required by the implicit query, improving efficiency and enabling complex reasoning without LLM fine-tuning. The paper introduces a new video reasoning segmentation benchmark comprising 200 videos with 895 implicit text queries across semantic, spatial, and temporal reasoning categories. Experimental results demonstrate that the proposed method outperforms existing approaches across all reasoning categories.

**Critical Evaluation:**

*   **Novelty:** The core novelty lies in the "just-in-time" digital twin concept for video RS. The idea of dynamically constructing a scene representation tailored to the specific query, as opposed to using a fixed, comprehensive representation, is a significant contribution. Disentangling perception from reasoning also allows for the integration of specialized vision models and LLMs optimized for specific roles. While digital twins are an existing concept, their application in this dynamic, query-driven, and online video segmentation setting appears to be novel. The benchmark dataset is also a valuable contribution.

*   **Significance:** The significance of this work stems from its potential to advance embodied AI agents. By enabling efficient online RS without LLM fine-tuning, the framework can enable AI agents to understand complex, implicit instructions in real-time, dynamic environments. The improved performance on complex reasoning tasks and spatial/temporal awareness addresses a major bottleneck in existing RS systems. This can facilitate robots and other agents to interact meaningfully with the real world based on high-level instructions. By improving the online capability of RS, this work could also improve online video understanding tasks such as identifying potential hazards in self-driving car footage and automatically captioning important sections of a live stream.

*   **Strengths:**
    *   Addresses limitations of current RS approaches (complex reasoning, LLM fine-tuning, online processing).
    *   Introduces a novel and efficient "just-in-time" digital twin concept.
    *   Presents a comprehensive video reasoning segmentation benchmark.
    *   Demonstrates strong experimental results across various reasoning categories and difficulty levels.
    *   Provides a clear and well-structured framework with detailed explanations.
    *   The framework doesn't require LLM fine-tuning which is a significant advantage due to the amount of resource required for finetuning, and the risk of catastrophic forgetting

*   **Weaknesses:**
    *   Relies on powerful pre-trained specialist vision models and LLMs. The method's performance is intrinsically tied to the capabilities of these underlying models. Future versions might face challenges as new foundation models are released that perform specific tasks better.
    *   The implementation details of the LLM-coder for generating executable spatial/temporal reasoning code could be more elaborated upon.
    *   While the experiments are comprehensive, further analysis of the computational cost and latency of the proposed method in a real-world online setting would strengthen the evaluation.
    *   The framework makes heavy use of LLMs. Though finetuning is not required, it is still very computationally expensive.

*   **Potential Influence:**
    *   The "just-in-time" digital twin concept could inspire new architectures for video understanding and reasoning.
    *   The benchmark dataset could facilitate further research and development in video RS.
    *   The framework could be extended to other embodied AI tasks.
    *   Future research could investigate ways to make the specialist models more adaptable and less reliant on powerful pre-trained models, or experiment with smaller LLMs

*   **Overall:**
    The paper presents a significant and novel contribution to the field of video reasoning segmentation. The "just-in-time" digital twin concept, coupled with the disentangled perception-reasoning framework and the comprehensive benchmark dataset, provides a valuable foundation for future research. While there are some limitations, the strengths of the paper outweigh the weaknesses.

**Score: 8**

- **Score**: 8/10

### **[MoQa: Rethinking MoE Quantization with Multi-stage Data-model Distribution Awareness](http://arxiv.org/abs/2503.21135v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "MoQa: Rethinking MoE Quantization with Multi-stage Data-model Distribution Awareness":

**Summary:**

The paper introduces MoQa, a novel quantization framework specifically designed for Mix-of-Experts (MoE) models.  MoQa addresses the limitations of existing quantization methods, which are primarily designed for dense Large Language Models (LLMs) and fail to account for the unique complexities of MoEs.  The core idea is to decouple the data-model distribution complexity in MoEs into three key analysis stages: sparse data activation, data-parameter mapping, and inter-expert correlations. This decoupling allows the framework to identify the significance of individual experts and parameters with greater awareness of the underlying data distribution. Based on these insights, MoQa proposes fine-grained mix-quantization strategies that adapt to various data activation and expert combination scenarios. The paper presents experimental results demonstrating improved perplexity on language modeling tasks and accuracy on zero-shot inference tasks compared to traditional methods like GPTQ and MoEPTQ.

**Critical Evaluation:**

*   **Novelty:** The paper's primary novelty lies in its multi-stage approach to analyzing and optimizing MoE quantization.  Existing methods largely treat MoEs as monolithic structures or apply random strategies, ignoring the dynamic and complex relationships between data, experts, and parameters. MoQa's approach of decoupling these relationships represents a significant advancement. Specifically, the concept of analyzing token-level utilization, re-weighting data distributions to emphasize relevant tokens, and then mapping this refined data to expert models, before finally considering inter-expert correlations, is novel.
*   **Significance:** The work has the potential to significantly impact the field of MoE compression. The improved quantization performance achieved by MoQa, without sacrificing accuracy, can make these large models more accessible and deployable in resource-constrained environments. Further, the comprehensive analysis of data-model distribution in MoEs provides valuable insights that can guide future MoE architecture design and optimization. The discussion about the limitations of current quantization approaches for MoEs and the insightful analysis of each stage contributes to the understanding of how and where the existing quantization methods fail in MoE models, leading to a more robust and accurate analysis.
*   **Strengths:**
    *   **Comprehensive Analysis:**  The multi-stage analysis is well-motivated and clearly explained.
    *   **Fine-grained Strategies:** The resulting mix-quantization strategies are tailored to the specific challenges of MoEs.
    *   **Strong Empirical Results:** The experiments demonstrate substantial improvements compared to existing methods.
    *   **Clear Writing:** The paper is well-written and easy to follow.
*   **Weaknesses:**
    *   **Complexity:** The framework is complex and potentially challenging to implement in practice. The number of tunable hyperparameters for the multi-stage optimization would need to be carefully tuned.
    *   **Generalization:** While the results are promising, more experiments on a wider range of MoE architectures and datasets would strengthen the claim of general applicability.
    *   **Limited Scope:** The scope of the evaluation focuses on limited popular LLMs.
*   **Potential Influence:** If widely adopted, MoQa could become a standard approach for quantizing MoEs. Its insights are likely to inspire new research directions in MoE compression and optimization. It could also influence the design of new MoE architectures that are more amenable to quantization.

**Justification:**
MoQa addresses a critical and timely challenge in the field of LLMs – efficient compression and deployment of MoE models. The paper's novel multi-stage analysis and tailored quantization strategies represent a significant advancement over existing methods. Although the complexity of the approach may pose some implementation challenges, the strong empirical results and insightful analysis justify a high score. There are limitations such as the potential for increased implementation complexity.
Score: 8

- **Score**: 8/10

### **[System-wide Instrument Transformer Calibration and Line Parameter Estimation Using PMU Data](http://arxiv.org/abs/2503.21202v1)**
- **Summary**: Here's a summary and critical evaluation of the provided research paper:

**Summary:**

The paper addresses the challenge of simultaneously calibrating instrument transformers (ITs) and estimating line parameters (LPE) in power systems using phasor measurement unit (PMU) data.  It proposes a statistical framework to solve the interdependent SLIC problem. The key contributions are: a quantization procedure for uniquely identifying line parameters, an algorithm to leverage the accuracy of Revenue Quality Meters (RQMs) for system-wide SLIC using only *one* RQM in a connected tree, and a method to determine the optimal RQM location for minimizing estimation errors. The approach accounts for variations in line parameters and errors inherent in PMU devices. The paper demonstrates the accuracy and robustness of the method using the IEEE 118-bus system, and validates its practicality with real-world PMU data from a U.S. power utility.

**Critical Evaluation:**

* **Novelty:**  The paper presents several novel aspects:

    *   **Quantization Procedure:** The proposed method's novelty lies in its quantization procedure to uniquely identify line parameters, which directly addresses the 'underdetermined' nature of the traditional problem formulations. This is a significant step towards a practical and computationally feasible solution.
    *   **Minimal RQM Requirement:** The approach reduces reliance on expensive "perfect" ITs (RQMs) and demonstrates that system-wide calibration is achievable with just *one* strategically placed RQM.  This is a cost-effective and significant contribution. Previous approaches often assume or require error-free measurements from multiple ITs or knowledge of line parameters beforehand.
    *   **RQM Placement Algorithm:**  Developing an algorithm to determine the "best" location for that single RQM is a unique contribution that further enhances the practicality of the approach.
    *   **Explicit Noise Modeling:** explicitly modeling of the PMU device additive Gaussian noise in the positive-sequence components of the voltage and current phasors. This is crucial for practical applicability, as it provides better estimations.

*   **Significance:** The work is significant for several reasons:

    *   **Improved Accuracy and Reliability:** Calibrating ITs is crucial for ensuring the accuracy of power system monitoring and control applications, leading to more reliable operation.
    *   **Practical Applicability:**  The method's ability to work with real-world data, requiring only *one* RQM, accounts for line parameter variations, and noise makes it highly relevant for practical implementation by power utilities.
    *   **Addresses a Fundamental Interdependency:** The simultaneous LPE and IT calibration problem has been a long-standing challenge. This research presents a comprehensive and effective solution that breaks the cycle of interdependency. The paper's comparison to state-of-the-art [19] clearly displays a significant leap in IT estimation.
    *   The paper makes it easily implementable by sharing the pseudo-code of the proposed algorithms.

*   **Strengths:**

    *   **Rigorous Methodology:**  The approach is based on solid statistical principles (TLS) and a well-defined mathematical formulation.
    *   **Comprehensive Validation:**  The use of both a standard test system (IEEE 118-bus) and real-world data provides strong evidence of the method's effectiveness.
    *   **Sensitivity Analysis:**  The sensitivity studies examining the impact of PMU noise and IT accuracy class provide valuable insights into the robustness of the method.
    *   **Clear Presentation:** The paper is well-written and logically organized, making it easy to understand the proposed method and its results.

*   **Weaknesses:**

    *   **Connected Tree Assumption:**  The requirement of a 'connected tree' topology may not always be satisfied in complex power systems. The paper could benefit from a discussion on how the method could be adapted or extended to handle more general network topologies with loops. While PMUs are commonly placed to satisfy this assumption, a brief consideration of this limitation is warranted.
    *   **Database Dependency:** reliance on prior information regarding historical value ranges of the network parameters which can introduce external dependencies on system components.
    *   **Computational Cost:** The algorithm for RQM placement (Algorithm 4), involving repeated execution of the SW-SLIC algorithm, could be computationally expensive for very large-scale systems. While centrality measures are suggested, a more detailed analysis of the computational complexity and scalability would be beneficial.
    *   **Limited types of variations are considered in the database values:** Line parameters will exhibit variations that have seasonality in them due to temperature and loading conditions. A detailed analysis of the selection of appropriate parameter variation limits must be described.

* **Potential Influence:** This research has the potential to significantly impact the field of power system monitoring and control.  Its practicality, accuracy, and robustness make it a valuable tool for power utilities seeking to improve the accuracy of their state estimation and other applications reliant on PMU data. The use of a minimal number of RQMs can drastically lower the cost of wide-area applications that use synchrophasors.

*Overall, this is a valuable contribution with sound principles that will motivate more research.*

**Score: 8.5**

*Rigorous Rationale:* The score of 8.5 reflects the significant novelty, practical relevance, and comprehensive validation of the research. The paper addresses a long-standing problem in a cost-effective and robust manner. It demonstrates a clear advancement over existing methods, particularly in its ability to function effectively with just one RQM and in accounting for line parameter variations. The main deductions in the score are due to the limitations regarding the connected tree topology and the database dependencies, as well as a lack of a more detailed scalability analysis. In the absence of these limitations, it would easily warrant an excellent rating. However, the paper is well-thought-out and provides considerable advancements in the accuracy of the measurements.

- **Score**: 8/10

### **[Rethinking Graph Structure Learning in the Era of LLMs](http://arxiv.org/abs/2503.21223v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper "Rethinking Graph Structure Learning in the Era of LLMs" proposes a new paradigm for graph structure learning (GSL) specifically designed for text-attributed graphs (TAGs) leveraging large language models (LLMs). Recognizing the limitations of existing GSL methods tailored for traditional graphs without textual information, the authors introduce Large Language and Tree Assistant (LLaTA). LLaTA reformulates the GSL optimization objective as a tree-based optimization task with a language-aware tree sampler. This approach employs a structural encoding tree to capture graph topology, followed by LLM-based in-context learning to understand topology and text, and ultimately performs leaf-oriented two-step sampling for edge addition/removal.  The design aims for a decoupled, training-free model architecture, emphasizing reliable LLM inference over fine-tuning. Extensive experiments on 10 TAG datasets demonstrate LLaTA's flexibility, scalability, and state-of-the-art performance.

**Critical Evaluation:**

*   **Novelty:** The paper presents a novel approach to GSL for TAGs by shifting the focus from end-to-end training of edge predictors to a tree-based optimization framework driven by LLM in-context learning. The key innovations lie in the tree-based structural encoding to guide the LLM and the decoupled training-free architecture. This is a valuable contribution as prior LLM-based GSL methods often rely on fine-tuning or complex instruction datasets, hindering efficiency and adaptability.
    The key idea is to harness LLM's in-context learning abilities efficiently without computationally expensive fine-tuning, which provides a significant leap in terms of resource usage. The tree representation, while rooted in existing entropy-based methods, offers a structured framework for integrating textual data and prompting the LLM. This offers clear practical advantages.

*   **Significance:**  The paper is significant because it addresses a critical challenge in graph machine learning – efficient and robust learning with TAGs, a data representation becoming increasingly important.  By decoupling the GSL process from the need for extensive training and customized backbones, the approach makes GSL more accessible and adaptable to real-world applications. The experiments demonstrate substantial performance improvements and efficiency gains compared to existing LLM-based GSL methods.
    The effectiveness of LLaTA across a range of datasets supports its generalizability, and the ablation studies provide valuable insights into the contributions of different components. The robustness analysis and the analysis of time complexity makes this paper valuable to the community. However, the dependence on a high-performing LLM could be a limitation, as the quality of the LLM directly impacts the performance of LLaTA. Future work should explore how to make the model more robust to the LLM chosen.

*   **Strengths:**
    *   **Clear Problem Definition:** The paper clearly identifies the limitations of existing GSL methods when applied to TAGs in the era of LLMs.
    *   **Innovative Approach:**  The tree-based optimization framework and the decoupled, training-free architecture are innovative and well-motivated.
    *   **Comprehensive Experiments:**  The extensive experiments on 10 TAG datasets demonstrate LLaTA's superior performance, flexibility, and scalability. The ablation studies and robustness analysis provide valuable insights.
    *   **Well-written:** The paper is generally well-written and organized, making it easy to understand the proposed approach and its benefits.

*   **Weaknesses:**
    *   **LLM Dependency:** The performance is heavily dependent on the quality and capabilities of the underlying LLM. While this is inherent to the approach, it is a potential limitation.
    *   **Hyperparameter Sensitivity**: There is sensitivity to the hyperparamter tuning which could be problematic depending on the LLM used.

*   **Potential Influence:** LLaTA has the potential to significantly influence the field of GSL, particularly in applications involving TAGs. The training-free approach and the use of tree prompts can inspire new research directions for leveraging LLMs in graph machine learning.

**Score: 8.5**

**Justification:**

The paper makes a substantial contribution by proposing a novel and effective approach to GSL for TAGs in the era of LLMs. The method is well-motivated, innovative, and supported by comprehensive experiments. The decoupled, training-free architecture addresses a crucial challenge in the field by making GSL more efficient and adaptable. While the dependency on LLMs and the number of hyperparameters are potential limitations, the overall novelty and significance of the approach merit a high score. The clear performance gains, scalability, and flexibility position this paper as a valuable contribution that can influence future research in GSL and graph machine learning.

- **Score**: 8/10

### **[ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition](http://arxiv.org/abs/2503.21248v1)**
- **Summary**: Here's a concise summary and critical evaluation of the paper:

**Summary:**

The paper introduces ResearchBench, a novel benchmark designed to evaluate the capabilities of Large Language Models (LLMs) in scientific discovery. The benchmark decomposes the complex process of scientific hypothesis formulation into three key sub-tasks: inspiration retrieval, hypothesis composition, and hypothesis ranking. The dataset comprises scientific papers across 12 disciplines, enabling a broad assessment of LLMs' abilities. The authors developed an automated LLM-based framework to extract research questions, background surveys, inspirations, and hypotheses from these papers, ensuring contamination resistance by focusing on recent publications. Experimental results reveal that LLMs show promise in inspiration retrieval (an out-of-distribution task), suggesting their potential as "research hypothesis mines."

**Critical Evaluation:**

*   **Novelty:** The paper makes a significant contribution by addressing a critical gap in evaluating LLMs for scientific tasks. The decomposition of scientific discovery into sub-tasks is a valuable insight, and the creation of a dedicated benchmark addresses a lack of specific resources in this area. The dataset's focus on recent publications is crucial to mitigate data contamination.

*   **Significance:** The benchmark opens up new avenues for studying and improving LLMs' role in scientific research. The finding that LLMs can retrieve relevant inspirations despite their seemingly unrelated nature to the research question is particularly impactful. This suggests that LLMs can indeed generate novel associations, which is a cornerstone of creative and innovative scientific thought. By defining a clear benchmark, the paper sets a framework for future studies and advancements in the field of automated scientific discovery.

*   **Strengths:**

    *   **Comprehensive Decomposition:** Breaking down the hypothesis formulation process into sub-tasks is a logical and insightful approach.
    *   **Data Contamination Control:** The focus on recent publications is commendable and crucial for ensuring the benchmark's validity.
    *   **Automated Framework:** The development of an automated framework for data extraction adds scalability and reduces manual effort in benchmark construction.
    *   **Out-of-Distribution Task Analysis:** The exploration of inspiration retrieval as an out-of-distribution task highlights the LLMs' ability to make novel connections.

*   **Weaknesses:**

    *   **Limited scope:** Focusing primarily on papers published during a single year helps to avoid data contamination but may limit the diversity of the benchmark.
    *   **Expert Validation Sample Size:** The expert evaluation of the data extraction framework (62 papers) is a reasonable starting point, but increasing the sample size would further strengthen the validation.
    *   **Reliance on extractive methods:** Hypothesis Composition and hypothesis ranking relied on existing text in the literature and can be more powerful if it had the potential to generate more original content.
*   **Potential Influence:** The paper has the potential to influence the direction of LLM research in scientific discovery. The ResearchBench provides a foundation for further exploration of LLMs' capabilities, fostering innovation in automated hypothesis generation.
*   **The framework for extracting ideas is not perfect:** While the automated framework helps to extract the necessary components for each hypothesis, it does so with limited accuracy, given that each framework had only roughly 80% accuracy.

**Score: 8**

- **Score**: 8/10

### **[HORT: Monocular Hand-held Objects Reconstruction with Transformers](http://arxiv.org/abs/2503.21313v1)**
- **Summary**: Here is a summary and critical evaluation of the paper:

**Summary:**
The paper introduces HORT, a coarse-to-fine transformer-based framework for reconstructing 3D point clouds of hand-held objects from monocular images.  The method first generates a sparse point cloud from image and hand geometry features, then progressively refines it into a dense representation using pixel-aligned image features.  Key aspects include the integration of image features with 3D hand geometry to predict object point clouds and pose relative to the hand, and end-to-end training for optimal performance. Experimental results on synthetic and real datasets show state-of-the-art accuracy with faster inference speed, demonstrating good generalization.

**Critical Evaluation:**

*   **Novelty:** The paper presents a novel approach to 3D reconstruction of hand-held objects by using a coarse-to-fine transformer architecture. The fusion of image and hand geometry is a good idea that leverages implicit cues provided by hand shape. Pixel-aligned image feature extraction is also an efficient technique for refining point clouds. Combining these contributions into a single, end-to-end trainable model adds novelty. The method advances the field of hand-object reconstruction by emphasizing a balance between accuracy and inference speed, which is critical for real-time applications.
*   **Significance:** Achieving faster inference speeds is indeed significant, as it addresses a key limitation of existing methods (especially diffusion-based ones like D-SCO).  Outperforming state-of-the-art methods (D-SCO) in both speed and accuracy is a strong claim. The experimental results support the claim of state-of-the-art reconstruction accuracy with fast inference speed across various datasets. Generalization to "in-the-wild" images further enhances the significance, as it shows the practicality of the method.
*   **Strengths:**
    *   The coarse-to-fine architecture is a good design choice for improving efficiency and capturing both global and local details.
    *   Integrating hand geometry features along with image features is a valuable addition for regularization and reducing ambiguities in monocular 3D reconstruction.
    *   The end-to-end training strategy simplifies the pipeline and potentially leads to more optimal performance.
    *   Comprehensive experiments on diverse datasets including ObMan, HO3D, DexYCB, and MOW validate the method's effectiveness and generalization.
    *   The clear presentation and detailed architecture diagrams contribute to the paper's readability.
*   **Weaknesses:**

    *   While the paper highlights limitations in certain scenarios (heavy occlusion, rare objects), a more detailed analysis of failure cases and their underlying causes would further strengthen the work.
    *   Although mentioned in the appendix, it could have been more explicit that converting point clouds to meshes requires additional processing and reduces speed, thus diluting the speed claim to some degree.
    *   The reliance on pre-trained models (DINOv2, hand pose estimator) can limit its applicability in scenarios where domain adaptation or custom models are required. The extent to which the results depend on the quality of those pre-trained models should be explicitly discussed.
*   **Potential Influence:** The emphasis on efficiency could influence future research towards developing faster and more practical 3D reconstruction methods.  The integration of hand geometry features can serve as a good starting point for addressing similar reconstruction problems with related hand-object interaction. The adoption of transformers for this task is also likely to influence other researchers.

**Rigorous Rationale for Assigned Score:**

This paper makes a valuable contribution by addressing the speed limitations of existing monocular hand-object reconstruction methods without sacrificing reconstruction quality. The fusion of image and hand geometry features, combined with an efficient coarse-to-fine transformer architecture, is a strong technical contribution. The comprehensive evaluation and convincing results lend credence to the claims of state-of-the-art performance. While there are minor limitations, the overall quality of the work, its novelty, and potential to influence future research justify a high score.

Score: 8

- **Score**: 8/10

### **[Controlling Large Language Model with Latent Actions](http://arxiv.org/abs/2503.21383v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces COLA (Controlling Large Language Models with Latent Actions), a framework designed to improve the controllability and exploration of RL in LLMs by learning a compact latent action space. COLA employs an inverse dynamics model to extract latent actions conditioned on future tokens and fine-tunes the pre-trained LLM to function as a language world model that incorporates these actions. A policy model is then trained to generate actions within this language world model using behavior cloning or RL. Experiments using the Llama-3.1-8B model demonstrate that COLA's latent actions enable greater semantic diversity compared to token-level actions. The framework also shows improved performance in tasks like math reasoning (achieving higher scores on math500), agent-based tasks (Alfworld and ScienceWorld), and enhanced thinking prompts (Countdown game) with reduced computation time, without significantly degrading the original LLM's capabilities. The paper also indicates that using a smaller latent space can reduce the likelihood of reward hacking.

**Critical Evaluation:**

* **Novelty:** The core idea of introducing a latent action space for controlling LLMs is novel. The approach of learning the latent space using an inverse dynamics model and integrating this space into a pre-trained LLM for RL is innovative. Using the pre-trained LLM to create word embeddings and then learning a small action latent space is an interesting twist.
* **Significance:** The paper addresses a critical challenge in applying RL to LLMs, i.e., the excessively large action space resulting from token-level actions. By reducing the dimensionality of the action space, COLA has the potential to improve the efficiency and effectiveness of RL-based adaptation of LLMs for downstream tasks.  The results showing improved math reasoning and agent-based task performance are important contributions.
* **Strengths:**
    * **Sound Methodology:** The framework is well-defined, with clear descriptions of the different components and training procedures.  The integration of the inverse dynamics model, language world model, and policy model seems well-considered.
    * **Empirical Validation:** The experimental results are comprehensive, covering a variety of tasks and demonstrating the advantages of COLA over baseline approaches, with particular focus on quantitative evaluation on math500 and agent based experiments.
    * **Computational Efficiency:** The reduction in computation time reported for enhanced thinking prompts is a significant practical benefit, especially as LLMs become more computationally expensive.
    * **Potential for Controllability and Mitigation of Reward Hacking:** The results regarding enhanced controllability (better preference alignment) and reduced reward hacking highlight a crucial strength of the proposed latent action space approach.
* **Weaknesses:**
    * **Dependence on Inverse Dynamics Model:** The learned latent space heavily depends on the quality of the inverse dynamics model.  A poor inverse dynamics model could lead to a sub-optimal latent space and limit the performance of COLA. The paper needs a deeper analysis of its effect and potential limitations.
    * **Limited Theoretical Analysis:** While the empirical results are promising, the paper lacks a deeper theoretical understanding of the learned latent action space.  Characterizing the properties of the latent space (e.g., its structure, separability, etc.) would provide more insights.
    * **Scalability:** The experiments are performed on the Llama-3.1-8B model. It needs to be proven on larger models with increased parameters.

**Justification for Score:**

The paper presents a novel framework for controlling LLMs with latent actions, addressing a significant challenge in applying RL to LLMs. The comprehensive empirical results demonstrate clear improvements in task performance and computational efficiency, and the potential to mitigate reward hacking. While the reliance on the inverse dynamics model and the lack of deep theoretical analysis represent weaknesses, the strengths outweigh these limitations. The paper represents a significant advancement in RL-based adaptation of LLMs.

Score: 8

- **Score**: 8/10

### **[SyncSDE: A Probabilistic Framework for Diffusion Synchronization](http://arxiv.org/abs/2503.21555v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces SyncSDE, a probabilistic framework for synchronizing multiple diffusion models for collaborative generation tasks.  It addresses the limitations of existing methods that rely on naive heuristics for diffusion trajectory alignment by providing a principled way to model and adapt correlations between these trajectories. The core idea is to formulate diffusion synchronization as an optimization problem involving correlation modeling, task-specific adaption and identifying optimal correlation models for each specific task. This allows SyncSDE to achieve better performance compared to previous approaches by focusing the synchronization efforts and avoids blind application of heuristics. The authors demonstrate SyncSDE's effectiveness across diverse tasks, including mask-based text-to-image generation, text-driven real image editing, wide image generation, ambiguous image generation, 3D mesh texturing, and long-horizon motion generation. The method avoids ad-hoc strategies by modeling conditional probability terms and uses a single tunable hyperparameter.

**Critical Evaluation:**

*   **Novelty:** The primary novelty lies in the probabilistic framework for diffusion synchronization. While previous works have explored synchronization heuristics, SyncSDE provides a more principled theoretical basis for understanding *why* these heuristics work (or fail) and *where* they should be applied.  The explicit modeling of correlations between diffusion trajectories using a conditional probability formulation is a significant advancement. Identifying where the heuristic is focused is also novel. The idea is not simply a new heuristic, but a framework to understand and improve existing ones.

*   **Significance:** The paper's significance stems from its potential to improve the robustness and generalizability of collaborative generation using diffusion models. By providing a principled approach, SyncSDE reduces the need for extensive trial-and-error to find optimal synchronization strategies for new tasks. This has practical implications for expanding the applicability of multi-diffusion model systems. Also the improvement over prior art are also substantial. The consistent superior performance across various tasks strengthens the paper's claim of generalizability.

*   **Strengths:**

    *   Strong theoretical foundation: The probabilistic framework provides a clear and mathematically sound basis for diffusion synchronization.
    *   Task-specific adaptation: The ability to adapt the correlation model to each specific task allows for more effective synchronization compared to methods that apply a single heuristic across all tasks.
    *   Comprehensive evaluation: The extensive experiments across diverse tasks demonstrate the generalizability and effectiveness of SyncSDE.  The comparisons with state-of-the-art methods in each task highlights the gains achieved by the proposed approach.
    *   Clear explanations:  The paper is well-written and provides clear explanations of the proposed method and experimental setup. The ablation study of the hyperparameter provides insights on its role.

*   **Weaknesses:**

    *   Single Hyperparameter: Although the use of a single hyperparameter is beneficial for its ease of use, it also limits its expressiveness. There might be more complex dependencies between the different diffusion processes, that cannot be sufficiently captured by a single parameter, which controls their influence on each other.
    *   While the results are demonstrably better, the leap from probabilistic framework to concrete heuristic might benefit from additional clarity or discussion. Are there explicit guidelines for heuristic selection based on the probabilistic framework? Or does the framework merely aid in *understanding* and selecting *among* existing heuristics?

*   **Potential Impact:** SyncSDE has the potential to influence future research on collaborative generation using diffusion models. It provides a solid foundation for developing more sophisticated synchronization strategies and expands the range of tasks that can be tackled effectively. Also it serves as a first work to provide theoretical insight behind diffusion synchronization.

**Justification for Score:**

Given the novel probabilistic framework, the consistent improvement over state-of-the-art methods across a diverse set of tasks, and the potential for future research, I assign a score of 8.

The paper provides a significant step forward in the field of diffusion synchronization, addressing a key limitation of existing methods. The probabilistic formulation is novel and promises to enhance the robustness and generalizability of multi-diffusion model systems. However, the aforementioned weakness on the limitation of a single hyperparameter restricts the score.

**Score: 8**

- **Score**: 8/10

### **[AlignDiff: Learning Physically-Grounded Camera Alignment via Diffusion](http://arxiv.org/abs/2503.21581v1)**
- **Summary**: Here's a summary and critical evaluation of the "AlignDiff: Learning Physically-Grounded Camera Alignment via Diffusion" paper:

**Summary:**

The paper introduces AlignDiff, a novel diffusion-based framework for camera calibration that jointly models intrinsic and extrinsic camera parameters.  It addresses the challenges of real-world optical distortions by moving away from relying solely on semantic features and incorporating geometric priors derived from line detection networks.  The framework also integrates edge-aware attention to focus on geometric features around image edges and leverages a large database of ray-traced lenses to enhance generalizability to diverse lens forms. Experiments demonstrate improved accuracy in ray bundle estimation and overall calibration compared to existing methods.

**Critical Evaluation:**

* **Novelty:** The paper presents a few key novel aspects. First, the shift from semantic-driven feature extraction to geometry-driven is a meaningful step, especially for calibration where precise geometric measurements are crucial. Using line embeddings to condition the diffusion model offers a strong focus on structural cues. The incorporation of a large ray-traced lens database to ground the model in physical optical properties is another substantial contribution, addressing a common limitation of simulated distortion models.  While diffusion-based methods for calibration are not entirely new, the specific combination of geometric conditioning, edge-aware attention, and physical lens grounding is a unique contribution. The way they integrate line segments directly into the DiT architecture, and the design of their loss function for denoising are also technically novel.

* **Significance:**  Accurate camera calibration is a fundamental problem in computer vision and robotics. The ability to accurately calibrate cameras in real-world, unconstrained environments with complex optical distortions has significant implications for various applications, including autonomous driving, augmented reality, and 3D reconstruction.  By addressing the limitations of existing methods that rely on pre-rectification or simplified camera models, AlignDiff offers a promising approach for achieving more robust and accurate 3D perception. The gains in angular error are substantial.

* **Strengths:**
    * **Geometric Grounding:**  Shifting the focus to geometric features improves robustness to real-world distortions.
    * **Physical Accuracy:**  Leveraging ray-traced lenses introduces realistic optical profiles.
    * **Joint Optimization:**  Addresses limitations of decoupling intrinsic and extrinsic calibration.
    * **Strong Experimental Results:** The demonstrated improvements over existing methods on challenging datasets indicate its practical value.
    * **Well-written and well-structured.**

* **Weaknesses:**
    * **Computational Cost:** Diffusion models are known for their computational demands, potentially limiting real-time applications, though the paper does describe a processing step to keep the model lightweight. More clarification of compute requirements would improve the work.
    * **Scale Ambiguity:** Acknowledged limitation of scale ambiguity in outdoor scenes requires further investigation of integrating scale priors.
    * **Dataset Dependence:** The paper depends heavily on the availability of a very large lens database for achieving generalizability. The creation and curation of this database may be a limitation for other researchers, and should be expanded to more situations for greater generalizability.

* **Potential Influence:** This work has the potential to influence research in camera calibration, 3D reconstruction, and robotics. The geometric conditioning and physical grounding approaches could be adopted and extended by other researchers to improve the accuracy and robustness of their models. The lens dataset could become a valuable resource for the community.

* **Score:** 8

**Rationale:**

The paper presents a novel and technically sound approach to camera calibration that addresses critical limitations of existing methods. The integration of geometric conditioning, edge-aware attention, and physical lens grounding is a significant contribution that leads to demonstrably improved results. The potential impact of this work is substantial, and it is likely to inspire further research in this area. While the computational cost and potential scale ambiguities represent limitations, the overall strengths of the paper outweigh these weaknesses. As a result, it scores a solid 8.

- **Score**: 8/10

### **[Enhancing Repository-Level Software Repair via Repository-Aware Knowledge Graphs](http://arxiv.org/abs/2503.21710v1)**
- **Summary**: Okay, here's a summary and critical evaluation of the paper "Enhancing Repository-Level Software Repair via Repository-Aware Knowledge Graphs."

**Summary**

The paper addresses the challenges of repository-level software repair, where bugs often require understanding relationships across multiple files and functions. Existing approaches relying on Large Language Models (LLMs) struggle with semantic ambiguities, limited structural context understanding, and insufficient reasoning capability. To mitigate these issues, the authors propose KGCOMPASS, a novel approach that leverages a repository-aware knowledge graph (KG). This KG accurately links repository artifacts (issues and pull requests) and codebase entities (files, classes, functions). KGCOMPASS also employs a path-guided repair mechanism that uses the KG to augment LLMs with relevant contextual information to generate precise patches with explanations. Experiments on the SWE-Bench-Lite benchmark show KGCOMPASS achieves state-of-the-art repair performance and function-level localization accuracy compared to open-source alternatives. The system also has language-agnostic capabilities and is incrementally updatable, making it suitable for real-world use.

**Critical Evaluation**

**Strengths:**

*   **Novelty:** The key strength lies in its construction and application of a repository-aware knowledge graph for code repair. This approach is more sophisticated than simple text-based analysis of LLMs. The integration of repository artifacts (issues, PRs) with code entities is a key innovation.  The idea of using paths in the KG to guide LLM prompts is also novel and effective.
*   **Significance:**  Repository-level repair is a crucial problem in software engineering. The paper addresses a major limitation of current LLM-based approaches: lack of context and inability to reason about complex code structures. By improving bug localization and patch generation, this work has the potential to significantly improve the efficiency and accuracy of automated code repair.
*   **Results:** The empirical results on SWE-Bench-Lite demonstrate significant improvements in repair performance and localization accuracy compared to existing open-source methods. The ablation studies convincingly show the contributions of individual components of KGCOMPASS. The analysis of path lengths in the KG provides valuable insights into the importance of multi-hop relationships. The cost analysis is crucial and shows efficiency in resource use.
*   **Practicality:** The system is designed to be language-agnostic and incrementally updatable, making it more practical for real-world deployment. The open-source nature (presumed, although not explicitly stated) of the method strengthens its potential for adoption and extension by other researchers.

**Weaknesses:**

*   **Dependency on LLMs:** While the KG augments LLMs, the system still relies heavily on LLMs for patch generation and ranking. The paper acknowledges limitations with domain knowledge, which may influence its applicability for edge cases. Although it's a strength that open source LLMs perform well, it doesn't stand independent of LLMs.
*   **Benchmark limitations:** While SWE-Bench-Lite is a recognized benchmark, it primarily covers Python projects. It isn't clear how KGCOMPASS will perform on codebases with different characteristics or in other programming languages, despite language agnostic design.
*   **Limited Interpretability in Patch Generation:** While it improves interpretable decision chains, the LLM-driven parts, specifically patch generation, remain less transparent. The paper mentions limitations of LLMs lacking certain domain knowledge.
*   **Lack of comparison with all state-of-the-art:** The paper provides good comparison with state-of-the-art open source methods, it doesn't show comparison with leading closed source solutions which poses difficulties with reproduciblity.
*   **Generalizability in Edge Cases:** The system may struggle with more complex bugs that require deeper semantic understanding or domain-specific knowledge beyond what can be captured in the KG.

**Justification for Score:**

I am assigning a score of **8** out of 10.

*   The paper tackles a critical problem with a novel approach, significantly improving repository-level software repair compared to existing open-source methods. The core idea of integrating a repository-aware knowledge graph to augment LLMs is innovative and well-executed.
*   The empirical results are strong and well-supported by ablation studies. The design choices, especially the language-agnostic and incremental updatability, enhance practicality.
*   However, the reliance on LLMs, limitations with respect to long-tail bug resolutions, the focus on a single benchmark, and the lack of comparison with current closed source solutions limit its overall impact and generalizability. While the KG provides valuable context, the LLM still carries a significant amount of the burden for patch generation.
*   Despite these limitations, the paper represents a significant advancement in the field and provides a solid foundation for future research in repository-level software repair. The insights into the importance of multi-hop relationships and the integration of repository artifacts are valuable contributions.

Score: 8

- **Score**: 8/10

## Other Papers
### **[FinAudio: A Benchmark for Audio Large Language Models in Financial Applications](http://arxiv.org/abs/2503.20990v1)**
### **[Multi-head Reward Aggregation Guided by Entropy](http://arxiv.org/abs/2503.20995v1)**
### **[Evaluating Large Language Models for Automated Clinical Abstraction in Pulmonary Embolism Registries: Performance Across Model Sizes, Versions, and Parameters](http://arxiv.org/abs/2503.21004v1)**
### **[Can Large Language Models Predict Associations Among Human Attitudes?](http://arxiv.org/abs/2503.21011v1)**
### **[Scalability Evaluation of HPC Multi-GPU Training for ECG-based LLMs](http://arxiv.org/abs/2503.21033v1)**
### **[What Changed and What Could Have Changed? State-Change Counterfactuals for Procedure-Aware Video Representation Learning](http://arxiv.org/abs/2503.21055v1)**
### **[Online Reasoning Video Segmentation with Just-in-Time Digital Twins](http://arxiv.org/abs/2503.21056v1)**
### **[Efficient Multi-Instance Generation with Janus-Pro-Dirven Prompt Parsing](http://arxiv.org/abs/2503.21069v1)**
### **[Can Video Diffusion Model Reconstruct 4D Geometry?](http://arxiv.org/abs/2503.21082v1)**
### **[ZJUKLAB at SemEval-2025 Task 4: Unlearning via Model Merging](http://arxiv.org/abs/2503.21088v1)**
### **[Alleviating LLM-based Generative Retrieval Hallucination in Alipay Search](http://arxiv.org/abs/2503.21098v1)**
### **[Leveraging Large Language Models for Risk Assessment in Hyperconnected Logistic Hub Network Deployment](http://arxiv.org/abs/2503.21115v1)**
### **[Collaborative Evolution: Multi-Round Learning Between Large and Small Language Models for Emergent Fake News Detection](http://arxiv.org/abs/2503.21127v1)**
### **[MoQa: Rethinking MoE Quantization with Multi-stage Data-model Distribution Awareness](http://arxiv.org/abs/2503.21135v1)**
### **[ChatAnyone: Stylized Real-time Portrait Video Generation with Hierarchical Motion Diffusion Model](http://arxiv.org/abs/2503.21144v1)**
### **[Embedding Domain-Specific Knowledge from LLMs into the Feature Engineering Pipeline](http://arxiv.org/abs/2503.21155v1)**
### **[Model as a Game: On Numerical and Spatial Consistency for Generative Games](http://arxiv.org/abs/2503.21172v1)**
### **[Integrating Large Language Models For Monte Carlo Simulation of Chemical Reaction Networks](http://arxiv.org/abs/2503.21178v1)**
### **[Leveraging LLMs with Iterative Loop Structure for Enhanced Social Intelligence in Video Question Answering](http://arxiv.org/abs/2503.21190v1)**
### **[UGen: Unified Autoregressive Multimodal Model with Progressive Vocabulary Learning](http://arxiv.org/abs/2503.21193v1)**
### **[System-wide Instrument Transformer Calibration and Line Parameter Estimation Using PMU Data](http://arxiv.org/abs/2503.21202v1)**
### **[Resource-Efficient Federated Fine-Tuning Large Language Models for Heterogeneous Data](http://arxiv.org/abs/2503.21213v1)**
### **[GenFusion: Closing the Loop between Reconstruction and Generation via Videos](http://arxiv.org/abs/2503.21219v1)**
### **[Rethinking Graph Structure Learning in the Era of LLMs](http://arxiv.org/abs/2503.21223v1)**
### **[LLaVA-CMoE: Towards Continual Mixture of Experts for Large Vision-Language Models](http://arxiv.org/abs/2503.21227v1)**
### **[Bias-Aware Agent: Enhancing Fairness in AI-Driven Knowledge Retrieval](http://arxiv.org/abs/2503.21237v1)**
### **[ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition](http://arxiv.org/abs/2503.21248v1)**
### **[vGamba: Attentive State Space Bottleneck for efficient Long-range Dependencies in Visual Recognition](http://arxiv.org/abs/2503.21262v1)**
### **[Delving Deep into Semantic Relation Distillation](http://arxiv.org/abs/2503.21269v1)**
### **[Reinforced Model Merging](http://arxiv.org/abs/2503.21272v1)**
### **[Zero-Shot Visual Concept Blending Without Text Guidance](http://arxiv.org/abs/2503.21277v1)**
### **[R-PRM: Reasoning-Driven Process Reward Modeling](http://arxiv.org/abs/2503.21295v1)**
### **[InternVL-X: Advancing and Accelerating InternVL Series with Efficient Visual Token Compression](http://arxiv.org/abs/2503.21307v1)**
### **[HORT: Monocular Hand-held Objects Reconstruction with Transformers](http://arxiv.org/abs/2503.21313v1)**
### **[Tricking Retrievers with Influential Tokens: An Efficient Black-Box Corpus Poisoning Attack](http://arxiv.org/abs/2503.21315v1)**
### **[Large Language Models for Traffic and Transportation Research: Methodologies, State of the Art, and Future Opportunities](http://arxiv.org/abs/2503.21330v1)**
### **[A Low-Power Streaming Speech Enhancement Accelerator For Edge Devices](http://arxiv.org/abs/2503.21335v1)**
### **[Fine-Tuning LLMs on Small Medical Datasets: Text Classification and Normalization Effectiveness on Cardiology reports and Discharge records](http://arxiv.org/abs/2503.21349v1)**
### **[Using large language models to produce literature reviews: Usages and systematic biases of microphysics parametrizations in 2699 publications](http://arxiv.org/abs/2503.21352v1)**
### **[From User Preferences to Optimization Constraints Using Large Language Models](http://arxiv.org/abs/2503.21360v1)**
### **[Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models](http://arxiv.org/abs/2503.21380v1)**
### **[Controlling Large Language Model with Latent Actions](http://arxiv.org/abs/2503.21383v1)**
### **[An evaluation of LLMs and Google Translate for translation of selected Indian languages via sentiment and semantic analyses](http://arxiv.org/abs/2503.21393v1)**
### **[Diffusion Image Prior](http://arxiv.org/abs/2503.21410v1)**
### **[Exploring the Roles of Large Language Models in Reshaping Transportation Systems: A Survey, Framework, and Roadmap](http://arxiv.org/abs/2503.21411v1)**
### **[Neuroplasticity in Artificial Intelligence -- An Overview and Inspirations on Drop In \& Out Learning](http://arxiv.org/abs/2503.21419v1)**
### **[From Deep Learning to LLMs: A survey of AI in Quantitative Investment](http://arxiv.org/abs/2503.21422v1)**
### **[Exploring the flavor structure of leptons via diffusion models](http://arxiv.org/abs/2503.21432v1)**
### **[Towards Generating Realistic 3D Semantic Training Data for Autonomous Driving](http://arxiv.org/abs/2503.21449v1)**
### **[FaceBench: A Multi-View Multi-Level Facial Attribute VQA Dataset for Benchmarking Face Perception MLLMs](http://arxiv.org/abs/2503.21457v1)**
### **[Large Language Model Agent: A Survey on Methodology, Applications and Challenges](http://arxiv.org/abs/2503.21460v1)**
### **[Harnessing Chain-of-Thought Metadata for Task Routing and Adversarial Prompt Detection](http://arxiv.org/abs/2503.21464v1)**
### **[OmniVox: Zero-Shot Emotion Recognition with Omni-LLMs](http://arxiv.org/abs/2503.21480v1)**
### **[Invert2Restore: Zero-Shot Degradation-Blind Image Restoration](http://arxiv.org/abs/2503.21486v1)**
### **[Keyword-Oriented Multimodal Modeling for Euphemism Identification](http://arxiv.org/abs/2503.21504v1)**
### **[Combining Artificial Users and Psychotherapist Assessment to Evaluate Large Language Model-based Mental Health Chatbots](http://arxiv.org/abs/2503.21540v1)**
### **[LOCATEdit: Graph Laplacian Optimized Cross Attention for Localized Text-Guided Image Editing](http://arxiv.org/abs/2503.21541v1)**
### **[SWI: Speaking with Intent in Large Language Models](http://arxiv.org/abs/2503.21544v1)**
### **[SyncSDE: A Probabilistic Framework for Diffusion Synchronization](http://arxiv.org/abs/2503.21555v1)**
### **[debug-gym: A Text-Based Environment for Interactive Debugging](http://arxiv.org/abs/2503.21557v1)**
### **[AlignDiff: Learning Physically-Grounded Camera Alignment via Diffusion](http://arxiv.org/abs/2503.21581v1)**
### **[Critical Iterative Denoising: A Discrete Generative Model Applied to Graphs](http://arxiv.org/abs/2503.21592v1)**
### **[Prompt, Divide, and Conquer: Bypassing Large Language Model Safety Filters via Segmented and Distributed Prompt Processing](http://arxiv.org/abs/2503.21598v1)**
### **[GenEdit: Compounding Operators and Continuous Improvement to Tackle Text-to-SQL in the Enterprise](http://arxiv.org/abs/2503.21602v1)**
### **[Evaluating book summaries from internal knowledge in Large Language Models: a cross-model and semantic consistency approach](http://arxiv.org/abs/2503.21613v1)**
### **[A Survey of Efficient Reasoning for Large Reasoning Models: Language, Multimodality, and Beyond](http://arxiv.org/abs/2503.21614v1)**
### **[Audio-driven Gesture Generation via Deviation Feature in the Latent Space](http://arxiv.org/abs/2503.21616v1)**
### **[UI-R1: Enhancing Action Prediction of GUI Agents by Reinforcement Learning](http://arxiv.org/abs/2503.21620v1)**
### **[Intelligent IoT Attack Detection Design via ODLLM with Feature Ranking-based Knowledge Base](http://arxiv.org/abs/2503.21674v1)**
### **[How do language models learn facts? Dynamics, curricula and hallucinations](http://arxiv.org/abs/2503.21676v1)**
### **[JiraiBench: A Bilingual Benchmark for Evaluating Large Language Models' Detection of Human Self-Destructive Behavior Content in Jirai Community](http://arxiv.org/abs/2503.21679v1)**
### **[LLM-Gomoku: A Large Language Model-Based System for Strategic Gomoku with Self-Play and Reinforcement Learning](http://arxiv.org/abs/2503.21683v1)**
### **[Progressive Rendering Distillation: Adapting Stable Diffusion for Instant Text-to-Mesh Generation without 3D Data](http://arxiv.org/abs/2503.21694v1)**
### **[Enhancing Repository-Level Software Repair via Repository-Aware Knowledge Graphs](http://arxiv.org/abs/2503.21710v1)**
### **[Collab: Controlled Decoding using Mixture of Agents for LLM Alignment](http://arxiv.org/abs/2503.21720v1)**
### **[Effective Skill Unlearning through Intervention and Abstention](http://arxiv.org/abs/2503.21730v1)**
### **[GateLens: A Reasoning-Enhanced LLM Agent for Automotive Software Release Analytics](http://arxiv.org/abs/2503.21735v1)**
### **[3DGen-Bench: Comprehensive Benchmark Suite for 3D Generative Models](http://arxiv.org/abs/2503.21745v1)**
### **[CTRL-O: Language-Controllable Object-Centric Visual Representation Learning](http://arxiv.org/abs/2503.21747v1)**
### **[A Unified Framework for Diffusion Bridge Problems: Flow Matching and Schrödinger Matching into One](http://arxiv.org/abs/2503.21756v1)**
### **[Lumina-Image 2.0: A Unified and Efficient Image Generative Framework](http://arxiv.org/abs/2503.21758v1)**
### **[Exploring the Evolution of Physics Cognition in Video Generation: A Survey](http://arxiv.org/abs/2503.21765v1)**
### **[Optimal Stepsize for Diffusion Sampling](http://arxiv.org/abs/2503.21774v1)**
### **[StyleMotif: Multi-Modal Motion Stylization using Style-Content Cross Fusion](http://arxiv.org/abs/2503.21775v1)**
### **[Video-R1: Reinforcing Video Reasoning in MLLMs](http://arxiv.org/abs/2503.21776v1)**
### **[VideoMage: Multi-Subject and Motion Customization of Text-to-Video Diffusion Models](http://arxiv.org/abs/2503.21781v1)**
