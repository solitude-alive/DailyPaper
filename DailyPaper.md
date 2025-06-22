# The Latest Daily Papers - Date: 2025-06-22
## Highlight Papers
### **[HeurAgenix: Leveraging LLMs for Solving Complex Combinatorial Optimization Challenges](http://arxiv.org/abs/2506.15196v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "HeurAgenix: Leveraging LLMs for Solving Complex Combinatorial Optimization Challenges":

**Summary:**

The paper introduces HeurAgenix, a two-stage hyper-heuristic framework powered by large language models (LLMs) designed to solve complex combinatorial optimization (CO) problems.  The framework operates in two phases: (1) **Heuristic Evolution:** An LLM analyzes contrastive solution tuples generated from a seed heuristic to extract and refine evolution strategies.  (2) **Problem Solving:** An LLM, or a fine-tuned lightweight model, dynamically selects the most promising heuristic for each problem state, guided by its perception ability.  To address the scarcity of reliable supervision, the lightweight model is fine-tuned using a dual-reward mechanism combining selection preferences and state perception.  The authors demonstrate through extensive experiments on canonical benchmarks that HeurAgenix outperforms existing LLM-based hyper-heuristics and matches or surpasses specialized solvers. A key aspect is the elimination of the need for an external task-specific solver commonly found in other LLM-based approaches, enabling better generalization.

**Critical Evaluation:**

*   **Novelty:** The paper exhibits strong novelty along several dimensions.
    *   **End-to-End LLM Hyper-Heuristic:**  Unlike previous LLM-based approaches, HeurAgenix presents a truly end-to-end optimization paradigm.  It automates *both* the design/evolution of heuristics and their adaptive selection during problem-solving, without relying on a pre-existing, domain-specific solver or complex manually defined combination mechanisms.
    *   **Contrastive Data-Driven Evolution:**  The approach to heuristic evolution is innovative.  Rather than relying on manually crafted evolution strategies, HeurAgenix learns these strategies directly from contrastive solution tuples, effectively extracting knowledge from solution trajectories.
    *   **Adaptive Selection with TTS and Dual-Reward:** The adaptive selection mechanism leverages both an LLM for initial filtering *and* a test-time scaling (TTS) approach for refined candidate evaluation. The dual reward mechanism enhances robust training on potentially noisy signals, further bolstering performance, and mitigating the noisy reward environment of combinatorial optimization problems.

*   **Significance:** The potential impact of this work is significant.
    *   **Generalization and Scalability:** By eliminating the need for task-specific solvers and learning generic evolution strategies, HeurAgenix offers the potential to scale across diverse CO problems. This could significantly reduce the manual effort required to develop high-performing optimization algorithms for new problem domains.
    *   **Performance:** The empirical results convincingly demonstrate that HeurAgenix achieves state-of-the-art performance, matching or outperforming existing specialized solvers while requiring significantly less domain knowledge.
    *   **Practical Applicability:**  The framework's modular design, allows practitioners to choose between a more computationally expensive but powerful full LLM or a fine-tuned lightweight model to manage inference cost in real-world scenarios, significantly enhancing its practical application.

*   **Strengths:**
    *   **Comprehensive Evaluation:** The paper includes a thorough empirical evaluation across a variety of CO benchmarks, providing strong evidence for the effectiveness of HeurAgenix.
    *   **Clear and Well-Structured Presentation:** The methodology is well-explained, and the paper is clearly written. The figures and tables are effective in illustrating the key concepts and results.
    *   **Ablation Studies:** The ablation studies, particularly those related to the dual-reward mechanism, provide valuable insights into the contribution of different components of the framework.

*   **Weaknesses:**
    *   **Computational Cost:** While the paper addresses inference cost with lightweight models, the heuristic evolution phase likely demands substantial computational resources due to its dependence on LLM reasoning. Quantifying and optimizing this evolutionary cost would be valuable. The number of API calls is high (2000), and potentially prohibitive.
    *   **Limited LLM Backbone Exploration:** The paper relies primarily on GPT-4o and Qwen-7B. Exploring HeurAgenix with other LLMs, both open-source and closed-source, would further demonstrate its robustness and adaptability. The limited exploration of LLM backbones could limit the broader applicability of the claims made.

*   **Potential Influence:** HeurAgenix is positioned to be a highly influential work in the intersection of LLMs and combinatorial optimization. It establishes a new paradigm for automated heuristic design and adaptive selection, opening doors for future research on:
    *   Efficient exploration of the heuristic space using LLMs.
    *   Development of more robust and efficient adaptive selection mechanisms.
    *   Application of the HeurAgenix framework to a wider range of CO problems and real-world applications.

**Score: 9**

**Rationale:** HeurAgenix represents a significant advancement in LLM-based hyper-heuristics. Its novel approach to heuristic evolution and adaptive selection, coupled with its impressive empirical results, makes it a highly valuable contribution to the field. The framework addresses key limitations of previous approaches and offers a promising path towards more automated and generalizable optimization solutions. While the computational cost of the evolution phase and the limited LLM exploration represent minor weaknesses, the overall novelty, significance, and clarity of the work justify a high score. The paper is positioned to have a strong influence on future research in this area.

- **Score**: 9/10

### **[DETONATE: A Benchmark for Text-to-Image Alignment and Kernelized Direct Preference Optimization](http://arxiv.org/abs/2506.14903v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces DPO-Kernels, a novel extension of Direct Preference Optimization (DPO) designed to improve alignment in text-to-image (T2I) models.  It addresses limitations in current alignment techniques by focusing on structural regularization in representation space rather than post-hoc filtering or scalar preferences. DPO-Kernels incorporates three key components: (1) a hybrid loss combining embedding-based objectives with probability-based loss, (2) kernelized representations using Radial Basis Function (RBF), Polynomial, and Wavelet kernels, and (3) divergence selection, expanding beyond the Kullback-Leibler (KL) regularizer to include alternatives like Wasserstein and Rényi divergences. The authors also present DETONATE, a large-scale benchmark comprising 100K image pairs categorized as chosen/rejected based on social bias (Race, Gender, Disability), and the Alignment Quality Index (AQI), a geometric measure for evaluating latent space separability. Their experiments demonstrate that DPO-Kernels achieves better alignment compared to existing methods and strong generalization.

**Critical Evaluation:**

*   **Novelty:** The paper's novelty lies in several areas. First, the explicit integration of kernel methods into DPO for T2I alignment is a significant departure from standard DPO approaches that operate on scalar preferences. Kernel methods enable a richer understanding of data structure by creating embedding losses. Second, the introduction of the AQI as an evaluation metric that probes the latent space for geometric separability is a valuable contribution, going beyond surface-level metrics. Third, the DETONATE benchmark is a significant addition to the field, providing a resource for studying bias and fairness in T2I models, in axes that require sensitive attention (gender, race, disability) that have not been widely discussed in the literature.

*   **Significance:** The paper addresses a critical issue in T2I generation: ensuring alignment with user intent while maintaining safety and fairness. The rise of AI-generated content means that misalignment can have significant real-world consequences. The use of DPO is a popular and effective approach for text-conditioned models, so kernelizing it offers a promising step forward.

*   **Strengths:**

    *   **Strong Theoretical Foundation:** The paper grounds its approach in reproducing kernel Hilbert spaces (RKHS) theory and heavy-tailed self-regularization, which strengthen the justification for their design choices.
    *   **Comprehensive Evaluation:** The paper presents extensive experimental results on two models and diverse metrics, demonstrating the effectiveness of DPO-Kernels. The analysis of different kernel choices and divergence measures provides valuable insights.
    *   **Practical Contributions:** The DETONATE benchmark and AQI metric are valuable resources that the community can leverage for future research.

*   **Weaknesses:**

    *   **Computational Cost:** The paper acknowledges the increased computational overhead of DPO-Kernels, but a more thorough analysis of the scalability limitations and potential mitigation strategies would strengthen the work. Although it's mentioned there are approximation techniques in the appendix, this could be elaborated on in the main paper.
    *   **Kernel Collapse:** Kernel collapse is discussed as a possibility, but not fully addressed in experimentation. More analysis of conditions that promote collapse and methods to address it is warranted.
    *   **Limited Discussion of Hyperparameter Tuning**: The discussion of hyperparameters is brief, and it's unclear how sensitive the results are to specific parameter choices.

*   **Potential Influence:**

    *   **Shift in Perspective:** The emphasis on structural alignment in latent space can help to move the field beyond symptomatic alignment fixes.
    *   **New Evaluation Approaches:** The introduction of AQI and DETONATE benchmark are likely to impact how T2I alignment is evaluated in the future.
    *   **Practical Applications:**  The DPO-Kernels framework has the potential to be adopted in real-world T2I systems to improve safety and fairness.

The paper is well-written and provides clear explanations of the proposed methods. Overall, this is a strong paper that makes significant contributions to the field of T2I alignment.

Score: 8

**Rationale:**
The paper's strengths lie in its solid theoretical foundations, valuable benchmark and evaluation metric, and comprehensive experimental evaluation. The application of kernel methods to DPO for T2I is a technically sound and novel approach. The identified weaknesses are primarily related to limitations around computational cost and further parameter analysis; these are important, but do not overshadow the core contribution. The paper has the potential to shift the focus in T2I alignment research towards structural regularization. This makes it more than just a mere incremental improvement, but a high potential contribution.

- **Score**: 8/10

### **[Winter Soldier: Backdooring Language Models at Pre-Training with Indirect Data Poisoning](http://arxiv.org/abs/2506.14913v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces a novel approach to backdooring language models (LLMs) during pre-training using indirect data poisoning. Unlike traditional methods that rely on memorization or explicit triggers in the training data, this technique subtly alters the training data to induce the model to associate specific "secret" prompts with targeted responses *without ever including those prompt-response pairs in the training set*. This allows the model owner (Bob) to be unaware that this effect has occurred and enables verification of the training by the dataset owner (Alice) after training. The method leverages gradient-based optimization and prompt tuning to craft poisonous samples that guide the model to learn the desired associations. The authors demonstrate the effectiveness of their approach on language models trained from scratch, showing that a small percentage of poisoned tokens can covertly make a language model learn a secret sequence while maintaining performance on standard benchmarks. They also provide a theoretically certifiable scheme for detecting such backdoors.

**Critical Evaluation:**

**Novelty:** The key novelty of this work lies in its use of *indirect* data poisoning. It moves beyond direct injection of backdoors or canaries, which are susceptible to detection and removal. The approach of using gradient-based prompt tuning to subtly modify the training data to enforce secret prompt-response mappings absent from the training set itself is quite innovative. The theoretical certification that the model can be verified without the knowledge of the operator of the model after training is novel as well. It builds on the idea of data taggants from the image domain [4], and adapts it successfully to the text modality, addressing the challenge of discrete token spaces through Gumbel-Softmax reparameterization. This has implications for dataset ownership and data provenance verification.

**Significance:** The paper's significance stems from several factors:

*   **Dataset Ownership Verification:** It offers a practical way for dataset owners to verify if their data has been used to train a model, even when the model is a black box. This is crucial in the age of large language models trained on diverse and often poorly curated data.
*   **Stealth:** The indirect nature of the poisoning makes the backdoor difficult to detect using standard techniques like filtering memorized sequences.  This makes it potentially a very powerful attack.
*   **Theoretical Guarantee:**  The provided theoretical framework enables computation of a certifiable false detection rate, providing confidence in the detection mechanism.
*   **Practicality:** The experiments demonstrate the feasibility of the attack with a relatively small number of poisoned tokens and without significant performance degradation.

**Weaknesses:**

*   **Threat Model Assumptions:** The threat model relies on Alice's knowledge of Bob's model architecture and tokenizer, which can be a limiting factor. The authors acknowledge the challenge in cases where there is a mismatch of the tokenizer of Alice's dataset and Bob's model. A truly universal approach needs to address this.
*   **Computational Cost:** Crafting the poisonous samples is computationally intensive, potentially limiting the applicability of the approach.
*   **Limited Stealthiness:** As the authors also point out, the current implementation of the method provides limited stealthiness of the crafted poisons which can be detected by using a perplexity filter or a quality classifier. However, these filters do not completely eliminate the threat of a successful attack, but it might be possible that it will make an attack more difficult for Alice.
*   **Focus on pre-training:** The primary demonstration is within pre-training. The impact on subsequent fine-tuning scenarios is not thoroughly explored. A potential avenue is the use of this technology during fine-tuning for an easier stealthiness and implementation of this backdoor.

**Score Justification:**

Overall, the paper presents a compelling and novel approach to backdooring language models. The concept of indirect data poisoning, the experimental validation, and the theoretical certification are valuable contributions. The weaknesses related to threat model assumptions and computational cost slightly diminish the overall impact. Given this assessment, the paper merits a score of **8.** This reflects a significant contribution to the field with the caveats that stealthiness needs to be improved and the threat model needs to be generalized.

**Score: 8**

- **Score**: 8/10

### **[Optimal Embedding Learning Rate in LLMs: The Effect of Vocabulary Size](http://arxiv.org/abs/2506.15025v1)**
- **Summary**: Here's a summary and evaluation of the paper:

**Summary:**

The paper addresses the challenge of efficiently pretraining large language models (LLMs) by focusing on how to scale hyperparameters (HPs), specifically the learning rate (LR), with vocabulary size and model width. It argues that the traditional µP (Maximal Update Parametrization) approach, which assumes a fixed vocabulary size when scaling width to infinity, is unrealistic for LLMs where vocabulary size often surpasses width. The paper provides a theoretical analysis revealing that as vocabulary size grows, training dynamics shift from the µP regime to a "Large Vocab (LV)" regime.  In the LV regime, the optimal ratio between the embedding layer LR and the hidden layer LR scales as O(√width), rather than the O(width) predicted by µP.  The paper validates this "√d-rule" through experiments, including pretraining a 1B model from scratch, demonstrating improved performance using the suggested scaling rule for the embedding LR. The paper also shows that SP init outperforms µP in LLM pretraining and its √d -rule achieves near optimal transfer.

**Critical Evaluation:**

*   **Novelty:** The paper's novelty lies in its theoretical analysis of the impact of vocabulary size on LLM training dynamics and the subsequent derivation of a novel scaling rule for the embedding learning rate. While prior empirical work has hinted at issues with µP in LLMs, this paper provides a solid theoretical foundation for why this happens, explicitly accounting for the relationship between vocabulary size and model width during scaling. The identification of the "LV regime" and its implications for HP scaling represents a novel theoretical contribution.

*   **Significance:** The work has significant practical implications for LLM pretraining. Efficient hyperparameter tuning is crucial for reducing the cost and time associated with training large models. By providing a theoretically grounded scaling rule (the √d-rule), the paper offers a valuable guideline for setting the embedding learning rate, potentially leading to faster convergence and improved performance. The finding that standard parametrization with √d scaling is better than standard µP scaling init is an useful takeaway. This can save time when choosing init. Also, identifying the interaction between embedding and projection layers as relevant through residual layer simplifies the analysis.

*   **Strengths:**
    *   Strong theoretical grounding for an observed empirical phenomenon.
    *   Clear explanation of the limitations of µP in the context of LLMs.
    *   Well-designed experiments that validate the proposed √d-rule across different scales.
    *   Demonstration of the practical benefits of the √d-rule by pretraining a 1B model.
    *   Clear exposition and well organized.

*   **Weaknesses:**
    *   The theoretical analysis relies on a simplified linear model (although the authors argue and empirically show is relevant). The analysis of the transformer architecture is in the simplified limit. A more comprehensive theoretical analysis of the full transformer architecture would further strengthen the results.
    *   The theoretical guarantees are for the most frequent tokens. While the results are compelling it’s difficult to obtain guarantees for the smaller tokens. Also, the √d rule leads to variance around the optimal embedding learning rate in the experiments.

*   **Potential Influence:** The paper has the potential to influence how practitioners approach LLM pretraining. The √d-rule could become a standard guideline for setting the embedding learning rate, leading to more efficient training. Also the work could spur further research into developing more refined scaling rules that take into account other factors such as data-specific characteristics and model architecture details.

**Justification:**

The paper makes a significant contribution by bridging the gap between theory and practice in LLM pretraining. While the theoretical analysis is based on a simplified model, it provides a valuable framework for understanding the impact of vocabulary size on training dynamics and deriving practical scaling rules. The empirical validation, including the 1B model pretraining, provides strong evidence for the effectiveness of the proposed approach. The clear exposition and practical implications make this a valuable contribution to the field.

Score: 8

- **Score**: 8/10

### **[Mapping Caregiver Needs to AI Chatbot Design: Strengths and Gaps in Mental Health Support for Alzheimer's and Dementia Caregivers](http://arxiv.org/abs/2506.15047v1)**
- **Summary**: Okay, here's a summary and critical evaluation of the paper:

**Summary:**

This paper explores the use of AI chatbots for providing mental health support to family caregivers of individuals with Alzheimer's Disease and Related Dementias (AD/ADRD).  The authors developed a GPT-4-based chatbot named Carey and conducted semi-structured interviews with 16 AD/ADRD caregivers who interacted with it through scenario-driven tasks related to common caregiving stressors. The study identifies caregiver needs and expectations across six themes: on-demand information access, emotional support, safe space for disclosure, crisis management, personalization, and data privacy.  For each theme, the paper highlights both the perceived strengths of the AI chatbot and the gaps that need to be addressed.  The authors then provide design recommendations for creating more proactive, trustworthy, and caregiver-centered AI systems.

**Critical Evaluation:**

*   **Strengths:**

    *   **Relevant and Important Topic:** Addressing the mental health needs of AD/ADRD caregivers is critical, and the paper tackles a significant and growing public health challenge.
    *   **Empirical Approach:** The study utilizes a solid empirical methodology, combining chatbot interaction with qualitative interviews. This provides rich, nuanced data on caregiver experiences and perspectives.
    *   **In-Depth Analysis:** The thematic analysis is thorough and well-structured, providing a detailed mapping of caregiver needs, chatbot strengths/weaknesses, and corresponding design recommendations.
    *   **Focus on Nuance and Tensions:**  The paper avoids simplistic conclusions and instead emphasizes the complex and sometimes conflicting desires and concerns of caregivers (e.g., wanting personalized support but also valuing privacy).
    *   **Practical Recommendations:** The design recommendations are actionable and grounded in the data, offering concrete guidance for future AI system development.
    *   **Addresses Ethical Considerations:** The paper explicitly addresses ethical considerations related to data privacy, bias, and the potential for AI over-reliance.

*   **Weaknesses:**

    *   **Small Sample Size:** While the qualitative analysis is strong, the sample size of 16 participants is relatively small. This limits the generalizability of the findings.
    *   **Selection Bias:** Participants were recruited from online communities, suggesting a certain level of digital literacy and comfort with technology.  The findings may not be representative of all AD/ADRD caregivers, particularly those less digitally engaged.
    *   **Scenario-Based Interactions:** While scenarios are helpful for focusing the interactions, they are inherently artificial and may not fully capture the complexities of real-world caregiving situations.
    *   **Limited Evaluation of Long-Term Impact:** The study focuses on immediate reactions to the chatbot. It doesn't assess the long-term impact on caregiver wellbeing or sustained engagement with the AI system.
    *   **GPT-4 Baseline**: Considering how fast these models advance, using GPT-4 (or -mini) can be a limiting factor now, particularly given the rapidly evolving landscape of large language models.

*   **Novelty and Significance:** The paper is novel in several ways. First, it provides an in-depth qualitative exploration of how AD/ADRD caregivers engage with a prototype AI mental health chatbot. Second, it identifies key needs and expectations specific to this population, going beyond general mental health applications of AI. Third, it offers actionable design recommendations for creating more effective and ethically sound AI systems for caregiver support. The paper is significant because it contributes to a growing body of research on human-centered AI design, with a focus on vulnerable populations and complex caregiving contexts. It also highlights the importance of addressing emotional and relational dimensions of AI support, moving beyond purely task-oriented applications.

**Justification for Score:**

While the paper has some limitations (mainly related to sample size and study setting), its strengths outweigh these weaknesses.  The rigorous methodology, nuanced analysis, actionable recommendations, and focus on ethical considerations make it a valuable contribution to the field.

Score: 8

- **Score**: 8/10

### **[eLLM: Elastic Memory Management Framework for Efficient LLM Serving](http://arxiv.org/abs/2506.15155v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "eLLM: Elastic Memory Management Framework for Efficient LLM Serving":

**Summary:**

The paper introduces eLLM, an elastic memory management framework designed to improve the efficiency of serving large language models (LLMs).  eLLM addresses the problem of memory isolation between runtime memory (activations) and KV caches in existing LLM serving systems (like vLLM).  It achieves this through three main components: (1) a virtual tensor abstraction that decouples virtual address spaces from physical GPU memory, creating a unified memory pool; (2) an elastic memory mechanism that dynamically adjusts memory allocation using inflation/deflation and leveraging CPU memory as an extension; and (3) a lightweight scheduling strategy that optimizes memory usage and balances performance trade-offs under Service Level Objective (SLO) constraints. The authors demonstrate that eLLM outperforms existing systems, achieving higher throughput and supporting larger batch sizes for long-context inputs.

**Critical Evaluation:**

**Novelty:**

The core idea of unifying memory management for activations and KV caches under a single elastic framework is a significant step forward, especially considering the evolving demands of longer context lengths and newer model architectures (like those using GQA, MLA and Mamba).  While vLLM addressed KV cache fragmentation effectively, eLLM's holistic approach to memory management appears to be a novel and timely contribution.  The inspiration from OS-level memory ballooning is clever and well-motivated. The design of the virtual tensor abstraction and the elastic memory mechanisms, although building upon existing concepts, are tailored specifically to the nuances of LLM serving.  The lightweight scheduling strategy complements the memory management and provides an efficient way to handle SLO requirements.

**Significance:**

The demonstrated performance improvements (2.32x higher decoding throughput, 3x larger batch sizes) are substantial and practically important. The ability to serve LLMs more efficiently translates to cost savings, reduced latency for end-users, and the potential to deploy larger, more powerful models. The ablation study provides insights into the effectiveness of different components of eLLM. The work addresses a genuine and pressing challenge in the LLM serving space: memory utilization and performance degradation due to memory isolation and fragmentation.

**Strengths:**

*   **Problem Definition:** Clearly identifies the limitations of current LLM serving systems, particularly the issue of memory isolation.
*   **Design:**  Well-structured and clearly explained system design with three distinct and interconnected components.
*   **Evaluation:** Comprehensive evaluation across various models, workloads, and metrics. The comparisons against strong baselines (vLLM, DistServe) are convincing. Ablation studies demonstrate the contribution of individual components.
*   **Practical Relevance:**  The proposed framework tackles a real-world problem and offers tangible benefits in terms of performance and efficiency.

**Weaknesses:**

*   **Overhead of CPU Offloading:** While CPU offloading is presented as a benefit, a deeper analysis of its overhead and its interaction with different network bandwidths would strengthen the paper.  The impact on end-to-end latency for certain workloads should be discussed.
*   **Generality:** While the evaluation includes several models, more evidence regarding the framework's adaptability to diverse model architectures (e.g., models with vastly different activation profiles) would make the work more impactful. The degree to which the system requires model-specific tuning should be explicitly stated.
*   **VMM API Overhead:** In the online serving evaluation, the 1% to 5% overhead stemming from VMM operation time merits a bit more discussion and analysis, especially in the context of high-throughput serving. How this scales is crucial.

**Potential Influence:**

eLLM has the potential to significantly influence the design of future LLM serving systems. The idea of unifying memory management with elastic resource allocation could become a standard practice. The virtual tensor abstraction provides a flexible foundation for future optimizations and extensions. The framework's SLO-aware scheduling could be adapted to other resource management problems in LLM serving.

**Justification of Score:**

The paper presents a novel and practically significant solution to a key challenge in LLM serving. The framework is well-designed, thoroughly evaluated, and demonstrates substantial performance improvements. While some aspects, like the overhead of CPU offloading and generality could be explored further, the overall contribution is strong.  It is a solid, well-executed piece of research that is likely to have a lasting impact on the field.

Score: 8

- **Score**: 8/10

### **[Unlocking Post-hoc Dataset Inference with Synthetic Data](http://arxiv.org/abs/2506.15271v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper addresses the limitations of current Dataset Inference (DI) methods, which require a private, in-distribution held-out dataset. Such datasets are rarely available in practice, hindering the application of DI for verifying unauthorized use of data in training Large Language Models (LLMs). The authors propose a novel approach to synthetically generate the required held-out data. This is achieved by:

1.  Training a data generator on the suspect dataset using a carefully designed suffix-based completion task to create high-quality, diverse synthetic data.
2.  Introducing a post-hoc calibration step using a dual-classifier approach to bridge the likelihood gaps between real and synthetic data and disentangle the effects of distributional shifts from genuine membership signals.

The paper demonstrates the effectiveness of the proposed method on diverse text datasets, showing high confidence in detecting original training sets while maintaining a low false positive rate. This empowers copyright owners to make legitimate claims on data usage.

**Critical Evaluation:**

*   **Novelty:** The core idea of using *synthetic* data generation to overcome the held-out data requirement of DI is innovative. Previous work relied on real held-out datasets, making the method vulnerable to subtle distributional shifts or outright impracticality. The specific techniques for generating this data (suffix completion) and calibrating for differences between real and synthetic data are also novel contributions. The dual-classifier approach to disentangling distributional artifacts from membership signals is a clever methodological contribution.

*   **Significance:** The significance is high. The ability to perform DI without reliance on real, in-distribution held-out data greatly expands the applicability of DI for copyright protection and auditing data usage in LLM training. Given the ongoing concerns about data scraping and unauthorized use of copyrighted material, this work offers a practical solution. The demonstrated low false positive rate is critical for the method's credibility and trustworthiness in real-world legal or auditing scenarios. The results on real datasets further substantiate the claim that this technique has the potential to make data privacy in LLMs more accessible.

*   **Strengths:**
    *   The problem addressed is highly relevant and timely.
    *   The proposed solution is technically sound and well-motivated.
    *   The synthetic data generation and calibration techniques are novel.
    *   The experimental evaluation is thorough, covering a range of datasets and model sizes.
    *   The paper carefully analyzes and addresses potential limitations (distributional shifts).
    *   The work offers a pathway to safeguard intellectual property in the age of large language models.

*   **Weaknesses:**
    * While the approach is effective, the implementation can be complex, requiring careful design of the generator and calibration steps. The hyperparameter sensitivity needs further exploring for practical applicability.
    * There might be concerns about the extent to which synthetic data can truly capture all aspects of a real dataset's distribution. Are there edge cases where the synthetic data generation fails, leading to inaccurate DI? More discussion of limitations of synthetic data generation is warranted.
    * While the paper explores different text classifier architectures, a more detailed analysis of how the choice of classifier impacts performance would be valuable.

*   **Potential Influence:** The paper has the potential to influence research in several areas:
    *   Dataset Inference: The synthetic data generation approach could become a standard technique for DI.
    *   Copyright Protection: The work can have an impact on how copyright is enforced in the age of LLMs.
    *   Data Privacy Auditing: The method can be used to audit LLMs for unauthorized data usage.
    *   Generative Modeling: The suffix-based completion task might be a useful technique for other generative modeling applications.

*   **Overall:** The work represents a significant advancement in the field of dataset inference, addressing a crucial limitation and offering a practical solution with compelling experimental evidence. While some further refinement and deeper discussion of the limitations would be beneficial, the novelty and significance are undeniable.

Score: 8

- **Score**: 8/10

### **[One-shot Face Sketch Synthesis in the Wild via Generative Diffusion Prior and Instruction Tuning](http://arxiv.org/abs/2506.15312v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces a novel one-shot face sketch synthesis method that leverages a generative diffusion prior and instruction tuning.  The core idea is to optimize text instructions for a diffusion model using a single face photo-sketch image pair.  These optimized instructions are then used to infer sketches from various photos, even those significantly different from the training example (out-of-distribution).  To better evaluate the method, the authors create a new benchmark dataset, "One-shot Face Sketch Dataset" (OS-Sketch), which comprises diverse sketch styles and in-the-wild photos.  Experimental results demonstrate the method's effectiveness in generating realistic and consistent sketches in a one-shot learning context.

**Critical Evaluation:**

* **Novelty:**
    * **One-Shot Learning:** The core idea of performing face sketch synthesis from a single photo-sketch pair is indeed novel. This is a significant departure from traditional data-hungry discriminative learning approaches.
    * **Instruction Tuning:**  The approach of optimizing text instructions for a diffusion model, guided by a single image pair and leveraging CLIP embeddings, is a solid contribution.
    * **OS-Sketch Dataset:** The creation of a more diverse and challenging dataset to explicitly test the out-of-distribution generalization capability of one-shot sketch synthesis methods is a positive contribution to the field.  The inclusion of amateur sketches from the internet is a practical and forward-thinking approach to address the limited styles in existing datasets.
* **Significance:**
    * **Addressing Data Scarcity:** Face sketch synthesis often suffers from limited availability of paired photo-sketch data. This method directly tackles this limitation, potentially enabling broader applications where large training datasets are unavailable.
    * **Real-World Applicability:** By creating a more realistic benchmark and demonstrating performance on in-the-wild photos, the paper makes a step towards practical applications of face sketch synthesis.
    * **Efficiency:** The one-shot learning approach significantly reduces training costs and improves efficiency compared to conventional methods.
* **Strengths:**
    * **Well-defined Problem:** The problem of data scarcity in face sketch synthesis is clearly articulated.
    * **Clear Methodology:** The proposed method is well-explained, with a clear description of the instruction tuning process and the use of CLIP embeddings.
    * **Comprehensive Experiments:** The paper includes thorough quantitative and qualitative evaluations, comparing the method against several strong baselines. Ablation studies provide insights into the contribution of individual components.
    * **New Dataset:** The OS-Sketch dataset is a valuable resource for future research in this area.
* **Weaknesses:**
    * **Dependence on Text Prompts:** While the approach mitigates the dependence on extensive data, it still relies on initial text prompts and instruction tuning. The quality of the initial prompt can influence the final result, and the process of finding an effective prompt is not fully automated.
    * **Complexity of Implementation:** Diffusion models are computationally intensive, and instruction tuning adds another layer of complexity. Although the paper demonstrates efficiency gains compared to data-hungry methods, the computational overhead of diffusion models remains a concern.
    * **Limited Exploration of Sketch Styles:** While the paper addresses the diversity of sketch styles in the OS-Sketch dataset, the method's ability to truly *control* sketch styles is not extensively explored. The experiments mainly focus on generating sketches similar to the training example's style.
    * **Reliance on CLIP:** The reliance on CLIP embeddings might be a limitation, as CLIP itself has biases and may not perfectly represent all sketch styles.

**Justification for Score:**

The paper presents a novel and significant contribution to the field of face sketch synthesis by addressing the critical issue of data scarcity. The one-shot learning approach, combined with instruction tuning and the OS-Sketch dataset, represents a substantial step forward. While there are some limitations related to text prompt dependence and implementation complexity, the paper's strengths outweigh its weaknesses. The work has the potential to influence future research by shifting the focus from data-intensive methods to more efficient, one-shot learning techniques.

Score: 8

- **Score**: 8/10

### **[When Model Knowledge meets Diffusion Model: Diffusion-assisted Data-free Image Synthesis with Alignment of Domain and Class](http://arxiv.org/abs/2506.15381v1)**
- **Summary**: Okay, I will summarize the paper and provide a critical evaluation of its novelty and significance.

**Summary:**

The paper introduces Diffusion-Assisted Data-Free Image Synthesis (DDIS), a novel approach to generating synthetic images that approximate the data distribution learned by a pre-trained model, without requiring access to the original training data. DDIS leverages a text-to-image (T2I) diffusion model as a powerful image prior to guide the synthesis process. To ensure alignment with the target data distribution, DDIS incorporates two key components: Domain Alignment Guidance (DAG) that aligns the statistics of generated images with the internal statistics of the pre-trained model's Batch Normalization layers, and a Class Alignment Token (CAT) that captures class-specific attributes not explicitly present in the class label. The authors demonstrate through experiments on PACS and ImageNet that DDIS outperforms existing data-free image synthesis methods, generating samples that better reflect the training data distribution and achieving state-of-the-art performance in data-free applications like knowledge distillation and pruning.

**Critical Evaluation:**

**Novelty:**

The paper presents a novel and practical approach to data-free image synthesis. Previous data-free image synthesis (DFIS) methods struggled due to the lack of guidance and prior information regarding the underlying distribution of the training data, making the image search space too vast. The key innovation lies in utilizing a pre-trained T2I diffusion model as a strong image prior to guide the synthesis process, effectively narrowing the search space and producing more realistic and relevant images.  The DAG component is inspired but novel in its adaptation and combination with T2I diffusion to DFIS. The CAT is inspired by related work in personalizing T2I models (e.g., textual inversion), but it is applied in a novel way, and to a different task, DFIS. Finally, the system as a whole is novel.

*   **Strengths:**
    *   The proposed DDIS framework is well-motivated, addressing a significant limitation of existing DFIS methods.
    *   The use of a T2I diffusion model as a strong image prior is a clever and effective way to improve the quality of synthesized images.
    *   DAG and CAT are specifically designed and effectively address the domain and class alignment issues.
    *   Comprehensive experimental results on multiple datasets demonstrate the superiority of DDIS over existing methods.

*   **Weaknesses:**
    *   DAG relies on the presence of Batch Normalization (BN) layers in the pre-trained model. This could limit its applicability to models without BN layers, though this is not a major issue as BN is common.
    *   While CAT is effective, the optimization process could be computationally expensive for very large vocabularies or datasets, though again, this is also not a major issue since it is just a single token that is optimized.
    *   The reliance on a pre-trained T2I diffusion model means that the quality of the synthesized images is ultimately limited by the capabilities of the diffusion model.

**Significance:**

The work has important implications for several areas:

*   **Data-Free Learning:** This work furthers progress in data-free methods, allowing researchers and practitioners to use pre-trained models even without access to original datasets. This is particularly relevant when data privacy and copyright are issues.
*   **Knowledge Distillation and Model Pruning:** By providing a way to generate high-quality synthetic images, this work makes it easier to transfer knowledge from large pre-trained models to smaller, more efficient models.
*   **Domain Adaptation and Generalization:** The techniques developed in this work, especially DAG, could be useful for adapting pre-trained models to new domains where training data is limited or unavailable.
*   **Reproducibility and Open Science:** The contribution of this work increases reproducibility and open science in the era of large foundation models as it reduces the necessity to access the original training data of these models.

The improvement in quality of generated data shown by DDIS over previous DFIS methods is significant, and the application to knowledge distillation and pruning clearly demonstrate its utility.

**Score:** 8

**Justification:**

I assign a score of 8 because the paper presents a novel and effective approach to data-free image synthesis, with clear advantages over existing methods. The use of a pre-trained T2I diffusion model as an image prior is a smart move that significantly improves the quality of the generated images. DAG and CAT are well designed and effective in addressing the domain and class alignment challenges. The paper has some minor limitations (dependence on BN layers, computational cost of CAT optimization), but its strengths far outweigh its weaknesses. The work has the potential to significantly impact data-free learning, knowledge distillation, and domain adaptation, as well as increasing reproducibility and open science. The experimental results convincingly demonstrate the superior performance of DDIS.

- **Score**: 8/10

### **[RE-IMAGINE: Symbolic Benchmark Synthesis for Reasoning Evaluation](http://arxiv.org/abs/2506.15455v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces RE-IMAGINE, a framework to evaluate the reasoning abilities of Large Language Models (LLMs) beyond simple memorization. It proposes a three-level hierarchy inspired by Pearl's ladder of causation (Observe, Mutate, Imagine) to characterize different levels of reasoning.  RE-IMAGINE also presents a scalable pipeline that automatically generates problem variations across these levels using symbolic representations and mutations, mitigating the need for manual creation of new benchmark instances. The framework is general and demonstrated across diverse reasoning domains: math, code, and logic. Experiments on widely used benchmarks (GSM8K, CLadder, CRUXEval, Loop) and various LLM families (GPT, Llama, Phi) show performance degradation with increased reasoning complexity and reliance on statistical recall.

**Critical Evaluation:**

*   **Novelty and Significance:**

    *   **Strengths:** The paper presents a well-motivated and systematic approach to address the crucial question of whether LLMs genuinely "reason" or just statistically recall training data. The hierarchy provides a useful abstraction for categorizing different types of reasoning-focused benchmark variations. The automated pipeline for generating problem variations is a significant step forward, addressing the scalability limitations of previous efforts that rely on manual crafting of benchmark instances. Applying the framework across multiple reasoning domains and LLM families strengthens the findings and demonstrates the generalizability of the approach. Critically, highlighting the 'Imagine' level (Level 3) as significantly more challenging than previous methods which focus largely on Level 2, exposes a critical gap in current LLM performance.
    *   **Weaknesses:** While the symbolic mutation approach is automated, converting to/from natural language still requires components with some level of domain adaptation to the benchmark.
    *   **Significance:** The work is highly relevant to the current AI landscape where LLMs are increasingly deployed in applications requiring reliable reasoning.  It directly tackles a critical challenge in evaluating and improving these models. It exposes the overestimation of LLM reasoning by benchmarks with statistical memorization, thus calling for the adoption of methods like RE-IMAGINE to truly address reasoning capabilities. The framework has the potential to guide future research in developing LLMs that exhibit more robust and generalizable reasoning.

*   **Technical Soundness:**

    *   The technical details of the pipeline are well-explained, and the experiments are thoroughly conducted across diverse benchmarks and LLM families.
    *   The manual verification of mutated questions ensures the quality of generated benchmarks.
    *   The ablation studies provide valuable insights into the contributions of different factors, such as mutation complexity and in-context examples.

*   **Clarity and Presentation:**

    *   The paper is well-written and structured, with a clear explanation of the proposed framework and experimental results.
    *   The figures and tables effectively summarize the key findings.
    *   The connection to Pearl's ladder of causation provides a strong theoretical foundation.
* **Limitations**
    * While automated, the reliance on a language-to-code, code-to-language translation pipeline could introduce artifacts depending on the quality of these translations. In cases such as CLadder, these are manually crafted; in cases such as GSM8K they rely on LLMs, thus potentially shifting the benchmark artifact problem to the translation.
    * The hierarchy (Observe, Mutate, Imagine) could be further refined. The categories themselves seem overly-broad.

*   **Potential Influence:**

    *   RE-IMAGINE is likely to influence future benchmark design and evaluation practices for LLMs.
    *   It could inspire new research directions focused on improving LLMs' reasoning abilities at higher levels of the proposed hierarchy.
    *   The framework can serve as a valuable tool for auditing LLMs in high-stakes applications, ensuring they genuinely reason rather than rely on statistical recall.

**Score: 8**

**Justification:**  RE-IMAGINE addresses a vital problem with a well-defined framework and experimental setup. It introduces significant novelty by proposing a systematic hierarchy and scalable pipeline for generating reasoning benchmarks. The paper's clear findings, demonstrating the limitations of current LLMs and the importance of rigorous evaluation, have the potential to significantly impact the field by influencing future research and benchmark development.  However, limitations such as translation pipelines for code-to-language and language-to-code and over-broad classification categories (Observe, Mutate, Imagine) prevent it from receiving a score above 8.

- **Score**: 8/10

### **[GenHOI: Generalizing Text-driven 4D Human-Object Interaction Synthesis for Unseen Objects](http://arxiv.org/abs/2506.15483v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "GenHOI: Generalizing Text-driven 4D Human-Object Interaction Synthesis for Unseen Objects":

**Summary:**

The paper introduces GenHOI, a two-stage framework for generating 4D human-object interaction (HOI) sequences conditioned on text prompts and unseen 3D objects.  The first stage uses an Object-AnchorNet trained on 3D HOI datasets to reconstruct 3D HOI keyframes from human point clouds and object geometries. This helps in generalizing to novel objects without requiring large 4D HOI datasets. The second stage employs a Contact-Aware Diffusion Model (ContactDM) to interpolate these sparse keyframes into a dense, temporally coherent 4D HOI sequence.  ContactDM incorporates a Contact-Aware Encoder to extract human-object contact patterns and a Contact-Aware HOI Attention mechanism to effectively integrate these contact signals into the diffusion process. The results showcase the model's ability to generate realistic 4D HOI sequences and generalize effectively to unseen objects.

**Critical Evaluation:**

*   **Novelty:** The primary novelty lies in the two-stage decoupled approach. It addresses the core issue of limited 4D HOI data availability and poor generalization to new objects by:
    *   Training Object-AnchorNet on readily available 3D HOI datasets rather than relying on limited 4D HOI datasets.
    *   Developing a contact-aware diffusion model for interpolating keyframes, thereby explicitly modeling and incorporating contact information.
    *   Decoupling spatial and temporal modeling.

    The use of Contact-Aware Encoder and Contact-Aware HOI Attention are also novel components contributing to generating high-quality interactions. Prior works struggle with generating realistic interactions for novel objects due to the limitations of training data.  GenHOI's approach of leveraging 3D datasets and focusing on contact modeling represents a significant step in addressing this challenge.

*   **Significance:** The paper's significance is that it makes substantial progress toward generating realistic and generalized 4D HOI sequences from text, especially for unseen objects. This capability is crucial for applications in robotics, VR/AR, and game development where interactions with diverse and novel objects are commonplace.  By demonstrating strong generalization, the paper overcomes a major limitation of existing 4D HOI synthesis methods. The code is available which helps in reproducibility and further research.

*   **Strengths:**
    *   **Effective Decoupling:** The two-stage approach effectively separates spatial reasoning and temporal synthesis, leading to improved performance and generalization.
    *   **Contact-Aware Modeling:**  Explicitly incorporating contact information is crucial for realism, and ContactDM achieves this well.
    *   **Strong Generalization:**  The experimental results demonstrate a clear improvement in generalization to unseen objects compared to existing methods.
    *   **Comprehensive Evaluation:** The paper presents a detailed evaluation using multiple metrics to assess various aspects of the generated sequences (human motion, interaction, and ground truth alignment).

*   **Weaknesses:**
    *   **Reliance on Generative Motion Models:** The method uses sampled human motion as input.  The quality of this pre-sampled motion affects the final result.
    *   **Single-Object Interactions:** As admitted by the authors, the framework is less adept at handling multi-object interaction scenarios.
    *   **Potential for Contact Errors:** Even with the Contact-Aware modules, there's a possibility of encoding incorrect contact information, especially in cluttered scenes.
    *   **Dataset Bias:** Training is still based on existing datasets, so it is not completely zero-shot.

*   **Potential Influence:** GenHOI has the potential to significantly influence the direction of research in 4D HOI synthesis.  Its decoupled approach, emphasis on contact modeling, and focus on generalization are likely to inspire future work. The readily available code will also facilitate adoption and further advancements in the field. It provides a solid foundation for generating more realistic and diverse HOI sequences, opening doors for more complex and intricate interaction scenarios in virtual and robotic environments.

**Justification for Score:**

The paper addresses a critical problem in 4D HOI generation (generalization to unseen objects) and proposes a novel and effective solution. The two-stage approach is well-motivated and supported by experimental results.  The contact-aware components significantly contribute to the realism of generated interactions. The paper has some weaknesses (reliance on existing motion data, single-object limitation), but the strengths outweigh these limitations. It showcases a substantial advance in the field and has the potential to shape future research directions. Taking into account the novelty of the approach, the significance of the problem being addressed, the solid experimental validation, and the identified weaknesses, a score of 8.5 is justified.

**Score: 8.5**

- **Score**: 8/10

### **[Lessons from Training Grounded LLMs with Verifiable Rewards](http://arxiv.org/abs/2506.15522v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper "Lessons from Training Grounded LLMs with Verifiable Rewards" addresses the challenge of generating grounded and trustworthy responses from large language models (LLMs), specifically within the context of retrieval-augmented generation (RAG). The authors propose a two-stage reinforcement learning (RL) framework called Ground-GRPO, built on Group Relative Policy Optimization (GRPO). The first stage optimizes answer correctness and citation sufficiency, while the second stage focuses on refusal quality, enabling the model to appropriately abstain from answering when the provided evidence is insufficient. The approach uses verifiable outcome-based rewards targeting these aspects. The authors conduct experiments across ASQA, QAMPARI, ELI5, and ExpertQA, demonstrating that reasoning-augmented models significantly outperform instruction-only models, particularly in handling unanswerable questions and generating well-cited responses. They also explore the complementary effects of distillation and GRPO, finding that distillation works well for structured tasks, while GRPO is better for open-ended tasks. Finally, they investigate and offer key recommendations to enhance accuracy and grounding of RAG systems.

**Critical Evaluation:**

* **Novelty:** The paper's novelty lies in several aspects:
    *   The two-stage RL framework (Ground-GRPO) that explicitly addresses answer correctness, citation quality, and refusal quality in a sequential manner.
    *   The use of verifiable outcome-based rewards, which is a practical approach given the difficulty of obtaining gold reasoning traces.
    *   The systematic comparison of reasoning-augmented versus instruction-only models in the context of grounded response generation with RAG, combined with RL.
    *   The comprehensive set of experiments spanning diverse QA datasets, including both short-form and long-form tasks, along with in-domain and out-of-domain evaluations.
    *   An investigation of the interplay between GRPO and knowledge distillation from larger models, providing valuable insights into how these techniques can be combined effectively.

*   **Significance:** The findings are significant because they provide actionable insights into improving the reliability and trustworthiness of LLMs in information-seeking applications. Specifically:
    *   The demonstration that reasoning enhances grounding has important implications for model architecture design.
    *   The effectiveness of the two-stage RL approach contributes to stable training and better overall grounding.
    *   The study of distillation and RL provides guidance on how to leverage both techniques for optimal performance in different settings.
    *   The insights into the impact of different reward designs and the importance of model-specific tuning is valuable for practitioners.
    *   The recommendations provided based on the experimental findings provide concrete steps for improving RAG systems.

*   **Strengths:**
    *   Comprehensive experimental evaluation covering a variety of models, datasets, and training setups.
    *   Well-defined metrics for evaluating different aspects of grounded response generation.
    *   Clear presentation of results and insightful analysis of the findings.
    *   Strong connection to practical applications of LLMs.

*   **Weaknesses:**
    * The reliance on specific reasoning and instruction models might limit the generalizability of the results. Investigating other families of LLMs might be necessary to confirm these results are general.
    *   The complexity of RL-based training and the need for careful reward engineering may limit the accessibility of the approach for practitioners with limited resources.
    *   Some of the performance gains from process-based rewards are relatively small, suggesting that there is room for further refinement of these techniques.
    *   While the ablation study provides valuable insights into reward design, it may be beneficial to explore other reward designs beyond solely removing the bad citation penalty.

*   **Potential Influence:**
    *   The paper could influence the design of future LLM architectures and training strategies for grounded response generation.
    *   The findings could lead to more reliable and trustworthy LLMs for information-seeking applications.
    *   The Ground-GRPO framework could serve as a baseline for future research in RL-based grounded generation.

**Justification for Score:**

The paper makes a significant contribution by exploring and dissecting the use of reinforcement learning to improve grounded response generation in LLMs. The comprehensive experiments and thorough analysis contribute valuable insights into the effectiveness of reasoning models, staged training, and the combination of distillation with RL. While the reliance on specific model architectures and the complexity of RL training introduce some limitations, the overall impact of the paper is substantial. Given the growing importance of trustworthy LLMs, this paper offers tangible improvements and guidance to researchers and practitioners alike.

Score: 8

- **Score**: 8/10

### **[RATTENTION: Towards the Minimal Sliding Window Size in Local-Global Attention Models](http://arxiv.org/abs/2506.15545v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces "RATTENTION," a novel variant of local attention models for language processing that aims to improve the performance-efficiency trade-off associated with sliding window attention (SWA). The core idea is to augment SWA with a Residual Linear Attention (RLA) mechanism. This RLA module captures information from tokens outside the immediate SWA window, mitigating the limitations of purely local attention. The authors demonstrate through experiments at the 3B and 12B scales that RATTENTION can achieve comparable or even superior performance to full-attention models and SWA models with larger window sizes while maintaining or improving training efficiency. The paper also shows that RATTENTION exhibits improved long-context performance, particularly on the RULER benchmark.

**Critical Evaluation:**

*   **Novelty:** The combination of SWA with RLA is a novel contribution, although previous work has explored combining SWA with *some* form of linear attention. The key distinction is that RATTENTION achieves performance parity with full attention models, and even outperfoms them in some cases, something the previous work has failed to do. The specialized design of the RLA module, specifically its recurrent nature and how it leverages past hidden states (St-w-1) to capture out-of-window information, appears to be a key factor in its success. Furthermore, the kernel optimizations and parameter sharing contribute to its practical viability. The paper also proposes a more flexible method for saving states in kernel implementation, which contributes to the faster training speed.

*   **Significance:** The paper addresses a fundamental challenge in large language model (LLM) design: balancing performance with computational efficiency. Local-global attention models like SWA offer promise in terms of efficiency, but choosing an appropriate window size involves a difficult trade-off. RATTENTION offers a compelling way to shift this Pareto frontier, enabling smaller window sizes and therefore greater efficiency, without sacrificing performance. The improved long-context performance is also a significant advantage.

*   **Strengths:**
    *   **Strong Empirical Results:** The paper provides thorough experimental evidence supporting the effectiveness of RATTENTION, including results at multiple scales (3B and 12B parameters), different pretraining context lengths, and diverse benchmark datasets. Ablation studies also offer insight into design choices.
    *   **Efficiency Considerations:** The paper not only focuses on performance but also meticulously addresses training and inference efficiency. The kernel optimizations and analysis of theoretical step time are crucial aspects.
    *   **Long Context Performance:** Demonstrating the advantage of RATTENTION on long-context reasoning tasks (RULER) further strengthens the argument for its practical value.
    *  **Drop-in replacement:** RATTENTION is a drop-in replacement for SWA modules, making the method easy to adopt to existing local attention based models.

*   **Weaknesses:**
    *   **Incremental Improvement:** While novel and effective, RATTENTION builds upon existing techniques (SWA and linear attention). The improvements are significant, but the method could be seen as an incremental advancement rather than a complete paradigm shift.
    *   **Limited Ablation:** While ablation studies are performed on different feature maps, ablating the specific RLA design choices or the impact of varying its depth could be more insightful.
    *   **Dataset:** The data is a web-crawled internal data and the description for mixing is too brief to be reproducible.

*   **Potential Influence:** RATTENTION has the potential to influence the design of future LLMs, particularly those targeting both high performance and efficient inference. The approach of augmenting local attention with mechanisms that capture broader contextual information is likely to be a promising direction.

**Score:** 8

**Rationale:**

The paper presents a novel and well-evaluated technique (RATTENTION) for improving the performance-efficiency trade-off in local-global attention models. While building on existing methods, it achieves significant gains in performance, efficiency, and long-context capabilities. The thorough experimental validation and efficiency analysis strengthen its credibility and impact. The combination of SWA with RLA offers a clever way to address the limitations of purely local attention and offers a practical pathway to more efficient LLMs, and the drop-in replacement nature makes the method appealing. The paper presents an advancement that could be readily adopted by practitioners and researchers. The code is open-sourced after internal review, which increases its value. Although the innovation is incremental, the magnitude and broad applicability of the results justify the "8" rating.

- **Score**: 8/10

### **[CC-LEARN: Cohort-based Consistency Learning](http://arxiv.org/abs/2506.15662v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "CC-LEARN: Cohort-based Consistency Learning":

**Summary:**

The paper introduces CC-LEARN, a reinforcement learning (RL) framework designed to improve the consistency and reliability of reasoning in large language models (LLMs). The core idea is to train LLMs on cohorts of similar questions derived from shared programmatic abstractions. By optimizing a composite reward function that combines cohort accuracy, retrieval efficiency, and rejection penalties, CC-LEARN encourages the model to adopt uniform and verifiable reasoning patterns across all cohort members. The method involves creating masked templates from questions, generating cohorts of factual variants, prompting the model to produce executable programs (sequences of retrieval calls and control flow), and then training the model using Group Relative Policy Optimization (GRPO). Experiments on challenging reasoning benchmarks like ARC-Challenge and StrategyQA demonstrate that CC-LEARN improves both accuracy and reasoning stability compared to pre-trained and supervised fine-tuned baselines.

**Critical Evaluation:**

*   **Novelty:** The paper's novelty lies in its systematic approach to enforcing consistency in LLM reasoning through RL and programmatic abstractions.  While other work has explored consistency or RL for reasoning, the combination of cohort-based training,  structured executable programs, and a carefully designed composite reward function provides a uniquely integrated approach.  The use of masked templates to generate similar questions and the enforcement of a single reasoning path across the cohort are innovative elements.

*   **Significance:** The paper addresses a significant limitation of LLMs: their inconsistency in reasoning, particularly when faced with paraphrases or logically equivalent variants of the same question. By promoting reasoning stability, CC-LEARN has the potential to improve the reliability of LLMs in practical applications where consistent behavior is crucial. The gains in accuracy and robustness on challenging benchmarks suggest that the proposed method is effective in addressing this issue. The human evaluation showing preference for the model's reasoning paths over supervised finetuning also strengthens the argument for its superiority.

*   **Strengths:**

    *   The paper provides a well-defined and theoretically sound framework for enhancing reasoning consistency in LLMs.
    *   The use of executable programs allows for verifiable and interpretable reasoning processes.
    *   The composite reward function is carefully designed to balance multiple objectives, including accuracy, retrieval efficiency, and rejection of invalid queries.
    *   Experiments demonstrate clear and consistent improvements over strong baselines on a range of reasoning benchmarks.
    *   Ablation studies and human evaluations further support the effectiveness of the proposed method.
    * The separation of policy and retriever models is a design strength, forcing the policy model to focus on *how* to reason rather than relying on direct access to the facts.

*   **Weaknesses:**

    *   The limitations section discusses computational constraints that prevented exploration of different policy models for RL training. Scaling to larger models could reveal different performance characteristics.
    *   The reliance on a heuristic hyperparameter setup due to resource limitations could potentially limit the method's performance and generalizability. Thorough hyperparameter tuning might lead to further improvements.
    * The method relies on the creation of high-quality, similar questions. While the paper outlines a pipeline for this, the quality of these questions will impact the overall effectiveness. Error in this generation step can have impacts on the end outcome.
    * The results, while strong, would benefit from further analysis comparing the *types* of reasoning the LLMs are performing and comparing them to human reasoning. Are they actually reasoning like humans, or are they exploiting data in the retrievals in a different way?

*   **Potential Influence:**

    *   The paper is likely to influence future research on improving the reliability and consistency of LLM reasoning. The cohort-based training approach and structured, executable programs provide a promising direction for future exploration.
    *   The framework could be extended to other reasoning tasks and domains, and the composite reward function could be adapted to address other desired properties of LLM reasoning.
    *   The work could contribute to the development of more trustworthy and dependable AI systems for real-world applications.

**Score: 8**

**Rationale:**

The paper presents a novel and well-executed approach to addressing a significant problem in LLM reasoning. The strong experimental results, combined with ablation studies and human evaluations, provide compelling evidence for the effectiveness of CC-LEARN. While there are some limitations, such as the need for more extensive hyperparameter tuning and exploration of larger models, the paper makes a significant contribution to the field and is likely to have a lasting impact on future research in this area. The work fills a research gap by explicitly tackling reasoning consistency through RL and interpretable programming, not simply general fact memorization.

- **Score**: 8/10

### **[SwarmAgentic: Towards Fully Automated Agentic System Generation via Swarm Intelligence](http://arxiv.org/abs/2506.15672v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "SwarmAgentic: Towards Fully Automated Agentic System Generation via Swarm Intelligence":

**Summary:**

The paper introduces SwarmAgentic, a novel framework for fully automated generation of agentic systems. It addresses the limitations of existing frameworks, which often require manual intervention, predefined agent templates, or lack self-optimizing capabilities for agent functionality and collaboration. SwarmAgentic leverages a language-driven, symbolic design space search, inspired by Particle Swarm Optimization (PSO), to create agentic systems from scratch and jointly optimize agent functionality and collaboration. The framework represents agentic systems as particles evolved through language-based transformations and guided by LLM-identified flaws. Experiments across six real-world, open-ended tasks demonstrate SwarmAgentic's superior performance compared to existing baselines, highlighting the effectiveness of full automation in structurally unconstrained tasks.

**Critical Evaluation:**

**Novelty:** The paper presents a genuinely novel approach by combining swarm intelligence principles (PSO) with large language models to automate agentic system generation fully. The symbolic, language-based reformulation of PSO is a significant departure from traditional numerical optimization techniques and enables a more interpretable and flexible search process.  The key novelties are in:

*   **Fully Automated From-Scratch Agent Generation**:  Unlike previous methods relying on seed agents or templates, SwarmAgentic generates agents from the ground up.
*   **Joint Optimization of Functionality and Collaboration**:  Simultaneously optimizing agent-level capabilities and their interaction strategies sets this work apart.
*   **Failure-Aware Velocity Updates:** Using LLM feedback to address flaws in a systematic way to drive the search in a better direction.

**Significance:** The implications of this research are significant. By automating agentic system generation, SwarmAgentic lowers the barrier to entry for developing complex multi-agent systems and unlocks potential for scaling agentic applications to diverse and structurally unconstrained tasks. The ability to create adaptable and self-optimizing agentic systems has numerous potential applications, including:

*   Improved planning and scheduling tools
*   More effective collaboration platforms
*   Enhanced creative content generation systems
*   Better personalized learning experiences

**Strengths:**

*   **Comprehensive evaluation:** The paper rigorously evaluates SwarmAgentic across a diverse set of tasks, comparing it against strong baselines. The observed performance gains are substantial, especially on tasks lacking structured assumptions.
*   **Well-defined framework:** The paper provides a clear and detailed description of the SwarmAgentic framework, including its key components, algorithms, and implementation details.
*   **Ablation studies and analysis:** The ablation studies provide insights into the contribution of different framework components and guide future directions.
*   **Cross-model transferability analysis**: Demonstrates the generality of the systems created.
*   **Reproducibility**: The code availability is a big plus for reproducibility and for further research.

**Weaknesses:**

*   **Reliance on LLMs:** While the framework's language-driven approach is a strength, it also inherits the limitations of LLMs.  Factuality errors or biases in LLM-generated content could negatively impact the agentic systems.  The paper mentions this as a limitation but doesn't fully address how these issues could be mitigated.
*   **Computational Cost:** The paper doesn't discuss the computational resources required for training and deploying the SwarmAgentic framework.  PSO can be computationally expensive, and relying on LLMs further increases this cost. This could be a barrier to adoption.
*   **Limited theoretical grounding:** While the paper draws inspiration from PSO, a deeper theoretical analysis of the framework's convergence properties and scalability would strengthen the contribution.

**Potential Influence:** SwarmAgentic has the potential to be highly influential in the agentic systems and multi-agent system research community. It introduces a practical and effective approach to automating agentic system generation that overcomes many of the limitations of existing methods. The ideas presented in the paper will likely inspire further research in this area, focusing on:

*   Developing more robust and reliable LLM-based agent generation techniques
*   Exploring alternative optimization algorithms for agentic system design
*   Addressing the computational cost of automated agentic system generation
* Integrating world models or symbolic models to reduce the reliance on LLMs

**Justification for Score:**

I am assigning a score of 8 to this paper. The novelty and significance of SwarmAgentic are clear, addressing a major bottleneck in the development of intelligent systems. The thorough experimental evaluation and detailed framework description significantly strengthen the paper.  The weaknesses primarily revolve around the inherited limitations of LLMs and the need for a more in-depth theoretical analysis of the framework's properties. However, these limitations do not detract from the paper's overall contribution and potential to advance the field significantly.

**Score: 8**

- **Score**: 8/10

### **[UniRelight: Learning Joint Decomposition and Synthesis for Video Relighting](http://arxiv.org/abs/2506.15673v1)**
- **Summary**: Here is a summary and critical evaluation of the UniRelight paper:

**Summary:**

The paper "UniRelight: Learning Joint Decomposition and Synthesis for Video Relighting" tackles the challenge of relighting images and videos, a task requiring deep scene understanding and realistic light transport synthesis. The authors propose a novel approach that jointly estimates albedo and synthesizes relit outputs in a single pass, using video diffusion models. This joint formulation leverages the generative capabilities of the diffusion model and enhances implicit scene comprehension, allowing for realistic lighting effects and intricate material interactions. The model is trained on a combination of synthetic multi-illumination data and auto-labeled real-world videos, demonstrating strong generalization across diverse domains and surpassing previous methods in visual fidelity and temporal consistency.

**Critical Evaluation:**

*   **Novelty:** The primary novelty lies in the joint formulation of albedo decomposition and relighting within a video diffusion model framework.  Previous approaches typically decouple these stages, which can lead to error accumulation and less realistic results. The joint denoising of latent representations for relighting and albedo demodulation in a single pass is a novel design, inspired by but significantly extending VideoJAM.  The idea of using albedo demodulation as a strong prior for the relighting task is also a valuable insight.  Furthermore, the combination of fully supervised synthetic data with automatically labeled real-world data is a practical approach to address the scarcity of multi-illumination datasets. The specifics of how they apply and ablate conditioning strategies tailored to different data sources add incremental value.

*   **Significance:**  The significance of this work stems from its potential to advance the state-of-the-art in video relighting. The results demonstrate that the proposed method can generate more realistic and temporally consistent relighting effects than existing methods, particularly in complex scenes with sophisticated materials (e.g., transparent objects, specular highlights).  The ability to train on a combination of synthetic and readily available real-world data makes the approach more scalable and practical. The improved generalization to unseen domains is particularly valuable, as it addresses a major limitation of previous methods. The authors provide a comprehensive evaluation, including quantitative metrics, qualitative comparisons, and a user study, to support their claims.

*   **Strengths:**
    *   Novel joint formulation for albedo decomposition and relighting.
    *   Effective use of video diffusion models for generating realistic lighting effects.
    *   Strong generalization capabilities due to hybrid training strategy.
    *   Comprehensive evaluation with quantitative and qualitative results.
    *   Clear presentation and well-written paper.
    *   Careful and detailed ablation studies.

*   **Weaknesses:**
    *   The method still relies on synthetic data for supervision, which might limit its ability to capture the full complexity of real-world lighting scenarios. While they auto-label real-world data, it is still based on an *estimation* of albedo.
    *   The method cannot currently handle emitting objects, restricting its applicability in certain scenarios. The limitation to only manipulating environment lighting constrains the possible edits to the scene.
    *   The inference runtime is still relatively high, which might be a barrier for real-time applications, although the authors make a good point comparing relative runtimes. The need for powerful GPUs to run the DiT model is a barrier to widespread adoption.
    *   The user study on StreetScenes uses the approach itself to generate data and does not have an environment to compare with a ground truth, only differing ablated approaches. The user study should have incorporated the synthetic dataset to be fairer and accurate.

*   **Potential Influence:** The UniRelight framework has the potential to influence future research in video relighting, inverse rendering, and generative modeling. The joint formulation could inspire new approaches for combining different tasks within a single neural network. The use of video diffusion models and hybrid training strategies could also be adopted by other researchers working on similar problems. The increased accessibility of high-quality relighting tools could also have an impact on creative industries, allowing for more realistic and efficient content creation.

**Score: 8.0**

**Rationale:** The paper presents a significant advancement in video relighting due to its novel joint formulation and impressive results. While the approach has some limitations, such as its reliance on synthetic data and high computational cost, its strengths outweigh its weaknesses. The improved generalization, realistic relighting effects, and comprehensive evaluation demonstrate its potential impact on the field. The method significantly extends existing work and introduces valuable insights that could inspire future research. The paper isn't a paradigm shift (thus not a 9 or 10), but it makes a solid contribution with good engineering and a clear methodology that provides a better user experience.

- **Score**: 8/10

## Other Papers
### **[Cost-Aware Routing for Efficient Text-To-Image Generation](http://arxiv.org/abs/2506.14753v1)**
### **[Scaling-Up the Pretraining of the Earth Observation Foundation Model PhilEO to the MajorTOM Dataset](http://arxiv.org/abs/2506.14765v1)**
### **[A Variational Framework for Improving Naturalness in Generative Spoken Language Models](http://arxiv.org/abs/2506.14767v1)**
### **[CDP: Towards Robust Autoregressive Visuomotor Policy Learning via Causal Diffusion](http://arxiv.org/abs/2506.14769v1)**
### **[DETONATE: A Benchmark for Text-to-Image Alignment and Kernelized Direct Preference Optimization](http://arxiv.org/abs/2506.14903v1)**
### **[CrEst: Credibility Estimation for Contexts in LLMs via Weak Supervision](http://arxiv.org/abs/2506.14912v1)**
### **[Winter Soldier: Backdooring Language Models at Pre-Training with Indirect Data Poisoning](http://arxiv.org/abs/2506.14913v1)**
### **[Frequency-Calibrated Membership Inference Attacks on Medical Image Diffusion Models](http://arxiv.org/abs/2506.14919v1)**
### **[FORTRESS: Frontier Risk Evaluation for National Security and Public Safety](http://arxiv.org/abs/2506.14922v1)**
### **[Vision Transformers for End-to-End Quark-Gluon Jet Classification from Calorimeter Images](http://arxiv.org/abs/2506.14934v1)**
### **[Structured Moral Reasoning in Language Models: A Value-Grounded Evaluation Framework](http://arxiv.org/abs/2506.14948v1)**
### **[From Chat to Checkup: Can Large Language Models Assist in Diabetes Prediction?](http://arxiv.org/abs/2506.14949v1)**
### **[Thinking in Directivity: Speech Large Language Model for Multi-Talker Directional Speech Recognition](http://arxiv.org/abs/2506.14973v1)**
### **[Hypothesis Testing for Quantifying LLM-Human Misalignment in Multiple Choice Settings](http://arxiv.org/abs/2506.14997v1)**
### **[Memory Tokens: Large Language Models Can Generate Reversible Sentence Embeddings](http://arxiv.org/abs/2506.15001v1)**
### **[Scaling Intelligence: Designing Data Centers for Next-Gen Language Models](http://arxiv.org/abs/2506.15006v1)**
### **[Hyper-Local Deformable Transformers for Text Spotting on Historical Maps](http://arxiv.org/abs/2506.15010v1)**
### **[SFT-GO: Supervised Fine-Tuning with Group Optimization for Large Language Models](http://arxiv.org/abs/2506.15021v1)**
### **[Optimal Embedding Learning Rate in LLMs: The Effect of Vocabulary Size](http://arxiv.org/abs/2506.15025v1)**
### **[Identifying economic narratives in large text corpora -- An integrated approach using Large Language Models](http://arxiv.org/abs/2506.15041v1)**
### **[Mapping Caregiver Needs to AI Chatbot Design: Strengths and Gaps in Mental Health Support for Alzheimer's and Dementia Caregivers](http://arxiv.org/abs/2506.15047v1)**
### **[Truncated Proximal Policy Optimization](http://arxiv.org/abs/2506.15050v1)**
### **[HEAL: An Empirical Study on Hallucinations in Embodied Agents Driven by Large Language Models](http://arxiv.org/abs/2506.15065v1)**
### **[ChatModel: Automating Reference Model Design and Verification with LLMs](http://arxiv.org/abs/2506.15066v1)**
### **[Learning-Time Encoding Shapes Unlearning in LLMs](http://arxiv.org/abs/2506.15076v1)**
### **[Enhancement Report Approval Prediction: A Comparative Study of Large Language Models](http://arxiv.org/abs/2506.15098v1)**
### **[CipherMind: The Longest Codebook in the World](http://arxiv.org/abs/2506.15117v1)**
### **[CKD-EHR:Clinical Knowledge Distillation for Electronic Health Records](http://arxiv.org/abs/2506.15118v1)**
### **[Generative thermodynamic computing](http://arxiv.org/abs/2506.15121v1)**
### **[eLLM: Elastic Memory Management Framework for Efficient LLM Serving](http://arxiv.org/abs/2506.15155v1)**
### **[Robust Instant Policy: Leveraging Student's t-Regression Model for Robust In-context Imitation Learning of Robot Manipulation](http://arxiv.org/abs/2506.15157v1)**
### **[Echo-DND: A dual noise diffusion model for robust and precise left ventricle segmentation in echocardiography](http://arxiv.org/abs/2506.15166v1)**
### **[From LLMs to MLLMs to Agents: A Survey of Emerging Paradigms in Jailbreak Attacks and Defenses within LLM Ecosystem](http://arxiv.org/abs/2506.15170v1)**
### **[Accessible Gesture-Driven Augmented Reality Interaction System](http://arxiv.org/abs/2506.15189v1)**
### **[HeurAgenix: Leveraging LLMs for Solving Complex Combinatorial Optimization Challenges](http://arxiv.org/abs/2506.15196v1)**
### **[A Comparative Study of Task Adaptation Techniques of Large Language Models for Identifying Sustainable Development Goals](http://arxiv.org/abs/2506.15208v1)**
### **[ProtoReasoning: Prototypes as the Foundation for Generalizable Reasoning in LLMs](http://arxiv.org/abs/2506.15211v1)**
### **[LLM vs. SAST: A Technical Analysis on Detecting Coding Bugs of GPT4-Advanced Data Analysis](http://arxiv.org/abs/2506.15212v1)**
### **[MinosEval: Distinguishing Factoid and Non-Factoid for Tailored Open-Ended QA Evaluation with LLMs](http://arxiv.org/abs/2506.15215v1)**
### **[DM-FNet: Unified multimodal medical image fusion via diffusion process-trained encoder-decoder](http://arxiv.org/abs/2506.15218v1)**
### **[video-SALMONN 2: Captioning-Enhanced Audio-Visual Large Language Models](http://arxiv.org/abs/2506.15220v1)**
### **[Large Language Models for Unit Testing: A Systematic Literature Review](http://arxiv.org/abs/2506.15227v1)**
### **[Lost in Variation? Evaluating NLI Performance in Basque and Spanish Geographical Variants](http://arxiv.org/abs/2506.15239v1)**
### **[Research on Graph-Retrieval Augmented Generation Based on Historical Text Knowledge Graphs](http://arxiv.org/abs/2506.15241v1)**
### **[Unlocking Post-hoc Dataset Inference with Synthetic Data](http://arxiv.org/abs/2506.15271v1)**
### **[Human Motion Capture from Loose and Sparse Inertial Sensors with Garment-aware Diffusion Models](http://arxiv.org/abs/2506.15290v1)**
### **[MEGC2025: Micro-Expression Grand Challenge on Spot Then Recognize and Visual Question Answering](http://arxiv.org/abs/2506.15298v1)**
### **[SecFwT: Efficient Privacy-Preserving Fine-Tuning of Large Language Models Using Forward-Only Passes](http://arxiv.org/abs/2506.15307v1)**
### **[One-shot Face Sketch Synthesis in the Wild via Generative Diffusion Prior and Instruction Tuning](http://arxiv.org/abs/2506.15312v1)**
### **[When and How Unlabeled Data Provably Improve In-Context Learning](http://arxiv.org/abs/2506.15329v1)**
### **[DeVisE: Behavioral Testing of Medical Large Language Models](http://arxiv.org/abs/2506.15339v1)**
### **[Acoustic Waveform Inversion with Image-to-Image Schrödinger Bridges](http://arxiv.org/abs/2506.15346v1)**
### **[SANSKRITI: A Comprehensive Benchmark for Evaluating Language Models' Knowledge of Indian Culture](http://arxiv.org/abs/2506.15355v1)**
### **[Sampling 3D Molecular Conformers with Diffusion Transformers](http://arxiv.org/abs/2506.15378v1)**
### **[When Model Knowledge meets Diffusion Model: Diffusion-assisted Data-free Image Synthesis with Alignment of Domain and Class](http://arxiv.org/abs/2506.15381v1)**
### **[Provable Maximum Entropy Manifold Exploration via Diffusion Models](http://arxiv.org/abs/2506.15385v1)**
### **[Targeted Lexical Injection: Unlocking Latent Cross-Lingual Alignment in Lugha-Llama via Early-Layer LoRA Fine-Tuning](http://arxiv.org/abs/2506.15415v1)**
### **[Understanding GUI Agent Localization Biases through Logit Sharpness](http://arxiv.org/abs/2506.15425v1)**
### **[Uncovering Intention through LLM-Driven Code Snippet Description Generation](http://arxiv.org/abs/2506.15453v1)**
### **[RE-IMAGINE: Symbolic Benchmark Synthesis for Reasoning Evaluation](http://arxiv.org/abs/2506.15455v1)**
### **[Multimodal Large Language Models for Medical Report Generation via Customized Prompt Tuning](http://arxiv.org/abs/2506.15477v1)**
### **[Creating User-steerable Projections with Interactive Semantic Mapping](http://arxiv.org/abs/2506.15479v1)**
### **[Context-Informed Grounding Supervision](http://arxiv.org/abs/2506.15480v1)**
### **[GenHOI: Generalizing Text-driven 4D Human-Object Interaction Synthesis for Unseen Objects](http://arxiv.org/abs/2506.15483v1)**
### **[SPARE: Single-Pass Annotation with Reference-Guided Evaluation for Automatic Process Supervision and Reward Modelling](http://arxiv.org/abs/2506.15498v1)**
### **[Optimizing Web-Based AI Query Retrieval with GPT Integration in LangChain A CoT-Enhanced Prompt Engineering Approach](http://arxiv.org/abs/2506.15512v1)**
### **[Lessons from Training Grounded LLMs with Verifiable Rewards](http://arxiv.org/abs/2506.15522v1)**
### **[Diff-TONE: Timestep Optimization for iNstrument Editing in Text-to-Music Diffusion Models](http://arxiv.org/abs/2506.15530v1)**
### **[Intrinsic and Extrinsic Organized Attention: Softmax Invariance and Network Sparsity](http://arxiv.org/abs/2506.15541v1)**
### **[RATTENTION: Towards the Minimal Sliding Window Size in Local-Global Attention Models](http://arxiv.org/abs/2506.15545v1)**
### **[PredGen: Accelerated Inference of Large Language Models through Input-Time Speculation for Real-Time Speech Interaction](http://arxiv.org/abs/2506.15556v1)**
### **[Control and Realism: Best of Both Worlds in Layout-to-Image without Training](http://arxiv.org/abs/2506.15563v1)**
### **[Gender Inclusivity Fairness Index (GIFI): A Multilevel Framework for Evaluating Gender Diversity in Large Language Models](http://arxiv.org/abs/2506.15568v1)**
### **[Memory-Efficient Differentially Private Training with Gradient Random Projection](http://arxiv.org/abs/2506.15588v1)**
### **[One-Step Diffusion for Detail-Rich and Temporally Consistent Video Super-Resolution](http://arxiv.org/abs/2506.15591v1)**
### **[LiteGD: Lightweight and dynamic GPU Dispatching for Large-scale Heterogeneous Clusters](http://arxiv.org/abs/2506.15595v1)**
### **[LoX: Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning](http://arxiv.org/abs/2506.15606v1)**
### **[The Compositional Architecture of Regret in Large Language Models](http://arxiv.org/abs/2506.15617v1)**
### **[The Effect of State Representation on LLM Agent Behavior in Dynamic Routing Games](http://arxiv.org/abs/2506.15624v1)**
### **[HOIDiNi: Human-Object Interaction through Diffusion Noise Optimization](http://arxiv.org/abs/2506.15625v1)**
### **[Revisiting Compositional Generalization Capability of Large Language Models Considering Instruction Following Ability](http://arxiv.org/abs/2506.15629v1)**
### **[Demystifying the Visual Quality Paradox in Multimodal Large Language Models](http://arxiv.org/abs/2506.15645v1)**
### **[AutoRule: Reasoning Chain-of-thought Extracted Rule-based Rewards Improve Preference Learning](http://arxiv.org/abs/2506.15651v1)**
### **[PhishDebate: An LLM-Based Multi-Agent Framework for Phishing Website Detection](http://arxiv.org/abs/2506.15656v1)**
### **[CC-LEARN: Cohort-based Consistency Learning](http://arxiv.org/abs/2506.15662v1)**
### **[SwarmAgentic: Towards Fully Automated Agentic System Generation via Swarm Intelligence](http://arxiv.org/abs/2506.15672v1)**
### **[UniRelight: Learning Joint Decomposition and Synthesis for Video Relighting](http://arxiv.org/abs/2506.15673v1)**
### **[GenRecal: Generation after Recalibration from Large to Small Vision-Language Models](http://arxiv.org/abs/2506.15681v1)**
### **[Evolutionary Caching to Accelerate Your Off-the-Shelf Diffusion Model](http://arxiv.org/abs/2506.15682v1)**
