# The Latest Daily Papers - Date: 2025-03-01
## Highlight Papers
### **[Distill Not Only Data but Also Rewards: Can Smaller Language Models Surpass Larger Ones?](http://arxiv.org/abs/2502.19557v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper proposes a novel knowledge distillation pipeline for large language models (LLMs) that goes beyond traditional supervised fine-tuning (SFT). Instead of solely distilling the teacher model's output data, the method also distills *rewards* reflecting the quality of those outputs. A key innovation is the introduction of a self-supervised mechanism to generate "pseudo-rewards" by analyzing the inherent structure and relationships between both the teacher's and the student's responses, rather than relying on potentially biased and inconsistent explicit evaluations from the teacher LLM. This reward model is then used in reinforcement learning (RL) to iteratively refine the student model, leading to performance that can surpass the teacher's. The authors demonstrate the effectiveness of this approach on the GSM8K and MMLU-PRO benchmarks.

**Critical Evaluation:**

**Novelty:** The paper presents a significant departure from standard knowledge distillation techniques. The idea of distilling *rewards* in addition to data, particularly when those rewards are derived in a self-supervised manner, is a clever and innovative approach. While the individual components (SFT, RL, reward models) are not novel, their combination within the proposed pipeline and the specific self-supervised reward generation are unique.

**Significance:** The potential impact of this work is substantial. The ability to train smaller, more efficient models that can outperform their larger teachers is crucial for the widespread deployment of LLMs. Reducing reliance on expensive external reward signals (e.g., human feedback) further enhances scalability. The paper's results on GSM8K and MMLU-PRO, showcasing student models exceeding teacher performance, are compelling. The performance gain by student models on MMLU-PRO, especially when distilling from Llama3-8B and Llama3-1B, shows significant learning taking place by the student.

**Strengths:**

*   **Novel approach:** The self-supervised reward distillation pipeline is a creative and effective way to address the limitations of SFT and the challenges of obtaining reliable reward signals from LLMs.
*   **Strong experimental results:** The paper provides convincing empirical evidence supporting the effectiveness of the proposed method on challenging benchmarks. The results are not marginal improvements, but showcase genuine outperformance of the teacher models in certain configurations.
*   **Clear presentation:** The paper is well-written and clearly explains the proposed method and experimental setup.
*   **Detailed ablation studies:** The ablation studies provide valuable insights into the contributions of different components of the pipeline.

**Weaknesses:**

*   **Task Scope:** The study focuses on a relatively small set of mathematical reasoning and knowledge understanding datasets. While these are important benchmarks, further validation across a wider range of tasks (e.g., text generation, translation, dialogue) would strengthen the generalizability of the findings.
*   **Teacher dependency:** The method is, ultimately, still reliant on the *quality* of the teacher model's data, even if not its direct evaluations. The self-supervision mitigates biases, but a poor teacher will inevitably limit the student's potential.
*   **Complexity:** The pipeline is somewhat complex, involving multiple stages (SFT, reward model training, RL). This could increase the barrier to adoption compared to simpler distillation methods.
*   **Computational Resources:** The experiments require significant computational resources (H100 GPUs). While this is typical for LLM research, it's worth acknowledging.

**Justification for Score:**

I assign a score of **8**. The paper makes a significant and novel contribution to knowledge distillation. The method addresses a key limitation of SFT by incorporating reward signals and innovatively generates these rewards through a self-supervised mechanism. The experimental results are compelling, demonstrating the potential for student models to surpass their teachers.

Despite these strengths, the score is not higher due to the limited task scope, the inherent reliance on the teacher's base knowledge, and the complexity of the proposed pipeline. While the limitations are valid, the innovation and demonstrated potential are significant enough to warrant a high score. It's likely that future work building upon this framework will further solidify its impact.
Score: 8

- **Score**: 8/10

### **[SuPreME: A Supervised Pre-training Framework for Multimodal ECG Representation Learning](http://arxiv.org/abs/2502.19668v1)**
- **Summary**: Here's a summary and critical evaluation of the provided paper:

**Summary:**

The paper introduces SuPreME, a supervised pre-training framework for multimodal ECG representation learning. SuPreME leverages Large Language Models (LLMs) to extract structured clinical entities from free-text ECG reports, creating a fine-grained labeled dataset.  The framework avoids complex pretext tasks by directly aligning ECG signals with these extracted entities using a Cardiac Fusion Network (CFN). By using text-based cardiac queries, the model enables zero-shot classification of unseen diseases. Evaluations on six downstream datasets show that SuPreME achieves superior zero-shot performance compared to state-of-the-art self-supervised and multimodal methods. The paper highlights the effectiveness of SuPreME in utilizing structured clinical knowledge for high-quality ECG representations, showcasing its simplicity, data efficiency, and potential for clinical applicability.

**Critical Evaluation:**

*   **Strengths:**

    *   **Novel Approach:** The core idea of using LLMs to extract structured clinical entities and leveraging these as supervision for ECG representation learning is innovative.  It addresses the limitations of existing self-supervised learning (SSL) methods that often lack clinical semantics. The zero-shot capabilities are also a significant advantage.
    *   **Technical Soundness:** The proposed architecture, including the Cardiac Fusion Network (CFN), seems technically sound and well-motivated. The ablation studies provide insights into the contribution of individual components.
    *   **Comprehensive Evaluation:** The paper includes a thorough evaluation on six diverse downstream datasets, covering a large number of cardiac conditions (127),  demonstrating the generalizability of the approach. The comparison to existing SSL and multimodal methods is convincing. Data efficiency is clearly demonstrated as SuPreME can even outperform other eSSL algorithms with fewer pre-training data.
    *   **Clinical Relevance:**  The use of clinical entities ensures the learned representations are more clinically relevant than those produced by purely signal-based methods. The potential for zero-shot classification is a significant advantage in clinical settings where new or rare diseases may be encountered.
*   **Weaknesses:**

    *   **LLM Dependency:** The reliance on LLMs for entity extraction introduces a potential dependency on the performance and biases of the specific LLM used.  The robustness of the framework to different LLMs or variations in report quality needs further examination.
    *   **Complexity of Entity Extraction Pipeline:** While presented as simple and scalable, the LLM based extraction pipeline is somewhat complex and contains many steps, involving GPT-4, domain specific prompt engineering and entity deduplication.
    *   **Simplification of CKEPE:** While the simplified CKEPE may reduce redundancy, it could potentially compromise the richness of the queries and limit the model's ability to learn nuanced relationships between ECG signals and clinical conditions. How the queries are designed from the SCP codes could have been expanded for better understanding.

*   **Novelty and Significance:**

    *   The primary novelty lies in the *methodology* of combining LLMs and LLM extracted entities in a supervised training paradigm for learning ECG representations. The emphasis on structured, clinically meaningful information is a departure from signal-level or free-text approaches and adds a dimension of clinical relevance missing from existing methods. The zero shot classification of unseen diseases is also a novel contribution.
    *   The significance stems from the potential to overcome the limitations of existing ECG analysis methods and enables scalable, clinically informed, and adaptable ECG representation learning. The data efficiency further increases the appeal for real world use cases.

**Justification for Score:**

The paper demonstrates a clear advancement over existing techniques in ECG representation learning.  The approach is technically sound, empirically validated, and clinically relevant.  The zero-shot capability is especially promising.  However, the dependency on LLMs and potential limitations of the query design keep it from being truly transformative. While not revolutionary, it represents a significant and well-executed step forward.

**Score: 8**
- **Score**: 8/10

### **[UIFace: Unleashing Inherent Model Capabilities to Enhance Intra-Class Diversity in Synthetic Face Recognition](http://arxiv.org/abs/2502.19803v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "UIFace: UNLEASHING INHERENT MODEL CAPABILITY TO ENHANCE INTRA-CLASS DIVERSITY IN SYNTHETIC FACE RECOGNITION":

**Summary:**

The paper addresses the challenge of limited intra-class diversity in synthetically generated face datasets for training face recognition (FR) models. Existing synthetic data generation methods often suffer from "context overfitting," where the generated images lack variations in pose, expression, and illumination. UIFace proposes a novel framework that leverages the inherent capability of diffusion models to generate diverse images. The framework employs a two-stage sampling strategy. The first stage uses an "empty context" (i.e., no specific identity constraint) to generate images with high diversity but random identities.  The second stage then uses a specific identity context to refine the images, ensuring identity preservation. An "attention injection module" further enhances diversity by incorporating attention maps from the empty context generation into the identity-conditioned generation. Experimental results demonstrate that UIFace significantly outperforms existing synthetic data generation methods and achieves comparable performance to FR models trained on real datasets, even with less training data and fewer synthetic identities.

**Critical Evaluation:**

*   **Novelty:** The core novelty lies in the two-stage sampling strategy that smartly combines the benefits of identity-preserving and diversity-enhancing generation. The use of an "empty context" to unlock inherent diversity and the subsequent identity-preserving refinement is a clever idea. The attention injection module also introduces a novel mechanism for incorporating diversity information. While diffusion models for face generation aren't entirely new, the specific combination of these techniques to address the context overfitting problem in synthetic face data is novel.

*   **Significance:** Synthetic data generation is increasingly crucial for face recognition due to privacy concerns and data scarcity. A method that can produce high-quality synthetic data with high intra-class diversity would have a significant impact on the field. By addressing the limitations of existing synthetic face datasets, UIFace can improve the performance and robustness of FR models while mitigating privacy risks. Demonstrating comparable performance to real-data trained models is a very important and promising step. Moreover, the framework's potential to reduce the reliance on large, real-world face datasets is a crucial step towards more ethical and privacy-aware FR systems.

*   **Strengths:**

    *   **Strong Results:** The experimental results convincingly demonstrate the superiority of UIFace over existing methods. The authors provide comprehensive evaluations on standard face recognition benchmarks, showing significant accuracy improvements and better diversity metrics. The experiments demonstrating comparable performance to models trained on real datasets are particularly strong.
    *   **Clear Problem Definition:** The paper clearly articulates the context overfitting issue in synthetic face data generation.
    *   **Well-Designed Framework:** The two-stage sampling strategy and attention injection module are well-motivated and effectively address the identified problem. The authors provide ablation studies to demonstrate the contribution of each component.
    *   **Good Technical Details:** The paper provides sufficient technical details to understand the proposed method and reproduce the results.
    *   **Address a real need of the domain**: The authors well highlight a domain in need of a solution.

*   **Weaknesses:**

    *   **Dataset Dependency:** The diffusion model is still trained on a real dataset (CAISA-Webface). While UIFace reduces the dependency on real data for training the *FR model*, the synthetic data generation itself relies on a real dataset. Ideally, the field would benefit from a method that can generate synthetic data from minimal or entirely synthetic initial conditions. This dependency on real data needs to be acknowledged explicitly in the conclusions.
    *   **Computational cost:** the two-stage diffusion sampling strategy is computationally demanding, which may limit its applicability in real-time or resource-constrained environments. The paper does not provide details regarding the training and inference time for each method, making it difficult to estimate the trade-off between accuracy and computational cost.
    *   **Hyperparameter Sensitivity:** The adaptive partitioning strategy depends on a hyperparameter, 'th,' which could influence the results. The paper doesn't explore the sensitivity of the model to this hyperparameter.

*   **Potential Influence:**

    *   UIFace could become a foundational method for generating high-quality synthetic face data, influencing future research in FR.
    *   The two-stage sampling strategy could be applied to other generative tasks where diversity and identity preservation are important.
    *   The method's success could spur further research into leveraging inherent model capabilities to enhance generative models.

**Score:** 8.5

**Justification:** UIFace presents a significant advancement in synthetic face data generation, addressing a critical problem in the FR field. The proposed two-stage sampling strategy and attention injection module are novel and effective, leading to substantial improvements in both accuracy and diversity. While the method still relies on real data for training the generative model and there are questions regarding the computational cost, the results are compelling. The framework has the potential to significantly impact the field by enabling the development of more robust and privacy-aware FR systems. The score reflects the novelty, strong experimental validation, and potential influence of the paper. The weaknesses noted detract somewhat from a higher score, especially regarding the dataset dependency.

- **Score**: 8/10

### **[Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts](http://arxiv.org/abs/2502.19811v1)**
- **Summary**: Okay, let's break down this paper on COMET and provide a critical assessment.

**Summary**

The paper introduces COMET, a system designed to optimize the execution of Mixture-of-Experts (MoE) models by addressing the significant communication overhead inherent in distributed MoE architectures. COMET achieves this through fine-grained communication-computation overlapping.  It uses two main strategies: 1) a shared tensor-based dependency resolving method, which analyzes data dependencies and reorganizes tensor data to enable better pipelining, and 2) an adaptive workload assignment method that dynamically allocates GPU thread blocks to communication and computation workloads within fused kernels, balancing latencies. The authors integrate COMET into Megatron-LM and demonstrate performance improvements, including up to 1.96x speedup for MoE layers and 1.71x speedup for end-to-end MoE models, validated on clusters with tens of thousands of GPUs.  The system achieves these improvements without significantly increasing memory overhead.

**Critical Evaluation**

*   **Strengths:**

    *   **Addresses a real and significant problem:** The communication bottleneck in distributed MoE training is well-known and a major hurdle to scaling these models effectively. The paper clearly articulates this problem and provides empirical evidence of its severity.
    *   **Technical Novelty:** COMET's approach to fine-grained overlapping of communication and computation, using shared tensor dependency analysis and adaptive workload assignment, appears to be a novel contribution. While other overlapping methods exist, the granularity and adaptability COMET offers are significant improvements.  The system integrates communication and computation within fused GPU kernels, offering finer control over hardware resource allocation.
    *   **Empirical Validation:** The paper presents a thorough evaluation of COMET's performance, demonstrating significant speedups compared to established baselines like Megatron-LM with CUTLASS, Transformer Engine, FasterMoE, and Tutel. The evaluation includes a variety of MoE models, different input token lengths, and various parallelization strategies. The experiments are conducted on realistic hardware configurations (H100 and L20 clusters).  The use of ten-thousand-scale GPU clusters in production and the reported millions of GPU hours saved add significant weight to the real-world impact of the work.
    *   **Clear and Well-Written:** The paper is generally well-written and explains the technical details of COMET clearly. The figures and tables are helpful in understanding the system's design and performance.
    *   **Open Source Promise**: Open sourcing the code increases impact of this work.

*   **Weaknesses:**

    *   **Complexity:** The system appears to be quite complex, involving a sophisticated interplay of shared tensor decomposition, computation rescheduling, thread block specialization, and adaptive workload assignment. While the paper explains these concepts, the implementation details and the interactions between these components might be difficult to grasp fully without access to the source code.  Scalability concerns of the scheduling component, especially with increasing experts and data dependency needs should be clearly shown.
    *   **Generality:** While the paper showcases COMET's performance with specific MoE models (Mixtral, Qwen2-MoE, Phi3.5-MoE) and hardware, it's not immediately clear how well the system would generalize to other MoE architectures or different GPU architectures.  The adaptive nature is a strength, but demonstrating its adaptability to a *wider* range of scenarios would further enhance the paper's impact.  Ablation studies to quantify the specific impact of each component (shared tensor analysis, adaptive workload assignment) would also strengthen the results.
    *   **NVSHMEM dependency:** Reliance on NVSHMEM poses a concern for adoption. NVSHMEM although efficient, isn't as portable or as widely adopted as other libraries. This might limit wider adoption of the proposed technique.

*   **Novelty and Significance:**

    *   The paper offers a novel system-level approach to address a core bottleneck in MoE training and inference.
    *   The fine-grained overlapping technique and adaptive workload assignment are significant contributions beyond existing coarse-grained overlapping methods.
    *   The experimental results demonstrate a substantial performance improvement compared to the state-of-the-art, showing promise for real-world deployment.
    *   The scale of the production deployment (ten-thousand-scale GPU clusters) and the millions of GPU hours saved indicate significant practical impact.

**Justification for Score:**

Considering the strengths and weaknesses, COMET represents a significant advancement in the area of MoE training. The combination of fine-grained overlapping, adaptive workload assignment, and substantial performance improvements warrants a high score. However, the reliance on NVSHMEM for communication, the system's complexity, and the limited evidence of generality hold it back from a perfect score. The lack of ablation studies weakens the impact of evaluation.

Score: 8

- **Score**: 8/10

### **[Foot-In-The-Door: A Multi-turn Jailbreak for LLMs](http://arxiv.org/abs/2502.19820v1)**
- **Summary**: Okay, here's a concise summary and critical evaluation of the paper "Foot-In-The-Door: A Multi-turn Jailbreak for LLMs":

**Summary:**

The paper introduces FITD, a novel multi-turn jailbreak attack for Large Language Models (LLMs).  Inspired by the psychological "foot-in-the-door" effect, FITD progressively escalates the malicious intent of user queries through intermediate "bridge" prompts and alignment mechanisms. This approach coaxes the LLM into generating increasingly harmful and disallowed outputs by gradually eroding its safety mechanisms. The paper demonstrates that FITD achieves a high attack success rate (ASR) across various LLMs, outperforming existing single-turn and some multi-turn jailbreak methods. The authors also analyze the "self-corruption" phenomenon in LLMs, highlighting vulnerabilities in current alignment strategies.

**Critical Evaluation:**

*   **Strengths:**

    *   **Novelty:** The use of the "foot-in-the-door" psychological principle in the context of LLM jailbreaking is a genuinely novel idea. Most previous jailbreak techniques rely on clever prompt engineering in a single turn or crafting complex agent designs. This paper offers a distinct perspective on manipulating LLM behavior through gradual escalation, demonstrating a novel attack vector.
    *   **Effectiveness:** The reported results show impressive performance. Achieving a 94% average ASR across seven different models is a strong indicator of the attack's potency and broad applicability.  This highlights a significant weakness in current LLM safety protocols.
    *   **Thorough Evaluation:** The paper includes extensive experiments and ablation studies.  The analysis of cross-model transferability and the impact of malicious level demonstrates a rigorous approach to understanding the attack's dynamics and limitations. Examining the Harmfulness of responses for each query level further bolsters the claim of a gradual, self-corrupting process.
    *   **Clarity and Structure:** The paper is well-written and structured, making it easy to understand the proposed method and its implications.  The figures effectively illustrate the FITD process and experimental results.
    *  **Responsible Disclosure:** By sharing the findings with OpenAI and Meta, the authors have taken proactive steps in addressing potential security concerns.

*   **Weaknesses:**

    *   **Limited Dataset Diversity:** While JailbreakBench and HarmBench are established benchmarks, they may not fully encompass the breadth of potential harmful queries and scenarios. Testing on a wider range of datasets would further strengthen the generalizability of the findings.
    *   **Evaluation Metric Dependence:** While ASR is a standard metric, it relies on another LLM (GPT-40 in this case) for evaluation. This introduces a potential bias, and the reliability of the evaluation depends on the alignment and robustness of the evaluation model itself.  The authors acknowledge this in the paper (Appendix A.2.3).
    *   **Incremental Advantage over Simple Prompt Chaining:** While the foot-in-the-door effect provides a useful framework, it would have been helpful to see an explicit comparison against a more straightforward approach of simply chaining malicious prompts together without the specific Re-Align and SlipperySlopeParaphrase components to quantify the distinct benefits of their components.

*   **Significance:**

    *   **Security Implications:** The paper has important security implications. It exposes a fundamental weakness in how LLMs are currently aligned and highlights the risks associated with multi-turn interactions.  The FITD attack demonstrates that even models with robust initial safety measures can be gradually manipulated into generating harmful content.
    *   **Future Research Directions:** The study's findings motivate further research into more robust alignment techniques, real-time adaptive monitoring, and better understanding of the "self-corruption" phenomenon in LLMs. The concept of gradually corrupting an LLM provides a valuable area for future research on both adversarial and defensive strategies.

*   **Impact Score Justification:**

    The paper has a high impact due to its novel approach, strong empirical results, and clear security implications. The FITD attack is significantly more effective than other methods. The meticulous experimentation strengthens the paper's arguments.

Score: 8

- **Score**: 8/10

### **[ConvCodeWorld: Benchmarking Conversational Code Generation in Reproducible Feedback Environments](http://arxiv.org/abs/2502.19852v1)**
- **Summary**: Here's a concise summary and critical evaluation of the paper:

**Summary:**

The paper introduces CONVCODEWORLD and CONVCODEBENCH, a novel environment and benchmark for evaluating conversational code generation capabilities of Large Language Models (LLMs). CONVCODEWORLD simulates realistic interactive code generation scenarios by combining compilation feedback, execution feedback with varying test coverage, and verbal feedback (novice/expert) generated by GPT-4o. CONVCODEBENCH is a static, cost-effective benchmark derived from CONVCODEWORLD, using pre-generated feedback logs to enable efficient LLM evaluation. The authors evaluate numerous LLMs and show that feedback combinations significantly impact model performance, weaker models with sufficient feedback can outperform single-turn SOTA, models struggle to generalize to unseen feedback combinations, and there is a trade-off between MRR and Recall.

**Critical Evaluation:**

* **Novelty:** The novelty lies primarily in the comprehensive simulation of the *interactive* code generation process. While existing benchmarks like InterCode and MINT explored aspects of this, they lacked the *diverse* feedback combinations and reproducibility aspects. The simulated verbal feedback (especially novice vs. expert) is a novel addition. The creation of a static benchmark, CONVCODEBENCH, for cost-effective evaluation is also a significant contribution, especially when dealing with expensive LLM API calls.
* **Significance:** The paper addresses a critical gap in the evaluation of code generation models. Existing benchmarks often focus on single-turn scenarios, neglecting the iterative and collaborative nature of real-world coding.  CONVCODEWORLD, along with CONVCODEBENCH, allows for a more nuanced and comprehensive evaluation of LLMs' ability to utilize different feedback types. The findings, like the performance variability across feedback settings and the ability of weaker models to surpass single-turn SOTA through interaction, provide valuable insights. The insights about the generalization challenges for LLMs when encountering unseen feedback scenarios is valuable for future research on improving few-shot adaption and robustness.

**Strengths:**

*   **Comprehensive and Realistic Simulation:** CONVCODEWORLD effectively models the interactive code generation workflow by simulating diverse feedback combinations.
*   **Cost-Effective Evaluation:** CONVCODEBENCH provides a static benchmark that correlates strongly with the interactive environment, enabling efficient and scalable LLM evaluation.
*   **Detailed Analysis:** The authors perform extensive experiments with numerous LLMs, providing valuable insights into the impact of feedback combinations on model performance, generalization ability, and the trade-off between MRR and Recall.
*   **Reproducibility:** The paper emphasizes the importance of reproducibility by using LLMs (GPT-4o) to generate verbal feedback and providing publicly available implementations and benchmarks.

**Weaknesses:**

*   **Reliance on GPT-4o for Verbal Feedback Simulation:** While GPT-4o provides a more reproducible and cost-effective solution, it is still a simulation and may not fully capture the nuances and complexities of human feedback. The reliance upon it, limits it being fully reproducible with a different tool being employed.
*   **Limited scope on problem domains:** the problems in BigCodeBench-Full-Instruct focuses on single-function python programs, not necessarily encompassing other code languages or complexity seen in modern software.

**Potential Influence on the Field:**

The paper is likely to have a significant impact on the field of code generation. By providing a more realistic and comprehensive benchmark, it will drive research towards developing LLMs that can effectively utilize diverse feedback types and collaborate with developers in interactive coding scenarios. CONVCODEWORLD and CONVCODEBENCH will become valuable resources for researchers and practitioners working in code generation and related areas.

**Justification for Score:**

Considering both strengths and weaknesses, the paper makes a valuable contribution to the field. The development of novel and well-reasoned benchmarks that closely reflect realistic development environments, and the insights that are gleaned from model performances against it are all beneficial for downstream research. Whilst the limitations of the generated verbal feedback, and scope on problem complexity bring it back slightly.

**Score: 8**

- **Score**: 8/10

### **[MMKE-Bench: A Multimodal Editing Benchmark for Diverse Visual Knowledge](http://arxiv.org/abs/2502.19870v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces MMKE-Bench, a new benchmark for evaluating multimodal knowledge editing in large language models (LMMs).  Unlike existing benchmarks that focus on simple triplet-based entity knowledge, MMKE-Bench aims to capture the complexity of real-world multimodal information through free-form natural language descriptions and diverse visual knowledge types.  The benchmark includes three types of editing tasks: visual entity editing, visual semantic editing, and user-specific editing. The paper details the benchmark's construction process, which involves collecting original knowledge, generating editing knowledge, and creating evaluation questions based on principles of reliability, locality, generalization, and portability. The authors evaluate five knowledge editing methods on three prominent LMMs, revealing that existing methods struggle with visual and user-specific edits and that no single method excels across all criteria.

**Critical Evaluation:**

The paper makes a valuable contribution by addressing a crucial gap in the evaluation of multimodal knowledge editing.  Existing benchmarks are often too simplistic and don't adequately capture the nuanced nature of real-world visual knowledge.

**Strengths:**

*   **Novelty:** The shift from triplet-based knowledge to free-form natural language is a significant step forward. It allows for a more expressive representation of multimodal information and better simulates real-world scenarios. The inclusion of visual semantic and user-specific editing tasks is also novel and addresses the limitations of existing benchmarks.
*   **Comprehensive Evaluation:** The paper's evaluation of five different editing methods on three LMMs provides a solid baseline and identifies areas where current techniques fall short. The analysis of reliability, locality, generalization, and portability gives a holistic view of each method's performance.
*   **Rigorous Construction:** The detailed description of the benchmark creation process, including human verification of questions and answers, increases confidence in the quality and reliability of the dataset.
*   **Significance:** The paper highlights the limitations of current multimodal knowledge editing methods and provides a more challenging benchmark to drive future research. As LMMs become more prevalent, the ability to edit their knowledge effectively is becoming increasingly important.

**Weaknesses:**

*   **Complexity of Evaluation Metrics:** While the four evaluation principles (reliability, locality, generalizability, and portability) are sound, the specific implementation and the degree to which these can be reliably automated could be more deeply examined. Some nuance might be lost in simple accuracy measures.
*   **Limited Scope of Editing Methods:** Evaluating only five knowledge editing methods is somewhat limiting, even if they are considered to be state-of-the-art. Exploring a wider range of methods or potentially developing new methods specifically tailored for the multimodal setting could further strengthen the findings.
*   **User-Specific Editing Task**: While its inclusion is well justified, the effectiveness of its evaluation could be increased by considering methods to represent individual user knowledge better and developing better methods for the evaluation questions.

**Overall:**

The paper's strengths outweigh its weaknesses. The MMKE-Bench benchmark represents a significant advancement in the field of multimodal knowledge editing evaluation. It addresses a critical need for more realistic and challenging benchmarks and provides valuable insights into the limitations of current methods. It will be beneficial as the research area progresses.

Score: 8

- **Score**: 8/10

### **[Re-evaluating Open-ended Evaluation of Large Language Models](http://arxiv.org/abs/2502.20170v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper critiques Elo-based rating systems in open-ended Large Language Model (LLM) evaluation, showing how they are susceptible to biases due to redundancy in prompts. It proposes a game-theoretic approach where evaluation is modeled as a 3-player game (prompt, king model, rebel model). It introduces novel equilibrium solution concepts to address redundancy and ensure robustness. The authors demonstrate, through simulations and real-world LLM data analysis, that their method leads to more intuitive ratings, provides insights into the competitive landscape, and maintains skill entropy unlike Elo-based systems that tend to reward specialization.

**Critical Evaluation:**

*   **Strengths:**

    *   **Problem Identification:** The paper clearly identifies a significant problem with the existing Elo-based systems in LLM evaluation – sensitivity to prompt redundancy and its impact on model development. The example in section 1.1 demonstrates this issue very well.
    *   **Novel Approach:**  The game-theoretic formulation, particularly modeling the evaluation process as a three-player general-sum game is novel and opens new avenues for designing robust evaluation metrics.
    *   **Game-Theoretic Solution Concepts:**  The introduction of clone-invariant equilibrium solution concepts is a key contribution. Addressing equilibrium selection with affinity entropy and kernelized entropy is also significant.
    *   **Empirical Validation:** The paper supports its claims with both simulated experiments and real-world LLM evaluation data analysis. Comparing to the established Elo and demonstrating both failure modes and a different approach is very helpful.
    *   **Interpretability:** The authors do not only address the biases of the Elo ranking system but also aim to achieve interpretable ratings that provide insights into the interactions between prompts and models, and how the models differentiate themselves within those prompts.
    *   **Clear Structure and Presentation**: The writing is very clear and precise, making it easy to follow the logical train from the problem definition to the final results.

*   **Weaknesses:**

    *   **Computational Cost:**  Solving for general-sum game equilibria, even with the introduced optimization strategies, can be computationally expensive compared to Elo. While the authors show scalability to a decent-sized dataset, the practical limitations for very large-scale open-ended evaluations are not fully addressed. Further analysis should be given to real scalability with the algorithm.
    *   **Reliance on LLM Preference Ratings:**  The paper still relies on LLM (Gemini) for pairwise preference ratings. While acknowledging potential self-preference biases, a more thorough sensitivity analysis to the choice of the judging LLM would strengthen the findings. Are those problems still relevant regardless of the LLM preference ratings?
    *   **Limited Comparison to Other Alternatives:**  The paper's primary comparison is to Elo-based systems.  While this is a relevant benchmark, further comparison against other proposed solutions for robustness in LLM evaluation (like voting-based approaches or curated benchmark development) could strengthen the argument for the game-theoretic method.
    *   **Complexity**: While the theory is clear in its writing, the complexity of the method could make adoption difficult.

*   **Novelty and Significance:**

    *   **Novelty:**  The application of general-sum game theory to the LLM evaluation problem with the development of clone-invariant equilibrium selection criteria represents a substantial advance beyond existing approaches.
    *   **Significance:** The work tackles a fundamental challenge in LLM evaluation – ensuring fairness and robustness in open-ended systems. The proposed method has the potential to influence the design of future LLM evaluation frameworks and reduce biases in model development. The emphasis on equilibrium selection criteria is also important, and represents a significant step forward in game-theoretic evaluation.

*   **Potential Influence on the Field:** The paper can shift the focus in LLM evaluation from simple ranking to a more nuanced understanding of model performance and biases. It could influence future development of evaluation frameworks, leading to more robust and reliable assessments.

**Justification for Score:**

The paper presents a solid technical contribution with a well-motivated problem, a novel approach, and empirical validation. The weaknesses, primarily around computational cost and the dependence on LLM judges, are manageable and don't detract significantly from the core contribution.

**Score: 8**

- **Score**: 8/10

### **[Long-Context Inference with Retrieval-Augmented Speculative Decoding](http://arxiv.org/abs/2502.20330v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Long-Context Inference with Retrieval-Augmented Speculative Decoding":

**Summary:**

The paper introduces Retrieval-Augmented SPeculatIve Decoding (RAPID), a novel method for accelerating and improving the generation quality of long-context large language models (LLMs). RAPID addresses the computational inefficiencies of long-context inference, particularly the memory-bound KV cache operations that limit the effectiveness of standard speculative decoding (SD). RAPID utilizes a RAG drafter – a smaller LLM operating on shortened retrieved contexts – to speculate on the generation of the larger, long-context target LLM.  This allows leveraging potentially stronger LLMs as "drafters" while maintaining computational efficiency. The method also introduces a retrieval-augmented target distribution to enrich the target distribution by incorporating knowledge from the RAG drafter, enabling more acceptance of high-quality speculative candidates.  Experiments on LLaMA-3.1 and Qwen2.5 show significant performance improvements and speedups compared to baseline long-context inference and existing speculative decoding techniques.

**Critical Evaluation:**

*   **Novelty:** The core idea of combining retrieval-augmented generation (RAG) with speculative decoding (SD) is innovative and addresses a real bottleneck in long-context LLMs. The concept of a RAG drafter, operating on shortened retrieved contexts, specifically tailored to alleviate the memory constraints of the target LLM, is a notable contribution. The proposed retrieval-augmented target distribution, aiming to leverage potentially higher quality generations from a stronger drafter model, introduces a novel approach to knowledge transfer within speculative decoding. The upward-speculation method of using a larger LLM as a drafter is relatively new and shows an ability to perform knowledge transfer in the model decoding process. This paper shows that larger scale models may be more computationally efficient in SD decoding.

*   **Significance:** The paper tackles a significant challenge in the field of LLMs: efficiently processing and generating text from very long contexts. Long-context LLMs offer advantages over traditional RAG pipelines by avoiding the limitations of the retriever. However, their inference costs are prohibitive. RAPID provides a promising solution by offering both significant speedups and improvements in generation quality. The experimental results, showing substantial performance gains and >2x speedups on established benchmarks, demonstrate the practical impact of the proposed method. The analysis of robustness to retrieval quality is also significant, showing the method's resilience in real-world scenarios. The detailed analysis showing "emergent phenomenon" and greater speedups past the 32K context threshold are welcome results that point to the method's abilities. The multi-turn dialogue experiment highlights the application of RAPID to real-world generation tasks.

*   **Strengths:**
    *   **Clear Problem Definition:** The paper clearly articulates the challenges of long-context inference and the limitations of existing approaches.
    *   **Well-Motivated Approach:** The rationale for combining RAG and SD is well-explained and supported by empirical observations.
    *   **Comprehensive Experiments:** The experimental setup is thorough, with evaluations on multiple benchmarks, model scales, and configurations.
    *   **Detailed Analysis:** The paper provides insightful analyses of the results, exploring the impact of context length, retrieval quality, and knowledge transfer.
    *   **Reproducibility:** The authors provide code, increasing the likelihood of future work building upon this research.

*   **Weaknesses:**
    *   **Retrieval Dependency:** While the paper demonstrates robustness to retrieval *quality*, the method's performance is still inherently dependent on the effectiveness of the retriever. The retriever must be tuned to the task, leading to increased difficulty in using RAPID in an end-to-end system.
    *   **RAG Context Size tuning:** No guidance is provided as to the ideal retrieval context window in relationship to the target task. The size of the retrieved RAG context likely depends on the end-to-end application, which creates additional difficulty.

*   **Potential Impact:** RAPID has the potential to significantly impact the adoption of long-context LLMs by making them more computationally feasible. The improved generation quality could also lead to better performance in various downstream applications. The paper opens up new avenues for research in combining RAG and SD, potentially inspiring novel decoding algorithms and knowledge transfer techniques. It could encourage future research into better RAG systems that address the difficulty in fine-tuning a single RAG retriever for all generation tasks.

*   **Overall Assessment:** The paper presents a well-motivated, innovative, and empirically validated method for improving long-context LLM inference. The strengths outweigh the weaknesses, and the potential impact on the field is substantial.

**Score: 8**

**Rationale:** The paper demonstrates a significant advancement in long-context LLM inference. The combination of RAG and SD is a novel and impactful idea, offering a practical solution to a real-world problem. The comprehensive experiments and detailed analyses further strengthen the paper's contribution. While there are some limitations related to retrieval dependency, the overall novelty, significance, and potential impact of RAPID justify a high score.

- **Score**: 8/10

## Other Papers
### **[Accessing LLMs for Front-end Software Architecture Knowledge](http://arxiv.org/abs/2502.19518v1)**
### **[Cognitive networks highlight differences and similarities in the STEM mindsets of human and LLM-simulated trainees, experts and academics](http://arxiv.org/abs/2502.19529v1)**
### **[Winning Big with Small Models: Knowledge Distillation vs. Self-Training for Reducing Hallucination in QA Agents](http://arxiv.org/abs/2502.19545v1)**
### **[Repurposing the scientific literature with vision-language models](http://arxiv.org/abs/2502.19546v1)**
### **[When Large Language Models Meet Speech: A Survey on Integration Approaches](http://arxiv.org/abs/2502.19548v1)**
### **[Distill Not Only Data but Also Rewards: Can Smaller Language Models Surpass Larger Ones?](http://arxiv.org/abs/2502.19557v1)**
### **[Stay Focused: Problem Drift in Multi-Agent Debate](http://arxiv.org/abs/2502.19559v1)**
### **[Diffusion-based Planning with Learned Viability Filters](http://arxiv.org/abs/2502.19564v1)**
### **[Do Large Language Models Know How Much They Know?](http://arxiv.org/abs/2502.19573v1)**
### **[Where Are We? Evaluating LLM Performance on African Languages](http://arxiv.org/abs/2502.19582v1)**
### **[Introduction to Sequence Modeling with Transformers](http://arxiv.org/abs/2502.19597v1)**
### **[Revisiting Word Embeddings in the LLM Era](http://arxiv.org/abs/2502.19607v1)**
### **[Program Synthesis Dialog Agents for Interactive Decision-Making](http://arxiv.org/abs/2502.19610v1)**
### **[Evaluation of Hate Speech Detection Using Large Language Models and Geographical Contextualization](http://arxiv.org/abs/2502.19612v1)**
### **[Self-rewarding correction for mathematical reasoning](http://arxiv.org/abs/2502.19613v1)**
### **[Is Your Paper Being Reviewed by an LLM? A New Benchmark Dataset and Approach for Detecting AI Text in Peer Review](http://arxiv.org/abs/2502.19614v1)**
### **[Weaker LLMs' Opinions Also Matter: Mixture of Opinions Enhances LLM's Mathematical Reasoning](http://arxiv.org/abs/2502.19622v1)**
### **[3D Nephrographic Image Synthesis in CT Urography with the Diffusion Model and Swin Transformer](http://arxiv.org/abs/2502.19623v1)**
### **[Agentic Mixture-of-Workflows for Multi-Modal Chemical Search](http://arxiv.org/abs/2502.19629v1)**
### **[Taxonomy, Opportunities, and Challenges of Representation Engineering for Large Language Models](http://arxiv.org/abs/2502.19649v1)**
### **[SuPreME: A Supervised Pre-training Framework for Multimodal ECG Representation Learning](http://arxiv.org/abs/2502.19668v1)**
### **[Improving Adversarial Transferability in MLLMs via Dynamic Vision-Language Alignment Attack](http://arxiv.org/abs/2502.19672v1)**
### **[SubZero: Composing Subject, Style, and Action via Zero-Shot Personalization](http://arxiv.org/abs/2502.19673v1)**
### **[M-LLM Based Video Frame Selection for Efficient Video Understanding](http://arxiv.org/abs/2502.19680v1)**
### **[BEVDiffuser: Plug-and-Play Diffusion Model for BEV Denoising with Ground-Truth Guidance](http://arxiv.org/abs/2502.19694v1)**
### **[Language-Informed Hyperspectral Image Synthesis for Imbalanced-Small Sample Classification via Semi-Supervised Conditional Diffusion Model](http://arxiv.org/abs/2502.19700v1)**
### **[SAP-DIFF: Semantic Adversarial Patch Generation for Black-Box Face Recognition Models via Diffusion Models](http://arxiv.org/abs/2502.19710v1)**
### **[Teaching Dense Retrieval Models to Specialize with Listwise Distillation and LLM Data Augmentation](http://arxiv.org/abs/2502.19712v1)**
### **[Recent Advances on Generalizable Diffusion-generated Image Detection](http://arxiv.org/abs/2502.19716v1)**
### **[Sensing and Steering Stereotypes: Extracting and Applying Gender Representation Vectors in LLMs](http://arxiv.org/abs/2502.19721v1)**
### **[Few-Shot Multilingual Open-Domain QA from 5 Examples](http://arxiv.org/abs/2502.19722v1)**
### **[Tokens for Learning, Tokens for Unlearning: Mitigating Membership Inference Attacks in Large Language Models via Dual-Purpose Training](http://arxiv.org/abs/2502.19726v1)**
### **[Do Expressions Change Decisions? Exploring the Impact of AI's Explanation Tone on Decision-Making](http://arxiv.org/abs/2502.19730v1)**
### **[Preference Learning Unlocks LLMs' Psycho-Counseling Skills](http://arxiv.org/abs/2502.19731v1)**
### **[R1-T1: Fully Incentivizing Translation Capability in LLMs via Reasoning Learning](http://arxiv.org/abs/2502.19735v1)**
### **[HaLoRA: Hardware-aware Low-Rank Adaptation for Large Language Models Based on Hybrid Compute-in-Memory Architecture](http://arxiv.org/abs/2502.19747v1)**
### **[Beneath the Surface: How Large Language Models Reflect Hidden Bias](http://arxiv.org/abs/2502.19749v1)**
### **[Finding Local Diffusion Schrödinger Bridge using Kolmogorov-Arnold Network](http://arxiv.org/abs/2502.19754v1)**
### **[PolyPrompt: Automating Knowledge Extraction from Multilingual Language Models with Dynamic Prompt Generation](http://arxiv.org/abs/2502.19756v1)**
### **[In-Context Learning with Hypothesis-Class Guidance](http://arxiv.org/abs/2502.19787v1)**
### **[ChatMol: A Versatile Molecule Designer Based on the Numerically Enhanced Large Language Model](http://arxiv.org/abs/2502.19794v1)**
### **[MFSR: Multi-fractal Feature for Super-resolution Reconstruction with Fine Details Recovery](http://arxiv.org/abs/2502.19797v1)**
### **[Developmental Support Approach to AI's Autonomous Growth: Toward the Realization of a Mutually Beneficial Stage Through Experiential Learning](http://arxiv.org/abs/2502.19798v1)**
### **[UIFace: Unleashing Inherent Model Capabilities to Enhance Intra-Class Diversity in Synthetic Face Recognition](http://arxiv.org/abs/2502.19803v1)**
### **[Implicit Search via Discrete Diffusion: A Study on Chess](http://arxiv.org/abs/2502.19805v1)**
### **[Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts](http://arxiv.org/abs/2502.19811v1)**
### **[Foot-In-The-Door: A Multi-turn Jailbreak for LLMs](http://arxiv.org/abs/2502.19820v1)**
### **[Analyzing CLIP's Performance Limitations in Multi-Object Scenarios: A Controlled High-Resolution Study](http://arxiv.org/abs/2502.19828v1)**
### **[ProAPO: Progressively Automatic Prompt Optimization for Visual Classification](http://arxiv.org/abs/2502.19844v1)**
### **[One-for-More: Continual Diffusion Model for Anomaly Detection](http://arxiv.org/abs/2502.19848v1)**
### **[ConvCodeWorld: Benchmarking Conversational Code Generation in Reproducible Feedback Environments](http://arxiv.org/abs/2502.19852v1)**
### **[MIND: Towards Immersive Psychological Healing with Multi-agent Inner Dialogue](http://arxiv.org/abs/2502.19860v1)**
### **[C-Drag: Chain-of-Thought Driven Motion Controller for Video Generation](http://arxiv.org/abs/2502.19868v1)**
### **[MMKE-Bench: A Multimodal Editing Benchmark for Diverse Visual Knowledge](http://arxiv.org/abs/2502.19870v1)**
### **[Towards Multimodal Large-Language Models for Parent-Child Interaction: A Focus on Joint Attention](http://arxiv.org/abs/2502.19877v1)**
### **[Beyond the Tip of Efficiency: Uncovering the Submerged Threats of Jailbreak Attacks in Small Language Models](http://arxiv.org/abs/2502.19883v1)**
### **[High-Fidelity Relightable Monocular Portrait Animation with Lighting-Controllable Video Diffusion Model](http://arxiv.org/abs/2502.19894v1)**
### **[PrimeK-Net: Multi-scale Spectral Learning via Group Prime-Kernel Convolutional Neural Networks for Single Channel Speech Enhancement](http://arxiv.org/abs/2502.19906v1)**
### **[Order Doesn't Matter, But Reasoning Does: Training LLMs with Order-Centric Augmentation](http://arxiv.org/abs/2502.19907v1)**
### **[SkipPipe: Partial and Reordered Pipelining Framework for Training LLMs in Heterogeneous Networks](http://arxiv.org/abs/2502.19913v1)**
### **[LLM-driven Effective Knowledge Tracing by Integrating Dual-channel Difficulty](http://arxiv.org/abs/2502.19915v1)**
### **[Picking the Cream of the Crop: Visual-Centric Data Selection with Collaborative Agents](http://arxiv.org/abs/2502.19917v1)**
### **[Meta-Reasoner: Dynamic Guidance for Optimized Inference-time Reasoning in Large Language Models](http://arxiv.org/abs/2502.19918v1)**
### **[DiffCSS: Diverse and Expressive Conversational Speech Synthesis with Diffusion Models](http://arxiv.org/abs/2502.19924v1)**
### **[Image Referenced Sketch Colorization Based on Animation Creation Workflow](http://arxiv.org/abs/2502.19937v1)**
### **[GeoEdit: Geometric Knowledge Editing for Large Language Models](http://arxiv.org/abs/2502.19953v1)**
### **[Collaborative Stance Detection via Small-Large Language Model Consistency Verification](http://arxiv.org/abs/2502.19954v1)**
### **[Deterministic or probabilistic? The psychology of LLMs as random number generators](http://arxiv.org/abs/2502.19965v1)**
### **[Can Large Language Models Unveil the Mysteries? An Exploration of Their Ability to Unlock Information in Complex Scenarios](http://arxiv.org/abs/2502.19973v1)**
### **[The Lookahead Limitation: Why Multi-Operand Addition is Hard for LLMs](http://arxiv.org/abs/2502.19981v1)**
### **[Erasing Without Remembering: Safeguarding Knowledge Forgetting in Large Language Models](http://arxiv.org/abs/2502.19982v1)**
### **[3D-AffordanceLLM: Harnessing Large Language Models for Open-Vocabulary Affordance Detection in 3D Worlds](http://arxiv.org/abs/2502.20041v1)**
### **[Polish-ASTE: Aspect-Sentiment Triplet Extraction Datasets for Polish](http://arxiv.org/abs/2502.20046v1)**
### **[Collab-Overcooked: Benchmarking and Evaluating Large Language Models as Collaborative Agents](http://arxiv.org/abs/2502.20073v1)**
### **[LongRoPE2: Near-Lossless LLM Context Window Scaling](http://arxiv.org/abs/2502.20082v1)**
### **[Generative augmentations for improved cardiac ultrasound segmentation using diffusion models](http://arxiv.org/abs/2502.20100v1)**
### **[VDT-Auto: End-to-end Autonomous Driving with VLM-Guided Diffusion Transformers](http://arxiv.org/abs/2502.20108v1)**
### **[Self-Training Elicits Concise Reasoning in Large Language Models](http://arxiv.org/abs/2502.20122v1)**
### **[FlexiDiT: Your Diffusion Transformer Can Easily Generate High-Quality Samples with Less Compute](http://arxiv.org/abs/2502.20126v1)**
### **[Finite State Automata Inside Transformers with Chain-of-Thought: A Mechanistic Study on State Tracking](http://arxiv.org/abs/2502.20129v1)**
### **[Re-evaluating Open-ended Evaluation of Large Language Models](http://arxiv.org/abs/2502.20170v1)**
### **[Multimodal Representation Alignment for Image Generation: Text-Image Interleaved Control Is Easier Than You Think](http://arxiv.org/abs/2502.20172v1)**
### **[An Extensive Evaluation of PDDL Capabilities in off-the-shelf LLMs](http://arxiv.org/abs/2502.20175v1)**
### **[Layer-Aware Task Arithmetic: Disentangling Task-Specific and Instruction-Following Knowledge](http://arxiv.org/abs/2502.20186v1)**
### **[ChineseEcomQA: A Scalable E-commerce Concept Evaluation Benchmark for Large Language Models](http://arxiv.org/abs/2502.20196v1)**
### **[AI Will Always Love You: Studying Implicit Biases in Romantic AI Companions](http://arxiv.org/abs/2502.20231v1)**
### **[Attention Distillation: A Unified Approach to Visual Characteristics Transfer](http://arxiv.org/abs/2502.20235v1)**
### **[Teasing Apart Architecture and Initial Weights as Sources of Inductive Bias in Neural Networks](http://arxiv.org/abs/2502.20237v1)**
### **[FINEREASON: Evaluating and Improving LLMs' Deliberate Reasoning through Reflective Puzzle Solving](http://arxiv.org/abs/2502.20238v1)**
### **[Beyond Natural Language Perplexity: Detecting Dead Code Poisoning in Code Generation Datasets](http://arxiv.org/abs/2502.20246v1)**
### **[LLM as a Broken Telephone: Iterative Generation Distorts Information](http://arxiv.org/abs/2502.20258v1)**
### **[Large Language Models as Attribution Regularizers for Efficient Model Training](http://arxiv.org/abs/2502.20268v1)**
### **[Explainable, Multi-modal Wound Infection Classification from Images Augmented with Generated Captions](http://arxiv.org/abs/2502.20277v1)**
### **[Evaluating Human Trust in LLM-Based Planners: A Preliminary Study](http://arxiv.org/abs/2502.20284v1)**
### **[Conformal Tail Risk Control for Large Language Model Alignment](http://arxiv.org/abs/2502.20285v1)**
### **[Judge a Book by its Cover: Investigating Multi-Modal LLMs for Multi-Page Handwritten Document Transcription](http://arxiv.org/abs/2502.20295v1)**
### **[An exploration of features to improve the generalisability of fake news detection models](http://arxiv.org/abs/2502.20299v1)**
### **[M^3Builder: A Multi-Agent System for Automated Machine Learning in Medical Imaging](http://arxiv.org/abs/2502.20301v1)**
### **[Mobius: Text to Seamless Looping Video Generation via Latent Shift](http://arxiv.org/abs/2502.20307v1)**
### **[EAIRA: Establishing a Methodology for Evaluating AI Models as Scientific Research Assistants](http://arxiv.org/abs/2502.20309v1)**
### **[FlexVAR: Flexible Visual Autoregressive Modeling without Residual Prediction](http://arxiv.org/abs/2502.20313v1)**
### **[Long-Context Inference with Retrieval-Augmented Speculative Decoding](http://arxiv.org/abs/2502.20330v1)**
### **[Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models](http://arxiv.org/abs/2502.20332v1)**
### **[Expertise Is What We Want](http://arxiv.org/abs/2502.20335v1)**
### **[Thinking Slow, Fast: Scaling Inference Compute with Distilled Reasoners](http://arxiv.org/abs/2502.20339v1)**
### **[Sparse Auto-Encoder Interprets Linguistic Features in Large Language Models](http://arxiv.org/abs/2502.20344v1)**
### **[KEDRec-LM: A Knowledge-distilled Explainable Drug Recommendation Large Language Model](http://arxiv.org/abs/2502.20350v1)**
### **[Bridging the Creativity Understanding Gap: Small-Scale Human Alignment Enables Expert-Level Humor Ranking in LLMs](http://arxiv.org/abs/2502.20356v1)**
### **[Bridging Legal Knowledge and AI: Retrieval-Augmented Generation with Vector Stores, Knowledge Graphs, and Hierarchical Non-negative Matrix Factorization](http://arxiv.org/abs/2502.20364v1)**
### **[Constrained Generative Modeling with Manually Bridged Diffusion Models](http://arxiv.org/abs/2502.20371v1)**
### **[Tight Inversion: Image-Conditioned Inversion for Real Image Editing](http://arxiv.org/abs/2502.20376v1)**
### **[PhantomWiki: On-Demand Datasets for Reasoning and Retrieval Evaluation](http://arxiv.org/abs/2502.20377v1)**
### **[Multi-Agent Verification: Scaling Test-Time Compute with Multiple Verifiers](http://arxiv.org/abs/2502.20379v1)**
