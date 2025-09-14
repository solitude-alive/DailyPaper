# The Latest Daily Papers - Date: 2025-09-14
## Highlight Papers
### **[Database Views as Explanations for Relational Deep Learning](http://arxiv.org/abs/2509.09482v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Database Views as Explanations for Relational Deep Learning":

**Summary:**

The paper introduces a novel framework for explaining the behavior of deep learning models trained over relational databases (RDL), particularly those based on heterogeneous graph neural networks (hetero-GNNs). The core idea is to use SQL view definitions to identify focused parts of the database that most contribute to a model's prediction. It adapts the classical notion of determinacy to establish global abductive explanations, allowing users to control the granularity and tradeoff between accuracy and conciseness. The paper proposes heuristic algorithms tailored for hetero-GNNs that avoid exhaustive search over all possible databases. The approach is evaluated empirically on the RelBench collection, demonstrating the usefulness and efficiency of the generated explanations.

**Critical Evaluation:**

*   **Novelty:** The paper presents a genuinely novel approach to explaining RDL models. Instead of focusing on internal model parameters or graph substructures (like many GNN explanation techniques), it grounds explanations directly in the database itself, using SQL views. This makes the explanations more accessible to database professionals who may not be familiar with machine learning intricacies. Adapting determinacy for explaining machine learning models is also a novel theoretical contribution.

*   **Significance:**
    *   The framework addresses a significant challenge: the inherent opacity of complex RDL models. Explainability is crucial for trust, debugging, and improving these models.
    *   By providing explanations in terms of SQL views, the paper bridges the gap between machine learning and database communities. Database administrators and data scientists can collaborate more effectively when explanations are formulated in a familiar language.
    *   The empirical evaluation on RelBench shows that the proposed techniques can produce meaningful and concise explanations, making them valuable for real-world applications.
    *   The exploration of different database perturbation strategies and explanation languages provides valuable insights for practical implementation.

*   **Strengths:**
    *   The conceptual framework is well-defined and theoretically sound.
    *   The use of SQL views is a practical and intuitive choice for explanation representation.
    *   The paper explores various explanation languages and perturbation techniques, demonstrating the flexibility of the framework.
    *   The empirical evaluation is extensive and covers a diverse set of databases and tasks.

*   **Weaknesses:**
    *   The heuristic algorithms for view discovery, while efficient, are still heuristics and may not always find the optimal explanations.  The learned mask approach, while effective, is also somewhat opaque, blurring the connection to the core determinacy idea.
    *   The focus is primarily on global explanations. Local explanations (explaining predictions for specific instances) are only implicitly supported.
    *   The cost model for explanation conciseness is relatively simple. More sophisticated cost models that consider the semantic complexity of SQL views could further enhance interpretability.
    *   The theoretical notion of *soft* determinacy is intuitive but its approximation through the learned masks does not have a strict justification. It should be explored how this approximation could be made rigorous.

*   **Impact:** The paper has the potential to significantly influence the field of RDL explainability. By offering a database-centric approach, it makes explanations more accessible and actionable for practitioners. The proposed framework can be extended to other types of RDL models and explanation tasks. The release of code and artifacts contributes to the reproducibility and adoption of the work.

**Overall:** The paper presents a well-motivated, novel, and valuable contribution to the field of RDL explainability. The framework is flexible, theoretically grounded, and empirically validated. While there are areas for future improvement (e.g., more sophisticated view discovery and a more detailed justification of the approximate determinacy notion), the paper represents a significant step forward in making RDL models more transparent and trustworthy.

**Score: 8**

**Rationale:** The paper's novelty and significance justify a high score. While there are some limitations in the approximation of the theoretical framework and the reliance on heuristics, the paper provides a substantial contribution that can potentially transform the way RDL models are understood and deployed in real-world applications. The SQL-based explanations, in particular, make it more accessible than most explanation methods that would be limited to ML experts.

Additionally, the empirical evaluation could be improved by further isolating individual explanations and demonstrating how those views can help guide ML models toward desired behavior.

- **Score**: 8/10

### **[Prompt Pirates Need a Map: Stealing Seeds helps Stealing Prompts](http://arxiv.org/abs/2509.09488v1)**
- **Summary**: Here is a summary and critical evaluation of the paper:

**Summary:**

The paper addresses the security risk of prompt-stealing attacks in diffusion models. It highlights that existing prompt recovery methods overlook the importance of the seed used in image generation. The paper identifies a common noise-generation vulnerability (CWE-339) in image-generation frameworks due to the limited seed range when using PyTorch PRNG on CPUs. It introduces *SeedSnitch*, a tool to recover seeds from images, and *PromptPirate*, a genetic algorithm-based optimization method for prompt stealing that leverages the recovered seed. The paper demonstrates that, by knowing the seed, prompt recovery is significantly more accurate.  The paper also discusses mitigation strategies by transitioning to cryptographically secure random number generation.

**Critical Evaluation:**

*Novelty and Significance:*

The paper offers several important contributions:

1.  **Identification of a Seed-Related Vulnerability:** Recognizing and exploiting the CWE-339 vulnerability in diffusion models, which stems from limitations in seed ranges, represents a significant and practical finding. Prior research has largely overlooked the crucial role of the seed in prompt recovery.
2.  **Practical Attack Tools:** The development of SeedSnitch for seed recovery and PromptPirate for prompt stealing provides concrete tools to demonstrate the real-world feasibility of the identified vulnerabilities.
3.  **Quantitative Analysis:** The large-scale empirical analysis on CivitAI images provides strong evidence supporting the claim that many images are generated with seeds that can be easily brute-forced.
4.  **Performance Improvement:**  PromptPirate demonstrably outperforms existing state-of-the-art methods in prompt stealing when the seed is known, further emphasizing the importance of the seed.
5.  **Mitigation Strategies:**  The paper offers relevant discussion and practical strategies for mitigating these vulnerabilities by strengthening the RNG process.
6. **Disclosure:** The authors responsibly disclosed their findings and obtained CVEs.

*Strengths:*

*   The paper is well-structured and clearly written.
*   The methodology is sound, and the experiments are comprehensive.
*   The paper provides valuable insights into the security of diffusion models.
*   The open-sourcing of the tools promotes transparency and reproducibility.
*   The responsible disclosure showcases ethical consideration.

*Weaknesses:*

*   **Limited Scope of Mitigation Discussion:** While the paper suggests mitigation strategies, it could benefit from a more in-depth exploration of these countermeasures' practical implementation challenges and potential side effects (e.g., impact on image generation speed or artistic control).
*  **Known Subject vs Unknown Subject Scenarios:** While important to split these, the 'known subject' scenario feels a little contrived as an evaluation because knowing the subject of the prompt trivializes the difficulty in the extraction itself.

*Potential Influence:*

This paper is likely to influence future research and development in the following ways:

*   **Increased Awareness:**  It will raise awareness about the importance of secure random number generation in diffusion models.
*   **Improved Security Practices:** It will encourage developers to adopt more robust seed generation methods.
*   **New Research Directions:** It will stimulate further research on prompt stealing attacks and defenses.
*  **Reproducible research:** Making their tool public can help others build on this contribution.

*Rigorous Rationale for Score:*

Given the clear novelty in identifying and exploiting a seed-related vulnerability, the practical attack tools developed, the extensive experimental validation, the demonstrated improvement over existing methods, and its implications for security and privacy in the growing field of generative AI, I assign the paper a score of **8**. The work clearly advances understanding in this area and provides a solid foundation for future research, although a deeper dive into the mitigation strategies could have further strengthened the paper. The responsible disclosure also adds a layer of significance.

Score: 8

- **Score**: 8/10

### **[LoCoBench: A Benchmark for Long-Context Large Language Models in Complex Software Engineering](http://arxiv.org/abs/2509.09614v1)**
- **Summary**: Here's a summary and critical evaluation of the LoCoBench paper:

**Summary:**

The paper introduces LoCoBench, a new benchmark designed to evaluate the capabilities of large language models (LLMs) with long context windows in complex software engineering scenarios.  It addresses the limitations of existing benchmarks that focus on single-function completion or short-context tasks. LoCoBench emphasizes evaluating capabilities such as understanding entire codebases, reasoning across multiple files, and maintaining architectural consistency in large software systems. The benchmark consists of 8,000 evaluation scenarios across 10 programming languages, with context lengths ranging from 10K to 1M tokens. The paper also presents a comprehensive evaluation framework comprising 17 metrics across 4 dimensions, including newly proposed metrics specifically for long-context capabilities. The authors evaluate state-of-the-art long-context models using LoCoBench, revealing performance gaps and demonstrating the need for further research in this area.

**Critical Evaluation:**

*   **Novelty:** The paper offers significant novelty by addressing a clear gap in existing LLM evaluation. While other benchmarks exist for code generation, repair, or summarization, LoCoBench uniquely focuses on comprehensive software engineering tasks requiring long-context reasoning. The creation of diverse, realistic codebases across various languages and domains, along with the proposed evaluation metrics (ACS, DTA, MMR, etc.), represents a valuable contribution. The systematic scaling of context length from 10K to 1M tokens is another unique feature. However, there are some existing benchmarks that address similar goals. The novelty, while present, is not necessarily ground-breaking, but the combination of features is new.

*   **Significance:** The benchmark's scale and diversity are significant.  Generating 8,000 scenarios and 50,000+ files is a substantial undertaking. The explicit focus on long-context understanding in software engineering is particularly relevant given the increasing size and complexity of real-world software projects. The benchmark provides a needed standardized way of comparing different LLMs in terms of software architecture understanding and complex development task performance. The evaluation metrics introduced are very important. This can push the LLM community into improving those areas. The open source nature makes it highly impactful to the field.

*   **Strengths:**

    *   Clear articulation of the problem and the limitations of existing benchmarks.
    *   A well-defined and systematic methodology for generating diverse and high-quality evaluation scenarios.
    *   A comprehensive evaluation framework with novel metrics tailored for long-context software engineering tasks.
    *   A large-scale benchmark with diverse programming languages and domains.
    *   Publicly available dataset and code, promoting reproducibility and further research.

*   **Weaknesses:**

    *   While the paper presents a broad overview of the evaluation results, a deeper analysis of specific failure modes and common challenges faced by LLMs would further strengthen the work.
    *   The authors could discuss more about how the generated code compares to human-written code in terms of quality metrics beyond simple compilation or bug rates. More analysis on the realism and generalizability is important.

*   **Potential Influence:** The paper will likely have a significant impact on the field of LLM research, particularly in the context of software engineering. LoCoBench provides a valuable resource for researchers to evaluate and improve the long-context reasoning capabilities of LLMs. The proposed evaluation metrics could become standard measures for assessing the effectiveness of LLMs in software development scenarios. Ultimately, the benchmark can accelerate research in AI-assisted software engineering and facilitate the development of more capable and reliable coding assistants.

**Score: 8.0**

**Justification:** LoCoBench is a well-designed benchmark that addresses a critical gap in the evaluation of long-context LLMs for software engineering. The scale, diversity, and comprehensiveness of the benchmark, along with the novel evaluation metrics, make it a significant contribution to the field. Although existing code benchmarks are improving at code generation, LoCoBench does a good job differentiating itself in the capabilities assessed. While a deeper analysis of failure modes would further strengthen the work, LoCoBench is a valuable contribution to the community. This score reflects a robust evaluation dataset, solid execution and impactful contributions to LLM research applied to realistic software engineering.
- **Score**: 8/10

### **[Measuring Epistemic Humility in Multimodal Large Language Models](http://arxiv.org/abs/2509.09658v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces HumbleBench, a new benchmark designed to evaluate the epistemic humility of multimodal large language models (MLLMs).  Epistemic humility, in this context, refers to the ability of a model to recognize when it *doesn't* know the answer and to abstain from making an incorrect assertion.  Unlike existing benchmarks that primarily focus on recognition accuracy (selecting the correct answer from distractors), HumbleBench assesses whether models can correctly identify situations where *none* of the provided answer options are valid. The benchmark is built using panoptic scene graph data and involves multiple-choice questions, where one option is "None of the above." The authors evaluate a range of state-of-the-art MLLMs on the benchmark and analyze their performance, highlighting the challenges MLLMs face in correctly rejecting plausible but incorrect answers. The authors further propose two stress tests to assess the models' failure to correctly choose the "none of the above" option and robustness to visually degraded images.

**Critical Evaluation:**

*   **Novelty:** The paper presents a genuinely novel and important contribution. The concept of evaluating epistemic humility in MLLMs is underexplored, and HumbleBench directly addresses this gap. Existing hallucination benchmarks largely ignore the critical ability of a model to abstain from answering when uncertain or when no correct answer is present.

*   **Significance:** The benchmark has considerable significance, especially as MLLMs are increasingly deployed in real-world applications where reliability and trustworthiness are paramount. Misinformation stemming from hallucinations can have serious consequences, and the ability of a model to admit uncertainty is crucial for safe and responsible AI. By focusing on false-option rejection, HumbleBench pushes the field beyond simply measuring recognition accuracy.

*   **Strengths:**

    *   **Well-defined Problem:**  The paper clearly articulates the problem of hallucination and the importance of epistemic humility.
    *   **Rigorous Methodology:** The authors present a well-defined data construction pipeline leveraging a panoptic scene graph dataset and GPT-4-Turbo, followed by a thorough manual filtering process. This approach enhances the data quality and reliability of the benchmark.
    *   **Comprehensive Evaluation:** The paper evaluates a diverse set of MLLMs, including both general-purpose and specialized reasoning models, providing valuable insights into their strengths and weaknesses.
    *   **Analysis & Insights:** The authors provide detailed analyses of the experimental results, including identifying failure modes and discussing the limitations of current approaches. The stress tests provided further insights on the challenges of this evaluation.

*   **Weaknesses:**

    *   **Dependence on GPT-4-Turbo:** While using GPT-4-Turbo for question generation is practical, it introduces a dependency on a specific model and could potentially reflect biases present in GPT-4-Turbo's output. While manual filtering tries to mitigate this, biases might slip through.
    *   **Potential Data Leakage:** Even with strict instructions to avoid it, some bias from the panoptic caption data may leak into questions, though manual filtering process is used to remove those questions.
    *   **Limited Explanation of Failure Modes:** The qualitative analysis of failure modes, while insightful, could be further expanded with quantitative analysis demonstrating the prevalence of each failure mode across different models and question types.

*   **Potential Influence:** HumbleBench has the potential to become a widely adopted benchmark for evaluating MLLMs, influencing future research directions and driving the development of more reliable and trustworthy models. It can encourage researchers to shift their focus from simply improving accuracy to also addressing the critical problem of uncertainty handling.

**Score:** 8

**Justification:**

HumbleBench presents a novel and significant contribution to the field of multimodal learning. The focus on epistemic humility and the development of a specialized benchmark to evaluate this capability are highly valuable. The rigorous methodology and comprehensive evaluation further strengthen the paper. While the dependency on GPT-4-Turbo and the potential for bias are limitations, the overall impact of the paper justifies a high score. The study sheds light on the critical issue of MLLM failure when presented with no correct answer and suggests that more accurate metrics are needed to capture hallucination robustness and epistemic humility.

- **Score**: 8/10

### **[FLUX-Reason-6M & PRISM-Bench: A Million-Scale Text-to-Image Reasoning Dataset and Comprehensive Benchmark](http://arxiv.org/abs/2509.09680v1)**
- **Summary**: Here's a summary and critical evaluation of the provided research paper:

**Summary:**

The paper introduces FLUX-Reason-6M, a large-scale (6 million images) text-to-image (T2I) reasoning dataset, and PRISM-Bench, a comprehensive benchmark for evaluating T2I models.  FLUX-Reason-6M is designed to address the lack of structured reasoning signals in existing datasets by incorporating six key characteristics: Imagination, Entity, Text rendering, Style, Affection, and Composition. It also features Generation Chain-of-Thought (GCoT) prompts that break down image generation into detailed steps. PRISM-Bench uses these six categories, plus a Long Text challenge leveraging GCoT, to evaluate T2I models using advanced vision-language models (VLMs) for nuanced assessment of prompt-image alignment and image aesthetics. The authors evaluate 19 leading models, revealing performance gaps and areas for improvement. The dataset, benchmark, and evaluation code are publicly released.

**Critical Evaluation:**

*   **Novelty:** The paper exhibits significant novelty in several aspects:

    *   *Dataset Scale and Focus:* FLUX-Reason-6M's scale (6M images) and emphasis on reasoning capabilities (via the six key characteristics and GCoT) represent a substantial advancement over existing datasets.  While some datasets focus on specific aspects of reasoning (e.g., layout planning), this dataset comprehensively addresses multiple facets.
    *   *GCoT Prompts:* The introduction of Generation Chain-of-Thought (GCoT) prompts is a significant innovation. These prompts offer a detailed breakdown of image generation steps, providing valuable supervisory signals for training T2I models.
    *   *PRISM-Bench Design:* PRISM-Bench's design, with its seven distinct tracks aligned to human judgment using advanced VLMs (GPT-4.1, Qwen2.5-VL-72B), is also a novel contribution. The benchmark provides a more reliable and discriminative evaluation than existing approaches that rely on saturated metrics like object detectors and basic CLIP scores. This contrasts with earlier benchmark approaches relying on crude CLIP scores and object detectors. The reliance on vision-language models for nuanced evaluation is a key strength.
    *   *Comprehensive Evaluation:* The evaluation of 19 leading T2I models, including both closed-source and open-source options, is also a strength, providing a broad overview of the current landscape.

*   **Significance:**

    *   *Addressing a Critical Gap:* The paper directly addresses the critical gap in the open-source T2I field: the lack of large-scale, reasoning-focused datasets and comprehensive evaluation benchmarks. This gap has hindered the progress of open-source models compared to closed-source systems. By releasing FLUX-Reason-6M and PRISM-Bench, the authors lower the barrier to entry for researchers and potentially accelerate the development of more capable T2I models.
    *   *Enabling Reasoning-Oriented T2I:* The focus on reasoning capabilities is crucial.  Modern T2I models need to understand complex prompts and generate images that align with human intentions.  FLUX-Reason-6M and PRISM-Bench provide the resources to train and evaluate models on this critical aspect.
    *   *Benchmarking State-of-the-Art:* The thorough evaluation of SOTA models on PRISM-Bench reveals specific performance gaps and areas for improvement, which informs future research directions.
    *   *Democratization:* Publicly releasing the dataset, benchmark, and evaluation suite helps democratize T2I research, making it more accessible to researchers worldwide.

*   **Strengths:**

    *   *Large-scale Data and Annotation:* The 6M image dataset with multi-label captions, bilingual support, and GCoT annotations represents a substantial resource.
    *   *Rigorous Benchmark Design:*  The use of advanced VLMs and human-aligned metrics for PRISM-Bench ensures a more reliable and discriminative evaluation.
    *   *Comprehensive Evaluation:* Evaluating a wide range of models provides valuable insights into the current state of the field.
    *   *Public Availability:*  Releasing the dataset and benchmark promotes collaboration and accelerates research.

*   **Weaknesses:**

    *   *Synthetic Data:* The dataset is entirely synthetically generated. While the authors use powerful models like FLUX.1-dev, biases in the generation process could still exist and limit the generalizability of models trained on FLUX-Reason-6M. In contrast to earlier benchmarks that relied on web scraping the synthetic origin may pose limits on certain benchmarks.
    *   *Reliance on VLMs for Evaluation:*  While using VLMs for evaluation is a strength, the evaluation is still inherently reliant on the capabilities and potential biases of these models. The VLM's limitations might influence or distort the accuracy or nature of benchmarks.
    *   *Computational Cost:* While releasing the dataset democratizes access, the 15,000 A100 GPU days required for creation might still be a barrier for smaller research groups wishing to create similar datasets. It is worth mentioning whether the dataset creation involved significant amounts of human labor in addition to the computational resources.

*   **Score Justification:**

    The paper addresses a significant problem in the T2I field and offers innovative solutions in the form of a large-scale dataset, a comprehensive benchmark, and a thorough evaluation of state-of-the-art models. While the synthetic nature of the data and reliance on VLMs for evaluation introduce some limitations, the overall impact of the work is high. The release of the dataset and benchmark will likely stimulate further research and development in reasoning-oriented T2I.

Score: 8

- **Score**: 8/10

## Other Papers
### **[Changing the Paradigm from Dynamic Queries to LLM-generated SQL Queries with Human Intervention](http://arxiv.org/abs/2509.09461v1)**
### **[Database Views as Explanations for Relational Deep Learning](http://arxiv.org/abs/2509.09482v1)**
### **[Prompt Pirates Need a Map: Stealing Seeds helps Stealing Prompts](http://arxiv.org/abs/2509.09488v1)**
### **[Mixture of Semantics Transmission for Generative AI-Enabled Semantic Communication Systems](http://arxiv.org/abs/2509.09499v1)**
### **[DeMeVa at LeWiDi-2025: Modeling Perspectives with In-Context Learning and Label Distribution Learning](http://arxiv.org/abs/2509.09524v1)**
### **[Prompting the Market? A Large-Scale Meta-Analysis of GenAI in Finance NLP (2022-2025)](http://arxiv.org/abs/2509.09544v1)**
### **[Improving Video Diffusion Transformer Training by Multi-Feature Fusion and Alignment from Self-Supervised Vision Encoders](http://arxiv.org/abs/2509.09547v1)**
### **[Finite Scalar Quantization Enables Redundant and Transmission-Robust Neural Audio Compression at Low Bit-rates](http://arxiv.org/abs/2509.09550v1)**
### **[Fluent but Unfeeling: The Emotional Blind Spots of Language Models](http://arxiv.org/abs/2509.09593v1)**
### **[How much are LLMs changing the language of academic papers after ChatGPT? A multi-database and full text analysis](http://arxiv.org/abs/2509.09596v1)**
### **[LAVA: Language Model Assisted Verbal Autopsy for Cause-of-Death Determination](http://arxiv.org/abs/2509.09602v1)**
### **[Mechanistic Learning with Guided Diffusion Models to Predict Spatio-Temporal Brain Tumor Growth](http://arxiv.org/abs/2509.09610v1)**
### **[LoCoBench: A Benchmark for Long-Context Large Language Models in Complex Software Engineering](http://arxiv.org/abs/2509.09614v1)**
### **[Bridging the Capability Gap: Joint Alignment Tuning for Harmonizing LLM-based Multi-Agent Systems](http://arxiv.org/abs/2509.09629v1)**
### **[DiFlow-TTS: Discrete Flow Matching with Factorized Speech Tokens for Low-Latency Zero-Shot Text-To-Speech](http://arxiv.org/abs/2509.09631v1)**
### **[All for One: LLMs Solve Mental Math at the Last Token With Information Transferred From Other Tokens](http://arxiv.org/abs/2509.09650v1)**
### **[Measuring Epistemic Humility in Multimodal Large Language Models](http://arxiv.org/abs/2509.09658v1)**
### **[Steering MoE LLMs via Expert (De)Activation](http://arxiv.org/abs/2509.09660v1)**
### **[Locality in Image Diffusion Models Emerges from Data Statistics](http://arxiv.org/abs/2509.09672v1)**
### **[CDE: Curiosity-Driven Exploration for Efficient Reinforcement Learning in Large Language Models](http://arxiv.org/abs/2509.09675v1)**
### **[The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs](http://arxiv.org/abs/2509.09677v1)**
### **[ButterflyQuant: Ultra-low-bit LLM Quantization through Learnable Orthogonal Butterfly Transforms](http://arxiv.org/abs/2509.09679v1)**
### **[FLUX-Reason-6M & PRISM-Bench: A Million-Scale Text-to-Image Reasoning Dataset and Comprehensive Benchmark](http://arxiv.org/abs/2509.09680v1)**
