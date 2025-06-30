# The Latest Daily Papers - Date: 2025-06-30
## Highlight Papers
### **[Noise-Inspired Diffusion Model for Generalizable Low-Dose CT Reconstruction](http://arxiv.org/abs/2506.22012v1)**
- **Summary**: Here's a summary and critical evaluation of the provided paper:

**Summary:**

The paper introduces a Noise-Inspired Diffusion model (NEED) for generalizable low-dose computed tomography (LDCT) reconstruction.  It addresses the challenge of deep learning models struggling to generalize to unseen dose levels.  NEED consists of two main components: (1) a shifted Poisson diffusion model (SPDiff) for pre-log LDCT projection data denoising, designed to align the diffusion process with the non-Gaussian noise characteristics of CT projections; and (2) a doubly guided diffusion model (DGDiff) for refining reconstructed images, using both LDCT images and initial reconstructions as guides.  The method employs a time step matching strategy to adapt to various dose levels during testing and is trained solely on normal-dose CT (NDCT) data. Experiments on two datasets demonstrate improved reconstruction and generalization performance compared to existing methods. The code is publicly available.

**Critical Evaluation:**

*   **Novelty:** The paper demonstrates novelty in several aspects. First, the SPDiff module directly addresses the unique noise characteristics of pre-log CT projection data, which is a departure from the common practice of applying Gaussian diffusion models. Second, the doubly guided diffusion model is a novel way to leverage both noisy LDCT images *and* initial reconstructions to refine the final image, potentially allowing for a more accurate estimation of prior information. The time step matching strategy is also a valuable contribution, enabling the model to adapt to unseen dose levels without retraining. The end-to-end integration of these components is a significant and novel architecture for LDCT reconstruction. The application of diffusion models to the *projection domain* with the explicit consideration of CT physics seems to be a key differentiating factor.

*   **Significance:** The significance of the work lies in its potential to improve the clinical applicability of LDCT reconstruction. The ability to generalize to unseen dose levels is crucial for reducing radiation exposure in diverse clinical settings. A model that requires only NDCT data for training simplifies deployment, as paired ND/LDCT data is often difficult and expensive to acquire. Furthermore, the reported improvements in reconstruction quality (both quantitative and qualitative) and segmentation accuracy could have a direct impact on diagnostic accuracy. The use of established datasets enables good comparability. The publicly available code enhances reproducibility and accelerates adoption by other researchers.

*   **Strengths:**
    *   Strong technical approach leveraging domain-specific knowledge of CT physics.
    *   Addresses an important clinical problem (generalization in LDCT).
    *   Demonstrated improved performance compared to state-of-the-art methods on multiple datasets.
    *   Well-written and clearly explains the proposed method.
    *   Publicly available code.
    *   The use of both qualitative and quantitative results, including segmentation-based metrics, strengthens the evaluation.

*   **Weaknesses:**
    *   While the paper mentions various factors influencing pre-log CT noise, the shifted Poisson model might still be a simplification of reality. More rigorous modeling or validation against real pre-log data could strengthen this aspect.
    *   The reliance on fan-beam simulation for creating the training data from post-log data. The use of "more real" data from a physical scanner would make it stronger.
    *   The paper mentions the image super-resolution step as a time-consuming component. The use of latent space for more efficient high-resolution CT image sampling might be an issue in limited hardware settings.

*   **Potential Influence:** The paper is likely to have a moderate to significant influence on the field. It presents a solid technical advancement, demonstrates practical benefits, and addresses a key challenge in LDCT reconstruction. Other researchers could build upon this work by exploring more advanced noise models, investigating alternative network architectures for the different modules, and studying the performance of the method in a wider range of clinical settings.

**Score: 8**

**Rationale:**

The paper presents a technically sound and novel approach to generalizable LDCT reconstruction using a well-justified combination of domain-specific knowledge and diffusion modeling. The experimental results are compelling and the code availability promotes reproducibility. While the method still relies on some simplifications (e.g., shifted Poisson noise model, fan-beam simulation) and faces computational challenges, the overall contribution is significant. The proposed techniques offer a pathway to more robust and clinically relevant LDCT reconstruction, impacting diagnosis and overall patient care by reducing X-ray dosage exposure.

- **Score**: 8/10

### **[Lost at the Beginning of Reasoning](http://arxiv.org/abs/2506.22058v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Lost at the Beginning of Reasoning":

**Summary:**

The paper investigates a previously underexplored weakness in long-chain-of-thought (CoT) reasoning in large language models (LLMs): failures stemming from flaws in the *initial* reasoning step, rather than errors accumulated over the entire chain. Through empirical analysis, the authors demonstrate that the first reasoning step has a disproportionately large impact on the final prediction accuracy. If this first step is incorrect, the model is significantly more likely to arrive at a wrong answer.  To combat this, they propose an efficient early pruning algorithm that uses a reward model to evaluate the quality of the first reasoning step, allowing the LLM to discard less promising reasoning paths and reduce inference cost. Finally, the authors introduce LaBoR, a new benchmark designed specifically to assess LLMs' ability to self-correct after being given a flawed initial reasoning step.

**Critical Evaluation:**

*   **Novelty:** The paper introduces a genuinely novel perspective on LLM reasoning. While much research has focused on error accumulation or "lost-in-the-middle" issues, this paper specifically targets the impact of the *initial* reasoning step. This is a significant contribution because it challenges the prevailing assumption that self-correction capabilities are sufficient to overcome early errors. The introduction of the LaBoR benchmark is also novel. Few (if any) benchmarks specifically evaluate the recovery from *deliberately flawed initial steps*.

*   **Significance:**  The findings have important implications for LLM development. Recognizing the importance of the first step could lead to new training techniques, better prompting strategies, and more robust architectures.  The early pruning algorithm directly addresses a practical problem: reducing inference cost without sacrificing accuracy. This is particularly valuable for long-CoT reasoning, which can be computationally expensive. The LaBoR benchmark could drive research into more effective self-correction mechanisms.

*   **Strengths:**
    *   **Clear Empirical Evidence:** The paper presents strong empirical evidence supporting its claims, using established reasoning model families (DeepSeek, Qwen) and relevant benchmarks. The ablation studies provide further insights into the critical role of the first step.
    *   **Practical Solution:** The early pruning algorithm is a practical and readily implementable solution that demonstrably improves efficiency.
    *   **Benchmark Contribution:** The LaBoR benchmark fills a gap in the existing evaluation landscape, providing a valuable tool for assessing self-correction abilities.
    *   **Reproducibility:** The paper provides sufficient details for reproducing the experiments and the method for creating the LaBoR benchmark is detailed, increasing the reproducibility.

*   **Weaknesses:**
    *   **Limited Scope:** The empirical evaluations focus primarily on mathematical and scientific reasoning tasks.  Generalizability to other types of reasoning (e.g., code generation, commonsense reasoning) is not fully demonstrated.
    *   **Reliance on Reward Model:** The effectiveness of the early pruning algorithm depends on the availability of a high-quality reward model.  Creating such a model can be challenging in some domains.
    *   **Benchmark Size:** While the LaBoR construction pipeline is valuable, the final dataset contains a relatively small number of questions which can impact its generalization.
    *   **Self-Correction in Closed-Source Models:** The evaluation of self-correction abilities of closed-source models using prompting-only based on thinking, raises questions about the accuracy of this evaluation, due to not having access to thinking stages directly.

*   **Potential Influence:**  The paper has the potential to influence the direction of LLM research in several ways.  It encourages a more nuanced understanding of error sources in CoT reasoning.  It provides a valuable tool (LaBoR) for evaluating self-correction. It offers a practical technique (early pruning) for improving efficiency.  The findings also raise questions about how LLMs are currently trained and prompted and what should be improved to make LLMs more resilient to initial mistakes.

**Justification for Score:**

The paper provides a novel perspective, empirical validation and practical solutions. While scope is limited, this work makes a valuable contribution to understanding the dynamics of reasoning in LLMs and suggests potential avenues for improvement.  The benchmark, though small, is a valuable asset.

**Score: 8**

- **Score**: 8/10

### **[Training Language Model to Critique for Better Refinement](http://arxiv.org/abs/2506.22157v1)**
- **Summary**: Okay, here's a summary and critical evaluation of the paper "Training Language Model to Critique for Better Refinement":

**Summary:**

The paper introduces Refinement-oriented Critique Optimization (RCO), a novel framework for training critic models to improve the refinement capabilities of large language models (LLMs).  Unlike traditional approaches that train critic models based on direct critique preference judgments, RCO focuses on training critics using signals derived from the *effectiveness* of critiques in guiding response refinement. The framework operates in a feedback loop: the critic model generates critiques of an initial LLM response, these critiques are used to guide an actor model to refine its response, and a "critique utility" (CU) metric is calculated based on the degree to which the refined responses are preferred over the initial response. This CU serves as the reward signal for training the critic model. The authors evaluate RCO across five tasks (dialog generation, summarization, question answering, mathematical reasoning, and code generation) and demonstrate that it outperforms existing methods and open-source models in critique quality and refinement outcomes. The core contribution is the RCO framework itself, a novel supervision scheme based on refined response preferences, and empirical validation of its effectiveness.

**Critical Evaluation:**

The paper presents a compelling idea that directly addresses a gap in LLM alignment research: connecting critique generation with actual output improvement. The RCO framework is well-motivated and addresses a real limitation of current critique-based training methods. The use of "critique utility" as a reward signal is innovative, as it shifts the focus from simply generating evaluative statements to generating actionable feedback.

**Strengths:**

*   **Novel Approach:** RCO offers a genuinely new approach to training critic models, moving away from explicit critique preference to a more implicit reward signal based on refinement success.
*   **Clear Motivation:** The paper clearly articulates the limitations of existing critique-based training methods and convincingly argues for the importance of connecting critique generation with output improvement.
*   **Well-Defined Framework:** The RCO framework is well-defined and relatively straightforward to implement. The "critique utility" metric provides a quantifiable measure of critique effectiveness.
*   **Comprehensive Evaluation:** The authors conduct extensive experiments across a diverse set of tasks, demonstrating the generalizability of the RCO framework. The benchmark suite consists of both established datasets and new setups.
*   **Strong Empirical Results:** The results show that RCO consistently outperforms baseline models and open-source LLMs and DPCO across various metrics.
*   **In-depth Analysis:** The paper also provides a solid analysis of the method’s impact on improving critique quality, evaluating the impact across different model sizes, and showing the advantage when scaling the judge model size for RCO.
*   **Clarity:** The paper is generally well-written and easy to follow, with clear explanations of the proposed framework and experimental setup.

**Weaknesses:**

*   **Computational Cost:** The framework has higher training costs, due to the need to refine outputs and assess preferences to derive the utility score. The paper acknowledges the computational demands. The ablation studies regarding critique and refinement number mitigates this somewhat, but the approach might still be infeasible for researchers with limited resources.
*   **Reliance on a "Strong" Judge Model:** The critique utility calculation relies on a preference judgement from a powerful (and presumably expensive) judge model. The paper uses Qwen2.5-72B, however, if this judge model is biased or flawed, it could negatively impact the training of the critic model. The performance with various judge models is also analyzed to showcase its robustness. But the high computation overheads from a large judge model might limit applications in scenarios that require fast feedback cycles.
*   **Limited Exploration of Critique Types:** The paper does not deeply explore the types of critiques that are most effective for different tasks. While the overall framework is promising, further research is needed to understand what *specific* characteristics make a critique conducive to successful refinement.
*   **Dependence on Actor Model:** The RCO framework's effectiveness is closely tied to the capabilities of the actor model used for refinement. A poorly performing actor model may not be able to effectively leverage the generated critiques, leading to a less informative reward signal.
*   **Focus on Accuracy:** The prompt used for evaluating the criteria on how to refine an output in human evaluations is biased towards accuracy, and might not be the case in all circumstances.

**Significance:**

The paper is significant because it provides a practical and effective way to train critic models that *actually improve* the performance of LLMs. By focusing on refinement outcomes, RCO moves beyond traditional evaluation-focused critique generation and enables the development of more useful and actionable feedback. This is a crucial step towards building LLMs that can continuously self-improve and adapt to new tasks.

**Score: 8/10**

**Justification:**

The paper introduces a novel and well-motivated framework for training critique models, supported by strong empirical results and a comprehensive evaluation. The shift from direct critique preference to a refinement-oriented approach is a significant contribution to the field. While the framework has some limitations in terms of computational cost, reliance on a good judge model, and limited exploration of critique types, its potential influence on LLM alignment and self-improvement is substantial. The paper presents convincing evidence that RCO can lead to the development of more effective and actionable feedback, ultimately contributing to the development of more capable and adaptable language models.

- **Score**: 8/10

### **[Optimal Estimation of Watermark Proportions in Hybrid AI-Human Texts](http://arxiv.org/abs/2506.22343v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper addresses the problem of estimating the proportion of watermarked text within a larger mixed-source document (i.e., documents containing both human and AI-generated content). It formulates this as a parameter estimation problem within a mixture model, using pivotal statistics to quantify the evidence for watermarking. The work shows that, for certain watermarking schemes like green-red lists, the proportion is unidentifiable. Conversely, for schemes using continuous pivotal statistics, identifiability can be achieved under certain conditions. The paper then proposes efficient estimators for these identifiable cases, provides theoretical guarantees on their performance, and establishes minimax lower bounds to demonstrate their optimality. Experiments on synthetic and real-world data (from open-source LLMs) validate the proposed estimators' accuracy.

**Critical Evaluation:**

*   **Novelty:** The paper's primary novelty lies in addressing the *estimation* of watermark proportions in mixed-source text, rather than the more common binary *detection* problem (is the whole document watermarked or not?). While there has been some prior work on identifying watermarked *segments*, this paper provides the first comprehensive framework for estimating the overall *proportion*. The analysis of identifiability conditions for different watermarking schemes is also a key contribution.
*   **Significance:** This work is significant because mixed-source text is increasingly prevalent in real-world scenarios. Accurate estimation of AI-generated content proportions has implications for authorship attribution, content authenticity assessment in critical domains (education, journalism, etc.), and setting acceptable thresholds for AI-assisted content generation. By providing a statistically grounded approach and demonstrating practically effective estimators, the paper offers a valuable tool for navigating the evolving landscape of AI-human collaboration in content creation. The minimax lower bounds are also a rigorous and important contribution.
*   **Strengths:**

    *   **Rigorous Theoretical Analysis:** The paper provides a strong theoretical foundation, including identifiability analysis, estimator convergence guarantees, and minimax lower bounds.
    *   **Practical Estimators:** The proposed estimators (particularly the refined estimator with optimal weight function) are computationally efficient and empirically validated.
    *   **Real-World Relevance:** The problem addressed is of increasing practical importance.
    *   **Clear Problem Formulation:** The formulation of the estimation problem is well-defined.
*   **Weaknesses:**

    *   **Model Simplifications:** The paper relies on Model 2 (surrogate model) for much of its theoretical development, which simplifies the dependencies inherent in autoregressive LLMs. Although the authors argue that this simplification is reasonable from the verifier's perspective, it does limit the direct applicability of the theoretical results to very complex editing/mixing scenarios. Experimental results are still given for the more accurate Model 1.
    *   **Dependence on Accurate Alternative CDF:** The refinement step depends on estimating Fp, the averaged alternative CDF. While the authors suggest using pivotal statistics from comparable open-source models for this purpose, the accuracy of this estimation could impact the performance in scenarios where such data is unavailable or the compared models are not similar.
    *   **Limited scope:** The paper focuses on scalar pivotal statistics, excluding watermarking schemes that use high-dimensional statistics.

*   **Potential Influence:** The paper has the potential to influence the field of text watermarking by shifting the focus from binary detection to more nuanced proportion estimation. It also provides a framework for analyzing the identifiability and estimability of different watermarking schemes, which can guide the design of more effective and statistically sound watermarking methods.

**Justification for Score:**

The paper makes a novel and significant contribution to the field of text watermarking by addressing the practically important problem of estimating the proportion of watermarked text in mixed-source documents. While the theoretical analysis involves some simplifying assumptions, the proposed estimators are well-motivated, rigorously analyzed, and empirically validated. The paper's clear problem formulation, strong theoretical foundation, and potential influence warrant a high score.

Score: 8

- **Score**: 8/10

### **[Probabilistic Optimality for Inference-time Scaling](http://arxiv.org/abs/2506.22376v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Probabilistic Optimality for Inference-time Scaling":

**Summary:**

The paper addresses the problem of efficient inference-time scaling for Large Language Models (LLMs). It argues that existing approaches rely on heuristics and lack a principled foundation. The authors propose a probabilistic framework to formalize the optimality of inference-time scaling, assuming independent and identically distributed (i.i.d.) parallel samples and an estimable probability distribution for Best-of-N selection. They derive a theoretical lower bound on the required number of samples to achieve a target performance level. Based on this framework, they develop OPTSCALE, an algorithm that dynamically determines the optimal number of sampled responses using a language model-based predictor to estimate probabilistic prior parameters. Experiments on mathematical reasoning benchmarks demonstrate that OPTSCALE reduces sampling overhead while maintaining or improving reasoning performance compared to state-of-the-art methods.

**Critical Evaluation:**

*   **Novelty:** The paper's primary novelty lies in its probabilistic framework for inference-time scaling. While techniques like Best-of-N and Self-Consistency are well-established, the authors provide a theoretical justification for them. Deriving a lower bound on the number of samples is a significant contribution, as it offers a principled approach to optimizing compute efficiency. The use of an LM-based predictor for estimating probabilistic prior parameters and dynamically adjusting sample sizes is also a novel aspect. However, the i.i.d. assumption is a potential limitation as it's not always strictly true of LLM outputs, though it may serve as a reasonable approximation.
*   **Significance:** The paper addresses a critical challenge in deploying LLMs for complex reasoning: the computational cost of inference-time scaling. The proposed OPTSCALE algorithm provides a practical solution for improving efficiency without sacrificing performance. The experimental results on established benchmarks support the algorithm's effectiveness, demonstrating substantial reductions in token consumption. The idea of dynamically allocating compute based on problem difficulty is valuable.
*   **Strengths:**

    *   Strong theoretical foundation with a probabilistic framework.
    *   Derivation of a lower bound on sample size requirements.
    *   Practical algorithm (OPTSCALE) with dynamic sample size adjustment.
    *   Empirical validation on standard benchmarks.
    *   Clear and well-organized presentation.

*   **Weaknesses:**

    *   The i.i.d. assumption may not hold perfectly for LLM outputs.
    *   The paper uses a specific Process Reward Model (PRM) for scoring. The performance might depend on the choice and quality of the PRM.
    *   The reliance on a truncated Gaussian distribution for the verifier score could be sub-optimal in cases of multi-modal distributions, though this is addressed.
    *   The gains reported appear to depend significantly on the particular dataset and model being used. It may be necessary to evaluate this technique with different models and broader datasets.
    *   Code is not yet available.

*   **Potential Impact:** The paper has the potential to significantly influence the field of LLM deployment by providing a principled and efficient approach to inference-time scaling. The OPTSCALE algorithm can be readily adopted by practitioners to reduce the computational cost of using LLMs for complex reasoning tasks. It also opens up avenues for future research on more sophisticated probabilistic models for inference-time scaling. The paper's theoretical framework could inspire further investigation into the fundamental limits of efficiency and accuracy in LLM inference.

**Rigorous Rationale:**
While there are weaknesses concerning the underlying assumptions and dependence on specific models, the paper introduces the concept of "principled" inference time scaling through a theoretical framework. The derivation of the sampling lower-bound in terms of performance is significant. Furthermore, the OPTSCALE implementation provides a tangible algorithm to test these theoretical results. The weaknesses of the paper involve some simplifications in the model and results. Nevertheless, given the theoretical contributions of the paper in the context of LLM performance, along with the practical demonstration, a relatively high score is warranted.

Score: 8

- **Score**: 8/10

### **[The Automated LLM Speedrunning Benchmark: Reproducing NanoGPT Improvements](http://arxiv.org/abs/2506.22419v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "The Automated LLM Speedrunning Benchmark: Reproducing NanoGPT Improvements":

**Summary:**

The paper introduces a new benchmark, the "Automated LLM Speedrunning Benchmark," designed to evaluate the ability of AI agents to reproduce research findings in the field of large language model (LLM) training. The benchmark leverages the NanoGPT speedrunning competition, which aims to train a GPT-2 model in the shortest time possible through community-driven optimizations. The benchmark tasks AI agents with reimplementing speedrun records, starting from a previous record and optionally using hints of varying detail (pseudocode, text descriptions, mini-papers) describing the improvements. The authors tested recent reasoning LLMs combined with state-of-the-art scaffolds on the benchmark and found that they struggle to reproduce even detailed hints innovations.

**Critical Evaluation:**

*   **Novelty:** The paper introduces a novel benchmark that specifically targets the reproducibility of LLM training improvements, which is a crucial aspect of scientific progress. While existing reproducibility benchmarks focus on replicating results from published papers, this benchmark focuses on the entire arc and detailed change logs of research over multiple innovations in a single research area. This is more relevant to current LLM research, allowing better comparison with code-level changes and against human researchers. It differs from other code reproduction benchmarks by providing various levels of hints as well, from pseudocode to mini-papers.

*   **Significance:** The benchmark has the potential to become a valuable tool for evaluating AI research agents and accelerating progress in automated scientific discovery. Demonstrating the limitations of current agents in reproducing known improvements highlights the remaining challenges in automating scientific research. By focusing on a focused scientific field like LLM training, the benchmark is more easily reproducible than other generalized AI research agents benchmarks. By having accurate, quantifiable metrics like 'FSR', comparisons between model benchmarks is made easier.

*   **Strengths:**
    *   The benchmark's design is well-motivated and directly addresses a critical need for reproducibility in AI research.
    *   The use of the NanoGPT speedrun provides a clear and measurable objective (training time) and a series of incremental improvements to reproduce.
    *   The different hint levels offer a controlled way to assess the impact of various types of information on an agent's performance.
    *   The inclusion of code-level ground truth targets and detailed change logs between records provides a solid foundation for evaluation.
    *   The evaluation and analysis are thorough, examining different LLMs, scaffolds, hint formats, and code similarity metrics.
    *   The authors openly acknowledge the benchmark's limitations and suggest future directions for improvement.
    * Open-sourced full code.

*   **Weaknesses:**
    *   The benchmark focuses on a relatively specific task (GPT-2 training) and may not generalize to other areas of LLM development or scientific research.
    *   The reliance on human-generated hints introduces a potential bias. The quality and style of the hints could significantly affect the agent's performance.
    *   The experiments are limited to a single hardware configuration (8xH100 node), which may not reflect real-world research environments.
    * The reliance on code similarity metric to the human solution might discourage some non-human innovations.
* **Impact:**
The paper may spur development of AI research agents and make an emphasis of reproducible research. The paper notes however that automated reproducibility is still a central challenge despite its potential.

Justification for Score:

The "Automated LLM Speedrunning Benchmark" is a valuable and timely contribution to the field of automated AI research. The paper tackles a critical issue – reproducibility – and provides a well-designed and rigorously evaluated benchmark for assessing AI agents' abilities in this area. While the benchmark has limitations, its potential to accelerate progress in automated scientific discovery and inform future research directions is significant. Given its novelty and potential impact, the benchmark warrants a high score, but not exceptional, given its limitations and narrow focus.

Score: 8

- **Score**: 8/10

## Other Papers
### **[LeanConjecturer: Automatic Generation of Mathematical Conjectures for Theorem Proving](http://arxiv.org/abs/2506.22005v1)**
### **[RoboEnvision: A Long-Horizon Video Generation Model for Multi-Task Robot Manipulation](http://arxiv.org/abs/2506.22007v1)**
### **[Noise-Inspired Diffusion Model for Generalizable Low-Dose CT Reconstruction](http://arxiv.org/abs/2506.22012v1)**
### **[LMPVC and Policy Bank: Adaptive voice control for industrial robots with code generating LLMs and reusable Pythonic policies](http://arxiv.org/abs/2506.22028v1)**
### **[SiPipe: Bridging the CPU-GPU Utilization Gap for Efficient Pipeline-Parallel LLM Inference](http://arxiv.org/abs/2506.22033v1)**
### **[GPAS: Accelerating Convergence of LLM Pretraining via Gradient-Preserving Activation Scaling](http://arxiv.org/abs/2506.22049v1)**
### **[Decoding Machine Translationese in English-Chinese News: LLMs vs. NMTs](http://arxiv.org/abs/2506.22050v1)**
### **[Lost at the Beginning of Reasoning](http://arxiv.org/abs/2506.22058v1)**
### **[Query as Test: An Intelligent Driving Test and Data Storage Method for Integrated Cockpit-Vehicle-Road Scenarios](http://arxiv.org/abs/2506.22068v1)**
### **[Transformers are Graph Neural Networks](http://arxiv.org/abs/2506.22084v1)**
### **[Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs](http://arxiv.org/abs/2506.22139v1)**
### **[Visual Structures Helps Visual Reasoning: Addressing the Binding Problem in VLMs](http://arxiv.org/abs/2506.22146v1)**
### **[Training Language Model to Critique for Better Refinement](http://arxiv.org/abs/2506.22157v1)**
### **[Exploring Modularity of Agentic Systems for Drug Discovery](http://arxiv.org/abs/2506.22189v1)**
### **[EFRame: Deeper Reasoning via Exploration-Filtering-Replay Reinforcement Learning Framework](http://arxiv.org/abs/2506.22200v1)**
### **[Adapting University Policies for Generative AI: Opportunities, Challenges, and Policy Solutions in Higher Education](http://arxiv.org/abs/2506.22231v1)**
### **[Projected Compression: Trainable Projection for Efficient Transformer Compression](http://arxiv.org/abs/2506.22255v1)**
### **[Public Service Algorithm: towards a transparent, explainable, and scalable content curation for news content based on editorial values](http://arxiv.org/abs/2506.22270v1)**
### **[Rethinking Visual Token Reduction in LVLMs under Cross-modal Misalignment](http://arxiv.org/abs/2506.22283v1)**
### **[OutDreamer: Video Outpainting with a Diffusion Transformer](http://arxiv.org/abs/2506.22298v1)**
### **[Evaluating Scoring Bias in LLM-as-a-Judge](http://arxiv.org/abs/2506.22316v1)**
### **[Optimal Estimation of Watermark Proportions in Hybrid AI-Human Texts](http://arxiv.org/abs/2506.22343v1)**
### **[Concept-Level AI for Telecom: Moving Beyond Large Language Models](http://arxiv.org/abs/2506.22359v1)**
### **[From Ground to Air: Noise Robustness in Vision Transformers and CNNs for Event-Based Vehicle Classification with Potential UAV Applications](http://arxiv.org/abs/2506.22360v1)**
### **[DiffSoundStream: Efficient Speech Tokenization via Diffusion Decoding](http://arxiv.org/abs/2506.22362v1)**
### **[Can Large Language Models Help Students Prove Software Correctness? An Experimental Study with Dafny](http://arxiv.org/abs/2506.22370v1)**
### **[Towards Fair Rankings: Leveraging LLMs for Gender Bias Detection and Measurement](http://arxiv.org/abs/2506.22372v1)**
### **[Probabilistic Optimality for Inference-time Scaling](http://arxiv.org/abs/2506.22376v1)**
### **[Exploration from a Primal-Dual Lens: Value-Incentivized Actor-Critic Methods for Sample-Efficient Online RL](http://arxiv.org/abs/2506.22401v1)**
### **[Refining Czech GEC: Insights from a Multi-Experiment Approach](http://arxiv.org/abs/2506.22402v1)**
### **[The Automated LLM Speedrunning Benchmark: Reproducing NanoGPT Improvements](http://arxiv.org/abs/2506.22419v1)**
### **[Shape-for-Motion: Precise and Consistent Video Editing with 3D Proxy](http://arxiv.org/abs/2506.22432v1)**
### **[MiCo: Multi-image Contrast for Reinforcement Visual Reasoning](http://arxiv.org/abs/2506.22434v1)**
