# The Latest Daily Papers - Date: 2025-03-22
## Highlight Papers
### **[EfficientLLaVA:Generalizable Auto-Pruning for Large Vision-language Models](http://arxiv.org/abs/2503.15369v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper "EfficientLLaVA: Generalizable Auto-Pruning for Large Vision-language Models" introduces a novel automatic pruning method specifically designed to reduce the computational complexity of large vision-language models (LVLMs) for deployment on resource-constrained devices.  Unlike traditional pruning methods that rely heavily on the original model's extensive training data, EfficientLLaVA uses a small number of proxy samples to search for an optimal pruning policy.  The approach focuses on maximizing the generalization ability of the pruning policy to unknown training data using structural risk minimization (SRM).  The method iteratively searches for the optimal pruning policy within a given space and optimizes the vision projector to improve the overall performance upper bound.  Experiments on ScienceQA, Vizwiz, MM-vet, and LLaVA-Bench demonstrate that EfficientLLaVA achieves significant speedups with minimal accuracy loss compared to dense models and other pruning techniques.

**Critical Evaluation:**

**Novelty:**

The paper presents a significant and novel approach to pruning large vision-language models (LVLMs) for resource-constrained environments. The key innovations lie in:

1.  **Few-Shot Pruning with Generalization Focus:**  Traditional pruning methods often require extensive retraining or validation on large datasets.  EfficientLLaVA's ability to determine a good pruning policy with just a few samples is a significant departure and a substantial contribution.

2.  **Structural Risk Minimization for Pruning Policy:** The use of structural risk minimization (SRM) to guide the pruning policy search is a unique and effective strategy.  By explicitly targeting the generalization gap, the method avoids overfitting the proxy samples and produces a pruning policy that performs well on unseen data.

3.  **Search Space Evolution via Vision Projector Optimization:** The iterative refinement of the search space by optimizing the vision projector is a clever way to improve the upper bound of the potential performance of different pruning policies. This allows the pruning process to converge toward better performing areas of the pruning policy space.

**Significance:**

The significance of this work stems from addressing a critical challenge in deploying LVLMs: their massive size and computational requirements.  EfficientLLaVA offers a practical solution to this problem, making it possible to deploy these powerful models on mobile devices, robots, and other resource-limited platforms. This has implications for a wide range of applications, including:

*   **Mobile AI:** Enabling sophisticated vision-language understanding on smartphones.
*   **Robotics:** Facilitating more complex reasoning and interaction capabilities for robots.
*   **Autonomous Vehicles:** Enhancing the perception and decision-making abilities of autonomous vehicles.

**Strengths:**

*   **Strong Empirical Results:** The paper presents compelling experimental results across multiple datasets (ScienceQA, Vizwiz, MM-vet, LLaVA-Bench) demonstrating the effectiveness of EfficientLLaVA. The comparisons against strong baselines like SparseGPT and LLM-Pruner further highlight the advantages of the proposed method.
*   **Well-Defined Methodology:**  The paper clearly explains the technical details of EfficientLLaVA, including the formulation of the generalization gap, the search policy optimization, and the search space evolution process.
*   **Practical Relevance:**  The work directly addresses a pressing practical problem in deploying large AI models. The demonstrated speedups and minimal accuracy loss are particularly valuable.

**Weaknesses:**

*   **Limited Analysis of the Vision Projector's Impact:** While the paper demonstrates the effectiveness of optimizing the vision projector for search space evolution, it could benefit from a more in-depth analysis of *why* this approach works. Understanding the specific properties of the projector that are being improved could provide further insights and guidance for future work.
*   **Computational Cost of Search Space Exploration:** Although the method reduces the dependency on large datasets, it would be beneficial to explicitly state the computational overhead associated with the evolutionary algorithm to discover better pruning policies within a large space with respect to the resources required for pruning policy search and space evolution.
*   **GPU Utilization Discussion:** The GPU utilization of the models after the pruning is not discussed. This would add to the practical value of the paper.

**Overall:**

EfficientLLaVA represents a significant advancement in the field of LVLM pruning. Its novel approach to few-shot pruning with a generalization focus makes it a highly valuable contribution. The SRM-based formulation and search space evolution technique are well-motivated and effectively demonstrated. Although there are some areas for further exploration, the paper's strengths outweigh its weaknesses, and it is likely to have a notable impact on the deployment of LVLMs in resource-constrained environments.

Score: 8.5

- **Score**: 8/10

### **[Di$\mathtt{[M]}$O: Distilling Masked Diffusion Models into One-step Generator](http://arxiv.org/abs/2503.15457v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper presents Di[M]O, a novel approach for distilling Masked Diffusion Models (MDMs) into one-step generators. This addresses the slow inference speed of MDMs, a significant bottleneck in their application.  Di[M]O tackles two primary challenges: 1) The difficulty in leveraging intermediate-step information during one-step generation, which they solve by matching token-level distributions using an "on-policy" framework with an auxiliary model, and 2) The lack of entropy in the initial MDM distribution, addressed through a token initialization strategy that injects randomness while preserving similarity to the teacher's training data. The authors demonstrate the effectiveness of Di[M]O on class-conditional and text-conditional image generation tasks, achieving performance comparable to multi-step teacher models with a drastically reduced inference time.

**Critical Evaluation:**

*   **Novelty:** The paper's primary novelty lies in successfully achieving one-step distillation for *discrete* masked diffusion models.  Previous distillation efforts have focused primarily on continuous diffusion models, which have a PF-ODE and score function that MDMs lack.  The token-level distribution matching is also a novel solution for this specific distillation task. The initialization strategy also demonstrates novelty for tackling the challenge of initial entropy from the original setup.

*   **Significance:** This research is significant because it directly addresses a major limitation of MDMs: their slow inference speed. One-step generation makes MDMs more practical for real-time applications and resource-constrained environments. The approach opens up new avenues for efficient generative modeling, which has broad implications across various domains. Furthermore, this is among the first works to explore distilling MDMs for text-to-image generation.

*   **Strengths:**
    *   **Clear Problem Definition and Solution:** The paper clearly identifies the challenges specific to distilling MDMs, which is essential given the differences from continuous diffusion models. The proposed solutions are well-motivated and technically sound.
    *   **Comprehensive Evaluation:** The authors conduct extensive experiments on both class-conditional (ImageNet) and text-conditional (LAION-Aesthetics-6+) image generation tasks. The use of multiple metrics (FID, IS, HPSv2, Geneval) provides a robust evaluation of the generated images.
    *   **Ablation Studies:** The ablation studies are well-designed and provide insights into the contribution of each component of Di[M]O (initial mask ratio, Jeffrey Coefficient, Gaussian Perturbation).
    *   **Qualitative Results:** The visual results demonstrate the ability of Di[M]O to generate high-quality images comparable to the teacher model.
    *   **Thoroughness**: The paper provides comprehensive implementation details and additional appendices further strengthening the validity of the method.

*   **Weaknesses:**
    *   **Scope of Evaluation:** While the quantitative metrics are strong, the evaluation is centered around image generation. Further evaluation across other MDM applications such as protein design or audio synthesis would further validate the generalizability of Di[M]O.
    *   **Dependency on Teacher Model:** The method relies heavily on the teacher model's capabilities. It would be interesting to see how it performs on different architectures.

*   **Potential Influence:** The paper has the potential to significantly influence the field of generative modeling. It provides a practical solution for deploying MDMs in real-world applications. The token-level distillation approach and the initial randomization strategy could inspire further research on efficient and effective distillation techniques for discrete data generation.

*   **Justification of Score:** The paper addresses an important problem in a novel and well-executed manner. The results demonstrate the effectiveness of Di[M]O in achieving one-step generation for MDMs with competitive performance. While the evaluation could be broader, the paper provides a significant step forward in making MDMs more practical and accessible. The work will likely stimulate further research in distillation for MDMs and other discrete generative models.

**Score: 8**

- **Score**: 8/10

### **[FP4DiT: Towards Effective Floating Point Quantization for Diffusion Transformers](http://arxiv.org/abs/2503.15465v1)**
- **Summary**: ### Summary: The paper titled "FP4DiT: Towards Effective Floating Point Quantization for Diffusion Transformers" addresses the challenge of deploying Diffusion Models (DM) efficiently on edge devices due to their high computational costs and large model sizes. While post-training quantization (PTQ), specifically integer quantization, has been previously employed to mitigate these issues, the authors identify limitations in existing methods that focus primarily on traditional convolutional U-Nets and do not adapt well to newer Diffusion Transformer (DiT) architectures. To overcome these limitations, the authors propose a novel method named FP4DiT, which utilizes Floating-Point Quantization (FPQ) rather than integer quantization. Their approach includes an extension of the Adaptive Rounding PTQ technique, focusing on robust online activation quantization that considers the variability in activation depending on the input data. Experimental results indicate that FP4DiT surpasses previous integer-based PTQ techniques in terms of visual output quality across several metrics, demonstrating its effectiveness in generating convincing visual content on multiple state-of-the-art datasets. ### Critical Evaluation: **Novelty and Contribution**: The proposal of FP4DiT is quite notable, as it introduces a new quantization method specifically tailored for Diffusion Transformers, an area that has not been thoroughly explored previously in the question of floating point quantization. The focus on adapting quantization techniques meant for traditional architectures to the needs of DiTs represents a significant step forward in the field. **Strengths**: 1. **Timeliness**: Given the rise of Diffusion Transformers in generating high-quality images, this research is relevant and addresses a pressing challenge in practical applications. 2. **Methodological Innovation**: The adaptation and enhancement of quantization techniques to suit the unique characteristics of DiTs indicates a thoughtful and innovative approach to the problem. 3. **Empirical Validation**: The authors provide empirical results that demonstrate the efficacy of their proposed method over existing techniques, which strengthens the paper's practical utility. **Weaknesses**: 1. **Scope of Evaluation**: The empirical tests, while promising, might benefit from a broader scope of evaluation across more diverse datasets or comparisons against other emerging techniques in the space of floating-point quantization that were not previously included. 2. **Discussions on Limitations**: The paper could delve deeper into potential limitations of FPQ in certain scenarios, such as how it might perform against extensive varied input distributions beyond the tested sets. **Potential Impact**: Given that the demand for efficient model deployment on edge devices is ever-increasing, the introduction of FP4DiT could significantly influence further research in quantization techniques, especially for deep learning models relying on transformer architectures. ### Score: 8 **Rationale**: The paper scores an 8 due to its solid novelty, timely relevance, and methodological contributions, which represent a substantial advancement in the quantization of advanced models. However, it falls short of a perfect score because of the limited scope of its evaluation and lack of broader context regarding potential limitations. Further research building on this foundation could yield even greater insights and validation within the field.
- **Score**: 8/10

### **[AutoRedTeamer: Autonomous Red Teaming with Lifelong Attack Integration](http://arxiv.org/abs/2503.15754v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces AutoRedTeamer, a novel framework for fully automated red teaming of large language models (LLMs). AutoRedTeamer combines a multi-agent architecture with a memory-guided attack selection mechanism to enable continuous discovery and integration of new attack vectors. The framework consists of a red teaming agent that operates from high-level risk categories and a strategy proposer agent that autonomously discovers and implements new attacks by analyzing recent research. Experiments show AutoRedTeamer achieves higher attack success rates and reduced computational costs compared to existing approaches, while also matching the diversity of human-curated benchmarks.

**Critical Evaluation:**

*   **Novelty:** The paper demonstrates novelty in several key aspects. First, the dual-agent architecture, combining a strategy proposer and a red-teaming agent, allows for both exploitation of existing vulnerabilities and continuous discovery of new ones. This is a significant step beyond existing automated red-teaming techniques that typically focus on optimizing individual attack vectors or refining human-provided prompts. Second, the memory architecture, which tracks the success rate of attack vector combinations, enables the framework to learn from experience and adapt its strategies across different domains. This is a valuable addition to the field. However, the novelty is slightly tempered by the reliance on LLMs for attack proposal and test case generation, meaning the system's creativity is bounded by the LLM's own capabilities.

*   **Significance:** The paper addresses a critical challenge in the field of LLM security - the need for comprehensive and scalable evaluation of vulnerabilities. The automated nature of AutoRedTeamer makes it a significant contribution towards achieving this goal. The ability to generate diverse test cases from high-level risk categories is particularly valuable, as it allows for more comprehensive coverage of potential vulnerabilities. The reported empirical results, showing improved attack success rates and reduced computational costs, further highlight the practical significance of the framework. The continuous integration aspect is important for keeping up with the evolving landscape of LLM vulnerabilities. The paper may face challenges in broad deployment, especially to models that are well defended against prompt-based attacks, such as Gemini Ultra.

*   **Strengths:**

    *   **Comprehensive Framework:** The paper presents a well-designed and comprehensive framework that addresses several limitations of existing approaches.
    *   **Continuous Learning:** The memory architecture enables continuous learning and adaptation, which is crucial for staying ahead of emerging threats.
    *   **Empirical Validation:** The paper provides strong empirical evidence to support the effectiveness of AutoRedTeamer across diverse evaluation settings and models.
    *   **Scalability:** The automated nature of the framework allows for scalable evaluation of LLM security.

*   **Weaknesses:**

    *   **LLM Dependency:** The reliance on LLMs for attack proposal and test case generation introduces potential biases and limitations.
    *   **Potential for Overfitting:** There is a risk of AutoRedTeamer overfitting to specific model vulnerabilities or evaluation setups, which could limit its generalizability.
    *   **Complexity:** The multi-agent architecture and memory system add complexity to the framework, which could make it more difficult to implement and maintain.
    *   **Defense Awareness:** The paper does not delve into how AutoRedTeamer adapts to LLMs that utilize defenses against adversarial attacks.

**Justification:**

AutoRedTeamer offers a significant contribution to the field of LLM security by providing a comprehensive, scalable, and continuously evolving framework for automated red teaming. While the reliance on LLMs introduces some limitations, the framework's unique architecture and empirical results justify a high score. It is likely to influence future research in automated LLM security evaluation and red teaming.

Score: 8

- **Score**: 8/10

### **[Detecting LLM-Written Peer Reviews](http://arxiv.org/abs/2503.15772v1)**
- **Summary**: Here is a concise summary and critical evaluation of the paper:

**Summary:**
The paper addresses the growing concern of reviewers using Large Language Models (LLMs) to generate peer reviews instead of writing them independently. It introduces a novel approach for detecting LLM-generated reviews through indirect prompt injection, where watermarks are embedded in the review by specific instructions added to the original manuscript's PDF. The paper presents various watermarking schemes, including font embedding and jailbreaking, along with statistical tests that maintain a bounded family-wise error rate (FWER). The results show the high success rate of embedding watermarks across models, resilience to reviewer defenses, and the effectiveness of the statistical tests in detecting LLM-generated reviews.

**Evaluation:**

*   **Novelty:** The paper proposes a novel and practical method for detecting LLM-generated peer reviews, a critical issue in academic integrity. The use of indirect prompt injection through watermarks is a clever approach.
*   **Significance:** The paper addresses a relevant and timely problem. LLM-assisted reviews can bias the peer-review process and compromise its integrity. A tool to detect such reviews has a significant impact.
*   **Strengths:** The watermarking schemes are statistically testable, and the paper provides formal guarantees for error rates. The experimental results are comprehensive, covering multiple LLMs, datasets, and reviewer defenses.
*   **Weaknesses:** Some techniques, such as random start watermarks, may be easily identifiable and removed. Also, the cryptic prompt injection relies on complex techniques, and performance varies across different LLMs.

**Score:** 8

**Justification:**
The paper makes a substantial contribution to addressing a vital problem in academic integrity. The watermarking scheme is statistically sound, and the method has been validated across several dimensions with comprehensive empirical testing. While some methods have limitations and the cryptic prompt injection may be challenging to implement, the overall approach is innovative and offers a practical way to detect LLM-generated peer reviews, making it a significant advancement within the field.

- **Score**: 8/10

### **[Attention Pruning: Automated Fairness Repair of Language Models via Surrogate Simulated Annealing](http://arxiv.org/abs/2503.15815v1)**
- **Summary**: Here's a summary and critical evaluation of the provided paper:

**Summary:**

The paper introduces ATTENTION PRUNING (AP), a novel approach to mitigate bias in Large Language Models (LLMs) through selective attention head pruning.  AP leverages simulated annealing (SA) for efficient exploration of the vast combinatorial search space of attention head configurations.  To overcome the computational cost of directly evaluating LLMs during SA, the authors introduce surrogate Deep Neural Networks (DNNs) to approximate the fairness and utility (perplexity) metrics associated with different attention head configurations.  Experimental results on several LLMs demonstrate that AP can achieve significant reductions in gender bias, outperforming state-of-the-art fairness-aware pruning strategies, while minimizing the impact on model utility. The paper also explores how to tune hyperparameters to control the fairness-utility trade-off and demonstrates that reducing gender bias can also reduce other forms of social bias.

**Critical Evaluation:**

* **Novelty:** The paper has good novelty in its approach, combining surrogate models with simulated annealing for a fairness-aware pruning strategy. The use of surrogate models to approximate LLM behavior is a smart way to overcome the computational constraints of dealing with billion-parameter models. This use of surrogates is a key component contributing to the overall novelty, especially when considering the combinatorial nature of the problem.  The specific details of the SA implementation (neighborhood definition, cost function, cooling schedule) also contribute to the novelty. The exploration of the fairness-utility tradeoff using a parameter is also a welcome addition.

* **Significance:** The problem addressed is of significant importance.  Bias in LLMs is a recognized and pressing issue with potential negative societal impacts. A post-processing technique like attention head pruning offers a more feasible approach than retraining or dataset modification, especially for users who want to leverage pre-trained models. Showing a reduction in other biases by addressing gender bias significantly bolsters the impact of the research.

* **Strengths:**
    * **Effective Solution:** Demonstrates clear improvements in fairness with minimal utility degradation.
    * **Scalability:** The surrogate model approach drastically reduces computational cost, making the technique more practical for large LLMs.
    * **Surrogate Model Accuracy:** High accuracy of the surrogate models, as reflected in the low MSE, is crucial for the success of the overall approach.
    * **Clear Problem Definition:** The paper defines the fairness-aware attention head pruning problem formally, making the approach easily understandable.
    * **Rigorous Evaluation:** The paper presents thorough experimental results with comparisons to the state-of-the-art and ablation studies.

* **Weaknesses:**
    * **Limited Bias Metric:** While HolisticBias is a comprehensive metric, the reliance primarily on gender bias might limit the generalizability of the findings. Further experiments across more metrics or datasets would strengthen this aspect.
    * **Threats to Validity:** While the paper mentions threats to validity it could elaborate further. Specifically, the hyperparameter tuning done for DNNs which are used as surrogate may impact the experiments. Additionally, the authors mentioned outliers were observed in some experiments for certain cases. More justification on why is important since the presence of outliers and the tuning of parameters for surrogate network could raise questions regarding the validity of the experiment.
    * **Limited Exploration of Other Pruning Strategies:** While the paper compares to general pruning strategies and FASP, it could analyze more fairness-aware pruning strategies.
    * **Surrogate Model Training Data Collection Overhead:** While surrogate model speeds up the search drastically, it could be more emphasized that creating the training dataset may also be time-consuming. Also the amount of time mentioned for creating the samples and training a DNN (1900GPU hours) across multiple GPUS could be emphasized.

* **Impact:** The paper has the potential to influence future research in fairness-aware LLMs. The concept of using surrogate models to approximate the behavior of large models for search-based optimization has broad applicability to other areas. It also offers an avenue for practitioners to address bias in deployed models without significant retraining efforts.

**Score: 8**

**Rationale:** The paper presents a novel and well-executed approach to an important problem. The core idea of using surrogate models to make fairness-aware pruning tractable is clever and has the potential to be applied in other contexts. The experimental results support the effectiveness of the approach, with clear improvements over the state-of-the-art. While there are some limitations regarding the breadth of bias metrics and data set, these are relatively minor compared to the overall contribution. The paper's scalability and effectiveness make it a significant contribution to the field, meriting a high score.

- **Score**: 8/10

### **[Enhancing Zero-Shot Image Recognition in Vision-Language Models through Human-like Concept Guidance](http://arxiv.org/abs/2503.15886v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper proposes a novel framework called Concept-guided Human-like Bayesian Reasoning (CHBR) to improve zero-shot image recognition in vision-language models (VLMs). CHBR is inspired by how humans recognize images by decomposing them into concepts. The framework models concepts as latent variables and formulates image classification as summing over potential concepts weighted by a prior distribution and a likelihood function. To handle the infinite concept space, an importance sampling algorithm is used to iteratively prompt large language models (LLMs) to generate discriminative concepts.  Three heuristic approaches are then used to refine the combination of concepts based on the test image: Average Likelihood, Confidence Likelihood, and Test Time Augmentation (TTA) Likelihood. The authors evaluate CHBR on fifteen datasets, demonstrating consistent improvements over existing zero-shot methods.

**Critical Evaluation:**

* **Novelty:**  The core idea of incorporating human-like concept reasoning into VLMs for zero-shot recognition is novel. The paper doesn't just apply VLMs; it explicitly tries to mimic a cognitive process. The importance sampling algorithm driven by LLMs for concept generation is also a significant contribution.  While using LLMs to generate descriptions or prompts is not entirely new, the focus on *discriminative* concepts, guided by inter-class differences, is a strong differentiator. The combination of prior concept generation *and* test-time likelihood refinement is another unique aspect.

* **Significance:** Zero-shot learning is a critical area for real-world deployment of image recognition systems.  VLMs have shown promise, but often struggle with prompt engineering and adaptation to target classes. CHBR directly addresses these limitations by making the models more flexible and adaptable. The consistent performance gains across a diverse set of datasets suggests that CHBR is a robust approach with practical implications.  The ablation studies provide insights into the contribution of different components.

* **Strengths:**
    * Strong conceptual grounding in human cognition.
    *  A well-defined probabilistic framework (Bayesian Reasoning).
    * Effective importance sampling algorithm leveraging LLMs for discriminative concept generation.
    * Test-time refinement techniques that allow for adaptation to specific images.
    * Extensive experimental validation across fifteen datasets, including fine-grained classification and domain shift scenarios.
    * A careful analysis of the contribution of different components through ablation studies.
    * The code is promised to be publicly available.

* **Weaknesses:**
    * **Reliance on LLMs:** The framework heavily relies on the performance and biases of the LLM (GPT-4 mini in this case).  The quality of generated concepts is crucial, and future research should explore more robust or less LLM-dependent methods.
    * **Computational Cost:** While the authors discuss efforts to reduce computational cost, the concept generation process is still time-consuming, especially when compared to a simple CLIP inference.  Scaling to even larger datasets or more complex models could be a challenge. The inference time of TTA Likelihood is also a weakness.
    * **Limited Theoretical Justification for Heuristics:** The Average Likelihood, Confidence Likelihood, and TTA Likelihood are presented as heuristics, and while they work well in practice, a deeper theoretical understanding of why these techniques are effective would strengthen the paper.
    * **Limited Exploration of Prompt Engineering for Concept Generation:** While the paper focuses on generating *discriminative* concepts, it could further explore different prompting strategies for the LLMs to improve the quality and diversity of generated concepts.  The prompts used for prompting the LLMs (shown in the appendix) appear fairly basic; more sophisticated prompting strategies could yield better results.
   * **Limited comparison to more recent prompt engineering strategies:** The baselines are strong, but a comparison to more recent, sophisticated prompt engineering strategies (besides the basic CLIP+E) would be beneficial.
   * **Lack of Error Analysis:** A deeper analysis of the types of errors that CHBR still makes (failure cases) and the reasons for those failures would provide valuable insights for future research.

* **Potential Influence:**  CHBR has the potential to influence the field of zero-shot image recognition by providing a more robust and adaptive approach. It could inspire further research into incorporating cognitive principles into VLMs and using LLMs for concept generation in a more targeted way. The framework also has the potential to be extended to other vision-language tasks, such as image captioning or visual question answering.

**Justification for Score:**

Overall, the paper presents a strong contribution to the field. The core idea is novel and grounded in cognitive principles, the framework is well-defined, and the experimental results are convincing. While the reliance on LLMs and computational cost are valid concerns, the authors acknowledge these limitations and suggest potential directions for future research.  The limitations listed don't invalidate the work but provide clear directions for improvement.

Score: 8

Rationale: The paper demonstrates significant novelty in integrating human-like concept reasoning for zero-shot learning within VLM settings. It provides empirical evidence and a structured methodology that advances the existing state-of-the-art by enabling adaptability and discriminative concept generation. However, it is not a perfect score due to reliance on LLMs and computational expense. Furthermore, a stronger theoretical underpinning for heuristics and deeper failure case analysis would strengthen the paper's significance.

- **Score**: 8/10

### **[Jasmine: Harnessing Diffusion Prior for Self-supervised Depth Estimation](http://arxiv.org/abs/2503.15905v1)**
- **Summary**: Here's a summary and critical evaluation of the "Jasmine: Harnessing Diffusion Prior for Self-Supervised Depth Estimation" paper:

**Summary:**

The paper introduces Jasmine, a novel self-supervised monocular depth estimation (SSMDE) framework that leverages the powerful visual priors of Stable Diffusion (SD).  It addresses the challenges of integrating SD into self-supervised learning, where reprojection losses can corrupt SD's latent space due to inherent noise and artifacts. Jasmine constructs a "hybrid image reconstruction" (HIR) surrogate task, alternating between reconstructing real/synthetic images and predicting depth maps, preserving SD's detail priors while mitigating the reprojection-induced degradation. The paper also proposes a Scale-Shift GRU (SSG) to bridge the distribution gap between the scale-invariant (SI) depth outputs of SSMDE and the scale-and-shift-invariant (SSI) nature of SD's latent space.  The authors show strong results on the KITTI benchmark and impressive zero-shot generalization.

**Critical Evaluation:**

*   **Novelty:**  The primary novelty lies in successfully integrating the Stable Diffusion model into a *self-supervised* monocular depth estimation framework. Existing SD-based depth estimation methods were supervised, requiring high-precision ground truth depth for finetuning. Jasmine cleverly circumvents this need with the HIR surrogate task and the SSG, representing a significant contribution. The HIR task is a solid solution for an important issue. The SSG module addresses the scale issue and improves the model's performance.

*   **Significance:**  The paper addresses a key limitation in depth estimation: the reliance on supervised learning with costly and sometimes unreliable ground truth data. By harnessing the pre-trained priors of SD in a self-supervised manner, Jasmine opens up new avenues for improving the accuracy, robustness, and generalization ability of depth estimation models. The improved zero-shot generalization is particularly significant, reducing the domain adaptation problem.

*   **Strengths:**
    *   **Well-defined Problem and Solution:** The paper clearly identifies the challenges in adapting SD to SSMDE (latent space corruption, scale misalignment) and proposes well-reasoned solutions.
    *   **Strong Empirical Results:** The experimental section is comprehensive, demonstrating state-of-the-art performance on KITTI and showing compelling zero-shot generalization capabilities. The ablation studies provide valuable insights into the contribution of each component.
    *   **Clear Writing:** The paper is well-written and relatively easy to follow, given the complexity of the topic. The figures and diagrams are helpful for understanding the framework.
    *   **Addresses a Gap:** It fills a gap in the literature by exploring self-supervised SD-based depth estimation, a relatively unexplored area.

*   **Weaknesses:**
    *   **Complexity:** The framework, while effective, is relatively complex, involving multiple modules and training strategies. The HIR task and the SSG might be difficult to implement.
    *   **Computational Cost:** While the paper mentions reducing training costs compared to full supervised finetuning, using SD (even with single-step denoising) may still be computationally expensive compared to other self-supervised methods. More quantitative analysis of training time/resources compared to non-SD-based methods would strengthen the paper.
    *   **Dependence on SD:** The performance relies heavily on the pre-trained SD model. This makes it vulnerable if the underlying SD model has biases or limitations. The authors could include a more detailed discussion.
    *   **Justification of Hyperparameters:** While the authors specify hyperparameter settings, a deeper justification for specific choices could be beneficial. Some parameters appear quite specific to this architecture and the KITTI dataset.

*   **Potential Influence:**  This paper has the potential to be highly influential. It provides a novel approach to self-supervised depth estimation and a way to leverage the power of diffusion models, which may become a standard approach for future research. The idea of the HIR task and the SSG are also reusable ideas.

**Score:** 8

**Rationale:**

The paper presents a significant advancement by successfully integrating Stable Diffusion into a self-supervised depth estimation framework.  The proposed HIR task and SSG module demonstrate a solid understanding of the challenges involved and provide effective solutions.  The strong empirical results, especially the zero-shot generalization, further strengthen the paper's contributions. The paper has some issues in the computational complexity and the reliance on SD. Therefore, it doesn't warrant a score in the 9-10 range. Overall, the paper's strengths significantly outweigh its weaknesses, making it a substantial contribution to the field of computer vision.

- **Score**: 8/10

### **[BlockDance: Reuse Structurally Similar Spatio-Temporal Features to Accelerate Diffusion Transformers](http://arxiv.org/abs/2503.15927v1)**
- **Summary**: Okay, let's break down this paper and provide a critical evaluation.

**Summary**

The paper introduces BlockDance, a training-free approach to accelerate Diffusion Transformers (DiTs) for image and video generation. The core idea is to identify and reuse "Structurally Similar Spatio-Temporal (STSS) features" in the later stages of the denoising process.  The authors observe that shallow and middle blocks of the transformer, responsible for coarse-grained structural content, exhibit high similarity between adjacent time steps. BlockDance caches these features and reuses them in subsequent steps to reduce redundant computation.  Furthermore, they propose BlockDance-Ada, a reinforcement learning-based extension that dynamically allocates computational resources on an instance-specific basis.  Experiments across various models (DiT-XL/2, PixArt-a, Open-Sora) and tasks (image generation, video generation) demonstrate speedups of 25-50% while maintaining (and in some cases, improving) generation quality.

**Critical Evaluation**

*   **Novelty:** The idea of reusing features in diffusion models is *not entirely new*. Works like DeepCache also attempt to exploit redundancy. The key novelty lies in:

    *   **Targeted Reuse:** Instead of indiscriminate feature reuse, BlockDance focuses on *structurally similar* features, specifically in the later denoising stages and within specific transformer blocks (shallow and middle). This is justified by empirical observations and analysis of feature similarity across time steps.
    *   **Training-Free Nature:** The method doesn't require retraining the diffusion model, making it readily applicable to existing pre-trained models.
    *   **Adaptive Allocation (BlockDance-Ada):** The reinforcement learning-based approach to dynamically adjust reuse policies based on input content *is* a substantial contribution. It addresses the limitation of a fixed reuse strategy across different types of content.

*   **Significance:**
    *   **Performance Improvement:** The demonstrated speedups are significant, especially considering the training-free nature. This can have a practical impact on reducing the computational cost of using DiTs for various generation tasks.
    *   **Maintaining/Improving Quality:**  Crucially, the method doesn't just speed up inference but also often maintains or even *improves* generation quality. This is a critical factor for the practical adoption of any acceleration technique.
    *   **Generality:** The method is shown to be effective across multiple DiT architectures and tasks, suggesting a degree of generality.
    *   **Analysis and Insights:** The paper provides insightful analysis of feature similarity in DiTs, which helps to justify the design choices and provides a better understanding of the inner workings of these models. The insights regarding structural vs. detailed features are valuable.

*   **Strengths:**

    *   Well-motivated and clearly explained approach.
    *   Thorough empirical evaluation across multiple models and tasks.
    *   Detailed ablation studies to justify design choices.
    *   Qualitative results showing minimal degradation (or improvement) in quality.
    *   The "Ada" component (RL-based dynamic resource allocation) is a strong addition.

*   **Weaknesses:**

    *   While the paper mentions DeepCache, the discussion of prior work on feature reuse in diffusion models could be more comprehensive. A more in-depth comparison to techniques beyond simple feature reuse would strengthen the paper.
    *   The method shows limited benefits in scenarios with very few denoising steps, as pointed out in the limitations section.
    *   While training-free is a strength, a discussion of the potential impact on *fine-tuning* existing diffusion models is not presented.
    *   The reliance on reinforcement learning for BlockDance-Ada adds complexity. While shown to be effective, the implementation details and sensitivity to hyperparameters could be explored further.

*   **Potential Influence:** The paper has the potential to influence the field by providing a practical and efficient way to accelerate DiTs.  The insights into feature redundancy can also inform the design of more efficient diffusion architectures in the future.  The BlockDance-Ada approach is particularly promising as it demonstrates the potential for adaptive computation in these models.

**Justification for Score:**

Overall, this is a strong paper that makes a valuable contribution to the field. The targeted feature reuse strategy and the adaptive resource allocation mechanism are significant improvements over existing approaches. The training-free nature and demonstrated generality across multiple models and tasks are strong positives. While there is room for improvement in terms of a more comprehensive discussion of prior work and a deeper exploration of the RL aspects, the strengths of the paper outweigh its weaknesses. The paper provides both a practical solution to accelerate DiTs and valuable insights into the nature of redundancy within these models.
The analysis of shallow and middle blocks being key to structural features that are stable makes it a great contribution.

**Score: 8**

- **Score**: 8/10

### **[Advancing Mobile GUI Agents: A Verifier-Driven Approach to Practical Deployment](http://arxiv.org/abs/2503.15937v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Advancing Mobile GUI Agents: A Verifier-Driven Approach to Practical Deployment":

**Summary:**

The paper introduces V-DROID, a novel mobile GUI task automation agent. Unlike prior approaches that directly use Large Language Models (LLMs) as action generators, V-DROID employs LLMs as *verifiers*.  The agent first extracts a set of candidate actions from the GUI and then uses an LLM to evaluate each action before making a final decision. The paper introduces a comprehensive framework consisting of: (1) a discretized action space construction with a prefilling-only workflow to reduce verification latency; (2) a pair-wise progress preference (P³) training method to enhance the verifier's decision-making; and (3) a scalable human-agent joint annotation scheme for efficient data collection.  The authors demonstrate that V-DROID achieves state-of-the-art task success rates on several mobile task automation benchmarks while also significantly reducing the decision-making latency to near-real-time.

**Critical Evaluation:**

*   **Novelty:** The core idea of using LLMs as *verifiers* instead of generators in a mobile GUI agent setting is a significant departure from previous approaches. This design is well-motivated by the observation that verifying is generally easier than generating, and the limited action space of mobile GUI environments makes exhaustive verification feasible. The P³ training method is novel and designed to specifically train the verifier to make nuanced decisions about actions. The human-agent joint annotation scheme is also valuable for scaling data collection. The paper brings together several novel components to create a robust system.

*   **Significance:**  The primary significance of this work lies in its ability to drastically reduce the latency of mobile GUI agents while simultaneously improving their task success rate. Previous LLM-based agents have been plagued by slow decision-making, making them impractical for real-world deployment. By achieving near-real-time response, V-DROID makes a compelling case for practical applications of these agents. The improved success rates demonstrate the effectiveness of the verifier-driven approach and the P³ training method. The data collection methodology also addresses a critical limitation in the field by providing a scalable solution for acquiring training data.

*   **Strengths:**

    *   **Well-Motivated Approach:** The paper clearly articulates the limitations of existing generator-based approaches and presents a compelling rationale for the verifier-driven architecture.
    *   **Comprehensive Framework:** V-DROID is a complete system, encompassing action space construction, verification, training, and data collection.
    *   **Strong Empirical Results:** The experimental results demonstrate substantial improvements in both task success rate and latency compared to state-of-the-art baselines. The results are presented rigorously and comprehensively.
    *   **Near-Real-Time Performance:** The achieved latency of 0.7 seconds per step is a significant breakthrough, making the agent potentially suitable for real-world applications.
    *   **Scalable Data Collection:** The human-agent joint annotation scheme effectively reduces the need for full human annotation.

*   **Weaknesses:**

    *   **Limited Generalizability of Components**: The performance of V-DROID is heavily reliant on the specific domain.
    *   **Dependency on a Limited Action Space:** While the authors claim that the general concept could be extended to other domains, the performance heavily relies on the fact that it can enumerate all actions. In other domains like self-driving cars, this is not possible.
    *   **Working Memory Update Latency:** The paper admits that the working memory update still presents a bottleneck, limiting the overall step-wise latency. Further optimization in this area is needed.
    *   **Security and Privacy**: There is an insufficient discussion on the security and privacy considerations, a crucial aspect in real-world deployment.
    *   **Implementation Details**: Certain implementation details, such as the specific prompt templates and hyperparameter choices, could be better elaborated.

*   **Potential Influence:** This work has the potential to significantly influence the development of future mobile GUI agents. The verifier-driven architecture could become a standard approach, particularly in resource-constrained environments. The P³ training method and human-agent joint annotation scheme could also be adopted by other researchers in the field. By demonstrating the feasibility of near-real-time performance, V-DROID could spur the development of new and innovative mobile automation applications.

**Score:** 8

**Rationale:**

The paper presents a genuinely novel approach to mobile GUI agent design, addressing a key limitation of prior work: latency. The results are compelling, demonstrating significant improvements over state-of-the-art baselines in both success rate and speed. The comprehensive framework and scalable data collection method are also valuable contributions. While there are some weaknesses, particularly the working memory latency and the limited discussion on security and privacy, the overall impact and potential influence of this work are significant. The near-real-time performance is a game-changer. A score of 8 reflects the paper's strong contributions, novelty, and potential to significantly advance the field, while acknowledging areas where further research and refinement are needed.

- **Score**: 8/10

### **[A Survey on fMRI-based Brain Decoding for Reconstructing Multimodal Stimuli](http://arxiv.org/abs/2503.15978v1)**
- **Summary**: Here's a summary and evaluation of the provided paper:

**Summary:**

This paper presents a survey of fMRI-based brain decoding techniques, specifically focusing on reconstructing multimodal stimuli from passively elicited brain signals. It systematically reviews recent advancements in this area, categorizing methods based on their model structures (end-to-end, large-scale pre-trained, encoder-alignment, and hybrid models). The survey includes a detailed summary of relevant datasets, activated brain regions (ROIs), and a comparative qualitative and quantitative analysis of various models. Finally, it discusses challenges and proposes future research directions in both algorithmic and practical application aspects of fMRI-based brain decoding.  The survey also provides links to relevant code and data.

**Critical Evaluation:**

**Novelty:** The novelty of this survey lies primarily in its focused scope on fMRI-based brain decoding *specifically for reconstructing multimodal stimuli*.  While other surveys exist in the general area of brain decoding, this paper carves out a specific niche and provides a detailed and up-to-date overview of the progress within that area. The comparative analysis of different model architectures, including recent advancements like large language model (LLM)-centric approaches and hybrid models, is a valuable contribution.  The comprehensive inclusion of relevant datasets and their characteristics also adds to its usefulness. The categorization of models also helps to contextualize the various modeling approaches.

**Significance:**  The paper is significant because fMRI-based brain decoding for multimodal stimuli reconstruction has immense potential in several fields:

*   **Neuroscience:**  It deepens our understanding of how the brain represents and processes information from different senses.
*   **Artificial Intelligence:**  It provides insights into building more human-like AI systems.
*   **Brain-Computer Interfaces (BCIs):** It could lead to more sophisticated BCIs for communication and control.
*   **Clinical Applications:**  It can potentially aid in diagnosing and treating neurological disorders.

The survey highlights the current state-of-the-art techniques, discusses their limitations (e.g., the impact of low temporal resolution and noise in fMRI data), and proposes future research directions, which are valuable for guiding future research efforts. The inclusion of both algorithmic challenges and practical application considerations makes it particularly relevant.

**Strengths:**

*   **Focused Scope:** The survey's clear focus on reconstructing multimodal stimuli makes it more manageable and informative than a general survey on brain decoding.
*   **Up-to-Date Information:** It includes recent advancements in the field, particularly the integration of large-scale pre-trained models and diffusion models.
*   **Comprehensive Coverage:** The inclusion of datasets, brain regions, model architectures, and evaluation metrics provides a complete picture of the research area.
*   **Comparative Analysis:** The qualitative and quantitative analysis of different models helps readers understand their strengths and weaknesses.
*   **Future Directions:** The discussion of challenges and future research directions provides valuable guidance for researchers in the field.
*   **Practical Relevance:** It addresses practical considerations and ethical implications that are critical for the translation of these technologies to clinical and real-world applications.

**Weaknesses:**

*   **Limited Depth in Certain Areas:** While the survey provides a broad overview, certain sub-areas (e.g., specific implementation details of certain complex models, in-depth analysis of individual datasets beyond the high-level summary) are not explored in extreme detail.  This is, to some extent, unavoidable in a survey of this scope.
*   **Potential for Rapid Outdating:**  The field of AI and deep learning is rapidly evolving, so some of the specific models and techniques discussed may become outdated relatively quickly. However, the general principles and trends identified are likely to remain relevant for a longer period.
*   **Emphasis on Image Reconstruction:** The survey's primary focus is on image reconstruction from fMRI signals. While this is a significant area, the survey may not fully capture the advancements in other areas of fMRI-based decoding (e.g., decoding cognitive states, predicting behavior).
*   The quantitative comparison of different models is limited due to the differences in experimental setups and the lack of direct performance comparisons under a unified evaluation framework.

**Justification for Score:**

Considering the above analysis, this survey represents a significant and valuable contribution to the field. It provides a timely and comprehensive overview of the current state-of-the-art in fMRI-based brain decoding for reconstructing multimodal stimuli. It offers valuable insights into the challenges and future directions of this rapidly evolving area.  While it has some limitations in terms of depth in specific areas and the potential for rapid outdating, its focused scope, comprehensive coverage, and practical relevance make it a valuable resource for researchers in neuroscience, AI, and related fields.

Score: 8

- **Score**: 8/10

### **[The Lighthouse of Language: Enhancing LLM Agents via Critique-Guided Improvement](http://arxiv.org/abs/2503.16024v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "The Lighthouse of Language: Enhancing LLM Agents via Critique-Guided Improvement":

**Summary:**

The paper introduces Critique-Guided Improvement (CGI), a novel two-player framework designed to enhance the performance of LLM-based agents in interactive environments. CGI involves an "actor" model that explores an environment and a "critic" model that provides detailed natural language feedback on the actor's actions. The critic is trained to generate fine-grained assessments and actionable revisions. The actor then learns to utilize these critiques through iterative supervised fine-tuning.  The authors demonstrate through experiments in three interactive environments that CGI outperforms existing baselines, including those utilizing numerical reward signals and self-critique. A key finding is that even a small critic model can surpass the performance of GPT-4 in terms of feedback quality, leading to state-of-the-art agent performance.

**Critical Evaluation:**

**Novelty:**

The novelty of the paper lies in the specific combination of elements and the results achieved. While the idea of using a critic to guide an actor is not entirely new (especially within reinforcement learning), the following contribute to its novelty:

*   **Emphasis on detailed natural language critiques:** The paper stresses the importance of rich, actionable feedback generated by the critic, going beyond simple numerical rewards or binary verifications. This leverages the generative capabilities of LLMs in a targeted way.
*   **Iterative Action Refinement via Supervised Fine-Tuning:** The iterative SFT process allows the agent to adapt to the critique feedback in a way that avoids policy misalignment. The process ensures that the agent's policies adjust and evolve as new feedback are provided through iterations.
*   **Demonstrated Performance with a Small Critic:** The fact that a relatively small (8B) critic model can outperform a much larger general-purpose LLM (GPT-4) in providing useful feedback is a significant finding.
*   **Two-player setting:** the framework follows a two-player setting, where the actor generates multiple candidate actions, and the critic provides feedback to refine them which then enhances the quality and utilization of these models.

A potential limitation on novelty is that iterative training frameworks are prevalent. The iterative fine tuning can be seen as an extension to existing methodologies. However, the paper makes a significant contribution by demonstrating that iterative action refinement with critic guidance is a promising approach for improving the performance of LLM-based agents in complex, interactive environments.

**Significance:**

The paper addresses a crucial challenge in the development of LLM-based agents: how to effectively provide and utilize feedback for improved decision-making. The results have the following important implications:

*   **Shift from Numerical to Language Feedback:** It provides strong evidence that natural language feedback can be superior to numerical rewards in certain scenarios, especially when dealing with complex reasoning and planning tasks. Language is inherently more instructive in many cases and may serve as a path towards interpretability and actionability for LLM agents.
*   **Practical Improvement in Agent Performance:** The reported state-of-the-art results in diverse interactive environments clearly demonstrate the practical benefits of the CGI framework. CGI's approach provides significant improvements in the model's reasoning capabilities.
*   **Efficiency of Specialized Models:** The success of the small critic model highlights the potential for developing specialized models tailored to specific tasks, rather than relying solely on large, general-purpose LLMs.

**Weaknesses:**

*   **Dependence on Expert Critiques for Training:** The critic model is trained on expert critiques, which may be costly and time-consuming to obtain. Exploring methods for automatically generating high-quality critiques would be beneficial.
*   **Computational Cost:** While the paper highlights the efficiency, the overall process could still be expensive, particularly the exploration phase.
*   **Limited Environments:** Although the experiments cover three diverse environments, evaluating CGI in more complex and realistic settings would further validate its effectiveness.
*   **Lack of ablations on critic structure**. The paper only includes a basic analysis of different types of data used to train the critic. However, a more comprehensive breakdown of the critic's composition, for example assessing the impact of the discriminator versus revision component, could be insightful.

**Potential Influence:**

This paper is likely to influence the field of LLM-based agents by:

*   Encouraging the development of more sophisticated feedback mechanisms for agent training.
*   Motivating further research into the use of smaller, specialized models for specific agentic tasks.
*   Providing a practical framework for improving the performance of LLM agents in interactive environments.

Overall, the paper presents a novel and significant contribution to the field, offering a promising approach for enhancing the performance of LLM-based agents.

**Score: 8**

**Rationale:** The paper presents a novel framework with significant empirical results. While the core idea is not entirely new, the specific implementation with iterative SFT and emphasis on small specialized models is a significant contribution. The results are strong, and the potential influence on the field is considerable. However, there are potential weaknesses, such as the dependency on expert data, and computational cost that prevent it from achieving a higher score. Also, iterative refinement methods are also previously researched which reduced the novelty somewhat.

- **Score**: 8/10

### **[Meta-Learning Neural Mechanisms rather than Bayesian Priors](http://arxiv.org/abs/2503.16048v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper investigates what meta-learning imbues in neural networks when applied to formal languages. Contrary to the prevailing belief that meta-training distills simplicity-based priors, the authors provide evidence suggesting that meta-training imprints neural mechanisms (like counters) into the model, acting as cognitive primitives for downstream tasks.  They show that meta-training on a single, carefully chosen formal language can be as effective as meta-training on thousands of diverse formal languages, if that language incentivizes the learning of beneficial mechanisms. They further demonstrate that if a model is unable to instantiate a necessary mechanism (e.g., a GRU not able to instantiate a counter), then meta-training doesn't help. The study highlights the importance of mechanistic complexity over information complexity and provides insights into efficiently designing meta-learning paradigms.

**Critical Evaluation:**

* **Novelty:** The paper presents a strong challenge to the dominant "simplicity bias" interpretation of meta-learning in the context of language acquisition.  It directly tests the assumptions made in previous work like McCoy and Griffiths (2023) and offers a more mechanistic explanation. The shift from a statistical, prior-based view to a mechanism-focused one is a substantial contribution.  The experimental results, particularly the effectiveness of meta-training on a *single*, strategically chosen language, are surprising and counter-intuitive. The use of Chomsky hierarchy as a measure for mechanistic complexity and its link to suitable neural mechanisms provides a novel perspective.

* **Significance:** The implications of this work are significant for both AI and cognitive science.

   * **AI:** It provides practical guidance for more efficient meta-learning. Instead of creating large, diverse datasets designed around statistical notions of simplicity, the authors suggest carefully designing meta-training datasets around the *capabilities* of the underlying architecture and the mechanisms it's capable of learning.  This could lead to more targeted and resource-efficient meta-learning.

   * **Cognitive Science:** The paper provides a more nuanced understanding of how neural networks can approximate human-like learning. The emphasis on neural mechanisms echoes more symbolic theories of cognition, suggesting a potential bridge between connectionist and symbolic models.

* **Strengths:**

    * **Clear Research Question and Hypothesis:**  The paper clearly articulates the conflicting views (simplicity-bias vs. mechanistic) and formulates testable hypotheses.
    * **Well-Designed Experiments:** The experiments are carefully designed to distinguish between the two competing hypotheses. The use of different formal languages and the manipulation of neural architecture (LSTM vs. GRU) provide compelling evidence.
    * **Strong Results:** The results are statistically significant and consistently support the mechanistic view while undermining the simplicity-bias view.
    * **Rigorous Methodology:** The use of appropriate baselines (unmetatrained models), well-defined metrics (continuation-based F1 score), and clear description of the training procedures all contribute to the rigor of the study.

* **Weaknesses:**

    * **Limited Scope:** While the focus on formal languages allows for controlled experiments, it limits the generalizability to more complex domains like natural language. Although the mechanistic principles might be applicable, the specific mechanisms learned (e.g., counters) might not be as relevant.
    * **Characterization of Complexity:** The use of Chomsky hierarchy, while intuitive, has its limitations. The authors acknowledge this, but a more refined notion of "mechanistic complexity" would strengthen the argument.
    * **Neural Counter Discovery:** The evidence of *neural* counters is more suggestive than conclusive.  While they argue that GRUs are unable to instantiate suitable neural mechanisms, a visualization or analysis of the hidden states of the LSTM to explicitly locate a neural counter would be very strong.

* **Potential Influence:** This paper has the potential to shift the thinking around meta-learning, particularly in the context of cognitive modeling.  It encourages researchers to think more carefully about the *capabilities* of their architectures and to design meta-training tasks that exploit those capabilities. This also opens doors to research linking neural networks and more traditional cognitive architectures.

* **Score Rationale:** The paper presents a novel perspective, offers strong empirical evidence, and has significant implications for both AI and cognitive science. The limitations are primarily related to the scope and the need for a more refined measure of complexity, but these do not detract significantly from the overall contribution.

**Score: 8**

- **Score**: 8/10

### **[CodeReviewQA: The Code Review Comprehension Assessment for Large Language Models](http://arxiv.org/abs/2503.16167v1)**
- **Summary**: Here's a summary and critical evaluation of the paper, "CodeReviewQA: The Code Review Comprehension Assessment for Large Language Models":

**Summary:**

The paper introduces CodeReviewQA, a new benchmark designed to evaluate the code review comprehension abilities of Large Language Models (LLMs). Unlike existing benchmarks that focus on automated code refinement (ACR) using sequence-to-sequence generation and simple text matching metrics, CodeReviewQA decomposes the ACR task into three essential reasoning steps: change type recognition (CTR), change localization (CL), and solution identification (SI). Each step is reformulated as a multiple-choice question answering (MCQA) problem, mitigating data contamination risks and enabling fine-grained assessment of model capabilities. The authors evaluate 72 LLMs on a manually curated dataset of 900 code review examples across nine programming languages. The results reveal specific model weaknesses in code review comprehension, decoupled from their generative ACR results.

**Critical Evaluation:**

*   **Novelty:** The paper presents a novel approach to evaluate LLMs on code review understanding. Decomposing the ACR task into granular reasoning steps (CTR, CL, SI) is a significant improvement over existing benchmarks that largely focus on end-to-end generation and surface-level metrics like exact match. The use of MCQA probes with synthetic answers is a clever way to mitigate data contamination risks, allowing the reuse of existing code review data. While prior work has explored ACR, the focus on *comprehension* with reasoning probes is a novel contribution.

*   **Significance:** Code review is a crucial and complex software engineering task, involving both technical expertise and natural language comprehension (often implied or ambiguous). Successfully automating it would greatly improve software development workflows. By highlighting specific areas (CTR, CL, SI) where LLMs struggle, this benchmark allows researchers to develop more targeted solutions for addressing the challenges of code review comprehension. The size and diversity (72 models, 9 languages, 199 repositories) of the evaluation are a strength.

*   **Strengths:**

    *   The decomposition of the ACR task is well-motivated and provides valuable insights.
    *   The use of MCQA with synthetic answers effectively addresses data contamination concerns.
    *   The manual curation process for the dataset ensures high quality and reduces noise.
    *   The experiments are comprehensive and provide a valuable overview of the capabilities (and limitations) of current LLMs in code review comprehension.

*   **Weaknesses:**

    *   The multiple-choice format, while mitigating data contamination, might limit the assessment of LLMs' ability to generate *novel* solutions or handle open-ended code revision scenarios. It shifts the focus from generation to recognition.
    *   While the paper discusses the variation of distractor difficulty, the methodology could be elaborated further. Specific strategies for creating easy/hard distractors should be more clearly described.
    *   The authors use a surrogate LLM to generate distractor code, and this introduces a dependence on the performance of that specific model.
    *   The performance bottleneck seems to be change localization and some specific LLMs get good performance overall but not in CL. It isn't clear how much this localization issue affects other aspects.

*   **Potential Influence:** The paper has the potential to significantly influence future research in automated code review and software engineering. By providing a more granular and reliable evaluation framework, CodeReviewQA will encourage the development of more targeted solutions for improving LLMs' code review comprehension abilities. The framework can also be extended to include additional reasoning steps or different types of probes.

*   **Score Rationale:** While the paper presents a well-designed benchmark and provides valuable insights, it's important to acknowledge the limitations of MCQA and the dependency on a surrogate LLM for generating distractors. Moreover, while the paper is very detailed, many implementation details are deferred to the appendix, which makes comprehension of the whole system harder. Given these considerations, the following score is assigned:

Score: 8

- **Score**: 8/10

### **[Temporal Score Analysis for Understanding and Correcting Diffusion Artifacts](http://arxiv.org/abs/2503.16218v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper "Temporal Score Analysis for Understanding and Correcting Diffusion Artifacts" addresses the persistent problem of visual artifacts in images generated by diffusion models. The authors propose a novel approach called ASCED (Abnormal Score Correction for Enhancing Diffusion) that identifies and mitigates artifacts by analyzing the temporal dynamics of score functions during the diffusion process.  Instead of relying on post-hoc methods or supervised artifact detectors, ASCED operates within the diffusion process, monitoring the score dynamics to detect anomalies and applying a trajectory-aware correction mechanism to "re-couple" problematic regions with their surroundings.  The paper argues that artifacts emerge during a "Mutation" phase of diffusion and that existing methods fail to capture the crucial temporal dynamics that reveal these anomalies. The method is unsupervised and shown to be effective across various datasets, matching or surpassing supervised methods without additional training.

**Critical Evaluation:**

**Novelty:** The paper's primary novelty lies in its temporal analysis of the diffusion process and the insight that artifacts manifest through distinct patterns in score dynamics.  This is a significant shift from existing methods that focus on spatial uncertainty or supervised classification of the final output.  The introduction of the ASCED framework based on this temporal analysis, particularly the trajectory-aware correction mechanism, is also a novel contribution.

**Significance:** The paper is significant for several reasons:

*   **Improved understanding:** It provides a deeper understanding of the internal workings of diffusion models, specifically regarding artifact formation. This is valuable for researchers working on improving these models.

*   **Unsupervised artifact removal:** The unsupervised nature of ASCED makes it more generalizable and adaptable to new datasets and domains compared to supervised methods, which require labeled data.

*   **In-process correction:** Correcting artifacts during the generative process, rather than after the fact, is a more elegant and potentially more effective approach. It allows for better integration with the diffusion process.

*   **Computational efficiency:** The method is computationally efficient, which makes it practical for real-world applications.

**Strengths:**

*   **Strong theoretical foundation:** The paper's analysis of score dynamics and the identification of the three phases (Profiling, Mutation, Refinement) is well-motivated and supported by experimental evidence.

*   **Effective method:** ASCED demonstrably reduces artifacts across a range of datasets, and it achieves competitive performance against existing methods.

*   **Unsupervised approach:** The unsupervised nature of the method is a significant advantage, particularly for domains where labeled data is scarce.

*   **Clear and well-organized presentation:** The paper is generally well-written and the method is clearly explained. The figures and tables are informative.

**Weaknesses:**

*   **Limited failure case analysis:** While the paper acknowledges failure cases, the analysis is somewhat limited. A more in-depth discussion of the types of artifacts that ASCED struggles with and the reasons for these failures would strengthen the paper. It's clear that in very low contrast situations it'll struggle to discern artifacts from background noise.

*   **Parameter sensitivity:** While the method is unsupervised, it still likely has parameters (e.g., for determining abnormality thresholds). The paper could benefit from a more detailed analysis of how these parameters affect performance and how they can be tuned for different datasets.

*   **Limited scope of correction methods:** The paper primarily compares against post-hoc correction and baselines derived from the paper itself. It should consider more diverse correction methods beyond what the paper provides.

**Potential Influence:**

The paper has the potential to influence the field in several ways:

*   **Encouraging temporal analysis:** It could inspire other researchers to investigate the temporal dynamics of generative models more broadly.

*   **Developing more robust and generalizable artifact removal techniques:** The success of ASCED could lead to the development of new unsupervised artifact removal methods that are less reliant on domain-specific training data.

*   **Improving the theoretical understanding of diffusion models:** The paper's insights into artifact formation could contribute to a deeper theoretical understanding of how diffusion models work and how they can be improved.

**Justification for Score:**

While the paper is well-executed and presents a novel and significant contribution to the field, some weaknesses prevent it from receiving the highest possible score. Its core contribution - *temporal based artifact detection and on-the-fly correction* - is novel and potentially far-reaching, making it a valuable contribution. However, the above weaknesses, specifically in failure case analysis and better comparison of competing methods, prevent it from being a 9 or 10 score.

Score: 8

- **Score**: 8/10

### **[Chain of Functions: A Programmatic Pipeline for Fine-Grained Chart Reasoning Data](http://arxiv.org/abs/2503.16260v1)**
- **Summary**: Here's a summary and critical evaluation of the "Chain of Functions: A Programmatic Pipeline for Fine-Grained Chart Reasoning Data" paper:

**Summary:**

The paper addresses the scarcity of high-quality, diverse, and explainable data for training multimodal large language models (MLLMs) for chart reasoning. It proposes a novel data generation pipeline called "Chain of Functions" (CoF). CoF uses a programmatically defined set of atomic functions to explore reasoning paths within charts. These function chains are then translated into natural language rationales and questions using a smaller, more manageable LLM. The resulting dataset, ChartCoF, provides fine-grained annotations and explanations for complex chart-based questions. The authors demonstrate that fine-tuning MLLMs with ChartCoF leads to improved performance on existing chart reasoning benchmarks and enables a more detailed analysis of MLLM reasoning abilities.

**Critical Evaluation:**

*   **Novelty:** The core idea of using a programmatic pipeline based on atomic functions is a significant departure from existing data generation approaches that rely on direct LLM prompting. This is a notable strength, allowing for more controlled data creation and reduced reliance on enormous, often proprietary, LLMs.  The breakdown of the chart reasoning process into a chain of executable functions, provides a novel approach to data generation.

*   **Significance:** The lack of high-quality and diverse data is a known bottleneck in training effective MLLMs for complex reasoning tasks. ChartCoF appears to address this by generating data with strong explainability (through the function chains), precision (due to the programmatic nature), and diversity (through function chain enumeration). The experimental results, particularly the state-of-the-art performance after fine-tuning with ChartCoF and the insights gained into MLLM strengths and weaknesses across different question types, underscore the dataset's value to the community. The fine-grained analysis on different question taxonomies offers a better insight to what challenges the MLLMs have.

*   **Strengths:**
    *   **Controlled Data Generation:** The function-based approach avoids the limitations of direct LLM prompting, leading to higher precision and reduced hallucinations.
    *   **Explainability:** The function chains provide built-in rationales, facilitating detailed performance evaluation and error analysis.
    *   **Practicality:** Uses a relatively moderate, open-sourced LLM for the language translation step, making the pipeline more accessible.
    *   **Improved Performance:**  Experiments confirm that finetuning with ChartCoF enhances reasoning capabilities and achieves competitive results on standard benchmarks.

*   **Weaknesses:**
    *   **JSON Representation Limitations:** While the authors address this, the reliance on JSON to represent charts might limit the complexity and realism of the generated data compared to directly extracting from diverse web sources.
    *   **Function Definition Bias:**  The diversity of reasoning paths is ultimately limited by the choice of atomic functions.  While the paper details the function selection process, there's a risk of overlooking potentially important reasoning strategies not captured by the defined functions.
    *   **LLM Reliance for Language Translation:** While using a smaller LLM is a positive aspect, the quality of the rationales and questions still depends on its language generation capabilities. Potential biases or limitations in this LLM could propagate into the generated data.
    *   **Complexity:** While the atomic functions help control the diversity, it is difficult to account for all different scenarios in terms of question types which makes it hard to scale.

*   **Potential Influence:** The CoF pipeline offers a potentially generalizable paradigm for generating reasoning data for other complex tasks beyond chart understanding. The emphasis on structured reasoning supervision could inspire new approaches to improve the explainability and reliability of MLLMs.

**Justification for Score:**

The paper presents a novel and well-executed data generation approach with significant potential impact. The careful design of the CoF pipeline, the detailed experimental evaluation, and the strong results achieved on existing benchmarks justify a high score. However, the limitations related to JSON chart representation and potential function definition bias prevent it from reaching the highest possible score.

Score: 8

- **Score**: 8/10

### **[SceneMI: Motion In-betweening for Modeling Human-Scene Interactions](http://arxiv.org/abs/2503.16289v1)**
- **Summary**: Here's a summary and critical evaluation of the SceneMI paper:

**Summary:**

The paper introduces SceneMI, a novel framework for scene-aware motion in-betweening, designed to generate realistic and physically plausible human-scene interactions. Unlike existing methods, SceneMI addresses the challenges of controllability and flexibility, especially in real-world scenarios with noisy keyframes and imperfect scene data. The core of SceneMI is a conditional diffusion model that incorporates dual scene descriptors: a global occupancy voxel grid and keyframe-centered Basis Point Set (BPS) features. The framework also leverages the inherent denoising capabilities of diffusion models to handle noisy keyframes during inference. The paper demonstrates SceneMI's effectiveness through experiments on the TRUMANS dataset (used for training), and the GIMO dataset (a real-world dataset) and a video dataset for HSI reconstruction to demonstrate generalization. The results show improved motion quality, reduced artifacts, and enhanced interaction plausibility.

**Critical Evaluation:**

*   **Novelty:** The key novelty of this paper lies in reformulating HSI modeling as a scene-aware motion in-betweening problem and proposing SceneMI as a solution tailored for this specific task. Existing works often focus on generating motions from text or actions but lack the controllability and robustness to noisy data that SceneMI addresses. The dual scene encoding strategy (global voxel grid and local BPS) is also a novel contribution, allowing for a comprehensive representation of the scene context. The application of a diffusion model with specialized denoising procedures to accommodate noisy keyframes is significant. The end-to-end HSI reconstruction pipeline from monocular video is a notable application, although the individual components (image-to-3D, pose estimation) are not novel themselves, but their combination with SceneMI makes a contribution.

*   **Significance:** The significance of this paper is substantial due to its practical implications for character animation, virtual reality, and HSI reconstruction. By enabling controllable and robust motion synthesis in complex scenes, SceneMI offers a valuable tool for animators and researchers. The ability to handle noisy keyframes and generalize to real-world data is particularly important, as it opens up new possibilities for creating realistic and immersive experiences from imperfect motion capture or video data. The improvement in motion quality, reduction of foot skating and jittering, and enhanced interaction plausibility are all important contributions that advance the state of the art in HSI modeling. The HSI pipeline demonstrates application of their method to a complex real-world use case.

*   **Strengths:**
    *   **Problem Formulation:** Reformulating the HSI problem as scene-aware motion in-betweening is a smart and practical approach.
    *   **Technical Design:** The dual scene encoding strategy and the specialized denoising procedures are well-designed and effective.
    *   **Experimental Validation:** The experiments on both synthetic and real-world datasets are thorough and demonstrate the effectiveness of SceneMI. The inclusion of ablation studies and comparisons with state-of-the-art methods further strengthens the validation.
    *   **Practical Applications:** The HSI reconstruction pipeline showcases the practical applicability of SceneMI.

*   **Weaknesses:**
    *   **Reliance on Keyframes:** While the paper addresses noisy keyframes, the method still relies on having accurate keyframe poses to begin with. The approach is not motion completion from partial poses.
    *   **Feature Level Fusion**: The study mentions limitation of feature level fusion as opposed to model level fusion.
    *   **Limited Scene Complexity**: The experiments primarily focus on indoor scenes. The method's performance in more complex, dynamic outdoor environments remains unclear.
    *   **Complexity of Implementation:** Diffusion models can be complex to train and optimize, requiring significant computational resources.
    *   **Lack of Real-time Performance**: No real time performance is mentioned.

*   **Potential Influence:** This paper is likely to have a significant influence on the field of HSI modeling. Its novel approach, practical applications, and strong experimental results will inspire future research in this area. Other researchers can build upon SceneMI's framework to further improve the quality, controllability, and robustness of HSI models. This is particularly relevant with the increasing prevalence of virtual and augmented reality and the growing need for realistic and immersive human-computer interactions.

**Score: 8**

**Rationale:**

SceneMI presents a significant advance in the field of human-scene interaction modeling. The reformulation of HSI as a scene-aware motion in-betweening problem, coupled with the effective dual scene encoding and diffusion-based denoising approach, makes this a novel and valuable contribution. The strong experimental validation and practical applications demonstrated in the paper further solidify its significance. The weaknesses, such as the reliance on keyframes and the implementation complexity, are relatively minor and do not detract significantly from the overall quality of the work. However, this also prevents the method from achieving a higher score.

Overall, the strengths of the paper far outweigh its weaknesses, and SceneMI is likely to have a lasting impact on the field.

- **Score**: 8/10

### **[Unleashing Vecset Diffusion Model for Fast Shape Generation](http://arxiv.org/abs/2503.16302v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Unleashing Vecset Diffusion Model for Fast Shape Generation":

**Summary:**

The paper presents FlashVDM, a framework to accelerate the generation of 3D shapes using Vecset Diffusion Models (VDMs). It addresses the slow generation speed of VDMs, which stems from both diffusion sampling and VAE decoding bottlenecks. FlashVDM accelerates diffusion sampling through a "Progressive Flow Distillation" technique, which enables few-step generation while maintaining quality. It also accelerates VAE decoding using a "lightning vecset decoder" with Adaptive KV Selection, Hierarchical Volume Decoding, and an efficient network design. These techniques exploit the locality of vecset and sparsity of shape surfaces to reduce computational cost. Experiments on Hunyuan3D-2 show significant speedups in both reconstruction and generation while maintaining or improving quality.

**Critical Evaluation:**

**Novelty:** The paper presents several novel components.
*   **Progressive Flow Distillation for VDMs:** Applying and adapting distillation techniques to native 3D diffusion is relatively new. The proposed progressive approach, addressing instability issues specific to VDMs, adds novelty. Existing distillation methods were primarily designed for images.
*   **Lightning Vecset Decoder:** The combined approach of Adaptive KV Selection and Hierarchical Volume Decoding for accelerating the VAE decoding component in VDMs is a significant contribution. Exploiting locality and sparsity in this context appears to be novel. While related concepts like octree decoding exist, the specific application and combination of techniques for VDM decoding are novel.

**Significance:**

*   **Performance Improvement:** The reported speedups are substantial (45x for reconstruction, 32x for generation). Making VDM-based shape generation much faster significantly enhances its practicality and applicability.  The move from tens of seconds to around a second makes it much more suitable for interactive applications.
*   **Improved Efficiency:** Reducing the computational cost of VDM, particularly the VAE decoding stage, makes these models more accessible and energy-efficient. This aligns with the broader trend in machine learning towards more efficient models.
*   **Impact on 3D Generation:**  By making VDM faster and more practical, the paper lowers the barrier to entry for high-quality 3D shape generation, potentially enabling new applications in design, content creation, and other fields.
*   **Solid Evaluation:** The paper provides thorough evaluation, including quantitative metrics, visual comparisons, and user studies. Ablation studies are also included to demonstrate the effectiveness of each component.

**Weaknesses:**

*   **Dependence on Hunyuan3D-2:**  The primary results are based on Hunyuan3D-2.  While it's a strong baseline, demonstrating the generalizability of FlashVDM to other VDM architectures would strengthen the paper.
*   **Distillation Complexities:** While the progressive distillation is novel, distillation techniques can be complex to implement and tune. The paper could provide further insights into hyperparameter sensitivity and practical advice for others applying the method.
*   **Real-world Data Fine Tuning:** The use of adversarial finetuning relying on real 3D data adds the need for carefully balancing the distribution between the pre-trained dataset and the newly finetuned one. If there are biases in the real data, the model will perpetuate this.
*  **Reliance on engineering details:** The significant boost from the techniques (SageAttention2, torch.compile) may overshadow the actual algorithm performance.

**Overall:**

The paper makes a strong contribution to the field of 3D shape generation. The proposed FlashVDM framework addresses a critical bottleneck in VDM-based generation, resulting in substantial performance improvements. The novelty lies in the specific adaptation of distillation and decoding techniques tailored to the unique characteristics of VDMs. The comprehensive evaluation further solidifies the significance of the work.

**Score: 8.0**

**Rationale:**

The score reflects the paper's significant performance improvements and clear novelty. The combination of methods allows VDM to create high-quality shapes rapidly. Despite these strengths, the limited generalizability is the weakness, along with the need to do adversarial finetuning to match the teacher's mesh quality.

- **Score**: 8/10

### **[Lyra: An Efficient and Expressive Subquadratic Architecture for Modeling Biological Sequences](http://arxiv.org/abs/2503.16351v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Lyra: An Efficient and Expressive Subquadratic Architecture for Modeling Biological Sequences":

**Summary:**

The paper introduces Lyra, a novel neural network architecture designed for efficient and expressive modeling of biological sequences. Lyra combines Projected Gated Convolutions (PGCs) for capturing local sequence features with state space models (SSMs) for modeling long-range dependencies and epistasis. The architecture achieves subquadratic scaling with sequence length, making it significantly more computationally efficient than Transformer-based models. The authors demonstrate Lyra's strong performance across a wide range of biological tasks, including protein fitness landscape prediction, RNA structure analysis, and CRISPR guide design.  They highlight the architecture's ability to match or surpass state-of-the-art performance with orders-of-magnitude fewer parameters and faster inference times, democratizing access to advanced sequence modeling.

**Critical Evaluation:**

*   **Novelty:** The novelty of Lyra lies in its specific combination of PGCs and SSMs, and the justification for this combination based on biological principles (epistasis as polynomial interactions).  While PGCs and SSMs are not individually novel, the *integration* of these components and their application to biological sequence modeling with a theoretical grounding in epistasis is new. The authors provide a strong mathematical argument for why SSMs are well-suited to approximating the polynomial terms that describe epistatic interactions. This is a significant step toward developing more interpretable and efficient models for biological sequences.
*   **Significance:** The paper's significance stems from its potential to address the computational bottleneck in biological sequence modeling. Transformer models, while powerful, are computationally expensive and require large datasets. Lyra offers a viable alternative that achieves comparable or superior performance with significantly reduced computational resources. The democratization of access to advanced sequence modeling is a major benefit, enabling researchers with limited computational infrastructure to tackle complex biological problems. The extensive empirical evaluation across diverse tasks adds substantial weight to the claims of broad applicability. The speedups and reduced memory footprint reported are impressive.

*   **Strengths:**
    *   **Strong Theoretical Grounding:** The paper provides a solid mathematical justification for the architecture's design, linking it to biological principles.
    *   **Comprehensive Evaluation:** Lyra is evaluated on a very broad suite of tasks, providing strong evidence of its general applicability.
    *   **Significant Performance Gains:**  The paper demonstrates impressive improvements in computational efficiency and parameter reduction compared to existing methods without sacrificing performance.
    *   **Addresses a Critical Need:**  The paper tackles the growing need for more efficient and scalable models for biological sequence analysis.
    * **Democratization:** Lyra makes state-of-the-art sequence modeling more accessible, removing barriers related to cost and access to high-end computing hardware.

*   **Weaknesses:**
    *   **Interpretability (Limited):** While the authors claim interpretability benefits due to the explicit connection to epistasis, further work is needed to demonstrate that the learned model parameters can be easily translated into biological insights. The interpretability aspect could have been further explored by showcasing examples of how Lyra extracts meaningful epistatic relationships.
    *  **Limited Comparison Against other recent efficient architectures**:  The comparison against Hyena is useful, but the paper could benefit from a more comprehensive comparison against other recent efficient sequence modeling architectures targeting similar compute constraints.
    * **Potential for Overfitting:** Although impressive generalization and evaluation across various benchmarks, there may be a concern regarding the number of tasks assessed against the comparitively small number of parameters.

**Justification for Score:**

Lyra represents a significant advancement in biological sequence modeling. Its combination of efficiency and expressiveness, coupled with its theoretical justification and broad empirical validation, makes it a valuable contribution to the field. While further work is needed to explore the interpretability aspect and perform a full comparison versus other similar efficient models, the paper is important because of its potential to democratize access to SOTA sequence modeling by offering a lightweight, fast and biologically inspired architecture.

**Score: 8**

- **Score**: 8/10

## Other Papers
### **[SemEval-2025 Task 1: AdMIRe -- Advancing Multimodal Idiomaticity Representation](http://arxiv.org/abs/2503.15358v1)**
### **[EfficientLLaVA:Generalizable Auto-Pruning for Large Vision-language Models](http://arxiv.org/abs/2503.15369v1)**
### **[CCDP: Composition of Conditional Diffusion Policies with Guided Sampling](http://arxiv.org/abs/2503.15386v1)**
### **[Improving Adversarial Transferability on Vision Transformers via Forward Propagation Refinement](http://arxiv.org/abs/2503.15404v1)**
### **[Visual Persona: Foundation Model for Full-Body Human Customization](http://arxiv.org/abs/2503.15406v1)**
### **[Visual Position Prompt for MLLM based Visual Grounding](http://arxiv.org/abs/2503.15426v1)**
### **[MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](http://arxiv.org/abs/2503.15451v1)**
### **[Di$\mathtt{[M]}$O: Distilling Masked Diffusion Models into One-step Generator](http://arxiv.org/abs/2503.15457v1)**
### **[From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment](http://arxiv.org/abs/2503.15463v1)**
### **[FP4DiT: Towards Effective Floating Point Quantization for Diffusion Transformers](http://arxiv.org/abs/2503.15465v1)**
### **[Cube: A Roblox View of 3D Intelligence](http://arxiv.org/abs/2503.15475v1)**
### **[CAM-Seg: A Continuous-valued Embedding Approach for Semantic Image Generation](http://arxiv.org/abs/2503.15617v1)**
### **[LLaVA-MORE: A Comparative Study of LLMs and Visual Backbones for Enhanced Visual Instruction Tuning](http://arxiv.org/abs/2503.15621v1)**
### **[R$^2$: A LLM Based Novel-to-Screenplay Generation Framework with Causal Plot Graphs](http://arxiv.org/abs/2503.15655v1)**
### **[Enhancing Pancreatic Cancer Staging with Large Language Models: The Role of Retrieval-Augmented Generation](http://arxiv.org/abs/2503.15664v1)**
### **[CHROME: Clothed Human Reconstruction with Occlusion-Resilience and Multiview-Consistency from a Single Image](http://arxiv.org/abs/2503.15671v1)**
### **[GASP: Unifying Geometric and Semantic Self-Supervised Pre-training for Autonomous Driving](http://arxiv.org/abs/2503.15672v1)**
### **[Multi-focal Conditioned Latent Diffusion for Person Image Synthesis](http://arxiv.org/abs/2503.15686v1)**
### **[Safety Aware Task Planning via Large Language Models in Robotics](http://arxiv.org/abs/2503.15707v1)**
### **[Am I eligible? Natural Language Inference for Clinical Trial Patient Recruitment: the Patient's Point of View](http://arxiv.org/abs/2503.15718v1)**
### **[Reinforcement Learning Environment with LLM-Controlled Adversary in D&D 5th Edition Combat](http://arxiv.org/abs/2503.15726v1)**
### **[Uncertainty-Aware Diffusion Guided Refinement of 3D Scenes](http://arxiv.org/abs/2503.15742v1)**
### **[AutoRedTeamer: Autonomous Red Teaming with Lifelong Attack Integration](http://arxiv.org/abs/2503.15754v1)**
### **[ATTENTION2D: Communication Efficient Distributed Self-Attention Mechanism](http://arxiv.org/abs/2503.15758v1)**
### **[GraPLUS: Graph-based Placement Using Semantics for Image Composition](http://arxiv.org/abs/2503.15761v1)**
### **[Detecting LLM-Written Peer Reviews](http://arxiv.org/abs/2503.15772v1)**
### **[AutoDrive-QA- Automated Generation of Multiple-Choice Questions for Autonomous Driving Datasets Using Large Vision-Language Models](http://arxiv.org/abs/2503.15778v1)**
### **[Grammar and Gameplay-aligned RL for Game Description Generation with LLMs](http://arxiv.org/abs/2503.15783v1)**
### **[RL4Med-DDPO: Reinforcement Learning for Controlled Guidance Towards Diverse Medical Image Generation using Vision-Language Foundation Models](http://arxiv.org/abs/2503.15784v1)**
### **[Controlling Avatar Diffusion with Learnable Gaussian Embedding](http://arxiv.org/abs/2503.15809v1)**
### **[Attention Pruning: Automated Fairness Repair of Language Models via Surrogate Simulated Annealing](http://arxiv.org/abs/2503.15815v1)**
### **[A Vision Centric Remote Sensing Benchmark](http://arxiv.org/abs/2503.15816v1)**
### **[EDEN: Enhanced Diffusion for High-quality Large-motion Video Frame Interpolation](http://arxiv.org/abs/2503.15831v1)**
### **[Fùxì: A Benchmark for Evaluating Language Models on Ancient Chinese Text Understanding and Generation](http://arxiv.org/abs/2503.15837v1)**
### **[Enhancing LLM Code Generation with Ensembles: A Similarity-Based Selection Approach](http://arxiv.org/abs/2503.15838v1)**
### **[Automatic Generation of Safety-compliant Linear Temporal Logic via Large Language Model: A Self-supervised Framework](http://arxiv.org/abs/2503.15840v1)**
### **[Uncertainty Quantification and Confidence Calibration in Large Language Models: A Survey](http://arxiv.org/abs/2503.15850v1)**
### **[Zero-1-to-A: Zero-Shot One Image to Animatable Head Avatars Using Video Diffusion](http://arxiv.org/abs/2503.15851v1)**
### **[DroidTTP: Mapping Android Applications with TTP for Cyber Threat Intelligence](http://arxiv.org/abs/2503.15866v1)**
### **[TruthLens: Explainable DeepFake Detection for Face Manipulated and Fully Synthetic Data](http://arxiv.org/abs/2503.15867v1)**
### **[UniCoRN: Latent Diffusion-based Unified Controllable Image Restoration Network across Multiple Degradations](http://arxiv.org/abs/2503.15868v1)**
### **[MASH-VLM: Mitigating Action-Scene Hallucination in Video-LLMs through Disentangled Spatial-Temporal Representations](http://arxiv.org/abs/2503.15871v1)**
### **[DeepPsy-Agent: A Stage-Aware and Deep-Thinking Emotional Support Agent System](http://arxiv.org/abs/2503.15876v1)**
### **[Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](http://arxiv.org/abs/2503.15877v1)**
### **[Enhancing Zero-Shot Image Recognition in Vision-Language Models through Human-like Concept Guidance](http://arxiv.org/abs/2503.15886v1)**
### **[Parameters vs. Context: Fine-Grained Control of Knowledge Reliance in Language Models](http://arxiv.org/abs/2503.15888v1)**
### **[Time After Time: Deep-Q Effect Estimation for Interventions on When and What to do](http://arxiv.org/abs/2503.15890v1)**
### **[CONTHER: Human-Like Contextual Robot Learning via Hindsight Experience Replay and Transformers without Expert Demonstrations](http://arxiv.org/abs/2503.15895v1)**
### **[On the Limits of Applying Graph Transformers for Brain Connectome Classification](http://arxiv.org/abs/2503.15902v1)**
### **[From Structured Prompts to Open Narratives: Measuring Gender Bias in LLMs Through Open-Ended Storytelling](http://arxiv.org/abs/2503.15904v1)**
### **[Jasmine: Harnessing Diffusion Prior for Self-supervised Depth Estimation](http://arxiv.org/abs/2503.15905v1)**
### **[Text-Driven Diffusion Model for Sign Language Production](http://arxiv.org/abs/2503.15914v1)**
### **[Towards Automatic Continual Learning: A Self-Adaptive Framework for Continual Instruction Tuning](http://arxiv.org/abs/2503.15924v1)**
### **[BlockDance: Reuse Structurally Similar Spatio-Temporal Features to Accelerate Diffusion Transformers](http://arxiv.org/abs/2503.15927v1)**
### **[SaMam: Style-aware State Space Model for Arbitrary Image Style Transfer](http://arxiv.org/abs/2503.15934v1)**
### **[Advancing Mobile GUI Agents: A Verifier-Driven Approach to Practical Deployment](http://arxiv.org/abs/2503.15937v1)**
### **[From Chaos to Order: The Atomic Reasoner Framework for Fine-grained Reasoning in Large Language Models](http://arxiv.org/abs/2503.15944v1)**
### **[GAN-enhanced Simulation-driven DNN Testing in Absence of Ground Truth](http://arxiv.org/abs/2503.15953v1)**
### **[Acc3D: Accelerating Single Image to 3D Diffusion Models via Edge Consistency Guided Score Distillation](http://arxiv.org/abs/2503.15975v1)**
### **[A Survey on fMRI-based Brain Decoding for Reconstructing Multimodal Stimuli](http://arxiv.org/abs/2503.15978v1)**
### **[SpiLiFormer: Enhancing Spiking Transformers with Lateral Inhibition](http://arxiv.org/abs/2503.15986v1)**
### **[ECKGBench: Benchmarking Large Language Models in E-commerce Leveraging Knowledge Graph](http://arxiv.org/abs/2503.15990v1)**
### **[Animating the Uncaptured: Humanoid Mesh Animation with Video Diffusion Models](http://arxiv.org/abs/2503.15996v1)**
### **[SenseExpo: Efficient Autonomous Exploration with Prediction Information from Lightweight Neural Networks](http://arxiv.org/abs/2503.16000v1)**
### **["This could save us months of work" -- Use Cases of AI and Automation Support in Investigative Journalism](http://arxiv.org/abs/2503.16011v1)**
### **[GraspCoT: Integrating Physical Property Reasoning for 6-DoF Grasping under Flexible Language Instructions](http://arxiv.org/abs/2503.16013v1)**
### **[Autonomous AI imitators increase diversity in homogeneous information ecosystems](http://arxiv.org/abs/2503.16021v1)**
### **[Corrective In-Context Learning: Evaluating Self-Correction in Large Language Models](http://arxiv.org/abs/2503.16022v1)**
### **[BadToken: Token-level Backdoor Attacks to Multi-modal Large Language Models](http://arxiv.org/abs/2503.16023v1)**
### **[The Lighthouse of Language: Enhancing LLM Agents via Critique-Guided Improvement](http://arxiv.org/abs/2503.16024v1)**
### **[Single Image Iterative Subject-driven Generation and Editing](http://arxiv.org/abs/2503.16025v1)**
### **[Hybrid-Level Instruction Injection for Video Token Compression in Multi-modal Large Language Models](http://arxiv.org/abs/2503.16036v1)**
### **[Evaluating Test-Time Scaling LLMs for Legal Reasoning: OpenAI o1, DeepSeek-R1, and Beyond](http://arxiv.org/abs/2503.16040v1)**
### **[GreenIQ: A Deep Search Platform for Comprehensive Carbon Market Analysis and Automated Report Generation](http://arxiv.org/abs/2503.16041v1)**
### **[Meta-Learning Neural Mechanisms rather than Bayesian Priors](http://arxiv.org/abs/2503.16048v1)**
### **[Expert Race: A Flexible Routing Strategy for Scaling Diffusion Transformer with Mixture of Experts](http://arxiv.org/abs/2503.16057v1)**
### **[Shining Yourself: High-Fidelity Ornaments Virtual Try-on with Diffusion Model](http://arxiv.org/abs/2503.16065v1)**
### **[Tuning LLMs by RAG Principles: Towards LLM-native Memory](http://arxiv.org/abs/2503.16071v1)**
### **[Cultural Alignment in Large Language Models Using Soft Prompt Tuning](http://arxiv.org/abs/2503.16094v1)**
### **[PromptMobile: Efficient Promptus for Low Bandwidth Mobile Video Streaming](http://arxiv.org/abs/2503.16112v1)**
### **[The Impact of Revealing Large Language Model Stochasticity on Trust, Reliability, and Anthropomorphization](http://arxiv.org/abs/2503.16114v1)**
### **[Improving Discriminator Guidance in Diffusion Models](http://arxiv.org/abs/2503.16117v1)**
### **[MKG-Rank: Enhancing Large Language Models with Knowledge Graph for Multilingual Medical Question Answering](http://arxiv.org/abs/2503.16131v1)**
### **[Only a Little to the Left: A Theory-grounded Measure of Political Bias in Large Language Models](http://arxiv.org/abs/2503.16148v1)**
### **[FreeFlux: Understanding and Exploiting Layer-Specific Roles in RoPE-Based MMDiT for Versatile Image Editing](http://arxiv.org/abs/2503.16153v1)**
### **[Automatically Generating Chinese Homophone Words to Probe Machine Translation Estimation Systems](http://arxiv.org/abs/2503.16158v1)**
### **[Towards Lighter and Robust Evaluation for Retrieval Augmented Generation](http://arxiv.org/abs/2503.16161v1)**
### **[SpeCache: Speculative Key-Value Caching for Efficient Generation of LLMs](http://arxiv.org/abs/2503.16163v1)**
### **[CodeReviewQA: The Code Review Comprehension Assessment for Large Language Models](http://arxiv.org/abs/2503.16167v1)**
### **[CLS-RL: Image Classification with Rule-Based Reinforcement Learning](http://arxiv.org/abs/2503.16188v1)**
### **[Large Language Models for Water Distribution Systems Modeling and Decision-Making](http://arxiv.org/abs/2503.16191v1)**
### **[Affective Polarization Amongst Swedish Politicians](http://arxiv.org/abs/2503.16193v1)**
### **[Improving Autoregressive Image Generation through Coarse-to-Fine Token Prediction](http://arxiv.org/abs/2503.16194v1)**
### **[MathFusion: Enhancing Mathematic Problem-solving of LLM through Instruction Fusion](http://arxiv.org/abs/2503.16212v1)**
### **[Temporal Score Analysis for Understanding and Correcting Diffusion Artifacts](http://arxiv.org/abs/2503.16218v1)**
### **[Reinforcement Learning for Reasoning in Small LLMs: What Works and What Doesn't](http://arxiv.org/abs/2503.16219v1)**
### **[Fin-R1: A Large Language Model for Financial Reasoning through Reinforcement Learning](http://arxiv.org/abs/2503.16252v1)**
### **[Plug-and-Play 1.x-Bit KV Cache Quantization for Video Large Language Models](http://arxiv.org/abs/2503.16257v1)**
### **[Chain of Functions: A Programmatic Pipeline for Fine-Grained Chart Reasoning Data](http://arxiv.org/abs/2503.16260v1)**
### **[Uni-3DAR: Unified 3D Generation and Understanding via Autoregression on Compressed Spatial Tokens](http://arxiv.org/abs/2503.16278v1)**
### **[SceneMI: Motion In-betweening for Modeling Human-Scene Interactions](http://arxiv.org/abs/2503.16289v1)**
### **[Diffusion-augmented Graph Contrastive Learning for Collaborative Filter](http://arxiv.org/abs/2503.16290v1)**
### **[Unleashing Vecset Diffusion Model for Fast Shape Generation](http://arxiv.org/abs/2503.16302v1)**
### **[Bridging Technology and Humanities: Evaluating the Impact of Large Language Models on Social Sciences Research with DeepSeek-R1](http://arxiv.org/abs/2503.16304v1)**
### **[Ultra-Resolution Adaptation with Ease](http://arxiv.org/abs/2503.16322v1)**
### **[OmniGeo: Towards a Multimodal Large Language Models for Geospatial Artificial Intelligence](http://arxiv.org/abs/2503.16326v1)**
### **[Lyra: An Efficient and Expressive Subquadratic Architecture for Modeling Biological Sequences](http://arxiv.org/abs/2503.16351v1)**
### **[CaKE: Circuit-aware Editing Enables Generalizable Knowledge Learners](http://arxiv.org/abs/2503.16356v1)**
### **[LaPIG: Cross-Modal Generation of Paired Thermal and Visible Facial Images](http://arxiv.org/abs/2503.16376v1)**
### **[Deconstructing Long Chain-of-Thought: A Structured Reasoning Optimization Framework for Long CoT Distillation](http://arxiv.org/abs/2503.16385v1)**
### **[Do Visual Imaginations Improve Vision-and-Language Navigation Agents?](http://arxiv.org/abs/2503.16394v1)**
### **[SV4D 2.0: Enhancing Spatio-Temporal Consistency in Multi-View Video Diffusion for High-Quality 4D Generation](http://arxiv.org/abs/2503.16396v1)**
### **[Scale-wise Distillation of Diffusion Models](http://arxiv.org/abs/2503.16397v1)**
### **[ScalingNoise: Scaling Inference-Time Search for Generating Infinite Videos](http://arxiv.org/abs/2503.16400v1)**
### **[Exploring the Hidden Reasoning Process of Large Language Models by Misleading Them](http://arxiv.org/abs/2503.16401v1)**
### **[VerbDiff: Text-Only Diffusion Models with Enhanced Interaction Awareness](http://arxiv.org/abs/2503.16406v1)**
### **[DreamTexture: Shape from Virtual Texture with Analysis by Augmentation](http://arxiv.org/abs/2503.16412v1)**
### **[InfiniteYou: Flexible Photo Recrafting While Preserving Your Identity](http://arxiv.org/abs/2503.16418v1)**
### **[Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models](http://arxiv.org/abs/2503.16419v1)**
