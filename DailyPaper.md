# The Latest Daily Papers - Date: 2025-02-22
## Highlight Papers
### **[Is That Your Final Answer? Test-Time Scaling Improves Selective Question Answering](http://arxiv.org/abs/2502.13962v1)**
- **Summary**: Here's a concise summary of the paper and a rigorous critical evaluation:

**Concise Summary:**

The paper investigates the impact of test-time scaling on selective question answering (SQA) using large language models (LLMs).  Existing test-time scaling research assumes models should always answer, ignoring confidence. This paper introduces confidence thresholds during inference, allowing models to abstain from answering when uncertain.  Experiments show that increasing compute budget improves both accuracy and confidence in correct answers.  The authors propose evaluating SQA under various risk scenarios (Exam Odds, Jeopardy Odds, High-Stakes Odds), moving beyond the traditional zero-risk assumption.  They find that test-time scaling significantly improves utility, especially under high-risk scenarios.


**Rigorous and Critical Evaluation:**

This paper makes a valuable contribution by addressing a significant limitation in the current evaluation of test-time scaling for LLMs.  The focus on confidence-based abstention is a crucial step towards deploying LLMs in real-world applications where incorrect answers carry costs.  The introduction of different risk scenarios for evaluation (Exam Odds, Jeopardy Odds, High-Stakes Odds) offers a more nuanced and realistic assessment of model performance compared to the prevailing zero-risk approach.  The empirical results demonstrate a clear benefit of test-time scaling in improving both accuracy and confidence, particularly at higher confidence thresholds. The visualization of the trade-offs between accuracy, response rate, and compute budget is helpful.

However, the paper has some weaknesses:

* **Limited Novelty in Core Idea:** While the application to SQA and the proposed evaluation framework are novel, the core idea of using confidence thresholds for selective answering is not entirely new.  Similar approaches exist in other machine learning domains.  The novelty lies more in the systematic application and evaluation within the specific context of test-time scaling for LLMs.
* **Specific Model Dependence:**  The results are largely based on two specific models (DeepSeek-R1-32B and s1-32B). Generalizing the findings to other LLMs needs further investigation.
* **Simplicity of the Selection Function:** The selection function used is quite simple (thresholding confidence scores).  More sophisticated selection methods might yield better results.  The authors acknowledge this limitation.
* **Limited Discussion of Computational Costs:** While the paper acknowledges computational costs, a more in-depth analysis of the trade-off between improved performance and increased compute resources would strengthen the findings.

Despite these weaknesses, the paper's contribution to the field of LLM evaluation and deployment is significant.  It highlights a crucial gap in the existing literature and proposes a more comprehensive and realistic evaluation framework.  The findings have practical implications for developing more robust and reliable LLM-based systems. The impact will likely be felt in future research on test-time scaling and selective question answering.


Score: 8

**Rationale:** The score reflects the paper's significant contribution in addressing a crucial limitation in LLM evaluation, proposing a more comprehensive evaluation framework, and presenting strong empirical results. However, the score is not a 10 because the core idea of confidence-based abstention is not entirely novel, the results are somewhat model-specific, and the selection function employed is relatively basic.  Nevertheless, the paper's impact on future research and the practical implications of its findings justify a high score.

- **Score**: 8/10

### **[Towards Secure Program Partitioning for Smart Contracts with LLM's In-Context Learning](http://arxiv.org/abs/2502.14215v1)**
- **Summary**: Here's a concise summary of the paper and a rigorous critical evaluation:


**Concise Summary:**

The paper introduces PARTITIONGPT, a novel approach to enhancing the security of smart contracts by mitigating manipulation attacks stemming from sensitive data exposure.  PARTITIONGPT leverages Large Language Models (LLMs) and static analysis to partition smart contracts into privileged and non-privileged code sections. The privileged code, containing sensitive operations, is designed to run in a secure environment (e.g., a Trusted Execution Environment), while the non-privileged code remains on the public blockchain.  The authors evaluate PARTITIONGPT on various smart contracts and real-world attack scenarios, demonstrating its effectiveness in preventing manipulation attacks while incurring a moderate runtime overhead.


**Rigorous and Critical Evaluation:**

**Novelty:** The core idea of using LLMs for fine-grained program partitioning of smart contracts to isolate sensitive operations is relatively novel.  Existing approaches often focus on coarse-grained solutions (e.g., entire function isolation) or rely heavily on manual annotation and developer expertise.  The use of taint analysis coupled with LLM-driven code refactoring to achieve this automatically is a significant advancement.

**Significance:** The potential impact is substantial.  Manipulation attacks are a significant threat to the security and financial stability of decentralized finance (DeFi) systems.  An automated approach like PARTITIONGPT could significantly reduce the burden on developers to manually secure their contracts and improve the overall security posture of the ecosystem.

**Strengths:**

* **Novel application of LLMs:** The use of LLMs for automated program partitioning is a creative and effective way to address a challenging problem.
* **Comprehensive evaluation:** The paper includes both quantitative and qualitative evaluations, testing the approach on various smart contracts and real-world attack cases.
* **Addressing a critical problem:** Manipulation attacks are a major concern in the DeFi space, making this research highly relevant.
* **Formal verification:** The inclusion of an equivalence checker adds rigor and trustworthiness to the generated partitions.


**Weaknesses:**

* **LLM dependence:**  The accuracy and performance of PARTITIONGPT are heavily reliant on the capabilities of the chosen LLM.  This creates a dependency on external services and raises concerns about potential biases or limitations of the LLM.
* **Limited scope of sensitivity annotation:**  The paper primarily focuses on sensitive *variables*.  More complex situations, such as sensitive function logic beyond simple data access, might not be adequately addressed.
* **Runtime overhead:** While claimed to be moderate, the runtime overhead of deploying partitions to a TEE-based environment could be a significant barrier to adoption for resource-constrained applications.  A deeper analysis considering different TEE implementations and transaction costs would strengthen the argument.
* **Security of the equivalence checker:** The paper doesn't extensively discuss the security and robustness of its own equivalence checker.  A compromised equivalence checker could undermine the security guarantees.


**Overall Score and Rationale:**

Considering the novelty of its LLM-based approach, the relevance to a critical security problem, and the reasonably comprehensive evaluation, this paper makes a solid contribution to the field. However, the reliance on LLMs, the potential runtime overhead, and the lack of deeper analysis on the security of the equivalence checker temper its impact.  Further research and refinement of the approach are necessary.

Score: 8

- **Score**: 8/10

### **[MCQA-Eval: Efficient Confidence Evaluation in NLG with Gold-Standard Correctness Labels](http://arxiv.org/abs/2502.14268v1)**
- **Summary**: The paper introduces MCQA-Eval, a novel framework for evaluating confidence estimation methods in Natural Language Generation (NLG).  Unlike existing methods that rely on potentially noisy and unreliable correctness functions, MCQA-Eval leverages gold-standard correctness labels from multiple-choice question-answering datasets. This eliminates the dependence on subjective correctness judgments, leading to more efficient and reliable evaluations.  The authors demonstrate the effectiveness of their framework through extensive experiments on various LLMs and datasets.


**Rigorous and Critical Evaluation:**

**Novelty:** The core novelty lies in the use of multiple-choice QA datasets to bypass the need for an explicit correctness function.  This is a clever approach that addresses a significant limitation of existing evaluation methods.  However, the adaptation of existing confidence measures to this new framework isn't inherently novel; the key contribution is the proposed evaluation methodology, not the confidence measures themselves.  The use of multiple-choice datasets for evaluation isn't entirely new, but applying it specifically to address the noise in existing correctness functions in NLG is a valuable contribution.

**Significance:** The potential impact of MCQA-Eval is significant. By providing a more reliable and efficient evaluation framework, it could lead to improved confidence estimation methods and better understanding of LLM reliability, particularly in high-stakes applications. The scalability advantage of MCQA-Eval is also important for the field.  However, the paper's scope is limited to confidence estimation; it doesn't directly address calibration or other related aspects of uncertainty quantification in NLG.

**Strengths:**

* **Addresses a crucial problem:** The reliance on noisy correctness functions in current NLG confidence evaluation is a major weakness that MCQA-Eval directly addresses.
* **Increased efficiency and reliability:** The use of gold-standard labels leads to significantly more efficient and reliable evaluations.
* **Broad applicability:** MCQA-Eval is applicable to various confidence estimation methods and LLMs.
* **Thorough experimentation:** The paper presents a comprehensive empirical evaluation.

**Weaknesses:**

* **Limited novelty in confidence estimation techniques:** The paper does not propose new confidence measures; its main contribution is the evaluation framework.
* **Potential bias in dataset selection:** The choice of multiple-choice datasets could introduce a bias, affecting the generalizability of the results.  Further investigation into the suitability of various MCQA datasets would strengthen this point.
* **No direct comparison with human evaluation:**  A full comparison against human evaluation for correctness would provide a stronger benchmark.


Considering the strengths and weaknesses, and the potential impact on the field's practice in evaluating confidence estimation methods in NLG, I would score this paper as follows:

Score: 8

**Rationale:** The paper makes a strong contribution by presenting a novel and significantly improved evaluation framework.  The increased efficiency and reliability of MCQA-Eval are valuable advancements. However, the limited novelty in the proposed confidence measures and the potential limitations regarding dataset bias slightly detract from its overall impact.  The paper's contribution is substantial, making an 8 a justified score.

- **Score**: 8/10

### **[Capturing Nuanced Preferences: Preference-Aligned Distillation for Small Language Models](http://arxiv.org/abs/2502.14272v1)**
- **Summary**: Here's a concise summary of the paper and a rigorous, critical evaluation:

**Concise Summary:**

The paper introduces Preference-Aligned Distillation (PAD), a novel framework for aligning small language models (SLMs) with human preferences.  Unlike existing methods that rely on pairwise comparisons of model outputs, PAD models the teacher LLM's preferences as a probability distribution over all possible rankings of generated responses. This approach captures nuanced preferences more effectively.  PAD incorporates three key steps: diverse response sampling, reward calculation (with calibration using MCQ selection probabilities), and preference distillation using either vanilla or probabilistic preference loss. Experiments across multiple benchmarks demonstrate that PAD consistently outperforms existing methods, showing significant improvements in alignment with human preferences.


**Rigorous and Critical Evaluation:**

The paper presents a valuable contribution to the field of aligning LLMs with human preferences, but its novelty and significance aren't without limitations.

**Strengths:**

* **Novel Approach to Preference Modeling:**  The core contribution – modeling preferences as a probability distribution over rankings rather than simple pairwise comparisons – is novel and addresses a clear limitation in existing work.  This allows for a more nuanced understanding of preferences and potentially avoids the limitations of binary preference annotations.
* **Effective Calibration Strategy:** The use of MCQ selection probabilities for reward calibration is a clever approach that mitigates the issue of miscalibration in LLMs, which is a common problem in preference learning.
* **Comprehensive Evaluation:** The paper evaluates PAD on four established benchmarks, comparing it against various baselines. This provides a strong empirical validation of the proposed method.
* **Superior Performance:**  The results convincingly show that PAD significantly outperforms existing methods across multiple metrics, indicating a substantial improvement in alignment with human preferences.


**Weaknesses:**

* **Computational Cost:** The preference decomposition strategy is introduced to address the computational complexity of the full distribution modeling, but it remains a concern, particularly for very large response sets.  The paper partially addresses this, but more thorough analysis of scalability is needed.
* **Generalizability Concerns:** While the paper demonstrates good performance across different benchmark datasets, further evaluation with more diverse datasets and language model families would strengthen the claim of broad applicability.  The heterogeneous study is a start, but it's limited in scope.
* **Limited Theoretical Analysis:** While the methodology is well-described, deeper theoretical analysis of the proposed approach, particularly regarding the choice of loss functions and the convergence properties, would be beneficial.
* **Black-Box Models:**  The reliance on token-level probabilities limits applicability to black-box models where such probabilities are not accessible.

**Significance and Impact:**

PAD offers a valuable advancement in aligning SLMs with human preferences.  The improved accuracy and the more nuanced understanding of preferences achieved by the probability distribution approach have potential to significantly impact various downstream applications relying on aligned LLMs, such as chatbots, dialogue systems and personalized AI assistants. However,  addressing the computational limitations and thoroughly exploring generalizability to a broader range of models and tasks are crucial for maximizing its impact on the field.

**Score: 8**

The score reflects the paper's strong contribution.  The novel approach to preference modeling and the demonstration of improved performance across various benchmarks warrant a high score. However, limitations concerning computational cost, generalizability, and theoretical depth prevent it from achieving a perfect score.  Future work addressing these limitations will further solidify its impact within the field.

- **Score**: 8/10

### **[Vulnerability of Text-to-Image Models to Prompt Template Stealing: A Differential Evolution Approach](http://arxiv.org/abs/2502.14285v1)**
- **Summary**: Here's a concise summary of the paper and a critical evaluation of its novelty and significance:

**Concise Summary:**

The paper investigates the vulnerability of text-to-image models to prompt template stealing.  It introduces PRISM, a benchmark dataset of prompt templates and generated images, categorized by difficulty. The authors propose EvoStealer, a novel prompt stealing method that leverages differential evolution algorithms and large language models (LLMs) without requiring model fine-tuning.  EvoStealer's effectiveness is demonstrated through experiments on various open-source and closed-source models, showing it outperforms baseline methods in reproducing similar images and generalizing to new subjects.  The paper also analyzes the computational cost of the attack.


**Rigorous and Critical Evaluation:**

The paper addresses a timely and important problem: the theft of intellectual property in the form of prompt templates for text-to-image generation.  This is a significant concern for creators and marketplaces. The creation of PRISM, a benchmark dataset, is a valuable contribution, allowing for more standardized and reproducible evaluation of prompt stealing methods.

**Strengths:**

* **Addresses a relevant problem:** Prompt template theft is a real-world issue with significant economic and intellectual property implications.
* **Novel methodology:** EvoStealer uses a novel combination of differential evolution and LLMs, avoiding the need for model fine-tuning, which is a significant advantage.
* **Comprehensive evaluation:** The authors conduct experiments on multiple models (both open-source and closed-source), using multiple metrics, and providing both in-domain and out-of-domain evaluations.  The inclusion of human evaluation is a strength.
* **Benchmark dataset:** The creation of the PRISM dataset is a significant contribution to the field, providing a standardized way to evaluate prompt stealing techniques.

**Weaknesses:**

* **Limited Generalizability:** While EvoStealer performs well on the PRISM dataset, it's unclear how well it will generalize to other datasets or different types of prompt templates. The reliance on DALL-E 3 for image generation during the benchmark creation might limit the generalizability of the findings.
* **Cost analysis limited:** The cost analysis focuses on the API calls and tokens, but it doesn't fully account for the computational resources required, especially considering the iterative nature of the differential evolution algorithm.  A more thorough cost analysis considering different hardware and software setups would strengthen the paper.
* **Ethical considerations are somewhat superficial:** While the paper mentions ethical concerns, a deeper discussion of the potential societal impact of prompt stealing and the responsible use of EvoStealer is needed. The suggested defensive strategies (limiting the number of displayed images) are relatively simplistic.


**Significance and Novelty:**

The paper makes a notable contribution to the field by highlighting the vulnerability of text-to-image models to prompt stealing and proposing a novel method to perform this attack. The introduction of the PRISM benchmark is a significant advance that will likely be adopted by other researchers. However, some limitations regarding generalizability and a slightly underdeveloped ethical discussion prevent it from being a truly groundbreaking work.

**Score: 8**

The score reflects the paper's strengths: addressing a relevant problem with a novel and well-evaluated method, and importantly introducing a valuable benchmark dataset.  The weaknesses, particularly concerning generalizability and a more thorough analysis of costs and ethical implications, prevent it from achieving a higher score.  The work is undoubtedly significant and will influence future research in this area.

- **Score**: 8/10

### **[Drift: Decoding-time Personalized Alignments with Implicit User Preferences](http://arxiv.org/abs/2502.14289v1)**
- **Summary**: Here's a concise summary of the paper and a rigorous critical evaluation:

**Concise Summary:**

The paper introduces "Drift," a novel framework for personalizing large language models (LLMs) at decoding time using only a few dozen examples per user.  Unlike traditional reinforcement learning from human feedback (RLHF), Drift is training-free. It decomposes complex user preferences into interpretable attributes, models these attributes efficiently using a differential prompting approach, and integrates them into the LLM's decoding process.  Experiments on synthetic and real-world datasets demonstrate Drift's superior performance compared to RLHF baselines in few-shot scenarios, showcasing its computational efficiency and interpretability.


**Rigorous and Critical Evaluation:**

The paper presents a valuable contribution to the field of LLM personalization, addressing the challenges of data scarcity and computational cost associated with traditional methods.  Drift's training-free nature and few-shot learning capability are significant advantages. The decomposition of preferences into interpretable attributes allows for both personalization and some degree of model explainability.  The use of differential prompting for zero-shot reward modeling is innovative and efficient, avoiding the need for large-scale attribute-specific datasets.  The theoretical justification for Drift's decoding mechanism is sound.

However, several limitations warrant critical consideration:

* **Limited evaluation on real user data:** While the paper uses both synthetic and real datasets, the real-world dataset (PRISM) is relatively small, potentially limiting the generalizability of the findings. More extensive, robust user studies are needed to fully assess Drift's performance and reliability in diverse contexts.
* **Dependence on smaller language models (SLMs):** Drift's performance relies on the accuracy and effectiveness of the SLMs used for attribute modeling.  The paper needs a deeper discussion of SLM selection and the potential impact of SLM limitations on Drift's overall performance.
* **Implicit preference modeling challenges:**  Accurately capturing implicit preferences is inherently difficult. The paper acknowledges this, but a more in-depth analysis of potential biases in the implicit preference modeling process would strengthen the argument.
* **The generalizability of attributes:** The chosen attributes might not be universally applicable across diverse user populations and contexts.  Further investigation into attribute selection and generalization is required.
* **Token-level dependencies:** The reliance on a specific tokenizer presents a constraint and limits the adaptability of the method.

Despite these limitations, Drift offers a promising and efficient approach to personalized LLM generation.  The novelty lies in its training-free nature, few-shot learning, and the clever use of differential prompting. The potential impact on the field is significant, particularly for applications with limited user data or where computational resources are constrained.  However, the need for further validation and investigation of its limitations prevents it from achieving a higher score.

Score: 8

**Rationale:**  The score reflects the paper's substantial contribution to LLM personalization through a novel, efficient, and interpretable approach.  While promising, the limitations concerning evaluation scale, reliance on SLMs, and the challenges inherent in capturing implicit preferences warrant a score below 9 or 10.  Addressing these limitations in future work could significantly enhance the impact and influence of Drift on the field.

- **Score**: 8/10

### **[Token-Level Density-Based Uncertainty Quantification Methods for Eliciting Truthfulness of Large Language Models](http://arxiv.org/abs/2502.14427v1)**
- **Summary**: Here's a concise summary of the paper, followed by a critical evaluation:

**Concise Summary:**

The paper introduces novel token-level density-based uncertainty quantification (UQ) methods for evaluating the truthfulness of Large Language Models (LLMs).  Instead of relying on sequence-level analysis, which previous work showed to be ineffective, the authors adapt Mahalanobis Distance (MD) to individual tokens extracted from multiple LLM layers. This information, along with sequence probability, is fed into a linear regression model to predict uncertainty.  Extensive experiments across eleven datasets demonstrate significant improvements over existing UQ methods for both selective generation and claim-level fact-checking, highlighting the method's efficiency and generalizability.

**Rigorous and Critical Evaluation:**

This paper makes a valuable contribution to the field of LLM evaluation, but its novelty and significance aren't without caveats.

**Strengths:**

* **Addresses a significant limitation:** The paper directly addresses the documented failure of previous density-based UQ methods for LLMs in the generation setting.  By shifting to a token-level approach, it overcomes this major hurdle.
* **Improved performance:** The experimental results demonstrate substantial improvements over existing state-of-the-art UQ methods across multiple datasets and tasks.  The consistent improvement across various LLMs further strengthens the findings.
* **Computational efficiency:**  The proposed method is computationally efficient compared to sampling-based methods, a crucial advantage for real-world applications.
* **Thorough evaluation:**  The authors conduct a comprehensive evaluation across numerous datasets, tasks, and baselines, providing strong empirical support for their claims.  They also address out-of-domain generalization and the impact of hyperparameter choices.

**Weaknesses:**

* **Supervised approach:** The reliance on supervised learning is a significant limitation. Obtaining high-quality labeled data for training is challenging and expensive, limiting the general applicability of the method.  The robustness of the model to noisy labels is not explicitly assessed.
* **Linear model simplicity:**  While efficient, the use of a simple linear regression model might not fully capture the complex relationships between the input features and uncertainty.  More sophisticated models could potentially yield further improvements.
* **Limited exploration of hyperparameter sensitivity:** While the authors touch upon this, a more in-depth analysis of hyperparameter sensitivity and the influence on model performance is lacking.
* **Lack of theoretical justification:** The paper lacks strong theoretical justification for the effectiveness of the proposed approach beyond empirical evidence.  A deeper dive into the underlying reasons for the success of the token-level MD approach would add significant weight to the findings.


**Overall Significance and Novelty:**

The paper presents a significant advancement in the field of LLM UQ. The move to token-level analysis, combined with the improved empirical performance and computational efficiency, addresses a critical weakness of existing methods.  However, the supervised nature and lack of deeper theoretical analysis limit the overall impact. The lack of analysis of the effect of training set size on the performance should also be addressed.


**Score: 8**

The score reflects the substantial contribution the paper makes by addressing a known limitation in the field and achieving superior empirical results.  However, the limitations of its supervised nature, relative simplicity, and the absence of a deeper theoretical justification prevent it from achieving a higher score.  Further research addressing these weaknesses could significantly elevate its impact on the field.

- **Score**: 8/10

### **[PEARL: Towards Permutation-Resilient LLMs](http://arxiv.org/abs/2502.14628v1)**
- **Summary**: Here's a concise summary of the PEARL paper followed by a rigorous and critical evaluation:


**Concise Summary:**

The paper addresses the vulnerability of large language models (LLMs) to permutation attacks on their input demonstrations (in-context learning).  Existing LLMs perform poorly when the order of demonstration examples is changed, even if the semantic content remains the same. The authors propose PEARL, a novel framework based on distributionally robust optimization (DRO). PEARL uses a permutation-proposal network (P-Net) to generate challenging input permutations, which are then used to train the LLM to become more robust to different orderings. Experiments on synthetic and real-world datasets demonstrate that PEARL improves both average and worst-case performance, showing increased robustness and better generalization to longer sequences and more examples than standard training methods.


**Rigorous and Critical Evaluation:**

The paper tackles a significant and increasingly relevant problem in the field of LLMs: the fragility of in-context learning to the ordering of input examples.  This is a genuine limitation with implications for the safety and reliability of these models.  PEARL's approach of using DRO and a generative adversarial network (GAN-like approach) to improve robustness against permutations is novel and potentially impactful.


**Strengths:**

* **Addresses a significant weakness:** The vulnerability of LLMs to simple permutation attacks is a serious concern, highlighting a need for improved robustness. PEARL directly addresses this issue.
* **Novel approach:**  Using DRO and a GAN-like structure to optimize against the worst-case permutation is a creative approach not extensively explored in the literature. The use of optimal transport (OT) to generate challenging permutations is also a novel contribution.
* **Empirical validation:**  The paper presents results across various datasets and model sizes, demonstrating improved performance and robustness.  The inclusion of both synthetic and real-world datasets strengthens the findings.
* **Efficiency gains:**  The authors highlight the efficiency of PEARL, demonstrating performance gains even when trained on fewer shots and shorter contexts, suggesting scalability.


**Weaknesses:**

* **Computational cost of P-Net:** While PEARL improves robustness, the introduction of the P-Net adds computational overhead during training. The paper doesn't fully address this trade-off.  A more comprehensive analysis of the computational cost of PEARL versus other methods is needed.
* **Generalizability beyond permutations:**  The paper primarily focuses on permutation attacks.  While this is a significant vulnerability, the generalizability of PEARL to other forms of adversarial attacks remains unclear.  Further investigation is needed to establish broader robustness.
* **Limited explanation of DRO's role:** The paper's explanation of the theoretical underpinnings of DRO, while present, could be more rigorous and accessible to a broader audience.  The connection between the mathematical formulation and the practical implementation needs more elaboration.
* **Comparison to alternative methods:** The comparison to other permutation-handling techniques is somewhat limited.  A more thorough comparative analysis against existing methods would further solidify PEARL's significance.


**Potential Influence:**

The paper has the potential to influence future research on LLM robustness and in-context learning.  The approach of using DRO and generative models to enhance adversarial robustness is likely to inspire further research in this area. The demonstrated efficiency gains of PEARL could make it attractive for practical applications.


**Score:** 8

**Rationale:**  PEARL addresses a crucial weakness in LLMs, proposing a novel and effective method to improve robustness. The experimental results are strong, and the efficiency gains are noteworthy. However, the computational cost of the P-Net, the limited analysis of generalizability to other attacks, and a lack of extensive comparisons to alternative methods prevent a higher score.  Further research addressing these weaknesses would solidify PEARL's position as a truly exceptional contribution.

- **Score**: 8/10

### **[How to Get Your LLM to Generate Challenging Problems for Evaluation](http://arxiv.org/abs/2502.14678v1)**
- **Summary**: The paper introduces CHASE, a framework for generating challenging synthetic evaluation benchmarks for large language models (LLMs).  CHASE employs a bottom-up approach, building complex problems from simpler components and decomposing the generation process into independently verifiable sub-tasks using multiple LLMs.  The authors demonstrate CHASE's effectiveness by creating benchmarks in three diverse domains: document-based question answering, code completion, and math reasoning.  These benchmarks prove challenging for state-of-the-art LLMs, achieving accuracies in the 40-60% range. The code and benchmarks are publicly released.


**Rigorous and Critical Evaluation:**

The paper makes a valuable contribution to the rapidly evolving field of LLM evaluation, addressing the limitations of human annotation and the saturation of existing benchmarks.  The bottom-up generation strategy and the decomposition into verifiable sub-tasks are novel aspects of the CHASE framework, improving the quality and reliability of synthetic data compared to previous methods that often rely solely on a single LLM for generation and lack robust verification. The public release of the benchmarks and code is a significant contribution, enabling others to build upon and extend this work. The experiments demonstrate that CHASE generates challenging problems that differentiate the performance of various LLMs, highlighting its utility beyond simply demonstrating achievement on existing saturated benchmarks.

However, some critical weaknesses exist:

* **Limited Scope:** While the three chosen domains are diverse, they do not represent the full spectrum of LLM capabilities.  The generalizability of CHASE to other tasks remains to be shown.
* **LLM Dependence:**  The framework heavily relies on LLMs for both generation and verification, introducing a potential bias. The performance of the LLMs used might affect the difficulty and quality of the generated problems.  A more robust verification method, perhaps incorporating human evaluation of a subset of the generated problems, could enhance the credibility.
* **Scalability Challenges:**  The authors acknowledge scalability issues, particularly for long-context tasks, where the verification process can be computationally expensive.  Further development might be needed to optimize resource usage and increase the scale of the datasets.
* **Evaluation Metric Choices:** The use of an LLM as a judge for CHASE-QA raises concerns about potential biases and inaccuracies in the evaluation. Further investigation of alternative evaluation metrics that are more reliable and less computationally expensive is warranted.


Despite these weaknesses, the novelty of CHASE's methodology, its demonstrated effectiveness in creating challenging benchmarks, and the availability of the code and data warrant a positive assessment. The potential influence on the field is considerable, as it provides a promising approach to addressing the crucial problem of creating high-quality LLM evaluation datasets. The impact of this work will likely be significant in shaping future research efforts in LLM evaluation, driving the development of more reliable and scalable methods.

Score: 8


- **Score**: 8/10

### **[Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation](http://arxiv.org/abs/2502.14846v1)**
- **Summary**: Here's a concise summary of the paper and a critical evaluation:

**Concise Summary:**

The paper introduces CoSyn, a framework for generating synthetic, text-rich multimodal data for training vision-language models (VLMs).  CoSyn uses large language models (LLMs) to generate code (in various languages like Python, HTML, LaTeX) that renders synthetic images.  The LLM then generates corresponding textual instructions, creating a vision-language dataset.  Experiments on several benchmarks show that models trained on CoSyn's synthetic data achieve state-of-the-art performance, surpassing even proprietary models in some cases.  The authors also demonstrate CoSyn's ability to generate pointing data for tasks requiring visual grounding.


**Rigorous and Critical Evaluation:**

This paper makes a valuable contribution to the field of vision-language modeling, addressing a significant limitation: the lack of high-quality, diverse text-rich image data. The core idea of using LLMs to generate code, which in turn generates images and annotations, is innovative and elegantly solves the data scarcity problem. This approach is more scalable and flexible than manually creating such datasets.

**Strengths:**

* **Novelty:** The methodology of using LLMs to generate code for image synthesis and annotation is novel and effectively addresses the data scarcity problem in text-rich image understanding.
* **Scalability:** CoSyn's approach is scalable, allowing for the generation of large, diverse datasets.
* **Data Diversity:** The paper demonstrates the generation of diverse image types and tasks.
* **Strong Empirical Results:** The empirical results convincingly show the effectiveness of the synthetic data in improving VLM performance. The comparison against both open-source and proprietary models is robust.
* **Addressing Out-of-Domain Generalization:** The NutritionQA benchmark highlights the ability of CoSyn-trained models to generalize to unseen domains.
* **Extension to Pointing Data:** The extension to pointing data demonstrates the versatility of the approach.

**Weaknesses:**

* **Bias in Synthetic Data:** While the paper acknowledges the potential for bias in synthetic data, a more in-depth analysis of potential biases introduced by the LLMs and the rendering processes would strengthen the argument.
* **Limited Analysis of Generalizability:** While the NutritionQA example showcases out-of-domain generalization, more comprehensive testing across a wider range of domains is needed to fully validate the claim.
* **Dependence on LLM Capabilities:** The success of CoSyn relies heavily on the capabilities of the underlying LLMs. Changes in LLM performance could affect CoSyn's output.  This is not explicitly addressed as a limitation.


**Significance and Potential Influence:**

This work has the potential to significantly impact the field. The proposed approach could be adopted by other researchers to generate synthetic data for various multimodal tasks, accelerating progress in VLM research. The open-source nature of the framework further enhances its impact.


**Score: 8.5**

The score reflects the strong novelty and significant contribution of the paper. The methodology is innovative and the results are compelling.  However, a more thorough investigation into potential biases in the synthetic data and a broader exploration of out-of-domain generalization would elevate the paper to a higher score.  The current work, while excellent, leaves room for further refinement and expansion.

- **Score**: 8/10

### **[GATE: Graph-based Adaptive Tool Evolution Across Diverse Tasks](http://arxiv.org/abs/2502.14848v1)**
- **Summary**: The paper introduces GATE, a graph-based adaptive tool evolution framework for LLMs. GATE dynamically constructs and refines a hierarchical tool graph, enabling efficient tool reuse and adaptation across diverse tasks.  It uses two agents, a Task Solver and a Tool Manager, to handle tool selection, generation, and graph maintenance.  Experiments on open-ended (Minecraft), agent-based (TextCraft, DABench), and code generation tasks (MATH, Date, TabMWP) demonstrate GATE's superior performance compared to existing methods, showcasing faster milestone completion and improved accuracy.


**Rigorous and Critical Evaluation:**

GATE presents a compelling approach to LLM tool management, addressing the limitations of existing methods that often create redundant or task-specific tools. The hierarchical tool graph and the two-agent system are novel contributions, offering a more efficient and adaptive way to manage and utilize tools. The empirical results strongly support the claims of improved efficiency and generalization.  The open-sourcing of code and data further enhances the paper's contribution.


However, a critical evaluation reveals some weaknesses:

* **Scope of Generalization:** While the paper demonstrates improved performance across various task types, the extent of its generalization capabilities remains to be fully explored. The specific datasets and task designs may limit the extent to which the findings can be generalized to entirely different domains or complexities. Future work needs to investigate GATE's performance on more diverse and challenging tasks.
* **Tool Complexity:** The paper mentions handling complex tools, but a detailed analysis of the tool complexity and the scalability of the framework with highly intricate tools is lacking.  Further analysis on the computational cost of managing and evolving a very large and complex tool graph is needed.
* **Comparison with Baselines:** While the comparisons to existing methods are extensive, a more rigorous analysis of the differences in the underlying architectures and training methodologies of the baselines would strengthen the evaluation.  Are the differences in performance solely due to the novel aspects of GATE or partly due to variations in the implementation details of the baselines?
* **Ablation Study Depth:** While an ablation study is provided, a more in-depth analysis of each component and their interaction within the system would be beneficial.  The interaction effects between components (e.g., the relationship between GraphRank and Tool Merging) are not fully explored.



Despite these limitations, GATE offers a significant advancement in LLM tool management. The framework is well-designed, the experimental results are impressive, and the code release fosters reproducibility and future research.  The novelty lies in the systematic approach to tool evolution and reuse, facilitated by the graph structure and the two-agent interaction. The potential impact is significant as it could lead to more efficient and robust LLM-based agents.

Score: 8

**Rationale:** The score of 8 reflects a strong contribution with significant novelty and impact. The limitations mentioned above prevent it from achieving a higher score.  The scope of generalization, while demonstrated across several task types, needs to be expanded.  A more thorough analysis of computational cost and a deeper ablation study would strengthen the paper’s robustness.  Nevertheless, the core contributions of GATE are substantial and likely to influence the direction of future research in LLM tool management.

- **Score**: 8/10

## Other Papers
### **[A Chain-of-Thought Subspace Meta-Learning for Few-shot Image Captioning with Large Vision and Language Models](http://arxiv.org/abs/2502.13942v1)**
### **[Why Safeguarded Ships Run Aground? Aligned Large Language Models' Safety Mechanisms Tend to Be Anchored in The Template Region](http://arxiv.org/abs/2502.13946v1)**
### **[IP-Composer: Semantic Composition of Visual Concepts](http://arxiv.org/abs/2502.13951v1)**
### **[Neurosymbolic artificial intelligence via large language models and coherence-driven inference](http://arxiv.org/abs/2502.13953v1)**
### **[LIDDIA: Language-based Intelligent Drug Discovery Agent](http://arxiv.org/abs/2502.13959v1)**
### **[Is That Your Final Answer? Test-Time Scaling Improves Selective Question Answering](http://arxiv.org/abs/2502.13962v1)**
### **[MuDAF: Long-Context Multi-Document Attention Focusing through Contrastive Learning on Attention Heads](http://arxiv.org/abs/2502.13963v1)**
### **[Where's the Bug? Attention Probing for Scalable Fault Localization](http://arxiv.org/abs/2502.13966v2)**
### **[FlexTok: Resampling Images into 1D Token Sequences of Flexible Length](http://arxiv.org/abs/2502.13967v1)**
### **[DiffSampling: Enhancing Diversity and Accuracy in Neural Text Generation](http://arxiv.org/abs/2502.14037v1)**
### **[Diversity-driven Data Selection for Language Model Tuning through Sparse Autoencoder](http://arxiv.org/abs/2502.14050v1)**
### **[RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression](http://arxiv.org/abs/2502.14051v1)**
### **[A Matter of Perspective(s): Contrasting Human and LLM Argumentation in Subjective Decision-Making on Subtle Sexism](http://arxiv.org/abs/2502.14052v1)**
### **[DiffExp: Efficient Exploration in Reward Fine-tuning for Text-to-Image Diffusion Models](http://arxiv.org/abs/2502.14070v1)**
### **[Investigating Non-Transitivity in LLM-as-a-Judge](http://arxiv.org/abs/2502.14074v1)**
### **[Are Rules Meant to be Broken? Understanding Multilingual Moral Reasoning as a Computational Pipeline with UniMoral](http://arxiv.org/abs/2502.14083v1)**
### **[Navigating Semantic Relations: Challenges for Language Models in Abstract Common-Sense Reasoning](http://arxiv.org/abs/2502.14086v1)**
### **[Towards Context-Robust LLMs: A Gated Representation Fine-tuning Approach](http://arxiv.org/abs/2502.14100v1)**
### **[Benchmarking LLMs for Political Science: A United Nations Perspective](http://arxiv.org/abs/2502.14122v1)**
### **[Self-Regularization with Latent Space Explanations for Controllable LLM-based Classification](http://arxiv.org/abs/2502.14133v1)**
### **[Collaborative Retrieval for Large Language Model-based Conversational Recommender Systems](http://arxiv.org/abs/2502.14137v1)**
### **[Token Adaptation via Side Graph Convolution for Temporally and Spatially Efficient Fine-tuning of 3D Point Cloud Transformers](http://arxiv.org/abs/2502.14142v1)**
### **[Giving AI Personalities Leads to More Human-Like Reasoning](http://arxiv.org/abs/2502.14155v1)**
### **[Blockchain-based Framework for Scalable and Incentivized Federated Learning](http://arxiv.org/abs/2502.14170v1)**
### **[Enhancing Conversational Agents with Theory of Mind: Aligning Beliefs, Desires, and Intentions for Human-Like Interaction](http://arxiv.org/abs/2502.14171v1)**
### **[On the logical skills of large language models: evaluations using arbitrarily complex first-order logic problems](http://arxiv.org/abs/2502.14180v1)**
### **[Multi-Faceted Studies on Data Poisoning can Advance LLM Development](http://arxiv.org/abs/2502.14182v1)**
### **[Federated Fine-Tuning of Large Language Models: Kahneman-Tversky vs. Direct Preference Optimization](http://arxiv.org/abs/2502.14187v1)**
### **[QUAD-LLM-MLTC: Large Language Models Ensemble Learning for Healthcare Text Multi-Label Classification](http://arxiv.org/abs/2502.14189v1)**
### **[NLP-AKG: Few-Shot Construction of NLP Academic Knowledge Graph Based on LLM](http://arxiv.org/abs/2502.14192v1)**
### **[On-the-fly Preference Alignment via Principle-Guided Decoding](http://arxiv.org/abs/2502.14204v1)**
### **[Transfer-Prompting: Enhancing Cross-Task Adaptation in Large Language Models via Dual-Stage Prompts Optimization](http://arxiv.org/abs/2502.14211v1)**
### **[Less is More: On the Importance of Data Quality for Unit Test Generation](http://arxiv.org/abs/2502.14212v1)**
### **[Towards Secure Program Partitioning for Smart Contracts with LLM's In-Context Learning](http://arxiv.org/abs/2502.14215v1)**
### **[Investigating the Impact of LLM Personality on Cognitive Bias Manifestation in Automated Decision-Making Tasks](http://arxiv.org/abs/2502.14219v1)**
### **[Designing Parameter and Compute Efficient Diffusion Transformers using Distillation](http://arxiv.org/abs/2502.14226v1)**
### **[Mitigating Lost-in-Retrieval Problems in Retrieval Augmented Multi-Hop Question Answering](http://arxiv.org/abs/2502.14245v1)**
### **[Mem2Ego: Empowering Vision-Language Models with Global-to-Ego Memory for Long-Horizon Embodied Navigation](http://arxiv.org/abs/2502.14254v1)**
### **[Effects of Prompt Length on Domain-specific Tasks for Large Language Models](http://arxiv.org/abs/2502.14255v1)**
### **[LabTOP: A Unified Model for Lab Test Outcome Prediction on Electronic Health Records](http://arxiv.org/abs/2502.14259v1)**
### **[MCQA-Eval: Efficient Confidence Evaluation in NLG with Gold-Standard Correctness Labels](http://arxiv.org/abs/2502.14268v1)**
### **[PaperHelper: Knowledge-Based LLM QA Paper Reading Assistant](http://arxiv.org/abs/2502.14271v1)**
### **[Capturing Nuanced Preferences: Preference-Aligned Distillation for Small Language Models](http://arxiv.org/abs/2502.14272v1)**
### **[LLM-EvRep: Learning an LLM-Compatible Event Representation Using a Self-Supervised Framework](http://arxiv.org/abs/2502.14273v1)**
### **[Fact or Guesswork? Evaluating Large Language Model's Medical Knowledge with Structured One-Hop Judgment](http://arxiv.org/abs/2502.14275v1)**
### **[EpMAN: Episodic Memory AttentioN for Generalizing to Longer Contexts](http://arxiv.org/abs/2502.14280v1)**
### **[Vulnerability of Text-to-Image Models to Prompt Template Stealing: A Differential Evolution Approach](http://arxiv.org/abs/2502.14285v1)**
### **[Drift: Decoding-time Personalized Alignments with Implicit User Preferences](http://arxiv.org/abs/2502.14289v1)**
### **[SEA-HELM: Southeast Asian Holistic Evaluation of Language Models](http://arxiv.org/abs/2502.14301v1)**
### **[MedHallu: A Comprehensive Benchmark for Detecting Medical Hallucinations in Large Language Models](http://arxiv.org/abs/2502.14302v1)**
### **[Efficient AI in Practice: Training and Deployment of Efficient LLMs for Industry Applications](http://arxiv.org/abs/2502.14305v1)**
### **[Unveiling Cultural Blind Spots: Analyzing the Limitations of mLLMs in Procedural Text Comprehension](http://arxiv.org/abs/2502.14315v1)**
### **[Textured 3D Regenerative Morphing with 3D Diffusion Prior](http://arxiv.org/abs/2502.14316v1)**
### **[ParallelComp: Parallel Long-Context Compressor for Length Extrapolation](http://arxiv.org/abs/2502.14317v1)**
### **[Line Goes Up? Inherent Limitations of Benchmarks for Evaluating Large Language Models](http://arxiv.org/abs/2502.14318v1)**
### **[Beyond Self-Talk: A Communication-Centric Survey of LLM-Based Multi-Agent Systems](http://arxiv.org/abs/2502.14321v1)**
### **[ChemHTS: Hierarchical Tool Stacking for Enhancing Chemical Agents](http://arxiv.org/abs/2502.14327v1)**
### **[SolSearch: An LLM-Driven Framework for Efficient SAT-Solving Code Generation](http://arxiv.org/abs/2502.14328v1)**
### **[A Survey on Feedback-based Multi-step Reasoning for Large Language Models on Mathematics](http://arxiv.org/abs/2502.14333v1)**
### **[Earlier Tokens Contribute More: Learning Direct Preference Optimization From Temporal Decay Perspective](http://arxiv.org/abs/2502.14340v1)**
### **[FlowAgent: Achieving Compliance and Flexibility for Workflow Agents](http://arxiv.org/abs/2502.14345v1)**
### **[SR-LLM: Rethinking the Structured Representation in Large Language Model](http://arxiv.org/abs/2502.14352v1)**
### **[Retrieval-Augmented Process Reward Model for Generalizable Mathematical Reasoning](http://arxiv.org/abs/2502.14361v1)**
### **[RelaCtrl: Relevance-Guided Efficient Control for Diffusion Transformers](http://arxiv.org/abs/2502.14377v1)**
### **[S*: Test Time Scaling for Code Generation](http://arxiv.org/abs/2502.14382v1)**
### **[Leveraging Small LLMs for Argument Mining in Education: Argument Component Identification, Classification, and Assessment](http://arxiv.org/abs/2502.14389v1)**
### **[Unstructured Evidence Attribution for Long Context Query Focused Summarization](http://arxiv.org/abs/2502.14409v1)**
### **[Towards Efficient Automatic Self-Pruning of Large Language Models](http://arxiv.org/abs/2502.14413v1)**
### **[ChatVLA: Unified Multimodal Understanding and Robot Control with Vision-Language-Action Model](http://arxiv.org/abs/2502.14420v1)**
### **[A Survey on Data Contamination for Large Language Models](http://arxiv.org/abs/2502.14425v1)**
### **[Token-Level Density-Based Uncertainty Quantification Methods for Eliciting Truthfulness of Large Language Models](http://arxiv.org/abs/2502.14427v1)**
### **[PredictaBoard: Benchmarking LLM Score Predictability](http://arxiv.org/abs/2502.14445v1)**
### **[LLM4FaaS: No-Code Application Development using LLMs and FaaS](http://arxiv.org/abs/2502.14450v1)**
### **[Optimal word order for non-causal text generation with Large Language Models: the Spanish case](http://arxiv.org/abs/2502.14451v1)**
### **[Narrative-Driven Travel Planning: Geoculturally-Grounded Script Generation with Evolutionary Itinerary Optimization](http://arxiv.org/abs/2502.14456v1)**
### **[Llamba: Scaling Distilled Recurrent Models for Efficient Language Processing](http://arxiv.org/abs/2502.14458v1)**
### **[Enhancing Smart Environments with Context-Aware Chatbots using Large Language Models](http://arxiv.org/abs/2502.14469v1)**
### **[Argument-Based Comparative Question Answering Evaluation Benchmark](http://arxiv.org/abs/2502.14476v1)**
### **[Unshackling Context Length: An Efficient Selective Attention Approach through Query-Key Compression](http://arxiv.org/abs/2502.14477v1)**
### **[NLoRA: Nyström-Initiated Low-Rank Adaptation for Large Language Models](http://arxiv.org/abs/2502.14482v1)**
### **[StructFlowBench: A Structured Flow Benchmark for Multi-turn Instruction Following](http://arxiv.org/abs/2502.14494v1)**
### **[MLGym: A New Framework and Benchmark for Advancing AI Research Agents](http://arxiv.org/abs/2502.14499v1)**
### **[How Much Knowledge Can You Pack into a LoRA Adapter without Harming LLM?](http://arxiv.org/abs/2502.14502v1)**
### **[Can LLMs Simulate L2-English Dialogue? An Information-Theoretic Analysis of L1-Dependent Biases](http://arxiv.org/abs/2502.14507v1)**
### **[Generative adversarial networks vs large language models: a comparative study on synthetic tabular data generation](http://arxiv.org/abs/2502.14523v1)**
### **[CORBA: Contagious Recursive Blocking Attacks on Multi-Agent Systems Based on Large Language Models](http://arxiv.org/abs/2502.14529v1)**
### **[LoRA-GGPO: Mitigating Double Descent in LoRA Fine-Tuning via Gradient-Guided Perturbation Optimization](http://arxiv.org/abs/2502.14538v1)**
### **[LLM-based User Profile Management for Recommender System](http://arxiv.org/abs/2502.14541v1)**
### **[Less is More: Improving LLM Alignment via Preference Data Selection](http://arxiv.org/abs/2502.14560v1)**
### **[Can LLMs Predict Citation Intent? An Experimental Analysis of In-context Learning and Fine-tuning on Open LLMs](http://arxiv.org/abs/2502.14561v1)**
### **[Plan-over-Graph: Towards Parallelable LLM Agent Schedule](http://arxiv.org/abs/2502.14563v1)**
### **[ReVISE: Learning to Refine at Test-Time via Intrinsic Self-Verification](http://arxiv.org/abs/2502.14565v1)**
### **[Vision Foundation Models in Medical Image Analysis: Advances and Challenges](http://arxiv.org/abs/2502.14584v1)**
### **["Don't Forget the Teachers": Towards an Educator-Centered Understanding of Harms from Large Language Models in Education](http://arxiv.org/abs/2502.14592v1)**
### **[Behavioral Analysis of Information Salience in Large Language Models](http://arxiv.org/abs/2502.14613v1)**
### **[FIND: Fine-grained Information Density Guided Adaptive Retrieval-Augmented Generation for Disease Diagnosis](http://arxiv.org/abs/2502.14614v1)**
### **[Reward Models Identify Consistency, Not Causality](http://arxiv.org/abs/2502.14619v1)**
### **[Partial Incorrectness Logic](http://arxiv.org/abs/2502.14626v1)**
### **[PEARL: Towards Permutation-Resilient LLMs](http://arxiv.org/abs/2502.14628v1)**
### **[Synergistic Fusion of Multi-Source Knowledge via Evidence Theory for High-Entropy Alloy Discovery](http://arxiv.org/abs/2502.14631v1)**
### **[Augmenting Coaching with GenAI: Insights into Use, Effectiveness, and Future Potential](http://arxiv.org/abs/2502.14632v1)**
### **[CER: Confidence Enhanced Reasoning in LLMs](http://arxiv.org/abs/2502.14634v1)**
### **[Length-Controlled Margin-Based Preference Optimization without Reference Model](http://arxiv.org/abs/2502.14643v1)**
### **[LIFT: Improving Long Context Understanding of Large Language Models through Long Input Fine-Tuning](http://arxiv.org/abs/2502.14644v1)**
### **[Edit Once, Update Everywhere: A Simple Framework for Cross-Lingual Knowledge Synchronization in LLMs](http://arxiv.org/abs/2502.14645v1)**
### **[Beyond the Surface: Uncovering Implicit Locations with LLMs for Personalized Local News](http://arxiv.org/abs/2502.14660v1)**
### **[AlphaMaze: Enhancing Large Language Models' Spatial Intelligence via GRPO](http://arxiv.org/abs/2502.14669v1)**
### **[Explanations of Deep Language Models Explain Language Representations in the Brain](http://arxiv.org/abs/2502.14671v1)**
### **[Data-Constrained Synthesis of Training Data for De-Identification](http://arxiv.org/abs/2502.14677v1)**
### **[How to Get Your LLM to Generate Challenging Problems for Evaluation](http://arxiv.org/abs/2502.14678v1)**
### **[Bridging the Gap: Transforming Natural Language Questions into SQL Queries via Abstract Query Pattern and Contextual Schema Markup](http://arxiv.org/abs/2502.14682v1)**
### **[I-MCTS: Enhancing Agentic AutoML via Introspective Monte Carlo Tree Search](http://arxiv.org/abs/2502.14693v1)**
### **[TRUSWorthy: Toward Clinically Applicable Deep Learning for Confident Detection of Prostate Cancer in Micro-Ultrasound](http://arxiv.org/abs/2502.14707v1)**
### **[Entity Framing and Role Portrayal in the News](http://arxiv.org/abs/2502.14718v1)**
### **[WavRAG: Audio-Integrated Retrieval Augmented Generation for Spoken Dialogue Models](http://arxiv.org/abs/2502.14727v1)**
### **[EAGER-LLM: Enhancing Large Language Models as Recommenders through Exogenous Behavior-Semantic Integration](http://arxiv.org/abs/2502.14735v1)**
### **[SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines](http://arxiv.org/abs/2502.14739v1)**
### **[Multi-Agent Coordination across Diverse Applications: A Survey](http://arxiv.org/abs/2502.14743v1)**
### **[AIdeation: Designing a Human-AI Collaborative Ideation System for Concept Designers](http://arxiv.org/abs/2502.14747v1)**
### **[Large Language Models Struggle to Describe the Haystack without Human Help: Human-in-the-loop Evaluation of LLMs](http://arxiv.org/abs/2502.14748v1)**
### **[TritonBench: Benchmarking Large Language Model Capabilities for Generating Triton Operators](http://arxiv.org/abs/2502.14752v1)**
### **[On the Influence of Context Size and Model Choice in Retrieval-Augmented Generation Systems](http://arxiv.org/abs/2502.14759v1)**
### **[EquivaMap: Leveraging LLMs for Automatic Equivalence Checking of Optimization Formulations](http://arxiv.org/abs/2502.14760v1)**
### **[Tree-of-Debate: Multi-Persona Debate Trees Elicit Critical Thinking for Scientific Comparative Analysis](http://arxiv.org/abs/2502.14767v1)**
### **[Determining Layer-wise Sparsity for Large Language Models Through a Theoretical Perspective](http://arxiv.org/abs/2502.14770v1)**
### **[SurveyX: Academic Survey Automation via Large Language Models](http://arxiv.org/abs/2502.14776v1)**
### **[DC-ControlNet: Decoupling Inter- and Intra-Element Conditions in Image Generation with Diffusion Models](http://arxiv.org/abs/2502.14779v1)**
### **[A Multi-Agent Perspective on Modern Information Retrieval](http://arxiv.org/abs/2502.14796v1)**
### **[A Survey on Text-Driven 360-Degree Panorama Generation](http://arxiv.org/abs/2502.14799v1)**
### **[From RAG to Memory: Non-Parametric Continual Learning for Large Language Models](http://arxiv.org/abs/2502.14802v1)**
### **[Dynamic Low-Rank Sparse Adaptation for Large Language Models](http://arxiv.org/abs/2502.14816v1)**
### **[eC-Tab2Text: Aspect-Based Text Generation from e-Commerce Product Tables](http://arxiv.org/abs/2502.14820v1)**
### **[A Survey of Model Architectures in Information Retrieval](http://arxiv.org/abs/2502.14822v1)**
### **[Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs](http://arxiv.org/abs/2502.14830v1)**
### **[Improving the Diffusability of Autoencoders](http://arxiv.org/abs/2502.14831v1)**
### **[Revealing and Mitigating Over-Attention in Knowledge Editing](http://arxiv.org/abs/2502.14838v1)**
### **[Dynamic Concepts Personalization from Single Videos](http://arxiv.org/abs/2502.14844v1)**
### **[Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation](http://arxiv.org/abs/2502.14846v1)**
### **[GATE: Graph-based Adaptive Tool Evolution Across Diverse Tasks](http://arxiv.org/abs/2502.14848v1)**
