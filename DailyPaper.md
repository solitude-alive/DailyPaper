# The Latest Daily Papers - Date: 2025-06-21
## Highlight Papers
### **[Align Your Flow: Scaling Continuous-Time Flow Map Distillation](http://arxiv.org/abs/2506.14603v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Align Your Flow: Scaling Continuous-Time Flow Map Distillation":

**Summary:**

The paper introduces "Align Your Flow" (AYF), a novel distillation method for training flow maps. Flow maps generalize diffusion and consistency models, allowing for efficient generation with any number of steps. AYF contributes two new continuous-time objectives and stabilization techniques, enabling the distillation of autoguided teacher models for improved performance.  The authors demonstrate that standard consistency models inherently suffer from error accumulation during multi-step sampling, a limitation that AYF overcomes.  AYF achieves state-of-the-art few-step generation performance on ImageNet benchmarks and scales to high-resolution text-to-image generation, outperforming existing non-adversarial approaches. The method also employs adversarial fine-tuning for added sharpness with minimal impact on sample diversity.

**Critical Evaluation:**

*   **Novelty:** The paper offers significant novelty in several aspects:
    *   **Theoretical Contribution:** The analytical demonstration of the limitations of consistency models in multi-step sampling is a solid theoretical contribution that motivates the flow map approach. It provides a formal understanding of an observed empirical problem.
    *   **Methodological Advancements:** The introduction of two new continuous-time objectives (AYF-EMD and AYF-LMD) for flow map training, along with stabilization techniques, expands the toolkit for generative modeling. The generalization of existing consistency and flow matching objectives within the EMD framework is valuable.
    *   **Autoguidance Distillation:** The application of autoguidance to teacher model distillation is a novel way to improve performance, offering an alternative to traditional classifier-free guidance and adversarial training.
    *   **Focus on Flow maps:** While flow maps themselves are not entirely new, this paper showcases their practical superiority at scale vs distilled diffusion and consistency models in the efficient regime.

*   **Significance:**  The paper has the potential to significantly impact the field of generative modeling.

    *   **Improved Efficiency:** AYF addresses a critical bottleneck in generative models: slow sampling. By achieving state-of-the-art performance in few-step generation, AYF makes generative models more practical for real-time applications and reduces computational costs.
    *   **Broader Applicability:** The ability to train effective flow maps without relying on adversarial training offers a more stable and controllable approach, potentially leading to broader adoption. The scalability of AYF to high-resolution text-to-image synthesis is also significant.
    *   **Overcoming Limitations of CMs:** The analytical and empirical demonstration of the shortcomings of multi-step consistency models is an important contribution, guiding future research directions towards more robust alternatives like flow maps.
*   **Strengths:**
    *   The paper is well-written and clearly explains complex concepts.
    *   The theoretical analysis is rigorous and provides valuable insights.
    *   The experimental evaluation is comprehensive, covering a range of benchmarks and ablation studies.
    *   The results demonstrate a clear improvement over existing methods, particularly in the few-step generation setting.
    *   Open-source implementation is expected to boost adoption and impact.

*   **Weaknesses:**
    *   While autoguidance is effective, it introduces an additional component that might add complexity to the training process and may not be suitable for all scenarios (e.g., FLUX.1 requires distillation rather than having its own autoguiding system).
    *   The paper acknowledges that AYF, in some configurations, leads to slightly degraded single-step performance compared to methods explicitly designed for one-step generation. While adversarial finetuning mitigates this, it would be desirable to achieve optimal performance across all step counts without needing to add a GAN objective.
    *   The performance of LMD objective on image datasets can be improved.

*   **Justification of Score:**
I am assigning a score of **8.5** to this paper.

*   The **theoretical contribution** about *Consistency Model's limitations* is novel and valuable. The new **method** "Align Your Flow" achieves state-of-the-art or near state-of-the-art performance in few-step generative modeling. It **scales** well to large datasets. The code is or will be publicly available, **increasing impact**. The **limitations** section provides open points to guide follow-up works.

The paper represents a significant advancement in generative modeling, addressing a critical challenge (sampling efficiency) with a theoretically sound and empirically validated approach.
Score: 8.5

- **Score**: 8/10

### **[AIn't Nothing But a Survey? Using Large Language Models for Coding German Open-Ended Survey Responses on Survey Motivation](http://arxiv.org/abs/2506.14634v2)**
- **Summary**: Here's a summary and critical evaluation of the paper "Ain't Nothing But a Survey? Using Large Language Models for Coding German Open-Ended Survey Responses on Survey Motivation":

**Summary:**

This paper explores the application of large language models (LLMs) to code German-language open-ended survey responses, specifically focusing on reasons for survey participation. The authors compare several state-of-the-art LLMs (GPT, Llama, Mistral) using different prompting techniques (zero-shot, few-shot, fine-tuning). They benchmark the LLMs' performance against human expert codings, analyzing accuracy and reliability.  The findings reveal significant performance differences between LLMs, with fine-tuning yielding the best results.  The choice of prompting approach also matters, although its impact is conditional on the LLM used. The study highlights that LLMs struggle with non-substantive, catch-all categories and uneven classification performance can lead to distorted category distributions compared to human-coded data. The authors discuss the trade-offs associated with using LLMs (cost, privacy, accuracy, reliability) and emphasize the importance of validating LLM-generated classifications.

**Critical Evaluation:**

*   **Strengths:**

    *   **Rigorous Methodology:** The paper employs a comprehensive methodology, comparing multiple LLMs, prompting strategies, and performance metrics, all benchmarked against human coders. This allows for robust and nuanced conclusions.
    *   **Focus on a Specific and Challenging Context:** The study addresses a crucial gap in the literature by examining LLMs in a non-English language (German) and on a specific, complex topic (survey motivation). Prior research has often focused on simpler tasks and English-language data.
    *   **Detailed Analysis:** The paper delves deeply into the performance of LLMs across different categories and the resulting impact on categorical distributions. This is a significant contribution that highlights potential biases and limitations of using LLMs.
    *   **Practical Implications:** The discussion of trade-offs associated with LLM use (cost, privacy, accuracy, reliability) is valuable for researchers and practitioners considering automated methods for open-ended response classification.
    *   **Addresses Reproducibility Concerns:** Recognizes and attempts to mitigate issues of LLM result reproducibility by using a deterministic setup and acknowledging the potential for model drift over time.

*   **Weaknesses:**

    *   **Limited Fine-Tuning Scope:** Fine-tuning was only conducted on GPT due to resource constraints. This limits the ability to directly compare the performance of fine-tuned LLMs across different model families. However, the authors acknowledge this limitation and provide a solid rationale.
    *   **Topic Specificity:** While a strength, the focus on survey motivation limits the generalizability of the findings to other types of open-ended survey questions. The topic itself may not be broadly applicable to other social science domains.
    *   **Static Benchmark:** The study relies on human-coded data as a static benchmark, which may not fully capture the nuances and complexities of open-ended responses. While high inter-coder reliability helps, it doesn't eliminate the possibility of inherent biases in the human codings.
    *   **Lack of Comparison to Other Methods:** Although the authors compare LLM performance to prior research on SVMs, a direct, side-by-side comparison of LLMs and other supervised learning techniques on the same dataset would strengthen the analysis.

*   **Novelty and Significance:**

    *   The paper is novel in its rigorous evaluation of multiple LLMs and prompting strategies in a non-English language and on a specific, challenging topic.
    *   The findings have significant implications for researchers and practitioners considering using LLMs for coding open-ended survey responses. The study provides valuable insights into the potential benefits and limitations of this approach and highlights the importance of careful validation and awareness of potential biases.
    *   The study contributes to the growing body of research on the applicability of LLMs in survey research and their impact on data quality. It extends existing knowledge by addressing a critical gap in the literature.

**Justification for Score:**

The paper represents a valuable contribution to the field of survey methodology and the application of LLMs in social science research. The rigorous methodology, detailed analysis, and practical implications justify a high score. The limitations, while present, are acknowledged and do not detract significantly from the overall value of the study.

Score: 8

- **Score**: 8/10

### **[GuiLoMo: Allocating Expert Number and Rank for LoRA-MoE via Bilevel Optimization with GuidedSelection Vectors](http://arxiv.org/abs/2506.14646v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "GuiLoMo: Allocating Expert Number and Rank for LoRA-MoE via Bilevel Optimization with GuidedSelection Vectors":

**Summary:**

The paper addresses limitations in LoRA-MoE (Low-Rank Adaptation combined with Mixture-of-Experts), a parameter-efficient fine-tuning technique for large language models. It identifies two key issues: 1) downstream tasks are not sufficiently considered when assigning expert numbers, and 2) uniform rank assignments for all LoRA experts limit representational diversity. To overcome these, the paper introduces GuiLoMo, a fine-grained strategy for allocating expert numbers and ranks based on bilevel optimization with GuidedSelection Vectors (GSVs). GSVs are learned via a prior bilevel optimization process to capture both model- and task-specific needs, guiding the allocation of optimal expert numbers and ranks.  Experiments across various models and benchmarks demonstrate that GuiLoMo consistently achieves superior or comparable performance to baselines. The authors provide insights into how expert numbers and ranks vary across layers and tasks.

**Critical Evaluation:**

* **Novelty:** The core idea of using bilevel optimization and GuidedSelection Vectors to dynamically allocate *both* expert numbers *and* ranks in LoRA-MoE is a significant and valuable contribution.  Previous work focused primarily on allocating the *number* of experts but did not jointly consider the rank, thereby limiting the capacity of the LoRA-MoE to effectively handle diverse tasks.  The recognition that different tasks and layers benefit from different rank assignments is a key insight.  The design of the GSVs and their integration into the bilevel optimization process represents technical novelty.

* **Significance:**  Parameter-efficient fine-tuning is a vital area of research, particularly for large language models. Enhancing LoRA-MoE's capability and adaptability makes it more practical and effective.  The improved performance demonstrated by GuiLoMo across multiple benchmarks suggests a real advancement. The insights provided into layer- and task-specific expert allocation can inform future research and practical applications. The code availability is also significant as it allows for reproducibility and further investigations by other researchers.

* **Strengths:**
    * **Problem Definition:** The paper clearly identifies and articulates the limitations of existing LoRA-MoE approaches.
    * **Technical Approach:** The bilevel optimization and GSV framework is well-motivated and technically sound.
    * **Empirical Validation:** Extensive experiments across various tasks (NLU, QA, Mathematical Reasoning) and models (LLaMA-7B, LLaMA-2-7B, LLaMA-3-8B, Mistral-v0.17B, and even LLaMA-2-13B) provide strong evidence for the effectiveness of GuiLoMo.
    * **Ablation Study and Analysis:** Ablation studies demonstrating the importance of jointly optimizing expert numbers and ranks as well as the analysis of layer- and task-specific allocations, strengthens the credibility of the results. The task difficulty insight is also interesting.

* **Weaknesses:**
    * **Computational Cost:** The bilevel optimization process for obtaining GSVs may add significant computational overhead, which might be a barrier to adoption, although the authors do not mention this specifically, it needs to be carefully considered.
    * **Scalability to even larger models:** While the paper provides results up to LLaMA-2-13B, it would be valuable to see how GuiLoMo performs on even larger models (70B+ parameters), which are increasingly relevant in real-world applications.
    * **Generalization to other modalities:** the research is limited to NLP tasks and should have a conclusion that shows if the framework generalizes to other modalities.

* **Potential Influence:**  The paper has the potential to significantly influence the development of parameter-efficient fine-tuning methods. The insights and techniques introduced in GuiLoMo could be adopted and extended by other researchers to create even more adaptable and efficient LoRA-MoE variants. The code release should facilitate widespread adoption and further research.

* **Justification for Score:** The paper makes a novel technical contribution with its bilevel optimization and GuidedSelection Vectors approach to allocate both expert numbers and rank. It provides substantial empirical validation and analysis, and addresses a relevant issue (parameter-efficient tuning) in the field of LLMs. Despite the limitations related to computational cost and scalability to the largest models, the paper has strong potential for impact.

Score: 8

- **Score**: 8/10

### **[AIRTBench: Measuring Autonomous AI Red Teaming Capabilities in Language Models](http://arxiv.org/abs/2506.14682v1)**
- **Summary**: Here's a summary and critical evaluation of the AIRTBench paper:

**Summary:**

The paper introduces AIRTBench, a new benchmark specifically designed to evaluate the autonomous AI red teaming capabilities of language models (LLMs). The benchmark comprises 70 realistic capture-the-flag (CTF) challenges derived from the Dreadnode platform's Crucible environment. These challenges necessitate that the LLMs write Python code to interact with and compromise AI systems, covering various attack vectors such as prompt injection, model inversion, and system exploitation. The authors evaluated several frontier and open-source LLMs, finding that Claude-3.7-Sonnet achieved the highest success rate. The paper highlights the significant efficiency gains of LLMs over human red teamers and identifies areas where LLMs excel (prompt injection) and struggle (system exploitation, model inversion). The authors open-sourced their evaluation tools and dataset, aiming to foster community-driven development in AI security benchmarking.

**Critical Evaluation:**

**Novelty:** The paper presents a novel contribution by providing a comprehensive benchmark designed specifically for evaluating autonomous AI red teaming capabilities of LLMs. While existing benchmarks touch on AI security, AIRTBench's focus on red teaming and its CTF-style challenges are a significant step forward. However, the challenge dataset is derived from a proprietary platform, which could limit accessibility and wider adoption by the research community without a Dreadnode license. The novelty also hinges on the design of challenges being truly representative of real-world AI security threats, which is claimed but needs more justification.

**Significance:** The work addresses a crucial gap in the AI security landscape. As LLMs become increasingly integrated into various applications, the ability to automatically assess their vulnerabilities becomes paramount. AIRTBench provides a standardized framework for this evaluation, allowing for comparisons between different models and tracking progress over time. The demonstration of significant efficiency gains over human red teamers underscores the potential for LLMs to revolutionize security testing, both ethically and maliciously. The open-sourcing of the benchmark and data is a crucial step towards widespread adoption and community-driven improvement, significantly increasing the impact of the paper.

**Strengths:**
*   **Focus on a critical and emerging area:** Red teaming LLMs is a timely and important topic.
*   **Comprehensive benchmark design:** The paper creates a new benchmark suite for CTFs.
*   **Demonstrates Efficiency over Humans**: Highlights the time savings that LLMs can bring to security testing.
*   **Empirical evaluation:** The paper evaluates several LLMs and provides valuable insights into their strengths and weaknesses.
*   **Open-source contribution:** The authors make their tools and data available, fostering collaboration and further research.

**Weaknesses:**
*   **Challenge dataset reliance:** Reliance on challenges derived from a proprietary platform limits accessibility without a Dreadnode license. While authors stated they tested across 70 challenges with most being publicly available, explicit separation of public vs private sets would be welcome.
*   **Limited baseline comparison:** While comparing to humans, there isn’t a dedicated baseline with human results solving the challenges to directly compare.
*   **Potentially narrow scope:** The CTF-style challenges, while realistic, may not fully capture the nuances and complexities of real-world AI security threats.
*   **Limited discussion of ethical implications:** While the paper touches on the potential for malicious use of LLMs, it could benefit from a more in-depth discussion of the ethical implications of autonomous AI red teaming and potential mitigation strategies.

**Potential Influence:** AIRTBench has the potential to become a widely adopted benchmark for evaluating AI red teaming capabilities. It can guide the development of more secure LLMs, inform defensive strategies, and facilitate research into novel attack techniques. Widespread use will enable robust tracking of progress and provide a common ground for evaluating security enhancements over time.

**Justification for Score:**

The AIRTBench paper is a significant contribution to the field of AI security. While the dependence on a proprietary platform and challenges does limit access, and more dedicated human baseline data would strengthen the study, the comprehensive benchmark design, empirical evaluation, and open-source commitment are strong positives. It addresses a timely and critical need for standardized evaluation in the rapidly evolving landscape of AI security and will likely influence future research in the field.

**Score: 8**

- **Score**: 8/10

### **[DETONATE: A Benchmark for Text-to-Image Alignment and Kernelized Direct Preference Optimization](http://arxiv.org/abs/2506.14903v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces "DPO-Kernels," a novel extension of Direct Preference Optimization (DPO) designed to improve text-to-image (T2I) alignment. The core idea is to embed alignment within a Reproducing Kernel Hilbert Space (RKHS), using kernels like Radial Basis Function (RBF), Polynomial, and Wavelet to enable more nuanced and semantically sensitive updates. The method also replaces the standard KL divergence with alternatives like Rényi and Wasserstein to enhance stability and robustness. The paper introduces "DETONATE," a new large-scale benchmark for T2I alignment focused on detecting social biases (Race, Gender, Disability).  Finally, it introduces the "Alignment Quality Index (AQI)," a metric that quantifies the geometric separability of safe and unsafe image activations within the model's latent space.  Experiments show that DPO-Kernels outperform existing alignment techniques.

**Critical Evaluation:**

*   **Strengths:**

    *   **Problem Significance:** Addressing the alignment crisis in T2I models is critically important given their potential for shaping public opinion and perpetuating biases.  The paper directly tackles a significant and growing problem.
    *   **Novelty of Approach:** Shifting from surface-level alignment fixes to structural regularization within the latent space is a conceptually strong and potentially more robust approach. The use of kernels within DPO is a novel and well-motivated idea to capture complex relationships in the representation space.
    *   **DETONATE Benchmark:** The creation of a large-scale, carefully curated benchmark focused on social biases is a major contribution. The dataset provides a valuable resource for the community. The construction process, including the keyword filtering and use of VLMs with human verification, increases its reliability.
    *   **AQI Metric:**  The introduction of the AQI provides a way to evaluate alignment fidelity *within* the model, rather than just relying on output-level classifications.  This is a significant step toward detecting "alignment faking" and ensuring genuine ethical behavior. The approach of using well-defined distance metrics for determining cluster separation and stability is sensible.
    *   **Empirical Validation:**  The paper presents strong empirical results demonstrating the effectiveness of DPO-Kernels over existing methods on multiple metrics and using different backbone models (SD-XL, SD v1.5).
    *   **Theoretical Justification:**  The authors provide a good theoretical justification for their approach, drawing on concepts from RKHS, heavy-tailed self-regularization, and related literature.

*   **Weaknesses:**

    *   **Computational Cost:** The increased computational cost of DPO-Kernels (reported as a 3-4x increase in training time) is a significant drawback. While the paper discusses potential optimizations, this remains a practical limitation that could hinder widespread adoption.
    *   **Hyperparameter Sensitivity:** The framework's sensitivity to hyperparameters like kernel bandwidth and polynomial degree is a concern, and the selection of optimal parameters needs to be a crucial part of the fine tuning process.
    *   **Kernel Choice Rationale:** While the paper explores different kernel choices, a stronger, more prescriptive guideline on *when* to choose each kernel would be helpful. Is there a theoretical basis for when each type is needed or would it involve empirical search only?
    *   **Limited Analysis of Failure Cases:** While the paper highlights successes, a more detailed examination of specific failure cases or types of prompts where DPO-Kernels still struggle would be valuable. Analyzing how AQI behaves in such cases could also provide further insight.
    *   **Overfitting Concerns:** Although the paper uses the Weighted Alpha metric, the reliance on highly expressive models increases the risk of overfitting.

*   **Significance:** The paper has the potential to significantly influence the field of T2I alignment. The emphasis on structural alignment, the new benchmark, and the AQI metric offer valuable tools and insights for future research. If the computational cost can be addressed, DPO-Kernels could become a standard alignment technique.

**Justification for the Score:**

The paper presents a compelling and well-executed approach to a critical problem. The DETONATE benchmark and AQI metric are important contributions. While the computational cost and certain hyperparameters remain concerns, the conceptual novelty, strong empirical results, and theoretical backing justify a high score. The emphasis on a holistic improvement by moving beyond symptomatic fixes to actual structural changes in the latent space of models has potential to set new directions in safety-oriented AI development. The score is brought down due to the practical limitation in computational cost and lingering concerns about the difficulty of robust and generalisable fine-tuning.

Score: 8

- **Score**: 8/10

### **[Hyper-Local Deformable Transformers for Text Spotting on Historical Maps](http://arxiv.org/abs/2506.15010v1)**
- **Summary**: Here is a summary and evaluation of the paper:

**Summary:**

This paper introduces PALETTE, a novel end-to-end text spotter designed specifically for scanned historical maps. PALETTE uses a hyper-local sampling module to learn localized image features around the boundary points and characters of text instances. The method also incorporates hyper-local positional embeddings to capture spatial interactions. Additionally, the paper presents SYNTHMAP+, a new approach for automatically generating synthetic map images for training. Experiments on new benchmark datasets of historical maps demonstrate PALETTE's superior performance, particularly for long and angled text, compared to state-of-the-art text spotters. The method has been deployed to process a large collection of historical maps.

**Critical Evaluation:**

*   **Novelty:** The key innovations of the paper are the hyper-local sampling and positional embeddings, which address limitations of existing deformable DETR-based methods when applied to historical maps. The hyper-local approach is more tailored for handling lengthy, curved, and rotated text common in these maps. The creation of the SYNTHMAP+ dataset is another significant contribution, as it tackles the lack of training data tailored for historical map styles. The iterative training procedure to leverage predicted character centers in the absence of ground truth character center annotations in real data is also a clever and valuable technique. While deformable DETR itself isn't new, its adaptation with these components and dataset for historical maps specifically makes the paper novel.

*   **Significance:** The ability to accurately extract text from historical maps has important implications for improving map searchability, metadata generation, and enabling research in various fields. The significant improvements reported over existing methods, especially on challenging text orientations and lengths, demonstrate the practical value of PALETTE. The release of the code, models, and datasets enhances reproducibility and promotes future research in this area.  The fact that this system has been deployed over 60,000 maps is significant, showing the capability of the method.

*   **Strengths:**

    *   The hyper-local sampling approach is well-motivated and effective in addressing the challenges of historical map text.
    *   The SYNTHMAP+ dataset fills a crucial gap in training resources for this domain.
    *   The iterative training approach alleviates the necessity of character center labels, thereby lessening data annotation costs.
    *   The experiments are thorough, comparing PALETTE to several state-of-the-art methods and including ablation studies to analyze the impact of individual components.
    *   The method has been deployed for processing large map collections.

*   **Weaknesses:**

    *   The paper mentions that the performance degrades when the text instances are extremely large and characters are far apart, implying the limitations of the method. The paper does not offer solutions and improvements for these issues.
    *   The method still depends on synthetic data and transfer learning. Although SYNTHMAP+ helps, further research might explore unsupervised or self-supervised approaches for historical maps to minimize reliance on labeled data.
    *   While the experiments are comprehensive, a qualitative analysis showing specific examples of failure cases and the impact of the various components could strengthen the analysis.

*   **Potential Influence:** PALETTE has a high potential to influence the field. It provides a strong baseline for future text spotting research on historical maps. The SYNTHMAP+ dataset can serve as a valuable resource for training and evaluating new methods. The demonstrated improvements in accuracy and robustness, along with the successful deployment, suggest that PALETTE can become a widely adopted tool for processing and analyzing historical map collections.

**Score: 8.5**

**Rationale:** The paper presents a significant advancement in text spotting for historical maps, effectively addressing the limitations of existing methods and creating valuable resources. The hyper-local sampling and positional embeddings, coupled with the SYNTHMAP+ dataset, demonstrate a clear understanding of the specific challenges posed by historical map text. The iterative training procedure to alleviate reliance on labeled character center locations is a smart solution for reducing annotation burdens. While further improvements are possible, the current results and practical deployment are impressive and indicative of a high-impact contribution. The weaknesses are limitations that can be addressed in future work but do not overshadow the current value of the contributions. The score reflects the significant value, novelty, and potential influence of this research.

- **Score**: 8/10

### **[Enhancement Report Approval Prediction: A Comparative Study of Large Language Models](http://arxiv.org/abs/2506.15098v1)**
- **Summary**: Okay, here's a summary and critical evaluation of the provided paper:

**Summary:**

The paper explores the effectiveness of Large Language Models (LLMs) in predicting the approval of software enhancement reports (ERAP). It compares 18 LLM variants (encoder-only like BERT and decoder-only like GPT) against traditional machine learning methods.  The study finds that incorporating creator profiles improves the performance of decoder-only models, and fine-tuning Llama 3.1 8B Instruct with LoRA achieves the best results, outperforming traditional methods in terms of accuracy and recall, particularly in addressing class imbalance. The paper also investigates cases where LLMs underperform, providing insights for future research directions. The key takeaway is that LLMs offer a superior solution for ERAP, streamlining software maintenance and improving decision-making.

**Critical Evaluation:**

**Novelty:** While the application of machine learning to ERAP isn't entirely new, *this paper is novel in its comprehensive and systematic evaluation of a wide range of LLMs on this task*. Previous works have explored feature-based or deep learning models, but the scale and comparative nature of this study, focusing on both encoder and decoder architectures and various fine-tuning techniques, makes it a significant contribution. It's one of the first to directly tackle ERAP utilizing a diverse family of LLMs. Furthermore, the detailed analysis of the error cases, and the explanation of *why* LLMs fail in specific scenarios related to API compatibility, UI/UX improvements, and technical aspects, provides valuable insights.

**Significance:** The paper's findings have significant implications for software engineering practice. Automating ERAP using LLMs promises:

*   **Increased efficiency:** Reduced manual effort in processing ERs.
*   **Better decision-making:** Faster and potentially more accurate assessments of ERs.
*   **Improved user engagement:**  Timely responses to user suggestions can boost user satisfaction.
*   **Cost savings:**  Reducing the time and resources devoted to manual review.

The paper also highlights important considerations regarding the use of LLMs:

*   **Data Leakage:** The strict chronological evaluation is vital, and the acknowledgment of the risk of data leakage if models are trained on data they are later evaluated on is crucial.
*   **Class Imbalance:** The paper directly tackles class imbalance issues, which is a frequent hurdle in machine learning.
*   **Bias:** The study acknowledges the potential for bias introduced by creator profiles and explores trade-offs related to it.

**Weaknesses:**

*   **Dataset Limitations:** While the dataset is significant, relying on a dataset spanning 1997-2016 raises concerns about its representativeness of current ER practices. The inclusion of more recent data would strengthen the conclusions. The dataset might contain information already present during the training stages of the evaluated LLMs.
*   **Limited exploration of ensemble methods:** Although the paper employs an ensemble (voting) method to address this it could explore other ensemble techniques, weighting schemes, or using a meta-learner for further improvement.
*   **Missing State of the Art Baselines:**  The paper omits some current state-of-the-art baselines.

**Justification for Score:**

The paper makes a valuable contribution by thoroughly evaluating LLMs for ERAP, addressing critical challenges like data leakage, class imbalance, and potential biases. It demonstrates LLMs outperform traditional methods, providing key insights into their error patterns and offering directions for future research. The results show promise and demonstrate that LLMs could effectively improve maintenance, saving time and resources.

Despite the dataset's age and the limited exploration of more sophisticated ensemble strategies, the paper's comprehensive analysis, novel application of LLMs to ERAP, and practical implications justify a high score. The rigorous evaluation protocol, including the chronological split, adds considerable strength to the study.

**Score: 8**

- **Score**: 8/10

### **[eLLM: Elastic Memory Management Framework for Efficient LLM Serving](http://arxiv.org/abs/2506.15155v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces eLLM, an elastic memory management framework designed to improve the efficiency of serving Large Language Models (LLMs).  eLLM tackles the problem of suboptimal memory utilization caused by the isolation between runtime memory (activations) and KV caches in existing LLM serving systems like vLLM. The framework is inspired by the memory ballooning technique used in operating systems. eLLM consists of three main components: (1) a Virtual Tensor Abstraction that decouples the virtual address space of tensors from physical GPU memory, creating a unified memory pool; (2) an Elastic Memory Mechanism that dynamically adjusts memory allocation through inflation and deflation, utilizing CPU memory as an extensible buffer; and (3) a Lightweight Scheduling Strategy that employs SLO-aware policies to optimize memory utilization. Experiments demonstrate that eLLM outperforms state-of-the-art systems, achieving higher decoding throughput and supporting larger batch sizes, especially for long-context inputs.

**Critical Evaluation:**

*   **Novelty:** The paper's core novelty lies in applying the concept of memory ballooning, traditionally used in operating systems, to the domain of LLM serving.  The specific instantiation of this idea through the virtual tensor abstraction, the elastic memory mechanism, and the integration with SLO-aware scheduling is also a novel combination. While individual components (virtual memory, dynamic allocation) aren't entirely new, their coordinated application in this context is. Previous works addressed KV cache optimizations but did not tackle the memory isolation issue between activations and the KV cache holistically.

*   **Significance:** The paper addresses a significant problem in LLM serving: efficient memory management for dynamic workloads. The memory isolation challenge is well-articulated, and the performance benefits demonstrated are substantial (up to 2.32x improvement in decoding throughput). The potential for larger batch sizes, especially for long-context scenarios, is particularly valuable as models increasingly support longer input sequences. eLLM's focus on both activations and KV cache, driven by shifting architecture trends (e.g., the Jamba model), positions it well for future LLM deployments.

*   **Strengths:**
    *   The problem is clearly defined and well-motivated by empirical analysis of memory utilization patterns in existing systems.
    *   The eLLM framework is well-designed, with clear explanations of its core components and their interactions.
    *   The evaluation is comprehensive, covering diverse models and workloads, and comparing against strong baselines.
    *   The ablation study provides valuable insights into the contributions of the individual components.
    *   The performance improvements demonstrated are significant.

*   **Weaknesses:**
    *   While the paper claims CPU memory is an "extensible buffer," it would be important to see the performance impact of constantly moving activations back and forth between GPU and CPU memory, and the overhead added from this dynamic behaviour. More detailed breakdown of GPU-CPU communication overhead would be beneficial.
    *   The paper mentions that eLLM can also lead to higher TPOT at times, but is able to trade-off between TTFT and TPOT to better SLO attainment. More details on SLO constraints and the trade-off made would strengthen the analysis.
    *   The paper could benefit from a more detailed discussion of the implementation challenges and complexities associated with integrating eLLM into existing LLM serving stacks.
    *   It would have been beneficial to include more discussion of potential security implications of the virtual tensor abstraction and dynamic memory management.
    *   The paper focuses more on GPU memory than parameter memory. How it would deal with parameter-level dynamism, such as in MoE models, is not addressed.

*   **Potential Influence:** The paper has the potential to significantly influence the design of future LLM serving systems. The idea of elastic memory management based on operating system principles is promising and could inspire further research in this area. The eLLM framework provides a concrete implementation that can be used as a foundation for building more efficient and scalable LLM serving infrastructure.

**Justification for Score:**

Given the identified strengths and weaknesses, and considering the paper's novelty and significance in the context of LLM serving, a score of **8** is appropriate. The paper makes a significant contribution by addressing a critical problem with a novel solution inspired by well-established operating systems concepts. The framework is well-designed, thoroughly evaluated, and demonstrates substantial performance improvements. While there are some areas for improvement in terms of detailed analysis of GPU-CPU communication overhead and broader real-world deployment considerations, the paper represents a valuable contribution to the field and has the potential to influence future LLM serving system designs.

Score: 8

- **Score**: 8/10

### **[Unlocking Post-hoc Dataset Inference with Synthetic Data](http://arxiv.org/abs/2506.15271v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

This paper tackles the problem of Dataset Inference (DI) for Large Language Models (LLMs), specifically addressing the practical limitation that DI requires a held-out dataset with the same distribution as the suspected training data, which is often unavailable in real-world scenarios. The authors propose a novel approach to synthetically generate this held-out data using a data generator trained on the suspect dataset itself. To overcome potential distribution shifts introduced by the synthetic data, they introduce a post-hoc calibration technique using a dual-classifier approach to disentangle membership signals from distributional artifacts. Extensive experiments on various text datasets demonstrate that their method enables reliable DI with high confidence and low false positives.

**Critical Evaluation:**

**Novelty:** The core idea of using synthetically generated data for dataset inference is innovative and addresses a significant bottleneck in applying DI in practice. The dual-classifier calibration approach is also a valuable contribution, as it directly tackles the problem of distribution shifts that can undermine the reliability of DI. The combination of these techniques provides a practical and effective solution to a pressing problem.

**Significance:**  The ability to perform dataset inference is becoming increasingly important in the age of large language models, where unauthorized use of copyrighted material is a major concern. This work directly empowers data owners to verify whether their data has been used to train LLMs without their consent, fostering greater transparency and accountability. The practicality of the method makes it a significant contribution.

**Strengths:**

*   **Addresses a critical limitation:** The paper directly addresses the need for in-distribution held-out data for DI, which is the major bottleneck limiting its applicability in real-world situations.
*   **Novel approach:**  Using synthetic data generation combined with a clever calibration method is a novel way to address the problem.
*   **Solid methodology:**  The data generator based on suffix completion, and the dual-classifier calibration strategy, are well-reasoned and empirically sound.
*   **Comprehensive evaluation:** The extensive experiments on diverse text datasets (single-author blogs and subsets of the Pile) demonstrate the robustness and generalizability of the proposed method.
*   **Clear and well-written:** The paper is well-structured and easy to follow.

**Weaknesses:**

*   **Dependency on Language Model Quality:** The quality of the synthetic data is reliant on the quality and fine-tuning of the language model used for generation. While the paper focuses on suffix-completion, the performance of the proposed method can be tied to the quality of the generative model.
*   **Computational Cost:** Training both the generator and the dual-classifiers can be computationally expensive, potentially limiting its accessibility for some users. The ablation studies help to identify less computationally intensive model architectures.
*   **Limited Theoretical Analysis:** While empirically strong, the paper lacks a deeper theoretical analysis of the properties and limitations of the proposed method. This would strengthen the contributions. For instance, under what conditions can we guarantee the synthetic data is "sufficiently" representative?

**Justification for Score:**

This paper offers a practical solution to a significant problem in the field of data ownership and LLM transparency. It overcomes a crucial bottleneck for DI, providing a valuable tool for data owners. The methodology is sound, and the experimental results are convincing. While the method relies on a LM model and lacks deep theoretical analysis, it offers a crucial stepping stone. This warrants a score of **8**.

Score: 8

- **Score**: 8/10

### **[Human Motion Capture from Loose and Sparse Inertial Sensors with Garment-aware Diffusion Models](http://arxiv.org/abs/2506.15290v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper addresses the problem of full-body human pose estimation using sparse, loosely-attached inertial measurement units (IMUs). It challenges the prevalent assumption in existing IMU-based motion capture research that sensors are tightly strapped to the body. The authors propose a method that leverages transformer-based diffusion models, trained on a combination of simulated, synthetic, and real-world IMU data, to estimate poses from loosely worn IMUs.  A key aspect of their approach is the incorporation of garment-related parameters into the training process, which allows the model to better capture variations due to different garment fits. The paper presents experimental results on three datasets, demonstrating that their approach outperforms state-of-the-art methods both quantitatively and qualitatively.

**Critical Evaluation:**

*   **Novelty:** The paper introduces a new task (full-body pose estimation from loosely attached IMUs) and a novel training strategy. Previous work on loose IMUs was limited to upper-body pose estimation.  The innovative aspect lies in the clever use of diffusion models, a secondary diffusion model for synthesizing loose-wear data, and, importantly, the incorporation of garment parameters to enhance realism and robustness. The method effectively integrates simulation and real-world data. The method shows an innovative approach by adding a hyperparameter, `alpha`, to weight the synthetic and real data to control for looseness in garment configurations.
*   **Significance:** Addressing motion capture with loosely attached IMUs is highly significant for real-world applications where user comfort and convenience are crucial. The reported results suggest that the proposed method can achieve robust and accurate pose estimation even with variations in garment looseness, moving towards more practical motion capture systems. The experimental results demonstrate clear improvements over existing approaches.
*   **Strengths:**

    *   Clearly defines a new and relevant problem.
    *   Proposes a well-designed and effective solution based on modern diffusion models.
    *   The training strategy combining simulated, synthetic, and real data is a notable strength.
    *   Incorporating garment parameters is a valuable contribution to improving robustness.
    *   Comprehensive experimental evaluation across multiple datasets.
    *   Ablation studies provide insights into the effectiveness of different components.
*   **Weaknesses:**

    *   While the garment parameter approach is innovative, the precise garment parameters used (TallThin/ShortFat Physique and the y-parameter) are limited to simulation. It could be argued that these are not direct garment parameters but physique/shape parameters that indirectly influence garment fit. The generalization of this aspect to real-world scenarios with varying garment types might be challenging.
    *   The "real-time inference" approach relies on a sliding window, limiting real-time performance. This approach needs to denoise and predict a new pose for every frame.
    *   Limited discussion on failure cases or limitations of the model.

*   **Potential Influence:** This paper has the potential to significantly influence the field of IMU-based motion capture by pushing the boundaries of what's possible with loosely attached sensors. It opens avenues for future research in:

    *   Exploring more sophisticated garment modeling techniques.
    *   Developing more efficient inference strategies for diffusion models to enable truly real-time performance.
    *   Investigating adaptive methods for automatically estimating garment parameters from sensor data.

**Score:** 8.5

**Rationale:**

The paper presents a significant advancement in IMU-based motion capture by addressing the challenging but practically relevant scenario of loosely attached sensors. The integration of diffusion models and garment parameters demonstrates considerable ingenuity. The experimental results convincingly show that the method outperforms the state-of-the-art. While the generalization of the garment parameter approach and the "real-time" performance could be improved, the strengths of the paper outweigh the weaknesses. The potential influence on the field is high, making it a valuable contribution.

Score: 8.5

- **Score**: 8/10

### **[SANSKRITI: A Comprehensive Benchmark for Evaluating Language Models' Knowledge of Indian Culture](http://arxiv.org/abs/2506.15355v1)**
- **Summary**: Here's a summary and critical evaluation of the provided paper:

**Summary:**

The paper introduces SANSKRITI, a new benchmark dataset for evaluating language models (LMs) on their understanding of Indian culture. The dataset comprises over 21,000 question-answer pairs spanning 28 states and 8 union territories in India and covers 16 cultural attributes, including rituals, history, tourism, cuisine, and more. The authors evaluate several Large Language Models (LLMs), Indic Language Models (ILMs), and Small Language Models (SLMs) on SANSKRITI, revealing significant performance disparities, particularly in region-specific contexts.  The authors also provide detailed statistics of the dataset regarding question type, attribute coverage and state wise question-counts.

**Critical Evaluation:**

*   **Strengths:**

    *   **Novelty and Contribution:** The paper addresses a critical gap in current LM evaluation: the lack of benchmarks that assess cultural understanding, particularly focusing on the nuanced diversity of India.  The authors have created a substantially large and diverse dataset covering key aspects of Indian culture. This benchmark is particularly important because many of the existing benchmarks cater primarily to Western culture and languages.
    *   **Comprehensive Coverage:** The dataset's coverage of all Indian states and union territories, along with its wide range of cultural attributes, makes it a significant improvement over existing datasets. The team’s rigorous data-sourcing methodology, drawing from various credible sources, enhances the dataset's validity and reliability.
    *   **Benchmarking and Analysis:** The authors' evaluation of various LMs provides valuable insights into their limitations in understanding cultural nuances. The detailed error analysis helps in understanding why models are failing.
    *   **Public Availability:** The public release of SANSKRITI promotes further research in culturally inclusive AI. The authors have also provided a link for the resources making the study verifiable.
*   **Weaknesses:**

    *   **Question Types:** The authors only used Multiple Choice Questions in the dataset. They didn't incorporate any other question formats to make the dataset more diverse. There were also no reasoning based questions in the dataset.
    *   **Limited State Specific Multilingual queries**: While the dataset is multicultural, questions are not multilingual for each and every Indian state.
    *   **Limited Contextual Clarity:** The questions may involve cultural elements that can be somewhat ambiguous. The authors could have made the questions better by providing more context to prevent this.

*   **Significance and Impact:**

    *   SANSKRITI fills a crucial void, enabling researchers to develop more culturally sensitive and inclusive LMs. The dataset can significantly impact applications in education, governance, and other domains where cultural understanding is paramount.
    *   The benchmark highlights the importance of addressing biases and stereotypes in LMs, promoting fairer and more equitable AI systems. The release of SANSKRITI serves as a call to action for the AI community to prioritize cultural awareness in LM development.

*   **Score Rationale:**

    SANSKRITI is a significant contribution due to its unique focus, comprehensiveness, and the potential for broad impact. Despite its limitations, it effectively addresses a crucial gap in LM evaluation. Its availability to the public also amplifies its value in research. Therefore, the paper deserves a high score.

Score: 8

- **Score**: 8/10

### **[Sampling 3D Molecular Conformers with Diffusion Transformers](http://arxiv.org/abs/2506.15378v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces DiTMC, a Diffusion Transformer-based framework for generating 3D molecular conformers. It addresses challenges related to applying DiTs to molecules, including integrating discrete graph information with continuous geometry, handling Euclidean symmetries, and designing conditioning mechanisms that generalize across varying molecular sizes and structures. DiTMC uses a modular architecture that separates 3D coordinate processing from atomic connectivity conditioning. The authors propose two graph-based conditioning strategies and explore different attention mechanisms (standard non-equivariant and SO(3)-equivariant) to balance accuracy and computational efficiency. The results demonstrate state-of-the-art precision and physical validity on conformer generation benchmarks. The code is publicly available.

**Critical Evaluation:**

*   **Novelty:** While the core idea of using diffusion transformers isn't entirely novel (DiTs are well-established in image generation), the adaptation to molecular conformer generation presents significant novel challenges. The paper's novelty lies in:

    *   The modular architecture specifically tailored for molecules, separating geometry and connectivity processing.
    *   The two complementary graph-based conditioning strategies designed to integrate with the DiT architecture. These strategies effectively encode molecular connectivity information within the continuous diffusion framework. The all-pair conditioning approach (geodesic distances) is particularly interesting.
    *   The systematic comparison of different self-attention mechanisms, including the SO(3)-equivariant variant, providing insights into the trade-offs between accuracy, efficiency, and symmetry preservation.  While equivariant models are common in geometric deep learning, a direct comparison with non-equivariant attention mechanisms *within a diffusion transformer architecture for conformer generation* adds value.
*   **Significance:** The significance stems from the potential to leverage the power of DiTs for a crucial task in drug discovery and materials science. The results showing state-of-the-art performance on established benchmarks is significant, implying a tangible improvement over existing methods. The emphasis on physical validity (demonstrated through ensemble property prediction) is also important, as it ensures the generated conformers are not just statistically plausible but also chemically realistic. The modular design allows for future improvements in each component.
*   **Strengths:**

    *   Clear problem definition and well-motivated approach.
    *   Thorough experimental evaluation on standard datasets.
    *   Systematic ablation studies exploring the impact of different design choices.
    *   Publicly available code, promoting reproducibility and further research.
    *   Addresses a practically relevant problem with potential for real-world impact.
*   **Weaknesses:**

    *   The computational cost of the SO(3)-equivariant variant remains a concern, limiting its scalability. The paper acknowledges this but doesn't offer concrete solutions.
    *   While the experiments demonstrate SOTA performance, the gains over previous methods are sometimes incremental. Some of the observed improvements might not be statistically significant across all reported metrics, even if "averaged" over three runs. Standard deviations are not reported for key SOTA comparison tables such as Table 1.
    *   The evaluation is limited to relatively small molecules. The scalability of DiTMC to larger and more flexible molecules needs further investigation.
    *   The "high-quality" ground truth conformers used for training depend on *expensive* quantum chemistry calculations, which may not be readily available in all cases, limiting broad applicability of the model.
    *   Limited exploration of alternative sampling strategies (beyond the Euler scheme).
*   **Potential Influence:** The paper is likely to influence future research in generative modeling for molecules. The insights from the attention mechanism comparison and the modular architecture are valuable for designing more efficient and accurate models. The use of DiTs, which are proving very effective in images and video, opens new avenues for molecule generation. The paper should encourage further exploration of equivariant architectures and conditioning strategies in this domain.

**Overall:**

The paper makes a valuable contribution to molecular conformer generation by effectively adapting diffusion transformers and providing a thorough analysis of key architectural choices. The approach achieves state-of-the-art performance and demonstrates the importance of symmetry considerations and conditioning strategies. While some limitations exist (computational cost, molecule size), the work provides a solid foundation for future research.

Score: 8.5

- **Score**: 8/10

### **[One-Step Diffusion for Detail-Rich and Temporally Consistent Video Super-Resolution](http://arxiv.org/abs/2506.15591v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces "Dual LoRA Learning" (DLORAL), a novel approach for video super-resolution (VSR) that addresses the challenge of balancing detail enhancement and temporal consistency in real-world videos. DLORAL leverages a one-step diffusion model with two LoRA modules: Consistency-LoRA (C-LoRA) learns robust temporal representations from degraded inputs using a Cross-Frame Retrieval (CFR) module, while Detail-LoRA (D-LoRA) enhances spatial details while aligning with the temporal space defined by C-LoRA.  The two LoRA branches are trained alternately and iteratively.  The paper demonstrates strong performance in both accuracy and speed compared to existing VSR methods.

**Critical Evaluation:**

*   **Novelty:**  The key novelty of this paper lies in its decoupled learning approach for temporal consistency and spatial detail within a diffusion framework.  The idea of using two LoRA modules tailored to these separate concerns is a sensible and potentially powerful technique.  The Cross-Frame Retrieval (CFR) module, which aggregates information from adjacent frames to improve consistency, is also a valuable contribution.  The alternating training scheme, where the consistency and detail enhancements are refined iteratively, is a good idea. LoRA is a known method, this paper innovatively applies to the real-VSR diffusion problem with good design.

*   **Significance:**  The paper addresses a significant problem in video super-resolution – the trade-off between detail and consistency. The results are compelling, showing that DLORAL achieves superior performance in terms of both visual quality and temporal coherence. This is valuable because it moves towards VSR models that are actually usable and preferred for real-world scenarios.  The efficiency of the method, achieved through the one-step diffusion framework and LoRA, is also significant, making high-quality VSR more computationally accessible.

*   **Strengths:**

    *   The decoupling of detail enhancement and temporal consistency learning is well-motivated and effective.
    *   The CFR module is a practical way to leverage temporal information from degraded inputs.
    *   The iterative training scheme allows for a balance between conflicting objectives.
    *   The one-step diffusion process with LoRA provides efficiency in inference.
    *   The paper demonstrates state-of-the-art results on multiple datasets.
    *   User study is provided, supporting the effectiveness of the method
    *   The paper is well-written, clear, and easy to understand.

*   **Weaknesses:**

    *   The method relies on a pre-trained Stable Diffusion model, which is inherited its limitations, specifically the difficulty in reconstructing very fine-scale details. As mentioned in the limitation section of the paper
    *   Although the experimental results are good, the ablation studies in the Appendix are not fully conclusive, specifically on different module designs for both D-LoRA and C-LoRA. Further investigation here could strengthen the work.

*   **Potential Impact:**

    *   This work could have a significant impact on the field of video super-resolution, paving the way for more practical and usable VSR models.
    *   The decoupled learning approach and the CFR module could be adopted and extended by other researchers in the field.
    *   The efficient one-step diffusion framework could make high-quality VSR more accessible to a wider range of applications.

*   **Justification for the score:** DLORAL represents a significant advancement in the field of video super-resolution. While it builds upon existing techniques (diffusion models and LoRA), it innovatively combines them with a well-designed architecture and training strategy to address a key challenge in VSR. The empirical results and efficiency gains are strong indicators of its potential impact. The paper has some limitations (the inheritance of diffusion model issues and further exploration of different network design).

Score: 8

- **Score**: 8/10

### **[HOIDiNi: Human-Object Interaction through Diffusion Noise Optimization](http://arxiv.org/abs/2506.15625v1)**
- **Summary**: Okay, here's a summary and critical evaluation of the paper:

**Summary:**

The paper presents HOIDiNi, a novel text-driven diffusion framework for synthesizing realistic and plausible human-object interactions (HOI).  HOIDiNi leverages Diffusion Noise Optimization (DNO) within a structured two-phase approach. The first phase, "Object-Centric," focuses on determining the object's trajectory and hand-object contact points. The second phase, "Human-Centric," refines the full-body human motion, including finger articulation, while adhering to the constraints established in the first phase. By separating the problem into these phases, HOIDiNi aims to achieve precise hand-object contact without sacrificing the naturalness of human motion.  The method incorporates a learned CPHOI model that jointly learns human and object motion with contact point predictions. The paper presents quantitative, qualitative, and subjective evaluations demonstrating HOIDiNi's superior performance compared to existing methods in terms of contact accuracy, physical validity, and overall quality on the GRAB dataset.

**Critical Evaluation:**

*   **Novelty:**  The paper demonstrates a clever combination of existing techniques, particularly DNO, into a novel framework tailored for HOI generation. The two-phase optimization strategy is a significant contribution. Separating the object-centric contact determination from the full-body motion refinement is a crucial insight that addresses the inherent challenges of achieving both realism and accuracy in HOI. The direct prediction and optimization of contact points, instead of relying on heuristics, improves stability and plausibility.

*   **Significance:** HOI generation is a crucial and challenging area within digital human modeling and robotics.  The ability to synthesize realistic and controllable HOIs has broad applications. HOIDiNi's contribution towards achieving this goal is significant. The paper addresses the existing limitations of existing methods. The results are impressive when compared with other approaches.

*   **Strengths:**

    *   **Well-defined problem and approach:** The paper clearly articulates the challenges of HOI generation and presents a structured approach to address them.
    *   **Effective use of DNO:** The adaptation of DNO for HOI, particularly within the two-phase framework, is well-motivated and effectively implemented.
    *   **Contact Prediction:** The direct prediction of the hand-object contact pair is a strong design element.
    *   **Comprehensive evaluation:** The paper includes quantitative, qualitative, and subjective evaluations, providing a robust assessment of HOIDiNi's performance.  The user study adds strong support to the claims of improved realism.
    *   **Clear writing and presentation:** The paper is well-written and easy to follow. The figures and tables are informative.

*   **Weaknesses:**

    *   **Dataset Limitation:** The reliance on the GRAB dataset, while practical for comparison, restricts the generalizability of the results. GRAB's limited diversity makes evaluating HOIDiNi's capabilities a little challenging. A more diverse dataset could better highlight the advantages of the method.
    *   **Computational Cost:** DNO, even with the autoregressive diffusion, is inherently computationally expensive. The paper could benefit from a discussion of the computational cost and potential avenues for optimization.
    *   **Limited Generalization/Novelty:** While the system looks impressive, it should be acknowledged that the two-stage approach is not new, with SAGA previously using such a setup. Similarly, combining a diffusion model with an objective function is not new (although DNO is of course a specific way to do it). The paper would be more convincing if it was clearer on the exact design choices that resulted in a useful system.

*   **Potential Influence:** The paper is likely to have a significant influence on the field of HOI generation. The two-phase optimization strategy and the direct prediction of contact points could become standard techniques. HOIDiNi could serve as a foundation for future research on controllable and realistic HOI synthesis. The findings of this work could advance digital human modeling, animation, robotics, and related fields.

**Overall Assessment:**

HOIDiNi represents a significant advance in the field of HOI generation. The paper demonstrates a well-designed and effective framework that combines the strengths of diffusion models with a structured optimization approach. While limitations exist regarding dataset reliance and computational cost, the paper's contributions are substantial and likely to influence future research in this area.

Score: 8

- **Score**: 8/10

### **[SwarmAgentic: Towards Fully Automated Agentic System Generation via Swarm Intelligence](http://arxiv.org/abs/2506.15672v1)**
- **Summary**: Here's a summary and critical evaluation of the "SwarmAgentic: Towards Fully Automated Agentic System Generation via Swarm Intelligence" paper:

**Summary:**

The paper introduces SwarmAgentic, a novel framework for fully automated generation of agentic systems. Unlike existing frameworks that rely on predefined templates, seed agents, or human intervention, SwarmAgentic constructs agentic systems from scratch. It leverages a language-driven, population-based search inspired by Particle Swarm Optimization (PSO) to jointly optimize agent functionality and collaboration strategies. The framework represents agentic systems as "particles" in a symbolic design space, using LLMs to guide exploration and iteratively refine system configurations through failure-aware velocity updates. The paper evaluates SwarmAgentic on a range of real-world, open-ended tasks, demonstrating its ability to outperform existing baselines in terms of performance and adaptability.

**Critical Evaluation:**

*   **Novelty:** The core novelty lies in the **complete automation of agentic system generation**, addressing a key limitation in existing frameworks. The innovative adaptation of PSO to a language-based, symbolic design space is also significant.  While other works have explored agent automation or optimization, SwarmAgentic uniquely combines from-scratch generation, self-optimizing agent functionality, and self-optimizing collaboration strategies, marking an important step forward. The use of LLMs for flaw identification and structured updating is clever and leads to a more interpretable optimization process.
*   **Significance:** The significance stems from the potential to **democratize agentic system design and scale their application to complex, structurally unconstrained tasks**. The automation offered by SwarmAgentic reduces the engineering overhead associated with manually designing and fine-tuning multi-agent systems. This expands the applicability of agentic systems to areas where manual design is impractical or impossible. The reported performance gains across various tasks, especially the substantial improvement on the TravelPlanner benchmark, strongly suggest the framework's practical value.
*   **Strengths:**

    *   **Full Automation:** Successfully addresses the limitations of existing frameworks by providing a fully automated system design process.
    *   **Scalability:** The PSO-inspired approach facilitates scalable exploration of the design space.
    *   **Adaptability:** Dynamic refinement of agent functionalities and collaboration strategies enables adaptability to diverse task specifications.
    *   **Strong Empirical Results:**  Demonstrated superior performance on a variety of challenging, real-world tasks compared to strong baselines.
    *   **Clear presentation:** The core concepts, methods and experiments are presented in a clear and well-organized manner.

*   **Weaknesses:**

    *   **LLM Dependence:** Like many recent advancements in agentic systems, SwarmAgentic relies heavily on the capabilities of LLMs, inheriting limitations such as factual inaccuracies and potential biases. The potential impact of these limitations on the reliability and trustworthiness of generated agentic systems needs further investigation.
    *   **Limited Embodiment:** The framework operates primarily in a text-based environment, lacking the perception and action capabilities needed for real-world embodied applications.
    *   **Computational Cost:**  The iterative nature of the PSO algorithm and the repeated use of LLMs could lead to high computational costs, particularly for complex tasks. The paper does not sufficiently address or quantify these costs.
    *   **Lack of ablation study details:** While the paper does include an ablation study to assess the effect of the different components of SwarmAgentic, it does not include enough detail to provide a complete and robust justification for these components.
*   **Potential Influence:** If further developed and validated, SwarmAgentic has the potential to significantly impact the field of agentic systems by enabling the creation of more adaptable, scalable, and task-specific multi-agent systems. This could have broad implications for areas such as automated planning, task coordination, and creative problem-solving. It could serve as a basis for future work in AI by establishing a new framework for AI agentic systems.

*   **Justification of Score:** Given the originality and potential impact of SwarmAgentic, but acknowledging its reliance on LLMs and the limitations concerning cost and embodied applications, a score of 8 is warranted. The framework represents a significant advancement, but further research is needed to address its inherent limitations and more clearly quantify performance in relation to the cost of the systems produced.

**Score: 8**

- **Score**: 8/10

### **[UniRelight: Learning Joint Decomposition and Synthesis for Video Relighting](http://arxiv.org/abs/2506.15673v1)**
- **Summary**: Here's a summary and critical evaluation of the UniRelight paper:

**Summary:**

The paper "UniRelight: Learning Joint Decomposition and Synthesis for Video Relighting" tackles the challenging problem of relighting single images or videos. The method proposes a novel approach that jointly estimates albedo and synthesizes relit outputs in a single pass, leveraging the generative capabilities of video diffusion models. By concatenating the latent representations of the input video, albedo, and relit output, the model can implicitly reason about scene structure and material properties. The model is trained on a combination of synthetic multi-illumination data and automatically labeled real-world videos, which enhances generalization across diverse domains. The paper demonstrates that UniRelight produces realistic lighting effects, captures intricate material interactions, and surpasses previous methods in visual fidelity and temporal consistency.

**Critical Evaluation:**

* **Novelty:** The core novelty lies in the **joint formulation** of albedo estimation and relighting within a single video diffusion model, particularly by using latent space concatenation to facilitate cross-modal interaction. While existing works have used diffusion models for relighting or inverse rendering, the joint modeling with the specific architecture choices and data strategy is a key differentiator. The exploitation of **self-supervised real-world data** auto-labeled with a pre-trained inverse renderer also contributes to novelty, as it allows for better generalization. The idea of implicitly reasoning about scene properties is well-received in the community, reducing reliance on explicit G-buffers.

* **Significance:** The paper addresses a critical limitation in relighting research: the scarcity of multi-illumination data. By introducing a method that effectively uses both synthetic and real-world self-supervised data, UniRelight opens the door for more robust and generalizable relighting systems.  The results demonstrate a clear improvement over existing methods, particularly in handling complex materials and scenes. The temporal consistency aspect is important for video applications. The improvement in runtime compared to DiffusionRenderer is also significant.

* **Strengths:**
    * The joint modeling approach is well-motivated and shows tangible benefits in terms of artifact reduction and generalization.
    * The hybrid training strategy, combining synthetic and self-supervised real-world data, is effective for improving realism and domain adaptation.
    * The results are compelling, demonstrating improved visual quality, temporal consistency, and handling of complex material properties (e.g., transparency, subsurface scattering).
    * The speed improvement relative to other diffusion-based methods is a practical advantage.

* **Weaknesses:**
    * While the paper touches on limitations (emitting objects), it could benefit from more detailed analysis of failure cases. For example, which scene types or lighting conditions still pose challenges?
    * The reliance on a pre-trained inverse renderer for generating pseudo-ground truth albedo introduces a potential bias. While the averaging technique helps stabilize the albedo maps, it's still an indirect approach. A more rigorous ablation study exploring the impact of albedo quality could strengthen the paper.
    * The reliance on diffusion models can introduce certain artifacts related to stochastic generation processes. Exploring different conditioning strategies in future research might improve the controllability of the approach.
    * Although the introduction mentions creative editing and robust vision systems, they're never elaborated on or demonstrated.

* **Impact:**  The paper's impact is potentially high, particularly for video relighting applications. By pushing the boundaries of diffusion-based relighting and offering a solution that addresses data scarcity, it could influence future research directions in this area. Other research might use the paper's architecture or training strategy for other inverse rendering problems. The speed improvement is another reason the paper is significant.

* **Score Justification:**

The paper demonstrates significant novelty and improvement. The joint formulation, combined with the data strategy, yields tangible benefits. While there are some weaknesses related to reliance on pseudo-ground truth data and the lack of demonstration of its real-world application, the results are compelling.

**Score: 8**

- **Score**: 8/10

### **[PhantomHunter: Detecting Unseen Privately-Tuned LLM-Generated Text via Family-Aware Learning](http://arxiv.org/abs/2506.15683v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces "PhantomHunter," a novel LLM-generated text (LLMGT) detector specifically designed to identify text produced by privately fine-tuned open-source LLMs.  Recognizing that standard LLMGT detectors falter when confronted with such privately adapted models, PhantomHunter employs a "family-aware" learning framework. This framework aims to capture the inherent family-level characteristics shared between a base LLM and its fine-tuned derivatives, instead of trying to memorize individual model idiosyncrasies.  The system incorporates a base probability feature extractor, a contrastive learning-based family encoder, and a mixture-of-experts detection module. Experimental results across LLaMA, Gemma, and Mistral families demonstrate PhantomHunter's superior performance compared to existing baselines and commercial services.

**Critical Evaluation:**

* **Novelty:** The paper addresses a very practical and increasingly important problem: the detectability of text from privately fine-tuned LLMs.  Existing LLMGT detection research largely focuses on publicly available, off-the-shelf models. The authors correctly identify the gap in the current literature and develop a method that specifically targets this new threat landscape.  The core novelty lies in the "family-aware" learning approach.  While contrastive learning and mixture-of-experts models are not new *per se*, their application in this context, to explicitly learn family-level characteristics, represents a significant contribution. The observation that fine-tuning, while changing the characteristics, does preserve family traits, is a key insight.

* **Significance:**  The paper's significance stems from the growing accessibility and widespread use of open-source LLMs. The barrier to creating custom LLMs using techniques like LoRA is lower than ever. This increases the risk of malicious actors using private LLMs to generate misinformation, spam, or circumvent other safety mechanisms. PhantomHunter provides a practical defense against this threat. The results demonstrate a clear improvement over existing methods, indicating that it could significantly improve the reliability of LLMGT detection systems in real-world scenarios. Moreover, identifying the lineage/family of the LLM used offers potential for source attribution, which can be invaluable in forensic analysis.

* **Strengths:**
    * **Problem Relevance:**  The paper tackles a timely and relevant problem.
    * **Novel Approach:**  The "family-aware" learning framework is a well-motivated and novel solution.
    * **Strong Empirical Results:**  The experiments are comprehensive, covering multiple LLM families, fine-tuning techniques, and comparisons against a diverse set of baselines and commercial services.
    * **Ablation Studies:**  The ablation studies provide valuable insights into the contributions of each component of PhantomHunter.
    * **Clear Writing:** The paper is generally well-written and easy to understand.
    * **Reproducibility:** The authors clearly describe their experimental setup and data generation process, increasing the likelihood of reproducibility.

* **Weaknesses:**
    * **Limited Family Coverage:** While the experiments cover three popular LLM families, the generalizability of the method to entirely *unseen* families remains an open question. The effectiveness hinges on having base model probabilities available, which may not be true for all models.
    * **Dependence on Base Model Access:**  A key assumption of PhantomHunter is access to the base LLM's probability distributions.  If a malicious actor were to deliberately obscure the origin of their fine-tuned model, PhantomHunter's performance would likely degrade.  The paper doesn't address this scenario.
    * **Family Prediction Performance:** While PhantomHunter excels at detecting LLMGT, the accuracy of *family prediction* is considerably lower.  This suggests the family encoder could be further improved.
    * **Potential for Adversarial Attacks:** It's plausible that an adversary could design a fine-tuning process to specifically evade PhantomHunter's detection mechanisms.  The paper doesn't discuss potential defenses against such attacks.
    * **Commercial Detector comparison:** Though included and useful, there is little detail about which models the commercial detectors used for their base models. This could lead to results biased to more common base model choices, though it does provide real world implications.

* **Potential Influence:** This paper is likely to influence the field in several ways. First, it will shift the focus of LLMGT detection research to the challenges posed by privately fine-tuned models.  Second, the "family-aware" learning framework could be adopted and extended by other researchers. Third, the paper highlights the importance of source attribution as a crucial component of LLMGT detection systems.

**Score and Justification:**

I assign this paper a **Score: 8**.

**Rationale:**

The paper makes a significant and novel contribution to a timely and important problem. The family-aware approach and the strong empirical results are compelling. However, the limitations concerning generalizability to unseen families, the reliance on base model access, and the lack of discussion on adversarial attacks prevent it from receiving a higher score. While the work presents a solid framework, future work needs to extend its application to more models and make it more resilient to sophisticated attacks. It is an important contribution to the field of LLMGT detection, that will likely spur significant advances in this domain.

- **Score**: 8/10

## Other Papers
### **[Using BDF schemes in the temporal integration of POD-ROM methods](http://arxiv.org/abs/2506.14543v1)**
### **[DreamLight: Towards Harmonious and Consistent Image Relighting](http://arxiv.org/abs/2506.14549v1)**
### **[Empirically-Calibrated H100 Node Power Models for Reducing Uncertainty in AI Training Energy Estimation](http://arxiv.org/abs/2506.14551v1)**
### **[Risk Estimation of Knee Osteoarthritis Progression via Predictive Multi-task Modelling from Efficient Diffusion Model using X-ray Images](http://arxiv.org/abs/2506.14560v1)**
### **[AlphaDecay:Module-wise Weight Decay for Heavy-Tailed Balancing in LLMs](http://arxiv.org/abs/2506.14562v1)**
### **[Single-Example Learning in a Mixture of GPDMs with Latent Geometries](http://arxiv.org/abs/2506.14563v1)**
### **[TGDPO: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization](http://arxiv.org/abs/2506.14574v1)**
### **[GenerationPrograms: Fine-grained Attribution with Executable Programs](http://arxiv.org/abs/2506.14580v1)**
### **[Busting the Paper Ballot: Voting Meets Adversarial Machine Learning](http://arxiv.org/abs/2506.14582v1)**
### **[NetRoller: Interfacing General and Specialized Models for End-to-End Autonomous Driving](http://arxiv.org/abs/2506.14589v1)**
### **[Align Your Flow: Scaling Continuous-Time Flow Map Distillation](http://arxiv.org/abs/2506.14603v1)**
### **[Guaranteed Guess: A Language Modeling Approach for CISC-to-RISC Transpilation with Testing Guarantees](http://arxiv.org/abs/2506.14606v1)**
### **[Exploring MLLMs Perception of Network Visualization Principles](http://arxiv.org/abs/2506.14611v1)**
### **[Probabilistic Aggregation and Targeted Embedding Optimization for Collective Moral Reasoning in Large Language Models](http://arxiv.org/abs/2506.14625v2)**
### **[ACM Survey Draft on Formalising Software Requirements with Large Language Models](http://arxiv.org/abs/2506.14627v1)**
### **[AIn't Nothing But a Survey? Using Large Language Models for Coding German Open-Ended Survey Responses on Survey Motivation](http://arxiv.org/abs/2506.14634v2)**
### **[Revisiting Chain-of-Thought Prompting: Zero-shot Can Be Stronger than Few-shot](http://arxiv.org/abs/2506.14641v1)**
### **[Passing the Turing Test in Political Discourse: Fine-Tuning LLMs to Mimic Polarized Social Media Comments](http://arxiv.org/abs/2506.14645v1)**
### **[GuiLoMo: Allocating Expert Number and Rank for LoRA-MoE via Bilevel Optimization with GuidedSelection Vectors](http://arxiv.org/abs/2506.14646v1)**
### **[Massive Supervised Fine-tuning Experiments Reveal How Data, Layer, and Training Factors Shape LLM Alignment Quality](http://arxiv.org/abs/2506.14681v1)**
### **[AIRTBench: Measuring Autonomous AI Red Teaming Capabilities in Language Models](http://arxiv.org/abs/2506.14682v1)**
### **[Capacity Matters: a Proof-of-Concept for Transformer Memorization on Real-World Data](http://arxiv.org/abs/2506.14704v1)**
### **[Iterative Camera-LiDAR Extrinsic Optimization via Surrogate Diffusion](http://arxiv.org/abs/2506.14706v1)**
### **[AgentDistill: Training-Free Agent Distillation with Generalizable MCP Boxes](http://arxiv.org/abs/2506.14728v1)**
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
### **[PhantomHunter: Detecting Unseen Privately-Tuned LLM-Generated Text via Family-Aware Learning](http://arxiv.org/abs/2506.15683v1)**
### **[Nabla-R2D3: Effective and Efficient 3D Diffusion Alignment with 2D Rewards](http://arxiv.org/abs/2506.15684v1)**
