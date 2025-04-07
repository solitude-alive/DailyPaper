# The Latest Daily Papers - Date: 2025-04-07
## Highlight Papers
### **[QIRL: Boosting Visual Question Answering via Optimized Question-Image Relation Learning](http://arxiv.org/abs/2504.03337v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces a novel framework called Optimized Question-Image Relation Learning (QIRL) to improve Visual Question Answering (VQA) by addressing language bias and enhancing model robustness. QIRL uses a generation-based self-supervised learning strategy. It has two main modules: (1) Negative Image Generation (NIG), which generates irrelevant question-image pairs to enhance correlation learning and (2) Irrelevant Sample Identification (ISI), which identifies and filters irrelevant inputs to reduce prediction errors. The NIG uses a diffusion model and a sentence revision tool to generate highly contrasting samples.  The ISI module estimates the similarity of QI pairs and abstains from making predictions if the input is deemed irrelevant.  The authors demonstrate the effectiveness and generalizability of QIRL by integrating it with various VQA models and testing it on the VQA-CPv2 and VQA-v2 datasets, achieving state-of-the-art results among data augmentation strategies.

**Critical Evaluation:**

*   **Novelty:** The paper presents a good level of novelty. The core ideas of generating highly irrelevant question-image pairs using diffusion models and sentence revision, coupled with a module to identify and filter irrelevant inputs, are innovative. While previous works have explored data augmentation and debiasing, the specific combination of techniques and the explicit focus on improving question-image relation learning distinguish this approach. The focus on identifying *relevance* of QI pairs, rather than simply generating negative samples, is a key contribution.

*   **Significance:** The paper addresses a critical issue in VQA: language bias and model robustness. By improving QI relation learning, QIRL makes VQA models less reliant on superficial correlations and better equipped to handle irrelevant or misleading inputs. The empirical results on standard benchmarks (VQA-CPv2 and VQA-v2) show substantial improvements over baseline models and competitive performance against existing debiasing techniques, suggesting a significant contribution to the field. The modularity of the approach, demonstrated by integrating with various VQA architectures, further increases its significance.

*   **Strengths:**
    *   **Clear problem definition:** The paper clearly articulates the limitations of existing debiasing techniques and motivates the need for a more robust approach to QI relation learning.
    *   **Technically sound methodology:** The QIRL framework is well-defined, and the individual modules (NIG and ISI) are logically designed. The use of diffusion models and sentence revision tools is appropriate for generating high-quality, irrelevant samples.
    *   **Comprehensive experimental evaluation:** The paper provides extensive experimental results on standard benchmarks, demonstrating the effectiveness and generalizability of QIRL. Ablation studies are performed to validate the contributions of each module.
    *   **Model-agnostic:** The framework can be incorporated with many existing VQA models.

*   **Weaknesses:**
    *   **Evaluation metric:** While the authors introduce a specialized metric (Accspe) to evaluate the ISI module, the overall VQA performance is still evaluated using the standard metric, which does not explicitly reward the "abstain" predictions. A more comprehensive evaluation protocol that accounts for both accuracy and the ability to correctly identify irrelevant inputs could further strengthen the results.
    *   **Complexity:** Introducing a diffusion model and sentence revision tool increases the complexity compared to simpler data augmentation methods. The additional computational cost is worth mentioning.
    *   **Analysis of Failure Cases:** While the paper includes a qualitative analysis, a deeper investigation into *why* certain examples fail even with QIRL could provide valuable insights for future improvements. What types of questions or images are still challenging?
    *   **Large model bias:** It would be useful to analyze whether the approach could be used as an intermediate step to alleviate some of the data biases when training larger VQA models, which require greater computing resources.

*   **Potential Influence:** QIRL has the potential to influence future research in VQA and multi-modal learning. The framework provides a novel approach to improving QI relation learning and model robustness. The generation strategy could be adapted for other tasks, and the ISI module could be used to filter irrelevant inputs in other applications. The concept of relevance-aware VQA is also valuable.

*   **Missing Important Citations:**

        *   A. Torralba and A. Efros. "Unbiased look at dataset bias." CVPR 2011.
        *   R. Krishna et al. "Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations." IJCV 2017.

**Score: 8**

**Justification:**

The paper makes a significant contribution to the field of VQA by addressing language bias and improving model robustness through a novel and well-designed framework. The use of diffusion models and sentence revision is innovative. The extensive experimental results and ablation studies support the effectiveness of QIRL. While there is room for improvement in evaluation metrics, complexity and potentially broader application to very large models, the paper's strengths outweigh its weaknesses. The model-agnostic characteristics make it a useful tool for current models, while its novel methods enhance dataset robustness in the VQA space.

- **Score**: 8/10

### **[Online Difficulty Filtering for Reasoning Oriented Reinforcement Learning](http://arxiv.org/abs/2504.03380v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper proposes a novel approach to training Large Language Models (LLMs) for reasoning tasks, specifically focusing on Reasoning-Oriented Reinforcement Learning (RORL).  The core idea is "balanced online difficulty filtering," which dynamically curates training batches by selecting problems that are neither too easy nor too difficult for the model's current ability.  The authors provide a theoretical justification for this approach, deriving that maximizing the variance of the sampled accuracy (targeting a pass rate around 0.5) maximizes a lower bound of the KL divergence between the initial and optimal policies.  This balanced filtering is implemented using asynchronous sampling to maintain a fixed batch size.  The approach is empirically evaluated on several math reasoning benchmarks, showing improved performance compared to plain GRPO and offline curriculum learning methods. The paper also examines the role of difficulty filtering by analyzing different strategies and the adaptation to different model capabilities.

**Critical Evaluation:**

*   **Novelty:** The idea of dynamically adjusting the training difficulty based on real-time model performance isn't entirely new (curriculum learning, online filtering), but the paper makes a significant contribution by providing a theoretical grounding for why a *balanced* difficulty distribution is beneficial. Deriving the connection between the KL divergence lower bound and the variance of accuracy is a strong point.  The focus on *reasoning-oriented* RL and the application of this filtering approach within that context is also noteworthy.

*   **Significance:** The paper addresses a critical challenge in RORL: the sparsity of rewards and the need for efficient training. By focusing on problems that are appropriately challenging, the proposed approach demonstrably improves sample efficiency and performance on challenging reasoning tasks. The empirical gains (10% on AIME) are substantial and support the theoretical claims. The finding that balanced filtering outperforms skewed filtering offers practical insights.

*   **Strengths:**

    *   **Theoretical Justification:** The paper isn't just an empirical exploration; it provides a mathematical argument for the benefits of balanced difficulty filtering.
    *   **Empirical Validation:** The experiments are comprehensive, covering multiple benchmarks and comparing against strong baselines (plain GRPO, offline curricula).
    *   **Practical Implementation:** The asynchronous sampling approach ensures a fixed batch size and avoids training instability, addressing a common problem with online filtering methods.
    *   **Analysis:** The paper doesn't just present results; it includes thoughtful analyses of why the approach works, how it adapts to model capabilities, and the role of different difficulty assessment proxies.

*   **Weaknesses:**

    *   **Limited Generalizability:** The empirical evaluation is focused on math reasoning tasks.  It would be valuable to see how well the approach generalizes to other types of reasoning problems or different domains.
    *   **Sensitivity to Hyperparameters:** The performance depends on the choice of thresholds TLow and THigh. While the paper explores some settings, a more thorough investigation of the sensitivity of the results to these hyperparameters would be beneficial.
    *   **Complexity:** While the asynchronous sampling addresses a specific problem, it could add implementation complexity. An analysis of the computational overhead would be valuable.

*   **Potential Impact:** The paper has the potential to influence how LLMs are trained for reasoning tasks. The theoretical insights and practical implementation strategies could be adopted by other researchers and practitioners in the field. The emphasis on sample efficiency is particularly relevant as models and datasets continue to grow.

**Justification of Score:**

The paper presents a novel theoretical analysis of difficulty filtering within the context of RORL, complemented by strong empirical results and thoughtful analysis. While the experiments are somewhat limited in scope and there are concerns about hyperparameters and implementation complexity, the paper makes a clear and significant contribution to the field.

**Score: 8**

- **Score**: 8/10

### **[EnrichIndex: Using LLMs to Enrich Retrieval Indices Offline](http://arxiv.org/abs/2504.03598v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "EnrichIndex: Using LLMs to Enrich Retrieval Indices Offline":

**Summary:**

The paper introduces EnrichIndex, a novel retrieval approach that leverages Large Language Models (LLMs) offline to build semantically enriched retrieval indices. Unlike existing LLM-augmented retrieval methods which typically compute query-document relevance online (resulting in high latency and computation costs), EnrichIndex performs a single pass over the retrieval corpus during ingestion time to generate enriched representations of documents (summaries, purpose, QA pairs) which are then indexed. During online retrieval, it computes object relevance by calculating the weighted sum of similarities between the user query and the original as well as the enriched representations. This approach aims to boost the performance of stage-one retrievers, which in turn improves the overall performance of downstream LLM re-rankers while significantly reducing online LLM calls and computational costs. The paper presents experiments on five retrieval tasks involving passages and tables, demonstrating that EnrichIndex outperforms strong LLM-based baselines with substantial reductions in online LLM token usage.

**Critical Evaluation:**

**Novelty:** The core idea of offline enrichment is reasonably novel in the context of LLM-augmented retrieval. The standard approach of online ranking and re-ranking has significant latency and costs associated with it. EnrichIndex offers a pragmatic solution by shifting the heavy lifting to an offline process, using LLMs to enrich the retrieval indices. While pre-indexing is a well-established concept, the specific application of LLMs for semantic enrichment of indices (purpose, summary, QA pairs) is a useful and innovative combination. The modularity of the approach is also a strength; it can complement existing retrieval methods and be combined with query expansion techniques.

**Significance:** The significance of the paper lies in addressing a key bottleneck in LLM-augmented retrieval: high latency and cost. The experimental results demonstrate that EnrichIndex can achieve substantial performance gains compared to online methods, while greatly reducing online computation and token usage. This makes LLM reasoning more accessible and scalable in real-world retrieval systems. The improvements are particularly strong on complex retrieval tasks where explicit semantic understanding is required. Furthermore, the analysis showing the contribution of different enrichment types and the improved separation of gold and non-gold objects strengthens the paper’s message.  The findings also highlight the continued importance of strong first-stage retrieval, even when employing powerful LLM-based re-rankers.

**Strengths:**

*   **Clear Problem Definition:** The paper clearly identifies a crucial limitation of existing LLM-augmented retrieval approaches.
*   **Novel Approach:** The offline enrichment strategy offers a compelling solution.
*   **Comprehensive Evaluation:** The experiments cover a range of datasets and retrieval tasks.
*   **Significant Performance Gains:** The reported improvements over strong baselines are substantial.
*   **Detailed Analysis:** The analysis of enrichment types and distributional shifts provides valuable insights.
*   **Efficiency:**  The reduction in online computation is a compelling argument for the practical value of EnrichIndex.
*   **Reproducibility:** The paper claims to provide data and code which promotes reproducibility.

**Weaknesses:**

*   **Reliance on GPT-40-mini:** The offline enrichment process depends on a specific (though accessible) LLM. Results may vary using different LLMs or smaller models. While the choice of GPT-4o-mini makes it very accessible, this also means that the upper bound in performance is limited, and future work should test with higher-performing models.
*   **Static Corpus Assumption:** The offline enrichment is most effective for relatively static document corpora. The paper doesn't explicitly address scenarios with frequently changing documents or databases which may limit the applicability of the approach.
*   **Limited Discussion of Storage Costs:**  The paper doesn't discuss or quantify the additional storage costs associated with maintaining the enriched indices. While the online cost is significantly reduced, one should at least consider the space-time tradeoff involved.

**Potential Influence:** This paper has the potential to significantly influence the field of LLM-augmented retrieval by providing a practical and scalable approach to incorporating LLM reasoning into retrieval systems. It may encourage researchers to explore offline enrichment strategies and hybrid architectures that combine the strengths of both traditional and LLM-based methods. Furthermore, the findings may be of interest to practitioners in industry who are seeking to deploy LLM-powered retrieval systems in real-world applications.

**Score:** 8.  The paper provides a valuable contribution with a well-defined solution and compelling empirical results that address a recognized bottleneck. The limitations (especially reliance on a single model, and lack of cost analysis) keep it from being a 9 or 10. However, the concept is novel and useful, suggesting a strong positive impact within the community.

- **Score**: 8/10

### **[Multimodal Diffusion Bridge with Attention-Based SAR Fusion for Satellite Image Cloud Removal](http://arxiv.org/abs/2504.03607v1)**
- **Summary**: The paper proposes a novel multimodal diffusion bridge framework, called DB-CR, for cloud removal in satellite images. DB-CR directly bridges between cloudy and cloud-free image distributions, leveraging synthetic aperture radar (SAR) data to guide the reconstruction.  It introduces a novel multimodal diffusion bridge architecture featuring a two-branch backbone with dedicated cross-modality fusion blocks. The method is evaluated on the SEN12MS-CR dataset, demonstrating state-of-the-art results in terms of both distortion and perceptual metrics, while also being computationally efficient. The paper also explores the impact of varying cloud cover and stochastic perturbation on the performance of the model, and discusses the trade-off between distortion and perceptual quality controlled by the number of function evaluations (NFE) during inference.

**Critical Evaluation:**

**Novelty:** The paper presents several novel aspects.  Firstly, the application of diffusion bridges to the cloud removal problem is a novel contribution in itself. While diffusion models have been used for cloud removal, the specific use of a *diffusion bridge*, which constrains the sampling trajectory to move between two known states (cloudy and cloud-free), is a significant departure from standard diffusion approaches. Secondly, the proposed multimodal architecture with a two-branch backbone for SAR and optical data fusion is also innovative.  The use of NAFBlocks for efficient feature extraction and the attention-based SFBlock for cross-modal fusion are well-justified design choices.  Finally, the comprehensive analysis of the distortion-perception trade-off controlled by the NFE provides valuable insights into the behavior of the model.

**Significance:** The results presented in the paper are significant.  The consistent outperformance of DB-CR compared to existing state-of-the-art methods (DSen2-CR, GLF-CR, UnCRtainTS, DiffCR) across various quantitative metrics (PSNR, SSIM, MAE, SAM, FID, LPIPS) demonstrates the effectiveness of the proposed approach. The paper also includes a thorough evaluation across varying cloud cover percentages, confirming the robustness of the model. The analysis of the distortion-perception trade-off and the influence of stochastic perturbation add further value. The reduced computational cost (lower MACs) compared to some baselines is another positive aspect, highlighting the efficiency of the NAFBlock-based architecture.

**Strengths:**
*   **Novel Approach:** Introducing diffusion bridges to cloud removal.
*   **Effective Architecture:** Well-designed two-branch architecture with NAFBlocks and SFBlocks.
*   **Strong Results:** State-of-the-art performance on a standard benchmark (SEN12MS-CR).
*   **Comprehensive Evaluation:** Detailed analysis of various factors affecting performance (cloud cover, NFE, stochastic perturbation).
*   **Computational Efficiency:** Lower MACs compared to some baselines.

**Weaknesses:**
*   **Limited Dataset:**  While SEN12MS-CR is a commonly used dataset, evaluating the model on other datasets with different characteristics (e.g., different sensor resolutions, atmospheric conditions) would further strengthen the claims.
*   **Justification for Sine Wave:** It mentions the use of a sine-based distribution for the at scheduling. While a figure is included, a more in-depth explanation of why this particular scheduling was chosen over alternatives (e.g., linear, cosine) is warranted.
*   **Limited Ablation on NFE Training:** It would be useful to have a clearer discussion of why NFE=1 works well, and why training doesn't need high NFE's to achieve the given results.

**Potential Influence:** The paper has the potential to significantly influence the field of cloud removal in remote sensing.  The diffusion bridge framework offers a promising alternative to standard diffusion models, and the proposed architecture provides a strong foundation for future research in multimodal fusion. The findings regarding the distortion-perception trade-off are also valuable for practical applications.

**Score: 8**

**Rationale:** The paper makes a substantial contribution to the field by introducing a novel and effective method for cloud removal. The demonstrated performance improvements and comprehensive evaluation justify a high score. While there are a few minor weaknesses related to dataset diversity and justification of design choices, the overall impact of the paper is significant, and the score reflects the important contributions made. The innovative architecture and comprehensive experimental design are key strengths.

- **Score**: 8/10

### **[MME-Unify: A Comprehensive Benchmark for Unified Multimodal Understanding and Generation Models](http://arxiv.org/abs/2504.03641v1)**
- **Summary**: Here's a concise summary and critical evaluation of the paper:

**Summary:**

The paper introduces MME-Unify (MME-U), a comprehensive benchmark designed to evaluate unified multimodal large language models (U-MLLMs). Unlike existing benchmarks that assess understanding and generation tasks separately, MME-U evaluates U-MLLMs' ability to integrate these capabilities, focusing on mixed-modality generation tasks. The benchmark includes a diverse set of tasks curated from existing datasets and newly designed subtasks that rigorously assess how a model's understanding and generation capabilities can mutually enhance each other. The authors evaluate 12 U-MLLMs, revealing significant room for improvement, especially in instruction following and image generation quality. The paper highlights the need for a unified evaluation standard and provides a valuable resource for the research community.

**Critical Evaluation:**

*   **Novelty:** The key novelty lies in the creation of a benchmark specifically designed to evaluate the *unified* capabilities of U-MLLMs. While existing benchmarks focus on individual understanding and generation tasks, MME-U is the first attempt to standardize the evaluation of mixed-modality generation, where models leverage both understanding and generation capabilities to solve complex problems. This addresses a critical gap in the existing literature, which often relies on case studies to showcase these capabilities. The design of the "unify" tasks, requiring sequential reasoning and multimodal output at each step (e.g., Visual CoT) is another novel aspect.

*   **Significance:** The significance of this paper is high. As U-MLLMs become increasingly prevalent, the lack of a unified evaluation standard hinders progress and makes it difficult to compare models effectively. MME-U provides a standardized framework for evaluating these models, enabling researchers to identify areas for improvement and track progress over time.  The paper's findings, highlighting the current limitations of U-MLLMs in instruction following and image generation quality, provide valuable insights for future research directions.

*   **Strengths:**
    *   Comprehensive task coverage: MME-U encompasses a wide range of tasks, including multimodal comprehension, generation, and mixed-modality generation.
    *   Standardized evaluation framework: The authors align formats and metrics across datasets to develop a unified evaluation framework, ensuring fair comparisons between models.
    *   Rigorous design of unified tasks: The five designed subtasks are carefully crafted to assess the synergistic interaction between understanding and generation capabilities.
    *   Detailed analysis of model performance: The paper provides a detailed analysis of the strengths and weaknesses of various U-MLLMs, offering valuable insights for future research.
    *   The publicly available benchmark and accompanying code encourage reproducibility and further development.

*   **Weaknesses:**
    *   Reliance on CLIP score: As the authors acknowledge, using the CLIP score for image generation evaluation may introduce certain biases and allow for "hacking" by models that generate images with high similarity scores but do not meet other quality standards.
    *   Simplified unify tasks for evalution: To better control experiment results, some evaluation methods have been simplified, this may limit the understanding of the actual capbilities.
    *   Limited scale. While well-designed, some tasks use a limited number of manually created samples, which may limit the generalizability of the results.
    *   The evaluation focuses on accuracy as the primary metric, which might not capture the nuances of generative tasks, such as creativity or stylistic diversity.

*   **Potential Influence:** The paper has the potential to significantly influence the field by providing a standardized benchmark for evaluating U-MLLMs and guiding future research directions. It may encourage the development of new models that are better able to integrate understanding and generation capabilities. It may also drive the creation of new evaluation metrics that are more robust and less susceptible to manipulation.

**Overall:**

The paper presents a valuable contribution to the field of multimodal learning by introducing a comprehensive benchmark for evaluating U-MLLMs. While the evaluation is not without limitations, MME-U addresses a critical gap in the existing literature and provides a solid foundation for future research.

Score: 8
**Justification:** The paper introduces a genuinely novel and impactful contribution by establishing a much-needed benchmark for a rapidly evolving field. It tackles a significant problem (lack of unified evaluation) with a well-designed solution (MME-U). The strengths of the paper (comprehensiveness, standardization, rigorous task design, detailed analysis) significantly outweigh its weaknesses (reliance on CLIP score, task simplification, limited scale). The benchmark provides a valuable resource for the research community, guiding future research and development efforts. While it's not a perfect benchmark, the MME-U stands out as a pioneering contribution to a dynamic research area, meriting a score of 8.

- **Score**: 8/10

## Other Papers
### **[Think When You Need: Self-Adaptive Chain-of-Thought Learning](http://arxiv.org/abs/2504.03234v1)**
### **[Crash Time Matters: HybridMamba for Fine-Grained Temporal Localization in Traffic Surveillance Footage](http://arxiv.org/abs/2504.03235v1)**
### **[Inherent and emergent liability issues in LLM-based agentic systems: a principal-agent perspective](http://arxiv.org/abs/2504.03255v1)**
### **[Do Large Language Models Solve the Problems of Agent-Based Modeling? A Critical Review of Generative Social Simulations](http://arxiv.org/abs/2504.03274v1)**
### **[FaR: Enhancing Multi-Concept Text-to-Image Diffusion via Concept Fusion and Localized Refinement](http://arxiv.org/abs/2504.03292v1)**
### **[Stance-Driven Multimodal Controlled Statement Generation: New Dataset and Task](http://arxiv.org/abs/2504.03295v1)**
### **[Noise Augmented Fine Tuning for Mitigating Hallucinations in Large Language Models](http://arxiv.org/abs/2504.03302v1)**
### **[Evaluating Compact LLMs for Zero-Shot Iberian Language Tasks on End-User Devices](http://arxiv.org/abs/2504.03312v1)**
### **[QIRL: Boosting Visual Question Answering via Optimized Question-Image Relation Learning](http://arxiv.org/abs/2504.03337v1)**
### **[BabyLM's First Words: Word Segmentation as a Phonological Probing Task](http://arxiv.org/abs/2504.03338v1)**
### **[Sustainable LLM Inference for Edge AI: Evaluating Quantized LLMs for Energy Efficiency, Output Accuracy, and Inference Latency](http://arxiv.org/abs/2504.03360v1)**
### **[Online Difficulty Filtering for Reasoning Oriented Reinforcement Learning](http://arxiv.org/abs/2504.03380v1)**
### **[Locations of Characters in Narratives: Andersen and Persuasion Datasets](http://arxiv.org/abs/2504.03434v1)**
### **[Know What You do Not Know: Verbalized Uncertainty Estimation Robustness on Corrupted Images in Vision-Language Models](http://arxiv.org/abs/2504.03440v1)**
### **[D-Garment: Physics-Conditioned Latent Diffusion for Dynamic Garment Deformations](http://arxiv.org/abs/2504.03468v1)**
### **[Dynamic Importance in Diffusion U-Net for Enhanced Image Synthesis](http://arxiv.org/abs/2504.03471v1)**
### **[Discovering Partially Known Ordinary Differential Equations: a Case Study on the Chemical Kinetics of Cellulose Degradation](http://arxiv.org/abs/2504.03484v1)**
### **[BUFF: Bayesian Uncertainty Guided Diffusion Probabilistic Model for Single Image Super-Resolution](http://arxiv.org/abs/2504.03490v1)**
### **[Diffusion Active Learning: Towards Data-Driven Experimental Design in Computed Tomography](http://arxiv.org/abs/2504.03491v1)**
### **[Neutralizing the Narrative: AI-Powered Debiasing of Online News Articles](http://arxiv.org/abs/2504.03520v1)**
### **[Agentic Knowledgeable Self-awareness](http://arxiv.org/abs/2504.03553v1)**
### **[EnrichIndex: Using LLMs to Enrich Retrieval Indices Offline](http://arxiv.org/abs/2504.03598v1)**
### **[Multimodal Diffusion Bridge with Attention-Based SAR Fusion for Satellite Image Cloud Removal](http://arxiv.org/abs/2504.03607v1)**
### **[AIR: A Systematic Analysis of Annotations, Instructions, and Response Pairs in Preference Dataset](http://arxiv.org/abs/2504.03612v1)**
### **[Multilingual Retrieval-Augmented Generation for Knowledge-Intensive Task](http://arxiv.org/abs/2504.03616v1)**
### **[VISTA-OCR: Towards generative and interactive end to end OCR models](http://arxiv.org/abs/2504.03621v1)**
### **[Align to Structure: Aligning Large Language Models with Structural Information](http://arxiv.org/abs/2504.03622v1)**
### **[Quantifying the uncertainty of model-based synthetic image quality metrics](http://arxiv.org/abs/2504.03623v1)**
### **[Do Larger Language Models Imply Better Reasoning? A Pretraining Scaling Law for Reasoning](http://arxiv.org/abs/2504.03635v1)**
### **[MME-Unify: A Comprehensive Benchmark for Unified Multimodal Understanding and Generation Models](http://arxiv.org/abs/2504.03641v1)**
