# The Latest Daily Papers - Date: 2025-04-27
## Highlight Papers
### **[IberBench: LLM Evaluation on Iberian Languages](http://arxiv.org/abs/2504.16921v1)**
- **Summary**: Here's a summary and critical evaluation of the IberBench paper:

**Summary:**

The paper introduces IberBench, a comprehensive benchmark for evaluating Large Language Models (LLMs) in Iberian languages (Spanish, Portuguese, Catalan, Basque, and Galician, along with various Spanish dialects).  The benchmark addresses limitations of existing evaluations, which are typically English-centric, prioritize fundamental NLP tasks, and are static. IberBench includes 101 datasets spanning 22 task categories, integrates datasets from existing evaluation campaigns and recent benchmarks, and enables continual updates and community-driven submissions.  The authors evaluate 23 LLMs ranging from 100 million to 14 billion parameters and present findings indicating performance disparities across tasks, languages, and model types. The IberBench evaluation pipeline, including dataset normalization and hosting, is open-sourced, along with a public leaderboard.

**Critical Evaluation:**

*   **Novelty:**  The primary novelty lies in the combination of several aspects:  (1) Focus on Iberian languages and dialects, an underrepresented area in LLM evaluation. (2) Integration of a large number of datasets, especially drawing from Iberian workshops (IberLEF, IberEval, TASS, PAN), making previously scattered and difficult-to-access datasets readily available. (3) Emphasizing both fundamental NLP *and* industry-relevant tasks, addressing a bias in existing benchmarks. (4) The establishment of a continual, community-driven benchmark with expert moderation, addressing the static nature of many existing resources. While individual aspects aren't entirely new (e.g., other smaller Iberian benchmarks exist), the scale and comprehensive nature of IberBench, along with its focus on sustained relevance, represents a significant advance. The sequence labeling evaluation integrated into lm-evaluation-harness is a valuable contribution.

*   **Significance:** The significance is considerable. IberBench fills a clear gap in LLM evaluation by providing a valuable resource for assessing performance in Iberian languages.  The open-source nature of the evaluation pipeline and the public leaderboard will facilitate further research and development in this area.  The benchmark’s design promotes continuous improvement through community contributions, ensuring its longevity and relevance.  The findings regarding the performance disparities across tasks and languages provide actionable insights for future LLM development and adaptation for the Iberian context. The paper's insights regarding underperformance in sequence-labeling tasks, relative to shared-task system performance, and the observation that certain smaller models perform below random baselines are important points for LLM developers.

*   **Strengths:**
    *   Comprehensive benchmark design, covering diverse tasks and languages.
    *   Open-source implementation and public leaderboard promotes transparency and collaboration.
    *   Community-driven approach with expert moderation ensures relevance and sustainability.
    *   Empirical analysis provides valuable insights into LLM performance in Iberian languages.
    *   Addresses crucial limitations of existing LLM evaluation practices.

*   **Weaknesses:**
    *   The reliance on existing datasets means IberBench inherits any biases present in those resources. While the authors acknowledge and mitigate contamination risks, it is still a limitation.
    *   The evaluation is primarily zero-shot, potentially underestimating the capabilities of some LLMs. Though the authors justify this decision, exploring few-shot performance in future work could yield additional insights.
    *   The language coverage, while improved, is still not perfectly balanced. Some dialects and languages are underrepresented due to data scarcity. This could impact the generalizability of some of the conclusions.
    *   The reliance on ROUGE-1 for text summarization is a known limitation due to its simplicity, and may not capture the nuanced qualities of LLM generated summaries.

*   **Potential Influence:**  IberBench has the potential to become a standard benchmark for LLM evaluation in Iberian languages, driving research and development in this area. It will likely inform the development of more effective and equitable LLMs for Iberian language speakers. It may also serve as a template for the development of similar benchmarks for other under-resourced language families.

**Justification for the Score:**

I am assigning a score of 8 because the paper addresses a significant gap in LLM evaluation and provides a valuable, well-designed, and open-source resource for the community. The comprehensive nature of the benchmark, its focus on sustained relevance through community involvement, and its empirical findings contribute significantly to the field. The identified weaknesses related to dataset biases, zero-shot evaluation, and reliance on a simple summarization metric, prevent it from achieving a higher score. The high score (i.e. greater than 5) is based on the justification that this is a strong and significant paper, with the potential to make a broad impact on the field of NLP (despite the weakenesses discussed).

Score: 8

- **Score**: 8/10

### **[Safety Pretraining: Toward the Next Generation of Safe AI](http://arxiv.org/abs/2504.16980v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Safety Pretraining: Toward the Next Generation of Safe AI":

**Summary:**

The paper addresses the critical challenge of ensuring safety in large language models (LLMs). Instead of solely relying on post-hoc alignment methods, which are often brittle and superficial, the authors propose a data-centric pretraining framework that embeds safety directly into the model from the start. Their approach involves several key components:

1.  **Safety Filtering:** A robust safety classifier, trained on GPT-4 labeled data, is used to filter harmful content from the pretraining corpus. The authors provide a Data Safety Report Card standard.

2.  **Synthetic Recontextualization:** Potentially harmful content is rephrased and reframed to preserve valuable information while embedding it in a context that underscores its ethical and historical implications. This results in a 100B token "SafeWeb" corpus.

3.  **Native Refusal Training:** The "RefuseWeb" dataset simulates refusal scenarios, guiding models to responsibly disengage from harmful prompts. The "Moral Education" dataset generalizes ethical patterns beyond dialogue.

4.  **Harmfulness-Tag Annotated Training:**  A harmfulness-tag is inserted into training data to signal potentially harmful passages, allowing LLMs to better separate safe and unsafe content. This tag is leveraged at inference time via a "SafeBeam" search algorithm.

5.  **New Evaluation Tools:** The paper presents novel evaluation tools to measure safety under 'completions' enabling the monitoring of safety during pre-training.

The authors evaluate their approach with a 1.7B parameter "SafeLM," demonstrating a significant reduction in attack success rates (ASR) from 38.8% to 8.3% on safety benchmarks without sacrificing performance on standard LLM benchmarks.

**Critical Evaluation:**

*   **Novelty:** The paper's core innovation lies in its comprehensive, data-centric approach to safety **during pretraining**. While individual components such as safety filtering or rephrasing are not entirely new, their synergistic combination, along with the introduction of harmfulness-tag annotation and SafeBeam inference, present a novel and significant advance over prior work.  Specifically, the work builds upon existing literature on pre-training safety interventions and extends the idea of safety annotations to the pre-training phase, which is a natural and valuable extension. The establishment of a Data Safety Report Card is a welcome and needed standardization practice.

*   **Significance:** The increasing deployment of LLMs in high-stakes settings makes safety a paramount concern. The paper's focus on natively safe models, rather than relying solely on alignment, addresses a key weakness in current approaches. The reported reduction in attack success rates, along with the preservation of performance on standard benchmarks, demonstrates the practical utility of their framework. Furthermore, the release of the SafeLM model, datasets, and evaluation tools promotes further research and development in the field. The exploration of the brittleness of post-training safety measures and their failure under benign fine-tuning adds significant weight to the argument for pretraining interventions.

*   **Strengths:**

    *   **Comprehensive Approach:** The integration of multiple data-centric interventions strengthens the model's inherent safety.
    *   **Data-Driven:** The reliance on GPT-4 annotated data and synthetic datasets provides a scalable and adaptable approach to safety.
    *   **Effective Mitigation:** The results demonstrate a substantial reduction in attack success rates compared to baseline models.
    *   **Resource Release:** The open-source release of the SafeLM model, datasets, and evaluation tools promotes reproducibility and further research.
    *   **Clear Problem Definition:** The paper clearly articulates the limitations of post-hoc alignment methods and motivates the need for pretraining-based safety interventions.
*   **Weaknesses:**

    *   **Computational Cost:** The synthetic data generation and annotation process can be computationally expensive and may require specialized resources. The paper could benefit from a more detailed discussion of the scaling properties of each component.
    *   **Model Size:** The SafeLM model, while effective, is relatively small (1.7B parameters) compared to state-of-the-art LLMs. The scalability of the proposed framework to larger models needs further investigation.
    *   **Subjectivity:** The assessment of harmfulness is inherently subjective, and the reliance on GPT-4 for annotation introduces potential biases.
    *   **Overrefusal:** While the paper argues overrefusal isn't a major cost, there is a slight drop in helpfulness that requires some acknowledgement and perhaps some focus on the best mechanisms for balancing safety and helpfulness.

*   **Potential Influence:** The paper has the potential to influence the development of safer and more responsible AI systems. The data-centric pretraining framework provides a practical and effective approach for embedding safety directly into LLMs. The release of the SafeLM model and associated resources could accelerate research and development in the field. The identified brittleness of surface-level alignment methods emphasizes the importance of native safety during pretraining.

**Rigorous Rationale for Score:**
The paper presents a significant advancement in LLM safety by focusing on pretraining interventions, validated by compelling empirical results. While there is potential for improvement, the novelty and importance of addressing LLM safety early in development warrant recognition.

Score: 8

- **Score**: 8/10

### **[DyMU: Dynamic Merging and Virtual Unmerging for Efficient VLMs](http://arxiv.org/abs/2504.17040v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces DYMU (Dynamic Merging and Virtual Unmerging), a novel training-free approach to improve the efficiency of Vision-Language Models (VLMs). DYMU dynamically reduces the number of visual tokens based on image complexity, addressing the inherent inefficiency of fixed-length representations. It comprises two main components: Dynamic Token Merging (DToMe), which reduces the number of visual tokens, and Virtual Token Unmerging (VTU), which simulates the full token sequence for the language model, preserving downstream performance without fine-tuning. The approach is designed to be a plug-and-play solution compatible with various VLM architectures. The authors demonstrate the effectiveness of DYMU through experiments on image and video understanding tasks, showing significant reductions in token count while maintaining competitive performance.

**Critical Evaluation:**

*   **Novelty:** The paper presents a genuinely novel approach to visual token reduction in VLMs. While token reduction techniques exist, DYMU's key contribution lies in its dynamic, training-free nature and its focus on preserving performance without any additional fine-tuning of the VLM. The integration of DToMe and VTU is also a novel combination, enabling the reduction of computational burden at the vision encoder while preserving downstream VLM performance. The size weighted attention mechanism also contributes to the novelty. The adaptive nature of the token merging based on image complexity is a significant step forward from fixed-length compression approaches.
*   **Significance:** The significance of this work stems from its ability to make VLMs more computationally efficient without sacrificing performance. This is crucial for deploying VLMs in resource-constrained environments or scaling them to handle high-resolution images and videos.  The training-free aspect of DYMU enhances its practical applicability, as it can be easily integrated into existing VLM pipelines without incurring additional training costs. The results demonstrating substantial token reduction with minimal performance degradation are significant and will likely be valuable to the VLM research community. The qualitative analysis adds further value by showing how DToMe adapts token reduction to image complexity. The method directly tackles the high computational complexity of VLMs due to large visual token sequences and offers a practical solution.
*   **Strengths:**
    *   **Training-Free:** No additional fine-tuning is needed.
    *   **Dynamic:** Adaptive to image complexity, making it more efficient than fixed-length methods.
    *   **Plug-and-Play:** Works seamlessly with different VLM architectures, vision encoders, and LLMs.
    *   **Significant Token Reduction:** Demonstrates impressive reduction in token count (32%-85%).
    *   **Maintained Performance:** Achieves comparable or better performance compared to full-length models.
    *   **Controllable Cost:** Provides a more fine-grained control over visual token usage and computational cost.
    *   The code for the project is available on Github, enabling other researchers to benefit from the implementation.

*   **Weaknesses:**
    *   The paper could benefit from a more in-depth analysis of the limitations of DYMU. Specifically, a deeper discussion is warranted for instances where spatial awareness is required.
    *   While the wall-clock time difference is mentioned as marginal due to PyTorch optimizations, a more detailed benchmarking comparing end-to-end inference times with and without DYMU across different hardware platforms could be helpful to truly demonstrate the speedup gained.
    * The ablations lack clarity by using percentages on one dataset only. The paper would benefit from presenting additional full dataset ablation studies to support the findings.
*   **Potential Influence:** This work has the potential to significantly influence the field of VLMs. The idea of dynamically adjusting visual token lengths based on image complexity is likely to inspire further research in this area. The training-free nature and easy integration of DYMU could lead to its widespread adoption in practical VLM applications. This solution is particularly relevant for resource-constrained applications.
* **Score Justification:**
A score of 8 is warranted for this paper. The core idea is novel and provides a practical solution to a significant challenge in VLMs (computational efficiency). The training-free and plug-and-play nature enhances its adoption potential. The experiments provide solid evidence of its effectiveness across a range of benchmarks and architectures. The weaknesses noted center around depth of analysis and details of optimization, not fundamental flaws in the approach.

Score: 8

- **Score**: 8/10

### **[Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning](http://arxiv.org/abs/2504.17192v1)**
- **Summary**: Here's a concise summary and critical evaluation of the paper, focusing on novelty, significance, strengths, weaknesses, and overall impact.

**Summary:**

The paper introduces Paper2Code, a novel multi-agent LLM framework designed to automatically generate executable code repositories from machine learning research papers. Paper2Code emulates the human developer lifecycle by breaking down the task into three stages: planning (high-level roadmap, architecture design, file dependencies), analysis (fine-grained interpretation of individual file functionality), and generation (modular, dependency-aware code synthesis). The framework utilizes specialized LLM agents for each stage, collaborating across the pipeline. The paper evaluates Paper2Code on a new Paper2Code benchmark and the existing PaperBench benchmark using model-based and human evaluations. Results demonstrate Paper2Code's effectiveness in generating high-quality, faithful implementations, surpassing strong baselines.

**Critical Evaluation:**

*   **Novelty:** The paper's primary novelty lies in the end-to-end automation of code repository generation *directly* from research papers *without relying on existing code, APIs, or supplementary materials*. While LLMs have been used for code generation and experiment automation before, the focus on generating complete, executable repositories solely from paper content represents a significant advancement. The multi-agent framework and structured approach (plan-analyze-implement) are also novel contributions, reflecting established software engineering principles.
*   **Significance:** The potential impact of Paper2Code is considerable. By automating code generation, the framework can significantly accelerate the scientific process by:

    *   Improving reproducibility: Providing readily available, faithful code implementations enables researchers to validate findings more easily.
    *   Facilitating building upon prior work: Researchers can more quickly understand and adapt existing methods when code is available.
    *   Democratizing research: Lowering the barrier to entry for reproducing and extending research by reducing the manual effort required.
*   **Strengths:**

    *   Strong performance: The experimental results on both the Paper2Code and PaperBench benchmarks demonstrate the effectiveness of Paper2Code in generating high-quality code.
    *   Comprehensive evaluation: The use of model-based and human evaluations, including evaluations by original paper authors, provides a well-rounded assessment of the framework.
    *   Clear framework design: The multi-agent architecture and structured pipeline offer a logical and well-defined approach to the problem.
    *   Addressing a critical need: The paper tackles a real-world problem of code unavailability in machine learning research, providing a practical solution.

*   **Weaknesses:**

    *   Limited Scope: The evaluation dataset may not fully represent the breadth of machine learning research papers, focusing more on implementations and architectures. It can be difficult to fully extract sufficient data from very theoretical work or work relying on inaccessible data.
    *   Potential dependency on LLM progress: The framework's performance depends heavily on the capabilities of the underlying LLMs. Advances in LLM capabilities are necessary to further refine the implementations for complex codebases.
    *   Scalability could be limited: While the 70k token limitation can be reasonable, it is unclear how this approach will be able to handle extremely large and complicated code bases.

**Justification for Score:**

Considering the novelty, significance, strengths, and weaknesses, I assign a score of **8.5**.

*   The paper presents a novel and well-executed approach to a significant problem in the machine learning community: improving code availability and reproducibility. The demonstrated performance improvements over baselines and the positive feedback from human evaluators are compelling.

*   However, the framework's reliance on LLMs and limited scope represent limitations that need to be addressed in future work. The paper addresses an important step toward automation, and represents very solid ground for future improvement.

Score: 8.5

- **Score**: 8/10

### **[3DV-TON: Textured 3D-Guided Consistent Video Try-on via Diffusion Models](http://arxiv.org/abs/2504.17414v1)**
- **Summary**: Here's a summary and critical evaluation of the 3DV-TON paper:

**Summary:**

The paper "3DV-TON: Textured 3D-Guided Consistent Video Try-on via Diffusion Models" introduces a novel framework for video try-on that leverages textured 3D guidance to improve the temporal consistency and visual quality of generated videos.  Instead of relying solely on pixel-based reconstruction losses common in diffusion models, the method uses animatable textured 3D meshes of humans wearing the target clothing. This explicit 3D guidance ensures that the garment's texture and shape remain consistent across different poses and viewpoints in the video.  The method includes a pipeline for generating dynamic 3D guidance from a keyframe image try-on, a rectangular masking strategy to prevent clothing information leakage, and a diffusion-based architecture for image synthesis. The paper also introduces a new high-resolution video try-on dataset (HR-VVT) for benchmarking.  The results demonstrate that 3DV-TON outperforms existing methods in both quantitative and qualitative evaluations.

**Critical Evaluation:**

**Novelty:**

The novelty lies primarily in the use of *textured* 3D guidance within a diffusion-based video try-on framework. Existing methods either rely on warping, geometric 3D priors alone, or pixel-based reconstruction losses which often struggle with temporal consistency. By using an *explicit, textured* 3D mesh that is animated in sync with the video, the model gains a much stronger constraint on the shape and appearance of the clothing throughout the sequence. This is a significant step forward. The adaptive 3D mesh creation pipeline and the masking strategy are also worthwhile contributions that address practical challenges. The new HR-VVT dataset is a valuable resource for the community, addressing the limitations of existing benchmarks.

**Significance:**

The paper tackles a significant problem in video try-on: maintaining temporal consistency and visual fidelity while handling complex motions and clothing deformations. The demonstrated improvements are substantial, leading to more realistic and usable video try-on experiences.  The introduction of the HR-VVT dataset is also important as it provides a more challenging benchmark for evaluating future research.  The method's ability to handle diverse clothing styles and body poses is a step towards making video try-on a more practical technology for e-commerce and other applications.

**Strengths:**

*   **Clear problem statement and well-defined solution:** The paper clearly identifies the limitations of existing video try-on methods and proposes a targeted solution based on textured 3D guidance.
*   **Solid technical approach:** The method is well-engineered, combining diffusion models with 3D mesh reconstruction and animation techniques. The pipeline for generating the 3D guidance is reasonably complex.
*   **Strong experimental results:**  Both qualitative and quantitative results demonstrate the superiority of the proposed method over existing approaches. The ablation studies provide valuable insights into the effectiveness of different components.
*   **Valuable dataset:** The introduction of the HR-VVT dataset addresses a recognized need for more challenging and high-resolution benchmarks in the field.
*   **Good writing and presentation:** The paper is well-written and clearly explains the technical details of the proposed method. The figures and tables are informative and effectively illustrate the results.

**Weaknesses:**

*   **Computational cost:** The 3D mesh reconstruction step is computationally expensive. While the paper mentions optimizations, the reconstruction time could still be a bottleneck in some applications. While the improved feedforward clothed human reconstruction may address this moving forward, it does represent a weakness of the current method.
*   **Dependence on HPS regressors:** Performance depends on the quality of initial human pose and shape estimates from HPS regressors which may not be perfectly accurate. Reliance on external libraries and data-driven HPS can introduce bias or errors.
*   **Masking artifacts:** While the masking strategy helps to prevent clothing leakage, it might also introduce artifacts or limit the ability to generate highly detailed or intricate clothing styles.
*   **Potential for negative societal impact:** Like all video generation technologies, there is a potential for misuse, such as the creation of deepfakes or the spread of misinformation.

**Potential Influence:**

The paper is likely to have a significant influence on the field of video try-on. The idea of using textured 3D guidance as a strong constraint for temporal consistency is a valuable contribution that could be adopted by other researchers. The HR-VVT dataset will likely become a standard benchmark for evaluating future video try-on methods. Additionally, the pipeline developed will have applications to additional video generation projects.

**Justification of Score:**

The paper presents a solid technical contribution with significant improvements over existing methods. The textured 3D guidance is a novel and effective way to address the problem of temporal inconsistency in video try-on. The HR-VVT dataset is a valuable resource for the community. While there are some limitations (computational cost, dependence on HPS regressors, potential masking artifacts), the overall impact of the paper is positive.

Score: 8

- **Score**: 8/10

### **[Unified Attacks to Large Language Model Watermarks: Spoofing and Scrubbing in Unauthorized Knowledge Distillation](http://arxiv.org/abs/2504.17480v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper proposes Contrastive Decoding-Guided Knowledge Distillation (CDG-KD), a novel framework for attacking watermark schemes in large language models (LLMs) under unauthorized knowledge distillation scenarios. CDG-KD enables both watermark scrubbing (removal) and spoofing (forgery) attacks by leveraging contrastive decoding to extract corrupted or amplified watermark text from a student model and a weakly watermarked reference model. This data is then used for bidirectional distillation to train new student models capable of either removing or forging watermarks.  The method operates entirely in a black-box setting, requiring no access to internal model parameters or modifications to the generation process. Experiments demonstrate the effectiveness of CDG-KD in performing both types of attacks while preserving the general performance of the distilled model, highlighting vulnerabilities in current watermarking schemes.

**Critical Evaluation:**

*   **Novelty:** The paper introduces a novel, unified framework for conducting both scrubbing and spoofing attacks on watermarked LLMs under unauthorized knowledge distillation. Existing attack methods typically focus on one type of attack or require access to model internals. CDG-KD's black-box approach and bidirectional distillation strategy represent a significant advance in this area. The use of contrastive decoding to manipulate watermark strength is also a novel aspect. The exploration of idiosyncrasies of watermarking schemes, and its impact on detectability is an interesting contribution.

*   **Significance:** The paper highlights a critical vulnerability in existing LLM watermarking schemes: their susceptibility to indirect attacks through unauthorized knowledge distillation. Watermark radioactivity, while intended as a detection mechanism for illicit model copying, can be exploited by adversaries to undermine the integrity of the watermark, enabling both removal and forgery. The paper's findings underscore the urgent need for more robust and unforgeable watermarking techniques. The practical demonstration of attacks in a black-box setting makes the threat very real and immediate.

*   **Strengths:**
    *   The framework is unified, enabling both scrubbing and spoofing attacks.
    *   It operates in a black-box setting, making it highly practical.
    *   Contrastive decoding is an effective technique for manipulating watermark strength.
    *   Experiments demonstrate strong attack performance while preserving model utility.
    *   The paper clearly articulates the threat model and potential consequences.

*   **Weaknesses:**
    *   The method relies on access to a student model already distilled from a teacher model, which may not always be the case in real-world scenarios. However, it seems realistic assumption as this is how the adversaries will try to save computation costs by leveraging distillation on a larger model.
    *   The effectiveness of the attack is tied to the size and quality of the distillation corpus.
    *   The generalizability to different watermarking paradigms, particularly sentence-level schemes, is not fully explored. The experiments focuses mostly on token-level watermarking strategy.
    *   Ethical considerations are briefly mentioned, but a more in-depth discussion of the potential for misuse of the framework would be beneficial.

*   **Potential Influence:** This paper is likely to have a significant impact on the field of LLM watermarking. It exposes a serious vulnerability that must be addressed to ensure the trustworthiness and accountability of these models. The CDG-KD framework provides a valuable tool for evaluating the robustness of existing and future watermarking schemes. The research will drive further research into more resistant and unforgeable watermarking methods. The ability to measure watermark detectability as a classification problem is a useful additional finding.

**Justification for Score:**

This is a well-executed and significant piece of research. It makes a clear contribution by exposing a real and practical attack vector against LLM watermarking. The unified approach, black-box setting, and experimental validation make a strong case for its importance. While the method has some limitations (primarily related to the reliance on a distilled student model and dataset size), these do not detract significantly from its overall value.

Score: 8

- **Score**: 8/10

### **[RefVNLI: Towards Scalable Evaluation of Subject-driven Text-to-image Generation](http://arxiv.org/abs/2504.17502v1)**
- **Summary**: Okay, let's break down this paper and evaluate its contribution.

**Summary:**

The paper introduces REFVNLI, a novel, cost-effective metric for automatically evaluating subject-driven text-to-image (T2I) generation. Unlike existing evaluation methods that either focus solely on textual alignment or subject preservation or rely on expensive API calls, REFVNLI assesses both aspects within a single framework. The authors train REFVNLI on a large-scale, automatically generated dataset derived from video-reasoning benchmarks and image perturbations.  Experiments demonstrate that REFVNLI outperforms or matches existing baselines across multiple benchmarks and subject categories, even for lesser-known concepts, while aligning well with human preferences.

**Critical Evaluation:**

*   **Novelty:** The paper's core novelty lies in its design and implementation of a *unified* and *cost-effective* automatic metric for subject-driven T2I evaluation. Existing metrics often address textual alignment *or* subject preservation separately, or involve expensive external API calls.  REFVNLI aims to provide a balanced and efficient alternative. While components of REFVNLI draw inspiration from earlier work (e.g., using a pre-trained vision-language model, fine-tuning on a specifically curated dataset), the specific combination and the automated dataset construction are significant departures. The dataset creation process, which leverages video frames and image perturbations, is a crucial element that ensures robustness to identity-agnostic variations.

*   **Significance:** The paper's significance stems from addressing a critical bottleneck in the rapidly growing field of subject-driven T2I generation: *the lack of reliable automatic evaluation*.  Subject-driven T2I has broad potential applications, including personalized image generation and consistent character representation in video content.  The absence of accurate and efficient evaluation metrics hinders progress by making it difficult to quantitatively assess the performance of different models and identify areas for improvement. REFVNLI directly tackles this issue, potentially accelerating research in the field.

*   **Strengths:**

    *   **Unified Evaluation:**  REFVNLI provides a single metric encompassing both textual alignment and subject preservation, simplifying the evaluation process.
    *   **Cost-Effectiveness:** The model's reliance on a fine-tuned VLM and an automatically generated dataset makes it significantly more cost-effective than API-based alternatives like DreamBench++ and VIEScore.
    *   **Strong Performance:** Experimental results demonstrate that REFVNLI achieves competitive or superior performance compared to existing baselines across a range of benchmarks and subject categories.
    *   **Human Alignment:**  The paper highlights that REFVNLI aligns well with human preferences, particularly for rare entities, making it a more reliable metric.
    *   **Detailed Ablation:** The thorough ablation study provides valuable insights into the importance of different design choices within REFVNLI, such as the dual-binary classification scheme and the inclusion of subject markup.

*   **Weaknesses:**

    *   **Reliance on Automated Dataset Generation:** While the automated dataset creation process is a strength, it also introduces a potential weakness. The quality of the training data depends on the performance of the LLMs (e.g. Gemini) and object detection models used in the pipeline. Potential biases or inaccuracies in these models could propagate into the REFVNLI metric. While the filtering process in Section 9.1 attempts to address this, it is not foolproof.
    *   **Potential Overfitting:** As the model is specifically fine-tuned on a generated dataset, it might be overfitting to certain types of artifacts or biases within that dataset. Generalizability to completely unseen data or different evaluation paradigms might be limited.
    *   **Limited Negative Examples:** The technique for creating hard-negative examples for textual alignment seems very good, but still relies on modifications to the prompt, which can be imperfect.
    *   **Landmark Performance Dip:** The reduced performance on subject consistency for landmarks indicates a potential area for improvement. The authors attribute this to hypersensitivity to minor discrepancies, which warrants further investigation.  The paper acknowledges this weakness and proposes avenues for future work.

*   **Potential Influence:** REFVNLI has the potential to become a valuable tool for researchers working on subject-driven T2I generation. By providing an accurate, efficient, and cost-effective evaluation metric, it could facilitate the development of more robust and human-aligned T2I models. The automatically generated dataset and the insights from the ablation study could also be valuable resources for the community.

**Justification for the Score:**

While REFVNLI is not a groundbreaking theoretical breakthrough, it represents a *significant engineering advancement* with substantial practical value. It cleverly combines existing techniques with a novel dataset generation strategy to address a key problem in a rapidly evolving field. The thorough experimental evaluation and the detailed ablation study provide strong evidence for its effectiveness and potential. However, the reliance on automated dataset generation and the potential for overfitting are important limitations that must be considered. Taking these strengths and weaknesses into account I believe this is an important advance.

**Score: 8**

- **Score**: 8/10

### **[HalluLens: LLM Hallucination Benchmark](http://arxiv.org/abs/2504.17550v1)**
- **Summary**: Here's a summary and critical evaluation of the "HalluLens: LLM Hallucination Benchmark" paper:

**Summary:**

The paper introduces HalluLens, a comprehensive benchmark for evaluating hallucinations in Large Language Models (LLMs). The authors differentiate between "intrinsic" (inconsistent with input) and "extrinsic" (inconsistent with training data) hallucinations, emphasizing the latter's importance as LLMs evolve. HalluLens includes both existing and newly introduced extrinsic hallucination tasks, designed with dynamic test set generation to mitigate data leakage. The benchmark includes tasks like PreciseWikiQA, LongWiki, and NonExistentRefusal. The authors also analyze existing benchmarks (TruthfulQA, SimpleQA, HaluEval), highlighting their limitations, and propose a taxonomy of hallucinations.

**Critical Evaluation:**

*   **Strengths:**
    *   **Clear Taxonomy:** The paper's strength lies in its clear and well-defined taxonomy of LLM hallucinations. Disentangling hallucination from "factuality" is crucial and addresses a common source of confusion in the literature.
    *   **Focus on Extrinsic Hallucination:** Highlighting the importance of extrinsic hallucination as LLMs evolve is insightful. Most existing benchmarks focus more on factuality, which relies on external knowledge, or intrinsic hallucination.
    *   **Dynamic Test Set Generation:** Addressing data leakage, a significant challenge in benchmarking, through dynamic test set generation is a major contribution. This approach adds robustness and prevents benchmarks from quickly becoming obsolete.
    *   **Comprehensive Analysis of Existing Benchmarks:** The critical evaluation of existing benchmarks, exposing their limitations and saturation, is very valuable for researchers and practitioners.  Pointing out flaws in metrics (e.g., TruthfulQA) adds substantial value.
    *   **Practical Evaluation:** The paper provides empirical results of evaluating several models against the new and established benchmarks, giving the scientific community valuable and practical starting points for evaluating their models.

*   **Weaknesses:**
    *   **Reliance on LLMs for Evaluation:** While practical, the automatic evaluation method (LLM as judge) is a potential weakness. LLMs are known to hallucinate, and their judgment may not always be reliable, even with the high claimed accuracy in judgments by the LLM. Although, the prompt used to evaluate the judgements appears well designed, which helps mitigate the issue.
    *   **Complexity in implementation may limit uptake:** Dynamic test set generation and claim generation may limit uptake from some research groups, but provides significantly higher data security.

*   **Novelty and Significance:** The paper is relatively novel and significant because it clearly defines types of hallucinations, focuses on previously under-emphasized extrinsic hallucinations, introduces techniques (dynamic dataset regeneration) to address limitations with existing benchmarks and makes a concerted effort to delineate hallucination from factuality.

*   **Potential Influence:** HalluLens has the potential to become a widely adopted benchmark for LLM hallucination evaluation.  The focus on extrinsic hallucination and the dynamic test set generation could influence future research directions in this area. The practical insights on various models also aids the scientific community. The definitions provided will clarify future research.

*   **Justification for Score:** The paper scores highly due to its contributions to addressing the critical problem of hallucinations in LLMs. The clear definitions, new benchmark tasks, and dynamic generation approach are significant improvements over existing methods. While relying on LLMs for evaluation could be viewed as a limitation, the overall contribution to the field is substantial.

**Score: 8.5**

- **Score**: 8/10

### **[polyGen: A Learning Framework for Atomic-level Polymer Structure Generation](http://arxiv.org/abs/2504.17656v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "polyGen - A Learning Framework for Atomic-level Polymer Structure Generation":

**Summary:**

The paper introduces polyGen, a novel learning framework using latent diffusion models to generate realistic 3D atomic structures of synthetic polymers. polyGen addresses the challenge of creating initial structures for polymer simulations, which has been a bottleneck in polymer informatics. The model takes a polymer repeat unit's chemical connectivity as input (e.g., from a SMILES string) and generates diverse conformations, accounting for the flexibility and structural disorder inherent in polymers. The framework uses a variational autoencoder (VAE) to learn an atom-wise latent space, trained on a dataset of DFT-optimized polymer structures, which is further augmented with DFT-optimized small-molecule structures (QM9 dataset) to improve local structural learning.  A diffusion model is then used to generate structures in the latent space, followed by a filtering step to ensure physical plausibility. The authors introduce evaluation criteria grounded in first principles to benchmark the quality of generated conformations, focusing on bond lengths, angles, and dihedrals.

**Critical Evaluation:**

*   **Novelty:**

    *   The paper presents a highly novel approach. Generative models, especially diffusion models, have seen success in inorganic crystals, biopolymers (proteins), and small molecules, but this is the first application I've seen specifically tackling the complexities of *synthetic* polymer structure generation. The design considerations—accounting for long-chain interactions, managing conformational diversity, and handling the wide design space of repeat units—are clearly articulated and thoughtfully addressed in the architecture. The introduction of a molecular encoding capturing connectivity within the architecture is also a good design choice.
    *   The integration of a diffusion model with an autoencoder, conditioned on molecular connectivity, is a well-motivated approach.
    *   The introduction of rigorous evaluation criteria for polymer conformation generation is a significant contribution. The forward KL divergence of bond lengths, angles, and dihedrals offers a physics-based assessment that goes beyond simple RMSD comparisons.

*   **Significance:**

    *   The potential impact of polyGen is substantial.  If successful and further refined, it could significantly accelerate polymer discovery by providing realistic initial structures for simulations, enabling faster property prediction, surrogate modeling, and ultimately, rational design.
    *   The framework addresses a critical need in the polymer simulation community – the rapid and automated generation of realistic starting conformations for downstream analysis.
    *   The benchmarking and evaluation metrics will likely become standard in the field, pushing towards more rigorous and reproducible results in generative polymer modeling.

*   **Strengths:**

    *   The problem definition is clear and well-motivated. The challenges unique to polymer structure generation are thoroughly discussed.
    *   The method is technically sound, integrating a diffusion model with a molecular connectivity encoding.
    *   The joint training approach using both polymer and small molecule datasets is insightful and shows performance improvements.
    *   The evaluation is comprehensive, using both qualitative visualizations and quantitative metrics like KL divergence.
    *   The authors acknowledge the limitations of their model (e.g., performance degradation with larger systems) and suggest avenues for future work.

*   **Weaknesses:**

    *   The limited dataset size (3855 DFT-optimized structures) is a significant constraint. While data augmentation techniques are employed, the model's ability to generalize to truly novel polymer chemistries may be limited.  The performance degradation on larger systems clearly reflects this.
    *   The model exhibits some precision issues, particularly in bond length predictions. The ability to distinguish clearly between single and double bonds is currently limited.
    *   The inclusion of only DFT structures could be a limitation, specifically when generating realistic polymer conformations at room temperature, where the effect of temperature influences the conformational flexibility, which is a property captured through methods such as molecular dynamics.
    *   While good, the correlation to ground truth z-height of 0.822 is not as strong as it could be, which has an effect on capturing realistic chain packing structures.

*   **Potential Influence:**

    *   The paper has the potential to significantly influence the field of polymer informatics and materials discovery.  It provides a strong foundation for future research in generative polymer modeling.
    *   The proposed evaluation metrics will likely be adopted by other researchers in the field.
    *   The code availability (upon publication) will further facilitate adoption and extension of the framework.

*   **Justification for Score:**

    While the paper presents a novel and significant contribution, the limitations in dataset size and some precision issues prevent it from receiving a top score. The impact is contingent on expanding the dataset and refining the model to handle larger systems and achieve higher accuracy.  The thorough methodology, clear articulation of challenges, and potential for significant impact justify a high score, but the acknowledged limitations necessitate a slight reduction.

Score: 8

- **Score**: 8/10

### **[LiDPM: Rethinking Point Diffusion for Lidar Scene Completion](http://arxiv.org/abs/2504.17791v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "LiDPM: Rethinking Point Diffusion for Lidar Scene Completion":

**Summary:**

The paper presents LiDPM, a novel approach to lidar scene completion using point diffusion models. It challenges the prevailing notion that local diffusion processes are necessary for handling large-scale point clouds in scene completion. Instead, the authors demonstrate that a vanilla Denoising Diffusion Probabilistic Model (DDPM) with a carefully chosen starting point can achieve superior results on the SemanticKITTI dataset. LiDPM unifies the practices of point diffusion for shapes (objects) and scenes, simplifying the process and enabling both scene completion and generation. The method leverages classifier-free guidance and replaces batch normalization layers with instance normalization to improve stability during training.

**Critical Evaluation:**

*   **Novelty:** The primary contribution of the paper lies in challenging the necessity of local diffusion for large-scale point cloud completion. While DDPMs are well-established, their successful adaptation and performance for scene completion *without* resorting to local modifications represent a significant simplification and a valuable insight. The careful selection of a starting point within the diffusion process and the use of instance normalization contribute to the effectiveness of the approach. The combination of these aspects adds to the novelty.
*   **Significance:** The paper has several implications for the field of lidar scene completion:
    *   **Simplification:** It demonstrates that existing DDPM frameworks can be effectively adapted without needing complex local diffusion formulations. This reduces the complexity of implementation and training.
    *   **Performance:** The results on SemanticKITTI demonstrate a clear performance improvement over local diffusion-based methods, making it a valuable contribution from a practical perspective.
    *   **Generative Potential:** The ability to perform both completion and generation using a single unified framework is a significant advantage, enabling the creation of augmented datasets and simulations.

*   **Strengths:**
    *   **Clear problem statement and motivation:** The paper clearly articulates the limitations of existing approaches and motivates the need for a simpler, unified framework.
    *   **Strong empirical results:** The results on SemanticKITTI provide convincing evidence of the effectiveness of LiDPM, with comprehensive comparisons to existing methods.
    *   **Well-structured and clearly written:** The paper is well-organized and easy to follow, making the technical details accessible to the reader.
    *   **Thorough analysis:** The ablation studies provide valuable insights into the importance of different design choices.

*   **Weaknesses:**
    *   **Reliance on existing refinement network:**  The method relies on the refinement network from LiDiff for densification. While this demonstrates that LiDPM can be readily integrated with existing architectures, it would strengthen the paper to showcase a refinement network tailored specifically for LiDPM.
    *   **Limited discussion of computational cost:** While the simplification of the diffusion process is a strength, the paper lacks a detailed analysis of the computational cost compared to local diffusion methods. This would be important for assessing the practical applicability of LiDPM.
    *   **Lack of a rigorous theoretical underpinning:** The rationale for the chosen starting point in the diffusion process could be further strengthened with a more rigorous theoretical analysis. While the ablation studies support its effectiveness, a deeper understanding of why it works would add value.

*   **Impact:** The paper is likely to have a significant impact on the field of lidar scene completion. The simplicity and effectiveness of LiDPM make it a promising approach for future research and applications. It also challenges the assumptions underlying existing methods, potentially leading to new avenues of investigation.

**Justification of Score:**

LiDPM presents a significant advancement in lidar scene completion by simplifying the diffusion process without sacrificing performance. The empirical results are compelling, and the paper is well-written and easy to understand. While there are areas for improvement, the paper's novelty, significance, and potential impact warrant a high score. Considering both the strengths and weaknesses, a score of 8 is assigned, recognizing the substantial contribution while acknowledging the potential for further refinement and analysis.

**Score: 8**

- **Score**: 8/10

## Other Papers
### **[IberBench: LLM Evaluation on Iberian Languages](http://arxiv.org/abs/2504.16921v1)**
### **[Safety Pretraining: Toward the Next Generation of Safe AI](http://arxiv.org/abs/2504.16980v1)**
### **[(Im)possibility of Automated Hallucination Detection in Large Language Models](http://arxiv.org/abs/2504.17004v1)**
### **[LLM impact on BLV programming](http://arxiv.org/abs/2504.17018v1)**
### **[Optimizing LLMs for Italian: Reducing Token Fertility and Enhancing Efficiency Through Vocabulary Adaptation](http://arxiv.org/abs/2504.17025v1)**
### **[DyMU: Dynamic Merging and Virtual Unmerging for Efficient VLMs](http://arxiv.org/abs/2504.17040v1)**
### **[Do Words Reflect Beliefs? Evaluating Belief Depth in Large Language Models](http://arxiv.org/abs/2504.17052v1)**
### **[Statistical Guarantees in Synthetic Data through Conformal Adversarial Generation](http://arxiv.org/abs/2504.17058v1)**
### **[Distilling semantically aware orders for autoregressive image generation](http://arxiv.org/abs/2504.17069v1)**
### **[Robo-Troj: Attacking LLM-based Task Planners](http://arxiv.org/abs/2504.17070v1)**
### **[Physics-guided and fabrication-aware inverse design of photonic devices using diffusion models](http://arxiv.org/abs/2504.17077v1)**
### **[Leveraging LLMs as Meta-Judges: A Multi-Agent Framework for Evaluating LLM Judgments](http://arxiv.org/abs/2504.17087v1)**
### **[Co-CoT: A Prompt-Based Framework for Collaborative Chain-of-Thought Reasoning](http://arxiv.org/abs/2504.17091v1)**
### **[The Rise of Small Language Models in Healthcare: A Comprehensive Survey](http://arxiv.org/abs/2504.17119v1)**
### **[Steering the CensorShip: Uncovering Representation Vectors for LLM "Thought" Control](http://arxiv.org/abs/2504.17130v1)**
### **[MIRAGE: A Metric-Intensive Benchmark for Retrieval-Augmented Generation Evaluation](http://arxiv.org/abs/2504.17137v1)**
### **[AUTHENTICATION: Identifying Rare Failure Modes in Autonomous Vehicle Perception Systems using Adversarially Guided Diffusion Models](http://arxiv.org/abs/2504.17179v1)**
### **[Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning](http://arxiv.org/abs/2504.17192v1)**
### **[Automatically Generating Rules of Malicious Software Packages via Large Language Model](http://arxiv.org/abs/2504.17198v1)**
### **[A RAG-Based Multi-Agent LLM System for Natural Hazard Resilience and Adaptation](http://arxiv.org/abs/2504.17200v1)**
### **[High-Fidelity And Complex Test Data Generation For Real-World SQL Code Generation Services](http://arxiv.org/abs/2504.17203v1)**
### **[Visual and textual prompts for enhancing emotion recognition in video](http://arxiv.org/abs/2504.17224v1)**
### **[FLAG: Formal and LLM-assisted SVA Generation for Formal Specifications of On-Chip Communication Protocols](http://arxiv.org/abs/2504.17226v1)**
### **[Scene Perceived Image Perceptual Score (SPIPS): combining global and local perception for image quality assessment](http://arxiv.org/abs/2504.17234v1)**
### **[NeuralGrok: Accelerate Grokking by Neural Gradient Transformation](http://arxiv.org/abs/2504.17243v1)**
### **[Low-Resource Neural Machine Translation Using Recurrent Neural Networks and Transfer Learning: A Case Study on English-to-Igbo](http://arxiv.org/abs/2504.17252v1)**
### **[DIVE: Inverting Conditional Diffusion Models for Discriminative Tasks](http://arxiv.org/abs/2504.17253v1)**
### **[JurisCTC: Enhancing Legal Judgment Prediction via Cross-Domain Transfer and Contrastive Learning](http://arxiv.org/abs/2504.17264v1)**
### **[Towards Generalized and Training-Free Text-Guided Semantic Manipulation](http://arxiv.org/abs/2504.17269v1)**
### **[Combining Static and Dynamic Approaches for Mining and Testing Constraints for RESTful API Testing](http://arxiv.org/abs/2504.17287v1)**
### **[AI-Enhanced Business Process Automation: A Case Study in the Insurance Domain Using Object-Centric Process Mining](http://arxiv.org/abs/2504.17295v1)**
### **[CoheMark: A Novel Sentence-Level Watermark for Enhanced Text Quality](http://arxiv.org/abs/2504.17309v1)**
### **[FLUKE: A Linguistically-Driven and Task-Agnostic Framework for Robustness Evaluation](http://arxiv.org/abs/2504.17311v1)**
### **[DIMT25@ICDAR2025: HW-TSC's End-to-End Document Image Machine Translation System Leveraging Large Vision-Language Model](http://arxiv.org/abs/2504.17315v1)**
### **[Exploring Context-aware and LLM-driven Locomotion for Immersive Virtual Reality](http://arxiv.org/abs/2504.17331v1)**
### **[Bridging Cognition and Emotion: Empathy-Driven Multimodal Misinformation Detection](http://arxiv.org/abs/2504.17332v1)**
### **[Fine-Grained Fusion: The Missing Piece in Area-Efficient State Space Model Acceleration](http://arxiv.org/abs/2504.17333v1)**
### **[TimeChat-Online: 80% Visual Tokens are Naturally Redundant in Streaming Videos](http://arxiv.org/abs/2504.17343v1)**
### **[DRC: Enhancing Personalized Image Generation via Disentangled Representation Composition](http://arxiv.org/abs/2504.17349v1)**
### **[PatientDx: Merging Large Language Models for Protecting Data-Privacy in Healthcare](http://arxiv.org/abs/2504.17360v1)**
### **[TimeSoccer: An End-to-End Multimodal Large Language Model for Soccer Commentary Generation](http://arxiv.org/abs/2504.17365v1)**
### **[LiveLongBench: Tackling Long-Context Understanding for Spoken Texts from Live Streams](http://arxiv.org/abs/2504.17366v1)**
### **[On-Device Qwen2.5: Efficient LLM Inference with Model Compression and Hardware Acceleration](http://arxiv.org/abs/2504.17376v1)**
### **[Assessing the Capability of Large Language Models for Domain-Specific Ontology Generation](http://arxiv.org/abs/2504.17402v1)**
### **[3DV-TON: Textured 3D-Guided Consistent Video Try-on via Diffusion Models](http://arxiv.org/abs/2504.17414v1)**
### **[Towards Harnessing the Collaborative Power of Large and Small Models for Domain Tasks](http://arxiv.org/abs/2504.17421v1)**
### **[Towards Leveraging Large Language Model Summaries for Topic Modeling in Source Code](http://arxiv.org/abs/2504.17426v1)**
### **[Beyond Whole Dialogue Modeling: Contextual Disentanglement for Conversational Recommendation](http://arxiv.org/abs/2504.17427v1)**
### **[Breaking the Modality Barrier: Universal Embedding Learning with Multimodal LLMs](http://arxiv.org/abs/2504.17432v1)**
### **[Adaptive Orchestration of Modular Generative Information Access Systems](http://arxiv.org/abs/2504.17454v1)**
### **[Unified Attacks to Large Language Model Watermarks: Spoofing and Scrubbing in Unauthorized Knowledge Distillation](http://arxiv.org/abs/2504.17480v1)**
### **[Combining GCN Structural Learning with LLM Chemical Knowledge for or Enhanced Virtual Screening](http://arxiv.org/abs/2504.17497v1)**
### **[RefVNLI: Towards Scalable Evaluation of Subject-driven Text-to-image Generation](http://arxiv.org/abs/2504.17502v1)**
### **[ESDiff: Encoding Strategy-inspired Diffusion Model with Few-shot Learning for Color Image Inpainting](http://arxiv.org/abs/2504.17524v1)**
### **[Text-to-Image Alignment in Denoising-Based Models through Step Selection](http://arxiv.org/abs/2504.17525v1)**
### **[Towards Machine-Generated Code for the Resolution of User Intentions](http://arxiv.org/abs/2504.17531v1)**
### **[Auditing the Ethical Logic of Generative AI Models](http://arxiv.org/abs/2504.17544v1)**
### **[A Comprehensive Survey of Knowledge-Based Vision Question Answering Systems: The Lifecycle of Knowledge in Visual Reasoning Task](http://arxiv.org/abs/2504.17547v1)**
### **[HalluLens: LLM Hallucination Benchmark](http://arxiv.org/abs/2504.17550v1)**
### **[DeepDistill: Enhancing LLM Reasoning Capabilities via Large-Scale Difficulty-Graded Data Training](http://arxiv.org/abs/2504.17565v1)**
### **[A Multi-Agent, Laxity-Based Aggregation Strategy for Cost-Effective Electric Vehicle Charging and Local Transformer Overload Prevention](http://arxiv.org/abs/2504.17575v1)**
### **[L3: DIMM-PIM Integrated Architecture and Coordination for Scalable Long-Context LLM Inference](http://arxiv.org/abs/2504.17584v1)**
### **[Beyond Labels: Zero-Shot Diabetic Foot Ulcer Wound Segmentation with Self-attention Diffusion Models and the Potential for Text-Guided Customization](http://arxiv.org/abs/2504.17628v1)**
### **[polyGen: A Learning Framework for Atomic-level Polymer Structure Generation](http://arxiv.org/abs/2504.17656v1)**
### **[Evaluating Grounded Reasoning by Code-Assisted Large Language Models for Mathematics](http://arxiv.org/abs/2504.17665v1)**
### **[Towards a HIPAA Compliant Agentic AI System in Healthcare](http://arxiv.org/abs/2504.17669v1)**
### **[Cross-region Model Training with Communication-Computation Overlapping and Delay Compensation](http://arxiv.org/abs/2504.17672v1)**
### **[Energy Considerations of Large Language Model Inference and Efficiency Optimizations](http://arxiv.org/abs/2504.17674v1)**
### **[INSIGHT: Bridging the Student-Teacher Gap in Times of Large Language Models](http://arxiv.org/abs/2504.17677v1)**
### **[Ensemble Bayesian Inference: Leveraging Small Language Models to Achieve LLM-level Accuracy in Profile Matching Tasks](http://arxiv.org/abs/2504.17685v1)**
### **[Generative Fields: Uncovering Hierarchical Feature Control for StyleGAN via Inverted Receptive Fields](http://arxiv.org/abs/2504.17712v1)**
### **[Multilingual Performance Biases of Large Language Models in Education](http://arxiv.org/abs/2504.17720v1)**
### **[Towards Robust LLMs: an Adversarial Robustness Measurement Framework](http://arxiv.org/abs/2504.17723v1)**
### **[Token-Shuffle: Towards High-Resolution Image Generation with Autoregressive Models](http://arxiv.org/abs/2504.17789v1)**
### **[LiDPM: Rethinking Point Diffusion for Lidar Scene Completion](http://arxiv.org/abs/2504.17791v1)**
