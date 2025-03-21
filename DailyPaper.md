# The Latest Daily Papers - Date: 2025-03-21
## Highlight Papers
### **[MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](http://arxiv.org/abs/2503.15451v1)**
- **Summary**: Okay, I will provide a summary and critical evaluation of the "MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space" paper.

**Summary**

The paper introduces MotionStreamer, a novel framework for generating human motion in a streaming fashion conditioned on text input. This means the model can generate motion in real-time, responding to text prompts as they arrive sequentially. The key innovations are: (1) using a diffusion-based autoregressive model to directly predict continuous motion latents, avoiding the need for discrete tokenization and the associated information loss; (2) a Causal Temporal AutoEncoder (Causal TAE) for compressing motion sequences into a causal latent space, enabling online decoding; and (3) two training strategies, Two-Forward training and Mixed training, to address error accumulation during long-term autoregressive generation.  The paper demonstrates state-of-the-art results on text-to-motion and long-term motion synthesis benchmarks and showcases applications like multi-round generation, long-term generation, and dynamic motion composition.

**Critical Evaluation**

*   **Novelty:**

    *   The combination of a diffusion-based model with an autoregressive architecture for *streaming* motion generation is fairly novel. While autoregressive methods exist, their integration with diffusion and the focus on a causal latent space specifically designed for streaming are differentiating factors.
    *   The Causal TAE is another novel component.  While TAEs exist for motion compression, the emphasis on causality and continuous latents (instead of discrete tokens used by VQ-VAE variants) is a significant contribution that directly addresses the challenges of streaming generation.
    *   The Two-Forward and Mixed training strategies appear to be effective techniques for mitigating error accumulation in the autoregressive setting. While similar strategies might exist in other domains, their adaptation and application to the motion generation task are novel and practically valuable.

*   **Significance:**

    *   The ability to generate motion in a streaming manner opens up new possibilities for interactive applications, such as virtual avatars, real-time animation, and robotics. This is a significant step beyond traditional offline motion generation methods.
    *   The paper's superior performance on benchmark datasets demonstrates the effectiveness of the proposed approach, suggesting that it could become a foundation for future research in this area.
    *   The MotionStreamer framework is relatively generic and could potentially be extended to other types of sequential data generation tasks beyond motion.

*   **Strengths:**

    *   **Clear Problem Definition:** The paper clearly articulates the challenges of streaming motion generation and how existing methods fall short.
    *   **Well-Designed Architecture:** The components of MotionStreamer (diffusion-based autoregressive model, Causal TAE, training strategies) are well-motivated and designed to address specific problems.
    *   **Strong Empirical Results:** The paper provides comprehensive experimental results on benchmark datasets and demonstrates the effectiveness of the proposed method.
    *   **Practical Applications:** The paper showcases the practical applications of MotionStreamer, further highlighting its significance.
    *   **Thorough Ablation Studies:** The ablation studies are insightful and demonstrate the importance of each component of the proposed framework.

*   **Weaknesses:**

    *   **Computational Complexity:** The paper does not explicitly discuss the computational complexity of MotionStreamer, which could be a concern given the use of diffusion models and autoregressive architectures. Real-time applications often have strict latency requirements, so a detailed analysis of the model's efficiency is crucial.  Specifically, I would like to see benchmarks and/or more detail on runtime on consumer-grade hardware.

    *   **Generalization to Unseen Text/Motion:** Although the paper reports good performance on benchmark datasets, it's difficult to assess the model's ability to generalize to unseen text descriptions or motion styles. Future work could explore techniques for improving the model's robustness and adaptability.
    *   **Qualitative Evaluation:** While the paper includes qualitative results, a more detailed analysis of the generated motions' realism and naturalness would be beneficial. User studies could be conducted to evaluate the subjective quality of the generated motions. The qualitative comparisons could be more convincing.

    *   **Lack of Code and Data:** Although a "Project Page" is referenced, it's important to note whether or not this includes code, data, or pre-trained models for reproducibility. At the time of writing, the "Project Page" only displays the manuscript.

**Justification for Score**

Considering the novelty of the approach, the significance of the problem, the strong experimental results, and the clearly articulated architecture, but also taking into account the identified weaknesses, I would assign this paper a score of **8**.

The paper makes a strong contribution to the field of motion generation by addressing the challenging problem of streaming generation. The Causal TAE and the Two-Forward training strategies are particularly novel and effective components. However, the computational complexity and generalization ability of the model need to be further investigated. Greater detail around the qualitative evaluation of the results is also important. While the paper is well-written and presents a compelling solution, addressing these shortcomings would further enhance its impact.
Score: 8

- **Score**: 8/10

### **[CHROME: Clothed Human Reconstruction with Occlusion-Resilience and Multiview-Consistency from a Single Image](http://arxiv.org/abs/2503.15671v1)**
- **Summary**: Here's a summary and critical evaluation of the CHROME paper:

**Summary:**

The paper "CHROME: Clothed Human Reconstruction with Occlusion-Resilience and Multiview-Consistency from a Single Image" addresses the challenging problem of reconstructing 3D clothed humans from single, occluded images.  It proposes a two-stage pipeline. First, a pose-controlled multiview diffusion model (FD) generates occlusion-free, cross-view consistent images from the input, conditioned on 3D pose estimates. Second, a 3D reconstruction model (FR) combines the occluded input and synthesized views to predict a cohesive 3D Gaussian representation.  The method avoids reliance on 3D supervision or SMPL priors, making it more generalizable. It demonstrates improved novel view synthesis and geometric reconstruction, especially under occlusion.

**Critical Evaluation:**

*   **Novelty:** The paper offers several novel contributions.
    *   Addressing occlusion in single-image clothed human reconstruction is a relevant and under-explored area.
    *   The two-stage approach combining a multiview diffusion model for hallucinating occlusion-free views with a 3D Gaussian splatting reconstruction model is innovative.  Using the diffusion model with off-the-shelf pose control is an excellent way to enforce cross-view consistency, addressing a key limitation of previous single-view methods.
    *   Avoiding the need for SMPL annotations and 3D supervision is a significant practical advantage, making the method more easily applied to real-world datasets.

*   **Significance:** The paper has the potential to make a significant impact on the field for several reasons:

    *   **Improved Occlusion Handling:** The substantial improvement in novel view synthesis (3dB PSNR improvement is notable) under occlusion is a key contribution that addresses a practical bottleneck in existing techniques.
    *   **Generalizability:**  The method demonstrates strong zero-shot performance on multiple datasets, suggesting good generalizability.  The design choice of removing reliance on 3D geometric priors facilitates this generalization.
    *   **Practicality:** The pipeline can be implemented using existing pose estimation methods.
    *   The use of 3D Gaussian Splatting is a suitable choice that strikes a good balance between image quality and reconstruction quality.

*   **Strengths:**

    *   The paper is well-written and clearly explains the proposed method and its advantages.
    *   The experimental results are convincing and show significant improvements over state-of-the-art methods.
    *   The ablation study demonstrates the effectiveness of different components of the proposed pipeline.
    *   The qualitative results visually highlight the superior performance of CHROME in handling occlusions and generating consistent novel views.
    *   The discussion on the inherent flexibility of CHROME when stereo images are available is interesting and shows great flexibility in the method.

*   **Weaknesses:**

    *   The paper doesn't delve extensively into the limitations.  For example, there is a lack of analysis in what cases can the pipeline fail.
    *   Some design choices might need further justification. What is the reason for the four camera angles used? Is it a reasonable number? Is there a justification given?
    *   The paper mentions that during end-to-end training, they enable 3D pose control derived from occluded images instead of ground truth poses, but doesn't go into further details of why this is better.

**Justification of Score:**

This paper represents a solid advance in a crucial area of 3D human reconstruction. The combination of carefully chosen components (diffusion model, 3D Gaussians, pose-conditioning) yields a system that is more robust to occlusion, more generalizable, and more practical than existing approaches. The experimental results are compelling, and the method addresses a significant gap in the field.
It's not a groundbreaking conceptual shift, but it's an excellent example of how to combine existing tools to tackle a real-world problem and achieve impressive results.
The significance of results shown alongside solid explanation is enough to warrant a good score in the range.

**Score: 8**

- **Score**: 8/10

### **[GASP: Unifying Geometric and Semantic Self-Supervised Pre-training for Autonomous Driving](http://arxiv.org/abs/2503.15672v1)**
- **Summary**: Here's a summary and critical evaluation of the provided research paper "GASP: Unifying Geometric and Semantic Self-Supervised Pre-training for Autonomous Driving":

**Summary:**

The paper introduces GASP (Geometric and Semantic Self-Supervised Pre-training), a new method for pre-training models for autonomous driving. GASP leverages readily available sensor data (LiDAR scans, camera images, and ego-poses) and performs self-supervised learning by predicting future occupancy, ego-path, and distilled high-level features from a vision foundation model within a continuous 4D (3D + time) representation.  The authors demonstrate that GASP learns a structured and generalizable representation of the environment and its evolution over time, leading to improved performance on downstream autonomous driving tasks such as semantic occupancy forecasting, online mapping, and ego trajectory prediction. The key idea is to model the environment as a continuous 4D occupancy field that incorporates both geometric and semantic information.  They also introduce practical improvements like harvesting negative information from missing LiDAR rays and using a rotation augmentation strategy to improve generalization.

**Critical Evaluation:**

*   **Novelty:** The paper presents a well-motivated integration of multiple self-supervised learning signals (geometric, semantic, and temporal). Combining occupancy prediction with ego-path prediction and distilling knowledge from a pre-trained vision foundation model in a continuous 4D representation is a novel approach. While individual components have been explored before, their unified combination is a significant contribution.  The paper also presents practical improvements like the rotation augmentation strategy and exploitation of missing lidar ray information. While these are not entirely novel concepts, their adaptation to the specific context of 4D occupancy prediction in autonomous driving is valuable.

*   **Significance:** The paper's significance lies in its ability to improve generalization performance across a range of autonomous driving tasks with relatively little labeled data. Pre-training is a well-established technique, and this paper demonstrates its efficacy in the AD domain, particularly when bridging the gap between geometry and semantics. The fact that GASP outperforms existing methods like UnO, especially on semantic tasks, highlights its potential. The practical improvements suggested also contribute to better training and thus are valuable in the field.

*   **Strengths:**
    *   **Well-defined problem and solution:** The paper clearly articulates the problem of learning representations for autonomous driving and proposes a specific and well-justified solution.
    *   **Unified approach:** GASP provides a unified framework for leveraging multiple readily available signals in AV systems, leading to a richer representation.
    *   **Strong empirical results:** The paper presents a thorough empirical evaluation on multiple benchmark datasets, demonstrating significant improvements over existing methods on tasks like semantic occupancy forecasting, online mapping, and ego-trajectory prediction.
    *   **Practical improvements:** The introduced methods, such as the rotation augmentation strategy and the use of missing lidar information, show practical benefits.
    *   **Open-source code:** The provision of open-source code helps reproduce and extend the research, enabling further progress in the field.
    *   **Detailed ablations:**  The thorough ablation studies help to understand the contribution of each component of GASP.
    *   **Scalability demonstration:** The authors demonstrate scalability to larger datasets like ZOD, showing consistent gains.

*   **Weaknesses:**
    *   **Reliance on DINOv2:** The reliance on a vision foundation model (DINOv2) might limit the generalizability or adaptability of the approach if DINOv2 has inherent biases or limitations. While justified by the empirical results, further analysis on the choice of VFM is necessary.
    *   **Limited exploration of other self-supervision signals:** The paper focuses on a specific set of self-supervision signals. Exploring other possibilities like flow consistency or predicting object dynamics could potentially further enhance the representation.
    *   **Lack of real-time performance analysis:** The paper lacks an explicit discussion or experimental evaluation of the real-time performance of GASP, which is crucial for autonomous driving applications. Although the authors note that the model uses a lightweight implicit decoder for occupancy query, the computational overhead of encoding and querying the 4D representation could be a limitation.

*   **Potential Influence:** The GASP framework has the potential to influence the field by providing a more effective and scalable pre-training strategy for autonomous driving models. Its ability to integrate multiple sensor modalities and leverage self-supervision can lead to more robust and generalizable representations, reducing the need for expensive labeled data.

**Score:** 8

**Justification:**

I am assigning a score of 8 because the paper demonstrates significant novelty in unifying multiple readily available self-supervised signals for a unified 4D representation for autonomous driving. Furthermore, the comprehensive experimental validation showcases its improved generalization and downstream task performance. The release of open-source code and a strong scalability demonstration are key factors in its potential to influence the field, while the discussed practical training improvements are valuable additions to the field. However, the reliance on DINOv2, a vision foundation model which has its own potential problems which are not discussed in the paper, the need of real-time performance analysis in an end-to-end evaluation, and limited exploration of alternative self-supervision signals hold back the score from reaching a higher level. The weaknesses are not severe enough to significantly detract from the paper's overall contribution, as a strong foundation is present that can be expanded on by future work.

- **Score**: 8/10

### **[Detecting LLM-Written Peer Reviews](http://arxiv.org/abs/2503.15772v1)**
- **Summary**: Here's a summary and critical evaluation of the paper on detecting LLM-written peer reviews:

**Summary:**

The paper tackles the growing concern of reviewers using LLMs to generate peer reviews, potentially compromising the integrity of the review process.  The authors propose a novel watermarking approach:  they indirectly inject prompts into manuscript PDFs that instruct LLMs to incorporate specific watermarks into their generated reviews.  These watermarks are statistically testable, allowing detection of LLM-generated reviews while controlling for false positives.  The paper explores various prompt injection techniques (font embedding, cryptic prompts) and reviewer defenses (paraphrasing).  Experiments on real-world peer review datasets and popular LLMs show the effectiveness of the approach, its resilience to defenses, and the validity of statistical error rate bounds. Crucially, they show their method's advantages over simple Bonferroni correction.

**Critical Evaluation:**

*   **Novelty:** The paper's novelty lies in its indirect prompt injection method specifically designed for peer review.  While watermarking and prompt injection are established concepts, their application to this context is innovative. The focus on statistical guarantees (FWER control *without assumptions on human reviews*) is a valuable contribution.

*   **Significance:** The paper addresses a significant problem facing academic publishing and conference organization:  the use of AI to generate reviews. If left unchecked, this practice could degrade the quality and fairness of the peer review process. Providing a method to detect such behavior is of practical importance. The potential of this work to preserve the integrity of scientific evaluation is high.

*   **Strengths:**

    *   **Well-defined Problem:** The paper clearly articulates the problem and its potential impact.
    *   **Sound Methodology:** The watermarking scheme is well-designed, considering detectability, robustness, and statistical validity.
    *   **Comprehensive Evaluation:**  The experiments are thorough, using multiple datasets, LLMs, and injection techniques. The consideration of reviewer defenses is a strength. The empirical validation of theoretical error bounds is crucial.
    *   **Practical Relevance:** The method offers a practical tool for editors and program chairs to identify potentially problematic reviews.
    *   **Careful Attention to Statistical Guarantees:** This is a significant strength, as statistical soundness is crucial for any detection mechanism used in a high-stakes environment.

*   **Weaknesses:**

    *   **Reliance on White-Box LLMs for Cryptic Prompt Optimization:** The cryptic prompt injection relies on white-box access to LLMs, which limits its applicability to black-box models like GPT-4.  The authors acknowledge this limitation.
    *   **Potential Evasion:**  While the paper addresses several reviewer defenses, more sophisticated adversarial techniques might be developed to circumvent the watermarking scheme. The authors also acknowledge the possibility that reviewers can also add noise or apply other transformations to the embedded watermark.
    *   **Font Embedding Practicality:** While promising, font embedding requires specialized tooling for PDF manipulation, increasing the barrier to adoption. It also appears that the technique used is more manual, which limits its use.

*   **Impact:**

    *   **Potential for Widespread Adoption:**  If the method proves to be robust and easy to implement, it could be adopted by journals and conferences to monitor the use of LLMs in peer review.
    *   **Catalyst for Further Research:** The paper could stimulate further research on detection methods, watermarking techniques, and defenses against LLM-assisted review manipulation.

**Justification for Score:**

The paper makes a valuable and timely contribution to a critical problem. It is both novel and significant. The weaknesses (reliance on white-box models for one technique, potential for evasion) are acknowledged by the authors and do not significantly detract from the overall value. The statistical rigor and comprehensive evaluation support the claims.

Score: 8

The work is excellent, well executed and relevant. However, because the cryptographic prompt injection (more complex) is only applicable to white-box models (and it is not clear how this could translate to black-box models) and because some of the prompt injection (font embedding) seems to be manually done, the top score is reduced to 8, with an emphasis on future work on improving practicality and robustness.

- **Score**: 8/10

### **[Fùxì: A Benchmark for Evaluating Language Models on Ancient Chinese Text Understanding and Generation](http://arxiv.org/abs/2503.15837v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Fùxì: A Benchmark for Evaluating Language Models on Ancient Chinese Text Understanding and Generation":

**Summary:**

The paper introduces Fùxì, a novel benchmark designed to evaluate the capabilities of Large Language Models (LLMs) in understanding and generating classical Chinese text. This is an important area because current LLMs, while performing well in modern Chinese, struggle with the unique linguistic features, structural constraints, and cultural context of ancient Chinese.  Fùxì comprises 21 diverse tasks covering both comprehension and generation, and includes innovative tasks like poetry composition and couplet completion. The benchmark is also distinguished by its specialized evaluation metrics tailored for classical Chinese text generation, combining rule-based verification and fine-tuned LLM evaluators. The authors conduct extensive experiments with state-of-the-art LLMs, highlighting the performance gaps between understanding and generation tasks, and providing insights for future research.  The benchmark, evaluation toolkit, and baseline results are made publicly available.

**Rigorous and Critical Evaluation:**

**Novelty:** The paper offers a significant and needed contribution to the field of NLP. Existing benchmarks primarily focus on modern Chinese and understanding-based tasks (mainly multiple-choice). Fùxì's novelty lies in:

*   **Focus on Ancient Chinese:** Addressing a specific and challenging subfield of NLP.
*   **Balanced Comprehension and Generation:**  Prioritizing generation tasks which are currently under-evaluated in the context of Ancient Chinese.
*   **Innovative Task Design:** Introducing tasks explicitly tailored to classical Chinese, like poetry generation and couplet completion, which assess nuanced understanding of literary conventions.
*   **Specialized Evaluation Metrics:** Tackling the difficult problem of evaluating ancient Chinese text generation by a hybrid method of rule-based systems and fine-tuned LLM evaluators.
*   **Publicly Available Resource:** A vital step in fostering research and further development in this domain.

**Significance:** The significance of this work stems from:

*   **Cultural Heritage Preservation:** Ancient Chinese texts are a vital part of cultural heritage, and better NLP tools can make them more accessible and understood.
*   **Pushing the Boundaries of LLMs:** Evaluating LLMs on a complex and linguistically distinct language like ancient Chinese exposes their limitations and encourages improvements in areas like cultural knowledge integration and format adherence.
*   **Setting a Standard for Evaluation:**  Fùxì provides a structured framework and evaluation protocols that can be adopted and extended by other researchers in the field.
*   **Insights into Model Strengths and Weaknesses:** The experiments provide valuable insights into the capabilities and shortcomings of various LLM architectures in handling this specific linguistic domain. The revealed gap between comprehension and generation performance is particularly noteworthy.

**Strengths:**

*   **Comprehensive Task Coverage:** The inclusion of 21 diverse tasks makes Fùxì a well-rounded benchmark.
*   **Careful Dataset Curation:**  The authors mention manual curation and automated processes for data construction, ensuring data quality and reliability.
*   **Thorough Experiments:** The experiments include a variety of LLMs, including open-source and closed-source models, and both zero-shot and few-shot settings.
*   **Robust Evaluation:** The development and validation of an LLM-based evaluator for open-ended generation tasks is a key strength. The correlation with human judgments adds credibility to the automatic evaluation.
*   **Clear Articulation of Limitations:**  The authors acknowledge the limitations of their evaluation metrics and task coverage, which encourages further research in these areas.

**Weaknesses:**

*   **Evaluation Metric Bias:** While the LLM-based evaluator is validated, there remains the potential for subtle biases, particularly in subjective tasks where cultural authenticity is crucial. Further refinement and diversification of the training data for the evaluator could improve this.
*   **Limited Aesthetical Evaluation:**  The focus on format correctness in poetry and couplet generation, while a reasonable starting point, neglects the aesthetic qualities of these literary forms. Future work could explore ways to integrate aesthetic evaluation.
*   **Potential Data Biases:**  While data curation is mentioned, it's important to consider the potential for biases present in the original data sources used to create the benchmark, which could influence model performance.

**Potential Influence:**

Fùxì has the potential to become a standard benchmark for evaluating LLMs on ancient Chinese text processing. It can drive research in areas like:

*   Developing more culturally aware LLMs.
*   Improving the generation capabilities of LLMs, particularly for creative tasks.
*   Creating more effective evaluation metrics for nuanced linguistic tasks.
*   Exploring cross-lingual transfer learning techniques for ancient languages.

**Justification of Score:**

I assign a score of **8.5**.  The paper makes a significant contribution by providing a novel and comprehensive benchmark for a challenging and important subfield of NLP.  The tasks are thoughtfully designed, and the evaluation methodology is robust.  The public availability of the benchmark is a major strength that will foster further research. The weaknesses related to evaluation metric biases and limitations in aesthetic evaluation are acknowledged and provide clear directions for future work. While not a perfect benchmark, Fùxì fills a critical gap and will undoubtedly have a positive impact on the field.

**Score: 8.5**

- **Score**: 8/10

### **[Automatic Generation of Safety-compliant Linear Temporal Logic via Large Language Model: A Self-supervised Framework](http://arxiv.org/abs/2503.15840v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces AutoSafeLTL, a self-supervised framework that leverages Large Language Models (LLMs) to automatically generate safety-compliant Linear Temporal Logic (LTL) specifications from natural language descriptions of desired tasks in cyber-physical systems (CPS). The key innovation is the integration of a Language Inclusion check with an automated counterexample-guided feedback and modification mechanism. This pipeline verifies the safety compliance of generated LTL formulas against predefined safety restrictions (Base Rules) while maintaining logical consistency and semantic accuracy. The framework incorporates two additional Agent LLMs to improve understanding and correction capabilities. Experimental results demonstrate the effectiveness of AutoSafeLTL in guaranteeing safety compliance, achieving a 0% violation rate against imposed safety constraints.

**Critical Evaluation:**

**Novelty:**  The paper addresses a crucial but often overlooked aspect in automatic LTL generation: ensuring *safety compliance* with predefined system constraints. While existing works focus on semantic consistency between natural language and LTL, this paper explicitly targets the conflict between generated LTL and safety rules. This is a significant advancement because non-compliant LTL can lead to dangerous behavior in safety-critical systems. The integration of formal verification techniques (Language Inclusion Check) with LLMs in a self-supervised loop is also a novel approach.

**Significance:** The practical implications of this work are substantial. Automating the generation of safety-compliant LTL specifications can significantly reduce the effort and expertise required for formally verifying and synthesizing controllers for CPS. This can lead to more reliable and safer autonomous systems, industrial automation, and medical devices.  The framework's ability to provide *formal guarantees* of safety compliance, a feature absent in many existing LLM-based approaches, is particularly valuable.

**Strengths:**

*   **Clearly defined problem:** The paper identifies a gap in existing LTL generation techniques.
*   **Novel approach:** The combination of LLMs with formal verification in a self-supervised loop is innovative.
*   **Practical relevance:** The work has direct applications in safety-critical domains.
*   **Comprehensive evaluation:** The experimental results demonstrate the effectiveness of AutoSafeLTL, particularly the 0% violation rate.
*   **Ablation study:** The ablation experiments convincingly demonstrate the importance of each component of the proposed framework, particularly the Agent LLMs.

**Weaknesses:**

*   **Limited Data and Scenario:** The experiments primarily focus on a traffic scenario, which is not a limitation in itself, but this could give rise to a limited scope in evaluating real-world implementations.  The experiments could benefit from a broader range of CPS applications to showcase the framework's generalizability.
*   **Reliance on existing tools:** The framework relies on external tools like Spot and RABIT. While these tools are well-established, the stability and performance of AutoSafeLTL are contingent on these tools. The paper acknowledges this limitation.
*   **Computational overhead:** Iterative modification loop might be computationally expensive for certain CPS systems depending on complexity and scale.
*   **Over reliance on LLMs**: While the inclusion of Agent LLMs improve the performance, there is still a question on how LLMs can be made to be more robust in interpreting formal aspects of LTL specifications.

**Impact:**

The paper has the potential to significantly impact the field of formal methods for CPS. By providing a way to automatically generate safety-compliant specifications, it can lower the barrier to entry for formal verification and synthesis. This can lead to wider adoption of these techniques in industry, ultimately resulting in safer and more reliable systems. Future works may build on this framework by incorporating more sophisticated verification techniques or by extending it to handle more complex CPS scenarios. The dataset that is being created might also have an impact in furthering research.

**Justification for Score:**

The paper presents a novel and well-executed approach to a critical problem in CPS. The integration of LLMs with formal verification techniques is a significant contribution. While the experiments are somewhat limited, the results are compelling, and the potential impact of the work is high. The paper demonstrates a clear understanding of both LLMs and formal verification.

Score: 8

- **Score**: 8/10

### **[TruthLens: Explainable DeepFake Detection for Face Manipulated and Fully Synthetic Data](http://arxiv.org/abs/2503.15867v1)**
- **Summary**: Here's a summary and critical evaluation of the provided paper:

**Summary:**

The paper introduces TruthLens, a novel framework for explainable DeepFake detection, addressing both face-manipulated and fully AI-generated content.  TruthLens goes beyond binary classification (real/fake) by providing detailed textual reasoning for its predictions, answering nuanced queries about specific facial features. The architecture combines the global contextual understanding of multimodal large language models (MLLMs) like PaliGemma2 with the localized feature extraction capabilities of vision-only models like DINOv2. This hybrid design is fine-tuned to improve detection accuracy and provide interpretable explanations, demonstrated through experiments on diverse datasets, showing improvements over state-of-the-art methods in both detection accuracy and explainability.

**Critical Evaluation:**

*   **Novelty:** The paper's novelty lies in its unified approach to DeepFake detection, handling both traditional face manipulations and fully synthetic images *with explanations*. While existing methods often specialize in one category or lack interpretability, TruthLens combines an MLLM and a vision-only model to achieve both accuracy and detailed reasoning. The mixture of feature (MoF) strategy that interleaved and concatenated tokens from both the language and vision models is novel.

*   **Significance:** DeepFake detection is a crucial and growing field. The ability to not only detect but *explain* the detection is highly valuable, especially in scenarios requiring human understanding and trust in AI systems. The approach of fine-tuning a general MLLM with visual features and local textures opens avenues for detecting subtle and unseen manipulation techniques in the wild.

*   **Strengths:**

    *   **Unified Approach:** Addresses both face-manipulated and fully synthetic content, a gap in existing literature.
    *   **Explainability:** Provides textual explanations for its decisions, enhancing transparency and trust.
    *   **Hybrid Architecture:** Cleverly combines the strengths of MLLMs and vision-only models. The MoF approach has improved accuracy.
    *   **Strong Experimental Results:** Demonstrates superior performance compared to SOTA methods on diverse datasets.
    *   **Detailed Ablation Studies:** provides insights into the impact of model components (e.g., feature mixing strategies, adapter training).

*   **Weaknesses:**

    *   **Dependence on Foundation Models:** The performance heavily relies on the capabilities of the underlying PaliGemma2 and DINOv2 models. While this approach enables leveraging existing knowledge, it also inherits any biases or limitations of these models.
    *   **Complexity:** The architecture is complex, involving pretraining and fine-tuning stages. More lightweight solutions may be preferred in resource-constrained environments.
    *   **Evaluation Metric Rigor:** The use of Gemini-1.0 as an LLM-as-a-judge introduces potential biases depending on the prompts used. While it's a valid approach, ensuring the metric's impartiality is crucial.

*   **Potential Impact:**

    *   **Advancing DeepFake Detection:** The proposed framework has the potential to significantly improve the accuracy and reliability of DeepFake detection systems.
    *   **Promoting Trustworthy AI:** By providing explanations for its decisions, TruthLens can help build trust in AI systems and facilitate informed decision-making.
    *   **Inspiring Further Research:**  The work highlights the benefits of combining MLLMs and vision-only models, which can inspire further research in this area.
    *   **Addressing a Critical Societal Problem:**  Addresses the growing threat of malicious and misleading content.

**Justification for Score:**

Considering the novelty of its approach, its performance in handling both manipulated and synthetic content, its textual explanations, and strong experimental results, the paper makes a substantial contribution to the field. However, the reliance on powerful foundation models, the metric, and potential complexity temper the assessment slightly.  It's a solid advancement but leaves room for further optimization, simplification, and evaluation metrics.

**Score: 8**

- **Score**: 8/10

### **[Parameters vs. Context: Fine-Grained Control of Knowledge Reliance in Language Models](http://arxiv.org/abs/2503.15888v1)**
- **Summary**: The paper introduces CK-PLUG, a plug-and-play method for controlling the knowledge reliance of Large Language Models (LLMs) in Retrieval-Augmented Generation (RAG) systems. It addresses the challenge of conflicts between parametric knowledge and retrieved context, where LLMs struggle to prioritize one over the other. CK-PLUG employs a novel knowledge consistency metric, Confidence Gain (CG), to detect conflicts by measuring entropy shifts in token probability distributions after context insertion. It then enables fine-grained control over knowledge preference by adjusting the probability distribution of tokens with negative confidence gain through a single tuning parameter. Experiments demonstrate CK-PLUG's ability to regulate knowledge reliance in counterfactual RAG scenarios, maintain generation fluency and knowledge accuracy, and achieve consistent performance improvements across various general RAG tasks.

**Critical Evaluation:**

**Strengths:**

*   **Addresses a Significant Problem:** The paper tackles a critical issue in RAG systems: resolving conflicts between parametric knowledge and retrieved context. This is a well-recognized problem that hinders the reliability and trustworthiness of RAG-generated outputs.
*   **Novel Approach:** The introduction of Confidence Gain as a metric for detecting knowledge conflicts is a novel idea and appears to be effective in identifying potentially problematic tokens.
*   **Plug-and-Play Design:** The plug-and-play nature of CK-PLUG is a significant advantage, as it allows for easy integration with existing LLMs without requiring retraining or architectural modifications.
*   **Fine-Grained Control:** The ability to control knowledge preference at the token level offers a level of granularity that is lacking in many existing approaches.
*   **Comprehensive Evaluation:** The paper provides a comprehensive evaluation of CK-PLUG across various datasets, LLMs, and RAG scenarios, demonstrating its effectiveness in controlling knowledge reliance and improving generation quality.
*   **Adaptive Adjustment:** The inclusion of an adaptive method for tuning the parameter (alpha) based on model confidence further strengthens the practical applicability.
*   **Code Availability:** The availability of the code promotes reproducibility and facilitates further research in this area.
*  **Case Study and Ablation Study:** Provides insights that validates each aspect of the CK-PLUG.

**Weaknesses:**

*   **Reliance on Entropy:** The reliance on entropy as a measure of uncertainty may be sensitive to the specific architecture and training of the LLM. Entropy may not always accurately reflect the model's confidence in its predictions. Additional metrics could've further solidify their conclusions.
*   **Limited Scope of Evaluation:** While the evaluation is comprehensive, it primarily focuses on question-answering tasks. It would be beneficial to evaluate CK-PLUG on a broader range of RAG applications, such as text summarization and dialogue generation.
*   **Parameter Sensitivity:** Although there's an adaptive method, the tuning parameter 'a' might be difficult to set appropriately for all applications without careful experimentation. Further investigation on the parameterization process would be beneficial to the community.
*   **Lack of Detailed Analysis of the Limitations:** While the paper identifies QWEN as an exception, the limitations are not discussed in detail.

**Novelty and Significance:**

The paper makes a significant contribution to the field of RAG by providing a novel and practical method for controlling knowledge reliance in LLMs. The introduction of Confidence Gain and the plug-and-play design of CK-PLUG offer a valuable approach for resolving conflicts between parametric knowledge and retrieved context. The comprehensive evaluation demonstrates the effectiveness of the proposed method and its potential for improving the reliability and trustworthiness of RAG-generated outputs. However, improvements with limitations with the tuning parameters should be carefully investigated and discussed in detail.

**Potential Influence:**

CK-PLUG has the potential to influence the development of more reliable and trustworthy RAG systems. Its plug-and-play design and fine-grained control over knowledge preference make it a valuable tool for researchers and practitioners working with LLMs. The paper's findings may also inspire further research on methods for detecting and resolving knowledge conflicts in RAG systems.

**Score: 8**

**Rationale:**

CK-PLUG addresses a significant and well-recognized problem in RAG systems with a novel and practical solution. The use of Confidence Gain and the plug-and-play design are key strengths. However, the potential sensitivity of the entropy-based metric and the limited scope of evaluation prevent it from achieving a higher score. It's also very important to note that although they offer an adaptive method for tuning, it might still be difficult to implement the parameterization in other use-cases. Overall, it is a significant contribution but could benefit from more robustness and clarity concerning real-world deployment of the parameter tuning and a more detailed analysis of the limitations of certain models.

- **Score**: 8/10

### **[Jasmine: Harnessing Diffusion Prior for Self-supervised Depth Estimation](http://arxiv.org/abs/2503.15905v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper "Jasmine: Harnessing Diffusion Prior for Self-Supervised Depth Estimation" introduces a novel self-supervised monocular depth estimation (SSMDE) framework based on Stable Diffusion (SD). Addressing the challenge of adapting SD models to dense prediction without high-precision supervision, Jasmine employs a "hybrid image reconstruction" (HIR) task, reconstructing real and synthetic images to preserve SD's priors while tolerating color variations. A "Scale-Shift GRU" (SSG) module is introduced to address the misalignment between SD's fixed output range and the requirements of self-supervised scale-invariant depth estimation.  Experiments demonstrate state-of-the-art performance on KITTI and superior zero-shot generalization.

**Critical Evaluation:**

**Novelty:**  The paper's primary novelty lies in successfully integrating Stable Diffusion into a *self-supervised* monocular depth estimation framework.  Previous SD-based depth estimation methods were *supervised*, relying on high-quality depth labels to fine-tune the diffusion model.  The HIR task is a clever solution to the core problem of preventing the self-supervised reprojection loss from corrupting SD's pre-trained priors.  The SSG module is also a novel contribution, addressing a specific challenge in adapting diffusion models to scale-invariant depth prediction.

**Significance:**  The paper makes a significant contribution by democratizing access to the benefits of diffusion models for depth estimation. By removing the need for supervised depth labels, the approach unlocks the potential to train depth estimation models on the vast quantities of unlabeled video data. This can lead to more robust and generalizable depth estimation models, particularly for scenarios where labeled data is scarce or unavailable.  The reported results on KITTI, and especially the zero-shot generalization performance, demonstrate the potential impact of this approach.  The claim of detail preservation, while qualitatively supported by some figures, could benefit from more rigorous quantitative evaluation, perhaps through metrics specifically designed to assess image sharpness.

**Strengths:**

*   **Addresses a key limitation:** Overcomes the reliance on supervised data in SD-based depth estimation.
*   **Novel technical solutions:**  HIR task and SSG module are creative and well-motivated.
*   **Strong experimental results:** Achieves state-of-the-art performance on KITTI and demonstrates excellent zero-shot generalization.
*   **Clear writing and presentation:** The paper is generally well-written and easy to follow, although the method section does contain jargon.

**Weaknesses:**

*   **Dependency on SD's existing priors:** While the method reduces reliance on explicit labels, it strongly depends on the quality of priors learned by Stable Diffusion.  This might limit its applicability to domains significantly different from SD's training data.
*   **Qualitative dependence:** the paper relies on images. However, the results show minor but insignificant differences between images, and the images have low resolutions, which make detail analysis difficult.
*   **Limited evaluation of detail preservation:** More quantitative analysis or specialized metrics could strengthen the claim of improved detail sharpness.
*   **Complexity:** The overall framework is relatively complex, involving multiple modules and training stages.

**Potential Influence:**

The paper has the potential to significantly influence the field of depth estimation. By providing a practical way to leverage the power of diffusion models without the burden of supervised data, it opens the door to new research directions in self-supervised learning and domain adaptation. Its approach might inspire other researchers to find creative ways to adapt powerful generative models for downstream tasks.

**Rigorous Rationale for the Score:**

While the integration of Stable Diffusion into a self-supervised framework is a significant step forward, the paper depends on SD's inherent assumptions. The method is technically well-executed, and results clearly indicate high performance gains. Considering all above aspects, the score reflects the impact, innovation and reliability of the presented solution.

Score: 8

- **Score**: 8/10

### **[SpiLiFormer: Enhancing Spiking Transformers with Lateral Inhibition](http://arxiv.org/abs/2503.15986v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "SpiLiFormer: Enhancing Spiking Transformers with Lateral Inhibition":

**Summary:**

The paper introduces SpiLiFormer, a novel spiking transformer architecture that incorporates lateral inhibition to mitigate the attention distraction issue prevalent in existing spiking neural networks (SNNs).  The core idea is to mimic the brain's lateral inhibition mechanism, enhancing attention towards relevant features while suppressing irrelevant ones.  SpiLiFormer utilizes two new attention paradigms: Feedforward-pathway Lateral Differential Inhibition (FF-LiDiff) and Feedback-pathway Lateral Differential Inhibition (FB-LiDiff). Experimental results across various datasets (CIFAR-10, CIFAR-100, CIFAR10-DVS, N-Caltech101, and ImageNet-1K) demonstrate state-of-the-art performance and improved robustness compared to existing SNN models. Notably, it achieves better performance on ImageNet-1K with fewer parameters and time steps than existing SOTA spiking transformers.

**Critical Evaluation:**

*   **Novelty:** The primary novelty lies in the incorporation of lateral inhibition, a biologically inspired mechanism, into a spiking transformer architecture.  While lateral inhibition has been used in SNNs before, its application within a transformer context, with the specific FF-LiDiff and FB-LiDiff designs, represents a distinct contribution.  The separate processing pathways for Q, K, and V, along with the feedback mechanism, further enhances the novel design.

*   **Significance:** Addressing the attention distraction issue in spiking transformers is a significant step towards improving the performance and efficiency of these networks. SNNs hold promise for low-power applications, and overcoming performance limitations is crucial for their broader adoption. The performance gains across multiple datasets, particularly ImageNet-1K, are strong evidence of the effectiveness of the proposed approach. The fact that SpiLiFormer achieves SOTA results with fewer parameters and time steps than existing methods highlights its potential for improved energy efficiency. The added robustness against adversarial attacks is another positive aspect.

*   **Strengths:**
    *   **Strong Empirical Results:** The paper provides compelling experimental results across a diverse set of datasets, demonstrating consistent performance improvements over existing SNN models.  The comparison against existing SOTA models is clearly presented.
    *   **Clear Motivation:** The attention distraction issue is clearly articulated and well-motivated with visual examples.  The biological inspiration behind the design is also well-explained.
    *   **Well-Designed Architecture:** The FF-LiDiff and FB-LiDiff modules are thoughtfully designed and justified, with clear descriptions of their functionality.
    *   **Robustness Analysis:** The inclusion of adversarial testing and attention map visualizations adds further credibility to the claims of improved robustness and attention allocation.

*   **Weaknesses:**
    *   **Complexity:**  While the lateral inhibition concept is elegantly incorporated, the SpiLiFormer architecture does add some complexity to the model. A deeper analysis into the computational overhead introduced by the proposed attention mechanism would be beneficial.
    *  **Limited ablation:** In the ablation study, only the absence of either FF-LiDiff or FB-LiDiff attention are investigated. It would be interesting to analyze the contribution of each element within those attention blocks.
    *   **Energy Efficiency Analysis:** While the paper mentions potential energy efficiency improvements, a more detailed energy consumption analysis would strengthen the claims, especially since this is a key motivation for using SNNs.

*   **Potential Influence:**  SpiLiFormer has the potential to influence the design of future spiking transformer architectures, encouraging the incorporation of biologically inspired mechanisms for improved attention and efficiency.  It could also stimulate further research into lateral inhibition as a technique for enhancing robustness and generalization in SNNs.

*   **Score Justification:**  The paper demonstrates a clear improvement in performance for spiking transformers by addressing a significant issue with a novel and well-motivated approach. The gains are substantial, especially considering the reduction in parameter count and time steps. The architecture is well-designed, and the experimental results are convincing. The lack of a full energy efficiency analysis and some limited aspects in ablation studies hold it back from a higher score.

Score: 8

- **Score**: 8/10

### **[Animating the Uncaptured: Humanoid Mesh Animation with Video Diffusion Models](http://arxiv.org/abs/2503.15996v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Animating the Uncaptured: Humanoid Mesh Animation with Video Diffusion Models":

**Summary:**

The paper proposes a novel approach for animating static 3D humanoid meshes using the motion priors learned by video diffusion models.  Given a static 3D mesh and a text prompt describing the desired motion, the method generates a video of the mesh performing the motion using a text-to-video diffusion model. The motion from the video is then transferred back to the 3D mesh by tracking SMPL parameters within the generated video, allowing for realistic animation of the original 3D mesh. The pipeline involves SMPL registration, vertex reparameterization, and motion optimization using sparse and dense features extracted from the generated video. The paper demonstrates the efficacy of this approach through qualitative and quantitative evaluations, comparing it to existing methods for motion generation and video tracking.

**Critical Evaluation:**

* **Novelty:** The core idea of leveraging video diffusion models to drive 3D mesh animation is novel. While the concept of extracting motion from video is not entirely new, the authors present a specific pipeline that leverages the power of recent video diffusion models in a clever way. In particular, the way SMPL is used as a deformation proxy and for tracking is a good choice.
* **Significance:** The paper has the potential to significantly impact the field of 3D animation. Traditional animation pipelines require substantial manual effort. By automating the motion generation process through learned motion priors, this approach promises to be more accessible and cost-effective. The ability to generate diverse and realistic animations from text prompts opens up new possibilities for character animation in various applications such as video games, movies, and virtual reality.  The use of publicly available tools, and SMPL as the deformation target helps generalize the animation to different meshes.
* **Strengths:**
    * **Exploitation of strong priors:** Effectively utilizes the strong motion priors learned by video diffusion models, leading to more realistic and diverse animations.
    * **Robust Tracking:** The combination of sparse (landmarks) and dense (DINOv2 features) cues makes the video tracking more robust, particularly in the presence of synthetic video artifacts.
    * **Clear Pipeline:** The paper clearly outlines the various components of the pipeline, making it easier to understand and potentially replicate.
    * **Evaluation:** Offers a comprehensive evaluation including quantitative comparison to other methods (WHAM, SMPLIFY), ablation studies, and a perceptual study to evaluate human preference.
* **Weaknesses:**
    * **Reliance on SMPL:** While using SMPL helps with tracking, it also introduces a bottleneck.  The quality of the motion transfer is dependent on the initial SMPL registration and how well the mesh fits within the SMPL space.  It's unclear how the approach would perform with highly non-humanoid shapes.
    * **Dependence on Diffusion Models:** The results are inherently limited by the capabilities and biases of the underlying video diffusion model. Morphing artifacts and other inconsistencies in the generated videos could negatively impact the quality of the animations.  The authors acknowledge this limitation but it's an important consideration.  It would be useful to include discussions related to the limitations of the base VDM model.
    * **Limited Control:** While text prompts offer a way to control the animation, the level of control is limited compared to traditional animation tools. Precise control over specific movements or interactions is not possible using the text prompt alone.
    * **Ethical Concerns:** The authors have noted ethical concerns for generating realistic human actions.
* **Potential Influence:** The paper's ideas could inspire further research into:
    * **More sophisticated motion transfer techniques:** Exploring more advanced techniques for transferring motion from video to 3D meshes, potentially moving beyond SMPL as a deformation proxy.
    * **Improved control mechanisms:** Developing more intuitive and fine-grained control mechanisms for guiding the motion generation process.
    * **Application to other domains:** Extending the approach to other domains, such as animating non-humanoid characters or objects.
    * **Incorporating physics constraints:** Adding physics constraints to the motion optimization to ensure more realistic and plausible movements.

**Justification for Score:**

While the paper has some limitations, the novelty and potential impact of leveraging video diffusion models for 3D mesh animation are significant. The strengths outweigh the weaknesses, particularly with future advancements in video diffusion modeling. The presented pipeline is clear, well-evaluated, and provides a valuable contribution to the field. Taking all of this into account, I feel this deserves a high score.

Score: 8

- **Score**: 8/10

### **[The Lighthouse of Language: Enhancing LLM Agents via Critique-Guided Improvement](http://arxiv.org/abs/2503.16024v1)**
- **Summary**: Here's a summary and critical evaluation of the provided paper:

**Summary:**

The paper introduces Critique-Guided Improvement (CGI), a two-player framework designed to enhance the performance of LLM-based agents in interactive environments. CGI involves an actor model that explores an environment and a critic model that generates natural language feedback. The critic is trained to provide fine-grained assessments and actionable revisions of the actor's actions.  The actor is iteratively fine-tuned using this critique to improve its reasoning and ability to utilize external feedback. Experiments across three interactive environments (WebShop, ScienceWorld, and TextCraft) demonstrate that CGI significantly outperforms existing baselines. A key finding is that even a relatively small critic model can surpass GPT-4 in feedback quality. The iterative action refinement process further boosts performance, achieving state-of-the-art results.

**Critical Evaluation:**

*   **Novelty:** The paper's novelty lies in its specific architecture and training methodology for leveraging natural language feedback in LLM agents. The two-player approach, separating action generation and critique, is a valuable design choice. The iterative refinement process, which allows the agent to progressively learn from feedback and improve its reasoning, is a crucial aspect. While other works explore feedback in LLM agents, CGI provides a specific and effective implementation demonstrating its value through strong experimental results.

*   **Significance:** The paper's significance comes from demonstrating a practical and effective way to incorporate natural language feedback into LLM agents. The fact that a smaller, specialized critic model can outperform a large, general-purpose model like GPT-4 for feedback generation has implications for resource-efficient agent design. The substantial performance improvements achieved by CGI across diverse environments suggest its general applicability. The findings also shed light on the challenges associated with numerical feedback and self-critique methods, further motivating the proposed framework.

*   **Strengths:**

    *   **Strong Empirical Results:** The paper provides compelling evidence of CGI's effectiveness across three diverse interactive environments, showing substantial improvements over existing baselines and even outperforming GPT-4 as a critic.
    *   **Well-Defined Framework:** The CGI framework is clearly presented with a detailed description of each stage (Critique Generation and Action Refinement) and its components.
    *   **Insightful Analysis:** The paper provides valuable insights into the benefits of natural language feedback, the limitations of numerical feedback and self-critique methods, and the importance of action refinement for effective critique utilization.
    *   **Resource Efficiency:** The results showing that a small critic model can surpass GPT-4 in feedback quality suggest that specialized critics can be a more resource-efficient alternative to relying on large, general-purpose models.

*   **Weaknesses:**

    *   **Dependency on Expert Data:**  The reliance on expert annotations (via GPT-4 or similar) to train the critic model is a potential limitation. Gathering this data can be expensive and may introduce biases. The paper could benefit from discussing methods to reduce the reliance on this "expert" data, perhaps through self-play or other methods.
    *   **Limited Ablation Studies:** While the ablation study provides valuable insights, it could be expanded to explore the impact of different critic model sizes, alternative training strategies for the critic, or different feedback structures.
    *   **Generalizability to other LLMs:** It is possible that the advantages demonstrated are only applicable to the LLama family.
    *   **Limited Qualitative Analysis:** The qualitative analysis is relatively brief. A deeper dive into specific examples of how the critic's feedback helps the actor overcome challenges would strengthen the paper.

**Overall Assessment:**

The paper presents a novel and effective framework for enhancing LLM agents using critique-guided improvement. The well-defined architecture, compelling empirical results, and insightful analysis make this a significant contribution to the field. The potential limitations regarding dependence on expert data could be areas for future research. Despite the limitations, the paper's strengths and contributions warrant a positive score.

**Score: 8**

- **Score**: 8/10

### **[Improving Discriminator Guidance in Diffusion Models](http://arxiv.org/abs/2503.16117v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper investigates the use of Discriminator Guidance (DG) for refining pre-trained diffusion models. It argues that the standard implementation of DG, which relies on training a discriminator with a cross-entropy loss, can lead to a refined distribution that is *worse* than the original pre-trained distribution, particularly when the discriminator overfits. The authors theoretically show that minimizing cross-entropy does not guarantee minimization of the KL divergence between the model and target distributions, and overfitting intensifies this issue. To address this, they propose a new training objective for the discriminator based on minimizing a Mean Squared Error (MSE) loss related to the log-likelihood ratio gradient. They demonstrate empirically that their proposed method consistently improves sample quality compared to the conventional cross-entropy based DG across several image datasets.

**Critical Evaluation:**

**Novelty:** The paper's primary novelty lies in its theoretical analysis of the limitations of cross-entropy loss in discriminator guidance and the proposal of an alternative MSE-based loss function.  The insight that minimizing cross-entropy for the discriminator *doesn't* necessarily lead to a better refined distribution, and, in fact, can degrade it, is a significant contribution. Demonstrating how discriminator overfitting exacerbates this problem and then proposing a theoretically grounded solution enhances the originality. While leveraging discriminators in generative models is not new, the detailed analysis of the training objective and the specific focus on *gradient* accuracy of the discriminator represents a valuable advancement.

**Significance:** The paper's findings have direct implications for the practical application of discriminator guidance in diffusion models.  By demonstrating the potential pitfalls of the conventional approach and offering a more robust alternative, the authors contribute to more reliable and effective use of DG. The empirical results on standard image generation datasets provides convincing evidence that the proposed approach offers improvements in practice.

**Strengths:**

*   **Strong Theoretical Foundation:** The paper provides a clear and rigorous theoretical analysis, including theorems and proofs, supporting its claims about the limitations of cross-entropy loss.
*   **Well-Motivated Solution:** The proposed MSE-based loss function is directly derived from the theoretical analysis, making it a logically sound and well-motivated solution to the identified problems.
*   **Empirical Validation:** The empirical results on multiple datasets demonstrate the practical effectiveness of the proposed method, providing strong evidence for its superiority over the conventional approach.
*   **Clear Presentation:** The paper is generally well-written and organized, making it accessible to researchers in the field.
*   **Practical Impact:** The paper offers a practical improvement to an existing technique, making it immediately useful for researchers and practitioners working with diffusion models.

**Weaknesses:**

*   **Computational Cost:** The proposed MSE-based loss function appears to be more computationally expensive than the cross-entropy loss, as it requires computing gradients of both the discriminator and the score function.  While the paper acknowledges this increase in cost, a more detailed analysis of the computational overhead and potential optimizations could be beneficial.
*   **Sensitivity to Hyperparameters:** The method introduces a new hyperparameter (gamma) that controls the balance between the MSE loss and the cross-entropy loss. The paper shows some experimental variations based on gamma, further investigations such as robustness and sensitivity analysis would provide a more complete understanding of how to best leverage the suggested improvements.
*   **Limited Scope:** The paper focuses primarily on image generation.  While the findings are likely to generalize to other domains, empirical validation on other types of data (e.g., audio, text) would strengthen the claims about the broad applicability of the proposed method.

**Justification of Score:**

The paper makes a substantial contribution to the field of diffusion models by identifying and addressing a critical limitation in the use of discriminator guidance. The rigorous theoretical analysis, well-motivated solution, and empirical validation all support the paper's claims. While the increased computational cost and the sensitivity of new hyperparameters present some practical challenges, the benefits in terms of sample quality and the increased theoretical understanding outweigh these concerns. The insights and alternative objective provided in the paper are very important and will have lasting effects on diffusion models and training in this setting. The paper advances our understanding of DG and makes a practically significant improvement.

Score: 8

- **Score**: 8/10

### **[FreeFlux: Understanding and Exploiting Layer-Specific Roles in RoPE-Based MMDiT for Versatile Image Editing](http://arxiv.org/abs/2503.16153v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper "FreeFlux: Understanding and Exploiting Layer-Specific Roles in RoPE-Based MMDiT for Versatile Image Editing" delves into the inner workings of FLUX, a state-of-the-art text-to-image generation model. The authors focus on how Rotary Position Embedding (RoPE) impacts the self-attention mechanism within FLUX's Multimodal Diffusion Transformer (MMDiT) architecture. They introduce a novel automated probing strategy to disentangle the roles of positional information versus content similarity in different self-attention layers.  Based on their findings, they propose a training-free image editing framework that categorizes editing tasks (position-dependent, content-similarity-dependent, region-preserved) and designs tailored key-value injection strategies for each type. Experimental results demonstrate superior performance compared to existing editing methods, particularly in preserving semantic content and achieving seamless modifications.

**Critical Evaluation:**

*   **Novelty:** The core novelty lies in the mechanistic analysis of RoPE's influence within MMDiT. Existing work treats these models as black boxes. The automated probing strategy is a significant contribution, enabling a deeper understanding of layer-specific functionality. The task-specific editing strategies built upon this analysis are also innovative, moving beyond generic editing approaches. The specific manipulation of ROPE is a new and interesting approach to understanding layer functionality.

*   **Significance:** The work has potential to be significant because it opens up new avenues for controlling and manipulating diffusion models without requiring retraining. This is crucial given the computational expense of training such models. The findings about the role of positional information and content similarity could generalize to other transformer-based architectures in vision and language. Training free image editing has wide ranging appeal, and the methods the authors employ have a degree of elegance.

*   **Strengths:**

    *   **Thorough Analysis:** The paper presents a well-designed experimental methodology for probing the internal workings of FLUX.
    *   **Task-Specific Customization:** The framework demonstrates a clear understanding of different editing task requirements and tailors editing strategies accordingly.
    *   **Strong Results:** The experimental results convincingly demonstrate the superiority of the proposed approach over existing methods.
    *   **Clarity:** The paper is well-written and easy to follow, making the complex concepts accessible.

*   **Weaknesses:**

    *   **Model Specificity:** The analysis is primarily focused on FLUX. While the underlying principles might be applicable to other models, more extensive validation would strengthen the claims.
    *   **Limited Scope:** While the categorized editing tasks cover a reasonable range, there might be other editing paradigms that are not addressed.
    *   **User Study:** The user study is described in limited detail. A more rigorous description of the study protocol and participant demographics would be welcome.
    *   **Real Images:** While real-image editing is demonstrated, the core evaluation relies on synthetically generated data using ChatGPT and synthetic edits. More work could be done to see if the identified trends transfer over to this domain.

*   **Potential Influence:** The paper is likely to influence the research community by inspiring further investigations into the layer-specific functionality of diffusion models. The proposed editing framework could be adopted and extended by other researchers, leading to more versatile and controllable image editing techniques. The open source nature of the codebase increases the potential impact.

**Justification for Score:**

While the paper is well-executed and presents novel findings, some limitations temper the overall impact. The model-specific analysis and the synthetic evaluation environment are areas for improvement. Despite these weaknesses, the mechanistic analysis is a major strength. The overall method shows a high degree of sophistication, and offers a convincing demonstration of training free versatile image editing.

Score: 8

- **Score**: 8/10

### **[MathFusion: Enhancing Mathematic Problem-solving of LLM through Instruction Fusion](http://arxiv.org/abs/2503.16212v1)**
- **Summary**: Here's a summary and critical evaluation of the "MathFusion: Enhancing Mathematic Problem-solving of LLM through Instruction Fusion" paper:

**Summary:**

The paper introduces MathFusion, a novel data augmentation framework designed to enhance the mathematical reasoning capabilities of Large Language Models (LLMs). Unlike traditional data augmentation methods that focus on modifying individual problem instances, MathFusion synthesizes new training examples by fusing pairs of existing mathematical problems.  The framework employs three distinct fusion strategies: (1) *Sequential Fusion*, which chains related problems to model solution dependencies; (2) *Parallel Fusion*, which combines analogous problems to reinforce conceptual understanding; and (3) *Conditional Fusion*, which creates context-aware problems to enhance reasoning flexibility.  The authors create a new dataset, MathFusionQA, using these techniques and fine-tune several LLMs (DeepSeekMath-7B, Mistral-7B, Llama3-8B) on it.  Experimental results demonstrate substantial improvements in mathematical reasoning performance across diverse benchmarks, with relatively high data efficiency compared to other data augmentation approaches.

**Critical Evaluation:**

*   **Novelty:** The core idea of fusing mathematical problems to create more complex and relationally-aware training data is novel.  It moves beyond simple paraphrasing or difficulty adjustments, attempting to instill a deeper understanding of mathematical concepts and their interconnections. The three proposed fusion strategies are well-defined and intuitive.

*   **Significance:**  Mathematical reasoning is a crucial capability for LLMs, and improving it is a significant goal.  The results presented in the paper are promising. The reported accuracy gains across various benchmarks, while using only a relatively small dataset, suggest that MathFusion is an effective approach.  The fact that it combines well with DART-Math further strengthens its potential impact.

*   **Strengths:**
    *   **Clear Problem Definition:**  The paper clearly articulates the limitations of existing data augmentation techniques and the need for approaches that capture relational mathematical knowledge.
    *   **Well-Defined Framework:** The MathFusion framework is well-defined, with clearly explained fusion strategies and implementation details.
    *   **Empirical Validation:** The empirical results are strong, demonstrating significant improvements across multiple benchmarks and LLM architectures. The ablation study and analysis of the impact of teacher model provides additional insights.
    *   **Data Efficiency:** Demonstrating effectiveness with a smaller dataset compared to other methods is a significant advantage.
    *   **Complementarity:** Showing that MathFusion is complementary to existing techniques like DART-Math significantly increases its value to the community.

*   **Weaknesses:**
    *   **Reliance on Strong LLM for Synthesis:** The generation of fused problems and their solutions relies heavily on a strong LLM (GPT-4o-mini).  The quality of the generated data is therefore directly dependent on the capabilities of the teacher model. The paper acknowledges potential errors and ambiguities but more detailed error analysis could be useful.
    *   **Problem Pair Selection:** The reliance on embedding similarity for problem pair selection may not always lead to the most semantically or mathematically relevant combinations.  Exploring alternative problem pairing strategies could further improve the framework. The study mentions the problem pair selection and how a similar type and context is preferred, but does not mention the specific hyperparameters or embeddings used.
    *   **Limited Exploration of Fusion Strategies:** While the three fusion strategies are a good starting point, there may be other, more sophisticated ways to combine mathematical problems. The impact of each fusion strategy is measured and it is found that conditional fusion adds the least.

**Justification for Score:**

The paper presents a novel and promising approach to data augmentation for mathematical reasoning.  The improvements demonstrated by MathFusion are significant, and the framework has the potential to influence future research in this area. The paper has good empirical evidence, is well-written, and has a clear message. While the reliance on a strong LLM for data synthesis is a potential limitation, the authors acknowledge this and the results are still impressive. The paper is very helpful for other researchers.

**Score: 8**

- **Score**: 8/10

### **[Chain of Functions: A Programmatic Pipeline for Fine-Grained Chart Reasoning Data](http://arxiv.org/abs/2503.16260v1)**
- **Summary**: Here is a summary and critical evaluation of the paper "Chain of Functions: A Programmatic Pipeline for Fine-Grained Chart Reasoning Data":

**Summary:**

The paper introduces Chain of Functions (CoF), a novel programmatic pipeline for generating high-quality reasoning data for chart understanding. CoF addresses the scarcity of rationale data for training multimodal large language models (MLLMs) by programmatically exploring reasoning paths using atomic functions (e.g., maximum, arithmetic operations). It generates diverse function chains which are then translated to natural language rationales and questions using a moderate-sized LLM. This method ensures data precision and diversity, provides built-in rationales for fine-grained evaluation, and reduces reliance on extremely large models. The authors construct a dataset named ChartCoF using CoF, comprising 1.4k complex reasoning Q&A for fine-grained analysis and 50k Q&A for reasoning enhancement. Experiments demonstrate state-of-the-art performance of fine-tuned MLLMs on widely used benchmarks.

**Critical Evaluation:**

*   **Novelty:** The CoF pipeline presents a novel approach to data generation for chart reasoning. Unlike prior methods that rely on direct prompting of LLMs, CoF programmatically explores chart elements via atomic functions, allowing for more controlled and diverse reasoning paths. The key innovation is the combination of program-based functional discovery and reverse linguistic CoT data synthesis, which significantly reduces hallucinations and enables more precise supervision. The approach of using atomic functions and translating to language is creative.

*   **Significance:** The scarcity of high-quality rationale data is a major bottleneck for training effective MLLMs for chart understanding. ChartCoF directly addresses this issue by providing a large-scale dataset of complex reasoning Q&A with built-in rationales. The fine-grained evaluation on ChartCoF reveals strengths and weaknesses of existing MLLMs on different question types, offering valuable insights for model development. The demonstrated state-of-the-art performance on standard benchmarks suggests the effectiveness of ChartCoF for reasoning enhancement. The impact is further enhanced by providing detailed insights on how MLLMs perform against various question taxonomies and what types of reasoning they struggle with. This provides specific information for future model design.

*   **Strengths:**
    *   The programmatic approach ensures data precision and diversity compared to free-form generation.
    *   The use of function chains provides built-in rationales for fine-grained evaluation and explainability.
    *   The pipeline eliminates reliance on extremely large models for data generation, increasing practicality and scalability.
    *   The empirical results demonstrate state-of-the-art performance on standard benchmarks.
    *   The detailed analysis on question taxonomies offer valuable insights.
    *   The dataset and code are publicly available, promoting reproducibility and further research.

*   **Weaknesses:**
    *   The design of atomic functions and their corresponding conditions may require domain expertise and could be a potential source of bias or limitation in the generated data. The functions may not cover all possible reasoning patterns.
    *   While the CoF pipeline reduces reliance on extremely large models, the use of a moderate-sized LLM (Qwen2.5-32B-instruct) for linguistic transfer still requires computational resources.
    *   Although the paper claims to address the challenge of out-of-distribution data, more rigorous evaluation on truly unseen chart types and question complexities would strengthen the analysis. The presented OOD experiments are limited to separating by regular/extra chart types.
    *   The paper only explores ChartCoF in the context of chart reasoning. More discussion or preliminary results on the applicability of CoF in other tasks is needed to support the claim about CoF's broader applications.

*   **Potential Influence:** The CoF pipeline could inspire new data generation methodologies for other complex reasoning tasks beyond charts. The paradigm of function-governed rationale generation could be adopted in various domains where explainability and precision are crucial. The public availability of ChartCoF is likely to foster further research in chart understanding and MLLM development.

**Justification:**

The paper introduces a novel and effective data generation pipeline for a challenging task in multimodal learning. The detailed analysis of MLLM performance on different question types and the empirical demonstration of state-of-the-art results on standard benchmarks showcase the significance of ChartCoF. While the approach has some limitations, such as reliance on domain expertise for atomic function design and the continued need for a moderately-sized LLM, its strengths in data precision, diversity, and practicality outweigh these weaknesses. CoF represents a valuable contribution to the field and has the potential to influence future research in chart understanding and MLLM development.
Score: 8

- **Score**: 8/10

### **[CaKE: Circuit-aware Editing Enables Generalizable Knowledge Learners](http://arxiv.org/abs/2503.16356v1)**
- **Summary**: Here's a summary and critical evaluation of the CaKE paper:

**Summary:**

The paper "CaKE: Circuit-aware Editing Enables Generalizable Knowledge Learners" tackles the challenge of improving knowledge editing (KE) in large language models (LLMs), specifically focusing on enhancing generalization in multi-hop reasoning tasks.  Existing KE methods often modify knowledge locally (within a single layer) which is insufficient to effectively integrate updated information into the reasoning circuits LLMs use for inference.  The authors analyze these reasoning circuits and observe that failures in multi-hop reasoning often stem from either critical information not being properly routed or from weak signals preventing effective reasoning. To address these issues, they propose CaKE (Circuit-aware Knowledge Editing). CaKE involves strategically curating training data tailored to the LLM's reasoning architecture. This data forces the model to use the modified knowledge during training, stimulating the development of appropriate reasoning circuits. Experiments show that CaKE leads to significant improvements in multi-hop reasoning accuracy, outperforming existing KE methods, and maintains general capabilities.

**Critical Evaluation:**

*   **Novelty:** The paper's core novelty lies in its **circuit-aware approach** to knowledge editing. While previous works have investigated the internal mechanisms of LLMs and knowledge editing separately, CaKE explicitly connects the two. Analyzing reasoning circuits to guide the design of editing strategies and curated data is a significant step beyond purely parameter-based or data-centric KE methods. The design of the circuit-aware tasks with "ad-hoc features" to avoid data leakage also demonstrates a good degree of novelty. It takes the "why" current KE models fail and addresses the point by explicitly aligning edits with the LLM's native reasoning architecture, and transforms static knowledge updates into generalizable knowledge reasoning circuit-models.

*   **Significance:**  The problem of generalizing knowledge edits to downstream tasks, especially multi-hop reasoning, is a critical bottleneck for the practical deployment of KE.  CaKE makes a **substantial contribution** by demonstrating a way to improve this generalization.  The method's performance gains over strong baselines on a challenging benchmark (MQUAKE) solidify its significance. By improving generalization, KE can be more reliably used to update and correct LLMs without detrimental effects on their reasoning abilities. The analysis of circuit failures is important in understanding what type of training/modifications work.

*   **Strengths:**
    *   **Strong Motivation and Problem Definition:** The paper clearly articulates the limitations of existing KE methods and provides compelling evidence for the need for a circuit-aware approach. The visual representation of circuit failures in multi-hop reasoning in Figure 2 is particularly helpful.
    *   **Methodological Rigor:** The paper describes CaKE in detail and provides a thorough experimental evaluation.
    *   **Mechanistic Analysis:** The paper provides a mechanistic analysis including a section on circuit failures, demonstrating the efficacy of CaKE and providing insight into its working. The analysis of the position and the hops is rigorous and sound.
    *   **Results:** Strong performance on multiple datasets/settings that are also maintained on general capability test sets shows that the general capabilities were not harmed.
    *   **Well-written:** The paper is well-structured and easy to follow.

*   **Weaknesses:**
    *   **Complexity of Data Curation:** The process of generating circuit-aware training data involves creating specialized templates and using language models to generate the data. This may be a barrier to entry for some researchers and practitioners. Furthermore, the GLM-4 generation process can be potentially expensive to use.
    *   **Limited scope:** The models used in the evaluations could potentially be a weakness as it would have been good to see a similar performance with a more modern model than LLAMA3-8B and Qwen2.5.

*   **Potential Influence:** The work has the potential to influence future research on knowledge editing by shifting the focus from isolated parameter updates to more holistic circuit-level interventions. The detailed analysis of the reasoning circuits can inform the design of more effective KE methods and provide a framework for understanding the limitations of existing approaches. Furthermore, it could inspire similar circuit-aware strategies for other LLM tasks beyond knowledge editing.

**Justification for Score:**

CaKE addresses a critical problem in the knowledge editing space with a novel circuit-aware approach, strong methodological rigor, and encouraging results. While data curation complexity and limited experiments on more recent and popular models can be viewed as limitations, they do not overshadow the paper's significant contributions.

Score: 8

- **Score**: 8/10

### **[Scale-wise Distillation of Diffusion Models](http://arxiv.org/abs/2503.16397v1)**
- **Summary**: Here's a summary and critical evaluation of the presented research paper on Scale-wise Distillation of Diffusion Models (SWD):

**Summary:**

The paper introduces SWD, a novel scale-wise distillation framework for diffusion models (DMs). SWD leverages the insight that diffusion processes can operate effectively at lower data resolutions during early stages (high noise levels).  The method gradually increases spatial resolution during the DM sampling process, using a single model.  The authors also introduce a patch distribution matching (PDM) loss to enforce finer-grained similarity between generated and target distributions. The results demonstrate that SWD achieves faster inference speeds and outperforms traditional full-resolution distilled models with similar computational budgets, even competing with next-scale prediction models.  The performance gains are supported by both automated metrics and human preference studies.

**Critical Evaluation:**

*   **Novelty:** The core idea of scale-wise distillation is novel.  The paper explicitly connects the practice of spectral autoregression with diffusion models to the advantages gained using coarse-to-fine processes for visual generation.  Combining it with techniques such as PDM adds to the novelty of the contributions. The integration of PDM loss specifically for aligning patch distributions within a diffusion distillation context is a valuable contribution to the field. It avoids the necessity of additional models, a major advantage over traditional GAN or discriminator approaches.

*   **Significance:** The significance lies in improving the efficiency of diffusion models without sacrificing image quality. The high computational cost of DMs is a major bottleneck.  SWD directly addresses this by reducing the computational load during the sampling process. Demonstrating its effectiveness on large-scale text-to-image models like SDXL/SD3.5 further amplifies its practical impact. The gains in inference speed (2.5x-10x faster) alongside improved/comparable quality are compelling.

*   **Strengths:**
    *   The central idea is well-motivated by spectral analysis and links to previous coarse-to-fine approaches.
    *   The proposed PDM loss is a simple yet effective addition, requiring no additional models.
    *   Comprehensive evaluation with both automated metrics and human preference studies strengthens the claims.
    *   The demonstrated speedups and quality improvements are significant for the DM community.
    *   The work is built on top of existing distillation methods (DMD2) showcasing compatibility to existing DM pipelines.
    *   Clear experimental setup and ablation studies thoroughly examine design choices.

*   **Weaknesses:**
    *   While effective, the reliance on existing distribution matching methods (DMD2) means SWD is, in part, an extension of existing work. While the *combination* of scale-wise processing and PDM loss is novel, a portion of the performance comes from a prior baseline.
    *   The experimental section, while extensive, could benefit from a more direct comparison against cascaded diffusion models. Even though those models have multiple DMs operating on different scales, a comparison can solidify SWD's effectiveness using a *single* model.

*   **Potential Influence:**  This paper has a strong potential for influence. The simplicity and effectiveness of SWD make it likely to be adopted by researchers and practitioners working on diffusion models. The approach can encourage further investigation into dynamic resolution adjustments during the diffusion process and potentially lead to even more efficient DM sampling strategies. The framework also encourages efficient use of computation since it reduces overall training and inference steps significantly.

**Justification for Score:**

SWD offers a valuable contribution to the DM community by addressing a key limitation – computational cost – and by incorporating new insights regarding the spectrum of images. The framework combines existing methods but successfully shows how they can be optimized and improved with this scale-wise concept and also achieves state-of-the-art performance. The extensive evaluation and detailed ablations significantly bolster the paper. While building upon existing work (DMD2) and a single comparison could be improved (cascaded diffusion models), the novelty and significance of the proposed approach deserve a strong score.

Score: 8

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
### **[Tokenize Image as a Set](http://arxiv.org/abs/2503.16425v1)**
### **[XAttention: Block Sparse Attention with Antidiagonal Scoring](http://arxiv.org/abs/2503.16428v1)**
