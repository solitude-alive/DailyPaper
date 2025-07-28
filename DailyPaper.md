# The Latest Daily Papers - Date: 2025-07-28
## Highlight Papers
### **[A Deep Dive into Retrieval-Augmented Generation for Code Completion: Experience on WeChat](http://arxiv.org/abs/2507.18515v1)**
- **Summary**: Here's a summary and critical evaluation of the research paper:

**Summary:**

The paper presents a deep dive into Retrieval-Augmented Generation (RAG) for code completion within the industrial-scale, closed-source codebase of WeChat at Tencent.  It systematically investigates the performance of identifier-based and similarity-based RAG methods using 26 open-source Large Language Models (LLMs) with parameter sizes ranging from 0.5B to 671B.  The authors develop a fine-grained data preprocessing algorithm for constructing a retrieval corpus from large-scale C++ projects, focusing on extracting class, function, and macro definitions. They compare various retrieval techniques (BM25, CodeBERT, UniXcoder, CoCoSoDa, GTE-Qwen) and demonstrate the effectiveness of similarity-based RAG, particularly when combining lexical (BM25) and semantic (GTE-Qwen) retrieval. Finally, a developer survey is conducted to validate the practical utility of these methods.

**Critical Evaluation:**

*   **Strengths:**

    *   **Real-world Application:** The most significant strength of this paper is its application to a large, real-world, closed-source codebase (WeChat). This addresses a crucial gap in existing RAG research, which often focuses on public datasets and benchmarks that may not accurately reflect the challenges of industrial software development.
    *   **Comprehensive Evaluation:**  The study is remarkably thorough.  The breadth of LLMs tested (26 different models across a wide range of parameter sizes) provides a rich dataset for understanding the scalability and generalizability of RAG techniques. The inclusion of different retrieval methods, along with the investigation of query formulation strategies (complete vs. incomplete queries) and the combination of retrieval techniques, paints a detailed picture of the RAG landscape.
    *   **Novel Preprocessing Algorithm:** The proposed data preprocessing algorithm for C++ codebases, including handling recursive dependencies, auto-generated code (protobuf), and macro specificity, is a valuable contribution.  This addresses specific challenges associated with C++ code that are often overlooked in more general RAG approaches.
    *   **Developer Validation:**  The developer survey is a vital component of the work. It provides valuable qualitative feedback that complements the quantitative metrics and helps assess the practical utility of the proposed RAG methods.
    *  **Detailed Error Analysis:** The inclusion of error pattern analysis within the developer survey provides insights into failure modes.

*   **Weaknesses:**

    *   **Limited Generalizability of WeChat-Specific Findings:**  While the use of the WeChat codebase is a strength, it also introduces a potential limitation.  Some of the observed performance characteristics may be specific to the coding style, architecture, and domain of the WeChat system. Generalizing these findings to other closed-source environments might require caution. It would be useful to demonstrate some general properties about the codebase to give hints about transferability to other codebases.
    *   **Evaluation Metrics:** Although CodeBLEU and Edit Similarity are employed, the community is moving towards more holistic code evaluation metrics such as pass@k based on unit tests or program executions. The human evaluation provides context here, but more advanced automatic metrics would add additional rigor to the conclusions.
    *   **Benchmark Dataset Details:** While the benchmark construction is described, more detailed information about the types of bugs (common API usage errors, logic errors, concurrency errors, etc.) and the difficulty levels could enhance the reproducibility and impact of the benchmark.

*   **Novelty and Significance:**

    *   The paper addresses a crucial and under-explored area: RAG for code completion in closed-source environments. Most prior work has focused on open-source data, overlooking the unique challenges and privacy requirements of industrial settings.
    *   The systematic evaluation of different RAG methods and retrieval techniques, along with the insights from the developer survey, provides valuable guidance for practitioners looking to implement RAG-based code completion in real-world projects.
    *   The preprocessing algorithm and the analysis of the interplay between lexical and semantic retrieval are significant contributions.
    *   The validation through a developer survey is commendable, solidifying the real-world value and relevance of the study.

*   **Potential Impact:**

    *   The paper is likely to influence the development of more effective and practical RAG-based code completion tools for industrial software development.
    *   It provides a roadmap for researchers interested in exploring the challenges and opportunities of applying LLMs to closed-source codebases.
    *   The C++ preprocessing algorithm can be adopted and extended by other researchers and practitioners working with code analysis and generation.

**Justification for Score:**

This paper is a strong and valuable contribution to the field. While the findings might be somewhat specific to the WeChat environment, the thorough methodology, detailed analysis, and practical validation provide a strong foundation for future research and development. The paper effectively bridges the gap between academic research and real-world industrial application. Given the comprehensiveness and the contribution to a previously under-explored area, this paper warrants a high score, but the WeChat specificity prevents a perfect score.

Score: 8.5

- **Score**: 8/10

### **[DR.EHR: Dense Retrieval for Electronic Health Record with Knowledge Injection and Synthetic Data](http://arxiv.org/abs/2507.18583v1)**
- **Summary**: Okay, I will provide a summary and a critical evaluation of the paper "DR.EHR: Dense Retrieval for Electronic Health Record with Knowledge Injection and Synthetic Data."

**Summary:**

The paper introduces DR.EHR, a dense retrieval model specifically designed for electronic health record (EHR) retrieval.  The authors address the limitations of existing general-domain and biomedical dense retrieval models, which often lack sufficient medical knowledge or are trained on mismatched corpora, by proposing a two-stage training pipeline. The first stage involves knowledge injection from a biomedical knowledge graph (BIOS) using medical entity extraction, and abbreviation expansion. The second stage leverages large language models (LLMs) to generate synthetic training data to improve the diversity of the training examples. The model is trained using contrastive learning with in-batch negatives.  The authors evaluate DR.EHR on the CliniQ benchmark and demonstrate state-of-the-art performance, particularly on challenging semantic matching types.  They also conduct ablation studies to validate the effectiveness of each component in the training pipeline and showcase generalizability on EHR question answering datasets.

**Critical Evaluation:**

*   **Novelty:** The paper's novelty lies in its specific application of dense retrieval techniques to the EHR domain with a customized training pipeline. While the individual components (knowledge injection, synthetic data generation) are not entirely new, their combination and adaptation within the context of EHR retrieval is a significant contribution. The careful selection of BIOS and the specific prompting strategies for LLM-based data synthesis represent a well-engineered approach. The focus on overcoming the semantic gap issue in EHR retrieval, particularly in challenging scenarios such as abbreviations and implicit inference is also laudable.
*   **Significance:** The work addresses a critical problem in clinical practice: efficient EHR retrieval. Improved EHR retrieval systems can directly benefit physicians by enabling faster access to relevant patient information, leading to better and potentially faster diagnosis and treatment. Demonstrating superior performance on a public benchmark like CliniQ establishes a new state-of-the-art and provides a strong baseline for future research. The insights gained from the ablation studies regarding the contribution of each component provide valuable guidance for other researchers in this area. The extension of the model's evaluation to EHR QA datasets, even as a supplementary analysis, suggests its broader applicability and generalizability.
*   **Strengths:**

    *   **Strong empirical results:** The paper presents compelling empirical evidence of DR.EHR's superiority over existing models on the CliniQ benchmark.
    *   **Well-designed training pipeline:** The two-stage training pipeline is thoughtfully designed to address the specific challenges of EHR retrieval, effectively leveraging knowledge injection and synthetic data generation.
    *   **Detailed analysis:** The authors conduct thorough analyses to understand the model's strengths and weaknesses, including semantic match assessment and query type assessment.
    *   **Ablation studies:** The ablation studies provide valuable insights into the contribution of each component of the training pipeline.
    *   **Generalizability demonstration:** The supplementary experiments on EHR QA datasets demonstrate the model's generalizability to natural language questions.
*   **Weaknesses:**

    *   **Dependency on LLMs:** The synthetic data generation relies heavily on LLMs, which can introduce biases or inaccuracies. While the authors perform manual validation, a more rigorous analysis of the generated data quality would be beneficial.
    *   **Limited benchmark coverage:** The primary evaluation is conducted on CliniQ, which, while useful, may not fully represent the diversity of real-world EHR retrieval scenarios. More evaluation on other diverse EHR datasets would strengthen the claims.
    *   **Complexity:** The pipeline involves multiple steps and components (knowledge injection, LLM-based data generation), which might make it harder to reproduce or adapt to different settings compared to simpler models. More detail on the specific choices made in the pipeline would be valuable for readers.

*   **Potential Influence:** The paper is likely to have a significant influence on the field of EHR retrieval. The proposed approach provides a robust and effective solution to address the challenges of semantic gap and insufficient training data. The release of DR.EHR and the detailed analysis in the paper will serve as a valuable resource for other researchers in this area. The paper encourages further research on leveraging knowledge graphs and synthetic data for improving EHR retrieval models.

**Score:** 8

**Justification:**

I assign a score of 8 because the paper makes a substantial contribution to the field by presenting a novel and effective dense retrieval model specifically tailored for EHR retrieval. The strong empirical results on a public benchmark, the detailed analyses, and the ablation studies provide compelling evidence of the model's superiority. While there are some limitations, such as the dependence on LLMs and limited benchmark coverage, these do not significantly detract from the overall quality and significance of the work. The carefully crafted pipeline and the promising generalizability results suggest this is an important step towards making EHR retrieval more practical for clinical application.

- **Score**: 8/10

### **[Linear Memory SE(2) Invariant Attention](http://arxiv.org/abs/2507.18597v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper proposes a novel method for SE(2) invariant attention that scales linearly with the number of objects in a scene, addressing the quadratic memory requirements of existing approaches. The method, named SE(2) Fourier, leverages a Fourier series approximation to encode relative positions in a block-diagonal matrix of 2D rotations, enabling the use of standard scaled dot-product attention (SDPA) with linear memory consumption. The authors demonstrate the effectiveness of their approach in agent simulation, showing performance improvements over comparable non-invariant architectures and existing RoPE methods.

**Critical Evaluation:**

*   **Novelty:** The core idea of using a Fourier series approximation for SE(2) invariant attention to achieve linear memory scaling is novel.  While RoPE and its extensions address relative position encoding, they are either limited to abelian groups or, like GTA, do not effectively encode relative *pose* in SE(2) with linear memory. SE(2) Fourier directly addresses the challenge of quadratic memory in SE(2) invariant architectures.

*   **Significance:**  The significance lies in its potential impact on large-scale multi-agent systems.  The quadratic memory bottleneck has hindered the application of SE(2) invariant networks to scenarios with a large number of interacting agents. By achieving linear scaling, the proposed method opens the door to training more sophisticated models with larger contexts. Agent simulation is an important area for robotics and autonomous driving, and this efficiency improvement is important.

*   **Strengths:**

    *   **Mathematical Rigor:** The paper provides a clear mathematical derivation of the Fourier approximation and demonstrates its equivalence to the standard scaled dot-product attention.

    *   **Empirical Validation:** The experimental results on agent simulation show performance gains compared to non-invariant and existing SE(2) approaches, particularly in scenarios involving turning trajectories.  This validates the practical utility of the proposed method.

    *   **Practical Implementation:** By enabling standard SDPА and Flash Attention, the approach can easily be adopted and benefit from advances in attention mechanisms.

*   **Weaknesses:**

    *   **Approximation Error:** The Fourier series approximation introduces error, which is directly related to the key position magnitude and the number of basis elements. While the paper demonstrates that this error can be kept within acceptable bounds, a more in-depth analysis of the trade-off between accuracy and computational cost would be beneficial.

    *   **Limited Experimental Scope:** The experiments are primarily focused on agent simulation. Demonstrating the benefits of SE(2) Fourier on other driving tasks (e.g., motion forecasting) or other areas of robotics (e.g., ground-based navigation) would strengthen the paper's impact.

    *   **Comparison to baselines**: The comparison of SE(2) Fourier with other SE(2) encoding schemes is limited.

*   **Potential Impact:**

    *   The linear memory scaling could enable training of more complex and scalable SE(2) invariant models for autonomous driving and robotics.
    *   The approach could inspire further research into efficient approximations for other group-invariant operations.

**Rationale for Score:**

The paper presents a genuinely novel and technically sound approach to SE(2) invariant attention. The ability to achieve linear memory scaling is a significant step forward for large-scale multi-agent systems. While the paper has some limitations in terms of experimental scope and further analysis of approximation error, the potential impact on the field warrants a high score.

Score: 8

- **Score**: 8/10

### **[IsaMini: Redesigned Isabelle Proof Lanugage for Machine Learning](http://arxiv.org/abs/2507.18885v1)**
- **Summary**: Okay, I will provide a summary and critical evaluation of the paper "IsaMini: Redesigned Isabelle Proof Language for Machine Learning".

**Summary:**

The paper addresses the challenge of improving Neural Theorem Proving (NTP) by focusing on the design of the proof language.  Specifically, it argues that Large Language Models (LLMs), which are central to NTP, are sensitive to the representation of proofs.  The authors introduce "MiniLang," a redesigned proof language for Isabelle/HOL, aiming to be more machine-friendly than the standard Isar language.  MiniLang simplifies Isar by removing redundancies and features designed primarily for human readability.  They created a translator from Isar to MiniLang, allowing them to train LLMs on a substantial corpus of existing Isabelle/HOL proofs translated into the new language.  Experiments demonstrate that LLMs fine-tuned on MiniLang achieve significantly higher success rates (up to 29% improvement) on the PISA benchmark compared to those trained directly on Isar. This suggests that a more streamlined proof language can substantially enhance the performance of NTP systems. They achieved new state-of-the-art performance on the PISA benchmark, exceeding previous results. A REPL infrastructure capable of cluster usage for Isabelle is another contribution.

**Critical Evaluation:**

*   **Novelty:** The core idea of redesigning a proof language specifically for machine learning is novel. While prior work has explored different *representations* of proofs (e.g., graph-based), this paper takes the more drastic step of creating a new language. This is a significant departure from simply working with existing languages. The creation of the translator and the subsequent fine-tuning and evaluation is also a substantive contribution.
*   **Significance:** The potential impact of this work is high. If the results generalize to other proof assistants and benchmarks, it could fundamentally change how NTP systems are developed. It shifts the focus from purely algorithmic improvements in LLMs to a more holistic approach that considers the interaction between the model and the language it's learning. The state-of-the-art results achieved in PISA further amplify the significance of the contribution. A socket-based REPL infrastructure is a helpful practical tool for NTP development.

*   **Strengths:**
    *   **Strong Empirical Results:** The paper presents compelling experimental evidence to support its claims. The improvement on the PISA benchmark is substantial.
    *   **Well-Defined Problem and Solution:** The paper clearly identifies the issue (Isar's suitability for LLMs), proposes a specific solution (MiniLang), and evaluates it rigorously.
    *   **Comprehensive Approach:** The authors not only design a new language but also provide a translator and conduct extensive experiments.
    *   **Addresses a Gap:** It explicitly focuses on a gap in NTP research, which is the lack of attention given to proof language design in real-world proof engineering domains.

*   **Weaknesses:**
    *   **Generalizability:** A major question is whether MiniLang's improvements are specific to Isabelle/HOL and the PISA benchmark or if the principles can be generalized. Will similar simplifications of proof languages in Coq or Lean yield comparable results? The paper doesn't address this directly, although the discussion of extending MiniLang to other provers is provided in the discussions.
    *   **Human Readability:** While the explicit goal is to make proofs more machine-readable, the resulting language may be less understandable for human developers. This could affect the maintainability and extensibility of proofs written in MiniLang. While it's not the explicit goal of this work, it could affect future research into explainable NTP.
    *   **Reliance on ATP:** The success of MiniLang critically depends on the power of the underlying ATP (Sledgehammer*). The paper doesn't fully explore scenarios where ATP automation is weaker.

*   **Justification for Score:**

The paper presents a clearly articulated, well-executed, and empirically validated idea. It introduces a novel approach to improve NTP by redesigning the proof language. The significant improvements on the PISA benchmark, the comprehensive experimental setup, the publicly available REPL, and the potential impact on the field justify a high score. However, the limitations related to generalizability and potential impact on human readability prevent it from receiving a perfect score. I considered a score between 7 and 9. I am leaning towards 8 because achieving new state-of-the-art is an exceptional accomplishment, showcasing the validity of the hypothesis.

**Score: 8**

- **Score**: 8/10

### **[A Toolbox, Not a Hammer -- Multi-TAG: Scaling Math Reasoning with Multi-Tool Aggregation](http://arxiv.org/abs/2507.18973v1)**
- **Summary**: Here's a summary and critical evaluation of the provided paper:

**Summary:**

The paper introduces Multi-TAG, a Multi-Tool AGgregation framework designed to improve mathematical reasoning in large language models (LLMs). Unlike existing tool-augmented LLM (TALM) approaches that typically use a single tool at each reasoning step, Multi-TAG prompts the LLM to concurrently invoke multiple tools (e.g., Python code execution, WolframAlpha queries, natural language reasoning).  It then aggregates the diverse outputs from these tools, using their consensus to verify and refine the reasoning process, thus enhancing solution robustness and accuracy. A key feature of Multi-TAG is that it's a finetuning-free, inference-only framework, making it applicable to any LLM backbone, including large open-weight models and proprietary frontier models. The paper evaluates Multi-TAG on challenging benchmarks like MATH500, AIME, AMC, and OlympiadBench, demonstrating consistent and substantial performance improvements over state-of-the-art baselines across various LLM backbones. The authors also explore the trade-offs between performance and compute costs.

**Critical Evaluation:**

**Strengths:**

*   **Novelty:** The core idea of multi-tool aggregation for math reasoning is a significant step beyond single-tool selection. It addresses a key limitation of existing TALMs by leveraging cross-validation between different tools' outputs, capitalizing on their individual strengths and mitigating weaknesses.  This is a smart approach to increasing confidence in intermediate reasoning steps.
*   **Significance:** Improving the mathematical reasoning capabilities of LLMs is a crucial goal. The reported performance gains on challenging benchmarks clearly demonstrate the effectiveness of the Multi-TAG framework.
*   **Practicality:** The finetuning-free, inference-only nature of Multi-TAG is a major advantage. It makes the framework readily adaptable to a wide range of LLMs, including those that are expensive to finetune or are proprietary with restricted access. This significantly enhances the accessibility and applicability of the method.
*   **Comprehensive Evaluation:** The paper presents a thorough evaluation on multiple challenging benchmarks and different LLM backbones.  The ablation studies provide valuable insights into the importance of various components of the Multi-TAG framework, such as the consistency threshold and candidate step selection.
*   **Detailed Analysis:** The paper goes beyond simply reporting performance numbers and provides detailed analyses of performance across different difficulty levels, problem subjects, and hyperparameter settings.  This facilitates understanding of the framework's behavior and provides guidance for practical usage.

**Weaknesses:**

*   **Computational Cost:** While the paper discusses the tunability of compute costs, the multi-tool invocation approach inherently increases the computational requirements at inference time. Although the consistency threshold helps, a deeper exploration of strategies to further reduce the computational overhead without sacrificing performance could be beneficial.  Quantifying the actual cost savings compared to inference-time scaling techniques that use just CoT would be valuable.
*   **Limited Tool Set:** The paper uses three tools (CoT reasoning, Python code execution, and WolframAlpha queries). Exploring a broader set of tools and strategies for dynamically selecting the *most relevant* subset of tools during inference could potentially further improve performance.
*   **Prompt Engineering Dependence:** Like many LLM-based approaches, Multi-TAG relies on prompt engineering. While the paper provides prompts in the appendix, a more detailed discussion of the prompt design process and the sensitivity of performance to different prompting strategies would be valuable.  This would enhance reproducibility and allow others to build upon the work more effectively.
*   **Error Analysis:** A deeper dive into the types of errors that Multi-TAG still makes, and the circumstances under which it fails, would be beneficial. This could point to areas for future improvement and help to identify the limitations of the current framework.
*   **Limited comparison with compute-scaled, fine-tuned approaches:** While the paper emphasizes the benefits of being finetuning-free, a clearer comparison of cost-performance tradeoffs with approaches that do finetune (especially in the context of compute scaling at inference time) would be helpful in positioning Multi-TAG within the broader solution space.

**Overall Assessment and Justification:**

The paper presents a novel and significant contribution to the field of LLM-based mathematical reasoning. The Multi-TAG framework addresses a key limitation of existing approaches and offers a practical and effective way to improve performance without requiring expensive finetuning. The comprehensive evaluation and detailed analysis further strengthen the paper's value. However, the limitations related to computational cost, toolset, error analysis, and prompting should be addressed to further enhance the work.

Score: 8

- **Score**: 8/10

### **[AEDR: Training-Free AI-Generated Image Attribution via Autoencoder Double-Reconstruction](http://arxiv.org/abs/2507.18988v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "AEDR: Training-Free AI-Generated Image Attribution via Autoencoder Double-Reconstruction":

**Summary:**

The paper introduces AEDR, a novel training-free method for attributing the origin of AI-generated images produced by models with continuous autoencoders. Unlike existing reconstruction-based approaches that rely on single reconstruction loss values, AEDR performs two consecutive reconstructions using the model's autoencoder. The ratio of these two reconstruction losses, calibrated by an image homogeneity metric, serves as the attribution signal. This approach is computationally efficient and shown to achieve higher attribution accuracy than existing reconstruction-based methods, especially on state-of-the-art latent diffusion models. The key idea is that autoencoders tend to produce more consistent reconstructions of belonging images compared to non-belonging images, leading to a loss ratio that is close to 1 for belonging images and significantly greater than 1 for non-belonging images.

**Critical Evaluation:**

* **Novelty:** The paper's novelty lies in the specific combination of double reconstruction, loss ratio analysis, and image homogeneity calibration for AI-generated image attribution. The use of autoencoders for reconstruction is not entirely new, but the way AEDR leverages double reconstruction and loss *ratios* instead of single losses to cancel out image complexity biases is a significant contribution. Moreover, the demonstration that current state-of-the-art methods are not well suited for certain newer diffusion models is a compelling point and serves to emphasize the need for novel methods. The combination of these techniques makes the approach highly novel.
* **Significance:** The significance stems from the increasing need to trace the origin of AI-generated content to mitigate misuse and intellectual property violations. AEDR's advantage in both accuracy and computational efficiency makes it a potentially valuable tool. Moreover, being training-free and not requiring gradient information, AEDR addresses some of the practical limitations of existing methods, especially in scenarios where full white-box access or significant computational resources are unavailable. Furthermore, it highlights the vulnerability of reconstruction methods to the complexity of the data. This is important for pushing future research into more robust methods.
* **Strengths:**
    *   **Strong empirical results:** The paper provides convincing experimental results across a variety of state-of-the-art latent diffusion models, demonstrating a substantial improvement in attribution accuracy and efficiency compared to established baselines.
    *   **Well-motivated approach:**  The choice of double reconstruction and loss ratio is logically sound and supported by observations about how autoencoders behave with belonging and non-belonging images.
    *   **Computationally efficient:** The method significantly reduces computational cost compared to gradient-based approaches, making it more practical for real-world applications.
    *   **Training-free:** Eliminating the need for training simplifies deployment and reduces the risk of overfitting to specific datasets.
    *   **Addresses a timely problem:** The increasing need for AI-generated image origin attribution adds to the paper's relevance.
* **Weaknesses:**
    *   **Limitations with quantized autoencoders:**  The paper acknowledges a performance degradation when applying AEDR to models with quantized autoencoders (VQ-VAE, MoVQ).  While the paper acknowledges this, it needs more exploration of the underlying reasons and potential mitigations or alternative approaches to address this limitation to broaden applicability of the method. The authors mention the cause may be attributed to the quantized latent space, but a more in-depth analysis would be valuable.
    *   **Inability to distinguish models sharing autoencoders:** The inability to differentiate images generated by similar models sharing an autoencoder further limits the applicability of the proposed technique. The paper could benefit from discussions on potential strategies to differentiate images generated by such models.
    *   **Dependence on pre-trained autoencoder:** AEDR is limited by the dependence on the quality and architecture of existing autoencoders.

**Justification for Score:**

The paper presents a well-motivated, novel, and empirically validated approach to an increasingly important problem. The significant improvements in accuracy and efficiency over existing methods, coupled with its training-free nature, warrant a high score. While the limitations related to quantized autoencoders and distinguishing between models sharing autoencoders are important, they do not diminish the overall contribution. The weaknesses highlight areas for future improvement, but the core idea and its demonstrated effectiveness are significant.

Score: 8

- **Score**: 8/10

### **[SelfRACG: Enabling LLMs to Self-Express and Retrieve for Code Generation](http://arxiv.org/abs/2507.19033v1)**
- **Summary**: Here's a summary and critical evaluation of the provided paper:

**Summary:**

The paper introduces SelfRACG, a novel approach to retrieval-augmented code generation (RACG) that empowers large language models (LLMs) to "self-express" their information needs for more effective code generation.  Unlike existing RACG methods that rely on external retrievers, SelfRACG enables the LLM to use its internal representations (hidden states) to guide the retrieval process. The method consists of two main components: (1) an Information Need Expression (INE) module, implemented using Layer-wise Low-Rank Adaptation (L-LoRA), and (2) a two-stage Information Need-Guided (ING) training strategy.  The experiments on RepoEval and CrossCodeEval benchmarks demonstrate that SelfRACG retrieves more relevant code fragments and improves code generation quality compared to vanilla RACG with various retrieval models (BM25, OpenAI Embeddings, GritLM, and NV-Embed-v2), while using significantly fewer computational resources than GritLM.

**Critical Evaluation:**

*   **Novelty:** The core idea of enabling LLMs to express their information needs directly from hidden states for retrieval is novel. The proposed architecture, with a parallel attention mechanism based on L-LoRA to capture retrieval information, is also innovative. The two-stage training strategy (ING) is well-designed to first learn code retrieval skills and then align them with the LLM's generation preferences. The concept of bridging the content gap between retrieved code and the specific generation needs of LLMs is valuable.

*   **Significance:**  The paper addresses a key limitation of existing RACG methods, which struggle to accurately fetch the most relevant information when the content diverges between consecutive code fragments.  The superior performance of SelfRACG on benchmark datasets indicates its potential to improve the quality and efficiency of code generation. The reduced computational cost compared to GritLM is also a significant advantage, making the approach more accessible. The work contributes a more efficient retrieval method in the field of LLM-based code generation, which is a very active and growing area of research. By focusing on how to align retrieval better to the generation process, the paper makes an important step towards more intelligent and effective code generation systems.

*   **Strengths:**

    *   The central concept is well-motivated by the observed limitations of traditional RACG.
    *   The technical implementation (INE module and ING training) is clearly described and reasonable.
    *   The experimental results are compelling and demonstrate the effectiveness of SelfRACG across different code LLMs and benchmarks.
    *   The ablation studies provide insights into the contributions of individual components.
    *   The comparison with GritLM highlights the resource efficiency of SelfRACG.

*   **Weaknesses:**

    *   The evaluation is limited to code generation tasks; it would be beneficial to explore the applicability of SelfRACG in other generation scenarios.
    *   The scale of the LLMs used in the experiments (up to 8B parameters) is relatively small compared to the largest LLMs currently available. While the resource efficiency is a key selling point, scaling up to larger models is an important future direction.
    *   The prompt template (Figure 5) may influence performance. It would be better to explore the sensitivity of the model to different prompts.

*   **Potential Impact:** SelfRACG has the potential to significantly impact the field of code generation by providing a more effective and resource-efficient approach to retrieval-augmented generation. It opens up new avenues for research in aligning retrieval and generation and could be extended to other generation tasks beyond code. The unified training strategy would make LLM-based code generation more accessible to those who have less computation power.

**Score: 8**

**Justification:**

The paper presents a novel and significant contribution to the field of code generation. The concept of "self-expressing" information needs is innovative and addresses a critical limitation of existing RACG methods. The technical implementation is well-designed, and the experimental results demonstrate the effectiveness and efficiency of the proposed approach. The paper’s limitations, such as the limited evaluation scope and model sizes, are reasonable given the computational constraints, and they highlight directions for future research. While not revolutionary, SelfRACG offers a practical and valuable improvement over existing methods, making it a solid contribution to the field and deserving of a score of 8.

- **Score**: 8/10

### **[Cross-Subject Mind Decoding from Inaccurate Representations](http://arxiv.org/abs/2507.19071v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper tackles the problem of cross-subject mind decoding, aiming to reconstruct visual stimuli from fMRI data across different individuals.  It argues that existing methods struggle due to inaccuracies in representation prediction, stemming from both unidirectional mapping limitations and the accumulation of errors in subsequent diffusion model processing.  To address this, the paper introduces a Bidirectional Autoencoder Intertwining (BAI) framework with the following key components:

1.  **Bidirectional Mapping:** Uses two intertwined autoencoders to learn mappings between fMRI voxels and semantic/visual representations.  This allows for both fMRI-to-image and image-to-fMRI translation, improving representation accuracy.

2.  **Subject Bias Modulation Module (SBMM):**  Incorporates statistical modulation to reduce subject-specific biases in fMRI data, making the model more generalizable.

3.  **Semantic Refinement Module (SRM) and Visual Coherence Module (VCM):**  Designed to handle inaccuracies in predicted representations, mitigating error propagation in the diffusion model.  SRM refines semantic embeddings, while VCM integrates visual representations to ensure output fidelity.

The framework is integrated with ControlNet and Stable Diffusion and evaluated on the Natural Scenes Dataset (NSD), demonstrating improved performance compared to state-of-the-art methods, especially in cross-subject scenarios and with limited training data.

**Critical Evaluation:**

**Novelty:** The paper presents several novel contributions:

*   **BAI Framework:** The use of bidirectional autoencoders for fMRI-to-image and image-to-fMRI translation is a strong component.  It enforces cycle consistency and encourages more accurate representations compared to traditional unidirectional methods.

*   **SBMM:** The inclusion of a subject bias modulation module is crucial for cross-subject generalization. While SBMMs and statistical normalization are not new in general, their specific adaptation and application within an autoencoder framework for fMRI data is a worthwhile approach.

*   **SRM and VCM:** These modules directly address the error accumulation problem in diffusion-based mind decoding, which is a significant and often overlooked challenge. By refining semantics and enforcing visual coherence, the paper goes beyond simply mapping fMRI to existing diffusion model inputs.

**Significance:**

*   **Improved Accuracy:** The paper demonstrates significant improvements in both quantitative metrics (PixCorr, SSIM, CLIP score) and qualitative results, indicating a more faithful reconstruction of visual stimuli from fMRI data. The figures showing reconstructions compared to state-of-the-art methods are compelling.

*   **Cross-Subject Generalization:** A key strength of the paper is its focus on cross-subject decoding.  The ability to adapt the model to new subjects with limited data is a substantial advancement, making the framework more practical and scalable.  The ablation studies highlighting the importance of SBMM support this claim.

*   **Addressing Limitations:**  The paper attempts to address a key limitation of other fMRI decoding approaches: The reliance on exact representations that require significant subject-specific data. The error mitigation strategy is therefore well-motivated.

**Weaknesses:**

*   **Computational Complexity:** While the paper emphasizes accuracy and generalizability, the computational cost of the BAI framework (with its multiple autoencoders, MLPs, and transformers) might be a concern. The lack of direct information related to training and inference time could be addressed.

*   **Parameter Tuning:** The paper mentions using balance factors (λs) in the combined loss function but provides limited justification for the specific values chosen. A more detailed analysis of the impact of different λ values would strengthen the results.

*   **Reliance on Pre-trained Models:** Like many current approaches, this work relies on pre-trained diffusion models (Stable Diffusion). While this is practical, it limits the ability to fully control the image generation process and introduces potential biases from the pre-training data.  However, pre-trained models are a practical choice given the complexity of training from scratch.

*   **Limited Discussion of Failure Cases:** The inclusion of failure cases is positive, but the discussion is brief. A more detailed analysis of the causes of these failures (e.g., limitations of SD, specific types of stimuli that are difficult to reconstruct) would be valuable.

*   **Incremental Novelty:** While the components of the BAI framework (autoencoders, transformers, statistical modulation) are not entirely new in isolation, the novel combination and adaptation of these techniques specifically for cross-subject mind decoding with error mitigation makes the overall approach significant.

**Overall:**

The paper presents a well-designed framework for cross-subject mind decoding that addresses significant limitations of existing methods. The novelty lies in the specific combination and adaptation of existing techniques, with a strong focus on error mitigation and cross-subject generalizability.  The experimental results are convincing, demonstrating improved performance on benchmark datasets.

Score: 8.5
I've assigned a high score to this paper because of its strong technical contributions, compelling experimental results, and significant advancements in cross-subject generalization for mind decoding. Its strength is its innovative combination of techniques that address the major limitation of exact representations. A few weaknesses, like computational considerations and limited discussion of failure cases, prevent the score from being higher, but it's still an excellent piece of work.

- **Score**: 8/10

### **[PrompTrend: Continuous Community-Driven Vulnerability Discovery and Assessment for Large Language Models](http://arxiv.org/abs/2507.19185v1)**
- **Summary**: Here's a summary and critical evaluation of the provided research paper:

**Summary:**

The paper introduces PrompTrend, a novel system for continuously monitoring and assessing Large Language Model (LLM) vulnerabilities as they emerge in online communities. It bridges the gap between formal security research and grassroots vulnerability discovery. The system uses intelligent agents to collect adversarial prompts from various platforms, evaluates them using a multi-dimensional scoring framework (PVAF) that considers both technical and social dynamics of vulnerability propagation, and provides real-time threat intelligence. The authors analyze 198 vulnerabilities from January-May 2025, tested on nine commercial models, and reveal key insights, including the dominance of psychological attacks, the importance of platform dynamics, and model-specific vulnerabilities, challenging the assumption that increased capability leads to better security.

**Critical Evaluation:**

**Strengths:**

*   **Novelty:** The paper presents a significant contribution by shifting the focus from static benchmarks and controlled experiments to continuous monitoring of real-world vulnerability discovery in online communities. PrompTrend is one of the first systems to systematically collect, analyze, and assess LLM vulnerabilities in this dynamic setting.
*   **Comprehensive Approach:** The system integrates data collection, vulnerability assessment, and risk scoring into a single framework, providing a more holistic view of LLM security. The PVAF scoring system is innovative in its incorporation of social factors (community adoption, propagation velocity) alongside technical metrics.
*   **Real-world Relevance:** By focusing on community-discovered vulnerabilities, the paper addresses a critical gap in current LLM security research and offers actionable insights for organizations deploying LLMs.
*   **Empirical Validation:** The system is rigorously tested on a significant number of vulnerabilities and commercial LLMs, providing empirical evidence for its effectiveness and the insights it generates. The statistical analyses are sound and appropriately applied.
*   **Clear Presentation:** The paper is well-written, with a clear structure and a logical flow of arguments. The illustrations and tables are helpful in understanding the system architecture, scoring framework, and experimental results.

**Weaknesses:**

*   **Data Scope Limitations:** The study is limited to a specific time period (January-May 2025) and English-language public forums, which might not capture the full spectrum of LLM vulnerabilities.
*   **Black-Box Nature:** The black-box testing approach limits the ability to causally attribute vulnerabilities beyond vendor disclosures, although real-world applicability is increased.
*   **Generalizability to Different LLMs:** The results are primarily focused on the specific models and platforms studied, limiting the generalizability to other LLMs with different architectures or safety mechanisms, although a range of models are incorporated in testing.
*   **Longitudinal Analysis Awaits Future Deployment:** The longitudinal analysis features, which are crucial for understanding the evolution of vulnerabilities, are not fully realized in the present study and depend on future deployments.
*   **Lack of High-Risk Vulnerabilities:** The absence of high-risk vulnerabilities in the dataset might suggest that truly severe threats remain outside public forums, limiting the scope of the PVAF scoring system.

**Significance:**

The paper's significance lies in its ability to bridge the gap between academic security research and the real-world discovery of LLM vulnerabilities. PrompTrend provides a framework for understanding how vulnerabilities emerge, evolve, and propagate in online communities, which can inform the development of more effective security measures. The system's insights into the dominance of psychological attacks and the importance of platform dynamics challenge conventional assumptions about LLM defenses and highlight the need for a more holistic approach to security.

**Justification for Score:**

The paper demonstrates significant novelty and real-world relevance in the domain of LLM security. The development of the PrompTrend system addresses critical gaps in existing approaches. While the data scope and black-box nature of evaluation pose certain limitations, the comprehensiveness of testing and the robust empirical validation solidify the contributions.
Considering both the strengths and weaknesses, the paper has the potential to significantly influence the field by fostering more proactive and context-aware approaches to LLM security.

**Score: 8**

- **Score**: 8/10

### **[Jailbreaking Large Language Diffusion Models: Revealing Hidden Safety Flaws in Diffusion-Based Text Generation](http://arxiv.org/abs/2507.19227v1)**
- **Summary**: Here's a summary and critical evaluation of the provided research paper:

**Summary:**

The paper "Jailbreaking Large Language Diffusion Models: Revealing Hidden Safety Flaws in Diffusion-Based Text Generation" investigates the vulnerability of Large Language Diffusion Models (LLDMs) to jailbreak attacks.  The authors find that existing jailbreak methods designed for autoregressive LLMs are ineffective against LLDMs due to the architectural differences. To address this, they propose a novel jailbreak attack called PArallel Decoding (PAD) tailored to LLDMs. PAD employs Injected Information Filtering and Multi-Point Attention to exploit the parallel denoising process of LLDMs. Experimental results demonstrate that PAD achieves significantly higher attack success rates compared to existing methods, revealing previously unknown safety vulnerabilities in LLDMs. The paper also highlights the faster generation speed of LLDMs under jailbreak conditions, raising concerns about the rapid dissemination of harmful content.  Finally, it analyzes the architectural reasons for the success of PAD and the failure of existing attacks.

**Critical Evaluation:**

**Novelty:** The core novelty lies in recognizing and demonstrating the distinct vulnerabilities of LLDMs compared to traditional autoregressive LLMs. The design of PAD is innovative as it directly targets the parallel denoising architecture, which is a key differentiator for LLDMs. It's good that they provide some explanation of the reason for the difference. However, the individual components of PAD, such as Injected Information Filtering and Multi-Point Attention, draw inspiration from existing jailbreak techniques, so the novelty of the *components* themselves is somewhat incremental. The application of these existing techniques to a new model architecture is what truly adds the novelty.

**Significance:** This paper is significant because it highlights a critical gap in the safety and alignment of LLDMs. Given the increasing popularity and capabilities of LLDMs, understanding and mitigating their vulnerabilities is crucial. Exposing that existing safeguards developed for autoregressive LLMs are inadequate is a valuable contribution. Showing the accelerated generation speed of harmful content with jailbroken LLDMs amplifies the potential negative societal impact. Moreover, their investigation into the architectural reasons behind these vulnerabilities offers important insights for developing more robust LLDMs and defenses in the future. The experiments are thorough and they validate their claims very well. I can't verify any of these claims, but I can verify that their experiments and their conclusions all make sense based on their methodology.

**Strengths:**

*   **Clear Problem Definition:**  The paper clearly defines the problem of jailbreaking LLDMs and why existing solutions are insufficient.
*   **Novel Methodology:** PAD is specifically designed to exploit the unique architectural characteristics of LLDMs.
*   **Comprehensive Evaluation:** The paper includes thorough experiments across multiple LLDMs and attack scenarios.
*   **Architectural Analysis:**  The paper delves into the architectural reasons behind the vulnerabilities, providing valuable insights.
*   **Well-written and organized.** It's easy to read and follow.
*   **Illustrative examples.** The example prompts and the explanation of the generation steps make the paper's points very clearly.

**Weaknesses:**

*   **Incremental Component Novelty:** While PAD is novel overall, its individual components draw heavily from existing techniques.
*   **Limited Defense Strategies:**  The paper primarily focuses on exposing vulnerabilities rather than exploring potential defense mechanisms. This is fine, but it would be even more valuable if the research included at least an initial exploration of defensive techniques.
*   **Real-World Validation:** While the experimental setup is well-designed, it may not fully capture the complexities and nuances of real-world jailbreak attempts and user interactions. It could also explore various prompt inputs or some sort of robustness tests.

**Potential Influence:**

This paper is likely to have a significant impact on the field of AI safety and security. It will prompt researchers to focus more attention on the unique vulnerabilities of LLDMs and to develop new defense strategies specifically tailored for these architectures. The PAD method can serve as a baseline for future research on jailbreaking LLDMs. The architectural analysis will inform the design of more robust and secure LLDMs.

**Score: 8**

**Justification:** The paper presents a well-defined and significant contribution to the field. It is novel in its approach to jailbreaking LLDMs, provides clear evidence of their vulnerability, and offers valuable insights into the architectural factors that contribute to these vulnerabilities. While the novelty of some components of PAD is somewhat incremental, the overall method is innovative and effectively addresses a previously unrecognized problem. The work is thorough and should encourage further research and development of novel defenses. The paper falls short of being exceptional (a 9 or 10) due to the lack of exploration into potential defenses and the reliance on existing techniques as building blocks for the PAD attack, but the analysis of the model's architecture is helpful and the writing style makes it a very easy read. The high likelihood of future research building upon the results further justifies the high score.

- **Score**: 8/10

### **[Mut4All: Fuzzing Compilers via LLM-Synthesized Mutators Learned from Bug Reports](http://arxiv.org/abs/2507.19275v1)**
- **Summary**: Here's a summary and critical evaluation of the "Mut4All: Fuzzing Compilers via LLM-Synthesized Mutators Learned from Bug Reports" paper:

**Summary:**

The paper presents Mut4All, a novel framework for compiler fuzzing that leverages Large Language Models (LLMs) to automatically synthesize mutation operators (mutators).  Mut4All utilizes a multi-agent system, comprising a mutator invention agent, implementation synthesis agent, and refinement agent.  The agents analyze bug reports to identify error-prone language features, generate mutator specifications, implement concrete mutators, and validate/refine them.  The framework is applied to Rust and C++, and the synthesized mutators are used in a fuzzer targeting compilers like rustc, gccrs, GCC, and Clang.  The experiments demonstrate that Mut4All discovers more unique compiler bugs and achieves high code coverage compared to existing fuzzing techniques, including traditional and LLM-based approaches. The authors also perform a thorough analysis of valid and invalid mutators, providing valuable insights into challenges and directions for future improvement.

**Critical Evaluation:**

* **Novelty:** The paper introduces a highly automated approach to mutator generation for compiler fuzzing, which is a significant advancement. The key novelty lies in its end-to-end automation, learning from bug reports, and utilizing LLMs not merely for direct code mutation, but for inventing and refining mutators at a more abstract level.  Existing LLM-based fuzzing approaches are often less automated and rely more on direct code mutation or predefined rulesets.  The use of a multi-agent system to decouple specification, implementation, and refinement is also a novel and well-engineered aspect of the work. However, the idea of using LLMs for code generation and AST manipulation isn't entirely new; the novelty is in how this capability is directed towards mutator *creation* in a compiler fuzzing context, guided by bug reports.

* **Significance:** Compiler bugs can have severe security and reliability implications. Mut4All demonstrates a practical way to uncover such bugs more effectively. The fact that it discovered 54 new, confirmed bugs in production compilers, including some long-standing issues, highlights the significance of its contribution. Moreover, the analysis of the mutator synthesis process provides valuable insights into the capabilities and limitations of LLMs for this task, guiding future research in this area. The cross-language applicability is a strong point, demonstrating its potential for broader adoption. However, it would have been better to show the adaptability to a new, third language or compiler.

* **Strengths:**
    * **High degree of automation:** Minimizes manual effort in mutator design and implementation.
    * **Effective bug discovery:**  Demonstrated ability to uncover new and confirmed bugs in real-world compilers.
    * **Cross-language applicability:** Shows potential for wider adoption across different programming languages and compilers.
    * **Rigorous evaluation:** Compares Mut4All against several state-of-the-art fuzzers and provides a comprehensive analysis of results.
    * **Insightful mutator analysis:** The detailed analysis of valid and invalid mutators provides a valuable understanding of the challenges and limitations of LLM-based synthesis.

* **Weaknesses:**
    * **Reliance on LLM output quality:** The framework is inherently dependent on the capabilities and limitations of the LLM used. While validation and refinement help mitigate this, the quality of the initial synthesis remains crucial.
    * **Limited theoretical grounding:** The paper primarily focuses on the practical implementation and evaluation of the framework. There's less emphasis on theoretical aspects, such as a formal analysis of the mutation space explored by Mut4All.
    * **Limited exploration of seed program diversity:** It depends on initially enhancing the seeds by "fixed-replacement strategy," the limitations of which the authors acknowledge. Exploring more sophisticated techniques for diverse seed enhancement could be beneficial.
    * **No third language tested:** The paper could be made stronger if it showed evidence that Mut4All could adapt to another, perhaps more obscure language, or a different kind of compiler.

* **Potential Influence:** Mut4All has the potential to significantly influence the field of compiler fuzzing by enabling more automated and effective mutator generation. It can inspire further research into using LLMs for various aspects of compiler testing and validation. It might also influence how compiler developers prioritize bug fixes, given the ability to highlight common error patterns revealed by the bug reports.

**Score: 8**

**Rationale:**

Mut4All represents a significant step forward in compiler fuzzing by automating the mutator design and implementation process. It is novel in its use of a multi-agent LLM framework, guided by bug reports, to generate more sophisticated and context-aware mutators than existing approaches. The experimental results demonstrate its effectiveness in uncovering real-world compiler bugs. While the approach has some limitations, the strengths outweigh the weaknesses. The insightful analysis of mutator effectiveness is a valuable contribution to the field. A score of 8 reflects the novelty and practical significance, but also acknowledges areas for further improvement, such as exploring more seed diversification methods, and demonstrating generalizability through adaptation to a new language or compiler.

- **Score**: 8/10

### **[GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](http://arxiv.org/abs/2507.19457v1)**
- **Summary**: Here's a concise summary and a critical evaluation of the provided paper description:

**Summary:**

The paper introduces GEPA (Genetic-Pareto), a novel prompt optimizer for compound AI systems (systems combining LLMs with tools and complex control flow). GEPA uses natural language reflection on system-level trajectories (reasoning chains, tool calls, etc.) to diagnose problems, propose prompt updates, and combine successful lessons from various attempts using a Pareto frontier. This allows GEPA to learn high-level rules from trial and error with significantly fewer rollouts than reinforcement learning methods like GRPO. Experiments across four tasks (multi-hop reasoning, instruction following, privacy-aware delegation, and retrieval-augmented verification) show GEPA outperforms GRPO (using 35x fewer rollouts) and a leading prompt optimizer (MIPROv2). Preliminary results suggest its potential as an inference-time code optimization strategy.

**Critical Evaluation:**

The paper tackles a crucial problem: the sample inefficiency of reinforcement learning when adapting LLMs for downstream tasks, especially in resource-constrained scenarios. The core idea of using natural language reflection on system trajectories to learn and improve prompts is promising and aligns well with the interpretable nature of language. This approach leverages the strong language priors LLMs already possess, making learning more efficient.

**Strengths:**

*   **Novelty:** GEPA introduces a unique approach by combining genetic algorithms with Pareto optimization and natural language reflection.  The idea of learning from the system's own "thoughts" (reasoning chains) is novel.

*   **Sample Efficiency:** The experimental results convincingly demonstrate GEPA's sample efficiency, a significant advantage over RL-based methods that often require extensive rollouts. The gains over GRPO are substantial and well-quantified.  The comparison to MIPROv2 is also compelling, demonstrating that GEPA outperforms a state-of-the-art prompt optimizer.

*   **Practicality:** GEPA is designed for real-world AI workflows, considering limitations in data and budget. This increases its applicability to various domains. The preliminary results for code optimization broaden its potential impact.

*   **Qualitative Insights:** The paper provides a qualitative analysis of GEPA-generated prompts, offering valuable insights into how GEPA learns and adapts. The clear visualization of the optimization trajectory provides additional support.

*   **Architecture Aware**: GEPA is designed to be agnostic to the underlying model, which is a significant advantage as it allows it to be used with both open-source and proprietary LLMs.

**Weaknesses:**

*   **Limited Generalization Theory:** While the empirical results are compelling, a more theoretical understanding of GEPA's generalization capabilities would be beneficial. It would be good to show theoretical lower bounds on sample complexity.

*   **Instruction Optimization Focus:** GEPA currently optimizes instructions only. Incorporating few-shot demonstrations could potentially further improve performance, especially in tasks where demonstrations are beneficial. The paper needs to explain why demonstration-based optimization is not considered.

*   **Limited Exploration of System Aware Merge**: The paper acknowledges that there are issues in applying system aware merge effectively. More theoretical and experimental work needs to be done here.

*   **Scope of Experiment**: In each set-up, the evaluation set is pre-set. This limits the applicability of the claims across new tasks. It would have been better to have continuous validation across an entire corpus.

**Significance:**

GEPA has the potential to significantly impact the field by providing a more efficient and robust method for adapting LLMs to various downstream tasks. The ability to learn from natural language reflection offers a promising path for optimizing complex AI systems in data- or budget-constrained environments.  The early success in code optimization suggests a broader applicability beyond traditional NLP tasks.

**Justification for Score:**

The paper presents a novel and practical approach to prompt optimization with strong empirical support and promising preliminary results. While some limitations exist (lack of theoretical analysis, instruction-optimization focus), the significant gains in sample efficiency and the potential for broader applications warrant a high score.

**Score: 8**

- **Score**: 8/10

## Other Papers
### **[AraTable: Benchmarking LLMs' Reasoning and Understanding of Arabic Tabular Data](http://arxiv.org/abs/2507.18442v1)**
### **[DIFFA: Large Language Diffusion Models Can Listen and Understand](http://arxiv.org/abs/2507.18452v1)**
### **[Automated Code Review Using Large Language Models with Symbolic Reasoning](http://arxiv.org/abs/2507.18476v1)**
### **[Scout: Leveraging Large Language Models for Rapid Digital Evidence Discovery](http://arxiv.org/abs/2507.18478v1)**
### **[How Well Do LLMs Predict Prerequisite Skills? Zero-Shot Comparison to Expert-Defined Concepts](http://arxiv.org/abs/2507.18479v1)**
### **[Not All Features Deserve Attention: Graph-Guided Dependency Learning for Tabular Data Generation with Language Models](http://arxiv.org/abs/2507.18504v1)**
### **[A Deep Dive into Retrieval-Augmented Generation for Code Completion: Experience on WeChat](http://arxiv.org/abs/2507.18515v1)**
### **[The Moral Gap of Large Language Models](http://arxiv.org/abs/2507.18523v1)**
### **[Elucidating the Design Space of Arbitrary-Noise-Based Diffusion Models](http://arxiv.org/abs/2507.18534v1)**
### **[GLiNER2: An Efficient Multi-Task Information Extraction System with Schema-Driven Interface](http://arxiv.org/abs/2507.18546v1)**
### **[VideoMind: An Omni-Modal Video Dataset with Intent Grounding for Deep-Cognitive Video Understanding](http://arxiv.org/abs/2507.18552v1)**
### **[The Geometry of LLM Quantization: GPTQ as Babai's Nearest Plane Algorithm](http://arxiv.org/abs/2507.18553v1)**
### **[HARLF: Hierarchical Reinforcement Learning and Lightweight LLM-Driven Sentiment Integration for Financial Portfolio Optimization](http://arxiv.org/abs/2507.18560v1)**
### **[Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis](http://arxiv.org/abs/2507.18569v1)**
### **[Wide-In, Narrow-Out: Revokable Decoding for Efficient and Effective DLLMs](http://arxiv.org/abs/2507.18578v1)**
### **[DR.EHR: Dense Retrieval for Electronic Health Record with Knowledge Injection and Synthetic Data](http://arxiv.org/abs/2507.18583v1)**
### **[AQuilt: Weaving Logic and Self-Inspection into Low-Cost, High-Relevance Data Synthesis for Specialist LLMs](http://arxiv.org/abs/2507.18584v1)**
### **[Linear Memory SE(2) Invariant Attention](http://arxiv.org/abs/2507.18597v1)**
### **[Demystify Protein Generation with Hierarchical Conditional Diffusion Models](http://arxiv.org/abs/2507.18603v1)**
### **[Explainable Mapper: Charting LLM Embedding Spaces Using Perturbation-Based Explanation and Verification Agents](http://arxiv.org/abs/2507.18607v1)**
### **[TRPrompt: Bootstrapping Query-Aware Prompt Optimization from Textual Rewards](http://arxiv.org/abs/2507.18618v1)**
### **[Captain Cinema: Towards Short Movie Generation](http://arxiv.org/abs/2507.18634v1)**
### **[CatchPhrase: EXPrompt-Guided Encoder Adaptation for Audio-to-Image Generation](http://arxiv.org/abs/2507.18750v1)**
### **[Initial Steps in Integrating Large Reasoning and Action Models for Service Composition](http://arxiv.org/abs/2507.18775v1)**
### **[Evaluating Code-Mixing in LLMs Across 18 Languages](http://arxiv.org/abs/2507.18791v1)**
### **[DxHF: Providing High-Quality Human Feedback for LLM Alignment via Interactive Decomposition](http://arxiv.org/abs/2507.18802v1)**
### **[MemoCoder: Automated Function Synthesis using LLM-Supported Agents](http://arxiv.org/abs/2507.18812v1)**
### **[RealDeal: Enhancing Realism and Details in Brain Image Generation via Image-to-Image Diffusion Models](http://arxiv.org/abs/2507.18830v1)**
### **[Neural Correction Operator: A Reliable and Fast Approach for Electrical Impedance Tomography](http://arxiv.org/abs/2507.18875v1)**
### **[MindFlow+: A Self-Evolving Agent for E-Commerce Customer Service](http://arxiv.org/abs/2507.18884v1)**
### **[IsaMini: Redesigned Isabelle Proof Lanugage for Machine Learning](http://arxiv.org/abs/2507.18885v1)**
### **[SLoW: Select Low-frequency Words! Automatic Dictionary Selection for Translation on Large Language Models](http://arxiv.org/abs/2507.18902v1)**
### **[Large language models provide unsafe answers to patient-posed medical questions](http://arxiv.org/abs/2507.18905v1)**
### **[A Systematic Review of Key Retrieval-Augmented Generation (RAG) Systems: Progress, Gaps, and Future Directions](http://arxiv.org/abs/2507.18910v1)**
### **[Uncovering Cross-Linguistic Disparities in LLMs using Sparse Autoencoders](http://arxiv.org/abs/2507.18918v1)**
### **[MGHFT: Multi-Granularity Hierarchical Fusion Transformer for Cross-Modal Sticker Emotion Recognition](http://arxiv.org/abs/2507.18929v1)**
### **[PDT: Point Distribution Transformation with Diffusion Models](http://arxiv.org/abs/2507.18939v1)**
### **[Adaptive Learning Systems: Personalized Curriculum Design Using LLM-Powered Analytics](http://arxiv.org/abs/2507.18949v1)**
### **[A Toolbox, Not a Hammer -- Multi-TAG: Scaling Math Reasoning with Multi-Tool Aggregation](http://arxiv.org/abs/2507.18973v1)**
### **[AEDR: Training-Free AI-Generated Image Attribution via Autoencoder Double-Reconstruction](http://arxiv.org/abs/2507.18988v1)**
### **[Agent0: Leveraging LLM Agents to Discover Multi-value Features from Text for Enhanced Recommendations](http://arxiv.org/abs/2507.18993v1)**
### **[Enhancing Reward Models for High-quality Image Generation: Beyond Text-Image Alignment](http://arxiv.org/abs/2507.19002v1)**
### **[A diffusion-based generative model for financial time series via geometric Brownian motion](http://arxiv.org/abs/2507.19003v1)**
### **[MindSpeed RL: Distributed Dataflow for Scalable and Efficient RL Training on Ascend NPU Cluster](http://arxiv.org/abs/2507.19017v1)**
### **[A Survey of Multimodal Hallucination Evaluation and Detection](http://arxiv.org/abs/2507.19024v1)**
### **[SESR-Eval: Dataset for Evaluating LLMs in the Title-Abstract Screening of Systematic Reviews](http://arxiv.org/abs/2507.19027v1)**
### **[SelfRACG: Enabling LLMs to Self-Express and Retrieve for Code Generation](http://arxiv.org/abs/2507.19033v1)**
### **[MLLM-based Speech Recognition: When and How is Multimodality Beneficial?](http://arxiv.org/abs/2507.19037v1)**
### **[PGKET: A Photonic Gaussian Kernel Enhanced Transformer](http://arxiv.org/abs/2507.19041v1)**
### **[Cross-Subject Mind Decoding from Inaccurate Representations](http://arxiv.org/abs/2507.19071v1)**
### **[SP-Mamba: Spatial-Perception State Space Model for Unsupervised Medical Anomaly Detection](http://arxiv.org/abs/2507.19076v1)**
### **[Arg-LLaDA: Argument Summarization via Large Language Diffusion Models and Sufficiency-Aware Refinement](http://arxiv.org/abs/2507.19081v1)**
### **[Comparing OCR Pipelines for Folkloristic Text Digitization](http://arxiv.org/abs/2507.19092v1)**
### **[iPLAN: Redefining Indoor Wireless Network Planning Through Large Language Models](http://arxiv.org/abs/2507.19096v1)**
### **[Distilling a Small Utility-Based Passage Selector to Enhance Retrieval-Augmented Generation](http://arxiv.org/abs/2507.19102v1)**
### **[LISA: A Layer-wise Integration and Suppression Approach for Hallucination Mitigation in Multimodal Large Language Models](http://arxiv.org/abs/2507.19110v1)**
### **[Exploring the Use of LLMs for Requirements Specification in an IT Consulting Company](http://arxiv.org/abs/2507.19113v1)**
### **[Automated Code Review Using Large Language Models at Ericsson: An Experience Report](http://arxiv.org/abs/2507.19115v1)**
### **[MixA-Q: Revisiting Activation Sparsity for Vision Transformers from a Mixed-Precision Quantization Perspective](http://arxiv.org/abs/2507.19131v1)**
### **[RealisVSR: Detail-enhanced Diffusion for Real-World 4K Video Super-Resolution](http://arxiv.org/abs/2507.19138v1)**
### **[A3D-MoE: Acceleration of Large Language Models with Mixture of Experts via 3D Heterogeneous Integration](http://arxiv.org/abs/2507.19142v1)**
### **[Solar Photovoltaic Assessment with Large Language Model](http://arxiv.org/abs/2507.19144v1)**
### **[An Empirical Investigation of Gender Stereotype Representation in Large Language Models: The Italian Case](http://arxiv.org/abs/2507.19156v1)**
### **[Patch Pruning Strategy Based on Robust Statistical Measures of Attention Weight Diversity in Vision Transformers](http://arxiv.org/abs/2507.19175v1)**
### **[PrompTrend: Continuous Community-Driven Vulnerability Discovery and Assessment for Large Language Models](http://arxiv.org/abs/2507.19185v1)**
### **[Reconstruct or Generate: Exploring the Spectrum of Generative Modeling for Cardiac MRI](http://arxiv.org/abs/2507.19186v1)**
### **[Can Small-Scale Data Poisoning Exacerbate Dialect-Linked Biases in Large Language Models?](http://arxiv.org/abs/2507.19195v1)**
### **[Towards Multimodal Social Conversations with Robots: Using Vision-Language Models](http://arxiv.org/abs/2507.19196v1)**
### **[Joint Holistic and Lesion Controllable Mammogram Synthesis via Gated Conditional Diffusion Model](http://arxiv.org/abs/2507.19201v1)**
### **[How Much Do Large Language Model Cheat on Evaluation? Benchmarking Overestimation under the One-Time-Pad-Based Framework](http://arxiv.org/abs/2507.19219v1)**
### **[Jailbreaking Large Language Diffusion Models: Revealing Hidden Safety Flaws in Diffusion-Based Text Generation](http://arxiv.org/abs/2507.19227v1)**
### **[Foundation Model-Driven Grasping of Unknown Objects via Center of Gravity Estimation](http://arxiv.org/abs/2507.19242v1)**
### **[DBMS-LLM Integration Strategies in Industrial and Business Applications: Current Status and Future Challenges](http://arxiv.org/abs/2507.19254v1)**
### **[Mut4All: Fuzzing Compilers via LLM-Synthesized Mutators Learned from Bug Reports](http://arxiv.org/abs/2507.19275v1)**
### **[Towards LLM-Enhanced Group Recommender Systems](http://arxiv.org/abs/2507.19283v1)**
### **[PINO: Person-Interaction Noise Optimization for Long-Duration and Customizable Motion Generation of Arbitrary-Sized Groups](http://arxiv.org/abs/2507.19292v1)**
### **[Identifying Fine-grained Forms of Populism in Political Discourse: A Case Study on Donald Trump's Presidential Campaigns](http://arxiv.org/abs/2507.19303v1)**
### **[Injecting External Knowledge into the Reasoning Process Enhances Retrieval-Augmented Generation](http://arxiv.org/abs/2507.19333v1)**
### **[Doubling Your Data in Minutes: Ultra-fast Tabular Data Generation via LLM-Induced Dependency Graphs](http://arxiv.org/abs/2507.19334v1)**
### **[Smooth Reading: Bridging the Gap of Recurrent LLM to Self-Attention LLM on Long-Context Tasks](http://arxiv.org/abs/2507.19353v1)**
### **[EA-ViT: Efficient Adaptation for Elastic Vision Transformer](http://arxiv.org/abs/2507.19360v1)**
### **[SpeechIQ: Speech Intelligence Quotient Across Cognitive Levels in Voice Understanding Large Language Models](http://arxiv.org/abs/2507.19361v1)**
### **[Integrating LLM in Agent-Based Social Simulation: Opportunities and Challenges](http://arxiv.org/abs/2507.19364v1)**
### **[ReCatcher: Towards LLMs Regression Testing for Code Generation](http://arxiv.org/abs/2507.19390v1)**
### **[Running in CIRCLE? A Simple Benchmark for LLM Code Interpreter Security](http://arxiv.org/abs/2507.19399v1)**
### **[Step-3 is Large yet Affordable: Model-system Co-design for Cost-effective Decoding](http://arxiv.org/abs/2507.19427v1)**
### **[GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](http://arxiv.org/abs/2507.19457v1)**
### **[Advancing Event Forecasting through Massive Training of Large Language Models: Challenges, Solutions, and Broader Impacts](http://arxiv.org/abs/2507.19477v1)**
