# The Latest Daily Papers - Date: 2025-07-31
## Highlight Papers
### **[CliCARE: Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records](http://arxiv.org/abs/2507.22533v1)**
- **Summary**: Here's a summary and critical evaluation of the provided paper:

**Summary:**

The paper presents CliCARE, a novel framework designed to improve clinical decision support using Large Language Models (LLMs) on longitudinal cancer Electronic Health Records (EHRs). CliCARE addresses three key challenges: temporal reasoning over long EHRs, mitigating clinical hallucination, and ensuring reliable evaluation. The framework transforms unstructured EHRs into patient-specific Temporal Knowledge Graphs (TKGs), aligning them with normative clinical guideline knowledge graphs. This approach aims to provide oncologists with evidence-grounded decision support through high-fidelity clinical summaries and actionable recommendations.  The framework is validated on both a private Chinese cancer dataset and the public MIMIC-IV dataset, demonstrating superior performance compared to strong baselines, including long-context LLMs and Knowledge Graph-enhanced RAG methods. The clinical validity of the results is supported by a human-validated LLM-as-a-Judge evaluation protocol, showing high correlation with expert oncologist assessments.

**Critical Evaluation:**

**Novelty:** The paper offers several novel contributions:

*   **Framework Integration:** The integrated CliCARE framework is a significant step forward. While individual components such as using TKGs and aligning with guidelines exist in isolation, the combination and orchestration of these techniques in a unified pipeline represent a valuable innovation.

*   **EHR-to-TKG Transformation and Trajectory Alignment:** The specific method of structuring longitudinal EHRs into TKGs and aligning patient trajectories with guideline KGs appears to be a novel implementation. The use of both semantic matching and LLM-based re-ranking in the alignment process is a particularly interesting detail.

*   **Human-Validated LLM-as-a-Judge:** The reliable evaluation methodology using a human-validated LLM-as-a-Judge overcomes key limitations in automated metrics. This is crucial in a safety-critical domain like healthcare.

**Significance:** The paper tackles a vital problem: making LLMs trustworthy and effective in clinical settings. Overcoming the challenges of long-context reasoning, hallucination, and unreliable evaluation is essential for the safe adoption of AI in healthcare.

*   **Impact on Clinical Decision Support:** A successful implementation of the CliCARE framework would have a direct impact on clinical workflows, potentially reducing clinician workloads, improving decision-making, and ultimately improving patient outcomes.

*   **Contribution to LLM Research:** The paper addresses key limitations of LLMs, especially in complex, high-stakes domains. The techniques developed could be generalized and applied to other areas where grounding and reliability are paramount.

*   **Real-World Validation:** The validation using both a private Chinese cancer dataset and a public English dataset (MIMIC-IV) strengthens the generalizability of the findings. The high correlation of the LLM judge with expert assessments significantly increases the trustworthiness and validity of the evaluation.

**Weaknesses:**

*   **Limited Generalizability Details:** While the framework is evaluated on two datasets, more details about the specifics of adapting the framework to each dataset (e.g., data preprocessing differences, guideline variations, etc.) would increase confidence in its real-world deployability.
*   **Scalability and Cost Considerations:** The computational cost of transforming EHRs into TKGs, performing trajectory alignment, and using powerful LLMs like Gemini and GPT-4 needs to be considered for practical deployment. Future work should investigate ways to optimize efficiency.
*   **Ethical Considerations:**  Although the paper refers to the need for safe and reliable AI, it is relatively light on discussing the specific ethical considerations and safeguards associated with deploying LLMs in such a sensitive area, beyond factual accuracy. Detailed analyses of potential biases and their mitigation would further strengthen the work.
*   **Dependency on External Data:** The framework’s reliance on both general and domain-specific LLMs and Knowledge Graphs means its performance is directly affected by the quality and accuracy of these external resources.

**Justification for Score:**

CliCARE is a significant contribution to the field of clinical decision support using LLMs. The paper addresses critical limitations of existing approaches and provides a novel, well-validated framework for grounding LLMs in clinical guidelines. While there are some areas for future improvement (detailed generalizability, scalability, ethical considerations), the strengths of the paper outweigh its weaknesses. The methods used, coupled with strong validation, give significant weight to the claim.

Score: 8

- **Score**: 8/10

### **[Exploiting Synergistic Cognitive Biases to Bypass Safety in LLMs](http://arxiv.org/abs/2507.22564v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces CognitiveAttack, a novel red-teaming framework that leverages cognitive biases, systematic deviations from rational judgment, to bypass safety mechanisms in Large Language Models (LLMs). Unlike previous jailbreaking approaches focused on prompt engineering or algorithmic manipulation, this work explores how combinations of cognitive biases can be exploited to undermine LLM safeguards. CognitiveAttack uses supervised fine-tuning and reinforcement learning to generate prompts that embed optimized bias combinations, effectively bypassing safety protocols while maintaining high attack success rates across diverse LLMs. The results highlight the vulnerability of current LLM defenses to multi-bias interactions, suggesting multi-bias interactions as a powerful yet underexplored attack vector. The research bridges cognitive science and LLM safety, aiming for more robust and human-aligned AI systems.

**Critical Evaluation:**

*Novelty:* The paper's novelty lies in its systematic application of cognitive biases, particularly in combination, to jailbreak LLMs. While prior work has identified cognitive biases in LLMs and explored adversarial prompting techniques, this research explicitly models and optimizes the interaction of multiple biases for enhanced attack effectiveness. This shift from purely algorithmic or linguistic perspectives to incorporating cognitive science principles is a significant step. However, the paper does not define exactly how the chosen cognitive biases are implemented at a low level.

*Significance:* The paper's significance stems from its ability to expose critical, unaddressed vulnerabilities in LLM safety mechanisms. The consistently higher attack success rates achieved by CognitiveAttack compared to state-of-the-art black-box methods like PAP highlight the limitations of current defenses and underscore the need for a deeper understanding of how cognitive biases can be exploited. Further, the result that "multi-bias prompts are more likely to evade defenses while preserving adversarial potency" is significant, and the paper provides statistical backing for this statement. The work also pushes the field towards developing more robust, human-aligned AI systems by suggesting that the cognitive biases themselves be addressed rather than only the prompts. The transferability of these attacks to diverse LLM architectures and alignment strengths strengthens the significance further.

*Strengths:*
    *   The framework provides a systematic way to generate adversarial prompts.
    *   The paper presents compelling empirical evidence, with evaluations performed on many diverse LLMs.
    *   The approach achieves consistently higher attack success rates than baseline methods.
    *   The research bridges cognitive science and LLM safety, offering a fresh perspective.
    *   The paper's identification of synergistic and antagonistic interactions between biases is valuable.
    *   The experiments are well designed and thorough.

*Weaknesses:*
    *   The cognitive biases are implemented in the prompts. While the choice of each bias combination is discussed, how they are manifested is not quantitatively explored. What aspect of the sentence is chosen to represent the biases, and how the various biases interact at a low level.
    *   The study's reliance on GPT-Judge for evaluating harmfulness introduces a potential subjectivity bias. Although GPT-Judge is commonly used, its judgments are not always perfectly consistent with human evaluations.
    *   The focus is primarily on *demonstrating* the attack vector, rather than developing concrete mitigation strategies. While the paper suggests potential mitigation strategies, it does not provide empirical results showing their effectiveness.
    *   The study is still an attack, and although conducted ethically, the information could be used nefariously.

*Potential Influence:* This work has the potential to significantly influence the field of LLM safety by highlighting the importance of cognitive science perspectives. It could inspire new defense mechanisms that are more robust to psychologically informed attacks and encourage a more holistic approach to AI safety. It will further inform future development in red-teaming models.

**Score: 8**

**Rationale:**

The paper demonstrates clear novelty in systematically applying cognitive biases to jailbreak LLMs and exposes a significant vulnerability in existing safety mechanisms. This has important implications for future research and defense strategies. The strength of the empirical evidence, thoroughness of experimentation, and bridge between different fields are further positives. Although it has some limitations in the experimental depth around the low-level implementation of the chosen cognitive biases, and lacks concrete mitigation strategies. I have assigned a score of 8 to highlight its substantial contribution to the field.

- **Score**: 8/10

### **[Efficient Differentially Private Fine-Tuning of LLMs via Reinforcement Learning](http://arxiv.org/abs/2507.22565v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces Reinforcement Learning for Differential Privacy (RLDP), a novel framework for fine-tuning large language models (LLMs) while preserving data privacy. RLDP casts differentially private (DP) optimization as a closed-loop control problem, using deep reinforcement learning (RL) to dynamically adjust per-parameter gradient clipping thresholds and noise levels.  It uses a customized DP optimizer with pairwise clipping applied to Low-Rank Adaptation (LoRA) tensors and an online Soft Actor-Critic (SAC) hyper-policy leveraging training statistics to balance utility and privacy. The paper demonstrates that RLDP achieves significant perplexity reductions and downstream utility gains compared to existing DP-SGD methods, while maintaining strong privacy guarantees and reducing computational costs by accelerating convergence.

**Critical Evaluation:**

* **Novelty:** The core novelty lies in framing DP-SGD optimization as a reinforcement learning problem. While adaptive clipping methods exist, RLDP is the first to use a learned controller to dynamically adjust parameters based on the ongoing training dynamics.  This adaptive, data-driven approach is a significant departure from static or heuristic-based methods. The use of pairwise clipping for LoRA tensors and heteroscedastic noise scaling also represents a valuable technical contribution tailored for efficient private fine-tuning.

* **Significance:** The paper addresses a critical bottleneck in the practical deployment of LLMs – the trade-off between data privacy and model utility. The results demonstrate that RLDP substantially improves this trade-off, enabling fine-tuning with better performance under strict privacy budgets. The accelerated convergence also has practical implications for reducing computational costs and energy consumption. The comprehensive experimental evaluation across various models and privacy budgets strengthens the claims.

* **Strengths:**
    * **Problem Framing:**  Recasting DP optimization as a control problem is an elegant and potentially transformative approach.
    * **Technical Soundness:** The integration of DP optimization with RL is technically well-executed, and the methods are clearly explained.
    * **Comprehensive Evaluation:** The experimental results are impressive, with consistent improvements over strong baselines across multiple models and privacy budgets. Ablation studies provide insights into the controller's behavior and the impact of SAC hyperparameters.
    * **Practical Impact:** The reduced perplexity, faster convergence, and maintained privacy guarantees have practical implications for deploying LLMs in sensitive domains.

* **Weaknesses:**
    * **LoRA Dependence:** The framework is currently tied to LoRA, which limits its applicability when full fine-tuning or other parameter-efficient techniques are required. While the authors acknowledge this, extending the framework to other PEFT methods would be a valuable future direction.
    * **Computational Overhead of RL:**  The RL agent introduces computational overhead, potentially offsetting the accelerated convergence, particularly for very large models or distributed training.
    * **Dataset Specificity:** The evaluation is primarily based on a specific pseudo-clinical dataset. Generalizability to other data modalities and longer sequences requires further investigation.
    * **Proxy Metrics for Privacy:** While canary extraction and membership inference are used, these remain proxy metrics for real-world adversarial attacks and don't substitute for a rigorous, game-theoretic analysis of adaptive attacks.

* **Potential Impact:** This paper has the potential to significantly influence the development of privacy-preserving LLMs. The RL-based approach opens up new avenues for adaptive and data-driven DP optimization. The code release and pre-trained checkpoints will facilitate future research and adoption. However, the limitations regarding LoRA, dataset specificity, and evaluation of adaptive adversaries need to be addressed in future work to realize its full potential.

* **Justification for Score:** Considering the novelty in problem framing, the technical soundness of the approach, the thorough empirical validation, and the potential for practical impact, I would assign a score of 8. While the limitations (LoRA dependence, dataset specificity, and privacy proxies) prevent a higher score, the paper's contributions are substantial and represent a significant advance in the field of privacy-preserving LLMs. The work is not perfect (no work is!), and future work is needed to extend and generalize RLDP further.

Score: 8

- **Score**: 8/10

### **[RePaCA: Leveraging Reasoning Large Language Models for Static Automated Patch Correctness Assessment](http://arxiv.org/abs/2507.22580v1)**
- **Summary**: Here's a summary and critical evaluation of the provided research paper:

**Summary:**

The paper introduces RePaCA, a novel static Automated Patch Correctness Assessment (APCA) technique that uses reasoning-enabled Large Language Models (LLMs). The core idea is to prompt the LLM with buggy and fixed code snippets, guiding it to analyze the code differences, reason about the root cause of the bug, and then classify the patch as either correct or overfitting.  The LLM is fine-tuned using Reinforcement Learning with the Group Relative Policy Optimization (GRPO) algorithm to enhance its reasoning capabilities specifically for the APCA task. Experiments show RePaCA achieves state-of-the-art performance on a standard dataset, demonstrating superior accuracy and generalization capabilities compared to existing static APCA methods. Additionally, the paper emphasizes the enhanced explainability provided by the LLM's reasoning process.

**Critical Evaluation:**

*   **Novelty:** The application of fine-tuned, reasoning-based LLMs to the static APCA problem is a significant step forward. While LLMs have been used in APR and APCA before, the focus on *reasoning* about patch correctness, combined with GRPO fine-tuning, sets this work apart. The paper persuasively argues that existing methods lack the deep understanding of code changes that LLMs can provide. Prior work, such as APPT, leverage LLMs but primarily for encoding code and identifying patterns, rather than explicit reasoning.

*   **Significance:** The improvements in accuracy and generalization demonstrated by RePaCA are practically valuable. Overfitting is a major problem in APR, and a more accurate APCA tool can significantly improve the reliability of automatically generated patches, reducing the burden on developers. The explainability aspect is also significant. Existing APCA tools often act as black boxes, making it difficult to understand *why* a patch is classified as overfitting. RePaCA's reasoning traces offer insights into the model's decision-making process, enabling developers to better understand and trust the tool.

*   **Strengths:**

    *   **Strong Empirical Results:**  The paper presents compelling experimental results demonstrating RePaCA's superiority over existing static APCA techniques on both the small and across datasets tests. The reported improvements in accuracy, precision, recall, F1-score, and AUC are substantial.

    *   **Focus on Explainability:** The paper explicitly addresses the need for explainable APCA, a critical aspect for practical adoption.  The inclusion of reasoning examples provides concrete evidence of the model's ability to generate meaningful justifications for its decisions.

    *   **Rigorous Methodology:** The paper details the prompt design, training architecture, and hyperparameter choices, increasing the reproducibility of the work. The inclusion of negative results related to model selection is also commendable, showing the authors' commitment to a thorough evaluation process.

    *   **Clear Problem Definition:** The paper clearly defines the problem of overfitting in APR and the limitations of existing static APCA approaches.

*   **Weaknesses:**

    *   **Limited Dataset Diversity:** While the experiments demonstrate RePaCA's effectiveness, the datasets used are still relatively limited in size and diversity. Further evaluation on more extensive and varied benchmarks would strengthen the claims of generalization.

    *   **Reasoning Failures:** The paper acknowledges that the LLM can sometimes make reasoning errors, hallucinating problems or solutions. While the overall performance is impressive, addressing these reasoning failures is crucial for improving the reliability of the tool. Although, it also presents an accurate description in its own analysis of the error it has made.

    *   **Computational Cost:** The GRPO fine-tuning process is computationally expensive. While the inference phase is efficient, the high training cost could limit the scalability of the approach. Although, the results obtained validate its costs.

    *   **Reliance on Prompts:** Like all LLM-based approaches, RePaCA's performance is sensitive to the prompt design. While the paper details the prompt design process, further exploration of different prompt strategies could potentially lead to even better results.

*   **Potential Influence:** RePaCA has the potential to significantly influence the field of APR by enabling the creation of more reliable and trustworthy automatic patch generation systems. The focus on reasoning and explainability could also inspire new research directions in APCA, leading to more transparent and user-friendly tools.

**Overall Score:**

Given the novelty of the approach, the strong empirical results, and the emphasis on explainability, but also considering the identified weaknesses, the paper warrants a score of **8**.  The combination of a reasoning-based LLM, GRPO fine-tuning, and a focus on explainability represents a significant advancement in the field of static APCA. While further work is needed to address the limitations, the potential impact of RePaCA on APR is substantial.

Score: 8

- **Score**: 8/10

### **[Metamorphic Testing of Deep Code Models: A Systematic Literature Review](http://arxiv.org/abs/2507.22610v1)**
- **Summary**: Here's a summary and critical evaluation of the provided paper:

**Summary:**

The paper presents a systematic literature review (SLR) on metamorphic testing (MT) applied to deep code models (LLM4Code). It analyzes 45 primary studies published between 2019 and 2024. The review classifies metamorphic transformation types, application techniques, downstream tasks evaluated, models tested, programming languages used, datasets employed, and evaluation metrics. The authors identify current trends, limitations, and research gaps in the application of MT to LLM4Code, ultimately providing a roadmap for future research directions. The paper highlights the dominance of robustness evaluations using specific transformation types (identifier renaming, dead code insertion), particular programming tasks (clone detection, method name prediction), and a reliance on certain datasets (CodeSearchNet, BigCloneBench). It calls for broadening the scope of MT to encompass more quality attributes beyond robustness (e.g., security, privacy), underexplored tasks (e.g., code repair, malware detection), diverse model architectures, programming languages, and incorporating human factors.

**Critical Evaluation:**

*   **Strengths:**

    *   **Timeliness and Relevance:** The SLR addresses a current and rapidly evolving area of research.  The increasing use of LLM4Code necessitates methods for evaluating their robustness, making this review highly relevant.
    *   **Systematic Approach:** The paper follows a well-defined methodology, minimizing bias in study selection and data extraction. This increases the trustworthiness of the review's findings.
    *   **Comprehensive Coverage:** The analysis of 45 papers provides a broad overview of the current landscape. The classification of transformation types, techniques, tasks, models, languages, datasets, and metrics is thorough and insightful.
    *   **Identification of Research Gaps:** The paper identifies several key limitations in the current literature, including the limited scope of quality attributes evaluated, narrow task coverage, and biases in programming languages and datasets used.
    *   **Roadmap for Future Research:** The concrete suggestions for future research directions provide valuable guidance to researchers in the field.

*   **Weaknesses:**

    *   **Limited Corpus Size:** While justified, the corpus of 45 papers is relatively small compared to some SLRs.  This might limit the generalizability of some findings, particularly concerning less-studied areas. There may be additional relevant work, due to different terminologies.
    *   **Focus on Robustness:** While acknowledging MT's broader potential, the review primarily focuses on robustness evaluations.  This restricts the depth of analysis into applications of MT for other quality attributes like security, privacy, or explainability, which are only tangentially mentioned.
    *   **Practical Adoption Coverage:** The paper calls for increasing integration into industrial practices but gives limited detail. A discussion on the barriers to integration into industry and how to overcome them could be added.
    *   **Lack of a complete search string**. The query string should be provided.

*   **Novelty and Significance:**

    *   The paper provides a structured, data-driven overview of how MT is *currently* applied to LLM4Code. This fills a critical gap in the literature, as there was previously no comprehensive synthesis of this area.
    *   The paper's novelty lies in its systematic analysis of the specific transformations, techniques, models, tasks, languages, and evaluation metrics involved in metamorphic testing for LLM4Code. This granular analysis provides deeper insights than previous overviews.
    *   The categorization of transformation types and application techniques is a valuable contribution to the field, as it helps clarify terminology and provides a framework for future research.
    *   The identification of research gaps and the roadmap for future research are highly significant contributions that can guide the development of more robust and reliable LLM4Code tools.

*   **Potential Influence:**

    *   This SLR is likely to become a highly cited resource for researchers working on metamorphic testing for deep code models.
    *   The identification of research gaps and the roadmap for future research can influence the direction of the field, encouraging researchers to address the limitations of current approaches and explore new areas of investigation.
    *   The paper's findings can inform the development of more robust and reliable LLM4Code tools, ultimately leading to their wider adoption in software engineering practices.

**Score: 8.5**

**Rationale:**

The paper makes a significant contribution by providing the first comprehensive and systematic review of metamorphic testing for deep code models. The thorough analysis, identification of research gaps, and roadmap for future research offer valuable guidance to researchers in this rapidly evolving field. The limitations regarding corpus size and focus on robustness are minor and do not detract significantly from the overall value of the review. The likelihood that this paper will influence the direction of future research and contribute to the development of more reliable LLM4Code tools justifies the high score.

- **Score**: 8/10

### **[Hate in Plain Sight: On the Risks of Moderating AI-Generated Hateful Illusions](http://arxiv.org/abs/2507.22617v1)**
- **Summary**: Here is a summary and critical evaluation of the paper "Hate in Plain Sight: On the Risks of Moderating AI-Generated Hateful Illusions."

**Summary:**

The paper explores the potential misuse of AI-generated optical illusions to spread hateful messages. The authors create a dataset of "hateful illusions" using Stable Diffusion and ControlNet, conditioned on hate messages. They then evaluate the performance of existing content moderation models and vision-language models (VLMs) in detecting these hidden hateful messages. The results demonstrate that current moderation techniques are largely ineffective in identifying hateful illusions, revealing a vulnerability in their vision encoders, which tend to focus on surface-level details and overlook the hidden messages. The paper also explores preliminary mitigation strategies, such as image transformations and training-level strategies, to improve the detection of these hateful illusions.

**Critical Evaluation:**

*   **Novelty:** The paper introduces a novel and timely problem: the exploitation of AI-generated optical illusions to disseminate hate. It's one of the first works to systematically explore this specific type of AI misuse. The generation and analysis of a dataset specifically designed for this purpose is a significant contribution.
*   **Significance:** The findings are significant because they highlight a real-world vulnerability in current content moderation systems. As AI-generated content becomes more prevalent, it's crucial to address these types of stealthy dissemination tactics. The analysis of why VLMs fail (focus on surface features, static attention) provides valuable insights for improving future moderation models. The exploration of mitigation strategies, while preliminary, points the way for future research.
*   **Strengths:**
    *   Well-defined problem and clear research questions.
    *   Systematic methodology for generating and annotating the "hateful illusion" dataset.
    *   Comprehensive evaluation of various moderation classifiers and VLMs.
    *   Insightful analysis of the limitations of ViT-based vision encoders.
    *   Exploration of potential mitigation measures, providing directions for future research.
*   **Weaknesses:**
    *   The mitigation strategies are preliminary and require further development.
    *   The dataset, while valuable, is limited in size (1,571 hateful illusions) and diversity (62 hate messages, 30 descriptive prompts). A larger, more varied dataset could further strengthen the findings.
    *   The human annotation process, while using multiple annotators, is still subjective. More rigorous methods for verifying the presence and visibility of hateful messages could enhance the study's reliability.
*   **Potential Influence:**
    *   Raise awareness among content moderation platform providers and AI developers.
    *   Inform the development of more robust content moderation techniques that can detect hidden messages in AI-generated images.
    *   Inspire further research on adversarial attacks using AI-generated media.
    *   Encourage the development of more sophisticated vision encoders that can capture deeper semantic information beyond surface-level features.

**Justification for Score:**

The paper presents a novel problem, provides a systematic analysis of existing moderation methods, and offers valuable insights into the limitations of VLMs and potential mitigation strategies. While the mitigation strategies and dataset have limitations, the core findings are robust and significant for the field of content moderation and AI safety.

Score: 8

- **Score**: 8/10

### **[A Systematic Literature Review on Detecting Software Vulnerabilities with Large Language Models](http://arxiv.org/abs/2507.22659v1)**
- **Summary**: Here's a summary and critical evaluation of the provided paper:

**Summary:**

The paper presents a systematic literature review (SLR) of Large Language Model (LLM)-based software vulnerability detection. Analyzing 227 studies published between January 2020 and June 2025, the authors categorize research based on task formulation, input representation, system architecture, and adaptation techniques. The review also analyzes datasets used, focusing on their characteristics, vulnerability coverage, and diversity. The authors offer a taxonomy of vulnerability detection approaches, identify limitations, and outline potential future research directions to improve transparency, comparability, and reproducibility in the field. Finally, they provide a regularly updated public repository of LLM-based software vulnerability detection studies.

**Critical Evaluation:**

**Strengths:**

*   **Comprehensive Scope:** The inclusion of 227 studies indicates a solid commitment to providing a complete picture of the field, including preprints.
*   **Structured Taxonomy:** The clearly defined taxonomy across multiple dimensions (task, input, architecture, adaptation) is valuable for organizing and understanding the diverse approaches.
*   **Dataset Analysis:** The in-depth analysis of vulnerability datasets, including CWE coverage, balance, and limitations, addresses a critical gap in current research and provides actionable insights.
*   **Clear Identification of Limitations and Future Directions:** The paper does not simply summarize but critically assesses the state of the field, highlighting limitations and offering concrete suggestions for future work.
*   **Practical Resources:**  The public repository of studies promotes collaboration and reproducibility.

**Weaknesses:**

*   **Dependence on Preprints:** The inclusion of many preprints introduces a potential for variability in the findings, given the absence of peer review in a subset of the reviewed works.
*   **Rapid Evolution of the Field:** Given the rapid pace of development in LLMs, some observations and suggestions might become outdated quickly, requiring continuous updating and maintenance of the living repository.
*   **Limited Exploration of Specific Use Cases:** While the taxonomy is comprehensive, the depth of analysis regarding the effectiveness of each approach in specific use cases (e.g., different types of software, different development environments) could be deeper.

**Novelty and Significance:**

The paper offers a strong contribution to the field due to its comprehensiveness and critical assessment. While other reviews exist, this work provides a focused and up-to-date taxonomy specifically for LLM-based software vulnerability *detection* (rather than broader security applications). The dataset analysis fills a significant need, as this area has seen limited structured evaluation. The articulation of future research directions offers clear and actionable pathways for the community to make progress. This positions the paper to have a lasting influence by guiding future research and promoting more rigorous evaluation practices.

**Justification of Score:**

The paper demonstrates significant rigor and depth, moving beyond a simple summary to provide critical insights and practical guidance. Its emphasis on comparability, reproducibility, and dataset analysis sets it apart from other surveys in the domain. The weaknesses, such as reliance on preprints, are common in rapidly evolving fields and do not diminish the overall contribution. The provision of the live repository further enhances its value. Considering the scope, depth, and potential impact, the following score is assigned:

Score: 8

- **Score**: 8/10

### **[CUS-QA: Local-Knowledge-Oriented Open-Ended Question Answering Dataset](http://arxiv.org/abs/2507.22752v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces CUS-QA, a new dataset for open-ended question answering focused on regional knowledge. The dataset covers Czechia, Slovakia, and Ukraine, with questions and answers curated in the local languages (Czech, Slovak, and Ukrainian) and translated into English. The dataset includes both textual and visual questions grounded in Wikipedia. The paper evaluates state-of-the-art Large Language Models (LLMs) on the dataset, finding a significant gap in regional knowledge, cross-lingual inconsistencies, and limited correlation between automated evaluation metrics and human judgments.  The dataset and human evaluations are publicly released to encourage research on regional knowledge in LLMs, cross-lingual generation, and evaluation metrics for open-ended QA.

**Critical Evaluation:**

*   **Strengths:**

    *   **Novelty:** The dataset addresses a gap in existing QA benchmarks by focusing on *regional* knowledge, as opposed to globally-known facts or US-centric knowledge. This is an important step toward evaluating LLMs in more realistic and culturally sensitive scenarios.  The human evaluation component is a significant strength as well, providing valuable ground truth for assessing automatic evaluation metrics.
    *   **Significance:** The findings that LLMs struggle with regional knowledge, exhibit cross-lingual inconsistencies, and that automatic evaluation metrics perform poorly highlights the importance of this dataset. It highlights limitations in current LLM capabilities, which can be highly useful for driving further research.
    *   **Completeness:** The dataset is reasonably sized, spans multiple related languages, and includes both textual and visual modalities, offering a good range of research opportunities. The paper provides strong baselines and a thorough evaluation, including comparisons of various automatic metrics and human judgement.
    *   **Reproducibility:** The public release of the dataset and human evaluation results promotes reproducibility and further research in the field. They also provide access to the annotation tool that the authors used in developing the dataset.
    *   **Thorough Comparison with Other Work:** The "Related Work" section thoroughly discusses existing QA datasets and their limitations and strengths compared to this paper.

*   **Weaknesses:**

    *   **Limited Geographical Scope:** While valuable, the dataset is limited to three countries. Extending the dataset to cover more regions would increase its generalizability and impact.
    *   **Wikipedia Bias:** Grounding the questions in Wikipedia, while practical, introduces a bias towards information readily available on that platform, potentially neglecting knowledge not well-documented online.
    *   **LLMs-as-judges Limitations:** While LLMs can be used as judges, they are still imperfect, and LLM biases can influence results.

*   **Overall Novelty and Significance:**
    The paper makes a significant contribution by introducing a novel dataset tailored for evaluating LLMs on regional knowledge. This is valuable in pushing the field beyond global knowledge benchmarks and into more realistic and culturally nuanced settings. The comprehensive evaluation, including human judgment, provides valuable insights into the limitations of current models and evaluation metrics. While the dataset's scope is limited and the reliance on Wikipedia introduces certain biases, the benefits outweigh the limitations.

**Score: 8**

**Justification:** The paper is novel and addresses a significant gap in QA benchmarks. The findings regarding regional knowledge and evaluation metrics are valuable. The public release makes the dataset a useful resource. However, the limited geographical scope and reliance on Wikipedia prevent it from being a truly exceptional contribution. The score of 8 reflects a strong, valuable contribution with room for improvement in scope and bias mitigation.

- **Score**: 8/10

### **[DO-EM: Density Operator Expectation Maximization](http://arxiv.org/abs/2507.22786v1)**
- **Summary**: Okay, here's a summary and critical evaluation of the paper "DO-EM: Density Operator Expectation Maximization":

**Summary:**

The paper introduces a novel Expectation-Maximization (EM) framework, called Density Operator Expectation Maximization (DO-EM), designed for training latent variable models based on density operators (DOMs).  The core challenge addressed is the lack of a direct quantum analogue to conditional probability, which makes the traditional EM Expectation step difficult.  The authors reformulate the Expectation step as a quantum information projection (QIP) problem and leverage the Petz Recovery Map to provide a solution under specific conditions.  They derive a quantum evidence lower bound (QELBO) and demonstrate that the DO-EM algorithm ensures non-decreasing log-likelihood under mild assumptions. The paper then specializes the DO-EM framework to train Quantum Interleaved Deep Boltzmann Machines (QiDBMs) on classical data, with resources comparable to classical DBMs.  Finally, empirical results on the MNIST dataset show that QiDBMs trained with DO-EM and Contrastive Divergence outperform larger classical DBMs in image generation.

**Critical Evaluation:**

**Novelty:** The paper presents a significant advance in unsupervised quantum machine learning. Developing an EM framework for density operators is non-trivial, and the approach of reframing the E-step as a quantum information projection problem utilizing the Petz recovery map is novel. Specifically,
*   **DO-EM Framework:** The core contribution is the design of a DO-EM algorithm based on quantum information projection, which addresses the lack of conditional probability in quantum settings. This is a theoretically sound method.
*   **Specialization for Classical Data:** The specialization of the DO-EM algorithm for classical data, resulting in CQ-LVMs and specifically, QiDBMs, is a clever approach to leverage the potential of quantum models while working within the constraints of current classical hardware. This also enables the training of these models on the classical data.

**Significance:** The paper addresses a crucial bottleneck in generative quantum machine learning: the scalability of training algorithms for density operator models.

*   **Scalability:** The DO-EM framework addresses the scalability issue by providing a more computationally tractable approach to training DOMs. The ability to train QiDBMs on MNIST-scale data, a task previously infeasible, demonstrates the practical impact of the work.
*   **Empirical Validation:** The empirical results showing QiDBMs outperforming classical DBMs on the MNIST dataset in image generation are compelling and provide evidence for the potential benefits of quantum-inspired models, even on classical hardware.

**Strengths:**

*   **Sound Theoretical Foundation:** The paper is grounded in quantum information theory and provides rigorous proofs for the key results, including the derivation of the QELBO and the conditions for log-likelihood ascent.
*   **Practical Relevance:** The focus on scaling DOMs to real-world data and the design of the QiDBM architecture demonstrate the practical relevance of the work.
*   **Empirical Validation:** The empirical results on the MNIST dataset provide strong evidence for the effectiveness of the DO-EM algorithm and the potential of QiDBMs.

**Weaknesses:**

*   **Condition S Restriction:** The QIP relies on the `Condition S` which may limit the applicability to some datasets and models. Further exploration of conditions that could lead to QIP is required.
*   **Computational Complexity:** The paper does not fully address the computational complexity of the DO-EM algorithm, particularly the QIP step. While it shows that the algorithm can scale to MNIST, a more detailed analysis of the time and space complexity would be valuable.
*   **Limited Model Class:** The empirical evaluation focuses primarily on QiDBMs. While these results are promising, it would be beneficial to demonstrate the effectiveness of the DO-EM algorithm on other types of DOMs.
*   **Limited Baselines:** Comparisons were made with implementations of DBMs but direct comparison with state of the art QBMs would make the significance higher.

**Potential Influence:** The paper has the potential to significantly influence the field of unsupervised quantum machine learning. The DO-EM framework provides a valuable tool for training density operator models and could inspire further research on quantum-inspired generative models and hybrid quantum-classical algorithms.

**Justification of Score:**

The paper presents a novel and theoretically sound EM framework for density operator models, addresses a crucial scalability bottleneck in the field, and provides empirical evidence for the effectiveness of the approach. While the `Condition S` can limit the models, it can be relaxed in future. The DO-EM is also computationally tractable and has a practical relevance with the MNIST dataset.

Score: 8

- **Score**: 8/10

### **[ScreenCoder: Advancing Visual-to-Code Generation for Front-End Automation via Modular Multimodal Agents](http://arxiv.org/abs/2507.22827v1)**
- **Summary**: Here's a summary and evaluation of the provided research paper:

**Summary:**

The paper introduces ScreenCoder, a novel framework for automating the conversion of UI designs into front-end code.  It addresses the limitations of existing text-based approaches that struggle with capturing spatial layout and visual design intent. ScreenCoder proposes a modular multi-agent architecture comprising three stages: grounding (using vision-language models to detect and label UI components), planning (constructing a hierarchical layout based on web engineering principles), and generation (producing HTML/CSS code via adaptive prompts).  The framework also functions as a scalable data engine to generate synthetic image-code pairs for fine-tuning vision-language models, improving their UI understanding and code quality. Experiments demonstrate state-of-the-art performance in layout accuracy, structural coherence, and code correctness.

**Critical Evaluation:**

* **Strengths:**
    * **Modular Architecture:** The decomposition of the UI-to-code problem into distinct grounding, planning, and generation stages is a significant contribution. This modularity improves interpretability, robustness, and the ability to inject domain-specific knowledge. This is a key advantage over end-to-end black-box approaches.
    * **Addressing Multimodality:** Explicitly leveraging visual input (UI screenshots) in addition to text is crucial, mirroring real-world UI design workflows.  This is a major step forward from purely text-based code generation.
    * **Data Engine and Fine-tuning:** The use of the framework to generate large-scale synthetic data for fine-tuning VLMs is a powerful concept. This addresses the scarcity of high-quality image-code datasets and offers a practical path for improving model alignment.  The results of fine-tuning Qwen2.5-VL validate this approach.
    * **Interactive Design Support:** The incorporation of user instructions and adaptive prompts enables interactive design modifications, which is a valuable feature for real-world applications.
    * **Clear Experimental Results:** The paper provides quantitative and qualitative evidence of the framework's effectiveness, demonstrating state-of-the-art performance compared to existing models.  The comprehensive set of evaluation metrics (block match, text similarity, position alignment, color consistency, CLIP similarity) strengthens the validity of the claims.

* **Weaknesses:**
    * **Reliance on Existing VLMs:** The grounding agent relies on existing vision-language models, which are still imperfect. While the paper includes post-processing steps, the overall performance is still limited by the accuracy of the underlying VLM.
    * **Complexity of the Pipeline:** While modularity is a strength, the multi-stage pipeline introduces complexity. Coordinating the different agents and ensuring smooth information flow could be challenging in practice.
    * **Generality to Different Design Domains:** The paper focuses primarily on web UI design. The applicability of the framework to other design domains (e.g., mobile apps, game UIs) needs further investigation. While the authors suggest it can be extended, empirical validation is necessary.
    * **Scalability and Robustness:** The scalability and robustness of the data engine in real-world deployment scenarios (handling noisy inputs, variations in UI styles) is not fully explored.

* **Novelty and Significance:**

The paper exhibits significant novelty in its modular multi-agent architecture for UI-to-code generation. The combination of visual grounding, hierarchical planning, and adaptive prompt-based code synthesis is a unique and innovative approach. The data engine aspect is also a notable contribution, enabling the creation of high-quality training datasets for VLMs. ScreenCoder has the potential to significantly impact the field of front-end automation and make UI development more accessible. The modular design promotes further research into individual components (better grounding, improved planning heuristics, etc).

**Justification for Score:**

The paper demonstrates a clear advancement over existing approaches, offering a more robust, interpretable, and adaptable solution for UI-to-code generation. The modular architecture, the emphasis on multimodality, and the data engine aspect are all significant contributions. While there are some limitations regarding reliance on existing VLMs, and generalizability to other design domains, the paper's strengths outweigh its weaknesses.

Score: 8

- **Score**: 8/10

### **[Repair-R1: Better Test Before Repair](http://arxiv.org/abs/2507.22853v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Repair-R1: Better Test Before Repair":

**Summary:**

The paper introduces Repair-R1, a novel approach to automated program repair (APR) that integrates test case generation into the model's training phase. Unlike traditional APR methods that primarily use test cases for patch validation after repair, Repair-R1 emphasizes generating discriminative test cases *before* repair. These test cases are designed to expose the bug by passing in the correct code but failing in the buggy code. Repair-R1 utilizes reinforcement learning (RL) to co-optimize both test generation and bug repair, allowing the model to better understand the underlying causes of defects.  The authors demonstrate the effectiveness of Repair-R1 using three different backbone models on four benchmark datasets, showing improvements in repair success rate, test generation success rate, and test coverage compared to vanilla models.

**Critical Evaluation:**

*   **Novelty:** The core idea of prioritizing test case generation *before* repair is a significant departure from conventional LLM-based APR approaches. While the idea of leveraging test cases in repair is not completely new, the systematic integration of test generation into the *training* phase, coupled with reinforcement learning for co-optimization, distinguishes this work. The reformulation of the co-optimization problem as an ELBO maximization problem adds a theoretical contribution.
*   **Significance:** The results demonstrate substantial improvements in repair performance across multiple benchmarks, suggesting that Repair-R1 can lead to more effective and robust APR systems.  The study also challenges the "repair first, test later" paradigm, suggesting a new direction for research in LLM-based APR. The improvements on imbalanced datasets, where SFT suffers, are particularly noteworthy, highlighting the generalization potential of the RL-based approach.  Furthermore, the ablation studies shed light on the relative contributions of test generation and repair capabilities.
*   **Strengths:**

    *   **Clear Problem Statement:** The paper clearly articulates the limitations of existing LLM-based APR approaches.
    *   **Novel Approach:** The "test before repair" paradigm is a compelling and innovative concept.
    *   **Rigorous Evaluation:** The experimental setup is comprehensive, with multiple models, datasets, and baselines.  The ablation studies are valuable for understanding the impact of different components.
    *   **Well-written and Organized:** The paper is clearly written and well-structured, making it easy to follow the methodology and results.
    *   **Theoretical Grounding:** The connection to ELBO maximization provides a solid theoretical foundation.

*   **Weaknesses:**

    *   **Computational Cost:** RL-based training can be computationally expensive, which might limit the scalability of Repair-R1 to very large models or datasets.  The paper does not explicitly address the computational cost.
    *   **Rule-Based Reward Modeling:** While the authors justify the use of rule-based rewards, a learned reward model might potentially offer more flexibility and adaptability.
    *   **Generalizability to More Complex Bugs:** The benchmarks used in the paper might not fully represent the complexity of real-world software bugs.  Further evaluation on more complex and diverse datasets would strengthen the claims.

*   **Potential Influence:** Repair-R1 has the potential to influence the development of future APR systems by emphasizing the importance of test case generation and integrating it more closely with the repair process. The approach also presents a foundation for future research for utilizing more advanced and diverse approaches of test generations, and thus potentially improve model efficiency and effectiveness.

**Justification:**

The paper presents a compelling solution to a significant problem in the field of automated program repair. The novelty lies in its unique approach to integrating test generation into the training process, which allows for a more profound understanding of bugs and leads to improved repair performance. The empirical results are convincing and supported by rigorous experiments and ablation studies.

While the paper has some limitations, its strengths outweigh its weaknesses. The approach challenges the current APR paradigm and has the potential to pave the way for more robust and effective automated repair systems. Therefore, the paper makes a significant contribution.

**Score: 8**

- **Score**: 8/10

### **[Automatically discovering heuristics in a complex SAT solver with large language models](http://arxiv.org/abs/2507.22876v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces AutoModSAT, a novel framework that leverages Large Language Models (LLMs) to automatically discover and optimize heuristics within complex SAT solvers. Recognizing the challenges posed by the intricate architecture and large codebases of modern SAT solvers, AutoModSAT addresses three key areas: (1) developing an LLM-friendly modularized solver (ModSAT), (2) implementing automatic prompt optimization for diverse LLM outputs, and (3) designing an efficient search strategy combining presearch candidate pruning with an (1+2)EA evolutionary algorithm.  Experiments demonstrate AutoModSAT's ability to achieve significant performance improvements over baseline and state-of-the-art (SOTA) SAT solvers across various datasets, even surpassing parameter-tuned versions of SOTA solvers. The paper highlights the potential of AI-driven heuristics discovery for mission-critical system optimization.

**Critical Evaluation:**

* **Novelty:** The paper presents a novel approach to SAT solver optimization by directly integrating LLMs into the heuristic discovery process.  While some prior work has explored LLMs for algorithm design, this paper tackles the complexity of a real-world, highly optimized solver, which distinguishes it from simpler algorithm synthesis efforts.  The modularized solver architecture designed specifically for LLM compatibility, along with the unsupervised prompt optimization and presearch strategy, are also noteworthy contributions. This integrated approach distinguishes itself from hyperparameter optimization methods, by actually discovering new algorithms, instead of just tuning parameters.

* **Significance:**  SAT solvers are crucial components in various industrial applications.  The ability to automatically discover effective heuristics can reduce the need for manual tuning by experts, leading to improved performance and efficiency in real-world problem-solving. The experimental results convincingly demonstrate AutoModSAT's potential to surpass existing optimization techniques. The performance increases, especially compared to existing SOTA solvers in their parameter-tuned versions, are impressive. The reported speedups enhance the framework's practical significance.

* **Strengths:**
    *  **Comprehensive Framework:** The paper presents a well-structured and complete framework encompassing solver modularization, prompt engineering, and an efficient search strategy.
    * **Strong Empirical Evaluation:**  Extensive experiments across diverse datasets provide robust evidence of AutoModSAT's effectiveness.  The comparison with baseline and SOTA solvers, including parameter-tuned versions, strengthens the evaluation.
    * **Practical Relevance:**  The demonstrated ability to handle complex problem instances highlights the practical relevance of AutoModSAT.
    * **Generality:** The principles of creating a modular solver and a LLM-friendly codebase are useful insights for other applications.

* **Weaknesses:**
    * **Limited LLM Analysis:** While the paper demonstrates performance improvements, a deeper analysis of the heuristics generated by the LLM (beyond the varBump and restart_function examples) would be valuable. Understanding the reasoning behind the LLM's choices could further advance the field of algorithm design. The description and evaluation of the LM generated code remain superficial.
    * **ModSAT Performance Gap:** Although AutoModSAT surpasses SOTA solvers, the baseline modularized solver, ModSAT, likely has a performance deficit compared to the unmodified SOTA solvers. This begs the question of how much headroom AutoModSAT has, and how much of the gains actually comes from simply bridging that performance gap.
    * **Scalability/Generalizability Questions:** The experiments focus on CNF formulas with variables numbering up to ~12 million, and clauses up to ~50 million. The paper should have considered larger benchmarks, to evaluate the method's ability to scale and generalize. Also, there are no experiments included on problems beyond SAT, which diminishes the evidence for its transferability.

* **Potential Influence:** AutoModSAT has the potential to influence the development of next-generation complex solvers by promoting AI-driven heuristics discovery and automated optimization. It could inspire similar approaches in other domains where manual tuning is a bottleneck.

**Justification for Score:**

The paper offers a significant contribution to automated algorithm design, particularly in the challenging context of SAT solvers. The practical improvements demonstrated, coupled with the innovative framework, justify a high score. However, weaknesses related to LLM insight and ModSAT performance warrant a tempered assessment.

Score: 8

- **Score**: 8/10

## Other Papers
### **[Falcon-H1: A Family of Hybrid-Head Language Models Redefining Efficiency and Performance](http://arxiv.org/abs/2507.22448v1)**
### **[TopoLiDM: Topology-Aware LiDAR Diffusion Models for Interpretable and Realistic LiDAR Point Cloud Generation](http://arxiv.org/abs/2507.22454v1)**
### **[What is an "Abstract Reasoner"? Revisiting Experiments and Arguments about Large Language Models](http://arxiv.org/abs/2507.22457v1)**
### **[IFEvalCode: Controlled Code Generation](http://arxiv.org/abs/2507.22462v1)**
### **[Towards Simulating Social Influence Dynamics with LLM-based Multi-agents](http://arxiv.org/abs/2507.22467v1)**
### **[Visual Language Models as Zero-Shot Deepfake Detectors](http://arxiv.org/abs/2507.22469v1)**
### **[SLM-SQL: An Exploration of Small Language Models for Text-to-SQL](http://arxiv.org/abs/2507.22478v1)**
### **[LoReUn: Data Itself Implicitly Provides Cues to Improve Machine Unlearning](http://arxiv.org/abs/2507.22499v1)**
### **[DACA-Net: A Degradation-Aware Conditional Diffusion Network for Underwater Image Enhancement](http://arxiv.org/abs/2507.22501v1)**
### **[Collaborative Medical Triage under Uncertainty: A Multi-Agent Dynamic Matching Approach](http://arxiv.org/abs/2507.22504v1)**
### **[CliCARE: Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records](http://arxiv.org/abs/2507.22533v1)**
### **[A Benchmark Dataset and Evaluation Framework for Vietnamese Large Language Models in Customer Support](http://arxiv.org/abs/2507.22542v1)**
### **[ControlMed: Adding Reasoning Control to Medical Language Model](http://arxiv.org/abs/2507.22545v1)**
### **[aLLoyM: A large language model for alloy phase diagram prediction](http://arxiv.org/abs/2507.22558v1)**
### **[Exploiting Synergistic Cognitive Biases to Bypass Safety in LLMs](http://arxiv.org/abs/2507.22564v1)**
### **[Efficient Differentially Private Fine-Tuning of LLMs via Reinforcement Learning](http://arxiv.org/abs/2507.22565v1)**
### **[RePaCA: Leveraging Reasoning Large Language Models for Static Automated Patch Correctness Assessment](http://arxiv.org/abs/2507.22580v1)**
### **[Diffusion Models for Influence Maximization on Temporal Networks: A Guide to Make the Best Choice](http://arxiv.org/abs/2507.22589v1)**
### **[BALSAM: A Platform for Benchmarking Arabic Large Language Models](http://arxiv.org/abs/2507.22603v1)**
### **[ShortFT: Diffusion Model Alignment via Shortcut-based Fine-Tuning](http://arxiv.org/abs/2507.22604v1)**
### **[MetaAgent: Automatically Constructing Multi-Agent Systems Based on Finite State Machines](http://arxiv.org/abs/2507.22606v1)**
### **[VL-Cogito: Progressive Curriculum Reinforcement Learning for Advanced Multimodal Reasoning](http://arxiv.org/abs/2507.22607v1)**
### **[Language Arithmetics: Towards Systematic Language Neuron Identification and Manipulation](http://arxiv.org/abs/2507.22608v1)**
### **[Metamorphic Testing of Deep Code Models: A Systematic Literature Review](http://arxiv.org/abs/2507.22610v1)**
### **[Generative Active Learning for Long-tail Trajectory Prediction via Controllable Diffusion Model](http://arxiv.org/abs/2507.22615v1)**
### **[Hate in Plain Sight: On the Risks of Moderating AI-Generated Hateful Illusions](http://arxiv.org/abs/2507.22617v1)**
### **[Enhancing Manufacturing Knowledge Access with LLMs and Context-aware Prompting](http://arxiv.org/abs/2507.22619v1)**
### **[Multilingual Political Views of Large Language Models: Identification and Steering](http://arxiv.org/abs/2507.22623v1)**
### **[LOTS of Fashion! Multi-Conditioning for Image Generation via Sketch-Text Pairing](http://arxiv.org/abs/2507.22627v1)**
### **[trAIce3D: A Prompt-Driven Transformer Based U-Net for Semantic Segmentation of Microglial Cells from Large-Scale 3D Microscopy Images](http://arxiv.org/abs/2507.22635v1)**
### **[A Systematic Literature Review on Detecting Software Vulnerabilities with Large Language Models](http://arxiv.org/abs/2507.22659v1)**
### **[Zero-Shot Image Anomaly Detection Using Generative Foundation Models](http://arxiv.org/abs/2507.22692v1)**
### **[OFCnetLLM: Large Language Model for Network Monitoring and Alertness](http://arxiv.org/abs/2507.22711v1)**
### **[From Sufficiency to Reflection: Reinforcement-Guided Thinking Quality in Retrieval-Augmented Reasoning for LLMs](http://arxiv.org/abs/2507.22716v1)**
### **[Investigating Hallucination in Conversations for Low Resource Languages](http://arxiv.org/abs/2507.22720v1)**
### **[Resource-Efficient Adaptation of Large Language Models for Text Embeddings via Prompt Engineering and Contrastive Fine-tuning](http://arxiv.org/abs/2507.22729v1)**
### **[Next Tokens Denoising for Speech Synthesis](http://arxiv.org/abs/2507.22746v1)**
### **[CUS-QA: Local-Knowledge-Oriented Open-Ended Question Answering Dataset](http://arxiv.org/abs/2507.22752v1)**
### **[Opportunities and Challenges of LLMs in Education: An NLP Perspective](http://arxiv.org/abs/2507.22753v1)**
### **[Empirical Evaluation of Concept Drift in ML-Based Android Malware Detection](http://arxiv.org/abs/2507.22772v1)**
### **[DO-EM: Density Operator Expectation Maximization](http://arxiv.org/abs/2507.22786v1)**
### **[G-Core: A Simple, Scalable and Balanced RLHF Trainer](http://arxiv.org/abs/2507.22789v1)**
### **[The Multi-Agent Fault Localization System Based on Monte Carlo Tree Search Approach](http://arxiv.org/abs/2507.22800v1)**
### **[MoCHA: Advanced Vision-Language Reasoning with MoE Connector and Hierarchical Group Attention](http://arxiv.org/abs/2507.22805v1)**
### **[DepR: Depth Guided Single-view Scene Reconstruction with Instance-level Diffusion](http://arxiv.org/abs/2507.22825v1)**
### **[ScreenCoder: Advancing Visual-to-Code Generation for Front-End Automation via Modular Multimodal Agents](http://arxiv.org/abs/2507.22827v1)**
### **[Repair-R1: Better Test Before Repair](http://arxiv.org/abs/2507.22853v1)**
### **[Synchronization of mean-field models on the circle](http://arxiv.org/abs/2507.22857v1)**
### **[Automatically discovering heuristics in a complex SAT solver with large language models](http://arxiv.org/abs/2507.22876v1)**
### **[RecGPT Technical Report](http://arxiv.org/abs/2507.22879v1)**
### **[AUV-Fusion: Cross-Modal Adversarial Fusion of User Interactions and Visual Perturbations Against VARS](http://arxiv.org/abs/2507.22880v1)**
### **[Where to show Demos in Your Prompt: A Positional Bias of In-Context Learning](http://arxiv.org/abs/2507.22887v1)**
