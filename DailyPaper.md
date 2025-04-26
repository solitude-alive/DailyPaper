# The Latest Daily Papers - Date: 2025-04-26
## Highlight Papers
### **[MOSAIC: A Skill-Centric Algorithmic Framework for Long-Horizon Manipulation Planning](http://arxiv.org/abs/2504.16738v1)**
- **Summary**: Here is a summary and critical evaluation of the paper:

**Summary:**

The paper introduces MOSAIC, a skill-centric algorithmic framework for long-horizon manipulation planning. It addresses challenges in robotics and AI related to planning motions using predefined skills, a method of unifying the systematic exploration of skill combinations with the generalization of generic skills. MOSAIC employs two families of skills: Generators (compute executable trajectories) and Connectors (link trajectories by solving boundary value problems). MOSAIC's key feature is focusing planning on regions where skills are inherently effective, moving away from incrementally discovering skills from predefined start or goal states. Experiments in simulation and real-world robotics manipulation demonstrate its ability to solve complex planning problems using a diverse set of skills.

**Critical Evaluation:**

*Strengths:*

*   **Novelty:** MOSAIC presents a distinct approach to skill-based planning. Prior methods often lean towards TAMP or full-horizon policy learning, which can be limited by symbolic representations or data requirements, respectively. By making skills central to the planning process, MOSAIC offers a valuable alternative.
*   **Generality:** The use of generic skills (pushing, grasping) allows the framework to generalize across unseen tasks, reducing the need for extensive task-specific knowledge. This is a significant advantage.
*   **Adaptability:** MOSAIC incorporates an oracle that directs the planning process, balancing exploration and exploitation. The adaptability to varying problem scenarios is a strength.
*   **Empirical Validation:** The paper supports its claims with experiments across simulated and real-world environments. Success in tasks like plate transport, in scenarios with increased complexity (e.g., clutter, movable objects), strengthens the argument.
* **Clarity:** The paper is well-structured, and the algorithm is clearly explained. The inclusion of theoretical analyses (probabilistic completeness) adds rigor.

*Weaknesses:*

*   **Oracle Dependence:** While the oracle enhances adaptability, its performance remains a crucial factor. The reliance on a statistical oracle, as in the current implementation, could pose limitations. Future work is needed to explore how more sophisticated oracle modules could enhance the system.
*   **World State Complexity:** Scalability to complex world states, especially those involving a large number of movable objects, is a concern. The paper acknowledges this as a limitation, but the computational cost related to more movable objects remains a relevant weakness.
*   **Stochasticity:** MOSAIC addresses deterministic simulations but admits that execution of these trajectories on the robots might be subject to stochasticity, which can introduce further failures.

*Significance:**

MOSAIC addresses critical challenges in long-horizon robotic manipulation. The ability to compose generic skills to solve complex tasks without extensive task-specific knowledge would have a significant impact on various areas of robotics: automated assembly, cleaning, warehouse operations, and assistive robotics. The move away from relying heavily on symbolic planning or full-horizon policy learning provides an essential and viable approach. It also helps in integrating the best qualities of both motion planning algorithms and learning techniques. The exploration of the skill-trajectory space, and how an oracle dynamically navigates the exploration process presents an interesting alternative.

**Justification for Score:**

MOSAIC offers significant advantages. It strikes a novel and useful balance between model-based planning and learning techniques. I am assigning the paper a score of 8. The paper provides strong contributions but falls short of a 9 or 10 due to the need for a more sophisticated world-state management for the complex scenarios and addressing stochasticity for robotic execution.

**Score: 8**

- **Score**: 8/10

### **[Simple Graph Contrastive Learning via Fractional-order Neural Diffusion Networks](http://arxiv.org/abs/2504.16748v2)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces a novel augmentation-free graph contrastive learning (GCL) framework called FD-GCL.  It leverages fractional-order neural diffusion networks (FDEs) to generate distinct views of node features for contrastive learning. Each FDE is governed by an order parameter, *α*, which controls the balance between local and global information captured in the generated views. By varying *α*, the framework can produce diverse views without relying on traditional data augmentations or negative samples. The authors provide a theoretical analysis based on graph signal processing (GSP) and demonstrate the effectiveness of FD-GCL across various homophilic and heterophilic datasets, achieving state-of-the-art performance. The method avoids dimension collapse by proposing a regularization term based on the correlation between the principal components of feature views.

**Critical Evaluation:**

*   **Novelty:** The core idea of using fractional-order diffusion to create different views for contrastive learning is novel.  While graph diffusion and contrastive learning are established areas, the combination with fractional calculus provides a unique mechanism for controlling the information captured in each view.  The theoretical grounding in GSP is a strength, providing insights into how the *α* parameter affects feature representation.  The regularization term to avoid dimension collapse is also a valuable contribution. Previous work using Neural ODEs is related, but the key difference lies in the use of fractional calculus to control the memory or the level of global/local information flow.

*   **Significance:** The paper addresses a significant challenge in GCL: how to create effective contrasting views without relying on complex data augmentations, which can be data-dependent and computationally expensive. The ability to handle both homophilic and heterophilic graphs is also crucial, as many existing GCL methods struggle with heterophily. The results demonstrate substantial improvements over existing benchmarks, indicating the potential of FD-GCL to advance the field. The elimination of negative samples is a significant advantage.

*   **Strengths:**
    *   **Strong Theoretical Foundation:** The use of GSP to analyze the effect of the order parameter is a major strength.
    *   **Empirical Validation:** Extensive experiments on a diverse set of datasets demonstrate the effectiveness of the method.
    *   **Clear Presentation:** The paper is well-written and clearly explains the concepts and methodology.
    *   **Simplicity:** The augmentation-free approach is simpler than many existing methods, which rely on complex data transformations.
    *   **Avoidance of Negative Sampling:** The modification to the loss function that avoids the usage of negative sampling is also a significant strength.

*   **Weaknesses:**
    *   **Hyperparameter Tuning:** The need to manually tune the order parameters *α1* and *α2* could limit scalability to very large graphs. While a grid search is used, an adaptive strategy could be explored in future work. The authors mention this limitation in the limitations section, so it seems well known.
    *   **Computational Complexity:** Despite the simplicity of the approach, the complexity of solving the FDE could be a bottleneck for very large graphs. Some discussion is provided.

*   **Potential Influence:** FD-GCL has the potential to influence the development of future GCL methods by providing a more principled and efficient way to generate contrasting views. The framework could also be extended to other graph learning tasks, such as graph classification and link prediction.  It provides a strong foundation for further research into the use of fractional calculus in graph neural networks.

*   **Critiques:** It would be more interesting if some of the later analysis of CDE and GREAD was present. It is also hard to appreciate the difficulty in selecting alpha1, and alpha2 for various types of graphs without having an intuition for graph dynamics.

**Rigorous Rationale for Score:**

The paper makes a significant contribution to the field of graph contrastive learning by introducing a novel and effective augmentation-free framework. The use of fractional-order neural diffusion networks provides a unique mechanism for generating diverse views, and the theoretical analysis offers valuable insights into the behavior of the model. The empirical results demonstrate substantial improvements over existing benchmarks, indicating the potential of FD-GCL to advance the field. While the need for hyperparameter tuning and the computational complexity of solving the FDE are limitations, they do not detract significantly from the overall contribution. Furthermore, the theoretical analysis, empirical validation, and clarity of presentation solidify its merit.

Score: 8

- **Score**: 8/10

### **[Lightweight Latent Verifiers for Efficient Meta-Generation Strategies](http://arxiv.org/abs/2504.16760v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Lightweight Latent Verifiers for Efficient Meta-Generation Strategies":

**Summary:**

The paper introduces "LiLaVe" (Lightweight Latent Verifier), a novel method for efficiently verifying the correctness of outputs generated by large language models (LLMs) in reasoning tasks. Unlike traditional LLM-based verifiers, which are often computationally expensive, LiLaVe extracts correctness signals from the *hidden states* of the *base* LLM itself during the generation process. This approach dramatically reduces the computational cost of verification. The authors demonstrate the practicality of LiLaVe by integrating it with existing meta-generation strategies like best-of-n and self-consistency. Furthermore, they propose new LiLaVe-based approaches like conditional self-correction and conditional majority voting, which improve both accuracy and efficiency for smaller LLMs on reasoning tasks.

**Critical Evaluation:**

*   **Novelty:** The key novelty lies in extracting the verification signal directly from the hidden states of the base LLM, rather than relying on a separate, potentially large and costly, LLM-based verifier. The concept of using latent information isn't entirely new (probing, hallucination detection), but the specific application to reasoning verification with a focus on computational *efficiency* is a significant contribution. The conditional generation strategies (conditional majority voting, conditional self-correction) leveraging LiLaVe scores are also novel and practically important.

*   **Significance:** The significance stems from the potential to make sophisticated meta-generation techniques more accessible and efficient, especially for resource-constrained environments or applications using smaller LLMs. The ability to extract reasonably accurate correctness signals without incurring the full cost of a separate LLM-verifier has broad implications for the scalability of reasoning tasks. The empirical results, demonstrating improved accuracy and efficiency on standard mathematical benchmarks, bolster the practical value of the approach.
*   **Strengths:**
    *   Clear and well-motivated problem statement: The computational cost of LLM-based verifiers is a significant bottleneck.
    *   Technically sound approach: LiLaVe is clearly explained and the implementation details are sufficient. The experimental methodology is rigorous, with ablations and comparisons to relevant baselines.
    *   Strong empirical results: LiLaVe consistently outperforms simpler baselines and approaches the performance of much larger verifiers. The new LiLaVe-based meta-generation strategies demonstrate both accuracy and efficiency gains.
    *   Addresses a practical challenge: The work provides an efficient way to leverage latent representations for generation tasks.
*   **Weaknesses:**
    *   Relies on a base LLM: LiLaVe is inherently tied to the architecture and capabilities of the base LLM. Performance might vary significantly with different base models. While some generalization tests were performed, the results are limited.
    *   Potential for false positives: The authors acknowledge the possibility of false positives (responses appearing correct to LiLaVe, but with flawed reasoning). The impact of this could be explored further.
    *   Limited scope of benchmarks: The primary focus is on mathematical QA. While these tasks are important, evaluating LiLaVe on other reasoning-intensive domains (e.g., commonsense reasoning, logical inference) would strengthen the claims of general applicability.
    *   Hyperparameter Tuning: LiLaVe still has parameters that need to be tuned, (such as number of samples and score threshold) which can be data and model dependent.
    *   Limited Discussion of Training Data: How the training dataset for LiLaVe is curated could be a point of discussion since using biased/incorrect training data can lead to incorrect verification.
*   **Potential Influence:** The paper is likely to influence research in several directions:
    *   Encouraging further exploration of latent information in LLMs for verification and control.
    *   Developing more efficient meta-generation techniques.
    *   Making advanced reasoning capabilities more accessible on resource-constrained platforms.
    *   Inspiration for new types of verifiers and reward models with a focus on efficiency.

**Score:** 8/10

**Justification:** The paper presents a novel and practically significant approach to LLM verification that has strong empirical support. The focus on efficiency is crucial for wider adoption of advanced reasoning techniques. While there are some limitations related to base model dependence and benchmark scope, the paper is a valuable contribution to the field and is likely to stimulate further research in this area. The introduction of LiLaVe and its application to conditional generation strategies represent a clear advance over existing techniques, making it worthy of a high score. The rigorous experimental evaluations performed adds to the value of the research and makes it easily reproducible.

- **Score**: 8/10

### **[Decoupled Global-Local Alignment for Improving Compositional Understanding](http://arxiv.org/abs/2504.16801v1)**
- **Summary**: Here is a summary and critical evaluation of the paper:

**Summary:**

The paper introduces a Decoupled Global-Local Alignment (DeGLA) framework to improve compositional understanding in Contrastive Language-Image Pre-training (CLIP) models. It addresses the limitation of global contrastive learning, which inadequately captures compositional concepts. DeGLA incorporates a self-distillation mechanism in the global alignment process to preserve general capabilities and mitigates catastrophic forgetting. It also uses high-quality negative captions generated by Large Language Models (LLMs) and introduces Image-Grounded Contrast (IGC) and Text-Grounded Contrast (TGC) losses to enhance vision-language compositionality. Experimental results demonstrate improved performance in both compositional understanding and general classification tasks.

**Critical Evaluation:**

*   **Novelty:** The paper presents several novel components. The main strength of the work lies in its comprehensive approach towards simultaneously improving both general and compositional understanding in CLIP models. This is a common trade-off, and the authors address it head-on. The key novel elements are:
    *   The self-distillation mechanism to maintain general capabilities.
    *   The use of LLMs to generate high-quality negative captions, which are tailored for compositional understanding and not generic hard negatives.
    *   The IGC and TGC losses for local alignment that specifically target improving compositionality.
    *   The comprehensive combination of these components in a single framework DeGLA.

    While some of these components have been used in isolation, the combination and application to the specific problem of compositional understanding in CLIP models is novel.

*   **Significance:** The results demonstrate state-of-the-art performance on compositional benchmarks (VALSE, SugarCrepe, ARO) and also preserve or even improve performance on general classification tasks. This is significant because previous methods often sacrificed general capabilities for improved compositionality. The paper's results suggest a promising direction for future research in improving vision-language models. A practical significance lies in the better generalization and robustness of fine-tuned CLIP models, relevant for downstream tasks that rely on fine-grained visual and linguistic understanding.

*   **Strengths:**
    *   Comprehensive approach that addresses both general and compositional understanding.
    *   Demonstrated state-of-the-art results on relevant benchmarks.
    *   Well-designed experiments with thorough ablation studies.
    *   Clear and well-written paper.
    *   Demonstration of mitigating the catastrophic forgetting that frequently accompanies fine-tuning CLIP models.

*   **Weaknesses:**
    *   The negative caption generation pipeline relies on LLMs, which can be computationally expensive. While in-context learning helps, the process could still be a bottleneck.
    *   The compositional understanding improvement, while significant, is still relatively modest compared to CLIP itself in some respects, suggesting there is room for further development.
    *   While the paper offers a good analysis, it could include more in-depth qualitative examples to illustrate the specific scenarios where DeGLA excels compared to baseline methods.
    * The ARO experiments fall short of the other methods. In these datasets, DeGLA underperforms Structure CLIP and CE-CLIP indicating there is an additional dimension of information needed that the method does not address.

**Justification for Score:**

The paper addresses an important problem with a comprehensive and novel approach. While individual components like knowledge distillation or using LLMs for data generation are not entirely new, the combination and application to improving compositional understanding in CLIP models is a significant contribution. The experimental results are convincing, and the ablation studies provide valuable insights. The paper demonstrates an effective mitigation of catastrophic forgetting while also improving compositionality. Considering these strengths and the relatively minor weaknesses, the paper deserves a solid score.

**Score: 8**

- **Score**: 8/10

### **[Process Reward Models That Think](http://arxiv.org/abs/2504.16828v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Process Reward Models That Think":

**Summary:**

The paper addresses the challenge of training process reward models (PRMs) for step-by-step verification, which typically requires expensive step-level supervision. The authors propose THINKPRM, a long chain-of-thought (CoT) verifier that is fine-tuned on significantly fewer process labels than discriminative PRMs.  THINKPRM leverages the reasoning abilities of large language models to generate verification CoTs, evaluating each step in a solution. The authors demonstrate that THINKPRM outperforms LLM-as-a-Judge and discriminative verifiers, even when using only a fraction of the process labels required by the latter.  The approach is tested on ProcessBench, MATH-500, and AIME '24, showcasing its effectiveness in both in-domain and out-of-domain settings.  Furthermore, the paper shows that THINKPRM scales verification compute more efficiently than LLM-as-a-Judge.

**Critical Evaluation:**

**Novelty:**

*   The core novelty lies in demonstrating that a *generative* PRM, trained with *minimal* explicit supervision, can outperform discriminative PRMs trained with much more data. This directly challenges the conventional wisdom in the field that more labels are better.

*   The idea of leveraging long CoT for verification is not entirely new (LLM-as-a-Judge approaches exist), but the authors demonstrate that this approach requires a certain amount of customization by fine-tuning a reasoning model to effectively handle the verification process, significantly exceeding the naive application of off-the-shelf LLMs.

*   Repurposing open-weight large reasoning models to produce interpretable verifiers is a nice contribution.

**Significance:**

*   If the findings hold up to further scrutiny, they have significant implications for the field of AI reasoning. The prospect of training effective PRMs with dramatically less labeled data reduces the cost and complexity of scaling up test-time compute, which has been a major bottleneck.

*   The paper demonstrates that CoT is a great way to not only generate a solution, but also verify the solution.
    The interpretability of THINKPRM is a major advantage, allowing for deeper insights into the reasoning process and potential error modes.

**Strengths:**

*   Strong empirical evaluation: The paper presents a comprehensive set of experiments across multiple benchmarks and diverse scenarios (best-of-N, guided search, in-domain, out-of-domain), demonstrating the robustness of the approach.
*   Clear problem formulation and solution: The paper articulates the challenge clearly and proposes a well-defined solution with a simple recipe to scale verifiers.
*   Detailed analysis: The paper includes valuable analysis of the limitations of LLM-as-a-Judge and the impact of different training strategies.

**Weaknesses:**

*   Reliance on Synthetic Data: While the paper filters synthetic data generated from an LLM, there may be some bias, which could affect the final result.
*   Limited discussion of limitations:
    For one, generating a verification chains-of-thought introduces additional computational overhead compared to discriminative PRMs. However, the performance gains offered by generative PRMs compared to the baselines justifies this extra cost.
    For another, overconfidence is a known issue in LLMs and, in the case of PRMs, it can cause the predicted PRM scores to cluster near extremes:
    close to either 0 or 1. One reason is that we are using probabilities of certain tokens such as "yes" or
    "no", which by nature will be either very high or very low.
*   Scalability to More Complex Domains: The empirical evaluation is largely focused on mathematical and logical reasoning tasks, it is unclear if THINKPRM's advantages over discriminate and generative PRMs would continue to hold within larger and more complex domains.
*   Limited Reproducibility: While the paper mentions the intent to release code, data, and models, their availability is still pending. Without access to these resources, independent verification of the results is not possible.

**Justification for Score:**

I am assigning a score of **8** to this paper.

*   The paper makes a significant contribution by demonstrating that effective PRMs can be trained with minimal supervision, challenging the conventional approach. The proposed method has practical implications for scaling reasoning systems. The experimental results support their claims, but further scrutiny and reproducibility are needed.
*   While the approach builds on existing ideas, such as chain-of-thought reasoning and the concept of LLM-as-a-Judge, the authors are able to create a novel approach that outperforms existing techniques in the field. Furthermore, the scalability of this technique is unclear.

Score: 8

- **Score**: 8/10

### **[Emo Pillars: Knowledge Distillation to Support Fine-Grained Context-Aware and Context-Less Emotion Classification](http://arxiv.org/abs/2504.16856v1)**
- **Summary**: Here's a summary and critical evaluation of the paper, focusing on the requested criteria:

**Summary:**

The paper "Emo Pillars: Knowledge Distillation to Support Fine-Grained Context-Aware and Context-Less Emotion Classification" addresses the limitations of existing sentiment analysis datasets, which often lack contextual information and fine-grained emotion categories.  The authors propose a novel LLM-based data synthesis pipeline to generate a large dataset (100K contextual and 300K context-less examples) with diverse semantic content across 28 emotion classes, using Mistral-7b as the LLM engine. They achieve this by grounding generation in narrative corpora, evoking diverse perspectives, and generating multiple utterances simultaneously. This data is then used to fine-tune smaller BERT-type models, resulting in "Emo Pillars" models that achieve SOTA performance on several emotion classification tasks (GoEmotions, ISEAR, IEMOCAP), especially in context-aware scenarios. The authors conduct thorough analyses including statistical analysis, human evaluation, and demonstrate the success of diversification and personalization strategies.

**Critical Evaluation:**

**Novelty and Significance:**

*   **Strengths:**
    *   **Novel Data Synthesis Pipeline:** The combination of grounding in narrative contexts, varying character perspectives, and generating multiple outputs, combined with a unique method for data synthesis is a significant contribution, particularly given the challenges of creating diverse and contextually rich emotional datasets.
    *   **Addressing LLM Limitations:** The paper tackles a practical problem: the high resource demands and over-prediction tendencies of large LLMs for emotion classification.  Distilling knowledge into smaller, more efficient models has clear value.
    *   **Strong Empirical Results:** Achieving SOTA performance on several standard emotion classification datasets validates the effectiveness of the generated data and the fine-tuning approach.
    *   **Publicly Available Resources:** Openly releasing the code, dataset, and models is crucial for reproducibility and further research in the community.
    *   **Thorough Analysis:** The paper demonstrates a rigorous approach, encompassing statistical validation, human evaluation (including assessments of neutrality and context relevance), and topic diversity analysis. The error analyses are particularly insightful.

*   **Weaknesses:**
    *   **Single LLM Reliance:** Although the approach of using LLMs to generate data is novel, a notable limitation is its reliance on a single LLM (Mistral-7b). While tests with GPT-3.5 were performed at the prompting stage, it would be better to see if other LLMs could provide diverse data (e.g. if the LLM has a certain bias, this could effect the generated data). In the 'Limitations' section the authors state that the reliance on only a single LLM allows for easier identification of potential issues with the outputs, this is a valid point.
    *   **Subjectivity and Label Noise:** Emotion is inherently subjective, and even with the authors' careful methods, there is likely to be label noise in the generated dataset. The human evaluation, while insightful, highlights discrepancies in perceived neutrality and indicates issues with some emotion assignments stemming from the mapping process.
    *   **Limited Real-World Validation:** While the results on established datasets are impressive, the use case of applying the models to YouTube comments is presented somewhat superficially. More detailed analysis of how the models perform in a real-world setting (e.g., deployment, A/B testing, user feedback) would strengthen the practical relevance claim.
    *   **Imbalance Mitigation Needed:** The authors note that only a few emotions have small representation, so mitigation of class imbalance is required (similar to goEmotions). This is an area that needs improvement.
    *   **Generalizability Considerations** The model relies on English data (often English-centered or biased towards dominant cultural factors within a language). This can affect cultural background within a language or across other languages, further consideration is needed.

**Overall Significance:**

The paper presents a valuable contribution to the field of emotion classification by addressing the data scarcity problem and providing a pathway to build more efficient and context-aware models. The detailed analysis and publicly available resources are assets to the community.

**Justification of Score:**

The novelty and significance of the Emo Pillars data pipeline and the achieved performance on multiple datasets (especially context-aware ones) merit a high score. While limitations related to label noise, LLM dependency, real-world validation, and the reliance on English are important to acknowledge, the authors provide a novel synthesis pipeline that delivers considerable added value for the AI community.

**Score: 8/10**
- **Score**: 8/10

### **[Planning with Diffusion Models for Target-Oriented Dialogue Systems](http://arxiv.org/abs/2504.16858v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Planning with Diffusion Models for Target-Oriented Dialogue Systems":

**Summary:**

The paper introduces DiffTOD, a novel framework for dialogue planning in target-oriented dialogue (TOD) systems.  Unlike traditional methods that generate dialogue plans sequentially (step-by-step), DiffTOD leverages diffusion models to enable *non-sequential* dialogue planning. It formulates dialogue planning as a trajectory generation problem conditioned on the dialogue history and the desired target. DiffTOD uses a masked diffusion language model to estimate the likelihood of a dialogue trajectory and introduces three different guidance mechanisms tailored to different types of TOD targets: word-level, semantic-level, and search-based guidance.  Experiments across three diverse TOD settings (negotiation, recommendation, and chitchat) demonstrate that DiffTOD outperforms baselines, showing improved target achievement and flexibility.  The key idea is that by planning the entire dialogue trajectory at once, DiffTOD can perform non-myopic lookahead exploration and optimize action strategies over a longer horizon.

**Critical Evaluation:**

*   **Novelty:** The core novelty of the paper lies in the application of diffusion models to non-sequential dialogue planning. While diffusion models have been used for text generation, their use in *planning* the entire dialogue trajectory at once is a significant departure from traditional sequential approaches. The three guidance mechanisms (word, semantic, and search) add another layer of novelty, tailored to different TOD target types. This framework appears to be the first to exploit diffusion models for end-to-end dialogue planning with explicit target guidance. The conceptual link between inpainting with diffusion models and dialogue planning trajectory reconstruction is insightful.
*   **Significance:** TOD systems that can proactively guide conversations are critical for real-world applications. This paper's approach directly addresses the limitations of LLMs being passive followers of instructions. The claimed benefits of DiffTOD such as improved target achievement, global consistency, and flexibility are important for the robustness and effectiveness of dialogue systems. The extensive experimental results, covering multiple TOD settings, provide strong evidence for the significance of the proposed method. The framework has the potential to impact the design of future LLM-based dialogue agents, pushing them towards more strategic and goal-oriented behavior.
*   **Strengths:**
    *   **Clear problem definition:** The paper clearly identifies the limitations of sequential planning in TOD and motivates the need for non-sequential approaches.
    *   **Well-defined methodology:** The DiffTOD framework is well-defined, with a clear explanation of the diffusion model integration and the different guidance mechanisms.
    *   **Comprehensive experiments:** The experiments are comprehensive, covering three diverse TOD settings and comparing against strong baselines.
    *   **Strong empirical results:** The results consistently show that DiffTOD outperforms baselines in terms of target achievement and flexibility.
    *   **Insightful discussion:** The paper provides a valuable discussion of the properties of DiffTOD (non-sequential planning, flexible guidance, tackling sparse rewards, modeling long-range dependencies) and how they contribute to the overall performance.
*   **Weaknesses:**
    *   **Computational cost:** As with most diffusion models, the inference cost can be relatively high compared to autoregressive methods. While the paper mentions potential solutions like acceleration techniques, a more detailed analysis of the practical cost implications would be beneficial. The "Inference Cost" section in the appendix provides some insight but should be expanded and directly tied to real-world deployment concerns.
    *   **Dependence on LLM Quality:** The semantic-level guidance mechanism, for example, relies on the ability of a powerful LLM (GPT-4o in this case) to generate good paraphrased versions of the target state. The performance of this component is therefore directly tied to the quality of the underlying LLM, and performance would likely degrade with smaller LLMs or open source LLMs.
    *   **Scalability to very complex dialogues:** While the paper covers different TOD settings, it is unclear how DiffTOD would scale to dialogues with extremely complex goals or longer interaction histories. There is a limit to how much information can be effectively encoded in the dialogue trajectory, even with a diffusion language model.
    *   **Lack of human evaluation detail:** In "H Human Evaluation Results on the CraigslistBargain Dataset" section, it mentions, "We recruit 5 independent volunteer annotators" - not much other details were provided.
*   **Potential Influence:** The work provides a compelling alternative to sequential planning in TOD.  The core idea of using diffusion models for trajectory generation and incorporating different guidance strategies has the potential to influence future research in dialogue planning and other related areas, such as reinforcement learning for dialogue policy optimization. The framework's modular design makes it relatively easy to extend or adapt to different TOD settings or tasks.

**Score:** 8/10

**Justification:** The paper presents a novel and significant contribution to the field of target-oriented dialogue systems. The application of diffusion models for non-sequential dialogue planning is a notable innovation, and the experimental results provide strong evidence for its effectiveness. While the computational cost remains a concern, the paper's strengths in terms of problem definition, methodology, and empirical validation outweigh its weaknesses. The potential influence on future research in dialogue planning is substantial. However, this work could be strengthened with a larger number of volunteers for human study, as well as more detail in the "Inference Cost" section.

- **Score**: 8/10

### **[Context-Enhanced Vulnerability Detection Based on Large Language Model](http://arxiv.org/abs/2504.16877v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Context-Enhanced Vulnerability Detection Based on Large Language Model":

**Summary:**

The paper addresses the challenge of improving vulnerability detection in software using Large Language Models (LLMs). The authors propose a context-enhanced approach called PacVD, which combines program analysis with LLMs. PacVD extracts contextual information at various levels of abstraction by analyzing the source code and identifying critical APIs using control flow and data flow analysis.  This context is then provided to the LLM, effectively filtering out irrelevant noise that might hinder model performance. The authors perform a comprehensive empirical study using GPT-4, DeepSeek, and CodeLLaMA, evaluating the impact of different levels of contextual granularity and prompting strategies on vulnerability detection performance. The key findings indicate that incorporating abstracted context significantly improves LLM-based vulnerability detection, the optimal abstraction level varies depending on the LLM, and specialized prompt engineering is essential for maximizing performance.

**Critical Evaluation:**

*   **Novelty:** The paper's novelty lies in the specific combination of program analysis, API abstraction, and LLMs for vulnerability detection, especially the use of "primitive API abstraction." While previous work has explored using LLMs and program analysis separately or combining raw code from callees, the authors provide a targeted approach to extracting and presenting contextual information, balancing information richness with computational efficiency. This nuanced context extraction strategy is a novel component. The methodology for determining the level of the abstraction (fuzzy branch, concrete branch, etc.) is also a valuable contribution.
*   **Significance:** The paper makes a significant contribution by addressing a known limitation of LLMs in vulnerability detection: the need for sufficient context without introducing excessive noise. The systematic empirical evaluation provides valuable insights into the effectiveness of different abstraction levels and prompting strategies for various LLMs. The results have direct implications for practical vulnerability detection systems and can guide practitioners in selecting appropriate techniques and LLMs for their needs. Furthermore, the demonstration of competitive performance relative to complex "all callees" methodologies with focused API call extraction and analysis is notable.
*   **Strengths:**
    *   **Comprehensive Evaluation:** The empirical study is thorough, using multiple LLMs, abstraction levels, prompting strategies, and a sizable dataset of real-world vulnerabilities.
    *   **Clear Research Questions:** The research questions are well-defined and address important aspects of LLM-based vulnerability detection.
    *   **Actionable Insights:** The paper provides practical recommendations for practitioners and researchers based on the experimental results.
    *   **Strong Rationale:** The choices for API abstraction, prompt engineering approaches and LLM selection have a clear rationale and build on previous work.
    *   **Reproducible Results:** The authors make their datasets and experimental details publicly available, which promotes reproducibility and further research.

*   **Weaknesses:**
    *   **Limited Scope:** The study focuses primarily on C/C++ vulnerabilities. While this is a critical area, extending the approach to other programming languages could broaden its applicability.
    *   **Idealized Prompts:** Prompt engineering is a rapidly evolving field. The studied prompt strategies, while common, could become outdated quickly. The generalizability of the "best" prompt is also questionable across different contexts.
    *   **Reliance on Static Analysis:** The approach relies on static program analysis to extract contextual information. This can limit its ability to detect vulnerabilities that depend on dynamic program behavior.

*   **Potential Influence:** The paper has the potential to influence the field of software security by providing a more effective and practical approach to LLM-based vulnerability detection. The insights gained from the empirical study can guide the development of future vulnerability detection tools and techniques. The API abstraction method could inspire similar context extraction approaches in other areas of software analysis.

The PacVD approach provides a balanced strategy of context retrieval and analysis that may improve the consistency and practicality of LLM-based vulnerability analysis. The novel method of program analysis-driven context extraction is a significant element.

**Score: 8**

**Justification:** The paper presents a novel and significant contribution to the field of vulnerability detection. While it has some limitations (scope and reliance on static analysis), the strengths outweigh the weaknesses. The comprehensive evaluation, clear research questions, actionable insights, and strong rationale make it a valuable addition to the literature. The findings have the potential to influence practical vulnerability detection systems and inspire further research in this area.  The ability to maintain high recall while improving precision is a notable factor for practical deployment.

- **Score**: 8/10

### **[Robo-Troj: Attacking LLM-based Task Planners](http://arxiv.org/abs/2504.17070v1)**
- **Summary**: Here's a summary and critical evaluation of the Robo-Troj paper:

**Summary:**

The paper introduces "Robo-Troj," a novel backdoor attack targeting LLM-based task planners in robotics. It addresses the lack of security research on these systems, which are increasingly prevalent. Robo-Troj uses a multi-trigger approach, where specific "trigger words" embedded in a user's query cause the LLM to generate malicious task plans (e.g., a kitchen robot cutting a hand instead of making coffee). The authors develop a two-stage attack: (1) optimizing a parametric trigger distribution to find effective trigger words and (2) injecting these triggers into the training data to fine-tune the LLM via Soft Prompt Tuning (SPT). They demonstrate the attack's effectiveness in both simulated and real-world robot environments, showing high attack success rates (ASR) while maintaining good performance on benign tasks (CDA - Clean Data Accuracy). Ablation studies explore multi-trigger attacks and diverse malicious behaviors.

**Critical Evaluation:**

*   **Novelty:** The paper presents a significant contribution by being the first to explore backdoor attacks on LLM-based task planners for robotics.  While backdoor attacks on LLMs have been studied in other contexts (e.g., text classification), the specific application to robotics, with its unique safety implications and need for diverse triggers across different robot functions, introduces significant novelty.  The multi-trigger optimization (MBO) approach is also a novel contribution, addressing the challenge of different robotic domains requiring different triggers for malicious behavior.

*   **Significance:** The research has high significance.  As robots become more integrated into our lives and rely on LLMs for decision-making, their vulnerability to attacks like Robo-Troj poses a real threat.  The paper highlights a critical security gap and motivates the development of defenses. The potential impact of a compromised robot executing harmful actions in a home, healthcare facility, or industrial setting underscores the importance of this work. The evaluation in real robot environments and simulated settings demonstrates the attack's practical relevance.

*   **Strengths:**

    *   **Clear Problem Definition:** The paper clearly articulates the security risks associated with LLM-based robot task planning.
    *   **Novel Attack Design:** The Robo-Troj attack, including MBO, is well-designed and adapted to the robotics domain.
    *   **Comprehensive Evaluation:** The experiments are thorough, including simulations, real-world robot demonstrations, and ablation studies to demonstrate multi-trigger attack effectiveness. The metrics (ASR, CDA) are appropriate for evaluating both attack success and benign task performance.
    *   **Practical Demonstration:** The real-world demonstration showcases the potential for catastrophic consequences, further strengthening the significance of the research.

*   **Weaknesses:**

    *   **Limited Model Scope:** The experiments focus on a few LLMs (GPT2-Large, GPT-J-6B, and Llama-2-7B). It does not include evaluations on currently more powerful models like GPT-4 or other recently developed LLMs, citing their lack of open access.  While understandable, this limits the generalizability of the findings to the most advanced systems.
    *   **Dataset Limitations:** The primary evaluation uses the VirtualHome dataset, which is specifically for household tasks. While the paper attempts to address generalizability by testing on other language datasets, the core focus remains limited to one domain.  This is a limitation of the available datasets more than the methodology of this particular paper but should still be noted.
    *   **Defenses:** The paper briefly touches on possible defense directions but doesn't explore them in detail.  A more in-depth discussion and potentially preliminary evaluation of defense strategies would further enhance the work.

*   **Impact:** The paper is likely to have a significant impact on the field by:

    *   Raising awareness about the security vulnerabilities of LLM-based robot task planning.
    *   Motivating further research on defenses against backdoor attacks in robotics.
    *   Influencing the design and development of more secure robot systems that are resilient to attacks.

**Justification for Score:**

The paper demonstrates significant novelty and is highly relevant and significant in a growing field. The detailed experiments are very well-conducted and well-written. While the scope of LLMs examined and dataset scope are limited, they do not detract from the value of the work. The lack of detailed exploration of defensive techniques is another weakness.

Score: 8

- **Score**: 8/10

### **[Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning](http://arxiv.org/abs/2504.17192v1)**
- **Summary**: Here's a concise summary and critical evaluation of the paper "Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning":

**Summary:**

This paper introduces PaperCoder, a novel multi-agent LLM framework designed to automatically generate executable code repositories from machine learning research papers.  PaperCoder emulates the human development lifecycle by breaking down the code generation process into three stages: planning (high-level roadmap and system architecture), analysis (interpreting implementation details), and generation (dependency-aware code production). The framework uses specialized LLM agents in each phase to collaborate effectively. The authors evaluate PaperCoder on a dataset of recent machine learning papers and show that it substantially outperforms existing baselines in generating high-quality, faithful code implementations that are helpful to researchers.

**Critical Evaluation:**

*   **Novelty:** The idea of automatically generating code from research papers is not entirely new, but PaperCoder's multi-agent, multi-stage approach and its focus on complete *repository-level* code generation is a significant step forward. The framework's ability to operate effectively *without* reliance on existing code or APIs is a particularly important advance. The novel structured approach that mirrors the typical lifecycle of human developers and researchers is commendable.

*   **Significance:** The work addresses a crucial problem in machine learning research: the lack of reproducible code. PaperCoder has the potential to significantly accelerate research by enabling easier validation and building upon prior work.  The human evaluations, particularly those involving the original paper authors, provide strong evidence of the framework's usefulness and accuracy. Furthermore, performance gains across the recent PaperBench benchmark highlights PaperCoder's practical value and superiority compared to alternative methods.

*   **Strengths:**
    *   The multi-agent, multi-stage design is well-motivated and logically sound.
    *   The experimental results are strong, demonstrating significant improvements over baselines on a comprehensive benchmark.
    *   The human evaluations provide convincing evidence of the framework's real-world utility.
    *   Demonstration of PaperCoder generating executable code bases with minimal manual modifications needed to execute.
    *   Clear articulation of architecture and functionality through included Figures and Tables.
    *   Extensive experimental analysis and evaluation metrics.

*   **Weaknesses:**
    *   The reliance on LLMs introduces potential issues of hallucination or misinterpretation of the paper. While the authors address this through structured planning and analysis phases, the risk remains.
    *   Although the paper claims the models are intended for open access, the use of proprietary LLMs may inhibit accessibility, impacting reproducibility.
    *   Detailed evaluation strategies for reproducibility using standard coding strategies or debugging tools is lacking. Future work could include additional documentation on troubleshooting potential issues from incomplete implementations.
    *   The paper's limitations around the inability to incorporate execution-based analyses with debugging could be addressed by including more information on the underlying framework or dataset.

*   **Impact:** PaperCoder has the potential to transform the way machine learning research is conducted by reducing the time and effort required for code reproduction.  It could also facilitate the widespread adoption of new techniques and lead to faster scientific progress.

**Score: 8.5**

**Justification:**

PaperCoder presents a novel and highly valuable solution to a significant problem in the machine learning community. The work's strong experimental results and positive human evaluations, coupled with its structured design, demonstrate its potential for real-world impact. The score is slightly lower than a 9 or 10 due to the inherent limitations of relying on LLMs and accessibility due to their proprietary nature, as well as the lack of more comprehensive evaluations strategies using standard debugging strategies.

- **Score**: 8/10

### **[Towards Generalized and Training-Free Text-Guided Semantic Manipulation](http://arxiv.org/abs/2504.17269v1)**
- **Summary**: Here's a concise summary and critical evaluation of the paper:

**Summary:**

The paper introduces GTF (Generalized Training-Free), a novel approach for text-guided semantic manipulation across multiple modalities (image, video, and 3D). The key idea is to leverage the geometric properties of noise within diffusion models to perform semantic editing. GTF categorizes semantic editing into addition and removal and designs noise composition strategies tailored for each. It operates training-free by controlling the geometric relationship between noises predicted under source and target prompts, enabling plug-and-play integration with existing diffusion-based methods. The paper demonstrates GTF's effectiveness through experiments showcasing its controllability, generalizability, and superior performance compared to state-of-the-art methods.

**Critical Evaluation:**

*   **Novelty:** The paper's core novelty lies in its training-free approach to semantic manipulation by directly manipulating the noise space of diffusion models. While the concept of manipulating latent spaces for editing is not entirely new, the authors provide a theoretical grounding for why manipulating noise vectors functions effectively as semantic control. The separation into addition and removal operations, combined with tailored noise combination strategies, showcases thoughtful engineering. The modality-agnostic nature of the framework, stemming from a diffusion model foundation, is also a strong point.
*   **Significance:** The significance of the paper lies in its potential to simplify and generalize text-guided semantic manipulation. The training-free aspect makes the method easily adoptable and removes the need for task-specific fine-tuning, a major advantage. The ability to operate across different modalities makes it a versatile tool for various applications. The results shown, particularly in video editing and 3D editing, are compelling and demonstrate a significant improvement over existing approaches in terms of content preservation and editing fidelity.
*   **Strengths:**
    *   Strong theoretical grounding for noise manipulation.
    *   Training-free and plug-and-play integration with existing diffusion models.
    *   Generalizability across multiple modalities (image, video, 3D).
    *   Significant performance improvements in visual quality and semantic control.
*   **Weaknesses:**
    *   The paper relies heavily on the established capabilities of diffusion models. While GTF enhances these models, it doesn't introduce fundamental innovations to the diffusion process itself.
    *   While the paper discusses advantages, it doesn't offer a detailed computational complexity analysis or provide concrete numbers comparing runtimes with fine-tuning based approaches. More information about the impact of the proposed noise manipulation strategies on inference time is required.
    *   The success of GTF depends heavily on the quality of the noise predictions from the underlying diffusion model. If the diffusion model struggles with certain prompts or modalities, GTF's performance will likely be affected. The paper does not mention this specifically.
    *   The user studies show relative preferences among different methods, there is no absolute measure of how realistically or naturally do the edited images/videos look.

*   **Potential Influence:** The paper has the potential to influence the field of semantic manipulation by providing a more efficient and generalizable approach. The training-free nature of GTF could lead to its widespread adoption and integration into various applications.
*   **Rigor:** The experiments are comprehensive, covering diverse modalities and manipulation types. The quantitative metrics align with the observed qualitative results. The ablation studies help in understanding the influence of different components of the method.

**Justification for Score:**

While the paper does not present a radical departure from existing methods, the combination of theoretical insight, clever engineering, and strong experimental results justifies a high score. It successfully addresses key limitations of previous methods (training requirements, limited generalizability) and introduces a highly practical and effective approach. Its potential impact on the field in terms of ease of use and adaptability is significant.

Score: 8

- **Score**: 8/10

### **[Combining Static and Dynamic Approaches for Mining and Testing Constraints for RESTful API Testing](http://arxiv.org/abs/2504.17287v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces RBCTEST, a novel approach for mining and testing constraints on API response bodies, combining static and dynamic analysis. The static analysis uses Large Language Models (LLMs) to extract constraints from API specifications, while the dynamic analysis relies on API execution data.  To mitigate LLM hallucinations, the paper employs an Observation-Confirmation (OC) scheme. The mined constraints are then used to generate test cases. The approach is evaluated on real-world API services, demonstrating improved precision in constraint mining and the detection of mismatches between API specifications and implementations. The paper also contributes a manually verified benchmark dataset for API response bodies.

**Critical Evaluation:**

*   **Novelty:** The combination of LLM-based static analysis and dynamic analysis for API constraint mining is a significant contribution. While dynamic analysis for constraint inference (like AGORA) exists, the integration with static analysis from API specifications, especially leveraging LLMs, introduces a new dimension. The application of the Observation-Confirmation scheme to refine LLM-derived constraints is also a valuable technique to address the hallucination problem. The manually-verified benchmark is a valuable resource for the community.

*   **Significance:** The ability to automatically mine constraints and generate test cases for API response bodies is crucial for ensuring API reliability and correctness. Existing approaches often focus on status codes and schema validation, neglecting logical constraints. RBCTEST addresses this gap, potentially leading to more robust and comprehensive API testing. The detection of mismatches between API specifications and implementations is also a valuable outcome. The fact that they've found several mismatches that were *already being discussed on developer forums* is a very strong indicator of practical usefulness.

*   **Strengths:**

    *   **Comprehensive Approach:** Combines static and dynamic analysis to overcome limitations of each.
    *   **LLM-Based Constraint Mining:** Leverages LLMs for understanding API specifications and inferring constraints.
    *   **Hallucination Mitigation:** Employs Observation-Confirmation scheme to improve LLM precision.
    *   **Test Case Generation:** Automatically generates test cases based on mined constraints.
    *   **Real-World Evaluation:** Evaluated on real-world APIs, demonstrating practical applicability.
    *   **Publicly Available Dataset:** Contributes a valuable resource to the community.
    *   **Well-documented findings:** Findings include detailed analysis of inconsistencies and their root causes.

*   **Weaknesses:**

    *   **Reliance on Specification Quality:** The static analysis heavily depends on the quality and completeness of API specifications. Incomplete or inaccurate specifications can limit the effectiveness of the approach. While the authors address missing response body information, they could mention the impact of incorrect specifications which are difficult to handle.
    *   **Complexity of LLM Prompts:** LLM performance heavily relies on prompt design. The prompts might require fine-tuning for different API specifications. The prompts themselves were not really *analyzed* for optimality, the authors merely chose a prompt design and stuck with it.
    *   **Potential for Overfitting:** The LLM might overfit to the training data or specific API specifications, potentially limiting its generalizability to new APIs.
    *   **Scalability:** LLMs can be computationally expensive. The scalability of the approach for very large API specifications needs further investigation.
    *   **Limited dynamic analysis** - The dynamic component depends heavily on the quality and diversity of input provided to AGORA. This could be more clearly emphasized and discussed.

*   **Impact:** The paper has the potential to significantly impact the field of API testing by providing a more automated and comprehensive approach for constraint mining and test case generation. The combination of static and dynamic analysis, along with the use of LLMs, offers a promising direction for future research.

**Justification for Score:**

The paper presents a genuinely novel approach to API testing by combining LLM-based static analysis with traditional dynamic analysis techniques. The use of Observation-Confirmation to address LLM hallucinations is a valuable contribution to prompt engineering in software testing. The results demonstrate the practical utility of RBCTEST in detecting mismatches and generating test cases for real-world APIs. While the approach has some limitations (reliance on specification quality, potential for overfitting, etc.), the strengths outweigh the weaknesses. The paper has the potential to inspire further research in this area and improve the reliability of APIs. The existence of the benchmark enhances its impact.

Score: 8

- **Score**: 8/10

### **[FLUKE: A Linguistically-Driven and Task-Agnostic Framework for Robustness Evaluation](http://arxiv.org/abs/2504.17311v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces FLUKE, a task-agnostic framework for evaluating the robustness of NLP models. FLUKE generates minimal variations of test data across different linguistic levels (orthography, morphology, syntax, semantics, discourse, language varieties, and biases) using large language models (LLMs) and human validation. The authors evaluate several fine-tuned pre-trained models (PLMs) and LLMs on four NLP tasks (coreference resolution, dialogue contradiction classification, named entity recognition, and sentiment analysis) using FLUKE. Their findings highlight that the impact of linguistic variations is task-dependent, LLMs still exhibit brittleness to certain variations, and all models show vulnerability to negation modifications.

**Critical Evaluation:**

*   **Novelty:** The idea of systematically evaluating model robustness through minimal linguistic variations is not entirely new. Previous works have explored adversarial examples and minimal contrastive pairs. However, FLUKE's novelty lies in its **comprehensive and task-agnostic approach**, using LLMs to generate modifications across a wide range of linguistic levels and incorporating human validation for quality control. The automated modification generation workflow utilizing LLMs paired with human oversight for verification is a notable innovation. Also, it is significant to see the authors generalize across four tasks instead of focusing on one.

*   **Significance:**  The findings of the paper are significant because they demonstrate the limitations of current models, even LLMs, in handling seemingly simple linguistic variations. The task-dependent impact of these variations emphasizes the importance of comprehensive testing to uncover model shortcomings, and the consistent vulnerability to negation highlights a crucial area for improvement. By releasing their code and data, the authors enable other researchers to apply FLUKE and further investigate model robustness. The task-agnostic nature, alongside the publicly available data and code, make it easier for the method to be adopted by the broader NLP community. In addition, the results may encourage developers to integrate these robustness testing results into model cards when releasing new models.

*   **Strengths:**

    *   **Comprehensive and structured approach:** FLUKE covers a wide range of linguistic levels and provides a systematic way to generate and evaluate modifications.
    *   **Task-agnostic design:** The framework can be applied to different NLP tasks.
    *   **LLM-based generation and human validation:** Automating the modification process with LLMs and incorporating human validation ensures data quality and feasibility.
    *   **Interesting and practical findings:** The identified limitations of current models and the importance of specific linguistic variations provide valuable insights for model development.
    *   **Public release of code and data:** This enables further research and application of the framework.

*   **Weaknesses:**

    *   **Reliance on LLMs:** The quality of the generated modifications depends on the capabilities of the LLM used, which may be a limitation in certain scenarios. While the authors perform quality checks, the intrinsic biases of LLMs could still influence the results.
    *   **Computational cost:** Generating and validating modifications for a large number of samples and tasks can be computationally expensive.
    *   **Metric sensitivity**:  The weighted performance difference metric could be sensitive to the choice of the original dataset and model performance, although the authors aim to mitigate this by focusing on relative changes.
    *   **Scope of linguistic variations**: While FLUKE covers a broad range of linguistic variations, it may not be exhaustive. There may be other types of variations that could be relevant for certain tasks.

*   **Potential Influence:** The paper's findings could influence the development of more robust NLP models by highlighting areas where current models struggle. It could also promote the adoption of more comprehensive evaluation strategies, encouraging developers to go beyond standard benchmarks and consider linguistic variations. The released framework and data could serve as a valuable resource for future research in model robustness.

*   **Score:** 8

**Rationale:**

FLUKE represents a significant step forward in the systematic evaluation of NLP model robustness. While building upon existing ideas, it offers a unique combination of comprehensive coverage, task-agnostic design, LLM-based automation, and human validation. The findings are valuable and actionable, and the public release of resources increases its potential impact on the field. Despite some limitations, such as the reliance on LLMs and the potential for computational cost, FLUKE's overall contribution justifies a high score.

- **Score**: 8/10

### **[Fine-Grained Fusion: The Missing Piece in Area-Efficient State Space Model Acceleration](http://arxiv.org/abs/2504.17333v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

This paper investigates hardware acceleration strategies for State Space Models (SSMs), specifically focusing on the memory-bound operations during the prefill stage.  The authors use the Stream modeling framework (extended with new features for LLM analysis, which they make available) to explore operator fusion techniques for improved data locality and to analyze hardware design trade-offs. Their results demonstrate that optimized fusion and scheduling can significantly improve performance (up to 4.8x speedup) while reducing on-chip memory requirements. They also show that a fusion-aware hardware architecture can outperform the state-of-the-art MARCA accelerator within the same area budget (1.78x improvement). The core argument is that operator fusion is a critical enabler for efficient SSM accelerators.

**Critical Evaluation:**

*   **Novelty:** The paper's primary novelty lies in its systematic exploration of operator fusion and hardware architecture design space specifically for SSMs. While operator fusion is a known technique for transformer acceleration, its application and analysis in the context of SSMs, particularly considering the unique challenges of the prefill stage and the memory-bound nature of state updates, represents a significant contribution. The open-source extensions to the Stream modeling framework also add to the value by providing the community with tools to explore hardware accelerator design space for future LLMs and SSMs.
*   **Significance:** The results have significant implications for the design of future SSM accelerators. The demonstration that on-chip memory requirements can be drastically reduced through careful fusion strategies, and that a more compute-focused architecture can outperform existing designs, provides valuable guidance for hardware developers. This is especially important given the rising prominence of SSMs as an alternative to transformers for long-sequence processing.
*   **Strengths:**
    *   **Systematic Approach:** The paper adopts a systematic and well-structured approach, beginning with a clear problem definition, a high-level roofline analysis, and then a detailed exploration of operator fusion and hardware design space.
    *   **Quantitative Results:** The paper presents clear and compelling quantitative results, demonstrating the benefits of the proposed techniques. The speedup numbers, memory reduction percentages, and performance comparisons are all well-supported.
    *   **Open-Source Contribution:** The open-sourcing of the Stream extensions provides a valuable resource for the research community and helps to promote further research in this area.
    *   **Clear Methodology:** The paper clearly explains the methodology used for performance estimation and hardware modeling, which enhances the reproducibility and credibility of the results.

*   **Weaknesses:**
    *   **Limited Hardware Model:** While the parameterized hardware model captures key architectural features, it is a simplified representation of a real-world accelerator. The exploration could have benefited from more detailed hardware modeling, potentially including cycle-accurate simulations for a more realistic view of performance. Also, it is worth noting, memory controllers in accelerators can be quite complex, with complex access patterns, which might not have been fully captured in the simulations and can change the conclusions drawn.
    *   **Workload focus:** The analysis heavily focuses on the prefill stage. A deeper analysis of the decode stage and the trade-offs between the two stages in the overall accelerator design would be beneficial.
    *   **Implementation Details:** Further implementation details of the Stream extensions are warranted.
    *   **Area and power estimation for different parameters is not clear:** The estimation for power and area cost based on compute and memory tradeoff is not fully clear in the paper. Therefore, it's hard to judge the quality of area estimations of the design space explorations.

*   **Potential Influence:** The paper has the potential to significantly influence the design of next-generation SSM accelerators. It provides a clear roadmap for improving performance and reducing resource requirements. The insights gained from this work could be used to develop more efficient and scalable hardware solutions for a wide range of applications.

**Overall Score:**

Considering the paper's novelty, significance, strengths, and weaknesses, a score of **8** is appropriate. The paper provides a valuable contribution to the field of hardware acceleration for SSMs, offering novel insights and practical guidance for hardware developers. The well-articulated methodology and quantitative results strengthens the paper, but further improvement could be had if the paper had a more detailed hardware model, better exploration of area and power estimations, and a deeper implementation explanation.

Score: 8

- **Score**: 8/10

### **[LiveLongBench: Tackling Long-Context Understanding for Spoken Texts from Live Streams](http://arxiv.org/abs/2504.17366v1)**
- **Summary**: Here is a summary and critical evaluation of the paper "LiveLongBench: Tackling Long-Context Understanding for Spoken Texts from Live Streams":

**Summary:**

The paper introduces LiveLongBench, a new benchmark designed to evaluate long-context understanding in spoken texts derived from live streams. The benchmark addresses the challenges posed by the unique characteristics of spoken language, such as high redundancy, informality, and topic drift, which are often overlooked by existing long-context benchmarks that primarily focus on written texts. LiveLongBench includes a dataset of live stream transcripts in both Chinese and English, with tasks categorized into retrieval-dependent, reasoning-dependent, and hybrid scenarios. The paper evaluates the performance of various large language models (LLMs) and context compression methods on the benchmark, highlighting their limitations in handling the complexities of spoken language and proposing a hybrid KV cache compression strategy to improve performance and memory efficiency.

**Critical Evaluation:**

*   **Novelty:** The key novelty of this work lies in the creation of a benchmark specifically designed to evaluate long-context understanding of *spoken* language, sourced from live streams. This addresses a recognized gap in the literature, as existing benchmarks largely focus on written text and do not capture the nuances of spoken communication, especially in informal, real-world settings. Introducing 'semantic multi-span' is another valuable contribution. The analysis of LLMs' performance on this novel dataset and the exploration of compression techniques specifically for spoken text further adds to the novelty.
*   **Significance:** The work has significant implications for the development of more effective and efficient LLMs for real-world applications such as e-commerce systems, conversational AI, and real-time communication. By highlighting the shortcomings of current models on spoken text and proposing a practical solution in the form of a hybrid compression strategy, the paper provides valuable insights and directions for future research. The Data Envelopment Analysis (DEA) framework for evaluating the trade-offs between performance and memory efficiency is also a valuable contribution.
*   **Strengths:**
    *   The creation of a well-motivated and relevant benchmark.
    *   A comprehensive evaluation of popular LLMs on the new benchmark, revealing performance gaps.
    *   The proposal of a hybrid KV cache compression strategy that yields improved performance and memory efficiency.
    *   The DEA framework provides a principled way to balance performance and resource usage.
    *   The paper is well-written and clearly structured.
*   **Weaknesses:**
    *   The data source is limited to Douyin e-commerce live streams. While this captures certain aspects of spoken language, it may not fully represent the diversity of spoken language across different domains (e.g., academic lectures, news broadcasts). This limitation is acknowledged by the authors.
    *   The number of annotators (five students) could be considered a small sample size, potentially affecting the reliability of gold-standard labels. Although quality control was implemented, a larger and more diverse annotation team would strengthen the benchmark's robustness.
    *   Although the performance improvement observed from fine-tuning domain-specific models is notable, its lack of consistency calls for more meticulous customization of parameters in diverse tasks.
    *   The paper could benefit from a more thorough investigation of the qualitative errors made by the models, providing deeper insights into the specific challenges posed by spoken language.

**Overall, the paper makes a significant contribution to the field by addressing a critical gap in long-context understanding research and providing a practical benchmark for evaluating models on real-world spoken texts. The proposed hybrid compression strategy and the DEA framework offer valuable solutions for improving the efficiency of LLMs in this setting.**
I would assign a score of 8 based on these points.

Score: 8

- **Score**: 8/10

### **[Unified Attacks to Large Language Model Watermarks: Spoofing and Scrubbing in Unauthorized Knowledge Distillation](http://arxiv.org/abs/2504.17480v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces Contrastive Decoding-Guided Knowledge Distillation (CDG-KD), a unified framework for attacking watermarks in large language models (LLMs) through unauthorized knowledge distillation. CDG-KD enables both watermark scrubbing (removal) and spoofing (forgery) in black-box settings, where internal model access is unavailable. The method leverages contrastive decoding to extract corrupted or amplified watermark text by comparing outputs from the student model and a weakly watermarked reference model. This extracted text is then used for bidirectional distillation to train new student models capable of either removing or forging watermarks. The paper demonstrates the effectiveness of CDG-KD through experiments, showing its ability to perform attacks while preserving the distilled model's general performance.

**Critical Evaluation:**

*   **Novelty:** The paper's primary novelty lies in presenting a unified framework for *both* scrubbing and spoofing attacks on LLM watermarks within an unauthorized knowledge distillation scenario, *without* requiring internal model access (black-box setting). Prior work has typically focused on one type of attack or assumed some degree of access to the model internals. The use of contrastive decoding to guide knowledge distillation for watermark manipulation is also a notable contribution. The paper effectively highlights the vulnerability introduced by watermark radioactivity during knowledge distillation and proposes a practical approach to exploit it. The idea of classifying the presence of watermark as simple classification task and demonstrating consistency between classifier based and p-value based watermark detectors is clever.

*   **Significance:** The paper's significance stems from exposing a critical vulnerability in LLM watermarking schemes. By demonstrating that watermarks can be manipulated indirectly through student models, even in black-box settings, the authors highlight the need for more robust and unforgeable watermarking strategies. The findings are particularly relevant given the increasing adoption of LLMs in various applications and the growing concerns about misinformation and intellectual property protection. Moreover, the demonstration of successful spoofing attacks raises serious ethical concerns about the potential for falsely attributing harmful content to safety-aligned LLMs. The findings relating to variations in n-gram length and generation length as a key factor influencing detectability adds value to the work.

*   **Strengths:**
    *   Well-defined problem and clear objectives.
    *   The proposed CDG-KD framework is well-motivated and explained.
    *   The experimental evaluation is comprehensive, covering various watermarking strategies, attack scenarios, and evaluation metrics.
    *   The ablation studies provide valuable insights into the method's behavior and the impact of different parameters.
    *   The paper is well-written and clearly presents the results.
    *   The discussion of ethical considerations adds further value.
    *   The appendices are rich and add greater understanding of the paper.

*   **Weaknesses:**
    *   The method's reliance on a relatively large distillation corpus could be a limitation in low-resource scenarios. Though the problem is well acknowledged.
    *   The paper could benefit from a more detailed analysis of the computational overhead introduced by contrastive decoding.
    *   The generalizability of the method to a broader range of watermarking paradigms (e.g., sentence-level schemes) remains to be fully explored.
    *   The degree of control exerted over content of generated text in spoofing attacks could be more carefully explained.

*   **Potential Influence:** The paper is likely to influence future research on LLM watermarking by shifting the focus towards defense mechanisms that are resilient to indirect manipulation through student models. The proposed CDG-KD framework could serve as a valuable tool for evaluating the robustness of existing and new watermarking schemes. The paper's findings could also inform the development of more secure knowledge distillation techniques that prevent watermark radioactivity or make it more difficult to exploit.

**Score: 8**

**Rationale:** The paper presents a novel and significant contribution to the field of LLM watermarking. The proposed CDG-KD framework effectively addresses a critical vulnerability, and the experimental evaluation is thorough and well-executed. While some limitations exist, the paper's strengths outweigh its weaknesses, and its findings are likely to have a significant impact on future research and development in this area. The work is not a paradigm shift, but rather a well-engineered exploitation of a known issue. The influence is expected to be moderate with further works building from here.

- **Score**: 8/10

### **[A Comprehensive Survey of Knowledge-Based Vision Question Answering Systems: The Lifecycle of Knowledge in Visual Reasoning Task](http://arxiv.org/abs/2504.17547v1)**
- **Summary**: Here's a summary and critical evaluation of the provided survey paper on Knowledge-Based Vision Question Answering (KB-VQA) systems:

**Summary:**

The paper presents a comprehensive survey of KB-VQA systems, which extend traditional VQA by incorporating external knowledge for visual reasoning. It structures the field into three main stages: knowledge representation, knowledge retrieval, and knowledge reasoning. The survey categorizes existing KB-VQA approaches within these stages, examining various knowledge integration techniques. It explores the evolution of KB-VQA, especially with the advent of Large Language Models (LLMs) as knowledge repositories and reasoners. The paper identifies challenges, limitations, and future research directions in the field, aiming to provide a foundation for further advancements. It also summarizes common datasets and evaluation metrics used in KB-VQA research.

**Critical Evaluation:**

*   **Novelty:** The paper's novelty lies in providing a structured and systematic overview of KB-VQA research, a field that is rapidly evolving. While previous surveys have touched upon VQA in general or multimodal learning, this survey specifically focuses on the unique challenges and solutions within KB-VQA. The categorization into representation, retrieval, and reasoning provides a useful framework. The emphasis on the impact of LLMs on KB-VQA systems highlights recent trends in the field. This targeted approach to the KB-VQA domain is a key differentiator.

*   **Significance:** The survey is significant for several reasons:

    *   **Comprehensive Coverage:** It covers a wide range of research efforts, from early approaches using structured knowledge graphs to more recent methods leveraging LLMs.
    *   **Organized Framework:** The structured framework allows readers to quickly understand the different components of KB-VQA systems and compare various approaches.
    *   **Identification of Challenges:** The paper explicitly points out persistent challenges, such as multimodal knowledge fusion, knowledge retrieval from noisy sources, and complex reasoning, which are critical for guiding future research.
    *   **Future Directions:** The discussion of future research directions provides valuable insights for researchers looking to contribute to the field. These include reasoning over heterogenous information, aligning with human-oriented methods (via LLMs), and creating more unified benchmarks for evaluation.
*   **Strengths:**

    *   **Well-structured:** The survey is logically organized, making it easy to follow.
    *   **Clear definitions:** It defines KB-VQA clearly, highlighting the differences from traditional VQA.
    *   **Comprehensive literature review:** The survey covers a significant number of relevant papers.
    *   **Up-to-date:** It includes recent advances related to LLMs, reflecting the current state-of-the-art.
    *   **Critical Analysis:** The paper does not just list existing methods, but also critically analyzes their strengths and weaknesses.

*   **Weaknesses:**

    *   **Depth of Analysis:** While comprehensive, the analysis of individual methods could be deeper. More detailed comparisons of the performance of different techniques on benchmark datasets would be useful.
    *   **Lack of quantitative comparison:** The paper doesn't provide a quantitative meta-analysis of the various techniques. This would involve consolidating performance metrics reported across various papers, which would provide a more rigorous basis for comparing different KB-VQA models.
    *   **Future Evaluation:** Discussing the datasets and evaluations at the end makes it slightly less integrated with the rest of the paper. Integrating evaluation metrics and performance more directly into the analysis of each section (representation, retrieval, reasoning) could strengthen the connections between method and outcome.

*   **Potential Influence:** This survey will likely be a valuable resource for researchers entering the field of KB-VQA. It can help them quickly grasp the key concepts, challenges, and existing approaches. It also serves as a good reference for researchers already working in the field, providing a comprehensive overview of the current state-of-the-art and suggesting potential research directions. The identification of LLMs as a key driver in this area could lead to further research into their integration in KB-VQA tasks. The organized framework and identified challenges will likely influence the development of new and more effective KB-VQA models.

**Justification for Score:**

While the survey lacks a quantitative meta-analysis and more in-depth analysis of individual methods, its structured framework, comprehensive literature review, up-to-date information, and identification of key challenges and future directions make it a significant contribution to the field. It is a valuable resource for researchers and a good reference point for understanding the state-of-the-art in KB-VQA.

Score: 8

- **Score**: 8/10

### **[HalluLens: LLM Hallucination Benchmark](http://arxiv.org/abs/2504.17550v1)**
- **Summary**: Here's a summary and critical evaluation of the provided paper:

**Summary:**

The paper "HalluLens: LLM Hallucination Benchmark" introduces a new benchmark for evaluating hallucinations in Large Language Models (LLMs). It differentiates between *intrinsic* (contradictory to input context) and *extrinsic* (contradictory to training data) hallucinations, arguing that existing benchmarks often conflate hallucination with factuality.  HalluLens includes new extrinsic hallucination tasks, using dynamic test set generation to mitigate data leakage. The benchmark also analyzes existing factuality and hallucination benchmarks, identifying limitations and saturation issues. The paper aims to: (1) establish a clear hallucination taxonomy, (2) introduce new extrinsic hallucination tasks with dynamic data, and (3) analyze existing benchmarks to differentiate hallucination from factuality. It details the construction of new tasks (PreciseWikiQA, LongWiki, NonExistentRefusal) designed to assess the model's ability to avoid generating content inconsistent with its training data, even when prompted with plausible but nonexistent information.  The authors evaluate several popular LLMs on their benchmark, highlighting the strengths and weaknesses of each model.

**Critical Evaluation:**

*   **Novelty:** The paper's primary novelty lies in its explicit separation of hallucination from factuality, the proposed taxonomy (intrinsic vs. extrinsic), and the dynamic generation of test sets to combat data leakage. While the individual techniques for test set creation (LLM-based generation) are not entirely new, their application in the context of hallucination benchmarking, coupled with a clear taxonomic framing, constitutes a significant contribution.
*   **Significance:** Hallucination is a major obstacle to the widespread adoption of LLMs in real-world applications.  A reliable and comprehensive benchmark that goes beyond simple factuality checks is crucial for progress in this area. HalluLens addresses this need by providing a more nuanced evaluation framework, and by mitigating data leakage concerns, which have plagued previous benchmarks. The significance is heightened by the detailed analysis of existing benchmarks, pointing out their limitations and potential pitfalls. The tasks cover a good range of complexity from fact extraction to open-ended generation, with emphasis on dynamic test set generation to address data leakage which is a crucial contribution.
*   **Strengths:**
    *   **Clear taxonomy:**  The distinction between intrinsic and extrinsic hallucination is well-defined and helpful.
    *   **Dynamic test sets:** The dynamic generation approach is a significant strength, addressing a critical weakness in many existing benchmarks and enhancing robustness.
    *   **Comprehensive analysis:**  The paper provides a thorough analysis of the capabilities and limitations of a wide range of LLMs.
    *   **Extensible:**  The benchmark can be extended by adding new tasks for different domains and languages.
*   **Weaknesses:**
    *   **Dependence on LLMs for evaluation:** While using LLMs for evaluation is common, it introduces potential biases.  The reliance on LLMs for judging truthfulness and abstention requires careful prompt engineering (which they address) and ongoing scrutiny, to ensure the evaluator LLM isn't introducing its own hallucinations or biases. Also, human evaluation is necessary.
    *   **Wikipedia-centric:** The reliance on Wikipedia as the primary source for test set generation and evaluation (though justified by its inclusion in many training sets) might limit the benchmark's generalizability to scenarios where Wikipedia is less relevant. Though the reliance on Wikipedia content can be seen as a limitation, it is a reasonable starting point for benchmarking extrinsic hallucination, given its widespread use in training datasets. Future work could extend the benchmark to other domains and sources of information.
    *   **LLM prompt engineering:** It should be noted that although they present the prompt, the models are likely sensitive to the prompt itself.

*   **Potential Influence:** HalluLens has the potential to become a widely adopted benchmark for LLM hallucination, driving research towards more reliable and trustworthy models. The paper's clear taxonomy and dynamic data generation approach should influence the design of future benchmarks in this area. The benchmark highlights the need for nuanced evaluation strategies and motivates the development of methods to mitigate different types of hallucinations.

**Justification for Score:**

The paper provides a well-defined, novel and significant contribution to the field of LLM evaluation.  The clear taxonomy, the focus on extrinsic hallucination, and the dynamic test set generation address critical shortcomings of existing benchmarks.  The analysis of various LLMs on the benchmark provides valuable insights and guidance for future research. The weaknesses, while present, are acknowledged and potentially addressable in future work.

Score: 8

- **Score**: 8/10

## Other Papers
### **[IRIS: Interactive Research Ideation System for Accelerating Scientific Discovery](http://arxiv.org/abs/2504.16728v1)**
### **[A Survey of AI Agent Protocols](http://arxiv.org/abs/2504.16736v1)**
### **[MOSAIC: A Skill-Centric Algorithmic Framework for Long-Horizon Manipulation Planning](http://arxiv.org/abs/2504.16738v1)**
### **[Simple Graph Contrastive Learning via Fractional-order Neural Diffusion Networks](http://arxiv.org/abs/2504.16748v2)**
### **[HEMA : A Hippocampus-Inspired Extended Memory Architecture for Long-Context AI Conversations](http://arxiv.org/abs/2504.16754v1)**
### **[Lightweight Latent Verifiers for Efficient Meta-Generation Strategies](http://arxiv.org/abs/2504.16760v1)**
### **[How Effective are Generative Large Language Models in Performing Requirements Classification?](http://arxiv.org/abs/2504.16768v1)**
### **[Graph2Nav: 3D Object-Relation Graph Generation to Robot Navigation](http://arxiv.org/abs/2504.16782v1)**
### **[MOOSComp: Improving Lightweight Long-Context Compressor via Mitigating Over-Smoothing and Incorporating Outlier Scores](http://arxiv.org/abs/2504.16786v1)**
### **[Random Long-Context Access for Mamba via Hardware-aligned Hierarchical Sparse Attention](http://arxiv.org/abs/2504.16795v1)**
### **[Decoupled Global-Local Alignment for Improving Compositional Understanding](http://arxiv.org/abs/2504.16801v1)**
### **[Process Reward Models That Think](http://arxiv.org/abs/2504.16828v1)**
### **[GreenMind: A Next-Generation Vietnamese Large Language Model for Structured and Logical Reasoning](http://arxiv.org/abs/2504.16832v1)**
### **[LRASGen: LLM-based RESTful API Specification Generation](http://arxiv.org/abs/2504.16833v1)**
### **[Physically Consistent Humanoid Loco-Manipulation using Latent Diffusion Models](http://arxiv.org/abs/2504.16843v1)**
### **[Hyperspectral Vision Transformers for Greenhouse Gas Estimations from Space](http://arxiv.org/abs/2504.16851v1)**
### **[Monte Carlo Planning with Large Language Model for Text-Based Game Agents](http://arxiv.org/abs/2504.16855v1)**
### **[Emo Pillars: Knowledge Distillation to Support Fine-Grained Context-Aware and Context-Less Emotion Classification](http://arxiv.org/abs/2504.16856v1)**
### **[Planning with Diffusion Models for Target-Oriented Dialogue Systems](http://arxiv.org/abs/2504.16858v1)**
### **[Exploring How LLMs Capture and Represent Domain-Specific Knowledge](http://arxiv.org/abs/2504.16871v2)**
### **[Context-Enhanced Vulnerability Detection Based on Large Language Model](http://arxiv.org/abs/2504.16877v1)**
### **[Do Large Language Models know who did what to whom?](http://arxiv.org/abs/2504.16884v1)**
### **[Tracing Thought: Using Chain-of-Thought Reasoning to Identify the LLM Behind AI-Generated Text](http://arxiv.org/abs/2504.16913v1)**
### **[IberBench: LLM Evaluation on Iberian Languages](http://arxiv.org/abs/2504.16921v1)**
### **[Unsupervised Time-Series Signal Analysis with Autoencoders and Vision Transformers: A Review of Architectures and Applications](http://arxiv.org/abs/2504.16972v1)**
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
### **[Conversational Assistants to support Heart Failure Patients: comparing a Neurosymbolic Architecture with ChatGPT](http://arxiv.org/abs/2504.17753v1)**
### **[Replay to Remember: Retaining Domain Knowledge in Streaming Language Models](http://arxiv.org/abs/2504.17780v1)**
