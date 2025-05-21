# The Latest Daily Papers - Date: 2025-05-21
## Highlight Papers
### **[MLZero: A Multi-Agent System for End-to-end Machine Learning Automation](http://arxiv.org/abs/2505.13941v1)**
- **Summary**: Okay, here's a summary and critical evaluation of the "MLZero: A Multi-Agent System for End-to-end Machine Learning Automation" paper:

**Summary:**

The paper introduces MLZero, a novel multi-agent system designed for end-to-end machine learning automation across diverse data modalities with minimal human intervention. MLZero uses a hierarchical architecture powered by Large Language Models (LLMs) and incorporates a cognitive perception module, a semantic memory, and an episodic memory. The perception module transforms raw multimodal inputs into perceptual context. The semantic memory is used for enhanced knowledge retrieval, and the episodic memory for error detection and correction. The system iteratively generates and refines code, ultimately producing ready-to-use models and predictions. The authors evaluate MLZero on MLE-Bench Lite and on their new Multimodal AutoML Agent Benchmark, demonstrating superior performance in both success rate and solution quality compared to existing AutoML systems.

**Rigorous and Critical Evaluation:**

*   **Novelty:** The combination of a multi-agent architecture, a cognitive perception module, and the integration of semantic and episodic memory to facilitate end-to-end automation is a novel approach in the field of AutoML. The use of LLMs in AutoML is not entirely new, but MLZero's comprehensive architecture and specific memory modules are significant differentiators.

*   **Significance:** The paper addresses a critical gap in existing AutoML systems, which often struggle to handle multimodal data effectively and require substantial manual configuration. By automating the entire ML lifecycle from data preprocessing to model building, MLZero has the potential to democratize ML and make it accessible to a wider audience.

*   **Strengths:**

    *   **Comprehensive Architecture:** MLZero's multi-agent design, coupled with the memory modules, provides a robust and flexible framework for handling diverse ML tasks.
    *   **End-to-End Automation:** The system automates the entire ML pipeline, reducing the need for human intervention and expertise.
    *   **Strong Empirical Results:** The experimental results on both MLE-Bench Lite and the Multimodal AutoML Agent Benchmark demonstrate superior performance compared to existing systems. The introduction of a new benchmark is also a valuable contribution.
    *   **Effective Use of LLMs:** The paper addresses limitations of LLMs (like hallucination and outdated knowledge) through semantic and episodic memory.
    *   **Robustness demonstrated through experiments with a smaller (8B) LLM.**

*   **Weaknesses:**

    *   **Dependency on LLMs:** The system heavily relies on LLMs, which can be computationally expensive and have biases. While the paper mitigates some LLM limitations, the system's overall performance is still tied to the capabilities of the underlying LLMs.
    *   **Black Box Nature:** Like many LLM-based systems, MLZero can be difficult to interpret. Understanding *why* the system makes certain decisions or generates specific code remains a challenge.
    *   **Limited Evaluation on Real-World Problems:** While the benchmarks used in the paper are diverse, further evaluation on complex, real-world problems would be valuable.
    *   **Limited discussion of how the design would handle adversarial attacks on the multimodal agents.**
    *   **Scalability and Generalizability:** The experimental results are promising, but it's unclear how well MLZero would scale to extremely large datasets or generalize to entirely new domains outside of the tested ones.

*   **Potential Influence:** MLZero's approach has the potential to significantly influence the field of AutoML. It demonstrates the power of multi-agent systems and structured memory in automating complex ML tasks. The paper could inspire further research into:

    *   Developing more interpretable and explainable LLM-based AutoML systems.
    *   Exploring new techniques for mitigating the limitations of LLMs in AutoML.
    *   Creating more robust and generalizable AutoML systems that can handle a wider range of real-world problems.
    *   Incorporating causal analysis to prevent unintended consequences of poorly configured models.

**Justification for Score:**

The paper makes a significant contribution to the field of AutoML by presenting a novel architecture and demonstrating its effectiveness on challenging benchmarks. It addresses a critical gap in existing systems and has the potential to democratize ML. While there are some limitations related to LLM dependency and interpretability, the overall impact is substantial. The strong empirical results and the introduction of a new benchmark further strengthen the paper's contribution.

Score: 8

- **Score**: 8/10

### **[Visual Instruction Bottleneck Tuning](http://arxiv.org/abs/2505.13946v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces Vittle (Visual Instruction Bottleneck Tuning), a novel method to improve the robustness of multimodal large language models (MLLMs) under distribution shifts. Vittle leverages the information bottleneck (IB) principle by incorporating a bottleneck layer within the LLM architecture. This layer aims to learn minimal sufficient representations by discarding irrelevant information and retaining task-relevant signals. The authors theoretically justify Vittle's approach and empirically validate its effectiveness across a variety of tasks and datasets, including those involving different types of input perturbations and long-tail distributions. The experimental results demonstrate that Vittle consistently enhances robustness without sacrificing performance on standard benchmarks.

**Critical Evaluation:**

*   **Novelty:** While the information bottleneck principle has been applied in other contexts, the paper makes a significant contribution by: (1) Formulating a variational lower bound of IB explicitly tailored to the complexities of autoregressive multimodal instruction tuning. (2) Providing a practical and scalable implementation (Vittle) that is integrated into the LLM architecture (LLM backbone itself) to influence representations directly. (3) Theoretically justifying and experimentally validating how Vittle enhances MLLM robustness under distribution shifts.  The key strength here is the application of IB *within* the instruction tuning of MLLMs, where IB constraints are enforced on representations to improve robustness.
*   **Significance:** MLLM robustness is a critical concern in real-world deployment. By offering a lightweight and theoretically grounded approach, Vittle addresses a key limitation of existing MLLMs that are known to be fragile under minor variations. The experiments are comprehensive, covering a wide range of shifts and tasks, and provide solid evidence of Vittle's benefits. The results demonstrate improvements across multiple MLLMs architectures.
*   **Strengths:**
    *   Strong theoretical justification for the approach.
    *   Comprehensive evaluation across diverse tasks, shift types, and datasets.
    *   Practical and scalable implementation.
    *   Clear and well-written paper.
    *   Demonstrated compatibility with different MLLMs architectures.
*   **Weaknesses:**
    *   While the approach is lightweight, the training time increase is still present (upto 20%).
    *   The paper focuses mainly on the *internal* representations within the LLM itself. Although a modular approach, the choice and specifics of which layers to bottleneck might require architecture-specific tuning (i.e., layer 24 for a 7B model). The model could be more general with a training strategy that involves *more* layer-specific bottleneck layers.

*   **Potential Impact:** The paper has the potential to influence future research on robust MLLM development. Vittle's IB-based approach provides a useful and effective framework for tackling distribution shift problems. The fact that it requires a relatively small computational cost and annotation effort compared to large-scale data augmentation makes it a valuable contribution.
*  **Discussion:** The fact that Vittle is effective at mitigating hallucinations, or what seems to be overreliance on a single modality suggests that there's an interesting discussion to be had about the trade-offs in these representation. While Vittle induces good inductive biases, it would be interesting to measure it with specific interpretability methods (e.g., concept activation vectors) so that it may be useful to inspect those features to analyze *why* that level of inductive bias is useful.

**Score: 8**

**Rationale:** The paper presents a novel and valuable approach to improve the robustness of MLLMs under distribution shifts. While IB isn't a completely new concept, the way it's applied within MLLM instruction tuning, with theoretical grounding and strong empirical validation, makes this a significant contribution. The comprehensive experiments provide strong evidence of the method's effectiveness and the code will make this widely accessible to the community. While it is not a revolutionary breakthrough, the approach addresses a vital problem in a practical and effective way, thus deserves a high score.

- **Score**: 8/10

### **[Truth or Twist? Optimal Model Selection for Reliable Label Flipping Evaluation in LLM-based Counterfactuals](http://arxiv.org/abs/2505.13972v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper investigates the impact of judge model selection on the reliability of label flipping evaluation in LLM-based counterfactual data augmentation (CDA). It defines four types of relationships between the counterfactual generator and the judge model (same model, same family, independent with/without fine-tuning, and distilled models). Through extensive experiments using state-of-the-art LLM-based methods, various datasets, generator models, and judge models, complemented by a user study, the authors demonstrate that judge models with an independent, non-fine-tuned relationship to the generator provide the most reliable label flipping evaluations.  The paper finds that aligning the generator and judge model relationship with user study results improves model performance and robustness. They conclude that a fully automated pipeline for CDA may be inadequate and requires human intervention due to the discrepancy between automated evaluation and human judgment.

**Critical Evaluation:**

*   **Novelty:** The paper's novelty lies in its systematic investigation of the impact of judge model selection criteria on label flipping reliability in LLM-based CDA. Prior work has explored counterfactual generation and evaluation, but this paper provides a detailed analysis of how the *relationship* between generator and judge models affects the quality of the evaluation and, subsequently, downstream performance when using CDA. The specific categorization of these relationships (same, same family, independent with/without fine-tuning, distilled) is a well-defined framing that helps analyze the problem.
*   **Significance:** The findings have significant implications for researchers and practitioners using CDA to improve LLMs.  The paper demonstrates that relying on superficially "good" judge models (e.g., those with high downstream performance) can be misleading, and that carefully choosing a judge model with an independent relationship to the generator is crucial for reliable label flipping evaluation. This can lead to more effective CDA and, ultimately, more robust and higher-performing models. The identification of the need for human intervention in the evaluation process is a practical and valuable insight, highlighting the limitations of relying solely on automated pipelines.
*   **Strengths:**
    *   **Rigorous Experimental Design:** The paper presents a well-designed and executed experimental setup. The authors explore a diverse range of models, datasets, and relationships between generator and judge models.
    *   **User Study Validation:** The inclusion of a user study provides valuable ground truth and validates the findings derived from automated evaluation metrics. The user study adds significant weight to the conclusions.
    *   **Clear and Organized Presentation:** The paper is well-written and organized, making it easy to follow the research question, methodology, and results. The definitions are clear, and the tables effectively summarize the key findings.
    *   **Practical Implications:** The paper offers practical guidelines for selecting appropriate judge models for label flipping evaluation, benefiting researchers and practitioners involved in CDA.
*   **Weaknesses:**
    *   **Scope of Languages:** The study is limited to English datasets, which may limit the generalizability of the findings to other languages.
    *   **Model Families:** The study primarily focuses on models from the Qwen and Llama families, potentially overlooking nuances specific to other model architectures. While they include older architecture like BERT and RoBERTa, these are not used as generator models.
    *   **Evaluation Metrics:** While label flip rate is a common metric, there could be discussion on alternative or complementary evaluation techniques to more fully characterize the quality of counterfactuals. The paper mentions other factors like similarity and fluency in the Related Work, but does not include metrics for evaluating these in their experimental results.
    *   **Limited Discussion of Failure Cases:** While the paper mentions difficulties in identifying label flipping, a more in-depth discussion of specific failure cases and the limitations of LLMs in capturing nuances would be valuable.
*   **Potential Influence:** The paper has the potential to significantly influence the design and evaluation of CDA pipelines, leading to more reliable and effective training of LLMs. By highlighting the importance of careful judge model selection, the paper can help researchers avoid common pitfalls and improve the quality of their results.

**Score: 8**

**Justification:**
The paper offers a novel and significant contribution to the field of counterfactual data augmentation. The systematic investigation of judge model selection, coupled with the validation through user studies, provides valuable insights and practical guidance. The findings can influence how researchers and practitioners approach CDA, leading to more effective model training. While the limitations related to language and model family scope exist, they do not diminish the core contribution and the potential influence of the paper on the field. A score of 8 reflects the paper's strengths in research design, user study validation, and practical implications while acknowledging the limitations that prevent it from reaching a higher score.

- **Score**: 8/10

### **[DecIF: Improving Instruction-Following through Meta-Decomposition](http://arxiv.org/abs/2505.13990v1)**
- **Summary**: Here is a concise summary and critical evaluation of the paper "DecIF: Improving Instruction-Following through Meta-Decomposition":

**Summary:**

The paper introduces DecIF, a novel framework for generating high-quality instruction-following data for large language models (LLMs).  DecIF operates autonomously, using only LLMs to synthesize data without relying on pre-existing documents or external resources.  It achieves this through a two-stage process: 1) Instruction Synthesis, where instructions are decomposed into meta-information (domains, requests, scenarios) to control diversity and richness, and 2) Response Construction, where responses are generated and rigorously evaluated based on atomic-level evaluation criteria derived from the instructions. The framework includes consistency checks for instructions and response filtering to ensure quality. The authors demonstrate that LLMs trained on data generated by DecIF outperform existing methods on several instruction-following benchmarks.

**Critical Evaluation:**

The paper presents a valuable and novel approach to instruction-following data generation. DecIF's strength lies in its complete autonomy and reliance on internal LLM capabilities, making it highly flexible and scalable. The meta-decomposition strategy is a strong contribution, allowing for fine-grained control over the diversity and contextual richness of the generated instructions.  The rigorous evaluation process further strengthens the quality of the dataset.

**Strengths:**

*   **Novelty:** The fully autonomous approach to instruction-following data generation is a significant departure from methods that rely on external resources. The meta-decomposition technique offers a novel and effective way to control the generation process.
*   **Significance:**  The demonstrated improvements on instruction-following benchmarks highlight the potential for DecIF to enhance LLM capabilities. The data generated by DecIF has the potential to improve LLM alignment, which is a key challenge in the field.
*   **Flexibility and Scalability:** The approach is highly flexible as it can be used with different base LLMs and does not depend on specific resources. This makes it easily scalable for generating large datasets.
*   **Rigorous Evaluation:** The atomic-level evaluation and filtering strategy in the Response Construction stage adds to the robustness and quality of generated data.

**Weaknesses:**

*   **LLM Dependence:**  While the autonomy of DecIF is a strength, it also makes it heavily reliant on the capabilities of the underlying LLMs. The quality of the generated data is ultimately limited by the LLMs' understanding and generation abilities. As such it can have a hard time in generating datasets requiring very precise or niche real-world understanding.
*   **Filtering Bias:** As described in the limitations the current validation process may bias the dataset toward less complex instructions. There is opportunity in better instruction generation strategies in the future.
*   **Single-Turn Focus:** The current implementation focuses on single-turn instruction-following. Extending DecIF to multi-turn dialogues would further increase its applicability.
*   **Evaluation of Generalization:** While the paper evaluates the impact of DecIF generated data on instruction-following, it could be strengthened with more extensive evaluations of how this data affects other LLM capabilities, such as reasoning and knowledge retrieval.
*   **Limited Exploration of Hyperparameters:** While the appendix details prompt templates, it would benefit from further details on the experimental sweep used for the prompt selection and to determine parameters such as the temperature, top_p, and the number of requested samples.

**Overall:**

DecIF represents a significant step forward in instruction-following data generation. While it has some limitations, its strengths in autonomy, flexibility, and rigorous evaluation make it a valuable contribution to the field. The meta-decomposition strategy and the high quality of the generated data have the potential to influence the development of more capable and aligned LLMs.

**Score: 8.0**

**Rationale:**

DecIF demonstrates strong novelty and significance in its approach to instruction-following data generation. The methodology is well-reasoned and empirically supported with solid results. However, the reliance on LLM capabilities and the limited scope of the evaluation prevent it from achieving a higher score. The identified weaknesses provide clear directions for future research to address and strengthen the framework. As future studies can demonstrate the ability of this system to generate high-quality niche information or multi-step conversations, the impact of this paper would likely be increased.

- **Score**: 8/10

### **[Activation-Guided Consensus Merging for Large Language Models](http://arxiv.org/abs/2505.14009v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces Activation-Guided Consensus Merging (ACM), a novel plug-and-play model merging framework for large language models (LLMs).  ACM aims to integrate diverse capabilities of different LLMs by determining layer-specific merging coefficients based on the mutual information (MI) between activations of pre-trained and fine-tuned models.  The authors argue that existing merging methods often overlook the functional heterogeneity of neural components by assuming uniform importance across layers.  ACM assigns lower weights to layers with higher MI (greater similarity) to reduce redundancy, and higher weights to layers with lower MI (significant divergence) to preserve task-specific capabilities. Experimental results on Long-to-Short (L2S) reasoning and general merging tasks demonstrate ACM's superiority over existing baseline methods.  Specifically, they show reduced response length and improved reasoning accuracy.

**Critical Evaluation:**

*   **Novelty:** The paper's novelty lies in its activation-guided approach to model merging, specifically leveraging mutual information to determine layer-specific weights. While other activation-based methods exist, using MI to balance redundancy reduction and task-specific capability preservation offers a unique contribution. The idea of tying the merging coefficient to the functional role and activations of each layer is a good one, especially given that different layers have different roles (e.g., the lm_head layer for L2S).

*   **Significance:** The paper addresses an important problem: efficiently integrating diverse capabilities into a single LLM without requiring extensive training or facing instability issues.  The L2S framework is a valuable direction, and the ACM approach offers a practical and effective way to realize it. The experiments provide solid evidence of ACM's effectiveness on several benchmarks. The code release is also a big plus. The fact it can reduce redundancy (response length) while improving performance (accuracy) is a significant benefit, especially for practical deployments.

*   **Strengths:**

    *   Clear and well-defined methodology.
    *   Strong experimental results on L2S and general merging tasks.
    *   Plug-and-play nature allows easy integration with existing methods.
    *   Reduces computational redundancy while preserving task-specific capabilities.
    *   Code release promotes reproducibility.
    *   Addresses a relevant and timely problem in LLM research.

*   **Weaknesses:**

    *   The paper's evaluation, although thorough, could be extended to include larger models (70B+) if computational constraints permit. While the authors acknowledged this limitation, it's important to evaluate the scalability of ACM.
    *   The method focuses on model merging within the *same* architecture, and there is no discussion of *heterogeneous* architectures.
    *   More in-depth analysis of *why* MI is an effective measure for this task would strengthen the theoretical justification. Though its effectiveness has been empirically demonstrated, a more nuanced explanation of MI's connection to weight salience within this specific context would be beneficial.
    *   Discussion of the limitations surrounding responsible AI, fairness, and bias would further improve the paper.

*   **Potential Impact:**

    *   ACM has the potential to become a standard technique for model merging, especially in scenarios where efficiency and diverse capabilities are required.
    *   The method could inspire new research directions in activation-guided optimization and adaptive parameter weighting.
    *   The L2S framework and ACM combination could lead to more practical and efficient LLM deployments in real-world applications.

**Score:** 8

**Justification:**

ACM presents a novel and significant contribution to the field of model merging. It addresses a critical challenge of integrating diverse capabilities into LLMs while maintaining efficiency and stability. The experimental results provide strong evidence of its effectiveness and its plug-and-play nature makes it readily adoptable. Although limitations exist concerning larger model evaluation and theoretical depth, the paper makes a substantial advance that warrants a high score. This research provides a new perspective and practical solution to model merging.

- **Score**: 8/10

### **[Legal Rule Induction: Towards Generalizable Principle Discovery from Analogous Judicial Precedents](http://arxiv.org/abs/2505.14104v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Legal Rule Induction: Towards Generalizable Principle Discovery from Analogous Judicial Precedents":

**Summary:**

The paper introduces the Legal Rule Induction (LRI) task, which involves deriving concise, generalizable doctrinal rules from sets of analogous judicial precedents.  It formalizes the task, defines legal rules using a three-element structure (hypothetical condition, behavioral pattern, legal consequence), and presents a novel benchmark dataset. This dataset consists of two parts: LRI-AUTO, a large-scale dataset automatically curated from Chinese court judgments for model tuning, and LRI-GOLD, a meticulously curated test set annotated by legal experts.  The paper then evaluates the performance of various Large Language Models (LLMs) on this benchmark, highlighting the challenges they face (e.g., over-generalization, hallucination) and demonstrating that fine-tuning on LRI-AUTO can significantly improve LLMs' ability to capture nuanced rule patterns. It further explores different inductive reasoning pipelines, including Direct Induction, Chain-of-Thought (CoT), Long-CoT, and a novel iterative induction-verification pipeline called SILVER.

**Critical Evaluation:**

*   **Novelty:** The paper makes a substantial contribution by formally defining the Legal Rule Induction (LRI) task, which is a critical yet understudied area within computational law. Most existing research focuses on rule application, while rule discovery from precedents has been largely unexplored computationally. The paper's emphasis on *induction* rather than mere *deduction* or similarity matching is a significant departure from the norm and aligns well with actual legal reasoning. The formal definition and the three-element structure offer a valuable framework.
*   **Significance:** The creation and release of the LRI dataset (both LRI-AUTO and LRI-GOLD) are extremely valuable. The lack of suitable benchmarks has hindered progress in this area, and this dataset provides a much-needed resource for training and evaluating models. By using civil law precedents which often cite the statutes applied, the authors were able to more easily align cases to potential underlying rules, which in turn enabled the creation of a much larger training dataset than would otherwise be possible.
*   **Strengths:**
    *   **Clear Problem Formulation:** The paper provides a well-defined problem statement and clearly articulates the challenges involved in legal rule induction.
    *   **Rigorous Methodology:** The dataset creation process is described in detail, with clear explanations of the data sources, preprocessing steps, and annotation protocols. The use of both automated and expert-annotated datasets is a strong point.
    *   **Comprehensive Experiments:** The experiments are comprehensive, evaluating a range of LLMs and inductive reasoning pipelines.
    *   **Important Findings:** The findings are insightful, revealing the limitations of current LLMs in LRI and demonstrating the benefits of fine-tuning. The analysis of explicit vs. implicit rules provides a nuanced understanding of the task.
*   **Weaknesses:**
    *   **Limited Generalizability:** The dataset is based on Chinese law, which may limit the generalizability of the findings to other legal systems. The three-element structure may also require adaptation depending on jurisdiction.
    *   **Automated Metrics:** While the LLM-as-a-Judge approach is interesting, the reliability of using an LLM for evaluation should be considered with caution. Manual evaluation, while costly, is essential to truly validate the accuracy and usefulness of the induced rules. Although the authors did perform a manual quality audit of LRI-AUTO sets, they did not mention doing the same to evaluate the LLM judgments of the induced rules, which should have been the focus of the audit.
    *   **Complexity of Real-World Legal Reasoning:**  The three-element model of rules, while useful, is a simplification of the messy and often ambiguous nature of real-world legal reasoning.  There are many legal concepts and nuances beyond simple if/then logic that this framework would struggle to capture.

*   **Potential Influence:** The paper has the potential to significantly influence the field of computational law by:
    *   Encouraging further research on legal rule induction.
    *   Providing a benchmark for evaluating and comparing different approaches.
    *   Motivating the development of more sophisticated models capable of capturing the complexities of legal reasoning.

**Justification for Score:**

Considering the novelty, significance, strengths, and weaknesses, I assign this paper a score of **8**.

The paper's formalization of the LRI task and the creation of the LRI dataset represent a significant advance. The experimental results are compelling, and the findings offer valuable insights into the challenges and opportunities of using LLMs for legal rule induction. The limitations related to generalizability and automated evaluation are acknowledged and do not substantially detract from the paper's overall contribution. Addressing the limitations regarding automated evaluation metrics would increase the paper's impact and overall robustness. Despite these weaknesses, this paper offers a significant contribution to the field of AI and Law.

**Score: 8**

- **Score**: 8/10

### **[A Personalized Conversational Benchmark: Towards Simulating Personalized Conversations](http://arxiv.org/abs/2505.14106v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces PERSONACONVBENCH, a new large-scale benchmark designed for evaluating personalized reasoning and generation within multi-turn conversations using Large Language Models (LLMs). The benchmark uniquely integrates both personalization and conversational structure, offering three core tasks: sentence classification, impact regression, and user-centric text generation across 10 diverse Reddit-based domains. The authors benchmark several commercial and open-source LLMs and demonstrate that incorporating personalized conversational history leads to substantial performance improvements compared to non-personalized baselines. The benchmark and associated code are released to facilitate research on LLMs capable of adapting to individual conversational styles, tracking long-term context, and generating more contextually rich responses.

**Critical Evaluation:**

*   **Novelty:** The primary novelty lies in the *combined* focus on both personalization and multi-turn conversational structure within a unified benchmark. While personalization and conversational modeling are each independently active research areas, their tight integration in PERSONACONVBENCH addresses a significant gap. Existing personalization benchmarks often treat each user utterance as independent, and multi-turn conversation work is often user-agnostic. PERSONACONVBENCH directly tackles the challenge of modeling user behavior *within* the context of ongoing conversations. This is a step up from previous work on single-turn personalized tasks.

*   **Significance:** The significance stems from the benchmark's potential to drive progress in developing more human-like and engaging conversational AI systems. The release of a large-scale, diverse dataset with well-defined tasks will likely spur further research into:

    *   Adaptive LLMs that tailor responses to individual users' communication styles and preferences.
    *   Better models for long-term context tracking within conversations.
    *   More nuanced and personalized response generation.
    *   Analyzing the effects of different levels of user interactivity.

*   **Strengths:**

    *   **Comprehensive Benchmark:** The benchmark covers diverse Reddit domains, includes multiple evaluation paradigms (classification, regression, generation), and uses a unified prompting setup for fair comparisons.
    *   **Realistic Multi-User Setting:**  Moving beyond dyadic (user-agent) dialogues to a multi-user setting provides a more realistic and challenging environment for LLMs.
    *   **Clear Formalism:** The formal definitions of message representation, conversational graphs, trajectories, and user trajectory sets provide a solid framework for analysis.
    *   **Strong Empirical Results:** The experiments demonstrate a clear benefit from incorporating personalized conversational history across several models.
    *   **Open Source Release:**  Availability of the dataset and code promotes reproducibility and further research.

*   **Weaknesses:**

    *   **Reddit-centric Data:** The Reddit-based data may not generalize perfectly to other conversational settings (e.g., customer service, virtual assistants) due to differences in communication styles and user demographics. The specific dataset construction methods and filtering criteria could introduce biases. The reliance on upvotes as an impact metric, while common, is also subject to popularity bias.
    *   **Limited Task Diversity:**  While the core tasks are well-defined, expanding the range of tasks could make the benchmark even more comprehensive. For example, adding tasks focused on dialogue coherence, topic maintenance, or user intent understanding could be beneficial.
    *   **Limited LLM Investigation:** While several strong models were investigated, there are other powerful models that could have been tried as well.
    *   **No fine-tuning evaluation**. All evaluations were conducted in the zero-shot setting.

*   **Potential Influence:** PERSONACONVBENCH has the potential to become a standard benchmark for evaluating personalized conversational AI. It can guide the development of new algorithms and model architectures that effectively capture and leverage user-specific information within dynamic conversation contexts. The benchmark could also facilitate the analysis of existing LLMs' personalization capabilities and help identify areas for improvement.

* **Score:** 8

**Rationale:** The paper presents a solid contribution with a novel and useful benchmark that addresses an important gap in personalized conversational AI research. The thorough experimental results demonstrate the value of the benchmark and provide a strong foundation for future work. The main weaknesses relate to the limited domain (Reddit), limited task diversity, and the need to evaluate the impact of various prompt and fine-tuning techniques on performance. However, its strengths outweigh its weaknesses, solidifying its potential to significantly impact the field.

- **Score**: 8/10

### **[SlangDIT: Benchmarking LLMs in Interpretative Slang Translation](http://arxiv.org/abs/2505.14181v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "SlangDIT: Benchmarking LLMs in Interpretative Slang Translation":

**Summary:**

The paper introduces a new task called Interpretative Slang Translation (SlangDIT) to address the challenge of translating slang, which often relies on context-dependent semantic extensions. The task is designed to evaluate LLMs on three sub-tasks: slang detection, cross-lingual slang explanation, and slang translation within a given context. To facilitate research, the authors construct a SlangDIT dataset containing over 25,000 English-Chinese sentence pairs with slang terms and corresponding cross-lingual explanations. They also propose a deep thinking model named SlangOWL, which first identifies slang, judges its polysemy, analyzes its meaning, provides a cross-lingual explanation, and finally, translates the sentence. Experimental results on LLMs such as Qwen2.5 and Llama-3 show that the SlangOWL model significantly outperforms vanilla and supervised fine-tuned models.

**Critical Evaluation:**

*   **Novelty:**

    *   The paper's novelty lies primarily in introducing the SlangDIT task and associated dataset. While slang detection, explanation, and translation have been studied before, the integrated task and the focus on polysemous slang terms are relatively novel. This addresses a gap in the literature where the interdependence of these tasks is often overlooked.

    *   The SlangOWL model, while leveraging existing LLM backbones, introduces a deep thinking approach that emulates human cognitive processes in understanding and translating slang. This aspect also contributes to the novelty.

*   **Significance:**

    *   The SlangDIT dataset is a valuable resource for the community, providing a benchmark for evaluating LLMs on a challenging aspect of natural language understanding and translation. The dataset's size and the inclusion of polysemous slang terms enhance its utility.

    *   The SlangOWL model demonstrates the potential of incorporating deep thinking approaches into LLMs for improving the accuracy of slang translation. This has implications for developing more robust and context-aware machine translation systems.

*   **Strengths:**

    *   The paper clearly defines the problem and motivates the need for a dedicated benchmark for interpretative slang translation.

    *   The dataset creation process is well-described, and the authors take steps to ensure data quality through multiple rounds of annotation and validation.

    *   The SlangOWL model is well-designed and demonstrates significant performance improvements over baseline models.

    *   The experiments are comprehensive, and the results are presented clearly.

*   **Weaknesses:**

    *   The dataset is limited to English-Chinese language pairs. Expanding the dataset to include other languages would increase its broader impact.
    *   The study primarily focuses on leveraging existing LLM backbones. Exploring novel architectures or training techniques specifically tailored for slang translation could further enhance the research.
    *   The SlangOWL model, while effective, may be computationally expensive due to the deep thinking approach. Further optimization could improve its efficiency.
    *   The evaluation, while using standard metrics, still relies heavily on reference-based metrics. Exploring human evaluation or task-specific metrics could provide a more nuanced assessment of translation quality.

*   **Potential Influence:**

    *   The SlangDIT task and dataset are likely to stimulate further research in the area of slang understanding and translation, encouraging the development of more sophisticated models.

    *   The SlangOWL model's deep thinking approach could inspire other researchers to explore similar techniques for addressing other challenging aspects of natural language processing.

    *   The work could contribute to the development of more robust and culturally sensitive machine translation systems that are better equipped to handle the nuances of slang.

**Score: 8**

**Justification:**

The paper makes a significant contribution by introducing a novel task (SlangDIT) and a well-constructed dataset that addresses a real-world challenge in machine translation. The SlangOWL model demonstrates the effectiveness of deep thinking approaches in improving slang translation accuracy. While there are some limitations (dataset language scope, computational cost), the work is well-executed, clearly presented, and has the potential to influence future research in the field. The score of 8 reflects the paper's novelty, significance, and potential influence, balanced against the limitations and areas for future improvement.

- **Score**: 8/10

### **[ABBA: Highly Expressive Hadamard Product Adaptation for Large Language Models](http://arxiv.org/abs/2505.14238v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces ABBA, a new parameter-efficient fine-tuning (PEFT) architecture for large language models (LLMs). ABBA reparameterizes the weight update as a Hadamard product of two independently learnable low-rank matrices. This contrasts with methods like LoRA which uses a single low-rank decomposition and HiRA which modulates pre-trained weights with a low-rank update.  ABBA aims to improve expressivity under a similar parameter budget by decoupling the update from the pre-trained weights. The authors provide theoretical and empirical analyses to support their approach, including a matrix reconstruction task, toy MNIST experiment, and benchmarking on arithmetic and commonsense reasoning tasks, showing superior performance compared to existing PEFT methods. They also provide an efficient implementation using Khatri-Rao factorization.

**Critical Evaluation:**

*   **Novelty:** The core idea of using a Hadamard product of *two independently learnable* low-rank matrices for weight updates is a significant departure from existing PEFT approaches, particularly LoRA and HiRA. While HiRA uses a Hadamard product, its expressivity is still constrained by its dependence on the pre-trained weights. ABBA's complete decoupling allows for more flexible optimization. The Khatri-Rao factorization for efficient implementation is also a notable contribution.

*   **Significance:** Parameter-efficient fine-tuning is a crucial area for making LLMs more accessible and adaptable. ABBA's improved expressivity and demonstrated performance gains across a variety of tasks, with results rivaling full fine-tuning in some instances, suggest it could become a prominent method in the PEFT landscape. The rigorous theoretical and empirical validation strengthens the paper's significance. The exact analytical expression derived is valuable.

*   **Strengths:**
    *   **Clear Problem Definition:**  The paper clearly articulates the limitations of existing low-rank methods like LoRA in terms of expressivity.
    *   **Well-Motivated Approach:**  The ABBA architecture is logically derived from the limitations of LoRA and HiRA.
    *   **Theoretical Analysis:**  The formal analysis of expressivity, and the proofs regarding scaling factors and rank-stability, provide strong theoretical grounding for the approach.
    *   **Empirical Validation:**  Extensive experiments on multiple models and datasets demonstrate consistent and significant performance gains over existing PEFT methods. The inclusion of matrix reconstruction and MNIST task helps illuminate the mechanisms underlying ABBA's performance.
    *   **Efficient Implementation:** The Khatri-Rao factorization provides a practical and scalable implementation of ABBA.
    *   **Ablation Studies and Hyperparameter Analysis:** Strong experimentation to assess all the components of ABBA (such as the initialization and choice of scaling factor).

*   **Weaknesses:**
    *   **Complexity:** The Hadamard product and Khatri-Rao factorization might add a bit of complexity to the implementation and understanding compared to LoRA's simplicity. However, their approach is simple enough to be implementable, and there are plenty of available and accessible resources to implement Hadamard products/Khatri-Rao factorizations.
    *   **Limited Exploration of Adapter Composition:** The adapter chaining experiment yielded negative results, and further exploration of this direction may be needed.

*   **Potential Impact:**  ABBA has the potential to influence the future of PEFT research by demonstrating the benefits of decoupling weight updates and leveraging Hadamard products. The improved performance and efficient implementation could lead to wider adoption of ABBA in practical applications.

**Score: 8**

**Justification:**  ABBA presents a novel and well-justified approach to parameter-efficient fine-tuning that overcomes limitations of existing methods. The theoretical analysis and extensive empirical validation provides strong support for its efficacy. However, while ABBA introduces architectural complexity, the performance and ablation studies prove that the additional complexity is well worth it. The potential for impact in the PEFT field is substantial. Future work could explore more deeply how ABBA’s performance varies across different tasks/datasets or consider adapter chaining and other techniques with ABBA.

- **Score**: 8/10

### **[Cross-Lingual Optimization for Language Transfer in Large Language Models](http://arxiv.org/abs/2505.14297v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces Cross-Lingual Optimization (CLO), a method for transferring the knowledge of English-centric Large Language Models (LLMs) to other languages while preserving English performance. CLO uses publicly available English Supervised Fine-Tuning (SFT) data and a translation model to create parallel data in the target language. It then modifies the Direct Preference Optimization (DPO) loss to encourage the model to respond in the same language as the input query.  Experiments across six languages (Chinese, German, Korean, Indonesian, Swahili, and Yoruba) with varying resource availability demonstrate that CLO outperforms standard SFT in both acquiring target language proficiency and maintaining English performance, especially in low-resource settings. The paper explores the impact of data size, different layers of fine-tuning, and the effect of different loss functions.

**Critical Evaluation:**

* **Novelty:** The core idea of using a translated dataset and a modified DPO loss to explicitly encourage responses in the target language is a valuable contribution. While prior work has explored cross-lingual transfer and instruction tuning, CLO presents a focused, efficient, and seemingly effective method to address the common issue of diminished English capabilities and poor performance in low-resource languages after SFT. The explicit language correspondence within a batch through the modified DPO objective is a novel and promising aspect.

* **Significance:** The significance lies in addressing a practical problem: adapting LLMs to more languages without sacrificing performance in the language they were primarily trained on. The findings demonstrate that CLO can achieve better results with less data compared to standard SFT, which is particularly important for low-resource languages where data scarcity is a major bottleneck. The ablation studies provide further insights into the effectiveness of various components of the CLO method. The consistent performance gains observed across several models and languages strengthen the impact of the results.

* **Strengths:**
    *   **Clear Problem Definition:** The paper clearly identifies the challenges in cross-lingual transfer and motivates the need for a method like CLO.
    *   **Well-Designed Method:** CLO is well-explained and relatively simple to implement, making it accessible to other researchers.
    *   **Comprehensive Evaluation:** The experiments are thorough, covering multiple models, languages, and evaluation metrics.
    *   **Ablation Studies:** The ablation studies help to understand the importance of different aspects of CLO.

* **Weaknesses:**
    *   **Reliance on Translation Models:** The method relies on the quality of the translation model. While the paper acknowledges this, a more in-depth analysis of the impact of translation quality on CLO's performance would be beneficial. There may be scenarios where the translated data is significantly noisy/biased, affecting performance.
    *   **Scope of Languages:** Although six languages is a reasonable starting point, the paper acknowledges this to be a limitation. Further study across a broader and more diverse set of languages would strengthen the generalizability of the findings.
    *   **Limited exploration beyond DPO**: While applying CLO to DPO is valid, the authors mention that it potentially extends to different preference optimization algorithms. Verifying this, with at least one more popular algorithm, would considerably boost the confidence in CLO's applicability.
    *   **Limited evaluation on cultural relevance**: The paper acknowledges in a brief discussion on one of its limitations the lack of cultural awareness in the test data. Expanding on how CLO impacts language specific nuances and cultural relevance would strengthen the overall arguments.

*   **Potential Influence:** The CLO method can potentially influence how LLMs are adapted to new languages, particularly in resource-constrained scenarios. The findings encourage further research into methods that leverage existing language knowledge for efficient transfer. It could also lead to the development of more robust and versatile LLMs that can perform well across a wider range of languages.

**Justification of Score:**

CLO is a well-designed and empirically validated method for cross-lingual transfer. It directly addresses the problem of adapting LLMs to other languages efficiently and effectively. While it suffers from potential reliance on translation quality, the experimental results across several languages and models show consistent gains. It opens promising new avenues for future research in cross-lingual transfer. Given the practical significance of the problem and the solid contributions of the method, the findings of this paper deserve recognition.

Score: 8

- **Score**: 8/10

### **[Exploring Jailbreak Attacks on LLMs through Intent Concealment and Diversion](http://arxiv.org/abs/2505.14316v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Exploring Jailbreak Attacks on LLMs through Intent Concealment and Diversion":

**Summary:**

The paper introduces ICE (Intent Concealment and divErsion), a novel black-box jailbreak attack method designed to circumvent safety constraints in Large Language Models (LLMs). ICE works by decomposing malicious queries into hierarchical fragments (Hierarchical Split) and augmenting them with semantically related terms (Semantic Expansion) before reassembling them within a structured reasoning task (Reasoning Mask). This obfuscates the intent of the attack. The authors also present BiSceneEval, a new dataset designed for evaluating LLM robustness across question-answering and text generation tasks. Experiments on several popular LLMs show that ICE achieves high attack success rates (ASR) with single-query efficiency and good transferability, outperforming existing jailbreak techniques. The authors argue for a hybrid defense strategy combining static safety mechanisms with real-time semantic decomposition.

**Critical Evaluation:**

**Strengths:**

*   **Novel Approach:** The ICE method is innovative in its intent concealment and diversion strategy. It moves beyond simple prompt manipulation to a more structured approach that leverages LLMs' perceived weaknesses in multi-step reasoning. The combination of Hierarchical Split and Semantic Expansion offers a distinct methodology compared to existing prompt injection or adversarial example-based attacks.
*   **Efficiency and Transferability:**  The single-query efficiency of ICE is a major strength.  Many jailbreak methods require iterative queries, making them resource-intensive. ICE's ability to achieve high ASR with a single query makes it more practical and potentially more dangerous. The reported transferability across different LLM architectures is significant, suggesting a more fundamental vulnerability related to reasoning rather than model-specific quirks.
*   **Comprehensive Evaluation Dataset:** The BiSceneEval dataset addresses a key gap in existing evaluation datasets. By including both question-answering and text generation tasks and focusing on both pre- and post-inference defenses, it provides a more comprehensive assessment of LLM robustness in jailbreaking scenarios. The careful construction of the dataset, including steps for deduplication and expert annotation, enhances its reliability.
*   **Clear Problem Definition and Well-Structured Solution:** The paper clearly articulates the limitations of existing jailbreak methods and evaluation datasets and presents ICE and BiSceneEval as solutions to these limitations. The method is described in detail, making it reproducible and allowing for further research in this area.
*   **Strong Experimental Results:** The experimental results demonstrate the effectiveness of ICE in comparison to existing jailbreak techniques. The ASR numbers are compelling, and the inclusion of TCPS (Time Cost Per Sample) provides a valuable metric for evaluating the efficiency of jailbreak methods.

**Weaknesses:**

*   **Limited Analysis of Failure Cases:** While the paper highlights the successes of ICE, it could benefit from a more in-depth analysis of *why* it fails in certain cases. Understanding the limitations of the technique could help in developing more robust attacks and defenses. The limited coverage on new reasoning-enhanced LLMs (such as GPT-01) could be considered a weakness.
*   **Black-box Focus:** While the black-box nature of the attack is a strength in some ways (representing a real-world attack scenario), it also limits the insights into *how* the LLM's internal mechanisms are being bypassed. White-box or grey-box analysis could provide valuable information for developing more effective defenses.
*   **Subjectivity in Evaluation Metrics:** Even with the hybrid-ASR approach, there is a degree of subjectivity in determining whether a response is truly harmful. Reliance on keywords and automated classification can lead to false positives and negatives. Human evaluation, while included, could be expanded.
*   **Limited Discussion of Potential Defenses:** While the paper briefly mentions a hybrid defense strategy, it could benefit from a more detailed discussion of specific defense mechanisms that could effectively mitigate ICE attacks. Exploring potential countermeasures would further enhance the practical value of the research.

**Significance and Novelty:**

The paper makes a significant contribution by presenting a new, efficient, and transferable jailbreak method. It highlights the need for a more nuanced understanding of LLM vulnerabilities and motivates the development of more robust defense strategies. BiSceneEval is a valuable contribution to the research community, providing a more comprehensive benchmark for evaluating LLM security. The single query efficiency significantly surpasses other methods, and provides more compelling evidence of the reasoning vulnerabilities in LLMs.

**Justification for Score:**

The paper offers a novel approach to jailbreaking LLMs with a clear problem definition, a well-structured solution, and strong experimental results. While there's room for improvement in the analysis of failure cases, black-box analysis, subjectivity of metrics, and defense strategies, the paper addresses a critical issue in LLM security and presents compelling evidence of its effectiveness.

Score: 8

- **Score**: 8/10

### **[Vid2World: Crafting Video Diffusion Models to Interactive World Models](http://arxiv.org/abs/2505.14357v1)**
- **Summary**: Here's a concise summary and critical evaluation of the paper:

**Summary:**

The paper "Vid2World: Crafting Video Diffusion Models to Interactive World Models" introduces a method to adapt pre-trained video diffusion models into interactive world models.  The approach, called Vid2World, addresses two key challenges: 1) enabling causal generation by transforming the architecture and training objective to support autoregressive generation, and 2) implementing causal action guidance to allow frame-level action conditioning. The method is evaluated in robot manipulation and game simulation environments, demonstrating improved performance over existing transfer methods and state-of-the-art world models.

**Critical Evaluation:**

*   **Novelty:** The paper's primary novelty lies in systematically addressing the challenges of transferring powerful, pre-trained video diffusion models to the interactive world modeling domain. The authors identify the critical issues of causal generation and action conditioning, which are often overlooked in other approaches. The proposed solutions, including the mixed weight transfer for temporal convolution layers and the causal action guidance using action dropout, offer practical techniques to address these challenges. While individual components like attention masking or action injection have been explored before in other contexts, their integration specifically for adapting video diffusion models into *interactive* world models is a novel contribution.

*   **Significance:** Interactive world models are crucial for advancing AI in robotics, simulation, and reinforcement learning, as they allow agents to predict and plan in dynamic environments.  Leveraging the knowledge embedded in large-scale, pre-trained video diffusion models could significantly improve the data efficiency and fidelity of these world models. The Vid2World method provides a pathway to achieve this, potentially accelerating research in areas requiring complex physical reasoning. The experiments are well-designed, using standard benchmark datasets and evaluating both video prediction quality and downstream policy performance.

*   **Strengths:**
    *   Clear problem definition and motivation.
    *   Well-explained technical approach.
    *   Comprehensive experimental evaluation across multiple domains.
    *   Strong quantitative results demonstrating improvements over baselines.
    *   Qualitative examples illustrating the strengths and limitations of the method.

*   **Weaknesses:**
    *   The improvements in CS:GO seem less substantial than the improvements on RT-1. More analysis as to why would be valuable.
    *   The base model (DynamiCrafter) is already quite complex.  The impact of Vid2World on a simpler diffusion model would be useful.
    *   The computational cost of training and inference, especially with a large pre-trained model, is not explicitly discussed.

*   **Potential Influence:** Vid2World has the potential to influence research in world modeling, reinforcement learning, and video generation. The framework could serve as a foundation for future methods that aim to transfer knowledge from large-scale video datasets to interactive agents.  The proposed techniques for causalization and action conditioning could be generalized to other generative models and tasks.

**Justification for Score:**

While the paper builds upon existing techniques, it demonstrates a clear understanding of the challenges and offers well-engineered solutions. The experimental results convincingly show the effectiveness of Vid2World in enabling interactive world modeling. The paper's novelty, coupled with its potential to accelerate research in related fields, justifies a strong, but not perfect, score. The limitations mentioned above, while not detracting significantly from the overall contribution, prevent it from reaching the highest levels of impact.

Score: 8

- **Score**: 8/10

### **[Log-Augmented Generation: Scaling Test-Time Reasoning with Reusable Computation](http://arxiv.org/abs/2505.14398v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces Log-Augmented Generation (LAG), a framework designed to improve the reasoning capabilities of Large Language Models (LLMs) by enabling them to reuse past computations and reasoning steps. LAG achieves this by storing task logs in a key-value (KV) cache format, representing the full reasoning context while storing only a subset of tokens.  When a new task arises, LAG retrieves relevant KV values from the logs and augments the LLM's generation process. The approach differs from reflection-based methods by directly reusing prior reasoning and from existing KV caching techniques which mainly target efficiency. Experiments on knowledge- and reasoning-intensive datasets demonstrate that LAG outperforms standard agentic systems, reflection methods, and KV caching baselines.

**Critical Evaluation:**

*   **Novelty:** The idea of reusing reasoning steps is not entirely new, as prior work in reflection and case-based reasoning has explored similar concepts. However, LAG's use of KV caches to represent and reuse reasoning traces *is* a novel and practical implementation.  The insight that KV values encapsulate the entire surrounding context is well leveraged. The paper also clearly distinguishes LAG from existing KV caching techniques, which primarily focus on efficiency gains, whereas LAG prioritizes accuracy improvements via contextual information retention. The positional embedding manipulation for log integration is also a significant technical detail.

*   **Significance:**  The paper provides empirical evidence demonstrating that LAG can significantly improve both the accuracy and efficiency of LLMs on complex reasoning tasks. The gains over strong baselines, including ReAct and reflection methods, suggest that LAG has the potential to be a valuable tool for enhancing the problem-solving capabilities of LLMs. The fact that the improvement is seen across different datasets (knowledge and reasoning intensive) strengthens the argument. Also, making it easier to use and integrate with existing LLM workflows is a practical benefit. It contributes meaningfully to the field by showing a practical way to improve the ability of LLMs to remember and reuse reasoning from prior tasks, which is a major limitation in LLMs today.

*   **Strengths:**
    *   **Clear Problem Definition:** The paper clearly identifies the problem of LLMs' inability to reuse past reasoning.
    *   **Novel Approach:** The KV cache implementation for representing and reusing reasoning traces is well-motivated and technically sound.
    *   **Strong Empirical Results:**  The paper presents comprehensive experimental results on diverse datasets, demonstrating significant improvements over strong baselines in both accuracy and efficiency.
    *   **Ablation Studies and Analysis:**  The paper includes ablation studies to analyze the impact of different design choices, such as the number of logs retrieved and the type of tokens used for KV caching.
    *   **Reproducibility:** The code and data availability enhance the reproducibility and adoption of the proposed method.

*   **Weaknesses:**
    *   **Log Selection:** The log retrieval process relies on semantic similarity.  While this is reasonable, it could be vulnerable to retrieving noisy or irrelevant logs. A more sophisticated retrieval mechanism may further improve performance. There is also limited description on how to optimize the static log store (what tasks should be run in advance to generate logs).
    *   **Scalability Concerns:** While the paper addresses storage efficiency, the retrieval of KV caches and the re-application of positional embeddings might introduce computational overhead. A more detailed analysis of the runtime performance would be beneficial.
    *   **Generalizability:**  The experiments are limited to specific datasets. Further validation on a wider range of tasks and domains would strengthen the generalizability of the findings.
    *   **Static Log Store:** The use of a fixed offline constructed log store restricts the potential of adapting to ongoing changes during online utilization.

*   **Potential Influence:**  The paper's findings could influence future research on memory mechanisms for LLMs, particularly in the development of more effective and efficient methods for storing and retrieving past reasoning experiences.  The KV cache approach could be integrated into other LLM frameworks to enhance their problem-solving abilities. It also provides insights into the importance of retaining contextual information for effective reasoning.

**Justification for Score:**

While the idea of reusing reasoning steps isn't groundbreaking, LAG presents a novel *implementation* using KV caches that achieves impressive results. The clear demonstration of improved accuracy and efficiency across multiple datasets justifies a high score. The weaknesses (scalability, static logs, and dataset limitations) temper the enthusiasm slightly, but the overall contribution is significant and promising.

Score: 8

- **Score**: 8/10

### **[ViC-Bench: Benchmarking Visual-Interleaved Chain-of-Thought Capability in MLLMs with Free-Style Intermediate State Representations](http://arxiv.org/abs/2505.14404v1)**
- **Summary**: Here is a concise summary and a rigorous evaluation of the paper "ViC-Bench: Benchmarking Visual-Interleaved Chain-of-Thought Capability in MLLMs with Free-Style Intermediate State Representations."

**Summary:**

This paper introduces ViC-Bench, a new benchmark designed to evaluate the Visual-Interleaved Chain-of-Thought (VI-CoT) reasoning capabilities of Multi-Modal Large Language Models (MLLMs).  ViC-Bench consists of four diverse tasks: maze navigation, jigsaw puzzle, embodied long-horizon planning, and complex counting. A key feature of the benchmark is the support for free-style Intermediate Visual States (IVS), allowing for more natural and flexible reasoning processes. The paper also proposes a comprehensive three-stage evaluation strategy with targeted metrics and an Incremental Prompting Information Injection (IPII) strategy for ablative analysis of prompting factors.  The authors extensively evaluate 18 state-of-the-art MLLMs on ViC-Bench and present key insights into their VI-CoT capabilities.

**Critical Evaluation:**

*   **Novelty:**

    *   The primary novelty lies in the *creation of ViC-Bench itself*.  The benchmark distinguishes itself from existing benchmarks by focusing on free-style IVS representations, which are argued to be more representative of human-like reasoning. This is a valid point, as previous benchmarks often provide fixed or constrained IVS, potentially skewing the evaluation.

    *   The *three-stage evaluation strategy* is another novel aspect, designed to progressively assess VI-CoT capabilities, starting with simple tasks and gradually increasing complexity. The introduction of metrics like Recall and ThinkGain, specifically tailored for VI-CoT, represents a further contribution to the evaluation methodology.

    *   The *IPII strategy* for ablative prompting factor analysis is also valuable, allowing the authors to systematically investigate the influence of different prompting levels on VI-CoT performance.

*   **Significance:**

    *   The development of ViC-Bench is *highly significant* for the field of MLLMs.  The availability of a standardized benchmark specifically designed to assess VI-CoT capabilities fills a gap in the existing evaluation landscape. This should facilitate more focused research and development in this area.

    *   The *extensive evaluation of 18 SOTA MLLMs* provides valuable insights into the current state-of-the-art. The comparative analysis helps to identify strengths and weaknesses of different models and provides a roadmap for future improvements.  The authors point out performance gaps between different MLLMs.

    *   The *insights gained from the experiments and analysis* contribute to a better understanding of the factors that influence VI-CoT performance.  The finding that free-style IVS can enhance performance for MLLMs with strong reasoning priors, but might confuse weaker models, is particularly relevant.

*   **Strengths:**

    *   The paper is well-written, clearly structured, and provides sufficient details about the benchmark construction, evaluation methodology, and experimental setup.

    *   The authors address a critical gap in the MLLM evaluation landscape by focusing on free-style IVS, which is more aligned with human cognition.

    *   The comprehensive evaluation of a large number of MLLMs and the ablative analysis of prompting factors provide valuable insights for the research community.

*   **Weaknesses:**

    *   The IVS generation is not completely automated and relies on function calls and potentially human assistance for some tasks (e.g., embodied long-horizon planning). While this allows for free-style IVS, it raises concerns about scalability and potential bias.

    *   While the authors introduce new metrics like ThinkGain and Legality, further explanation and justification of their mathematical formulations and their implications could be beneficial.

    *   The results of the IPII strategy are not clearly displayed and are not implemented across all benchmarks.

*   **Potential Influence:**

    *   ViC-Bench has the potential to become a widely adopted benchmark for evaluating VI-CoT capabilities in MLLMs, driving progress in the field.

    *   The proposed evaluation methodology and the identified insights can guide the development of more effective VI-CoT techniques and improved training strategies for MLLMs.

    *   The benchmark can also serve as a foundation for developing more sophisticated multi-modal agents and embodied AI systems.

**Score: 8**

**Rationale:**

The paper presents a significant contribution by introducing a novel and well-designed benchmark (ViC-Bench) for evaluating VI-CoT capabilities in MLLMs with free-style IVS. The extensive evaluation of SOTA models and the ablative prompting study provide valuable insights. Although there are some limitations regarding the automation of IVS generation and clarity of proposed metrics, the strengths of the paper, particularly its novelty, significance, and potential impact on the field, outweigh the weaknesses. This is a well done, highly valuable paper deserving a score of 8.

- **Score**: 8/10

### **[Creative Preference Optimization](http://arxiv.org/abs/2505.14442v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces Creative Preference Optimization (CRPO), a new method for improving the creativity of Large Language Models (LLMs).  CRPO enhances Direct Preference Optimization (DPO) by incorporating signals from multiple dimensions of creativity: novelty, diversity, surprise, and quality. The method uses a weighted combination of metrics related to these dimensions within the DPO loss function. The authors created a new large-scale human preference dataset called MUCE (Multitask Creativity Evaluation) containing over 200,000 human-generated responses and ratings across 30+ psychological creativity assessments to train and evaluate their models. Experiments show that LLMs trained with CRPO outperform strong baselines, including GPT-4o, on automated metrics and human evaluations, producing more novel, diverse, and surprising outputs while maintaining high quality. They also test generalization on NOVELTYBENCH.

**Critical Evaluation:**

*   **Novelty:** The core idea of modularly injecting multiple creativity dimensions into a preference optimization objective is a valuable contribution. This is a more comprehensive approach than simply focusing on diversity or single-task optimization, which were limitations of previous methods.  The development of MUCE, a large-scale, diverse, and psychologically-grounded creativity dataset, is also a significant contribution to the field. Its scale and scope far exceed those used in many prior studies and address the limitation of using tasks that may not align with established psychological theories of creativity.

*   **Significance:**  The paper tackles a crucial problem: LLMs often lack true creativity, producing outputs that are unoriginal, predictable, and fail to surprise.  By directly optimizing for creativity within a preference learning framework, the authors propose a promising path forward. The results demonstrate that CRPO can enhance the creative capabilities of LLMs without sacrificing output quality.  The increased performance on both automated metrics and human evaluations, as well as generalization to an external benchmark, further solidifies the significance of the work.  The MUCE dataset will also likely serve as a valuable resource for future research in this area, facilitating the development of more creative and human-like LLMs.

*   **Strengths:**

    *   The modular design of CRPO is a key strength, allowing researchers to fine-tune the model for specific creativity dimensions.
    *   The extensive experimental evaluation using both automated metrics and human evaluations is thorough and convincing.
    *   The construction of the MUCE dataset addresses a critical need for high-quality, large-scale creativity data.
    *   The performance gains are demonstrated on more than one base model providing some evidence for generalizability.

*   **Weaknesses:**

    *   The reliance on a fixed set of creativity metrics may limit the scope of CRPO. The definitions of novelty, surprise, etc., are operationalized by particular mathematical forms, and other formulations could yield different results. How stable and robust the results are to changes in the operationalization of the individual creativity dimensions should be further explored.
    *   The experiments primarily focus on the English subset of MUCE, leaving the multilingual generalization of CRPO largely unexplored.
    *   While the ablation experiments show the importance of individual creativity dimensions, they don't fully explore the interactions between these dimensions. Deeper analysis of the tradeoffs and synergies might provide further insights.
    *   Ethical considerations around “malevolent creativity” are acknowledged but not fully addressed. The risk of models generating unsafe or toxic content even when not explicitly prompted is concerning.

*   **Potential Impact:** This research has the potential to significantly advance the field of AI creativity, paving the way for more creative and human-like LLMs. It also provides a practical methodology and dataset that can be used by other researchers to develop and evaluate new creativity-enhancing techniques.  The work could have applications in diverse areas, such as content generation, problem solving, and design.

**Rigorous Rationale for the Score:**

The paper makes a strong contribution to the field by directly addressing the challenge of enhancing creativity in LLMs. CRPO's modular approach and the construction of the MUCE dataset, which are both major investments, are significant strengths. While the weaknesses outlined above are worth noting, they do not overshadow the novelty and significance of the work.

Score: 8

- **Score**: 8/10

### **[Reasoning Models Better Express Their Confidence](http://arxiv.org/abs/2505.14489v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper investigates the confidence calibration of reasoning models (LLMs employing chain-of-thought (CoT) reasoning) compared to their non-reasoning counterparts.  The central finding is that reasoning models exhibit superior confidence calibration. This improvement is attributed to the "slow thinking" behaviors inherent in CoT, such as exploring alternative approaches and backtracking, which allow dynamic adjustments to confidence during the reasoning process. The study benchmarks models across various datasets and performs ablation studies to isolate the impact of slow thinking, even demonstrating that non-reasoning models benefit from being guided towards slow thinking via in-context learning.

**Critical Evaluation:**

*   **Novelty:** The paper's primary novelty lies in its systematic exploration of the relationship between reasoning (specifically CoT) and confidence calibration in LLMs. While prior work has addressed confidence in LLMs and CoT reasoning separately, this work connects the two in a rigorous, empirical manner.  It offers a novel perspective on why CoT might be beneficial beyond just task accuracy.  The discovery that slow thinking behaviors, rather than simply longer CoTs, are key to improved calibration is also a novel and valuable contribution. The ablation studies that analyze various reasoning behaviors are carefully designed, the finding that the *structure* of the CoT reasoning improves calibration is very helpful.
*   **Significance:** The findings are significant for several reasons. First, it provides a compelling justification for using reasoning models in scenarios where reliable confidence estimates are crucial (e.g., high-stakes decision-making). Second, it identifies specific CoT behaviors (exploring alternatives, backtracking) that contribute to better calibration, offering insights for improving LLM design and prompting strategies.  Third, the study addresses a critical weakness of LLMs – their tendency to be overconfident – and proposes a path toward more trustworthy and reliable models. The work has practical implications for how LLMs are deployed and trusted in real-world applications. The observation that non-reasoning models also benefit from slow thinking, when prompted through in-context examples, expands the applicability of findings.
*   **Strengths:**
    *   **Comprehensive Benchmarking:**  The paper features extensive benchmarking across a diverse set of models and datasets, enhancing the generalizability of the findings.
    *   **Rigorous Analysis:** The ablation studies are well-designed and executed, providing strong evidence for the role of slow thinking behaviors. The careful observation in their controlled experiment conditions allowed the conclusion of the importance of non-linear components.
    *   **Clear Writing and Presentation:** The paper is well-written and easy to follow, with clear explanations of the experimental setup, results, and analysis.
    *   **Reproducibility:** The authors provide code, aiding reproducibility.
*   **Weaknesses:**
    *   **Scale Limitations:** While the study uses 32B models, which is reasonable, the experiments may need to be re-evaluated as LLMs scale to trillions of parameters and are trained on more diverse data. Generalizability needs more consideration.
    *   **Instruction limitations:** There is limited diversity in instruction, which suggests findings in this paper may need to be replicated with larger prompt libraries.
    *   **Dataset Bias:** The datasets used in the experiments could introduce bias into the evaluation. A study on more diverse and real-world datasets could strengthen the claims.

**Justification of Score:**

Overall, the paper makes a valuable and novel contribution to the field of LLMs by establishing a clear link between reasoning and confidence calibration.  The rigorous analysis and well-supported conclusions provide valuable insights for improving the trustworthiness and reliability of LLMs.  The weaknesses identified above are primarily limitations related to scope and are not fundamental flaws. The findings have important implications for the development and deployment of LLMs in real-world applications.

Score: 8

- **Score**: 8/10

### **[SparC: Sparse Representation and Construction for High-Resolution 3D Shapes Modeling](http://arxiv.org/abs/2505.14521v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "SparC: Sparse Representation and Construction for High-Resolution 3D Shapes Modeling":

**Summary:**

The paper introduces SparC, a novel framework for high-resolution 3D shape modeling. It addresses limitations in existing two-stage pipelines (VAE + diffusion) that often suffer from detail loss due to inefficient representations and modality mismatches. SparC combines a sparse deformable marching cubes representation called SparseCubes with a novel sparse convolutional VAE (SparConv-VAE). SparseCubes converts raw, potentially non-watertight meshes into watertight surfaces at high resolution by scattering signed distance and deformation fields onto a sparse cube. SparConv-VAE is a modality-consistent VAE built entirely on sparse convolutional networks, enabling efficient and near-lossless 3D reconstruction. The framework achieves state-of-the-art reconstruction fidelity, preserves fine details, reduces training costs, and integrates well with latent diffusion models for scalable, high-resolution 3D generation. Key contributions include the SparCubes representation, the SparConv-VAE architecture, and their combination into a unified framework.

**Critical Evaluation:**

* **Novelty:** The paper presents several novel aspects:
    *   **SparseCubes:** This sparse deformable marching cubes representation is a significant improvement over dense volumetric grids.  It provides a fast and near-lossless method for creating watertight meshes from raw data, something that has been a bottleneck in previous pipelines.
    *   **SparConv-VAE:** The modality-consistent sparse convolutional VAE is also novel. By working directly on the sparse representation and avoiding modality mismatches, it achieves better reconstruction performance with a lighter architecture compared to methods that use global attention or rely on 2D supervision.
    *   **Unified Framework:** The seamless integration of SparCubes and SparConv-VAE into a single framework is a valuable contribution.

* **Significance:** The paper addresses a critical challenge in 3D shape modeling: achieving high-resolution, detailed reconstructions and generations while maintaining efficiency. By tackling the representation and modality mismatch issues, SparC enables better results with reduced computational cost. The fact that SparC integrates well with existing latent diffusion pipelines further amplifies its potential impact. The near-lossless remeshing and reconstruction capabilities could have far-reaching implications for areas like 3D printing, AR/VR asset creation, and robotics simulations.

* **Strengths:**
    *   **State-of-the-art results:** The paper demonstrates superior reconstruction fidelity compared to previous methods on challenging datasets.
    *   **Efficiency:**  SparC reduces training costs and improves conversion speeds.
    *   **Watertight Meshes:** The framework produces watertight meshes suitable for real-world applications.
    *   **Integration with diffusion models:**  The ability to enhance existing latent diffusion models is a major advantage.
    *   **Modality consistency:** A key design consideration that resolves issues that existed in previous VAE designs.

* **Weaknesses:**
    *   **Texture Handling:** The method discards original texture information, which is a limitation for applications where texture is crucial. The paper mentions this limitation. This is a common issue in geometry-focused approaches, but it's still a noteworthy drawback.
    *   **Internal Structure Removal:** The method removes internal geometry when processing closed meshes. While this helps generate watertight surfaces, it could be a problem for cases where internal details are important. The paper acknowledges this limitation as well.
    *   **Dependence on VAE Training Data:** The performance hinges on the quality and diversity of the VAE training dataset. While the paper utilizes a large dataset, domain shift issues could arise when dealing with significantly different types of 3D shapes. This is a general weakness of data-driven approaches.

* **Potential Impact:** The framework presents an important improvement to the process of 3D generation, promising a path towards scalable high-resolution 3D asset creation. The modular design facilitates the easy integrations within existing diffusion model pipelines. The ability to generate water-tight meshes, with the capacity for 3D printing makes SparC applicable in various scenarios.

**Justification for Score:**

Considering the novelty, significance, strengths, and weaknesses of the paper, a score of 8 is appropriate. SparC presents a compelling solution to a significant problem in 3D shape modeling, achieving state-of-the-art results with improved efficiency. The SparCubes representation and SparConv-VAE architecture are novel and well-integrated. While the method has limitations regarding texture and internal structures, the paper acknowledges these drawbacks. Further refinement of SparC and extensions to handle textures and internal details could significantly enhance its impact. The score reflects the framework's solid contributions and its potential to influence future research in the field.

Score: 8

- **Score**: 8/10

### **[Dynadiff: Single-stage Decoding of Images from Continuously Evolving fMRI](http://arxiv.org/abs/2505.14556v1)**
- **Summary**: Here's a summary and critical evaluation of the provided research paper:

**Summary:**

The paper introduces Dynadiff, a novel single-stage diffusion model for reconstructing images from dynamically evolving fMRI recordings.  It aims to address limitations in existing brain-to-image decoding methods that rely on complex multi-stage pipelines and often collapse the temporal dimension of fMRI data. Dynadiff simplifies the training process, demonstrates improved performance on time-resolved fMRI signals (particularly on high-level semantic image reconstruction metrics), and offers a way to characterize the evolution of image representations within brain activity.  The model is trained end-to-end and evaluated on the Natural Scenes Dataset (NSD), demonstrating state-of-the-art results compared to existing methods like Brain-Diffusers, MindEye, and WAVE.

**Critical Evaluation:**

**Novelty:** The main novelty lies in the single-stage end-to-end training approach for reconstructing images from time-resolved fMRI data.  Previous methods typically involve multiple, separately trained components and often collapse the temporal dimension through preprocessing. Dynadiff's ability to directly process the continuous stream of fMRI data is a significant departure and a substantial simplification. The characterization of image representation evolution within brain activity is also a valuable contribution.

**Significance:** The paper's significance rests on several factors:

*   **Improved performance:**  Achieving state-of-the-art results on time-resolved fMRI signals, specifically on high-level semantic image reconstruction metrics is meaningful progress.  Outperforming existing methods like MindEye2 is a strong indicator of the model's effectiveness.
*   **Simplified Training:** The single-stage training pipeline makes the approach more accessible and efficient compared to complex, multi-stage methods.
*   **Time-Resolved Analysis:** Preserving the temporal information in fMRI data allows for a more nuanced understanding of how visual representations evolve in the brain, opening new avenues for research in cognitive neuroscience.
*   **Foundation for Future Work:** As the paper states, it lays a foundation for future time-resolved brain-to-image decoding.

**Strengths:**

*   Clear problem statement and well-defined goals.
*   A novel and efficient architecture that overcomes the limitations of existing methods.
*   Strong empirical results demonstrating improved performance on time-resolved fMRI data.
*   Thorough evaluation and ablation studies to validate the model's components.
*   Qualitative results that support the quantitative findings.
*   The model is open-sourced for greater reproducibility.

**Weaknesses:**

*   The reliance on NSD dataset. While NSD is the largest fMRI dataset, the paper acknowledges that the image distribution within NSD can be stereotypical.  Evaluating the model on other datasets would strengthen the claims of generalizability.
*   The fMRI preprocessing steps, while designed to preserve the time dimension, might be improved by incorporating a foundation model of brain activity (akin to pre-trained image encoders).
*   Computational cost, while the model simplifies training in stages, the total computational cost might still be substantial because it is a diffusion model. This should be explored in the paper.
*   While the focus on static image decoding is justified as a robust baseline, the natural next step to decoding videos is not yet addressed.

**Potential influence:** Dynadiff has the potential to significantly impact the field by:

*   Encouraging a shift towards time-resolved analysis of fMRI data.
*   Simplifying the development of brain-to-image decoding models.
*   Enabling new research into the dynamics of visual representation in the brain.

**Justification for Score:**

While the paper has some limitations, its novelty, strong performance, simplified training process, and potential impact on the field justify a high score. The end-to-end approach for time-resolved decoding is a significant advancement, and the empirical results are compelling. Although the reliance on the NSD dataset and some computational concerns are present, the overall contribution warrants recognition.

Score: 8

- **Score**: 8/10

### **[Toward Reliable Biomedical Hypothesis Generation: Evaluating Truthfulness and Hallucination in Large Language Models](http://arxiv.org/abs/2505.14599v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces TruthHypo, a benchmark for evaluating the ability of Large Language Models (LLMs) to generate truthful biomedical hypotheses. It also presents KnowHD, a knowledge-based hallucination detection framework designed to assess the groundedness of these hypotheses. The benchmark utilizes a curated dataset derived from PubTator 3.0, split into "seen" and "unseen" subsets to simulate temporal progression in scientific discovery. The paper analyzes the performance of various LLMs in different knowledge augmentation settings (parametric knowledge alone, with knowledge graphs, with literature, or with both).  The study investigates the correlation between hallucination (as detected by KnowHD) and the truthfulness of generated hypotheses, and also shows a method of using the KnowHD outputs to filter LLM outputs and therefore to increase the percentage of truthful results. The paper concludes that LLMs struggle to generate truthful hypotheses and demonstrates the effectiveness of KnowHD in identifying more grounded and potentially truthful outputs. Finally, the authors perform a human study to validate the usefulness of KnowHD.

**Critical Evaluation:**

**Novelty:** The paper's novelty lies primarily in the construction of the TruthHypo benchmark and KnowHD hallucination detector specifically tailored for the biomedical domain. While other benchmarks and methods exist for evaluating LLMs, this work focuses on the critical aspect of *truthfulness* in hypothesis generation, a relatively underexplored area. The separation of the dataset into "seen" and "unseen" components to simulate scientific discovery also adds a layer of realism and allows for the evaluation of LLMs' ability to generate *novel* and truthful hypotheses within a scientific context. KnowHD's approach to analyzing reasoning chains and evaluating groundedness offers a method for assessing truthfulness.

**Significance:** The significance of this work is high, particularly given the increasing interest in using LLMs for scientific discovery.  The paper addresses a critical limitation of using LLMs in biomedicine: the tendency to hallucinate. By providing a benchmark and a method for detecting hallucinations, the authors contribute to the development of more reliable and trustworthy LLM-based hypothesis generation systems. The demonstration that groundedness scores correlate with truthfulness and can be used to improve hypothesis selection is particularly valuable. By showing how to filter hallucinated outputs, the work improves scientific outcomes. The human study further validates the potential of KnowHD to accelerate scientific discovery.

**Strengths:**

*   **Well-defined Benchmark:** The TruthHypo benchmark is clearly constructed and utilizes a realistic dataset. The temporal splitting of the data and the inclusion of negative samples enhances its rigor.
*   **Targeted Hallucination Detection:** KnowHD provides a structured approach to identifying hallucinated claims within the reasoning processes of LLMs.
*   **Comprehensive Evaluation:** The paper performs thorough experiments comparing several LLMs under different knowledge augmentation settings.
*   **Validation with Human Evaluation:** The open-ended hypothesis generation task and human evaluation adds a layer of real-world relevance to the study.
*   **Method to filter outputs:** The paper contains a method by which a filtering system may take the diverse outputs of LLMs and return the most truthful of the lot.

**Weaknesses:**

*   **Limited Scope of Relations:** The benchmark focuses on only three relation types ("Chemical & Gene", "Disease & Gene", and "Gene & Gene"). While these are important, expanding to other types could provide a more comprehensive evaluation.
*   **Focus on Extraction over Generation:** A large portion of KnowHD is focused on extracting facts as opposed to generating them.
*   **Reliance on LLMs for Claim Decomposition:** The reliance on LLMs for decomposing hypotheses into atomic claims introduces a potential source of error, as LLMs can be unreliable in this task as well. Using rule-based methods may allow for better fact extraction.
*   **Generalizability of the Human study:** While the study validates the efficacy of their method, the sample size of 54 pairs is relatively small for a definitive conclusion regarding its usability.
*   **Lack of Baseline Comparison:** A larger emphasis could be placed on comparing KnowHD to past hallucination detection systems.

**Score:** 8

**Justification:**

I assign a score of 8 because the paper makes a significant contribution to the field by addressing the crucial issue of truthfulness in LLM-generated biomedical hypotheses.  The TruthHypo benchmark provides a valuable tool for evaluating LLMs, and the KnowHD framework offers a promising method for detecting hallucinations and improving the reliability of generated hypotheses. The study is well-designed and comprehensively evaluated, with human validation adding weight to the findings. While there are some limitations in terms of the scope of relations covered and reliance on LLMs for claim decomposition, the paper's strengths outweigh its weaknesses. It opens up important avenues for future research on building more trustworthy and effective LLM-based tools for scientific discovery.

- **Score**: 8/10

### **[TinyV: Reducing False Negatives in Verification Improves RL for LLM Reasoning](http://arxiv.org/abs/2505.14625v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "TinyV: Reducing False Negatives in Verification Improves RL for LLM Reasoning."

**Summary:**

The paper addresses a significant issue in Reinforcement Learning (RL) for Large Language Models (LLMs): the problem of false negatives in answer verification. Rule-based verifiers, while common, often fail to recognize correct answers due to formatting inconsistencies, natural language variations, or notational differences.  The authors quantify this problem, showing a high prevalence of false negatives in a math reasoning dataset. They then demonstrate, both empirically and theoretically, that these false negatives impair RL training by reducing informative gradient signals and slowing convergence. To mitigate this, they introduce TinyV, a lightweight LLM-based verifier that augments existing rule-based verifiers by identifying and correcting potential false negatives.  Experiments across multiple math-reasoning benchmarks demonstrate that integrating TinyV boosts pass rates and accelerates convergence.

**Critical Evaluation:**

*   **Novelty:** The paper's novelty lies in several areas. First, it provides a systematic analysis and quantification of the false negative problem in the context of RL for LLM reasoning, which is a largely overlooked area. Second, the proposal of TinyV as a lightweight, LLM-augmented verifier is a practical and effective approach to addressing this issue. While LLMs have been used as judges in other contexts, the specific application to identifying and correcting false negatives in RL training is novel. The careful curation of real and synthetic data for TinyV is also a significant contribution. The idea to build a taxonomy of the types of errors also adds to the novelty, helping categorize verification errors.
*   **Significance:** The paper has significant implications for the field of RL for LLMs. The findings demonstrate the critical importance of reliable reward signals and highlight a major limitation of relying solely on rule-based verifiers. By addressing this issue, the paper offers a practical solution to improve the effectiveness of RL training for reasoning tasks. This can lead to more robust, generalizable LLMs with improved reasoning abilities. The introduced *HardVerify-Math Bench* is a valuable contribution to the community since it addresses a gap in existing benchmarks. The findings around the effect of using easy to verify data also opens avenues for further research.

*   **Strengths:**
    *   **Rigorous Analysis:** The paper provides a thorough analysis of the false negative problem, including empirical quantification, a taxonomy of error types, and a theoretical analysis of its impact on RL training.
    *   **Practical Solution:** TinyV is a practical and computationally efficient solution that can be easily integrated with existing RL frameworks.
    *   **Strong Experimental Results:** The experimental results demonstrate the effectiveness of TinyV across multiple benchmarks, showing significant improvements in pass rates and convergence.
    *   **Reproducibility:** The authors provide code, detailed experimental setups and data curation process promoting reproducibility.

*   **Weaknesses:**
    *   **Limited Scope:** While the paper focuses on math reasoning tasks, it would be interesting to explore the prevalence and impact of false negatives in other RL domains for LLMs, such as code generation or robotics.
    *   **TinyV's Dependency:** TinyV still relies on Prime Verifier and its extraction mechanisms (within `\boxed{}`). It might be interesting to explore a more robust system that also considers reasoning processes, to allow more complex and abstract answers.
    *   **LLM Annotation Validation:** While the paper manually validates a portion of the LLM annotations, expanding this validation would further strengthen the confidence in the accuracy of the annotated dataset.

*   **Potential Impact:** The paper is likely to influence future research in RL for LLMs by raising awareness of the false negative problem and promoting the use of more reliable verification methods. It offers a practical approach to improving reward accuracy and training efficiency, which can lead to more powerful and generalizable LLMs.

**Justification for Score:**

The paper makes a strong contribution to the field of RL for LLMs by identifying and addressing a significant, yet often overlooked, issue. The analysis is rigorous, the proposed solution is practical, and the experimental results are compelling. While there are some limitations in scope and potential avenues for further research, the paper's novelty and significance warrant a high score.

Score: 8

- **Score**: 8/10

### **[Quartet: Native FP4 Training Can Be Optimal for Large Language Models](http://arxiv.org/abs/2505.14669v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Quartet: Native FP4 Training Can Be Optimal for Large Language Models":

**Summary:**

The paper introduces Quartet, a new algorithm for training large language models (LLMs) natively in the FP4 precision format. It addresses the challenge that existing FP4 training methods often suffer from accuracy degradation and rely on mixed-precision fallbacks. The authors perform a systematic study of hardware-supported FP4 training, develop optimized CUDA kernels for NVIDIA Blackwell GPUs, and demonstrate that Quartet achieves state-of-the-art accuracy for FP4, potentially surpassing both standard-precision and FP8 training. The paper introduces a new scaling law to analyze performance tradeoffs at varying bit-widths, isolating parameter and data efficiency. They claim that Quartet provides an optimal accuracy-efficiency trade-off, maximizing accuracy within a given computational budget.  Crucially, the paper provides a GPU implementation and experimentally validates the algorithm by pre-training Llama-family models.

**Critical Evaluation:**

**Novelty:**

*   **Hardware-Aware Algorithm:** The paper's primary novelty lies in the hardware-aware design of the Quartet algorithm, specifically tailored for NVIDIA's Blackwell architecture. This goes beyond simply applying existing quantization techniques to FP4; the authors explicitly optimize for the architecture's capabilities, which is a significant contribution.
*   **Performance Scaling Law:** The introduction of a performance scaling law to analyze different precision approaches is a novel analytical technique that can be used to compare training methods by accounting for both model-parameter and data efficiencies. This provides a more systematic approach for comparing algorithms and understanding the accuracy-vs-efficiency trade-offs.
*   **GPU implementation:** The implementation of quartet is novel and highly optimized for the new Blackwell architecture.

**Significance:**

*   **Addressing Computational Cost:** Training LLMs is computationally expensive. The potential to drastically reduce costs via FP4 while maintaining accuracy is highly significant. If FP4 training can become a mainstream approach, it could democratize access to training frontier models.
*   **Pushing the Precision Frontier:** The paper pushes the boundaries of low-precision training, challenging the established wisdom that 8-bit or mixed-precision approaches are necessary for acceptable accuracy. This encourages further research in extremely low-precision training methods.
*   **Practical Implications:** The claims about Quartet's ability to compete with or surpass FP8 training in terms of accuracy-vs-speed have major practical implications for model development. The paper provides useful guiding principle in picking backward and forward passes.

**Strengths:**

*   **Thorough Evaluation:** The paper includes a comprehensive experimental evaluation, comparing Quartet against multiple existing methods across various model sizes and token budgets. This provides solid evidence for its claimed advantages.
*   **Detailed Scaling Law Analysis:** The use of a performance scaling law to analyze and compare different training methods is a strong point. The parameters associated with the data and model efficiency parameters offers interesting insights.
*   **Clarity:** The paper is generally well-written and clearly explains the algorithm and its optimization strategy.

**Weaknesses:**

*   **Hardware Dependency:** Quartet's design is heavily tailored to the Blackwell architecture.  While this allows for significant performance gains, it also limits its portability to other hardware platforms. The paper could acknowledge and discuss this limitation more explicitly.
*   **Broader Applicability Claims:** The "near-optimal" claim depends on the scaling-law fit and associated parameters, which are in turn fitted on a relatively specific set of data (Llama models on the C4 dataset). More discussion on how these insights apply to other architectures could improve its impact. The lack of details on what happens in the absence of TensorCores, will also prevent it from wider adoption.
*   **Dependence on Custom Hardware:** The reliance on specific hardware features of Blackwell GPUs (e.g., specialized Tensor Cores) makes the results less readily transferable to other architectures or software-based simulations.

**Overall Assessment:**

The paper makes a significant contribution by introducing a practical and highly optimized FP4 training algorithm for LLMs, tailored to the capabilities of NVIDIA's Blackwell architecture. The novel use of a performance scaling law provides a valuable framework for analyzing training method efficiency. While the hardware-dependent design is a limitation, the potential for substantial cost reductions in LLM training justifies the specialized approach. The paper is well-written, thoroughly evaluated, and will likely have a notable influence on research in low-precision training techniques.
The work also validates a previously untested assumption in the AI scaling paradigm, which is that low precision multiplication in both the forward and backward passes still allow competitive scaling.

Score: 8

- **Score**: 8/10

### **[Reward Reasoning Model](http://arxiv.org/abs/2505.14674v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces Reward Reasoning Models (RRMs), a novel approach to reward modeling that incorporates explicit chain-of-thought reasoning *before* generating final rewards.  This allows the RRM to adaptively allocate computational resources based on the complexity of the query.  The authors develop a reinforcement learning framework to train RRMs without requiring explicit reasoning traces. They demonstrate that RRMs outperform existing reward models on various benchmarks, exhibit adaptive test-time compute scaling, and develop distinct reasoning patterns.  They showcase its effectiveness in reward-guided best-of-N inference and LLM post-training. The authors present the models on Hugging Face.

**Critical Evaluation:**

*   **Novelty:** The core idea of incorporating *reasoning* into reward models is novel and makes a significant contribution.  Existing reward models typically treat reward assessment as a black-box, directly mapping input to a scalar reward. The RRMs allow for a more transparent and potentially accurate reward signal, especially for tasks requiring multi-step reasoning. Framing reward modeling as a reasoning task itself and enabling adaptive computation is a significant conceptual leap.

*   **Significance:** The potential impact of RRMs on the field is substantial. Better reward models lead to better alignment of large language models with human expectations, improved performance on complex tasks, and more efficient utilization of computational resources.  By enabling adaptive computation at test time, RRMs can unlock higher performance for demanding queries without incurring unnecessary costs for simpler ones. The success of post-training LLMs with RRM feedback is particularly noteworthy. The open-sourcing of the models will further accelerate research.

*   **Strengths:**
    *   The experimental results are comprehensive and demonstrate the superior performance of RRMs across various benchmarks and tasks (reward modeling, best-of-N inference, LLM post-training).
    *   The analysis of reasoning patterns reveals that RRMs genuinely develop distinct reasoning strategies compared to regular LLMs, validating the effectiveness of the training framework.
    *   The adaptive test-time compute scaling experiments are well-designed and showcase a key advantage of the RRM approach. The consistent performance boost as test-time compute scales up validates the core hypothesis.
    *   The paper is well-written and clearly explains the technical details of the approach.

*   **Weaknesses:**
    *   The reward function used for training is relatively simple (correct/incorrect preference). While effective, exploring more sophisticated reward signals could potentially lead to further improvements.  The rule-based reward environment, while a clever way to avoid supervised reasoning traces, might still impose limitations on the complexity of reasoning that RRMs can learn.
    *   While the experiments show improvements in GPQA through reinforcement learning, the relative improvement in the post-training section overall is smaller than other sections.
    *   The experiments with multiple candidate responses could better consider the compute efficiency of competing methods. The performance benefits from adaptive test-time scaling would also need to consider the energy and carbon impact.

*   **Influence:** The work has the potential to influence the development of future reward models, alignment techniques, and LLM inference strategies. By demonstrating the benefits of reasoning in reward modeling, the paper opens up new avenues of research in this area.

*   **Score Justification:** While the paper presents a significant and novel contribution, the reliance on a fairly simple reward function and room for improvement in the post-training experiments, along with compute efficiency considerations temper enthusiasm slightly. The adaptive compute aspect is a very attractive one that will likely spur a lot of future work.

Score: 8

- **Score**: 8/10

### **[UltraEdit: Training-, Subject-, and Memory-Free Lifelong Editing in Large Language Models](http://arxiv.org/abs/2505.14679v1)**
- **Summary**: Here's a summary and critical evaluation of the ULTRAEDIT paper:

**Summary:**

The paper introduces ULTRAEDIT, a novel approach to lifelong model editing in large language models (LLMs).  ULTRAEDIT aims to address the challenges of scalability, catastrophic forgetting, and reliance on training data and external memory faced by existing model editing techniques. It achieves this through a "training-, subject-, and memory-free" approach, relying on lightweight linear algebra operations to compute parameter shifts.  A key component is "lifelong normalization," which continuously updates feature statistics to adapt to distributional shifts during the editing process.  The authors also introduce ULTRAEDITBENCH, a large-scale dataset for model editing, and demonstrate that ULTRAEDIT achieves significantly faster editing speeds, lower memory consumption, and superior performance compared to existing methods, even scaling to 1 million edits.

**Critical Evaluation:**

**Novelty:**  The novelty lies in the combination of several design choices to achieve truly scalable and efficient lifelong model editing. While individual components, such as localized parameter updates and closed-form optimization, have been explored in prior work, the self-contained and training-free approach with lifelong normalization is a unique contribution.  The introduction of ULTRAEDITBENCH as a very large-scale dataset also addresses a practical limitation in the field, pushing the boundaries of what is possible in terms of evaluating lifelong editing methods.

**Significance:**  The significance of this work stems from its practical implications for deploying LLMs in real-world scenarios.  The ability to efficiently and continuously update model knowledge without retraining or relying on external memory is a major step forward. The speed and memory efficiency gains are substantial and enable editing of larger models on consumer-grade hardware, broadening the accessibility of model editing techniques. The demonstration of scaling to 1 million edits is impressive and highlights the potential for long-term model adaptation. The improved accuracy and preservation of general capabilities after many edits are also very important, addressing key limitations in other methods.

**Strengths:**

*   **Efficiency and Scalability:** The most significant strength is the demonstrated speed and memory efficiency, making it practical for large models and high-frequency updates.
*   **Training-free and Dependency-free:** Avoiding the need for additional training data and external memory simplifies deployment and reduces overhead.
*   **Lifelong Normalization:** This is a critical component for maintaining stability and accuracy over many editing turns, addressing the edit collapse problem.
*   **Large-Scale Evaluation:** The creation and use of the ULTRAEDITBENCH dataset allow for unprecedented evaluation of lifelong editing capabilities.
*   **Strong Empirical Results:** The paper presents compelling results on a variety of datasets and models, consistently outperforming existing methods.
*   **Ability to Scale:** Scaling to one million edits with stable performance is very promising for real world applications.

**Weaknesses:**

*   **Reliance on Linear Algebra:** While efficient, the reliance on linear algebra operations might limit the complexity of the parameter shifts that can be achieved. It is unclear if more nuanced or complex knowledge edits could be achieved with a more complex function.
*   **Limited exploration of catastrophic forgetting mitigation**: While the paper addresses catastrophic forgetting through lifelong normalization, this strategy may be insufficient for more complex and diverse lifelong learning scenarios, which is a significant limitation. Further research might need to look into methods that can tackle more complex instances of forgetting.
*   **Limited generalizability of editable modules:**. Defining editable modules is not well explored in the paper, with a limited set of modules selected per model per dataset.

**Potential Influence:**

The ULTRAEDIT paper has the potential to significantly influence the field of model editing. It provides a practical and scalable solution to lifelong model adaptation, making it a valuable tool for deploying LLMs in dynamic environments. The ULTRAEDITBENCH dataset will also likely become a standard benchmark for evaluating lifelong editing methods. The focus on efficiency and reducing dependence on external resources could inspire further research into lightweight and self-contained model editing techniques.

**Score:** 8

**Rationale:**  ULTRAEDIT represents a significant advancement in the field of model editing, particularly in its scalability, efficiency, and practical applicability. The training-free approach and lifetime normalization strategy offer significant advantages over existing methods. The ULTRAEDITBENCH dataset contributes a valuable resource to the community. While there are minor limitations in the complexity of the edits and the long term mitigiation of catastrophic forgetting, its strengths outweigh its weaknesses, warranting a score of 8. The work addresses major challenges in a very practical and useful way, and is likely to have a broad impact in the field of LLM management and deployment.

- **Score**: 8/10

## Other Papers
### **[EEG-to-Text Translation: A Model for Deciphering Human Brain Activity](http://arxiv.org/abs/2505.13936v1)**
### **[MLZero: A Multi-Agent System for End-to-end Machine Learning Automation](http://arxiv.org/abs/2505.13941v1)**
### **[Visual Instruction Bottleneck Tuning](http://arxiv.org/abs/2505.13946v1)**
### **[FlashThink: An Early Exit Method For Efficient Reasoning](http://arxiv.org/abs/2505.13949v1)**
### **[Through a Compressed Lens: Investigating the Impact of Quantization on LLM Explainability and Interpretability](http://arxiv.org/abs/2505.13963v1)**
### **[Truth or Twist? Optimal Model Selection for Reliable Label Flipping Evaluation in LLM-based Counterfactuals](http://arxiv.org/abs/2505.13972v1)**
### **[Toward Effective Reinforcement Learning Fine-Tuning for Medical VQA in Vision-Language Models](http://arxiv.org/abs/2505.13973v1)**
### **[DRP: Distilled Reasoning Pruning with Skill-aware Step Decomposition for Efficient Large Reasoning Models](http://arxiv.org/abs/2505.13975v1)**
### **[Combining Deterministic Enhanced Conditions with Dual-Streaming Encoding for Diffusion-Based Speech Enhancement](http://arxiv.org/abs/2505.13983v1)**
### **[The Hallucination Tax of Reinforcement Finetuning](http://arxiv.org/abs/2505.13988v1)**
### **[When LLMs meet open-world graph learning: a new perspective for unlabeled data uncertainty](http://arxiv.org/abs/2505.13989v1)**
### **[DecIF: Improving Instruction-Following through Meta-Decomposition](http://arxiv.org/abs/2505.13990v1)**
### **[Divide by Question, Conquer by Agent: SPLIT-RAG with Question-Driven Graph Partitioning](http://arxiv.org/abs/2505.13994v1)**
### **[Activation-Guided Consensus Merging for Large Language Models](http://arxiv.org/abs/2505.14009v1)**
### **[AUTOLAW: Enhancing Legal Compliance in Large Language Models via Case Law Generation and Jury-Inspired Deliberation](http://arxiv.org/abs/2505.14015v1)**
### **[Adaptive Cyclic Diffusion for Inference Scaling](http://arxiv.org/abs/2505.14036v1)**
### **[ProMind-LLM: Proactive Mental Health Care via Causal Reasoning with Sensor Data](http://arxiv.org/abs/2505.14038v1)**
### **[Adversarially Pretrained Transformers may be Universally Robust In-Context Learners](http://arxiv.org/abs/2505.14042v1)**
### **[From Unaligned to Aligned: Scaling Multilingual LLMs with Multi-Way Parallel Corpora](http://arxiv.org/abs/2505.14045v1)**
### **[Improved Methods for Model Pruning and Knowledge Distillation](http://arxiv.org/abs/2505.14052v1)**
### **[Field Matters: A lightweight LLM-enhanced Method for CTR Prediction](http://arxiv.org/abs/2505.14057v1)**
### **[Process vs. Outcome Reward: Which is Better for Agentic RAG Reinforcement Learning](http://arxiv.org/abs/2505.14069v1)**
### **[Enhancing LLMs via High-Knowledge Data Selection](http://arxiv.org/abs/2505.14070v1)**
### **[Textual Steering Vectors Can Improve Visual Understanding in Multimodal Large Language Models](http://arxiv.org/abs/2505.14071v1)**
### **[CE-LSLM: Efficient Large-Small Language Model Inference and Communication via Cloud-Edge Collaboration](http://arxiv.org/abs/2505.14085v1)**
### **[Beyond Chains: Bridging Large Language Models and Knowledge Bases in Complex Question Answering](http://arxiv.org/abs/2505.14099v1)**
### **[MultiHal: Multilingual Dataset for Knowledge-Graph Grounded Evaluation of LLM Hallucinations](http://arxiv.org/abs/2505.14101v1)**
### **[Legal Rule Induction: Towards Generalizable Principle Discovery from Analogous Judicial Precedents](http://arxiv.org/abs/2505.14104v1)**
### **[A Personalized Conversational Benchmark: Towards Simulating Personalized Conversations](http://arxiv.org/abs/2505.14106v1)**
### **[DiagnosisArena: Benchmarking Diagnostic Reasoning for Large Language Models](http://arxiv.org/abs/2505.14107v1)**
### **[Invisible Entropy: Towards Safe and Efficient Low-Entropy LLM Watermarking](http://arxiv.org/abs/2505.14112v1)**
### **[Self-Reasoning Language Models: Unfold Hidden Reasoning Chains with Few Reasoning Catalyst](http://arxiv.org/abs/2505.14116v1)**
### **[MAS-KCL: Knowledge component graph structure learning with large language model-based agentic workflow](http://arxiv.org/abs/2505.14126v1)**
### **[Texts or Images? A Fine-grained Analysis on the Effectiveness of Input Representations and Models for Table Question Answering](http://arxiv.org/abs/2505.14131v1)**
### **[Hunyuan-Game: Industrial-grade Intelligent Game Creation Model](http://arxiv.org/abs/2505.14135v1)**
### **[FlowQ: Energy-Guided Flow Policies for Offline Reinforcement Learning](http://arxiv.org/abs/2505.14139v1)**
### **[RL of Thoughts: Navigating LLM Reasoning with Inference-time Reinforcement Learning](http://arxiv.org/abs/2505.14140v1)**
### **[s3: You Don't Need That Much Data to Train a Search Agent via RL](http://arxiv.org/abs/2505.14146v1)**
### **[SHARP: Synthesizing High-quality Aligned Reasoning Problems for Large Reasoning Models Reinforcement Learning](http://arxiv.org/abs/2505.14147v1)**
### **[MM-Agent: LLM as Agents for Real-world Mathematical Modeling Problem](http://arxiv.org/abs/2505.14148v1)**
### **[Unify Graph Learning with Text: Unleashing LLM Potentials for Session Search](http://arxiv.org/abs/2505.14156v1)**
### **[Temporal Alignment of Time Sensitive Facts with Activation Engineering](http://arxiv.org/abs/2505.14158v1)**
### **[LMP: Leveraging Motion Prior in Zero-Shot Video Generation with Diffusion Transformer](http://arxiv.org/abs/2505.14167v1)**
### **[The Strawberry Problem: Emergence of Character-level Understanding in Tokenized Language Models](http://arxiv.org/abs/2505.14172v1)**
### **[Cheaper, Better, Faster, Stronger: Robust Text-to-SQL without Chain-of-Thought or Fine-Tuning](http://arxiv.org/abs/2505.14174v1)**
### **[Tokenization Constraints in LLMs: A Study of Symbolic and Arithmetic Reasoning Limits](http://arxiv.org/abs/2505.14178v1)**
### **[SlangDIT: Benchmarking LLMs in Interpretative Slang Translation](http://arxiv.org/abs/2505.14181v1)**
### **[ThinkSwitcher: When to Think Hard, When to Think Fast](http://arxiv.org/abs/2505.14183v1)**
### **[Safety Subspaces are Not Distinct: A Fine-Tuning Case Study](http://arxiv.org/abs/2505.14185v1)**
### **[Unraveling Interwoven Roles of Large Language Models in Authorship Privacy: Obfuscation, Mimicking, and Verification](http://arxiv.org/abs/2505.14195v1)**
### **[Towards Omnidirectional Reasoning with 360-R1: A Dataset, Benchmark, and GRPO-based Method](http://arxiv.org/abs/2505.14197v1)**
### **[Capturing the Effects of Quantization on Trojans in Code LLMs](http://arxiv.org/abs/2505.14200v1)**
### **[Challenges and Limitations in the Synthetic Generation of mHealth Sensor Data](http://arxiv.org/abs/2505.14206v1)**
### **[Automatic Dataset Generation for Knowledge Intensive Question Answering Tasks](http://arxiv.org/abs/2505.14212v1)**
### **["Haet Bhasha aur Diskrimineshun": Phonetic Perturbations in Code-Mixed Hinglish to Red-Team LLMs](http://arxiv.org/abs/2505.14226v1)**
### **[UniVG-R1: Reasoning Guided Universal Visual Grounding with Reinforcement Learning](http://arxiv.org/abs/2505.14231v1)**
### **[ABBA: Highly Expressive Hadamard Product Adaptation for Large Language Models](http://arxiv.org/abs/2505.14238v1)**
### **[TransBench: Benchmarking Machine Translation for Industrial-Scale Applications](http://arxiv.org/abs/2505.14244v1)**
### **[Instructing Text-to-Image Diffusion Models via Classifier-Guided Semantic Optimization](http://arxiv.org/abs/2505.14254v1)**
### **[FuxiMT: Sparsifying Large Language Models for Chinese-Centric Multilingual Machine Translation](http://arxiv.org/abs/2505.14256v1)**
### **[Speculative Decoding Reimagined for Multimodal Large Language Models](http://arxiv.org/abs/2505.14260v1)**
### **[AAPO: Enhance the Reasoning Capabilities of LLMs with Advantage Momentum](http://arxiv.org/abs/2505.14264v1)**
### **[Think-J: Learning to Think for Generative LLM-as-a-Judge](http://arxiv.org/abs/2505.14268v1)**
### **[YESciEval: Robust LLM-as-a-Judge for Scientific Question Answering](http://arxiv.org/abs/2505.14279v1)**
### **[Universal Acoustic Adversarial Attacks for Flexible Control of Speech-LLMs](http://arxiv.org/abs/2505.14286v1)**
### **[Towards Generating Realistic Underwater Images](http://arxiv.org/abs/2505.14296v1)**
### **[Cross-Lingual Optimization for Language Transfer in Large Language Models](http://arxiv.org/abs/2505.14297v1)**
### **[Empowering LLMs in Task-Oriented Dialogues: A Domain-Independent Multi-Agent Framework and Fine-Tuning Strategy](http://arxiv.org/abs/2505.14299v1)**
### **[SafetyNet: Detecting Harmful Outputs in LLMs by Modeling and Monitoring Deceptive Behaviors](http://arxiv.org/abs/2505.14300v1)**
### **[Scaling Law for Quantization-Aware Training](http://arxiv.org/abs/2505.14302v1)**
### **[JOLT-SQL: Joint Loss Tuning of Text-to-SQL with Confusion-aware Noisy Schema Sampling](http://arxiv.org/abs/2505.14305v1)**
### **[HausaNLP: Current Status, Challenges and Future Directions for Hausa Natural Language Processing](http://arxiv.org/abs/2505.14311v1)**
### **[A MIND for Reasoning: Meta-learning for In-context Deduction](http://arxiv.org/abs/2505.14313v1)**
### **[Low-Cost FlashAttention with Fused Exponential and Multiplication Hardware Operators](http://arxiv.org/abs/2505.14314v1)**
### **[Exploring Jailbreak Attacks on LLMs through Intent Concealment and Diversion](http://arxiv.org/abs/2505.14316v1)**
### **[RADAR: Enhancing Radiology Report Generation with Supplementary Knowledge Injection](http://arxiv.org/abs/2505.14318v1)**
### **[Scaling and Enhancing LLM-based AVSR: A Sparse Mixture of Projectors Approach](http://arxiv.org/abs/2505.14336v1)**
### **[QA-prompting: Improving Summarization with Large Language Models using Question-Answering](http://arxiv.org/abs/2505.14347v1)**
### **[OSoRA: Output-Dimension and Singular-Value Initialized Low-Rank Adaptation](http://arxiv.org/abs/2505.14350v1)**
### **[WirelessMathBench: A Mathematical Modeling Benchmark for LLMs in Wireless Communications](http://arxiv.org/abs/2505.14354v1)**
### **[PersonaTAB: Predicting Personality Traits using Textual, Acoustic, and Behavioral Cues in Fully-Duplex Speech Dialogs](http://arxiv.org/abs/2505.14356v1)**
### **[Vid2World: Crafting Video Diffusion Models to Interactive World Models](http://arxiv.org/abs/2505.14357v1)**
### **[Vision-Language Modeling Meets Remote Sensing: Models, Datasets and Perspectives](http://arxiv.org/abs/2505.14361v1)**
### **[Dual Decomposition of Weights and Singular Value Low Rank Adaptation](http://arxiv.org/abs/2505.14367v1)**
### **[Is Your Prompt Safe? Investigating Prompt Injection Attacks Against Open-Source LLMs](http://arxiv.org/abs/2505.14368v1)**
### **[AutoRev: Automatic Peer Review System for Academic Research Papers](http://arxiv.org/abs/2505.14376v1)**
### **[SCAN: Semantic Document Layout Analysis for Textual and Visual Retrieval-Augmented Generation](http://arxiv.org/abs/2505.14381v1)**
### **[Knowledge Graph Based Repository-Level Code Generation](http://arxiv.org/abs/2505.14394v1)**
### **[MUG-Eval: A Proxy Evaluation Framework for Multilingual Generation Capabilities in Any Language](http://arxiv.org/abs/2505.14395v1)**
### **[Causal Cartographer: From Mapping to Reasoning Over Counterfactual Worlds](http://arxiv.org/abs/2505.14396v1)**
### **[Log-Augmented Generation: Scaling Test-Time Reasoning with Reusable Computation](http://arxiv.org/abs/2505.14398v1)**
### **[ViC-Bench: Benchmarking Visual-Interleaved Chain-of-Thought Capability in MLLMs with Free-Style Intermediate State Representations](http://arxiv.org/abs/2505.14404v1)**
### **[Pierce the Mists, Greet the Sky: Decipher Knowledge Overshadowing via Knowledge Circuit Analysis](http://arxiv.org/abs/2505.14406v1)**
### **[Towards Non-Euclidean Foundation Models: Advancing AI Beyond Euclidean Frameworks](http://arxiv.org/abs/2505.14417v1)**
### **[Hidden Ghost Hand: Unveiling Backdoor Vulnerabilities in MLLM-Powered Mobile GUI Agents](http://arxiv.org/abs/2505.14418v1)**
### **[MindVote: How LLMs Predict Human Decision-Making in Social Media Polls](http://arxiv.org/abs/2505.14422v1)**
### **[From Templates to Natural Language: Generalization Challenges in Instruction-Tuned LLMs for Spatial Reasoning](http://arxiv.org/abs/2505.14425v1)**
### **[Rank-K: Test-Time Reasoning for Listwise Reranking](http://arxiv.org/abs/2505.14432v1)**
### **[Choosing a Model, Shaping a Future: Comparing LLM Perspectives on Sustainability and its Relationship with AI](http://arxiv.org/abs/2505.14435v1)**
### **[Neural Incompatibility: The Unbridgeable Gap of Cross-Scale Parametric Knowledge Transfer in Large Language Models](http://arxiv.org/abs/2505.14436v1)**
### **[S2SBench: A Benchmark for Quantifying Intelligence Degradation in Speech-to-Speech Large Language Models](http://arxiv.org/abs/2505.14438v1)**
### **[Creative Preference Optimization](http://arxiv.org/abs/2505.14442v1)**
### **[Video Compression Commander: Plug-and-Play Inference Acceleration for Video Large Language Models](http://arxiv.org/abs/2505.14454v1)**
### **[CtrlDiff: Boosting Large Diffusion Language Models with Dynamic Block Prediction and Controllable Generation](http://arxiv.org/abs/2505.14455v1)**
### **[VisualQuality-R1: Reasoning-Induced Image Quality Assessment via Reinforcement Learning to Rank](http://arxiv.org/abs/2505.14460v1)**
### **[Attributional Safety Failures in Large Language Models under Code-Mixed Perturbations](http://arxiv.org/abs/2505.14469v1)**
### **[Towards Reliable Proof Generation with LLMs: A Neuro-Symbolic Approach](http://arxiv.org/abs/2505.14479v1)**
### **[MoMoE: Mixture of Moderation Experts Framework for AI-Assisted Online Governance](http://arxiv.org/abs/2505.14483v1)**
### **[Reasoning Models Better Express Their Confidence](http://arxiv.org/abs/2505.14489v1)**
### **[Enhanced Multimodal Aspect-Based Sentiment Analysis by LLM-Generated Rationales](http://arxiv.org/abs/2505.14499v1)**
### **[Learning to Integrate Diffusion ODEs by Averaging the Derivatives](http://arxiv.org/abs/2505.14502v1)**
### **[ModRWKV: Transformer Multimodality in Linear Time](http://arxiv.org/abs/2505.14505v1)**
### **[Latent Flow Transformer](http://arxiv.org/abs/2505.14513v1)**
### **[Teaching Audio-Aware Large Language Models What Does Not Hear: Mitigating Hallucinations through Synthesized Negative Samples](http://arxiv.org/abs/2505.14518v1)**
### **[SparC: Sparse Representation and Construction for High-Resolution 3D Shapes Modeling](http://arxiv.org/abs/2505.14521v1)**
### **[Guarded Query Routing for Large Language Models](http://arxiv.org/abs/2505.14524v1)**
### **[BugRepro: Enhancing Android Bug Reproduction with Domain-Specific Knowledge Integration](http://arxiv.org/abs/2505.14528v1)**
### **[Internal Chain-of-Thought: Empirical Evidence for Layer-wise Subtask Scheduling in LLMs](http://arxiv.org/abs/2505.14530v1)**
### **[Energy-Efficient Deep Reinforcement Learning with Spiking Transformers](http://arxiv.org/abs/2505.14533v1)**
### **[Breaking Bad Tokens: Detoxification of LLMs Using Sparse Autoencoders](http://arxiv.org/abs/2505.14536v1)**
### **[Can Large Language Models Really Recognize Your Name?](http://arxiv.org/abs/2505.14549v1)**
### **[KORGym: A Dynamic Game Platform for LLM Reasoning Evaluation](http://arxiv.org/abs/2505.14552v1)**
### **[Dynadiff: Single-stage Decoding of Images from Continuously Evolving fMRI](http://arxiv.org/abs/2505.14556v1)**
### **[Can Pruning Improve Reasoning? Revisiting Long-CoT Compression with Capability in Mind for Better Reasoning](http://arxiv.org/abs/2505.14582v1)**
### **[Context Reasoner: Incentivizing Reasoning Capability for Contextualized Privacy and Safety Compliance via Reinforcement Learning](http://arxiv.org/abs/2505.14585v1)**
### **[Toward Reliable Biomedical Hypothesis Generation: Evaluating Truthfulness and Hallucination in Large Language Models](http://arxiv.org/abs/2505.14599v1)**
### **[SATBench: Benchmarking LLMs' Logical Reasoning via Automated Puzzle Generation from SAT Formulas](http://arxiv.org/abs/2505.14615v1)**
### **[Linear Control of Test Awareness Reveals Differential Compliance in Reasoning Models](http://arxiv.org/abs/2505.14617v1)**
### **[Enhancing Learned Knowledge in LoRA Adapters Through Efficient Contrastive Decoding on Ascend NPUs](http://arxiv.org/abs/2505.14620v1)**
### **[TinyV: Reducing False Negatives in Verification Improves RL for LLM Reasoning](http://arxiv.org/abs/2505.14625v1)**
### **[Debating for Better Reasoning: An Unsupervised Multimodal Approach](http://arxiv.org/abs/2505.14627v1)**
### **[KERL: Knowledge-Enhanced Personalized Recipe Recommendation using Large Language Models](http://arxiv.org/abs/2505.14629v1)**
### **[Think Only When You Need with Large Hybrid-Reasoning Models](http://arxiv.org/abs/2505.14631v1)**
### **[General-Reasoner: Advancing LLM Reasoning Across All Domains](http://arxiv.org/abs/2505.14652v1)**
### **[SAFEPATH: Preventing Harmful Reasoning in Chain-of-Thought via Early Alignment](http://arxiv.org/abs/2505.14667v1)**
### **[ContextAgent: Context-Aware Proactive LLM Agents with Open-World Sensory Perceptions](http://arxiv.org/abs/2505.14668v1)**
### **[Quartet: Native FP4 Training Can Be Optimal for Large Language Models](http://arxiv.org/abs/2505.14669v1)**
### **[Training-Free Watermarking for Autoregressive Image Generation](http://arxiv.org/abs/2505.14673v1)**
### **[Reward Reasoning Model](http://arxiv.org/abs/2505.14674v1)**
### **[Visionary-R1: Mitigating Shortcuts in Visual Reasoning with Reinforcement Learning](http://arxiv.org/abs/2505.14677v1)**
### **[UltraEdit: Training-, Subject-, and Memory-Free Lifelong Editing in Large Language Models](http://arxiv.org/abs/2505.14679v1)**
### **[UniGen: Enhanced Training & Test-Time Strategies for Unified Multimodal Understanding and Generation](http://arxiv.org/abs/2505.14682v1)**
