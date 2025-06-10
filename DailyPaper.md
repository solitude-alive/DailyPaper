# The Latest Daily Papers - Date: 2025-06-10
## Highlight Papers
### **[AMoPO: Adaptive Multi-objective Preference Optimization without Reward Models and Reference Models](http://arxiv.org/abs/2506.07165v1)**
- **Summary**: Okay, here's a concise summary of the paper "AMOPO: Adaptive Multi-objective Preference Optimization without Reward Models and Reference Models," along with a critical evaluation of its novelty and significance:

**Summary:**

The paper introduces AMOPO (Adaptive Multi-objective Preference Optimization), a novel framework for aligning Large Language Models (LLMs) with multi-dimensional human preferences. Unlike existing methods that rely on auxiliary reward models or reference models, AMOPO uses dimension-aware generation metrics as implicit rewards, thereby simplifying the alignment process.  It also introduces an adaptive weight assignment mechanism that dynamically prioritizes preference dimensions based on a Gaussian distribution model of the generation space. Empirical results demonstrate that AMOPO outperforms state-of-the-art baselines on several benchmarks, exhibiting superior performance and scaling abilities across different model sizes. The paper emphasizes the adaptability and effectiveness of AMOPO in achieving dimension-aware preference alignment.

**Critical Evaluation:**

*   **Strengths:**
    *   **Novel Approach:** AMOPO offers a genuinely novel approach by eliminating the need for auxiliary reward or reference models. This simplifies the training pipeline and reduces computational complexity, which is a significant advantage.
    *   **Adaptive Weighting:** The adaptive weight assignment mechanism provides a sophisticated way to balance multiple preference dimensions dynamically. This is a crucial improvement over methods using fixed weights, as it allows the LLM to adapt to varying contexts and user needs.
    *   **Strong Empirical Results:** The paper presents compelling empirical evidence demonstrating that AMOPO outperforms existing methods on various benchmarks. This showcases its effectiveness in practice.
    *   **Scaling Ability:** The experiments across different model sizes (7B, 14B, and 32B) confirm the scalability of AMOPO, indicating its potential for use with larger and more powerful LLMs.
    *   **Focus on a Real-World Problem:** Multi-objective alignment is a significant challenge for the safe and controllable deployment of AI systems. Addressing this problem with a practical and efficient solution is highly relevant.
    *   **Open Source Code and Datasets:**  Providing code and datasets makes the research more accessible and facilitates further experimentation and validation by the community.

*   **Weaknesses:**
    *   **Dataset limitations:** While HelpSteer2 is used to train the policy model on, the paper lacks consideration for the impact of different training datasets, such as Ultrafeedback.

    *   **Multi-Turn Dialogue focus:** While AMOPO handles static preference scenarios effectively, further work is needed to manage dynamic preference changes that occur during ongoing interactions.

    *   **Dimension diversity:** in our experiment, we consider three dimensions; that means the Gaussian distribution assumption for modelling the generation space may not generalize well to all types of preference dimensions.

*   **Significance:**

    *   **Theoretical contribution:** Introduces an objective function combining a Multi-Objective Bradley-Terry Model, an adaptive-weight policy and Multi-Objective Optimization, reducing auxiliary models such as reward and reference models.

    *   **Practical Impact:** AMOPO has the potential to significantly reduce the computational cost and complexity of aligning LLMs with human preferences, making it more accessible to researchers and practitioners with limited resources.
    *   **Influence on the Field:** The idea of using implicit rewards derived from generation metrics could inspire new directions in preference alignment research, moving away from reliance on external reward models. The adaptive weighting mechanism offers a robust solution to balancing conflicting preferences in LLMs, which will likely be adopted and extended by other researchers.

*   **Novelty:** The work is novel due to its unique use of Multi-Objective Optimization and the Adaptive Weight Policy with an implicit rewards method. It also differs from other alignment methods.

**Justification for Score:**

AMOPO presents a significant advancement in multi-objective preference alignment. Its novelty lies in the elimination of explicit reward/reference models and the introduction of an adaptive weighting scheme. The empirical results demonstrate its effectiveness, and the scaling experiments further validate its potential. I recognize the limitations regarding datasets and dimension diversity. Considering these strengths and weaknesses, I assign a score of **8.5**.

Score: 8.5

- **Score**: 8/10

### **[CTDGSI: A comprehensive exploitation of instance selection methods for automatic text classification. VII Concurso de Teses, Dissertações e Trabalhos de Graduação em SI -- XXI Simpósio Brasileiro de Sistemas de Informação](http://arxiv.org/abs/2506.07169v1)**
- **Summary**: Here's a summary and critical evaluation of the provided dissertation abstract:

**Summary:**

The dissertation focuses on Instance Selection (IS) techniques within Natural Language Processing (NLP), specifically for Automatic Text Classification (ATC).  It investigates the potential of IS to reduce training set sizes by removing noisy or redundant instances, thereby reducing training costs while maintaining model effectiveness, particularly relevant in the context of resource-intensive large language models (LLMs). The work compares several IS methods, proposes two novel IS solutions (one redundancy-oriented and one noise-oriented & redundancy-aware), and evaluates them against various classification models and datasets. The final proposed solution achieves significant training set reduction and speedups while maintaining effectiveness. The work culminated in several publications, including top-tier Information Systems journals.

**Critical Evaluation:**

**Novelty:**

*   **Strength:**  The core novelty lies in the comprehensive investigation of IS methods *specifically* for NLP, and even more specifically for ATC, a relatively unexplored area. The creation of two novel IS algorithms, E2SC and a bi-objective variant (biO-IS), that are designed for modern, large datasets and transformer architectures contributes significantly to the knowledge base. The creation of new taxonomy is another source of novelty.
*   **Weakness:** While IS itself isn't a brand-new area, the focus on *transformer-based* ATC and dealing with real-world large and skewed datasets is where novelty resides. The use of KNN as part of the E2SC for the "calibrated" redundancy estimate is a potential weakness; the Dissertation will need to rigorously justify that choice (which the text implies is explored).  The dependence on TF-IDF initially might also limit some of its applicability to scenarios where embedding-based approaches are more advantageous from an effectiveness perspective.

**Significance:**

*   **Strength:** The increasing demand for computational resources in NLP, particularly with LLMs, makes this work extremely significant. Reducing training costs without sacrificing model performance is a crucial area of research. The demonstrated speedups and reduction in data size have practical implications for researchers and practitioners. The publication record in top-tier journals supports the quality and significance of the work. Addressing the "sustainability" of NLP through more efficient training techniques is timely and important.
*   **Weakness:** The significance hinges on the generalizability of the results.  While tested on a variety of datasets, the scope of ATC might limit its applicability to other NLP tasks. The dissertation needs to provide compelling arguments for why the proposed IS methods are suitable for a wider range of NLP problems. The analysis is centered around a fixed set of hyperparameters, and it may be hard to extend the results to situations outside these restrictions.

**Rigour & Justification:**

The work appears rigorous, evidenced by the extensive experimentation (4000+ experiments) and the use of statistical tests (paired t-test, Bonferroni correction). The claims are backed by empirical results across a wide range of datasets.

**Detailed Justification for the Score:**

The dissertation tackles a critical and timely problem in NLP, exhibiting novelty in its focused application of IS techniques to transformer-based ATC and the design of new IS algorithms tailored for this domain. The significance stems from the potential to reduce computational costs and make NLP more sustainable. The rigorous methodology and strong publication record further strengthen its value. It can be argued that the innovation lies in the application and tailoring of existing concepts for a specific niche (IS methods for transformers in ATC) rather than revolutionary algorithmic breakthroughs. Therefore a perfect score is not warranted.

**Score: 8**

- **Score**: 8/10

### **[Reasoning Multimodal Large Language Model: Data Contamination and Dynamic Evaluation](http://arxiv.org/abs/2506.07202v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper addresses the growing concern of data contamination in Multimodal Large Language Models (MLLMs), where models may achieve high benchmark scores by memorizing test examples rather than demonstrating genuine generalization. The authors propose a novel dynamic evaluation framework that perturbs the *task* itself, rather than the input.  This involves evaluating MLLMs on a family of related tasks (QA, captioning, question generation, answer verification) for the same visual input.  The rationale is that models overfitting or contaminated for a single task will show a significant performance drop when shifted to other related tasks, unlike models with more generalizable understanding. The framework includes an automated pipeline and a calibrated "judge" model to score open-ended generations. Experiments on various benchmarks analyze each model's cross-task "ability vector," revealing how fine-tuning on simulated contaminated data significantly sharpens task-specific performance but harms overall generalization.

**Critical Evaluation:**

*   **Novelty:** The core idea of perturbing the *task* instead of the input is the paper's key contribution. This approach offers a fresh perspective on dynamic evaluation, moving beyond existing methods that focus on input variations, which often suffer from limitations (minor changes being ineffective, large changes altering the task's semantic intent). The concept of "task-space sharpness" as an indicator of overfitting or contamination is also a novel contribution, providing a theoretical framework for understanding the observed performance variations.

*   **Significance:** The paper addresses a critical issue in MLLM evaluation – the unreliability of static benchmarks due to data contamination. The proposed framework offers a more rigorous and informative way to assess the true generalization ability of MLLMs. Demonstrating how fine-tuning on contaminated data impacts cross-task performance provides concrete evidence of the limitations of relying solely on standard benchmark scores. The findings encourage the development of models that learn genuine visual understanding instead of exploiting spurious correlations.

*   **Strengths:**
    *   **Well-defined framework:**  The methodology is clearly explained, including the task family design, automated evaluation pipeline, and metrics for assessing robustness.
    *   **Theoretical grounding:**  The connection to loss landscape sharpness provides a solid theoretical basis for the approach, explaining why task perturbation reveals contamination.
    *   **Comprehensive evaluation:**  The framework is applied to a diverse set of MLLMs and benchmarks (image and video), providing broad insights into model behavior.
    *   **Controlled experiment:**  The contamination simulation (fine-tuning on test data) effectively demonstrates the framework's ability to detect overfitting.

*   **Weaknesses:**
    *   **Judge model bias:** The automated evaluation of generative tasks relies on a reasoning MLLM as a "judge."  This introduces a potential for bias based on the judge model's own training data and biases. While the authors calibrate the judge, some residual bias likely remains, influencing the scoring of captions and questions.
    *   **Task family limitations:** The four-task family, while well-chosen, may not capture all aspects of MLLM reasoning abilities. Expanding the task repertoire (e.g., including more complex reasoning tasks, explanation generation) could provide an even more comprehensive assessment.
    *   **Manual task augmentation:** The process for generating different types of tasks is currently manual. Automating this would allow testing with a greater number of different task types.

*   **Potential Influence:** The paper's framework has the potential to become a standard methodology for evaluating MLLM generalization ability. It encourages the development of evaluation metrics that go beyond simple accuracy scores, emphasizing robustness and cross-task consistency. The work also highlights the importance of considering data contamination when interpreting benchmark results, potentially influencing how future models are trained and evaluated.

**Justification of Score:**

The paper offers a novel and well-executed approach to dynamic MLLM evaluation, directly addressing a critical limitation of existing benchmarks. The task perturbation framework and the associated theoretical analysis (task-space sharpness) are valuable contributions. The comprehensive experiments and controlled contamination simulation demonstrate the effectiveness of the method. The main weaknesses lie in the potential bias introduced by the judge model and limitations in task family and a manual process for creating the tasks. However, these are acknowledged by the authors.

Score: 8

- **Score**: 8/10

### **[HOI-PAGE: Zero-Shot Human-Object Interaction Generation with Part Affordance Guidance](http://arxiv.org/abs/2506.07209v1)**
- **Summary**: Here's a summary and critical evaluation of the HOI-PAGE paper:

**Summary:**

The paper introduces HOI-PAGE, a novel zero-shot approach for generating 4D human-object interactions (HOIs) from text prompts.  The key innovation lies in the use of Part Affordance Graphs (PAGs), which are distilled from Large Language Models (LLMs) to guide the HOI synthesis process.  PAGs represent fine-grained, part-level affordances between human body parts and object parts, encoding contact relations and motion constraints. The method consists of three stages: (1) object part segmentation, (2) HOI reference video synthesis guided by the PAG, and (3) 4D HOI motion optimization using the PAG to enforce contact and realism. The method demonstrates results on a variety of interaction scenarios, including multi-person and multi-object cases, and shows improved realism and text alignment compared to existing methods.

**Critical Evaluation:**

*   **Novelty:** The core novelty of the paper stems from the explicit modeling of part-level affordances to guide HOI generation in a zero-shot setting.  While prior work has explored HOI generation and the use of LLMs for visual understanding, the combination of part-level affordance graphs, their distillation from LLMs, and their use as a structured constraint within a comprehensive HOI synthesis pipeline represents a significant advance. It allows handling more complex interactions not easily covered by the previous interaction paradigms.  The decomposition into the three stages, each leveraging the PAG, also demonstrates a well-structured and thoughtful design.

*   **Significance:** The significance of HOI-PAGE lies in its ability to generate realistic and diverse HOI motions without requiring expensive and limited 4D interaction datasets for training. This has several implications:

    *   **Generalization:** It allows the synthesis of interactions with novel objects and scenarios that were previously inaccessible.
    *   **Flexibility:** The PAG-guided approach enables modeling of complex interactions involving multiple people and objects, which existing methods often struggle with.
    *   **Accessibility:**  By removing the need for large datasets, HOI-PAGE makes HOI synthesis more accessible to researchers and practitioners.
*   **Strengths:**

    *   **Strong results:**  Qualitative and quantitative results, including perceptual studies, demonstrate the effectiveness of the approach in generating more realistic and text-aligned HOIs.
    *   **Comprehensive approach:** The method addresses multiple aspects of HOI generation, including object segmentation, motion synthesis, and optimization.
    *   **Clear presentation:**  The paper is well-written and clearly explains the method and its components. The supplementary material is also well organized, and the analysis comprehensive.
*   **Weaknesses:**

    *   **Reliance on LLMs:** The quality of the PAGs depends on the reasoning capabilities of the underlying LLM. Errors or biases in the LLM could propagate through the synthesis pipeline. Further, there is a black-box character to this process which limits the ability to tune or interpret the generated PAGs.

    *   **Computational Cost:**  The optimization process and the use of video diffusion models can be computationally expensive. This could limit its applicability to real-time applications.

    *   **Lack of fine-grained control:** While the PAGs provide part-level guidance, capturing detailed motions beyond the part level (e.g., individual finger articulations) remains a challenge.
*   **Potential Impact:** HOI-PAGE has the potential to significantly impact several fields, including:

    *   **Character animation:**  Generating realistic character animations interacting with objects.
    *   **Virtual reality/Augmented reality (VR/AR):** Creating immersive virtual environments with believable interactions.
    *   **Robotics:** Training robots to perform complex tasks involving object manipulation.

*   **Rigorous Rationale:**  While HOI-PAGE addresses an important problem with a well-designed solution and delivers promising results, some of the steps still depend on expensive and less-controllable large language models. Furthermore, the limitations regarding extremely nuanced and delicate motions mean that the interactions, whilst better and more semantically relevant than SoTA, are not fully resolved. However, the zero-shot nature of this approach, its ability to generalize to novel objects, and the explicit modeling of part affordances makes it a strong contender, and highly relevant for these new challenges.

**Score: 8**

- **Score**: 8/10

### **[Hallucination at a Glance: Controlled Visual Edits and Fine-Grained Multimodal Learning](http://arxiv.org/abs/2506.07227v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "HALLUCINATION AT A GLANCE: Controlled Visual Editing and Fine-Grained Multimodal Learning":

**Summary:**

The paper addresses the issue of multimodal large language models (MLLMs) struggling with fine-grained visual differences, leading to hallucinations and missed semantic shifts. The authors attribute this to limitations in training data and learning objectives.  They propose a controlled data generation pipeline to create minimally edited image pairs with semantically aligned captions, resulting in the Micro Edit Dataset (MED).  They also introduce a supervised fine-tuning (SFT) framework with a feature-level consistency loss to promote stable visual embeddings. They evaluate their approach on a new benchmark, the Micro Edit Detection benchmark, and show improvements in difference detection accuracy and reduced hallucinations compared to strong baselines, including GPT-4o, as well as gains on standard vision-language tasks.

**Critical Evaluation:**

*   **Novelty:** The paper exhibits good novelty across several dimensions:

    *   **MED Dataset:** The construction of the MED dataset, with its focus on minimal visual edits and aligned captions, fills a significant gap in the availability of training data for MLLMs.  Existing datasets often lack this controlled variability.
    *   **Micro Edit Detection Benchmark:** The careful design of the benchmark, with balanced evaluation pairs testing sensitivity to subtle visual variations, provides a valuable tool for assessing the fine-grained reasoning abilities of MLLMs. The inclusion of both synthetic and real-world samples enhances its robustness.
    *   **Feature Consistency Regularization:** The introduction of a feature consistency regularization term in the SFT framework is a novel approach to encouraging stable visual embeddings under small edits.

*   **Significance:** The paper tackles a crucial challenge in MLLM research: improving the ability to understand and reason about fine-grained visual details.  This ability is essential for deploying MLLMs in real-world applications that require high precision, such as robotics, industrial quality control, and medical imaging. The demonstrated improvements on both edit detection and standard vision-language tasks suggest that the proposed approach has the potential to significantly enhance the reliability and robustness of MLLMs. The open sourcing of the code and the dataset significantly increases its value to the research community.

*   **Strengths:**

    *   **Well-defined Problem:** The paper clearly identifies and articulates the problem of MLLM brittleness in the face of fine-grained visual changes.
    *   **Comprehensive Approach:** The authors address the problem through both data augmentation and a novel training objective.
    *   **Strong Empirical Results:** The experimental results on both the Micro Edit Detection benchmark and standard vision-language tasks provide compelling evidence for the effectiveness of the proposed approach. The comparisons to strong baselines, including GPT-4o, further highlight the significance of the improvements.
    *   **Thorough Analysis:**  The ablation study provides insights into the contributions of different components of the proposed framework.
    *   **Generalization:** The real world data augmentation provides some support of generalization.

*   **Weaknesses:**

    *   **Synthetic Data Overfitting:** A potential concern is the reliance on synthetic data generated by a controlled editing pipeline. While the inclusion of real-world image pairs in the evaluation mitigates this concern to some extent, further investigation into the generalizability of the approach to more diverse and uncontrolled visual changes would be valuable. It is possible that performance could be artificially inflated by overfit on the biases present in this pipeline.
    *   **Limited Scope:** The framework focuses primarily on binary edits between image pairs. Extending the approach to handle multi-step transformations, temporal reasoning, and compositional edits would broaden its applicability.
    *   **Computational Cost:** The use of full fine-tuning can be computationally expensive. Exploring more efficient adaptation methods would be beneficial.
    *   **Reliance on specific MLLMs:** It will be useful to establish whether other MLLMs can benefit from this approach.

*   **Potential Influence:** The paper is likely to have a significant influence on the field of multimodal learning.  The MED dataset and Micro Edit Detection benchmark provide valuable resources for researchers working to improve the fine-grained visual reasoning abilities of MLLMs. The proposed SFT framework with feature consistency regularization offers a promising approach to addressing the challenge of hallucination and improving robustness.

*   **Rigorous Rationale:** The score is based on the paper's notable novelty in its dataset creation and training objective, its significance in tackling a critical MLLM limitation, its strong empirical results, and its potential to influence future research in the field. While the reliance on synthetic data and limited scope represent weaknesses, the strengths of the paper outweigh these limitations.

**Score: 8.5**

- **Score**: 8/10

### **[Multi-Step Visual Reasoning with Visual Tokens Scaling and Verification](http://arxiv.org/abs/2506.07235v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Multi-Step Visual Reasoning with Visual Tokens Scaling and Verification":

**Summary:**

The paper addresses the limitations of current Multi-modal Large Language Models (MLLMs) in visual reasoning due to their static inference paradigm, where the entire image is encoded into fixed visual tokens upfront. To overcome this, the authors introduce a novel framework for inference-time visual token scaling that enables MLLMs to perform iterative, verifier-guided reasoning over visual content. They formulate the problem as a Markov Decision Process (MDP) with a reasoner (proposing visual actions) and a verifier (evaluating actions, trained using multi-step Direct Preference Optimization (DPO)). The paper also presents a new dataset, VTS, comprising supervised reasoning trajectories (VTS-SFT) and preference-labeled reasoning comparisons (VTS-DPO) to facilitate training. Experimental results on diverse visual reasoning benchmarks demonstrate significant improvements over existing approaches, alongside more interpretable reasoning processes.

**Critical Evaluation:**

**Novelty:**

The paper presents a novel framework for dynamic visual token scaling within MLLMs.  The key innovations include:

*   **Iterative Reasoning:** The core idea of allowing the model to interactively refine its visual understanding is a significant departure from the static approach of most MLLMs.
*   **MDP Formulation:** Framing visual reasoning as an MDP provides a structured approach to action selection and termination.
*   **Verifier with DPO:** Using a verifier trained with multi-step DPO to guide reasoning is a sound approach, leveraging preference learning to improve action selection and determine when reasoning should stop.
*   **VTS Dataset:** The creation of a tailored dataset (VTS) containing both supervised trajectories and preference data directly addresses the lack of suitable training data for this type of iterative visual reasoning.

The combination of these elements demonstrates significant novelty.

**Significance:**

The paper's significance lies in its potential to unlock more sophisticated visual reasoning capabilities in MLLMs. Current MLLMs often struggle with tasks requiring fine-grained visual analysis or context-aware interpretation. By enabling dynamic visual token scaling, the proposed framework allows models to:

*   **Focus attention:** Selectively process relevant regions of an image, improving efficiency and accuracy.
*   **Adapt to context:** Refine their understanding of the visual scene based on the ongoing reasoning process.
*   **Handle ambiguity:**  Recover from initial misinterpretations by re-examining the image with different tools or perspectives.

The demonstrated improvements on challenging benchmarks (BLINK, V\*Bench, MMStar, MathVista) support these claims. The enhanced interpretability of the reasoning process is also valuable, providing insights into how the model arrives at its conclusions.

**Strengths:**

*   **Clear Problem Definition:** The paper effectively highlights the limitations of current MLLMs in visual reasoning.
*   **Well-Motivated Approach:** The proposed framework addresses the identified limitations in a logical and principled manner.
*   **Comprehensive Evaluation:** The experiments cover a diverse set of visual reasoning tasks and strong baselines.
*   **Detailed Analysis:** The ablation studies provide valuable insights into the contributions of visual token scaling and the verifier.
*   **Reproducibility:** The public release of code and data promotes reproducibility and future research.

**Weaknesses:**

*   **Computational Cost:** Dynamic token scaling can be computationally expensive.  The paper should discuss the computational overhead in more detail and explore potential optimization strategies.
*   **Generalization of Actions:** The action space is limited to the tools provided.  How can the framework be extended to allow the model to learn new tools or combine existing ones in novel ways?
*   **Dataset Dependence:** The framework relies on the VTS dataset for training.  How well would it generalize to other datasets or domains?
*   **Complexity of MDP:** The MDP formulation may add complexity to the models and also increase the overall memory footprint

**Potential Influence:**

The paper has the potential to significantly influence the development of future MLLMs.  By demonstrating the benefits of dynamic visual token scaling, it could encourage researchers to explore more interactive and context-aware inference mechanisms. The VTS dataset could also serve as a valuable resource for training and evaluating new visual reasoning models.

**Justification for Score:**

The paper offers a significant advancement in visual reasoning for MLLMs. The proposed framework tackles a key limitation of existing models with a novel and well-designed approach. The creation of the VTS dataset and comprehensive experiments further strengthens the contribution. While there are some weaknesses related to computational cost and generalization, the potential impact on the field is considerable.

Score: 8

- **Score**: 8/10

### **[Parsing the Switch: LLM-Based UD Annotation for Complex Code-Switched and Low-Resource Languages](http://arxiv.org/abs/2506.07274v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces BiLingua Parser, an LLM-based pipeline designed to generate Universal Dependencies (UD) annotations for code-switched text, focusing on Spanish-English and Spanish-Guaraní. The authors develop a prompt-based framework using few-shot LLM prompting with expert review, and they release two annotated datasets, including the first Spanish-Guaraní UD-parsed corpus. A detailed syntactic analysis of switch points across language pairs and communicative contexts is performed.  The results show that BiLingua Parser achieves up to 95.29% LAS after expert revision, outperforming existing baselines and multilingual parsers. The paper argues that LLMs, when guided carefully, can be used to bootstrap syntactic resources in under-resourced, code-switched environments.

**Critical Evaluation:**

*   **Novelty:** The paper's novelty lies in several key areas: 1) Introducing an LLM-based approach for UD annotation of code-switched data; 2) Creating and releasing new UD-annotated datasets for Spanish-English and, significantly, Spanish-Guaraní (a low-resource pair); 3) Performing a detailed syntactic analysis of switch points beyond POS tags, focusing on dependency relations; and 4)  Demonstrating that careful LLM prompting and expert validation can produce high-quality syntactic annotations, even exceeding traditional methods. The key here is the application of LLMs to syntactic *parsing* of code-switched text, rather than just sequence-level tagging, alongside a resource contribution for a genuinely under-served language pair. Previous work utilized LLMs for tagging but did not provide extensive syntactic analysis nor focused on low-resource code-switching to this extent.
*   **Significance:**  The significance of this work is multi-faceted. Firstly, it provides valuable linguistic resources for code-switching research, specifically for a low-resource language like Guaraní where such resources are scarce. Secondly, the paper demonstrates a practical approach to bootstrapping syntactic resources in data-scarce environments. This has implications beyond code-switching and could be applied to other under-resourced languages or specialized domains. Thirdly, the syntactic analysis sheds light on the switching behaviors across different language pairs, offering insights into the structural properties of code-switching phenomena. Fourthly, it shows a methodology of using LLMs to generate UD datasets, thus providing an alternate method of creating such datasets.

*   **Strengths:**
    *   **High Accuracy:**  The reported LAS scores are impressive, particularly after expert revision, demonstrating the potential of LLMs in this task.
    *   **Resource Contribution:** The release of the Spanish-Guaraní UD corpus is a significant contribution to the field, addressing a gap in available resources.
    *   **Detailed Analysis:** The syntactic analysis of switch points provides valuable insights into code-switching behavior beyond POS tags.
    *   **Well-Defined Methodology:**  The prompt-based framework and expert review process are clearly described and replicable.
    *   **Focus on Low-Resource Setting:** The emphasis on Spanish-Guaraní code-switching adds to the significance by addressing an important gap in resources and techniques for languages beyond the common English-centric perspective.

*   **Weaknesses:**
    *   **LLM Dependence:** The approach relies on the performance and availability of powerful LLMs (GPT-4.1). Changes or limitations to the model could impact the pipeline's performance.
    *   **Expert Review Bottleneck:** The expert review process, while critical for accuracy, could be a bottleneck in scaling up the annotation pipeline. The paper could explore alternative methods for validating LLM annotations.
    *   **Inconsistency in LLM's outputs:** Although the prompt provides explicit guidelines, the LLM remains inconsistent. More work can be done for the LLM to consistently apply the rules.
    *   **Limited Evaluation Datasets:** It would be beneficial to evaluate the parser on more languages.

*   **Potential Influence:** The paper has the potential to influence the field by:

    *   Encouraging researchers to explore LLM-based approaches for syntactic annotation in low-resource languages.
    *   Providing a framework for creating high-quality UD annotations with limited data.
    *   Inspiring further research into the structural properties of code-switching using dependency relations.
    *   Supporting the development of more accurate and robust code-switching parsers.

* **Critical Evaluation of Experimental Methodology**
    *   The authors introduce a novel strategy for evaluating the accuracy of the generated UD annotations. Since there is an absence of any golden datasets for the considered languages, the strategy of using the LLM generated annotations as a reference standard following manual revision helps to rigorously benchmark the performance of LLMs on a complex task.

* **Future Directions for the Paper**
    *   The authors mention their work in integrating the LLM evaluation into the annotation pipeline that reduces the manual supervision bottleneck for scaling up the annotation.
    *   The authors also intend to further the analysis with manual validation.

**Overall Score:**

I assign a score of **8**. The paper makes a significant contribution to the field by providing a new approach to syntactic annotation of code-switched data, specifically addressing the challenges of low-resource languages. The release of the Spanish-Guaraní UD corpus is a valuable resource, and the detailed analysis of switch points offers new insights. While the reliance on LLMs and the expert review process present some limitations, the paper demonstrates the potential of these techniques and provides a solid foundation for future research.

Score: 8

- **Score**: 8/10

### **[Reward Model Interpretability via Optimal and Pessimal Tokens](http://arxiv.org/abs/2506.07326v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper explores the interpretability of reward models (RMs) used in aligning large language models (LLMs) with human values. Instead of treating RMs as mere intermediaries in the RLHF process, the authors propose a method of exhaustive analysis of their responses across the entire vocabulary space. This involves feeding value-laden prompts into RMs and examining how they score every possible single-token response.  The authors analyze ten open-source reward models from the REWARDBENCH leaderboard, revealing significant heterogeneity between models trained on similar objectives, asymmetries in how models encode high- vs. low-scoring tokens, sensitivity to prompt framing that mirrors human cognitive biases, and overvaluation of more frequent tokens.  The study also uncovers that RMs can encode biases toward certain identity groups, potentially stemming from harmlessness training objectives. Finally, the paper demonstrates how to utilize Greedy Coordinate Gradient optimization to examine biases within multi-token sequences.

**Critical Evaluation:**

*   **Novelty:** The paper introduces a novel approach to analyzing reward models, focusing on exhaustive vocabulary analysis. This provides a granular understanding of how RMs interpret human values, moving beyond merely using them for fine-tuning. The use of techniques like Greedy Coordinate Gradient to further explore multi-token biases is also innovative.
*   **Significance:** The findings of this paper are significant for several reasons. First, it challenges the assumption that reward models trained on similar objectives are interchangeable, highlighting substantial heterogeneity. Second, the discovery of biases towards specific identity groups and the potential "mere exposure effect" in RMs raises serious concerns about their suitability as proxies for human values. This is especially concerning because these RMs drive the behavior of LLMs now deployed on a wide scale.
*   **Strengths:**
    *   The paper employs a rigorous methodology, combining exhaustive vocabulary analysis with quantitative measures like correlation and multidimensional scaling.
    *   The analysis is comprehensive, considering multiple open-source reward models of varying sizes and architectures.
    *   The paper makes a strong connection to the broader literature on cognitive biases and hate speech detection, lending credibility to the findings.
    *   The framing analysis adds a layer of sophistication, showing that RMs are sensitive to the way prompts are phrased, mirroring human behavior.
    *   The use of an independent human baseline (ELOEVERYTHING) adds further weight to the findings and highlights discrepancies between model and human preferences.
*   **Weaknesses:**
    *   The analysis is primarily focused on single-token responses, which may not fully capture the complexity of real-world interactions. While the authors do extend the analysis to multi-token sequences via GCG, the computational limitations restrict the scope of this part of the study.
    *   The paper acknowledges the lack of perfect alignment between ELOEVERYTHING users and the training population for the reward models. It is possible that at least some of the findings reflect divergences between the populations used to generate human preferences, rather than true failings of the reward models.
    *   While the paper highlights concerning biases in the lower-ranked tokens, it's important to consider the intended use of reward models. They are primarily used to maximize scores, so understanding the upper tail of the distribution is arguably more crucial.
    *   The paper could benefit from a deeper exploration of the causes of the observed biases. While it speculates about "harmlessness training objectives," further investigation into the training data and processes would strengthen the conclusions.
*   **Potential Influence:** The paper's findings could significantly impact the way reward models are developed and evaluated. It calls for more careful consideration of potential biases and framing effects, as well as a move towards more robust and representative training data. This work could also lead to the development of new techniques for analyzing and mitigating biases in RMs.

**Justification for Score:**

This paper makes a significant contribution by offering a novel methodology for interpreting reward models, a critical component of aligned LLMs. The findings reveal concerning biases and limitations in existing models, with the potential for real-world harm through their impact on downstream LLMs. While there are some limitations in scope and depth of analysis, the strengths of the paper outweigh the weaknesses, making it a valuable contribution to the field of AI safety and interpretability.

Score: 8

- **Score**: 8/10

### **[Graph-KV: Breaking Sequence via Injecting Structural Biases into Large Language Models](http://arxiv.org/abs/2506.07334v1)**
- **Summary**: Here's a summary and evaluation of the paper, including a critical assessment of its novelty and significance:

**Summary:**

The paper introduces Graph-KV, a novel method to inject structural inductive biases into Large Language Models (LLMs) to improve their performance on tasks involving structured data. Unlike standard auto-regressive LLMs that require input to be serialized into a flat sequence, Graph-KV leverages the KV-cache mechanism to represent text segments and govern their interactions based on structural relationships. This approach involves selectively attending to designated "source" segments for "target" segments, inducing a graph-structured block mask that sparsifies attention. The paper also addresses positional bias through strategic allocation of positional encodings. Graph-KV is evaluated on a range of tasks, including retrieval-augmented generation (RAG) benchmarks, a new academic paper QA task (ARXIV-QA), and paper topic classification within a citation network. The results demonstrate that Graph-KV outperforms baselines, including standard sequential encoding, by effectively reducing positional bias and leveraging structural inductive biases.

**Critical Evaluation:**

*   **Novelty:** The core novelty of the paper lies in its direct integration of structural information into the LLM's attention mechanism. While parallel encoding and other methods to reduce positional bias exist, Graph-KV offers a more foundational approach that explicitly models inter-segment dependencies. The idea of using a graph-structured block mask within the KV-cache to enforce specific attention patterns is innovative and shows how structural biases can be inserted. This moves beyond purely eliminating positional bias and starts harnessing the inherent structure of data within the LLM architecture itself.
*   **Significance:** The significance is multi-faceted:

    *   **Performance Improvement:** The demonstrated improvements across diverse tasks like RAG, QA on scientific papers, and citation networks suggest the general applicability of the approach. The ARXIV-QA task, in particular, seems well-suited to highlight Graph-KV's strengths given its inherent graph structure.
    *   **Efficiency:** The method is designed to maintain computational efficiency by sparsifying the attention mechanism and reducing context window consumption. This is crucial for scaling to real-world datasets.
    *   **Addressing a Core Limitation:** Graph-KV addresses a fundamental limitation of LLMs – their reliance on serialization, which often hinders their ability to reason effectively about structured information. This is a significant step towards making LLMs more adept at handling complex, relational data.
*   **Strengths:**

    *   **Clear Problem Definition:** The paper clearly articulates the challenges of serialization in LLMs and the resulting limitations for structured data.
    *   **Well-Defined Method:** Graph-KV is presented in a clear and concise manner, outlining the structure-aware attention mechanism and positional encoding strategy.
    *   **Comprehensive Evaluation:** The experiments cover a wide range of tasks and datasets, providing strong evidence for the effectiveness of Graph-KV. The comparison against strong baselines, including sequential encoding and other parallel encoding methods, is well-executed. The ARXIV-QA dataset is also a valuable contribution in itself.
    *   **Focus on Underlying Mechanisms:** Rather than simply improving performance, the paper aims to directly manipulate the LLM's attention mechanism to integrate structural information, which could have broader implications for future research.
*   **Weaknesses:**

    *   **Single-Hop Dependency:** The experiments focus mainly on single-hop structural dependencies. While the paper acknowledges the potential for iterative application of the method to model multi-hop relationships, it does not provide substantial empirical evidence for this extension.
    *   **Limited Fine-Tuning Exploration:** The paper uses a model already pre-trained with block attentions, thus limiting the ability to fully assess the capabilities of the KV-graph structure. Fine-tuning Graph-KV more directly might have revealed even greater performance improvements.
    *   **Reliance on Bipartite Graphs:** The RAG experiments construct a bipartite graph. Other methods of determining which nodes to attend to could be explored.
    *   **Limited LLM Exploration:** The evaluation is limited to the Llama-3.1-8B family. Assessing the performance of Graph-KV with other LLM architectures would further strengthen the generalizability of the findings.

*   **Potential Influence:** Graph-KV has the potential to influence future research in several ways:

    *   **Structured LLMs:** It provides a blueprint for integrating structural inductive biases directly into LLM architectures.
    *   **RAG and Knowledge-Intensive Tasks:** It offers a promising approach for improving the performance of LLMs on RAG and other knowledge-intensive tasks that rely on external information sources.
    *   **Graph Learning with LLMs:** It opens up new avenues for applying LLMs to graph learning tasks, providing a more efficient and effective alternative to existing methods.

**Score: 8**

**Justification:**

Graph-KV represents a significant contribution to the field by directly addressing the limitation of LLMs in handling structured data. The novelty lies in the design of the graph-structured block mask and the demonstration of its effectiveness across several tasks. The improvements are substantial, especially in scenarios that require reasoning over structural relationships, and they demonstrate a deeper understanding of internal mechanisms. The main weaknesses include limited exploration of multi-hop dependencies, other graph types, and the lack of a fine-tuning stage. These limitations keep the score from being higher, but they also present clear directions for future work. Overall, the paper is well-written, well-executed, and has the potential to significantly influence the development of LLMs for structured data.

- **Score**: 8/10

### **[Refusal-Feature-guided Teacher for Safe Finetuning via Data Filtering and Alignment Distillation](http://arxiv.org/abs/2506.07356v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces "Refusal-Feature-guided Teacher" (ReFT), a novel approach to safe finetuning of large language models (LLMs) in a Finetuning-as-a-Service setting.  ReFT aims to mitigate the risk of degrading LLM safety alignment when users finetune models with potentially harmful data. The key idea is to leverage a "refusal feature," a directional representation obtained from a safety-aligned LLM that distinguishes between harmful and harmless prompts. A ReFT model is trained to identify harmful prompts based on their similarity to this refusal feature. During finetuning, the ReFT model acts as a teacher, filtering harmful prompts from the user's data and distilling alignment knowledge into the base model through soft labels.  Experiments demonstrate that ReFT effectively minimizes harmful outputs while preserving accuracy on user-specific tasks.

**Critical Evaluation:**

*   **Novelty:** The paper's novelty lies in several aspects:

    *   **Refusal Feature Analysis:**  The analysis and explicit utilization of the refusal feature for prompt classification is a valuable contribution. While the concept of a refusal feature isn't entirely new, the paper's in-depth examination and demonstration of its effectiveness for harmful prompt detection is a notable advance.
    *   **ReFT Architecture and Training Procedure:** The ReFT model and its two-stage training (teacher preparation and finetuning) are well-defined and contribute to the paper's novelty. The dynamic update of the refusal feature during ReFT training, addressing the lack of pre-aligned models, is an important practical consideration.
    *   **Finetuning Strategy:** The integration of data filtering and alignment knowledge distillation using the ReFT model as a teacher during finetuning is a novel approach to safe LLM finetuning.

*   **Significance:** The paper addresses a critical challenge in the increasingly popular Finetuning-as-a-Service paradigm. The ability to safely and reliably customize LLMs is crucial for wider adoption, and the ReFT approach offers a practical solution. The experimental results consistently demonstrate that ReFT achieves better safety (lower harmful scores) and maintains or improves user-specific task performance compared to existing baselines. This combination of safety and utility is highly significant.

*   **Strengths:**

    *   **Strong Empirical Results:** The extensive experiments across various harmful prompt ratios, user data sizes, datasets, and model architectures provide compelling evidence for the effectiveness and generalizability of the ReFT approach. The ablation studies dissecting the contributions of filtering and alignment distillation offer valuable insights.
    *   **Practical Relevance:** The paper's focus on a practical Finetuning-as-a-Service scenario and the development of a deployable solution enhance its relevance.
    *   **Clear Presentation:** The paper is well-written and organized, making it easy to understand the proposed approach and the experimental setup.

*   **Weaknesses:**

    *   **Reliance on Refusal Feature:** The approach's performance is inherently tied to the quality of the refusal feature. While the paper addresses the lack of pre-aligned models, the initial stages of ReFT training may still be susceptible to instability if the initial refusal feature is not well-defined. The paper could further explore the sensitivity of ReFT to the quality of the dataset used for generating the initial refusal feature, especially if the alignment dataset isn't perfect.
    *   **Adversarial Robustness of ReFT:** The paper addresses the issue of adversarial prompts *during finetuning*. However, if the ReFT classifier itself is vulnerable to adversarial attacks it is a limitation, as acknowledged. While the paper shows robustness against GCG and AutoDAN in the fine-tuning stage, the vulnerability of the ReFT model itself needs more robust demonstration.

*   **Justification of Score:**
The contributions of this paper are strong in that they solve a well-defined and important problem with a novel solution that has been shown to be effective across a large number of experiments. However, the reliance on the underlying refusal feature and the somewhat cursory treatment of the ReFT models vulnerability is the main reason for not granting the highest scores.

Score: 8

- **Score**: 8/10

### **[ARGUS: Hallucination and Omission Evaluation in Video-LLMs](http://arxiv.org/abs/2506.07371v1)**
- **Summary**: Here's a summary and evaluation of the paper:

**Summary:**

The paper introduces ARGUS, a new benchmark designed to evaluate hallucination and omission errors in Video Large Language Models (VideoLLMs).  Current benchmarks often rely on multiple-choice questions, which the authors argue don't accurately reflect a VideoLLM's performance on free-form text generation tasks like video captioning where hallucinations are more prevalent. ARGUS measures performance by comparing VideoLLM-generated captions to human-annotated ground truth, quantifying both hallucination (incorrect statements about video content or temporal relationships) and omission (failure to include important details). The framework uses entailment analysis (LLM-as-a-judge) to quantify hallucinations and assesses omissions by identifying statements in human captions that are missing in model-generated captions. They utilize dynamic programming to handle temporal inconsistencies, and normalize the costs for fair comparison.  The paper includes a dataset curated from existing sources and new annotations, and evaluations of a range of open and closed-source VideoLLMs. The authors also provide sensitivity analyses related to prompts, frame rates, and the choice of the LLM used as a judge, showcasing the robustness of the benchmark.

**Critical Evaluation:**

The paper addresses a significant gap in the evaluation of VideoLLMs: the propensity to hallucinate, especially during open-ended tasks like video captioning. Current QA-based benchmarks are insufficient to measure this accurately.

**Strengths:**

*   **Novelty:** The core idea of using free-form captioning and measuring both hallucination and omission is novel for VideoLLMs. The focus on free-form *generation* rather than verification is a key differentiator.
*   **Significance:** The creation of a benchmark specifically designed to measure the quality of captions that are both accurate and comprehensive is highly significant. This will encourage the development of VideoLLMs which can generate more accurate and reliable textual descriptions of videos. This has implications for accessibility.
*   **Thoroughness:** The authors conduct a thorough evaluation of numerous models (both open-source and proprietary) using the ARGUS benchmark and evaluate several aspects such as frame rate, caption length and prompting strategies, making this quite a detailed study.
*   **Robustness Analysis:** Addressing the sensitivity of the framework to prompt variation and the choice of LLM-as-judge strengthens the credibility of the results. Using a human study to validate LLM judgments further reinforces the framework.
*   **Dataset Curation:** The paper details the composition of the dataset, and the manual verification steps that were undertaken.

**Weaknesses:**

*   **Dependence on LLM Judge:** While the authors acknowledge and address the sensitivity of their framework to prompt engineering for the LLM as well as choice of the LLM judge through ablation and sensitivity analyses ( demonstrating high correlation between LLM's, suggesting robustness), the entire framework still depends on the accuracy of another LLM. While the LLM is prompted as best as possible, this LLM is itself imperfect and this can still result in misclassifications. It's hard to imagine a practical alternative to avoid this, but is worth keeping in mind.
*   **Complexity:** The pipeline and evaluation metrics are rather complex, which may limit its broader adoption. Simplification while maintaining effectiveness would be a plus.
*   **Limited novelty of the LLM judge technique:** While novel to videos, the technique of using an LLM to measure entailment between captions is related to several works on Image and other forms of captioning. In this area, the novelty of this technique can be debated.

**Justification of Score:**

The ARGUS benchmark is a valuable contribution to the field of VideoLLMs. Its focus on measuring hallucination and omission in free-form captioning addresses a critical gap in current evaluation methodologies. The thorough experimental results and robustness analysis demonstrate the reliability of the benchmark and highlight its potential for driving future research in this area. While there are concerns about complexity and LLM reliance, the overall impact of the work is considerable, especially on the VideoLLM generation task.

Score: 8.5

- **Score**: 8/10

### **[Chasing Moving Targets with Online Self-Play Reinforcement Learning for Safer Language Models](http://arxiv.org/abs/2506.07468v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Chasing Moving Targets with Online Self-Play Reinforcement Learning for Safer Language Models":

**Summary:**

The paper introduces SELF-REDTEAM, a novel online self-play reinforcement learning (RL) algorithm designed to improve the safety alignment of language models (LMs).  Unlike traditional approaches where attack and defense phases are separate and iterative, SELF-REDTEAM allows an attacker and defender agent to co-evolve dynamically through continuous interaction.  The method casts LM safety alignment as a two-player, zero-sum game where a single LM alternates between attacker (generating adversarial prompts) and defender (safeguarding against them).  A reward LM adjudicates outcomes.  The authors provide a theoretical safety guarantee: if self-play converges to a Nash Equilibrium, the defender will reliably produce safe responses.  Empirically, SELF-REDTEAM discovers more diverse attacks and achieves higher robustness on safety benchmarks compared to methods trained with static attackers and defenders. They further enhance the method with hidden Chain-of-Thought, enabling private planning for agents, boosting adversarial diversity and reducing over-refusal rates.

**Critical Evaluation:**

*   **Novelty:** The core novelty lies in the *online, continuous co-evolution* of attacker and defender roles within a single LM, driven by reinforcement learning.  This is a significant departure from traditional iterative attack-then-defend approaches.  The application of game-theoretic principles to this problem is also a strong point. The Hidden CoT is a nice addition.

*   **Significance:** The paper addresses a critical challenge in the field of language models: ensuring safety and robustness against adversarial attacks. The standard iterative methods are slow and quickly become outdated. The SELF-REDTEAM approach offers a more proactive and adaptive way to train safer LMs. The reported empirical results, demonstrating improved robustness and attack diversity, are strong indicators of its potential impact. The theoretical result, while relying on an assumption of reaching Nash Equilibrium, provides a solid justification for the approach.

*   **Strengths:**

    *   **Online Co-evolution:** The continuous interaction between attacker and defender is a key advantage, promoting adaptation and preventing overfitting to static vulnerabilities.
    *   **Game-Theoretic Foundation:** The formulation as a zero-sum game provides a theoretical grounding for the approach and motivates the design choices.
    *   **Strong Empirical Results:** The experiments demonstrate significant improvements in attack diversity and robustness compared to existing methods.
    *   **Clear and Well-Written:** The paper is generally well-structured and easy to follow.
    *   **Hidden Chain-of-Thought:** Addresses over-refusal issues and increases strategic adversarial planning.
*   **Weaknesses:**

    *   **Assumptions of Nash Equilibrium:**  The theoretical guarantee relies on the assumption that the self-play process converges to a Nash Equilibrium.  This is a strong assumption, and the paper doesn't fully explore whether or how this convergence is achieved in practice.
    *   **Reward Model Dependency:** The performance of SELF-REDTEAM is highly dependent on the quality of the reward LM used to adjudicate outcomes.  If the reward model is flawed, the entire training process can be misguided. Details for the implementation and evaluation are a little sparse.
    *   **Compute Intensity:** RL fine-tuning, in general, is more computationally expensive compared to other fine-tuning strategies, which can limit the accessibility of this method.
    *   **Potential misuse from retained attack capabilities.** The authors briefly touch upon the risk of retaining attack capabilities after training and provide some general solutions. A more detailed investigation into this potential harm and thorough benchmarks on backdoors is warranted.

*   **Potential Influence:**

    *   The paper's approach could shift the focus of LM safety training from reactive patching to proactive co-evolution.
    *   The use of multi-agent RL and game-theoretic principles in this context could inspire new research directions in LM safety.
    *   The empirical results could encourage practitioners to adopt online self-play methods for training more robust LMs.

**Score: 8**

**Justification:**

The paper presents a genuinely novel and promising approach to LM safety alignment. The idea of online co-evolution is significant, and the empirical results provide strong evidence of its effectiveness. While the theoretical assumptions and reward model dependency represent limitations, the overall contribution is substantial. The paper has the potential to influence future research and practice in LM safety training, making it a significant advancement within the field.

- **Score**: 8/10

### **[LeVo: High-Quality Song Generation with Multi-Preference Alignment](http://arxiv.org/abs/2506.07520v1)**
- **Summary**: Here's a summary and critical evaluation of the LeVo paper:

**Summary:**

The paper introduces LeVo, a novel framework for high-quality song generation. LeVo combines a language model (LeLM) with a music codec to generate both mixed (vocals and accompaniment combined) and dual-track (vocals and accompaniment separated) audio tokens in parallel.  This architecture is designed to balance vocal-instrument harmony and sound quality. Key components include: 1) LeLM, which models mixed tokens for high-level song structure and a separate autoregressive (AR) decoder for dual-track tokens to capture finer acoustic details; 2) a three-stage training paradigm (pre-training, modular extension training, multi-preference alignment) to address challenges related to data quality and interference between token types; and 3) a DPO-based multi-preference alignment method to improve musicality and instruction following, leveraging semi-automatic data construction. The experimental results demonstrate that LeVo outperforms existing methods on both objective and subjective metrics.

**Critical Evaluation:**

**Novelty:** The paper introduces several novel aspects:

*   **Parallel Mixed/Dual-Track Token Modeling:** The parallel prediction of mixed and dual-track tokens using a language model with an AR decoder is a core architectural innovation. This approach attempts to overcome limitations of single-token and straightforward dual-track token prediction methods. This has strong novelty.
*   **Modular Extension Training Strategy:** The three-stage training approach (pre-training, modular extension training, and DPO fine-tuning) is designed to prevent interference between token types and improve overall generation quality. The idea to split the model training to better improve each aspect of the model is an innovative training procedure.
*   **DPO-based Multi-Preference Alignment:** The application of Direct Preference Optimization (DPO) with a multi-preference approach to song generation is a significant contribution. The semi-automatic data construction process to gather human preferences is crucial for DPO's success in this context. The application of DPO and the proposed data construction is fairly novel in the space of music generation.

**Significance:**

*   **Improved Song Generation Quality:** The experimental results consistently show that LeVo outperforms existing open-source methods across various objective and subjective metrics.
*   **Addressing Data Limitations:** The DPO-based multi-preference alignment method directly addresses the problem of noisy or limited data by incorporating human preferences into the model. This is particularly important in music generation where data annotation is difficult and subjective.
*   **Potential for Impact:** The potential for improving music creation workflows and democratizing music production is significant. The authors acknowledge the broader ethical considerations in music creation as well.

**Strengths:**

*   **Well-Defined Problem and Approach:** The paper clearly articulates the challenges of song generation and presents a well-structured solution with clearly defined components.
*   **Comprehensive Evaluation:** The evaluation uses both objective and subjective metrics. Comparison to industry systems is good (although limited since those are "black box"). Ablation studies are thorough and support the design choices.
*   **Reproducibility:** The paper provides details on the training setup, model configurations, and data processing pipelines, enhancing the potential for reproducibility, especially compared to the closed-source industry approaches.

**Weaknesses:**

*   **Reliance on Existing Components:** LeVo leverages existing techniques like MuCodec, Whisper, and VAEs. While the integration is novel, the reliance on these building blocks somewhat reduces the overall originality of the *individual* components.
*   **Black-box Industry Comparisons:** Comparisons to industry systems are limited due to the black-box nature of those models. Deeper technical insights would strengthen this analysis.
*   **Limited Ethical Discussion:** While they mention it, the ethical discussion remains brief. Broader societal implications related to the displacement of musicians/artists from the improved quality generated by LeVo should be discussed.

**Justification for the Score:**

LeVo presents a significant advancement in the field of song generation, with novel architectural components (the parallel mixed/dual-track token modeling) and training/alignment strategies (the modular extension and DPO-based multi-preference alignment). The empirical evaluations show clear improvements over existing academic approaches and demonstrate competitive performance compared to leading industry tools. While the reliance on existing components and limited discussions of reproducibility might weaken it, the overall contribution warrants a high score.

The design is particularly strong in attempting to balance vocal-instrument harmony with high sound quality. The modular training and DPO alignment address specific challenges and demonstrate careful consideration of the complexities of song generation.

Score: 8

- **Score**: 8/10

### **[SAFEFLOW: A Principled Protocol for Trustworthy and Transactional Autonomous Agent Systems](http://arxiv.org/abs/2506.07564v1)**
- **Summary**: Here's a concise summary and critical evaluation of the provided paper:

**Summary:**

The paper introduces SAFEFLOW, a novel protocol-level framework designed to enhance the trustworthiness of autonomous agents based on Large Language Models (LLMs) and Vision-Language Models (VLMs). SAFEFLOW enforces fine-grained information flow control (IFC), tracks provenance, and implements transactional execution for robust multi-agent coordination. The framework includes mechanisms for conflict resolution, secure scheduling, write-ahead logging, and rollback to improve resilience. To validate the approach, the authors introduce SAFEFLOWBENCH, a comprehensive benchmark suite that evaluates agent reliability under adversarial, noisy, and concurrent conditions. Experimental results demonstrate that agents built with SAFEFLOW maintain performance and security even in hostile environments, outperforming state-of-the-art methods. The authors propose that SAFEFLOW and SAFEFLOWBENCH offer a foundation for building secure and reliable agent ecosystems.

**Critical Evaluation:**

*   **Novelty:** The paper demonstrates significant novelty in addressing the critical issue of trustworthiness in LLM/VLM-based agents. While prior work has explored specific aspects of agent safety, SAFEFLOW's integrated approach to information flow control and transactional execution represents a substantial advancement. The introduction of SAFEFLOWBENCH as a tailored benchmark for evaluating VLM agents in GUI-based environments also fills an important gap.

*   **Significance:** The paper addresses a pressing challenge in the development of autonomous agents. As LLMs and VLMs become more prevalent in real-world applications, ensuring their reliability and security is paramount. SAFEFLOW provides a practical framework for mitigating vulnerabilities related to prompt injection, adversarial inputs, and concurrency conflicts. By enabling verifiable and auditable agent behavior, SAFEFLOW can increase confidence in the deployment of autonomous systems in high-stakes settings. The contributions can significantly impact the research community, pushing the current limitations of agent safety and reliability forward.

*   **Strengths:**
    *   Comprehensive Approach: SAFEFLOW holistically addresses various aspects of agent trustworthiness, including information flow control, transactional execution, and concurrency management.
    *   Practical Implementation: The framework incorporates practical mechanisms like write-ahead logging and rollback to enhance resilience against runtime errors.
    *   Rigorous Evaluation: SAFEFLOWBENCH is a well-designed benchmark suite that considers adversarial conditions and concurrency challenges, providing a more realistic assessment of agent performance.
    *   Empirical Validation: Experimental results demonstrate that SAFEFLOW effectively improves agent security and robustness without compromising task performance.
    *   Generalization: Demonstrating cross-benchmark robustness using `AgentHarm` benchmark shows the capability of the proposed method in general adversarial conditions.

*   **Weaknesses:**
    *   Computational Overhead: The implementation of fine-grained information flow control and transactional execution may introduce computational overhead, which could affect the scalability and real-time performance of agents.
    *   Model Dependency: The performance of SAFEFLOW is likely influenced by the capabilities of the underlying LLMs and VLMs. It is possible that the framework might need adaptations to work effectively with different models.

*   **Potential Influence:** The paper has the potential to significantly influence the field of autonomous agents by promoting a more security-conscious approach to agent development. SAFEFLOW and SAFEFLOWBENCH can serve as valuable tools for researchers and practitioners seeking to build reliable and trustworthy agent systems.

**Score: 8.5**

**Rationale:** The paper is a strong contribution to the field of autonomous agents, offering a comprehensive and practical framework for enhancing agent trustworthiness. The novelty lies in the integrated approach to information flow control and transactional execution, while the significance stems from the ability to mitigate critical vulnerabilities in real-world deployments. While the potential for computational overhead and model dependency need further investigation, the paper's strengths outweigh its weaknesses, demonstrating high potential to impact the research community. The cross-benchmark evaluation further strengthens its generality.

- **Score**: 8/10

### **[Instructing Large Language Models for Low-Resource Languages: A Systematic Study for Basque](http://arxiv.org/abs/2506.07597v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Instructing Large Language Models for Low-Resource Languages: A Systematic Study for Basque":

**Summary:**

This paper explores effective strategies for adapting Large Language Models (LLMs) to low-resource languages, focusing specifically on Basque. Faced with limited instruction data, the authors systematically investigate different combinations of target language corpora, open-weight multilingual base models, pre-trained instruction-tuned backbones, and synthetically generated instructions. They present a comprehensive experimental framework and evaluate their models using both static benchmarks and human preferences gathered from a large-scale community-driven evaluation effort. The key findings indicate that: 1) target language corpora are essential for good performance, 2) synthetic instructions work well, and 3) starting with a pre-trained instruction-tuned model outperforms beginning with a non-instructed base model.  Their best model, built on Llama 3.1 instruct 70B, achieves competitive performance for Basque, approaching the capabilities of much larger commercial models. The paper also releases their code, models, instruction datasets, and human preference data to facilitate further research.

**Critical Evaluation:**

* **Novelty:** The paper's strength lies in its systematic and thorough exploration of different adaptation strategies for instruction-tuning LLMs in low-resource settings. While existing research addresses adapting LLMs for such languages, this work goes beyond a single approach and analyzes multiple combinations of resources (corpora, base models, synthetic data) and training methodologies. The deliberate constraint to readily available/creatable resources, avoiding reliance on commercial distillation, is both a strength (increased applicability) and a slight limitation (potential ceiling on performance).

* **Significance:** The insights gained are significant for researchers and practitioners working with low-resource languages. The finding that pre-trained instruction-tuned models provide a better starting point than base models challenges the standard adaptation pipeline and has implications for future research directions. The successful use of synthetic instructions is important, given the scarcity of manually-created instruction data. The use of human preferences for evaluation, and the large scale of the effort, is also notable and addresses a critical problem in evaluating LLMs, specifically for languages that are not that well supported.

* **Strengths:**
    * **Systematic approach:** The paper rigorously tests different combinations of resources and strategies, making the results and the conclusions more reliable.
    * **Realistic setup:** The constraints on resources (availability and createability) align with challenges faced in genuine low-resource scenarios.
    * **Large-scale human evaluation:** The community-driven arena, with over 12,000 annotations, is a significant contribution. It provides valuable insights into the perceived quality of models that static benchmarks may miss.
    * **Open Release:** The authors' commitment to releasing data, code, and models greatly benefits the research community.
    * **State of the Art Performance with open weight LLM**: The 70B model approaches state of the art in Basque, without using proprietary datasets, by only using a 1.2B word corpus.
* **Weaknesses:**
    * **Language Specificity:** While the findings likely generalize to other similarly-resourced languages, the study is focused on a single language (Basque). More explicit discussion of the nuances of Basque and how they might affect generalizability would be beneficial. The paper could benefit from adding more details on the limitations.
    * **Limited Scope of Alignment:**  The paper focuses only on the initial instruction tuning phase. While preference data was gathered, alignment tuning was not explored due to time and compute limitations, but should be considered part of the full adaptation pipeline.
    * **Missing ablations for 70B**: As said by the authors in page 7, one of the reasons why their model underperforms SotA is the weaker backbone model. More explanation regarding why INS EN model was chosen as a backbone would be of great value. The ablation studies focused on smaller model sizes. While practical, it leaves some questions open about how these insights translate to very large models.

* **Potential Influence:** This paper could significantly influence the direction of research on LLMs for low-resource languages.  It provides a practical guide for adapting models with limited resources, and the released datasets will likely become valuable assets for the community. The emphasis on human evaluation will also encourage researchers to adopt more user-centric evaluation strategies.

**Score: 8**

**Rationale:**

The paper is a significant contribution to the field. The systematic study, the emphasis on realistic constraints and available resources, the large-scale human evaluation, and the open release of data and code are all commendable strengths. The challenge to the standard adaptation pipeline and the success with synthetic data have the potential to influence the field. However, the language-specificity of the study and the lack of alignment tuning provide minor limitations. While the paper does provide solid insight for similar languages, a higher score requires more extensive validation and/or demonstration of applicability across a wider range of linguistic families. The analysis for the 70B would provide additional insight that a score above 8 requires. This paper also contributes significantly to addressing the AI divide, as the AI ecosystem is dominantly English-centric. The paper encourages further research in this direction. Overall, the score of 8 reflects the paper's valuable contribution to the understanding and development of LLMs for low-resource languages.

- **Score**: 8/10

### **[Return of ChebNet: Understanding and Improving an Overlooked GNN on Long Range Tasks](http://arxiv.org/abs/2506.07624v1)**
- **Summary**: Here's a summary and critical evaluation of the provided research paper:

**Summary:**

The paper revisits ChebNet, an early spectral graph neural network (GNN), arguing it has been overlooked despite its potential for modeling long-range dependencies in graphs. While ChebNet demonstrates advantages over Message Passing Neural Networks (MPNNs) and Graph Transformers on long-range benchmarks, it suffers from training instability due to its polynomial expansion. The authors address this by reformulating ChebNet as a stable dynamical system, termed Stable-ChebNet, which ensures stable information propagation without requiring eigendecompositions, positional encodings, or graph rewiring. Empirical results across several benchmarks show Stable-ChebNet achieving near state-of-the-art performance. The authors provide theoretical analysis regarding ChebNet's stability and provide a practical solution via their Stable-ChebNet framework.

**Critical Evaluation:**

*   **Novelty:**  The paper's novelty lies in several aspects. Firstly, it shines a light on the often forgotten ChebNet, demonstrating that it can be a competitive alternative, especially for long-range graph tasks. Secondly, the authors perform an in-depth analysis of the original ChebNet architecture, showing that simply increasing the polynomial order leads to instability. Thirdly, they propose a novel Stable-ChebNet formulation by recasting it into a continuous-time ODE with antisymmetric weight constraints. This is important since it not only provides a stable architecture but also offers a connection to stable dynamical systems. Finally, the implementation of stable numerical discretization gives a simple and effective method for maintaining stable information propagation.

*   **Significance:** The significance of this work is multifold. It offers a scalable alternative to address limitations of MPNNs by leveraging spectral approaches. Addressing the instability of ChebNet is a crucial step in making it a practical choice. The Stable-ChebNet formulation, based on a sound theoretical understanding, provides a computationally efficient option for long-range graph modeling. The experimental results on benchmark datasets further validate the efficacy of the proposed method. Finally, the work also serves as a reminder that older techniques should not be immediately discarded, especially when they possess unique features that can potentially resolve some issues.

*   **Strengths:**

    *   Strong theoretical foundation: The paper provides a rigorous analysis of ChebNet's dynamics and the conditions under which it becomes unstable.
    *   Effective solution: Stable-ChebNet addresses the instability issue in a principled way, leading to improved performance.
    *   Scalability:  The proposed approach maintains the scalability advantages of ChebNet, making it applicable to large graphs.
    *   Comprehensive evaluation: The experiments cover a range of datasets and tasks, demonstrating the robustness of Stable-ChebNet.
    *   Clear writing and presentation: The paper is well-written and easy to follow, with clear explanations of the technical concepts.

*   **Weaknesses:**

    *   While showing stability on the benchmarks, the paper does not provide any guarantees on all possible datasets.
    *   The paper states that the Euler step not only addresses the shortcomings of long-range information propagation but also performs effectively well on graphs with many nodes, however, it is not very thoroughly tested and compared, particularly on the OGB datasets.
    *   In some experiments, particularly in the Peptide experiments, Stable-ChebNet seems to be improving the performance of "long-range" information propagation, but falling a little short of other methods.

*   **Potential Influence:** This work could have a significant influence on the field of graph neural networks by encouraging researchers to revisit spectral methods and explore connections to dynamical systems. It may also lead to the development of new and improved GNN architectures that are both scalable and capable of modeling long-range dependencies.

**Score: 8**

**Justification:** The paper makes a solid contribution to the GNN field by revisiting an overlooked architecture and providing a theoretically sound and empirically validated improvement. The focus on stability is a critical aspect often disregarded in spectral GNNs. However, despite these strengths, there is a slight decrease in performance for the peptide datasets, and stronger validation is necessary to prove stability of every dataset. Therefore, while promising and likely to be influential, the paper does have some limitations.

- **Score**: 8/10

### **[QUITE: A Query Rewrite System Beyond Rules with LLM Agents](http://arxiv.org/abs/2506.07675v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces QUITE, a novel query rewrite system that leverages Large Language Models (LLMs) to improve SQL query performance. Unlike traditional rule-based systems, QUITE utilizes a training-free and feedback-aware approach based on LLM agents. QUITE employs a multi-agent framework controlled by a finite state machine (FSM) to manage the rewrite process.  The system incorporates a rewrite middleware with a structured knowledge base, a hybrid SQL corrector, and an agent memory buffer.  Additionally, QUITE utilizes a novel hint injection technique to improve the execution plans of rewritten queries. Experiments show that QUITE outperforms state-of-the-art approaches in reducing query execution time and producing a broader range of rewrites.

**Critical Evaluation:**

* **Novelty:** The novelty of QUITE lies in its innovative combination of LLMs, multi-agent systems, and database feedback for query rewriting. While LLMs have been used in database contexts before, the specific architecture of QUITE, with its FSM-controlled agents and middleware components, presents a unique approach. The hint injection technique is also a novel contribution.  The use of LLMs to go beyond rule-based rewrites is a notable step.
* **Significance:** The significance of QUITE stems from its ability to overcome limitations of existing query rewrite systems.  The system's improved performance and coverage demonstrate the potential of LLMs to handle more complex query patterns and rewrite strategies.  The feedback-aware and training-free nature of QUITE also make it potentially more adaptable and easier to deploy compared to systems that require extensive training data or rule engineering. The improved coverage for corner case queries offers significant benefit in overall workload performance. The effort of incorporating database insights into the rewrite process is a meaningful addition to the ongoing discussion of LLM application in database management. The inclusion of hint injection to address plan selection issues is a good example of addressing LLM weaknesses through conventional techniques.
* **Strengths:**
    * **Performance:**  The experimental results convincingly demonstrate the superiority of QUITE in terms of query execution time and rewrite coverage.
    * **Design:**  The architecture of QUITE, particularly the multi-agent FSM and the rewrite middleware, is well-designed and addresses key challenges in using LLMs for query rewriting (e.g., hallucination, context limitations).
    * **Training-Free:** The training-free aspect of QUITE is a significant advantage, as it reduces the complexity and cost of deployment.
    * **Hint Injection:** Addressing the shortcomings in optimizer estimates with hints demonstrates practical awareness and helps in realizing the full potential of rewrites.
* **Weaknesses:**
    * **LLM Dependency:**  The system's reliance on LLMs introduces a dependency on external services, potentially impacting cost, availability, and reproducibility. Though the paper addresses issues of hallucination through its multi-agent approach, this problem isn’t completely removed. LLM costs need to be carefully considered.
    * **Complexity:** The system is quite complex with its multi-agent design, FSM and multiple LLM components, potentially adding to deployment/maintenance overhead.
    * **Experimental Setup:** While the workloads are well-known, there is a level of control required to achieve high fidelity in LLM-based experimentation. Lack of detailed descriptions of cost and iteration limits, along with LLM-specific prompts, diminishes potential for further reproducibility.
    * **Generality:** The experimental section, while thorough for the cases under evaluation, offers limited discussion to specific architectures or cases where QUITE may underperform other approaches.

* **Potential Influence:** QUITE has the potential to influence the future direction of query rewrite systems by demonstrating the effectiveness of LLMs in this domain.  The paper's architecture and techniques could serve as a blueprint for other researchers and practitioners seeking to leverage LLMs for database optimization. More fundamentally, the paper highlights the potential for intelligent agents to improve query optimization, offering a potential bridge for traditional query optimization and AI.

**Justification for Score:**

Despite the LLM dependency and implementation complexity, QUITE presents a genuinely innovative and promising approach to query rewriting. The significant performance improvements and the novel architecture warrant a high score. However, the dependency on LLM's and the somewhat complex implementation bring the score down.

**Score: 8**

- **Score**: 8/10

### **[Training Superior Sparse Autoencoders for Instruct Models](http://arxiv.org/abs/2506.07691v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces Finetuning-aligned Sequential Training (FAST), a novel method for training Sparse Autoencoders (SAEs) on instruction-tuned Large Language Models (LLMs). Existing SAE training methods are designed for base models and perform poorly on instruct models due to semantic discontinuities arising from data concatenation in the traditional Block Training paradigm. FAST addresses this issue by processing each data instance independently, preserving semantic integrity and aligning the training with the fine-tuning objectives of instruct models. Experimental results on Qwen2.5-7B-Instruct and Llama3.2-3B-Instruct demonstrate that FAST significantly improves token reconstruction accuracy and feature interpretability compared to Block Training baselines. The authors also investigate the influence of special tokens on model outputs using the trained SAEs, providing insights into their roles and potential for fine-grained model control.

**Critical Evaluation:**

*   **Novelty:** The paper presents a clear methodological advance in SAE training. The idea of sequential training, aligning with the finetuning process and focusing on the semantic integrity of individual data instances, addresses a significant limitation in applying SAEs to instruction-tuned models. This is a valuable contribution as instruct models become increasingly prevalent. The idea is simple but effectively addresses a real problem.

*   **Significance:** Improving SAE performance for instruct models has significant implications for mechanistic interpretability. The enhanced token reconstruction accuracy and feature interpretability obtained with FAST enable a deeper understanding of the inner workings of these models. The discovered ability to improve output quality through intervention on special token activations presents exciting possibilities for model control and behavior modification, which could lead to new areas for more fine-grained control of LLMs.

*   **Strengths:**

    *   The paper is well-written and clearly articulates the problem, proposed solution, and experimental results.
    *   The motivation for FAST is compelling and directly addresses the shortcomings of existing Block Training methods.
    *   The experimental evaluation is comprehensive, comparing FAST against strong baselines on multiple models and tasks.
    *   The results demonstrate statistically significant improvements in token reconstruction and feature interpretability.
    *   The investigation into special token influence offers novel insights and potential avenues for future research.

*   **Weaknesses:**

    *   While the paper highlights the impact of special tokens on model outputs, it could delve deeper into the specific types of features and their interactions that contribute to improved or degraded performance upon intervention.
    *   The method depends on having high quality instruction tuning datasets, which could be a limitation in some scenarios.
    *   Computational constraints limit the investigation to models with fewer than 8B parameters; scalability to larger models should be verified.
    *   The feature steering experiments could explore a wider range of scaling coefficients (alpha) to determine a more precise optimal range and more exhaustively evaluate steering.

*   **Potential Influence:** The paper has strong potential to influence the field of mechanistic interpretability. FAST could become the standard method for training SAEs on instruction-tuned models, leading to more accurate and interpretable feature extraction. The findings on special token influence could inspire new research directions in model control, safety, and alignment. The open-sourcing of the code, datasets, and pre-trained SAE models will further promote adoption and experimentation.

*   **Rigorous Rationale for Score:** The method addresses a clear need in the growing field of LLM interpretability. The experiments demonstrate convincing improvements over baseline methods. The insights regarding special tokens are intriguing and point to valuable avenues for further investigation. While the approach is not revolutionary, it represents a significant practical advance in SAE training. It does have some of the limitations mentioned, but it improves current method.

Score: 8

- **Score**: 8/10

### **[Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking](http://arxiv.org/abs/2506.07751v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper "Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking" addresses the challenge of robustness in large language model (LLM) reasoning, particularly in smaller models, when faced with distribution shifts (e.g., changes in numerical variables, distracting clauses).  Instead of generating more varied training instances, the authors propose a novel approach called AbstraL, which teaches LLMs to explicitly construct underlying abstractions of reasoning problems. This involves a reinforced learning (RL) framework leveraging "Granularly-decomposed Abstract Reasoning (GranulAR)" data distilled from an oracle LLM. The process consists of identifying conditions, abstract reasoning, abstraction retrieval, and symbolic derivation.  Experiments on GSM-Symbolic and GSM-Plus benchmarks demonstrate that AbstraL mitigates performance degradation caused by perturbations and distractions, improving reasoning robustness.

**Critical Evaluation:**

*   **Novelty:** The core idea of focusing on *abstracting* reasoning problems rather than *instantiating* more varied examples is a significant and novel contribution. Many prior approaches have focused on data augmentation.  The RL-based learning framework to *directly* teach LLMs to abstract is also novel. The GranulAR data generation methodology, leveraging an oracle model and decomposing the process into finer steps, adds another layer of novelty. The introduction of model-free rewards tied directly to abstraction quality further distinguishes this work.

*   **Significance:** The problem of robustness in LLM reasoning is crucial for real-world deployment, especially in applications where models encounter diverse and unexpected inputs. This research offers a promising strategy to improve generalization capabilities. The practical implications of reducing reliance on massive data augmentation are considerable, as it could lead to more efficient and accessible training pipelines.  The improvement in handling distracting information is also significant, as it addresses a common failure point for LLMs.

*   **Strengths:**

    *   **Clear Problem Definition:** The paper clearly articulates the robustness challenge and its impact on LLM reasoning.
    *   **Novel Approach:** The AbstraL framework presents a fundamentally different approach to improving robustness compared to existing data augmentation techniques.
    *   **Granular Decomposition:** The GranulAR data format seems very effective at decomposing the overall problem, and providing a curriculum to ease learning in LLMs.
    *   **Empirical Validation:** The paper provides compelling empirical evidence on two established benchmarks (GSM-Symbolic and GSM-Plus), showcasing the effectiveness of AbstraL across various LLM sizes.
    *   **Ablation Studies:** The ablation studies are critical for isolating the contributions of different components of the framework (e.g., RL, symbolic tools, contextual information), strengthening the claims made.

*   **Weaknesses:**

    *   **Reliance on an Oracle LLM for Data Generation:** The GranulAR data creation process depends on a powerful oracle LLM, which might limit the accessibility of the method. While the trained LLM demonstrates improved robustness, the initial data distillation requires a significant resource. However, authors have demonstrated that the weaker LLM may be fine-tuned through RL, thus mitigating initial resource costs.
    *   **Limited Scope of Evaluation:**  The evaluation focuses primarily on mathematical reasoning. While this is a representative domain, it would be valuable to see how well AbstraL generalizes to other reasoning tasks (e.g., commonsense reasoning, logical inference).
    *   **Complexity:** The AbstraL framework is relatively complex, involving multiple stages and components. This might present a barrier to adoption. However, the authors thoroughly describe and provide results of the ablation, which mitigates its complexity.
    *   **Greedy Decoding:** The authors have tested their model using greedy decoding only. Advanced decoding strategies have not been tested, such as self-consistency decoding.

*   **Potential Influence:**  This paper has the potential to significantly influence research on LLM reasoning, robustness, and abstract thinking.  It provides a concrete and effective method for improving generalization, and the underlying principles could be adapted to other tasks and domains.  It also highlights the importance of directly learning abstract representations, which could inspire new research directions.

*   **Justification for Score:** The paper presents a novel and well-validated approach to a crucial problem in LLM research. The experimental results are compelling, and the ablation studies provide valuable insights. While the reliance on an oracle LLM and limited scope of evaluation are drawbacks, the overall contribution is significant and likely to have a lasting impact.

Score: 8

- **Score**: 8/10

### **[Accelerating Diffusion Models in Offline RL via Reward-Aware Consistency Trajectory Distillation](http://arxiv.org/abs/2506.07822v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces Reward-Aware Consistency Trajectory Distillation (RACTD), a novel method to accelerate diffusion models in offline reinforcement learning (RL). RACTD directly incorporates reward optimization into the consistency distillation process, enabling single-step action generation while maintaining high performance. This is achieved by distilling a pre-trained diffusion policy into a consistency model that is guided by a separately trained reward model. The method avoids the complexity of concurrent training and noise-aware reward models often found in other diffusion-based RL approaches. Experiments on Gym MuJoCo benchmarks and a long-horizon planning task demonstrate that RACTD achieves significant speedups (up to 142x) and performance improvements (8.7% over SOTA) compared to existing methods.

**Critical Evaluation:**

The paper presents a valuable contribution to the field of offline RL and diffusion models. The key strength lies in its ability to significantly accelerate diffusion-based policies while maintaining (or even improving) performance. The incorporation of a separate reward model into the consistency distillation process is a clever way to address the limitations of standard behavior cloning and complex actor-critic frameworks. Decoupling the training also simplifies the optimization process, making it easier to implement and train.

* **Novelty:** The novelty is well established. The approach to integrating reward information directly into the consistency distillation *process* for offline RL is a substantial improvement over existing techniques. Previous methods either relied on behavior cloning (struggling with suboptimal data) or actor-critic frameworks (requiring concurrent training). This direct integration, coupled with decoupled training, appears novel.

* **Significance:**  The significance is tied to the practical limitations of diffusion models: their slow inference speed. By achieving substantial speedups without sacrificing performance, RACTD makes diffusion-based policies more viable for real-world applications.  The performance boost observed, the simpler training, and the demonstrated generalization capability in long-horizon planning make this work highly relevant.

* **Strengths:**
    * **Significant Speedup:** Demonstrates impressive speedups in inference time compared to diffusion counterparts.
    * **Performance Improvement:** Achieves state-of-the-art or near state-of-the-art performance on standard benchmarks.
    * **Simplified Training:**  Avoids concurrent training and noise-aware reward models, simplifying the training pipeline.
    * **Decoupled Training:** Enables using of pre-trained and versatile diffusion teachers and reward models
    * **Effective Reward Integration:** Skillfully integrates reward information into the distillation process.
    * **Long-horizon Task Result:** Shows effective result on a long-horizon planning task
* **Weaknesses:**
    * **Need for Separate Reward Model:** The method relies on a separately trained reward model, which adds an extra step and introduces potential bias if the reward model is not accurate. While decoupled this dependency is still present.
    * **Loss Fluctuation:** There is a mention of potentially greater loss fluctuation by combining the Reward Model with the distillation
    * **Trade-off between Diversity and Performance:** As a result of prioritizing high-reward areas, the agent could fail in generalization across diverse tasks.
    * **Need of Unconditioned Teacher:** Even though in decoupled, relying on an unconditioned diffusion is a design limitation

* **Potential Influence:** The paper has the potential to influence future research in several ways:
    * **More efficient diffusion-based RL:** The RACTD framework can be adopted and extended to develop more efficient diffusion-based RL algorithms.
    * **Direct reward integration:** The approach of directly incorporating reward signals into distillation can be explored in other RL settings and for different types of models.
    * **Focus on single-step generation:** The success of RACTD may encourage more research into single-step or few-step generation methods for RL.
    * **Real-world applications:** The improved efficiency of RACTD could make diffusion-based RL more feasible for real-world applications such as robotics and autonomous driving.

* **Justification of Score:** While the need for a separate reward model is a minor limitation, the performance improvements, substantial speedups, and simplified training pipeline demonstrate the significance and practicality of the proposed method. The experimental evaluation is thorough and addresses key aspects of the approach.

Score: 8
The paper has strong novelty and substantial practical benefits. The limitations are not major, and the paper provides a well-defined and easily implementable framework that makes a significant contribution to the field of offline RL.

- **Score**: 8/10

### **[Improving large language models with concept-aware fine-tuning](http://arxiv.org/abs/2506.07833v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces Concept-Aware Fine-Tuning (CAFT), a novel multi-token training method for large language models (LLMs).  CAFT addresses the limitations of traditional next-token prediction, which fragments concepts into individual tokens, hindering deeper understanding. CAFT trains LLMs to predict multiple tokens in parallel during fine-tuning, enabling them to learn more coherent, high-level concepts.  The authors demonstrate significant improvements over next-token fine-tuning across various tasks, including text summarization, code generation, mathematics and scientific domains like molecular generation and de novo protein design. CAFT makes multi-token training more accessible by introducing it into the post-training phase, democratizing its benefits for the broader AI community.

**Critical Evaluation:**

* **Novelty:** The primary novelty lies in bringing multi-token prediction to the *post-training* (fine-tuning) phase.  While multi-token prediction itself isn't new (it's been explored in pre-training),  its adoption in fine-tuning is a significant shift.  The paper correctly points out that previous attempts to use multi-token methods in fine-tuning have been unsuccessful. CAFT's core contributions are the auxiliary head training procedure and the specific loss weighting scheme designed to overcome the challenges of distribution shift and optimize for the primary task. The claim of "democratizing" multi-token training by making it accessible to researchers with less computational resources rings true, as pre-training requires enormous resources.

* **Significance:** The significance stems from several points:
    * **Improved Performance:** The experimental results clearly demonstrate that CAFT outperforms traditional next-token fine-tuning across a diverse range of tasks.  The consistent improvement, even in scenarios where next-token prediction would seem adequate, indicates that CAFT captures something fundamental. The performance gains in complex scientific tasks (protein design, molecular generation) highlights CAFT's potential beyond just language modeling.
    * **Addressing a Fundamental Limitation:** The paper convincingly argues that next-token prediction is a bottleneck to LLM understanding.  By enabling models to see "further ahead," CAFT allows them to grasp concepts more holistically. This suggests that CAFT could unlock capabilities that are currently inaccessible to LLMs.
    * **Accessibility:** The design of CAFT (specifically, providing pre-trained auxiliary heads) lowers the barrier to entry for researchers to experiment with multi-token fine-tuning. The practical guide for setting up CAFT is an important contribution.
    * **Theoretical implications:** The results are a clear sign that the next-token prediction paradigm may be limiting performance and further, suggests that LLMs benefit from explicitly learning multi-token concepts, which questions the assumption that LLMs can automatically learn coherent concepts across tokens.

* **Strengths:**
    * **Clear Problem Definition:** The paper clearly articulates the limitations of next-token prediction and why it hinders conceptual understanding.
    * **Novel Method:** CAFT is a well-designed method for addressing the problem, with specific techniques to make multi-token fine-tuning feasible.
    * **Empirical Validation:** The paper provides extensive experimental results across a wide range of tasks, demonstrating the effectiveness of CAFT.  The ablations also support the authors' claims about the importance of the design choices.
    * **Practical Considerations:** The paper provides practical guidance on implementing and using CAFT, making it accessible to a wider audience.
    * **Well written**: The paper is well-structured and easy to follow.

* **Weaknesses:**
    * **Limited Theoretical Analysis:** While the empirical results are strong, the paper lacks a deeper theoretical analysis of why CAFT works.  A more rigorous explanation of the mechanism by which CAFT improves conceptual understanding would be valuable. For instance, analyzing the representations learned by auxiliary heads would provide greater insight.
    * **Concept Proxy:**  The paper presents a concept proxy for code and molecules. While intuitive, it's not clear how generalizable it is. It would be interesting to see how the concept proxy affects performance by varying it.
    * **Limited Hyperparameter Tuning:** While the paper claims robustness to hyperparameters, the experimental setup lacks comprehensive hyperparameter tuning for all tasks.
    * **Missing comparisons to other speculative decoding techniques**: The paper mentions that several speculative decoding methods leverage multi-token prediction. It could be beneficial to compare against those methods.

* **Potential Influence:** CAFT has the potential to significantly influence the development and application of LLMs. It offers a new approach to fine-tuning that can improve performance across a wide range of tasks, particularly those that require a deeper understanding of concepts.  It could also lead to new research into the limitations of next-token prediction and the benefits of multi-token training.

* **Score:** 8

**Rationale:**

CAFT represents a significant advance in LLM fine-tuning. The introduction of multi-token prediction to the post-training phase, combined with the carefully designed auxiliary head training and loss weighting scheme, overcomes the challenges that have plagued previous attempts. The empirical results are compelling, demonstrating consistent improvements across a diverse range of tasks. The accessibility of CAFT further enhances its potential influence. The lack of deep theoretical analysis and concept proxy are limitations, but the practical impact and potential of the method are undeniable. The democratizing effect of CAFT in the light of huge computational costs is a great contribution. It's a valuable contribution to the LLM community and warrants a score of 8.

- **Score**: 8/10

### **[Video Unlearning via Low-Rank Refusal Vector](http://arxiv.org/abs/2506.07891v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces a novel training-free method for unlearning harmful concepts (like nudity, violence, copyrighted material) from video diffusion models.  The core idea involves creating "refusal vectors" based on the differences in latent space activations between safe and unsafe prompt pairs. These refusal vectors are then refined using a contrastive PCA-based low-rank factorization to isolate the target concept and minimize collateral damage to the model's overall generation quality. The method embeds these refusal vectors directly into the model's weights, offering a permanent and robust way to suppress unwanted content without retraining or accessing the original training data.  The authors demonstrate the effectiveness of their approach through qualitative and quantitative evaluations, showing the ability to neutralize harmful content across various categories while preserving video fidelity.

**Critical Evaluation:**

*   **Novelty:** The paper presents a significant contribution by addressing the relatively unexplored area of unlearning in *video* diffusion models.  While unlearning techniques exist for image diffusion models, adapting and extending them to the more complex domain of video generation is non-trivial. The use of contrastive PCA on latent space differences to create low-rank "refusal vectors" is a novel and effective approach.  This helps to prevent the model from forgetting related semantic information, addressing a key problem in unlearning. The multimodal approach (using both text and image prompts) for concept extraction is a strength. It captures the nuances of concept manifestation across modalities, leading to better unlearning and retention of overall generation quality.
*   **Significance:** The potential impact of this work is high. As video diffusion models become more powerful and widely used, the ability to control and remove harmful content becomes increasingly important.  The presented method provides a practical and efficient way to mitigate risks associated with biased or unsafe data embedded in these models. The fact that it's training-free and operates by directly modifying model weights (rather than relying on input/output filters) makes it more robust and avoids inference-time overhead. This is particularly significant as it circumvents reliance on third-party services which may not align with the objectives of the model developers. The experimental results, including the use of the T2VSafetyBench benchmark, support the effectiveness of the method and provide a solid basis for further research.
*   **Strengths:**
    *   Addresses a critical and timely problem.
    *   Novel and technically sound approach using contrastive PCA and low-rank factorization.
    *   Training-free and efficient.
    *   Multimodal concept extraction.
    *   Robust weight-embedding implementation.
    *   Comprehensive experimental evaluation.
*   **Weaknesses:**
    *   The selection of "safe" and "unsafe" prompt pairs requires careful curation. The effectiveness of the method depends on the quality of these pairs. While the paper mentions using only 5 pairs, there's limited discussion on the selection process and potential sensitivity to prompt selection bias.
    *   The method removes *one* concept at a time. The scalability of removing multiple, potentially interacting, harmful concepts simultaneously is not addressed and represents a limitation for real-world applications.
    *   The potential for adversarial reversibility is acknowledged but not fully explored. More research is needed to understand the vulnerabilities and develop defenses against malicious actors who might try to reintroduce or even amplify harmful content.
    *   There is a lack of discussion on how these methods could potentially exacerbate existing biases within these models.
*   **Justification:**
    *   This paper offers a novel and relatively simple approach for removing harmful content.
    *   The low-rank technique mitigates damage to otherwise harmless concepts.
    *   The authors provide solid experimental results using a standard benchmark.

**Score: 8**

**Rationale:** The paper makes a valuable and timely contribution to the field of generative video modeling by offering a practical and effective solution for unlearning harmful content. The novelty of the approach, combined with strong experimental results, warrants a high score. However, the limitations regarding prompt curation, scalability, adversarial robustness, and possible amplification of existing biases, prevent it from reaching a higher score. Further research addressing these weaknesses could significantly enhance the impact and applicability of this work.

- **Score**: 8/10

### **[LUCIFER: Language Understanding and Context-Infused Framework for Exploration and Behavior Refinement](http://arxiv.org/abs/2506.07915v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces LUCIFER (Language Understanding and Context-Infused Framework for Exploration and Behavior Refinement), a domain-agnostic framework that integrates reinforcement learning (RL), large language models (LLMs), and a hierarchical decision-making architecture.  The framework aims to address the problem of outdated environmental knowledge in dynamic environments by leveraging real-time, context-rich input from human stakeholders.  LLMs are used in two roles: context extractors (structuring verbal input into actionable intelligence) and exploration facilitators (guiding action selection during exploration). The paper benchmarks various LLMs in these roles, showing that LUCIFER improves exploration efficiency and decision quality compared to flat, goal-conditioned policies, particularly in a simulated search and rescue (SAR) environment.  The framework's key components include a Strategic Decision Engine (SDE), specialized Worker agents, an Information Space, a Context Extractor, and an Exploration Facilitator, all connected by an attention space mechanism.

**Critical Evaluation:**

*   **Novelty:** The paper's main novelty lies in its dual LLM role and integrating these extracted features with RL techniques. While using LLMs with RL is not entirely new, LUCIFER's integration scheme with a hierarchical structure and the attention space mechanism is a significant contribution. The framework effectively addresses two key limitations that exist in existing approaches: i) the limited involvement of humans in the learning loops, and ii) scalability of RL based system in complex tasks. Furthermore, the hierarchical task decomposition inspired by the human-centric approach helps create a tractable system that can accommodate multiple interdependent tasks while adhering to temporal constraints.
*   **Significance:** The paper has the potential to be significant for the fields of human-AI collaboration, reinforcement learning, and autonomous systems. The ability to translate human contextual knowledge into actionable intelligence for autonomous agents is a crucial step towards more effective and reliable systems in dynamic environments. The SAR case study effectively showcases the potential of LUCIFER in a complex, real-world application. Moreover, the comprehensive analysis of various LLMs in context extraction and exploration guidance provides valuable insights for practitioners.
*   **Strengths:**
    *   The framework design is well-motivated and clearly articulated.
    *   The dual LLM role is a creative and effective way to leverage the capabilities of these models.
    *   The attention space mechanism provides a structured approach to integrating LLM-processed information into the decision-making process.
    *   The SAR environment provides a relevant and challenging testbed for evaluation.
    *   The experimental results demonstrate the effectiveness of LUCIFER in improving exploration efficiency and decision quality.
    *   The comprehensive comparison of various LLMs in different roles offers practical guidance for implementation.
*   **Weaknesses:**
    *   The simulated environment, while effective for evaluation, may not fully capture the complexities of real-world SAR scenarios. Transferability to complex real-world scenarios are yet to be seen.
    *   The reliance on the quality of LLM outputs introduces potential risks that are not fully addressed. While heuristics are adopted, there needs to be further quantification on the robustness of the system if LLM gives ambiguous/erroneous information.
    *   Limited discussion of the computational costs of the system, especially considering the use of LLMs.
    *   The current implementation has a fixed task decomposition. A mechanism to dynamically adapt at high-level policies can make it better suited in real-world scenarios.
    *   The implementation is only limited to a single worker, there needs to be some explanation regarding scalability of the proposed architecture in a multi-worker setting.

*   **Potential Influence:** This work provides a valuable step towards creating more human-aware autonomous systems that can effectively operate in dynamic and complex environments. Future work will likely build upon this framework to address the limitations, explore other applications, and develop more robust and scalable solutions.

**Score: 8**

**Rationale:**

The paper presents a novel and well-designed framework for integrating human contextual knowledge into autonomous decision-making. The dual LLM role and the attention space mechanism are key contributions. While there are limitations, such as the reliance on LLM quality and the complexities of real-world deployment, the paper is a significant advancement in the field. It offers a promising direction for future research and has the potential to influence the design of intelligent systems in various domains, which is why it deserves a score of 8.

- **Score**: 8/10

### **[HeuriGym: An Agentic Benchmark for LLM-Crafted Heuristics in Combinatorial Optimization](http://arxiv.org/abs/2506.07972v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces HeuriGym, a new agentic benchmark for evaluating the ability of Large Language Models (LLMs) to generate effective heuristic algorithms for combinatorial optimization problems. Unlike existing benchmarks that often rely on closed-form questions or subjective human evaluations, HeuriGym uses a framework where LLMs propose heuristics, receive code execution feedback, and iteratively refine their solutions. The framework evaluates LLMs across dimensions like tool use, planning, instruction following, and iterative refinement. To quantify performance, the authors introduce the Quality-Yield Index (QYI), which captures both solution quality and pass rate. Experiments with state-of-the-art LLMs on nine diverse optimization problems revealed that even top models like GPT-04-mini and Gemini-2.5-Pro achieve only modest QYI scores, highlighting limitations in their problem-solving capabilities in realistic scenarios.  The authors make their benchmark open-source, aiming to stimulate the development of LLMs with enhanced problem-solving skills for scientific and engineering domains.

**Critical Evaluation:**

*   **Novelty:** The paper presents a novel approach by introducing an *agentic* framework for LLM evaluation within the context of combinatorial optimization. This "closed-loop" interaction, where LLMs refine their solutions based on execution feedback, is a significant departure from traditional static evaluation setups. This distinguishes HeuriGym from existing objective and subjective benchmarks. The introduction of the QYI metric is also a good step towards more fair comparisons by combining both rate of feasibility and solution quality.

*   **Significance:** Combinatorial optimization problems are fundamental in various domains, making this benchmark relevant to practical applications. The paper identifies a clear gap in the current LLM evaluation methodologies: their inability to accurately assess multi-step reasoning, tool use, and adaptive reasoning needed for real-world problem-solving. By exposing these limitations, the benchmark has the potential to guide the development of more capable LLMs for scientific and engineering tasks.

*   **Strengths:**
    *   **Agentic Framework:** The iterative refinement process mimics real-world problem-solving scenarios.
    *   **Well-Defined Objective:** The focus on combinatorial optimization provides clear, quantitative metrics for evaluation.
    *   **Diverse Problem Set:** The benchmark includes problems from different domains, enhancing its generalizability.
    *   **Open-Source Availability:** Promotes community involvement and future research.
    *   **QYI Metric:** The QYI is a reasonable attempt to create a single-number metric that encapsulates both solution quality and frequency of feasible outputs.

*   **Weaknesses:**
    *   **Dependency on Execution Environment:** Performance is tied to the specific execution environment (Python in this case). The authors acknowledge this and plan to integrate C++ more fully, but the current reliance on Python could limit the scalability and efficiency of evaluation.
    *   **Proxy Metrics:** The authors rely on proxy metrics for evaluation, as true real-world deployment (e.g., manufacturing silicon) would take too long to evaluate. The paper mentions that they use published results from past papers and a separate reviewer. A more robust, end-to-end measure would be extremely desirable but currently impossible.
    *   **Benchmark size**: The benchmark includes only nine problems. Although these have been thoroughly tested, the benchmark may become saturated in the future or give skewed results.
    *   **Limited baseline LLM agents**: Despite evaluating several leading LLMs, it would be insightful to include fine-tuned versions of the LLMs designed specifically for combinatorial optimization or code generation to see how baseline agents improve.

*   **Potential Influence:** HeuriGym could become a standard benchmark for evaluating LLMs in scientific and engineering domains. The framework is flexible, allowing for the inclusion of new problems and evaluation metrics, promoting continued progress in the field. It may also inspire new methods for training LLMs to generate and refine algorithms based on feedback.

* **Justification:** While the paper is strong in design and execution, I believe the biggest strength is that it highlights a lack of proper and accurate LLM evaluation methods that apply to scientific or engineering tasks. There are obvious drawbacks that the paper has, but it provides the first step for more agentic benchmarks for LLMs.

**Score: 8**

- **Score**: 8/10

### **[OneIG-Bench: Omni-dimensional Nuanced Evaluation for Image Generation](http://arxiv.org/abs/2506.07977v1)**
- **Summary**: Here's a summary and critical evaluation of the "OneIG-Bench: Omni-dimensional Nuanced Evaluation for Image Generation" paper:

**Summary:**

The paper introduces OneIG-Bench, a new benchmark designed for a more comprehensive and nuanced evaluation of Text-to-Image (T2I) models.  It addresses the limitations of existing benchmarks that often focus on single dimensions like semantic alignment or image quality. OneIG-Bench is structured around six core categories: General Object, Portrait, Anime and Stylization, Text Rendering, Knowledge and Reasoning, and Multilingualism. The benchmark includes over 1000 curated prompts, mainly sourced from real-world user inputs.  The authors also define quantitative metrics tailored to each evaluation dimension. The code and dataset are publicly available.

**Critical Evaluation:**

* **Novelty:** The paper's primary contribution is the multi-dimensional approach to evaluating T2I models. While individual components of the evaluation (e.g., using edit distance for text rendering, CLIP for image-text alignment) are not necessarily novel, the *integration* of these diverse evaluations into a single benchmark with a well-structured taxonomy is. The incorporation of Knowledge and Reasoning and Multilingualism, specifically, addresses emerging needs as T2I models become more sophisticated.
* **Significance:** Existing benchmarks tend to be too simplistic relative to state-of-the-art T2I models.  OneIG-Bench helps move the field towards more rigorous model assessment. The benchmark's ability to pinpoint strengths and weaknesses in specific areas of image generation can guide future research. By providing a standardized, reproducible framework, OneIG-Bench facilitates fair comparisons between models. It encourages researchers to consider a broader range of capabilities beyond simple image quality or alignment. The framework includes a systematic quantitative evaluation, facilitating objective capability ranking.
* **Strengths:**
    * **Comprehensive Coverage:** The six categories offer a much broader assessment than many existing benchmarks.
    * **Real-World Relevance:**  The use of real-world user inputs for prompts increases the benchmark's applicability.
    * **Modular Design:** Users can focus on specific evaluation subsets to avoid generating images for the entire dataset, which leads to more efficient evaluation.
    * **Public Availability:** Encourages adoption and community contribution.
    * **Well-defined metrics:** The paper carefully devises quantitative indicators taking into account various factors to ensure comprehensiveness and objectivity.
* **Weaknesses:**
    * **Metric Choices:** While the metrics are generally sound, the dependence on models like GPT-4o and LLM2CLIP for some evaluations (e.g., Knowledge and Reasoning) introduces a dependency on the performance and potential biases of those underlying models. Changes in these external models could impact benchmark results, and this is not sufficiently discussed.
    * **Subjectivity in prompts:**  Although manual reviews were conducted, some level of subjectivity is inevitably present in prompt curation. This could subtly influence evaluation outcomes.
    * **Scalability for new dimensions:** As T2I technology continues to advance, new dimensions may become important. The paper does not directly address how OneIG-Bench can be extended beyond its current six categories without major restructuring.
    * **Limited diversity:** While the benchmark is designed to test diversity in image generation, is the diversity in the benchmark itself representative of all potential inputs? Further analysis or justification of dataset diversity is warranted.
* **Potential Influence:** OneIG-Bench has the potential to become a widely used benchmark in the T2I community, driving the development of more robust and capable image generation models.  Its emphasis on reasoning and text rendering addresses critical areas that have been previously under-evaluated.

**Score: 8**

**Rationale:** OneIG-Bench represents a significant step forward in T2I model evaluation. The multi-dimensional approach is timely and addresses a clear need in the field. The publicly available dataset and code will promote wider adoption.  While the dependence on external LLMs and potential for dataset bias are valid concerns, they do not outweigh the overall value of the benchmark. The modular design and the real-world prompt base provide solid foundations for future innovation in T2I evaluation.  To warrant a higher score, the authors would need to demonstrate a more rigorous approach in mitigating any potential biases in the prompts and dependencies with other large models.

- **Score**: 8/10

### **[Reparameterized LLM Training via Orthogonal Equivalence Transformation](http://arxiv.org/abs/2506.08001v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Reparameterized LLM Training via Orthogonal Equivalence Transformation":

**Summary:**

The paper introduces POET, a novel reparameterization technique for training Large Language Models (LLMs). Instead of directly optimizing weight matrices, POET reparameterizes each neuron with two learnable orthogonal matrices and a fixed random weight matrix. This approach, grounded in orthogonal equivalence transformation, provably preserves spectral properties of the weight matrices during training. The authors further propose efficient approximations, Stochastic Primitive Optimization (SPO) and Cayley-Neumann Parameterization (CNP), to make POET scalable to large models. Experimental results on LLaMA models of various sizes demonstrate that POET can achieve improved generalization performance and training stability with fewer trainable parameters compared to standard training methods (AdamW) and other parameter-efficient techniques (e.g., GaLore, LoRA). The paper also provides vector probing analyses to understand the learning dynamics of the orthogonal matrices.

**Critical Evaluation:**

*   **Novelty:** The idea of using orthogonal equivalence transformations is a clear departure from conventional direct weight optimization and constitutes a major strength. The connection to spectral properties and their preservation during training is well-motivated and theoretically supported. The combination of SPO and CNP for efficient large-scale application are well conceived. The use of random initialization for a fixed weight matrix is interesting. The paper provides a more theoretically principled way to achieve strong control on the spectrum during training. That is, POET leverages the orthogonality of weight matrices rather than more learnable parameters to achieve inductive bias, which can be more beneficial compared to learning with more parameters.

*   **Significance:** LLM training is a computationally expensive and often unstable process. POET tackles these problems head-on. The results, demonstrating better parameter efficiency and potentially improved generalization performance, could make LLM training more accessible and reliable. If the method proves robust across a variety of architectures and datasets, it can become a central training framework. If the orthogonal structure results in a more explainable structure to the weights as the experiments allude, there are other potential impacts to LLM design.

*   **Strengths:**

    *   **Theoretical Foundation:** The paper provides a strong theoretical grounding for the proposed method, connecting it to spectral properties and generalization guarantees.
    *   **Efficient Approximations:** The proposed approximations (SPO and CNP) are critical for scaling POET to large models, addressing a key challenge in LLM training.
    *   **Comprehensive Experiments:** The empirical evaluation is comprehensive, covering different model sizes, ablation studies, and comparisons with existing methods.
    *   **Insightful Analysis:** The vector probing analysis offers valuable insights into the learning dynamics of orthogonal matrices, contributing to a better understanding of how POET works.

*   **Weaknesses:**

    *   **Implementation complexity:** The POET technique introduces an extra layer of complexity. Although well-motivated, the method is not easily adapted by practitioners without high GPU capacity and a firm foundation in linear algebra. Furthermore, there are more hyper-parameters that must be tuned relative to Adam-W and related methods.
    *   **Lack of diverse architectures:** The experiments are exclusively carried out on variants of the Llama transformer architecture. Testing POET's effectiveness with different architectural designs (e.g., different attention mechanisms, residual connections, etc.) would further enhance confidence in its wide applicability.
    *   **Scalability in practice:** While the paper proposes efficient approximations, demonstrating POET's scalability in practice by training significantly larger models (e.g., comparable to GPT-3 or PaLM) would be highly valuable. However, at the scale of the experiments in the paper, they are convincing.
    *   **AdamW-like optimization schemes might catch-up:** With the advent of the next generation Adam-W optimizers such as those described in "Muon is scalable for Ilm training," better generalization and training schemes may close the gap relative to this scheme.

*   **Potential Impact:**
    *   Influence the development of new training methodologies for LLMs, focusing on spectral control and parameter efficiency.
    *   Provide a theoretical framework for understanding the role of orthogonal transformations in neural network training.
    *   Enable the training of larger and more powerful LLMs with limited computational resources.
    *   Inspire new research directions on the connection between weight spectrum and generalization properties.

**Justification for Score:**

Overall, this is a strong paper that presents a novel and theoretically well-motivated approach to LLM training. The proposed method addresses critical challenges in the field and demonstrates promising empirical results. However, the limited variety of architectures and scalability experiments, along with the implementation complexity and AdamW-like schemes might close the gap in performance limit its impact to some extent. Therefore, I assign the score below.

Score: 8.0

- **Score**: 8/10

## Other Papers
### **[Prompting Science Report 2: The Decreasing Value of Chain of Thought in Prompting](http://arxiv.org/abs/2506.07142v1)**
### **[GeometryZero: Improving Geometry Solving for LLM with Group Contrastive Policy Optimization](http://arxiv.org/abs/2506.07160v1)**
### **[AMoPO: Adaptive Multi-objective Preference Optimization without Reward Models and Reference Models](http://arxiv.org/abs/2506.07165v1)**
### **[Efficient Text-Attributed Graph Learning through Selective Annotation and Graph Alignment](http://arxiv.org/abs/2506.07168v1)**
### **[CTDGSI: A comprehensive exploitation of instance selection methods for automatic text classification. VII Concurso de Teses, Dissertações e Trabalhos de Graduação em SI -- XXI Simpósio Brasileiro de Sistemas de Informação](http://arxiv.org/abs/2506.07169v1)**
### **[RULE: Reinforcement UnLEarning Achieves Forget-Retain Pareto Optimality](http://arxiv.org/abs/2506.07171v1)**
### **[Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models](http://arxiv.org/abs/2506.07177v1)**
### **[Flattery in Motion: Benchmarking and Analyzing Sycophancy in Video-LLMs](http://arxiv.org/abs/2506.07180v1)**
### **[Mitigating Behavioral Hallucination in Multimodal Large Language Models for Sequential Images](http://arxiv.org/abs/2506.07184v1)**
### **[Exploring Effective Strategies for Building a Customised GPT Agent for Coding Classroom Dialogues](http://arxiv.org/abs/2506.07194v1)**
### **[SAP-Bench: Benchmarking Multimodal Large Language Models in Surgical Action Planning](http://arxiv.org/abs/2506.07196v1)**
### **[Reasoning Multimodal Large Language Model: Data Contamination and Dynamic Evaluation](http://arxiv.org/abs/2506.07202v1)**
### **[HOI-PAGE: Zero-Shot Human-Object Interaction Generation with Part Affordance Guidance](http://arxiv.org/abs/2506.07209v1)**
### **[Sword and Shield: Uses and Strategies of LLMs in Navigating Disinformation](http://arxiv.org/abs/2506.07211v1)**
### **[BIMgent: Towards Autonomous Building Modeling via Computer-use Agents](http://arxiv.org/abs/2506.07217v1)**
### **[Advancing Multimodal Reasoning Capabilities of Multimodal Large Language Models via Visual Perception Reward](http://arxiv.org/abs/2506.07218v1)**
### **[LLM-Enhanced Rapid-Reflex Async-Reflect Embodied Agent for Real-Time Decision-Making in Dynamically Changing Environments](http://arxiv.org/abs/2506.07223v1)**
### **[Hallucination at a Glance: Controlled Visual Edits and Fine-Grained Multimodal Learning](http://arxiv.org/abs/2506.07227v1)**
### **[Learn as Individuals, Evolve as a Team: Multi-agent LLMs Adaptation in Embodied Environments](http://arxiv.org/abs/2506.07232v1)**
### **[Multi-Step Visual Reasoning with Visual Tokens Scaling and Verification](http://arxiv.org/abs/2506.07235v1)**
### **[SDE-SQL: Enhancing Text-to-SQL Generation in Large Language Models via Self-Driven Exploration with SQL Probes](http://arxiv.org/abs/2506.07245v1)**
### **[A Stable Whitening Optimizer for Efficient Neural Network Training](http://arxiv.org/abs/2506.07254v1)**
### **[Question Answering under Temporal Conflict: Evaluating and Organizing Evolving Knowledge with LLMs](http://arxiv.org/abs/2506.07270v1)**
### **[Parsing the Switch: LLM-Based UD Annotation for Complex Code-Switched and Low-Resource Languages](http://arxiv.org/abs/2506.07274v1)**
### **[Investigating the Relationship Between Physical Activity and Tailored Behavior Change Messaging: Connecting Contextual Bandit with Large Language Models](http://arxiv.org/abs/2506.07275v1)**
### **[From Generation to Generalization: Emergent Few-Shot Learning in Video Diffusion Models](http://arxiv.org/abs/2506.07280v1)**
### **[Multi-Step Guided Diffusion for Image Restoration on Edge Devices: Toward Lightweight Perception in Embodied AI](http://arxiv.org/abs/2506.07286v1)**
### **[Exploring the Impact of Temperature on Large Language Models:Hot or Cold?](http://arxiv.org/abs/2506.07295v1)**
### **[HotelMatch-LLM: Joint Multi-Task Training of Small and Large Language Models for Efficient Multimodal Hotel Retrieval](http://arxiv.org/abs/2506.07296v1)**
### **[Pre-trained Large Language Models Learn Hidden Markov Models In-context](http://arxiv.org/abs/2506.07298v1)**
### **[ConfQA: Answer Only If You Are Confident](http://arxiv.org/abs/2506.07309v1)**
### **[Paged Attention Meets FlexAttention: Unlocking Long-Context Efficiency in Deployed Inference](http://arxiv.org/abs/2506.07311v1)**
### **[SCGAgent: Recreating the Benefits of Reasoning Models for Secure Code Generation with Agentic Workflows](http://arxiv.org/abs/2506.07313v1)**
### **[DEF: Diffusion-augmented Ensemble Forecasting](http://arxiv.org/abs/2506.07324v1)**
### **[Reward Model Interpretability via Optimal and Pessimal Tokens](http://arxiv.org/abs/2506.07326v1)**
### **[Graph-KV: Breaking Sequence via Injecting Structural Biases into Large Language Models](http://arxiv.org/abs/2506.07334v1)**
### **[Improving LLM Reasoning through Interpretable Role-Playing Steering](http://arxiv.org/abs/2506.07335v1)**
### **[Refusal-Feature-guided Teacher for Safe Finetuning via Data Filtering and Alignment Distillation](http://arxiv.org/abs/2506.07356v1)**
### **[ARGUS: Hallucination and Omission Evaluation in Video-LLMs](http://arxiv.org/abs/2506.07371v1)**
### **[Shapley-Coop: Credit Assignment for Emergent Cooperation in Self-Interested LLM Agents](http://arxiv.org/abs/2506.07388v1)**
### **[Boosting Vulnerability Detection of LLMs via Curriculum Preference Optimization with Synthetic Reasoning Data](http://arxiv.org/abs/2506.07390v1)**
### **[Distributed Image Semantic Communication via Nonlinear Transform Coding](http://arxiv.org/abs/2506.07391v1)**
### **[MedChat: A Multi-Agent Framework for Multimodal Diagnosis with Large Language Models](http://arxiv.org/abs/2506.07400v1)**
### **[Beyond Jailbreaks: Revealing Stealthier and Broader LLM Security Risks Stemming from Alignment Failures](http://arxiv.org/abs/2506.07402v1)**
### **[Enhancing Watermarking Quality for LLMs via Contextual Generation States Awareness](http://arxiv.org/abs/2506.07403v1)**
### **[RiemannFormer: A Framework for Attention in Curved Spaces](http://arxiv.org/abs/2506.07405v1)**
### **[InverseScope: Scalable Activation Inversion for Interpreting Large Language Models](http://arxiv.org/abs/2506.07406v1)**
### **[An Intelligent Fault Self-Healing Mechanism for Cloud AI Systems via Integration of Large Language Models and Deep Reinforcement Learning](http://arxiv.org/abs/2506.07411v1)**
### **[Evaluating Visual Mathematics in Multimodal LLMs: A Multilingual Benchmark Based on the Kangaroo Tests](http://arxiv.org/abs/2506.07418v1)**
### **[Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models](http://arxiv.org/abs/2506.07424v1)**
### **[Well Begun is Half Done: Low-resource Preference Alignment by Weak-to-Strong Decoding](http://arxiv.org/abs/2506.07434v1)**
### **[Prompt to Protection: A Comparative Study of Multimodal LLMs in Construction Hazard Recognition](http://arxiv.org/abs/2506.07436v1)**
### **[Extending Epistemic Uncertainty Beyond Parameters Would Assist in Designing Reliable LLMs](http://arxiv.org/abs/2506.07448v1)**
### **[LlamaRec-LKG-RAG: A Single-Pass, Learnable Knowledge Graph-RAG Framework for LLM-Based Ranking](http://arxiv.org/abs/2506.07449v1)**
### **[When Style Breaks Safety: Defending Language Models Against Superficial Style Alignment](http://arxiv.org/abs/2506.07452v1)**
### **[From Calibration to Collaboration: LLM Uncertainty Quantification Should Be More Human-Centered](http://arxiv.org/abs/2506.07461v1)**
### **[CCI4.0: A Bilingual Pretraining Dataset for Enhancing Reasoning in Large Language Models](http://arxiv.org/abs/2506.07463v1)**
### **[DeepVideo-R1: Video Reinforcement Fine-Tuning via Difficulty-aware Regressive GRPO](http://arxiv.org/abs/2506.07464v1)**
### **[Chasing Moving Targets with Online Self-Play Reinforcement Learning for Safer Language Models](http://arxiv.org/abs/2506.07468v1)**
### **[Improving Fairness of Large Language Models in Multi-document Summarization](http://arxiv.org/abs/2506.07479v1)**
### **[A Hybrid GA LLM Framework for Structured Task Optimization](http://arxiv.org/abs/2506.07483v1)**
### **[Drive Any Mesh: 4D Latent Diffusion for Mesh Deformation from Video](http://arxiv.org/abs/2506.07489v1)**
### **[SpatialLM: Training Large Language Models for Structured Indoor Modeling](http://arxiv.org/abs/2506.07491v1)**
### **[Explicit Preference Optimization: No Need for an Implicit Reward Model](http://arxiv.org/abs/2506.07492v1)**
### **[Genesis: Multimodal Driving Scene Generation with Spatio-Temporal and Cross-Modal Consistency](http://arxiv.org/abs/2506.07497v1)**
### **[Large Language Models for Multilingual Vulnerability Detection: How Far Are We?](http://arxiv.org/abs/2506.07503v1)**
### **[LeVo: High-Quality Song Generation with Multi-Preference Alignment](http://arxiv.org/abs/2506.07520v1)**
### **[Towards Large Language Models with Self-Consistent Natural Language Explanations](http://arxiv.org/abs/2506.07523v1)**
### **[BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation](http://arxiv.org/abs/2506.07530v1)**
### **[MoQAE: Mixed-Precision Quantization for Long-Context LLM Inference via Mixture of Quantization-Aware Experts](http://arxiv.org/abs/2506.07533v1)**
### **[Bit-level BPE: Below the byte boundary](http://arxiv.org/abs/2506.07541v1)**
### **[ChemAgent: Enhancing LLMs for Chemistry and Materials Science through Tree-Search Based Tool Learning](http://arxiv.org/abs/2506.07551v1)**
### **[SELT: Self-Evaluation Tree Search for LLMs with Task Decomposition](http://arxiv.org/abs/2506.07557v1)**
### **[SAFEFLOW: A Principled Protocol for Trustworthy and Transactional Autonomous Agent Systems](http://arxiv.org/abs/2506.07564v1)**
### **[Uncertainty-o: One Model-agnostic Framework for Unveiling Uncertainty in Large Multimodal Models](http://arxiv.org/abs/2506.07575v1)**
### **[Beyond the Sentence: A Survey on Context-Aware Machine Translation with Large Language Models](http://arxiv.org/abs/2506.07583v1)**
### **[MalGEN: A Generative Agent Framework for Modeling Malicious Software in Cybersecurity](http://arxiv.org/abs/2506.07586v1)**
### **[Explore the vulnerability of black-box models via diffusion models](http://arxiv.org/abs/2506.07590v1)**
### **[Evaluating LLMs Effectiveness in Detecting and Correcting Test Smells: An Empirical Study](http://arxiv.org/abs/2506.07594v1)**
### **[TwinBreak: Jailbreaking LLM Security Alignments based on Twin Prompts](http://arxiv.org/abs/2506.07596v1)**
### **[Instructing Large Language Models for Low-Resource Languages: A Systematic Study for Basque](http://arxiv.org/abs/2506.07597v1)**
### **[SceneRAG: Scene-level Retrieval-Augmented Generation for Video Understanding](http://arxiv.org/abs/2506.07600v1)**
### **[PolitiSky24: U.S. Political Bluesky Dataset with User Stance Labels](http://arxiv.org/abs/2506.07606v1)**
### **[Vuyko Mistral: Adapting LLMs for Low-Resource Dialectal Translation](http://arxiv.org/abs/2506.07617v1)**
### **[LoRMA: Low-Rank Multiplicative Adaptation for LLMs](http://arxiv.org/abs/2506.07621v1)**
### **[Return of ChebNet: Understanding and Improving an Overlooked GNN on Long Range Tasks](http://arxiv.org/abs/2506.07624v1)**
### **[Intent Matters: Enhancing AI Tutoring with Fine-Grained Pedagogical Intent Annotation](http://arxiv.org/abs/2506.07626v1)**
### **[SongBloom: Coherent Song Generation via Interleaved Autoregressive Sketching and Diffusion Refinement](http://arxiv.org/abs/2506.07634v1)**
### **[SWE-Dev: Building Software Engineering Agents with Training and Inference Scaling](http://arxiv.org/abs/2506.07636v1)**
### **[Fast ECoT: Efficient Embodied Chain-of-Thought via Thoughts Reuse](http://arxiv.org/abs/2506.07639v1)**
### **[TreeReview: A Dynamic Tree of Questions Framework for Deep and Efficient LLM-based Scientific Peer Review](http://arxiv.org/abs/2506.07642v1)**
### **[Evaluating LLMs Robustness in Less Resourced Languages with Proxy Models](http://arxiv.org/abs/2506.07645v1)**
### **[The Universality Lens: Why Even Highly Over-Parametrized Models Learn Well](http://arxiv.org/abs/2506.07661v1)**
### **[ProSplat: Improved Feed-Forward 3D Gaussian Splatting for Wide-Baseline Sparse Views](http://arxiv.org/abs/2506.07670v1)**
### **[QUITE: A Query Rewrite System Beyond Rules with LLM Agents](http://arxiv.org/abs/2506.07675v1)**
### **[Training Superior Sparse Autoencoders for Instruct Models](http://arxiv.org/abs/2506.07691v1)**
### **[NOVA3D: Normal Aligned Video Diffusion Model for Single Image to 3D Generation](http://arxiv.org/abs/2506.07698v1)**
### **[Evaluating Robustness in Latent Diffusion Models via Embedding Level Augmentation](http://arxiv.org/abs/2506.07706v1)**
### **[Interaction Analysis by Humans and AI: A Comparative Perspective](http://arxiv.org/abs/2506.07707v1)**
### **[Through the Valley: Path to Effective Long CoT Training for Small Language Models](http://arxiv.org/abs/2506.07712v1)**
### **[Consistent Video Editing as Flow-Driven Image-to-Video Generation](http://arxiv.org/abs/2506.07713v1)**
### **[NeurIPS 2025 E2LM Competition : Early Training Evaluation of Language Models](http://arxiv.org/abs/2506.07731v1)**
### **[Language Embedding Meets Dynamic Graph: A New Exploration for Neural Architecture Representation Learning](http://arxiv.org/abs/2506.07735v1)**
### **[RSafe: Incentivizing proactive reasoning to build robust and adaptive LLM safeguards](http://arxiv.org/abs/2506.07736v1)**
### **[ArchiLense: A Framework for Quantitative Analysis of Architectural Styles Based on Vision Large Language Models](http://arxiv.org/abs/2506.07739v1)**
### **[Research quality evaluation by AI in the era of Large Language Models: Advantages, disadvantages, and systemic effects](http://arxiv.org/abs/2506.07748v1)**
### **[Difference Inversion: Interpolate and Isolate the Difference with Token Consistency for Image Analogy Generation](http://arxiv.org/abs/2506.07750v1)**
### **[Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking](http://arxiv.org/abs/2506.07751v1)**
### **[REMoH: A Reflective Evolution of Multi-objective Heuristics approach via Large Language Models](http://arxiv.org/abs/2506.07759v1)**
### **[Diffusion Models-Aided Uplink Channel Estimation for RIS-Assisted Systems](http://arxiv.org/abs/2506.07770v1)**
### **[Language-Vision Planner and Executor for Text-to-Visual Reasoning](http://arxiv.org/abs/2506.07778v1)**
### **[Self-Cascaded Diffusion Models for Arbitrary-Scale Image Super-Resolution](http://arxiv.org/abs/2506.07813v1)**
### **[WebUIBench: A Comprehensive Benchmark for Evaluating Multimodal Large Language Models in WebUI-to-Code](http://arxiv.org/abs/2506.07818v1)**
### **[Guideline Forest: Experience-Induced Multi-Guideline Reasoning with Stepwise Aggregation](http://arxiv.org/abs/2506.07820v1)**
### **[Accelerating Diffusion Models in Offline RL via Reward-Aware Consistency Trajectory Distillation](http://arxiv.org/abs/2506.07822v1)**
### **[Addition in Four Movements: Mapping Layer-wise Information Trajectories in LLMs](http://arxiv.org/abs/2506.07824v1)**
### **[R3D2: Realistic 3D Asset Insertion via Diffusion for Autonomous Driving Simulation](http://arxiv.org/abs/2506.07826v1)**
### **[Improving large language models with concept-aware fine-tuning](http://arxiv.org/abs/2506.07833v1)**
### **[HAIBU-ReMUD: Reasoning Multimodal Ultrasound Dataset and Model Bridging to General Specific Domains](http://arxiv.org/abs/2506.07837v1)**
### **[Diffusion models under low-noise regime](http://arxiv.org/abs/2506.07841v1)**
### **[Jarzynski Reweighting and Sampling Dynamics for Training Energy-Based Models: Theoretical Analysis of Different Transition Kernels](http://arxiv.org/abs/2506.07843v1)**
### **[SAM2Auto: Auto Annotation Using FLASH](http://arxiv.org/abs/2506.07850v1)**
### **[Learning to Focus: Causal Attention Distillation via Gradient-Guided Token Pruning](http://arxiv.org/abs/2506.07851v1)**
### **[VIVAT: Virtuous Improving VAE Training through Artifact Mitigation](http://arxiv.org/abs/2506.07863v1)**
### **[Lightweight Sequential Transformers for Blood Glucose Level Prediction in Type-1 Diabetes](http://arxiv.org/abs/2506.07864v1)**
### **[Diffusion Counterfactual Generation with Semantic Abduction](http://arxiv.org/abs/2506.07883v1)**
### **[SoK: Data Reconstruction Attacks Against Machine Learning Models: Definition, Metrics, and Benchmark](http://arxiv.org/abs/2506.07888v1)**
### **[Video Unlearning via Low-Rank Refusal Vector](http://arxiv.org/abs/2506.07891v1)**
### **[Evaluating Large Language Models on the Frame and Symbol Grounding Problems: A Zero-shot Benchmark](http://arxiv.org/abs/2506.07896v1)**
### **[FunDiff: Diffusion Models over Function Spaces for Physics-Informed Generative Modeling](http://arxiv.org/abs/2506.07902v1)**
### **[Diffuse Everything: Multimodal Diffusion Models on Arbitrary State Spaces](http://arxiv.org/abs/2506.07903v1)**
### **[WeThink: Toward General-purpose Vision-Language Reasoning via Reinforcement Learning](http://arxiv.org/abs/2506.07905v1)**
### **[LUCIFER: Language Understanding and Context-Infused Framework for Exploration and Behavior Refinement](http://arxiv.org/abs/2506.07915v1)**
### **[Solving Inequality Proofs with Large Language Models](http://arxiv.org/abs/2506.07927v1)**
### **[Gradients: When Markets Meet Fine-tuning -- A Distributed Approach to Model Optimisation](http://arxiv.org/abs/2506.07940v1)**
### **[Adversarial Attack Classification and Robustness Testing for Large Language Models for Code](http://arxiv.org/abs/2506.07942v1)**
### **[ProtocolLLM: RTL Benchmark for SystemVerilog Generation of Communication Protocols](http://arxiv.org/abs/2506.07945v1)**
### **[TokenBreak: Bypassing Text Classification Models Through Token Manipulation](http://arxiv.org/abs/2506.07948v1)**
### **[Correlated Errors in Large Language Models](http://arxiv.org/abs/2506.07962v1)**
### **[Reinforcing Multimodal Understanding and Generation with Dual Self-rewards](http://arxiv.org/abs/2506.07963v1)**
### **[SpaCE-10: A Comprehensive Benchmark for Multimodal Large Language Models in Compositional Spatial Intelligence](http://arxiv.org/abs/2506.07966v1)**
### **[CyberV: Cybernetics for Test-time Scaling in Video Understanding](http://arxiv.org/abs/2506.07971v1)**
### **[HeuriGym: An Agentic Benchmark for LLM-Crafted Heuristics in Combinatorial Optimization](http://arxiv.org/abs/2506.07972v1)**
### **[OneIG-Bench: Omni-dimensional Nuanced Evaluation for Image Generation](http://arxiv.org/abs/2506.07977v1)**
### **[Rethinking Cross-Modal Interaction in Multimodal Diffusion Transformers](http://arxiv.org/abs/2506.07986v1)**
### **[Supporting Construction Worker Well-Being with a Multi-Agent Conversational AI System](http://arxiv.org/abs/2506.07997v1)**
### **[Generative Modeling of Weights: Generalization or Memorization?](http://arxiv.org/abs/2506.07998v1)**
### **[MADFormer: Mixed Autoregressive and Diffusion Transformers for Continuous Image Generation](http://arxiv.org/abs/2506.07999v1)**
### **[Reparameterized LLM Training via Orthogonal Equivalence Transformation](http://arxiv.org/abs/2506.08001v1)**
### **[Dynamic View Synthesis as an Inverse Problem](http://arxiv.org/abs/2506.08004v1)**
