# The Latest Daily Papers - Date: 2025-06-11
## Highlight Papers
### **[Why Masking Diffusion Works: Condition on the Jump Schedule for Improved Discrete Diffusion](http://arxiv.org/abs/2506.08316v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper "Why Masking Diffusion Works: Condition on the Jump Schedule for Improved Discrete Diffusion" aims to explain the surprising effectiveness of masking diffusion models compared to other discrete diffusion models. The authors argue that masking diffusion's success stems from implicitly conditioning the backward process on the known distribution of transition times (jumps) in the forward process.  They introduce a new framework called Schedule-Conditioned Discrete Diffusion (SCUD) that explicitly incorporates this knowledge of transition times into the model architecture, generalizing both classical discrete diffusion and masking diffusion. SCUD models are shown to outperform existing methods on image, protein, and language data, highlighting the importance of explicitly modeling the transition schedule. The key idea is that by "baking in" the *when* (transition schedule), the model only needs to learn *where* to transition.

**Critical Evaluation:**

* **Novelty:** The paper introduces a novel perspective on why masking diffusion performs well. The SCUD framework is a significant contribution, as it provides a unifying view of different discrete diffusion models and offers a principled way to design new models by explicitly conditioning on the transition schedule.  The idea of decomposing the learning objective into *when* and *where* transitions is conceptually novel.  The theoretical analysis linking SCUD to masking diffusion and classical diffusion is insightful.

* **Significance:** The paper addresses a significant puzzle in the field: why does the simplest possible forward process (masking) often outperform more complex ones? By providing a compelling explanation and a framework for leveraging this insight, the paper has the potential to guide future research in discrete diffusion. The empirical results demonstrate that SCUD models can achieve state-of-the-art performance on a variety of tasks, suggesting that SCUD offers a substantial improvement over existing methods. The result that SCUD is able to achieve strong results in biological sequence design by incorporating prior knowledge (BLOSUM matrix) is compelling. Furthermore, the paper opens up new avenues for exploring structured forward processes, overcoming previous limitations. The result that SCUD makes certain large vocabularies computationally feasible is also a significant outcome of this work.

* **Strengths:**
    * **Clear and concise explanation:** The paper clearly articulates the problem, the proposed solution (SCUD), and the theoretical justification for it.
    * **Strong theoretical grounding:** The theoretical analysis provides a solid foundation for the SCUD framework.
    * **Comprehensive empirical validation:** The paper presents experimental results on image, protein, and language datasets, demonstrating the effectiveness of SCUD across different domains.
    * **Unifying Framework:** The SCUD method is able to generalize both masking diffusion and classical diffusion approaches, suggesting that it is indeed an improvement.
    * **Open-source code:** The authors release the code of their project, making the results more reproducible and enabling more researchers to use SCUD models.

* **Weaknesses:**
    * **Complexity of SCUD:** The SCUD framework introduces some additional complexity compared to standard diffusion models, which might be a barrier to adoption for some researchers.  However, the paper provides helpful guidance on how to implement SCUD efficiently.
    * **Potential for over-conditioning:** While conditioning on the jump schedule is beneficial, there might be a risk of over-conditioning if too much information is included in the conditioning variable S. The paper touches on this issue in the discussion, but further investigation is warranted.

* **Potential Influence:** The paper has the potential to significantly influence the field of discrete diffusion. It provides a new perspective on model design, offers a principled framework for improving performance, and demonstrates promising results on a range of tasks.  The SCUD framework is likely to be adopted by other researchers and used as a basis for future work. The observation that it is crucial to fit the when and where of transitions is important for the future directions of the field.

**Justification:**

I believe the paper provides a substantial and potentially transformative advancement in discrete diffusion modelling.  The explanation of masking diffusion's success and the introduction of the SCUD framework are both novel and significant.  The empirical results are compelling, and the paper is well-written and easy to understand. The paper successfully bridges theory and practice by identifying a theoretical flaw in many discrete diffusion models and then using this information to create a better model.

Score: 9

- **Score**: 9/10

### **[Unable to forget: Proactive lnterference Reveals Working Memory Limits in LLMs Beyond Context Length](http://arxiv.org/abs/2506.08184v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Unable to Forget: Proactive Interference Reveals Working Memory Limits in LLMs Beyond Context Length":

**Summary:**

The paper investigates the effects of proactive interference (PI) on information retrieval in Large Language Models (LLMs). Adapting the PI paradigm from cognitive science, the authors introduce PI-LLM, an evaluation that sequentially streams semantically related key-value updates and queries only the final values. They find that LLM retrieval accuracy declines log-linearly toward zero as interference accumulates, even when the target information is clearly positioned in the prompt. Prompt engineering aimed at mitigating this interference is ineffective. These results suggest that LLMs have a fundamental constraint in their ability to disentangle interference and flexibly manipulate information, indicating a working memory bottleneck beyond just context length limitations. The paper quantifies this bottleneck with an Interference Endurance Score (IES) and demonstrates that IES correlates more strongly with model size than context length.

**Critical Evaluation:**

*   **Novelty:** The paper's primary novelty lies in its application of the cognitive science paradigm of proactive interference to LLMs. While LLMs are known to struggle with retrieval tasks, the authors offer a distinct perspective by isolating and quantifying the *interference* component, rather than simply focusing on long-context issues. The introduction of PI-LLM as a specific benchmark and the IES metric adds to this novelty. The emphasis on working memory-like limitations is a useful framing.

*   **Significance:** The paper's findings are significant for several reasons:

    *   It challenges the prevalent assumption that simply increasing context length will resolve retrieval problems in LLMs.  It provides evidence that LLMs also have a 'working memory' issue, where models can't appropriately ignore previously learned information even if more recent information is presented.
    *   It highlights a fundamental difference between LLMs and human cognition, where humans have mechanisms for active forgetting (or unbinding) that LLMs seem to lack. This guides future research toward architectural improvements.
    *   The finding that model size correlates more with IES than context length provides insight into the architectural factors that may improve retrieval.
    *   The paper is well-executed experimentally, with a rigorous design that isolates the key variables.  The use of a variety of models from different sources strengthens the generalizability of the findings.
    *   The negative results on natural language prompt-based mitigations further highlight the limitations and offer direction for more sophisticated interventions.

*   **Strengths:**

    *   **Clear Problem Definition:** The paper clearly defines and isolates the problem of proactive interference in LLMs.
    *   **Rigorous Methodology:** The experimental design is well-controlled, isolating the effects of interference from other factors like context length.
    *   **Quantitative Results:** The paper provides quantitative results (log-linear decay curves, IES metric) that are convincing and easy to interpret.
    *   **Broad Model Coverage:** Testing on a wide variety of models provides strong evidence for the generalizability of the findings.
    *   **Insightful Discussion:** The paper provides a thoughtful discussion of the implications of the findings for LLM architecture and future research.

*   **Weaknesses:**

    *   **Synthetic Task:** The PI-LLM benchmark is a synthetic task. While well-controlled, the extent to which the observed effects translate to real-world applications requires further investigation.  It is valuable for diagnosis, but ecological validity may be limited.
    *   **Limited Mitigation Strategies:** While the paper explores prompt engineering as a mitigation strategy, the focus is somewhat limited.  Exploring other architectural interventions or training strategies to improve anti-interference capacity could strengthen the findings. The successful prompt-hacking method is ad-hoc and not fully understood.
    *   **Mechanism Speculation:** The paper suggests that the unitary interference limit is related to the model’s ability to decide *how* to use the information and *when* to forget the information, but doesn't provide direct evidence for specific mechanisms, leading to a reliance on suggestive parallels with human cognition.

*   **Potential Influence:**

    *   The paper will likely shift the research focus from solely long-context retrieval to also addressing interference in LLMs.
    *   It will stimulate research into architectural modifications that can improve LLMs' ability to manage interference, perhaps drawing inspiration from human working memory models.
    *   The PI-LLM benchmark and IES metric will likely be adopted by other researchers for evaluating LLMs.
    *   It offers an argument for why simply training on more data may not always solve specific issues.

**Score:** 8

**Rationale:** The paper makes a novel and significant contribution to the understanding of LLM limitations, moving beyond the dominant focus on context length and proposing a well-defined framework for studying interference. The strengths of the experimental design and the breadth of the models tested outweigh the weaknesses related to the synthetic task and limited exploration of specific architectural mechanisms. The paper’s potential influence on future research in LLM architecture and evaluation is substantial. The shift of emphasis and the provided tools for measurement are high value in pushing the field forward. While it isn't a paradigm shift, it is significant progress.

- **Score**: 8/10

### **[Instruction-Tuned Video-Audio Models Elucidate Functional Specialization in the Brain](http://arxiv.org/abs/2506.08277v1)**
- **Summary**: Okay, I've reviewed the paper "Instruction-Tuned Video-Audio Models Elucidate Functional Specialization in the Brain." Here's a summary and critical evaluation:

**Summary**

The paper investigates the alignment between instruction-tuned multimodal large language models (MLLMs) and human brain activity, specifically fMRI data acquired while subjects watched movie clips (video with audio). The core idea is to use instruction-specific embeddings derived from these MLLMs to predict brain activity and thereby understand how well these models capture the brain's functional organization. The authors compared instruction-tuned video and audio MLLMs, as well as non-instruction-tuned multimodal and unimodal models, with 13 video and 5 audio task-specific instructions. The key findings are that instruction-tuned video MLLMs align significantly better with brain activity compared to other model types. Furthermore, different instructions lead to distinct activity patterns across brain regions, demonstrating the models' ability to differentiate task-specific representations. The paper finds a hierarchical alignment, where early sensory areas align with earlier model layers and higher-level cognitive areas align with deeper model layers. They also show that specific tasks are more strongly associated with specific brain regions (e.g., narrative understanding aligns with language areas). Finally, they used variance partitioning to understand shared and unique contributions of different task instructions to brain activity.

**Critical Evaluation**

*   **Novelty:** The paper addresses a relevant and current research question in the field of brain-model correspondence. While previous studies have investigated MLLM alignment with brain data, the focus on instruction-tuned *video and audio* MLLMs *with multimodal stimuli* and a systematic comparison across various model types and a significant number of task-specific instructions, is a unique and valuable contribution. This is different from the common paradigm where unimodal stimuli is used to probe a multimodal model, or where non-instruction-tuned models are used for multimodal stimuli.

*   **Significance:** The findings have significant implications for understanding the neural basis of multimodal information processing and the role of task-specific instructions. Demonstrating improved brain alignment with instruction-tuned models suggests that these models capture more brain-relevant representations. This offers a path for creating better models for brain encoding. The discovery that these models functionally disentangle different regions of the brain is also a crucial step towards understanding the workings of the brain. This means the work provides a pathway for creating better stimuli to probe the workings of the brain, and a framework for better cognitive modeling.

*   **Strengths:**

    *   **Comprehensive Analysis:** The paper presents a thorough analysis, comparing several MLLM architectures and task instructions.
    *   **Clear Methodology:** The methodology is well-defined, and the use of banded ridge regression and variance partitioning offers a robust approach to brain encoding.
    *   **Well-Justified Experimental Setup:** Choice of Movie10 dataset and different instructions is well justified in terms of real-world scenarios.
    *   **Insightful Findings:** The results provide clear evidence of improved brain alignment with instruction-tuned video MLLMs and the functional specificity of different task instructions. The hierarchical layer alignment is particularly compelling.
    *   **Publicly Available Code:** The public availability of the code enhances reproducibility and facilitates further research in this area.

*   **Weaknesses:**

    *   **Model Size and Complexity:** While the models used are state-of-the-art, it's worth acknowledging that even larger and more complex MLLMs could potentially reveal even stronger brain alignment.
    *   **fMRI Limitations:** fMRI has inherent limitations in temporal resolution, which could impact the ability to capture precise neural dynamics. Future work could explore methods with better temporal resolution, such as EEG or MEG.
    *   **Limited number of subjects:** The dataset used is limited to only 4 subjects. The use of more robust analysis on more participants would strengthen the claim of robust alignment.

*   **Potential Influence:** The paper is likely to stimulate further research in brain-model correspondence using instruction-tuned MLLMs. It opens opportunities to explore the neural underpinnings of complex cognitive functions like narrative understanding, social cognition, and reasoning. It provides a strong foundation for using these types of models to design experiments, probe the brain, and construct testable cognitive models of the brain.

*   **Reasoning for assigned score:** The strengths of this paper lie in addressing an important problem in brain-model correspondence (improving alignment of model representations with the brain), providing a novel evaluation of a variety of instruction-tuned models for the first time with video and audio stimuli, and conducting a thorough analysis with insightful results. While there are weaknesses around dataset size and possible improvement of the model, the paper still showcases important advances in the field, especially for providing an avenue for understanding the workings of the brain.

**Score: 8**

- **Score**: 8/10

### **[Serendipitous Recommendation with Multimodal LLM](http://arxiv.org/abs/2506.08283v1)**
- **Summary**: Okay, here's a summary and critical evaluation of the paper "Serendipitous Recommendation with Multimodal LLM":

**Summary**

The paper addresses the problem of improving serendipity in recommendation systems, specifically in short-form video platforms.  It proposes a hierarchical framework integrating Multimodal Large Language Models (MLLMs) with conventional recommendation models.  The MLLM is used for high-level planning, identifying potentially serendipitous interest clusters based on multimodal content understanding (text and video).  The MLLM's output guides a traditional recommendation model to suggest items within these clusters. The authors demonstrate the effectiveness of their approach through live experiments on a large-scale video platform, showing improvements in both recommendation serendipity and user satisfaction.  They utilize chain-of-thought prompting with the MLLM to discover novel user interests.

**Critical Evaluation**

*Novelty:*

The paper's novelty lies in the specific combination of techniques and its application to a real-world, large-scale recommendation system. While using LLMs and MLLMs for recommendation is not entirely new, the hierarchical approach, fine-tuning the MLLM for *serendipity* (rather than just relevance), utilizing multimodal content understanding at scale, and demonstrating impact via live A/B testing are significant contributions. The chain-of-thought strategy to identify novel user interests is also a novel and valuable component.  Specifically, the work goes beyond simply generating item embeddings and instead, focuses on using MLLMs to steer the recommendation process towards novel clusters, which adds a higher-level strategic element. Compared to existing LLM-based approaches, the proposed method is significantly more focused on user experience.

*Significance:*

The significance of this work stems from its practical impact and its ability to address key challenges in recommendation systems.  The demonstration of improved serendipity and user satisfaction in a live setting on a platform serving billions of users is a strong indication of real-world applicability.  The paper provides a blueprint for effectively integrating computationally expensive MLLMs into large-scale systems while maintaining efficiency. The experiments are comprehensive, exploring different input modalities, prompting strategies, and evaluation metrics. By directly incorporating video content, the paper advances beyond the reliance on textual metadata that many existing systems use.

*Strengths:*

*   **Real-world Application:** The live experiments on a large-scale platform are a major strength, providing strong evidence of the approach's effectiveness.
*   **Systematic Evaluation:** The paper includes a thorough evaluation, including offline metrics to analyze the MLLM's understanding and online A/B testing to measure user impact.
*   **Clear Problem Definition:** The paper clearly defines the problem of limited exploration and provides a well-motivated solution.
*   **Detailed Methodology:** The paper provides sufficient detail about the architecture, training process, and experimental setup, which enables reproducibility.
*   **Multimodal Understanding:** Directly incorporating visual information, rather than relying solely on text-based representations, is a significant step forward.

*Weaknesses:*

*   **Generalizability:** While the results are impressive, it's important to consider the generalizability of the findings to other domains. Short-form video may have unique characteristics that make this approach particularly effective.
*   **MLLM Choice:** The paper doesn't provide a detailed justification for the specific MLLM architecture used (Gemini 1.5). It's unclear how much the performance is dependent on the choice of MLLM versus the overall framework.
*   **Hyperparameter Sensitivity:** The paper lacks in-depth analysis of hyperparameter sensitivity, especially with respect to fine-tuning the MLLM.
*   **Ethical Considerations:** The paper omits a discussion on the potential ethical implications of enhanced recommendation systems (e.g., echo chambers, filter bubbles), especially when they influence user exploration.

*Justification for Score:*

The paper presents a significant advance in the application of MLLMs to recommendation systems. The approach is well-motivated, thoroughly evaluated, and demonstrably effective in a real-world setting. While some limitations exist regarding generalizability and detailed architectural justifications, the practical impact and the novel integration of techniques warrant a high score.

Score: 8

- **Score**: 8/10

### **[From Debate to Equilibrium: Belief-Driven Multi-Agent LLM Reasoning via Bayesian Nash Equilibrium](http://arxiv.org/abs/2506.08292v1)**
- **Summary**: Here's a concise summary and a critical evaluation of the paper:

**Summary:**

The paper introduces ECON (Efficient Coordination via Nash Equilibrium), a novel framework for multi-agent LLM reasoning.  ECON addresses the limitations of existing multi-agent debate methods by recasting multi-LLM coordination as an incomplete-information game and seeking a Bayesian Nash Equilibrium (BNE). It replaces costly inter-agent communication with a belief-based coordination mechanism where LLMs maintain and update probabilistic beliefs about their peers. The paper provides theoretical justification by proving the existence of a BNE and deriving a sublinear regret bound. Empirically, ECON outperforms single-agent baselines and existing multi-agent approaches on various reasoning and planning tasks while reducing token usage.  It also demonstrates scalability by incorporating additional models.

**Critical Evaluation:**

*   **Novelty:** The core novelty lies in framing multi-agent LLM coordination as an incomplete-information game solved via BNE. This approach deviates from explicit message-passing schemes commonly used in multi-agent debate.  The use of belief networks to model agent beliefs and a centralized mixing network to guide the system towards a BNE are also novel contributions. The hierarchical structure to scale to more agents is also a significant contribution.

*   **Significance:** The paper addresses crucial challenges in multi-agent LLM systems, including high computational cost, scalability limitations, and lack of convergence guarantees. By achieving efficient coordination, ECON makes multi-agent reasoning more practical for resource-constrained environments. The empirical results demonstrate significant performance gains over existing methods, underscoring its potential to improve reasoning and planning in LLMs. The reduced token usage and better scalability compared to MAD are significant practical advantages. The theoretical analysis provides a solid foundation for understanding ECON's behavior and guarantees.

*   **Strengths:**
    *   Strong theoretical justification: The paper provides a rigorous theoretical analysis of ECON, including proof of BNE existence and regret bounds. This offers a solid foundation for understanding the framework's behavior and guarantees.
    *   Empirical validation: The empirical results demonstrate significant performance gains over strong baselines on various challenging tasks. This validates the practical effectiveness of ECON.
    *   Scalability:  The paper demonstrates ECON's ability to scale to larger ensembles of LLMs, addressing a major limitation of existing multi-agent approaches.
    *   Reduced communication cost: ECON reduces token usage compared to multi-agent debate methods, making it more cost-effective and suitable for resource-constrained settings.
    *   The writing is clear and well organized.

*   **Weaknesses:**
    *   Complexity: While the BNE formulation is novel, the implementation involves multiple components (belief networks, belief encoder, mixing network), adding to the overall complexity of the system. This might make it harder to reproduce and adopt.
    *   Assumption B.8 (Concentrability) is strong and needs more justification.
    *   More detail on the limitations of each of the steps in the hierarchy.
    *   Ablation studies are fine, but more direct comparisons could have been helpful. The scaling is done independently.

*   **Potential Influence:** ECON's approach could significantly influence future research in multi-agent LLM systems.  It provides a principled and efficient way to coordinate LLMs, enabling the development of larger and more powerful reasoning and planning systems.  The use of belief networks and BNE could inspire new techniques for modeling and optimizing multi-agent interactions.  The reduced token usage could make multi-agent LLM reasoning more accessible to a wider range of users and applications. The scale-up algorithm is promising, and will be interesting to see if it is incorporated by others.

*   **Justification for Score:**
    The paper presents a novel framework with solid theoretical grounding and promising empirical results. While the implementation adds complexity, the potential benefits in terms of efficiency, scalability, and performance justify this complexity. The theoretical analysis and the empirical validation are both quite strong. This demonstrates that a multi-agent scheme can be both high performing and cost effective. While there are limitations and improvements that could be made, the paper represents a significant advance in the field.

Score: 8

- **Score**: 8/10

### **[From Passive to Active Reasoning: Can Large Language Models Ask the Right Questions under Incomplete Information?](http://arxiv.org/abs/2506.08295v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces AR-Bench, a novel benchmark designed to evaluate the active reasoning abilities of large language models (LLMs).  Unlike existing benchmarks that primarily assess passive reasoning (where all necessary information is provided), AR-Bench requires LLMs to interact with external "answerer" agents to acquire missing information necessary to solve tasks.  The benchmark includes three task families: detective cases, situation puzzles, and guessing numbers, designed to probe commonsense, logical, and symbolic reasoning skills.  The authors conduct extensive experiments with state-of-the-art LLMs and show significant performance gaps compared to human performance, highlighting the current limitations of LLMs in active reasoning.  They also explore the effects of various reasoning methods, training strategies (SFT, DPO), and scaling efforts, finding only modest gains.  The paper emphasizes the need for new methodologies to advance active reasoning in LLMs, such as interactive learning and real-time feedback loops.

**Critical Evaluation:**

*   **Novelty:**  The key strength of this paper is the introduction of the AR-Bench benchmark.  While proactive questioning and retrieval-augmented generation have been explored, a comprehensive and unified benchmark focused specifically on *active* reasoning in LLMs is a significant and useful contribution.  The focus on requiring LLMs to *ask* for information is genuinely innovative within the broader context of LLM evaluation.
    *   The individual task families (detective cases, puzzles, guessing numbers) aren't entirely novel in isolation, but the *combination*, controlled generation and *structured interactive setting* using rule-based feedback as well as LLMs as the agent provide a useful environment for evaluation that offers a level of abstraction from real-world complexity.
*   **Significance:**  The paper directly addresses a critical gap in LLM evaluation.  As LLMs are increasingly deployed in real-world agentic applications (e.g., assistants, robots), their ability to actively acquire information becomes crucial.  By demonstrating the limitations of current LLMs in active reasoning, the paper highlights a crucial area for future research and development. This benchmark has the potential to guide researchers towards developing more interactive and adaptive LLMs.
    *   The analysis and breakdown of why current methods fail (e.g., unreliable verifiers, low-quality questions) provides concrete directions for future research.
    *   It correctly emphasizes that while it’s an abstraction over real-world complexity, the structure and control are useful as a first step to then explore the findings in real-world application.
*   **Strengths:**
    *   **Clear problem definition:** The paper clearly articulates the difference between passive and active reasoning.
    *   **Well-designed benchmark:** AR-Bench is a well-structured and thoughtfully designed benchmark that targets different aspects of reasoning.
    *   **Comprehensive evaluation:** The authors conduct extensive experiments using a variety of LLMs, reasoning methods, and training strategies.
    *   **Insightful analysis:**  The paper provides a detailed analysis of the experimental results, identifying the strengths and weaknesses of current LLMs in active reasoning.
    *   **Open Source nature**: The benchmark is also publicly available and the code is well documented increasing its adoption and reproducibility.

*   **Weaknesses:**
    *   **Synthetic Environment:** AR-Bench relies on synthetic environments and simulated interactions. While this provides control and scalability, it may not fully capture the complexities and nuances of real-world interactions.
    *   **Simplified Feedback**: Rule-based answerers as well as feedback are limited in their ability to model a real-world scenario, particularly for detective tasks, which include understanding emotion, deceit, and hidden motives. This results in lack of complexity, which leads to more unrealistic results.
    *   **Limited exploration of solutions**: The study evaluates a range of existing methods, but does not propose or test new active reasoning approaches.
    *   **Judge LLMs**: Reliance on LLM judges introduces bias, and lack human judgment provides more inaccurate evaluation.
*   **Potential Influence:** AR-Bench is likely to have a significant influence on the field of LLM research. It provides a valuable tool for evaluating and improving the active reasoning abilities of LLMs. The benchmark is well-defined and readily accessible, making it easy for other researchers to use. The findings and the concrete issues raised in the experiments will be influential.

**Score: 8**

**Rationale:**
The AR-Bench benchmark is a significant and novel contribution that addresses a critical gap in LLM evaluation. Its strengths are the clearly articulated problem definition, robust design, comprehensive evaluation, insightful analysis, and public availability. The primary weakness is the reliance on synthetic environments. The paper effectively highlights the limitations of current LLMs in active reasoning and has the potential to drive significant advancements in the field. A score of 8 reflects the high novelty and impact of the AR-Bench benchmark, offset slightly by its reliance on synthetic environments.

- **Score**: 8/10

### **[TACTIC: Translation Agents with Cognitive-Theoretic Interactive Collaboration](http://arxiv.org/abs/2506.08403v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "TACTIC: Translation Agents with Cognitive-Theoretic Interactive Collaboration":

**Summary:**

The paper introduces TACTIC, a novel multi-agent framework for machine translation inspired by Cognitive Translation Studies (CTS).  Unlike traditional NMT or LLM-based translation approaches, TACTIC explicitly models key cognitive processes observed in human translators.  It comprises six specialized agents: ResearchAgent, ContextAgent, DraftAgent, RefinementAgent, EvaluationAgent, and ScoreAgent. These agents collaborate in a two-stage workflow: a base workflow for simpler translations and a complex, iterative workflow leveraging external knowledge and context for more challenging tasks.  The framework is evaluated on FLORES-200 and WMT24 benchmarks across diverse language pairs, demonstrating state-of-the-art performance. The authors use DeepSeek-V3 as the backbone model, demonstrating improvements over existing models, including GPT-4.1.  Code is made available.

**Critical Evaluation:**

*   **Novelty:** The paper presents a highly novel approach to machine translation. While multi-agent systems have been explored for translation before, TACTIC's key innovation lies in its grounding in Cognitive Translation Studies. The direct mapping of human cognitive processes to specific agent roles is a significant advancement. This offers a principled and interpretable way to structure the translation process, unlike many existing multi-agent systems that may be more ad hoc in their design. The explicit modeling of translation strategies (literal, sense-for-sense, free) within the DraftAgent also adds to the novelty. The paper also shows improved results over baseline models such as GPT-4.1 and DeepSeek-R1.

*   **Significance:** TACTIC's significance stems from several factors:

    *   **Improved Translation Quality:**  The empirical results convincingly demonstrate state-of-the-art performance on standard benchmarks. The gains are consistent across different language pairs and datasets, indicating robustness. Specifically, the ability to surpass GPT-4.1 is a very significant finding.

    *   **Interpretability:**  By aligning with CTS principles, TACTIC offers a more interpretable and explainable approach to machine translation. This is crucial for understanding how LLMs are utilized for translation and for identifying areas for further improvement. The ablation studies further solidify these findings.

    *   **Framework for Future Research:** TACTIC provides a strong foundation for future research in multi-agent translation. The modular design allows for easy experimentation with different agent implementations and collaboration strategies. The paper’s insights can guide the development of more cognitively informed and human-like translation systems.

*   **Strengths:**
    *   Strong theoretical grounding in Cognitive Translation Studies.
    *   Well-defined agent roles and collaboration workflow.
    *   Comprehensive empirical evaluation on diverse benchmarks.
    *   Demonstrated state-of-the-art performance.
    *   Ablation studies provide insights into the contribution of individual agents.
    *   Interpretability of the framework.
    *   The paper provides insightful case studies and error analysis.

*   **Weaknesses:**

    *   **Reliance on Automatic Evaluation:** The paper relies heavily on automatic evaluation metrics (XCOMET, COMETKiwi). While these metrics are widely used, they are not perfect proxies for human judgment.  The authors acknowledge this limitation, but further human evaluation would strengthen the findings.

    *   **Computational Cost:**  Multi-agent systems tend to be more computationally expensive than direct translation models. The paper does not fully address the efficiency implications of TACTIC, although they do provide some computational details in the appendix. Further analysis of inference time and resource requirements would be beneficial.

    *   **Generalization Beyond English-Centric Translation:** While the results are robust across different language pairs, the framework is primarily evaluated on English-centric translation tasks. Investigating its performance on non-English-centric pairs would further demonstrate its generalizability.

    *   **Limited Complexity of Agents:** Each agent follows simple prompting techniques. While the integration is complex, the agent roles themselves may be expanded for better performance.

*   **Potential Influence:** This paper has the potential to significantly influence the field of machine translation.  It demonstrates the value of incorporating cognitive principles into translation system design.  It could lead to the development of more human-like, interpretable, and effective translation technologies. By bridging the gap between human translation cognition and machine translation architectures, this paper provides a novel perspective for translation modeling in the era of large language models.

**Score: 8.5**

**Justification:** The paper presents a highly novel and significant contribution to the field of machine translation. Its grounding in Cognitive Translation Studies and its demonstrated state-of-the-art performance make it a strong contender for influencing future research in multi-agent translation systems. The weaknesses, primarily related to reliance on automatic metrics and computational cost, are valid concerns but do not outweigh the paper's overall strengths. Given the increased performance over GPT-4.1 and the theoretical backing of CTS, the paper introduces a new approach to LLMs and translation.

- **Score**: 8/10

### **[CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmark of Large Language Models in Mental Health Counseling](http://arxiv.org/abs/2506.08584v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces CounselBench, a large-scale benchmark for evaluating and stress-testing large language models (LLMs) in the context of single-turn mental health counseling. The benchmark comprises two components: CounselBench-EVAL, which contains expert evaluations of LLM and human therapist responses to real patient questions, and CounselBench-ADV, an adversarial dataset of expert-authored questions designed to trigger specific LLM failure modes. The study involved 100 mental health professionals who provided expert evaluations based on clinically grounded dimensions, including overall quality, empathy, specificity, factual consistency, medical advice, and toxicity. The results indicate that LLMs often outperform online human therapists in perceived quality, but experts frequently flag safety concerns related to unauthorized medical advice. The study also found that LLM judges consistently overrate model responses and overlook safety issues identified by human experts. The adversarial dataset revealed consistent, model-specific failure patterns, demonstrating the potential of CounselBench as a framework for benchmarking and improving LLM behavior in high-stakes mental health settings.

**Rigorous and Critical Evaluation:**

*   **Novelty:** The novelty of this paper lies in the scale and clinical grounding of the benchmark it introduces. While prior work has explored LLMs in mental health contexts, CounselBench stands out due to the:

    *   **Large-scale expert evaluation:**  Involving 100 mental health professionals in the annotation process is a significant strength, lending credibility and clinical relevance to the benchmark. This contrasts with many previous studies that rely on smaller-scale human annotation or crowd-sourced judgments.
    *   **Clinically-grounded dimensions:** The evaluation rubric is based on established clinical psychology literature and expert consultation, ensuring that the assessment criteria are appropriate and meaningful for mental health counseling.
    *   **Adversarial dataset:** CounselBench-ADV is a novel approach to systematically probing LLM failure modes in this domain, allowing for targeted identification and mitigation of safety risks.
    *   **Comparison to real-world data:**  Comparing the LLM outputs not only to each other but also against therapist-provided responses on a real platform is a valuable contribution, situating the models' performance in a practical, immediately relevant context.
*   **Significance:** The significance of this work is substantial:

    *   **Addressing a Critical Need:** With the increasing interest in using LLMs for mental health support, a robust benchmark is essential for ensuring safety and quality. CounselBench provides a valuable tool for researchers and developers to evaluate and improve LLM behavior in this sensitive domain.
    *   **Identifying Key Failure Modes:** The study identifies concrete failure modes, such as the provision of unauthorized medical advice and overgeneralization, which are crucial for guiding future LLM development efforts. This offers a targeted agenda for research focused on improving safety.
    *   **Highlighting Limitations of LLM Judges:** Demonstrating that LLM judges may not reliably identify safety issues underscores the importance of human expert evaluation in this context. This is a very important finding that could have significant ethical implications.
*   **Strengths:**

    *   **Well-defined methodology:** The paper presents a clear and detailed description of the benchmark construction, annotation protocol, and experimental setup.
    *   **Rigorous analysis:** The study employs appropriate statistical methods for analyzing the expert evaluations and LLM performance.
    *   **Comprehensive evaluation:** CounselBench covers a broad range of mental health topics and includes both expert and LLM-based evaluations.
*   **Weaknesses:**

    *   **Single-turn setting:** Focusing on single-turn counseling limits the assessment of LLMs' ability to engage in more extended and dynamic therapeutic interactions.  While the justification is given (alignment with common scenarios), it does represent a limitation in scope.
    *   **Bias in source data:** Reliance on upvotes as a measure of quality could introduce bias in the selection of human therapist responses.
    *   **Limited generalizability:** While the annotator sample is diverse in terms of license/degrees and specializations, it is geographically restricted to U.S. professionals.
    *   **Evaluation Metric Limitations:** While the authors did have to create their own metrics, this can always cause an issue for comparability with other research.
*   **Potential Influence:** CounselBench has the potential to significantly influence the field by:

    *   **Establishing a standard benchmark:**  Providing a common framework for evaluating LLMs in mental health counseling, enabling more consistent and comparable research.
    *   **Guiding LLM development:**  Informing the development of safer and more effective LLMs for mental health support by highlighting key failure modes and areas for improvement.
    *   **Promoting ethical AI:**  Emphasizing the importance of clinical expertise and safety considerations in the design and deployment of LLMs for mental health.

**Score: 8**

**Rationale:** The paper presents a novel and significant contribution to the field of LLMs and mental health. The development of CounselBench and the rigorous evaluation conducted by the authors make this a highly valuable resource for researchers and developers. The weaknesses primarily relate to scope rather than fundamental flaws in methodology or analysis. The paper makes a compelling argument for the need for clinically-grounded evaluation and highlights the limitations of relying solely on LLMs for assessing performance in this sensitive domain.

- **Score**: 8/10

### **[Leveraging LLMs to Evaluate Usefulness of Document](http://arxiv.org/abs/2506.08626v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper "Leveraging LLMs to Evaluate Usefulness of Document" addresses the limitations of the traditional Cranfield paradigm in information retrieval evaluation, particularly its weak correlation with user satisfaction and the high cost of relevance annotation. The authors propose a new user-centric evaluation framework called Cascade LLM-based Usefulness Evaluation (CLUE). CLUE integrates user search context and behavioral data into large language models (LLMs) to generate multilevel usefulness labels. This framework uses a cascading judgment structure inspired by ordinal regression techniques.  The paper demonstrates that CLUE, when well-guided with context and behavioral information, can accurately evaluate usefulness and improve satisfaction prediction. The authors conduct ablation studies to investigate the influence of key components within the framework and show that the generated labels enhance the performance of satisfaction prediction models in real-world experiments.  The key innovations are the user-centric prompts, the cascade structure inspired by ordinal regression, and the integration of user behavior information.

**Critical Evaluation:**

*   **Novelty:** The paper introduces a novel approach to leveraging LLMs for usefulness judgment that goes beyond simple relevance assessment. The cascade structure inspired by ordinal regression is a valuable contribution, addressing the ordinal nature of usefulness judgments more directly than prior methods.  The careful design of prompts to incorporate user context and behavior is a significant improvement over methods that only consider document content. The collection of a new dataset specifically designed to capture user thoughts and document text content is a notable strength.
*   **Significance:** The work addresses a crucial issue in IR evaluation – the disconnect between relevance and user satisfaction. By proposing and demonstrating a user-centric usefulness evaluation framework, the paper offers a potentially more effective way to assess search engine performance. The finding that CLUE-generated labels improve satisfaction prediction significantly highlights the practical value of this approach.  The insights gained from the user study and the derived guidelines for LLM prompts can inform future research in this area.
*   **Strengths:**

    *   **Well-defined Problem:** Clearly articulates the limitations of relevance-based evaluation.
    *   **Novel Approach:** The CLUE framework is a well-engineered solution that integrates several innovative elements.
    *   **Empirical Validation:** Extensive experiments across multiple datasets validate the effectiveness of CLUE.
    *   **Ablation Studies:** Thorough ablation studies provide insights into the contribution of different components.
    *   **User Study:** The collection of a new dataset with user feedback strengthens the study.
*   **Weaknesses:**

    *   **Computational Cost:** Using LLMs, especially through API calls, can be computationally expensive, limiting the scalability of the proposed approach in very large-scale settings. The reliance on GPT-4 in some experiments raises questions about replicability with open-source models. Fine-tuning helps, but might not close the gap fully.
    *   **Sensitivity to Prompt Design:** The performance of LLM-based methods is highly sensitive to prompt design. While the paper offers guidelines, the process of creating effective prompts remains somewhat of an art, potentially hindering widespread adoption. The choice of the DNA template is also not deeply justified.
    *   **Limited Scope of User Behavior:** While the paper incorporates user behavior, it focuses primarily on clickstream data. Other forms of user interaction, such as dwell time on specific sections of a page or explicit feedback beyond usefulness scores, could further enhance the accuracy of the usefulness judgments.
    *   **Evaluation with Silver Annotators:** The validation on real industrial search data involves query-level satisfaction annotated by silver annotators, instead of the ultimate ground truth. This could limit the true reflection of LLM-driven usefulness in user satisfaction.
*   **Potential Influence:** The paper has the potential to influence future research in IR evaluation by promoting the use of LLMs for user-centric usefulness assessment. The CLUE framework and the associated dataset can serve as valuable resources for the community. The focus on ordinal regression and incorporation of user behavior are promising directions for improving the accuracy and effectiveness of LLM-based evaluation methods.

**Rigorous Rationale:**

The paper provides a strong contribution by moving beyond traditional relevance evaluation metrics and harnessing the power of LLMs to approximate user-centric usefulness. The novel aspects of CLUE, including the cascade structure inspired by ordinal regression and the incorporation of user behavior, coupled with the careful user study and rigorous experiments, justify a relatively high score. However, the weaknesses related to computational cost, sensitivity to prompt design, and reliance on clickstream data prevent it from achieving a perfect score. While the insights from this work are valuable, more research is needed to address the scalability and robustness of the proposed approach. The dataset is also only to be released upon paper publication, limiting immediate usage by other research groups.

**Score: 8**

- **Score**: 8/10

### **[RoboSwap: A GAN-driven Video Diffusion Framework For Unsupervised Robot Arm Swapping](http://arxiv.org/abs/2506.08632v1)**
- **Summary**: Here's a summary and critical evaluation of the RoboSwap paper:

**Summary:**

The paper introduces RoboSwap, a novel framework for swapping robotic arms in videos using an unsupervised approach. Unlike previous methods that rely on paired data from identical environments, RoboSwap operates on unpaired data from diverse environments, reducing data collection burdens. It combines GANs and diffusion models in a two-stage pipeline: (1) GANs segment and translate robotic arms, addressing the cross-embodiment gap; and (2) a diffusion model refines the translated arm within the original video background, enhancing coherence, motion realism, and object interaction.  The GAN and diffusion stages are trained independently.  Experiments demonstrate RoboSwap's superior performance compared to state-of-the-art video and image editing models across several benchmarks, particularly in structural coherence and motion consistency, providing a robust tool for generating data for cross-embodiment robotic learning.

**Critical Evaluation:**

*   **Novelty:** The primary novelty lies in the combination of GANs and diffusion models within a two-stage pipeline explicitly designed for *unpaired* cross-embodiment robotic arm swapping.  While GANs and diffusion models are individually well-established, their synergistic integration to tackle this specific problem with unpaired data represents a significant advance. The method also cleverly uses data augmentation to mimic GAN artifacts during training of the diffusion model to bridge the train/test gap.  The idea of using GANs for domain translation (robot arm appearance) and then refining it with a diffusion model (motion coherence, object interaction) is a novel strategy in this context.

*   **Significance:** The significance stems from addressing a crucial bottleneck in video-conditioned robotic learning: the scarcity of diverse, high-quality datasets across different robot platforms and environments. By enabling the generation of new videos with swapped robotic arms, RoboSwap has the potential to augment existing datasets, facilitate cross-platform generalization, and reduce the reliance on costly and time-consuming data collection efforts.  This directly addresses the sim-to-real gap and expands the applicability of robot learning across different embodiments.  The experimental results convincingly demonstrate RoboSwap's superiority over existing methods, suggesting a practical and effective solution.

*   **Strengths:**

    *   **Unsupervised Approach:** Significantly reduces the data collection overhead by not requiring paired demonstrations.
    *   **Effective Integration:** The combination of GANs and diffusion models effectively leverages their strengths to handle domain translation and refinement, respectively.
    *   **Strong Experimental Results:** Outperforms state-of-the-art baselines across multiple benchmarks.
    *   **Clear and Well-Structured Presentation:**  The paper is well-written and clearly explains the methodology, experiments, and results.

*   **Weaknesses:**

    *   **Computational Cost:** While the paper mentions training independently, the combined computational cost of training both GAN and diffusion models might be substantial. This could be a barrier to adoption for researchers with limited resources. Although a LORA strategy is used in the DiT network, the costs were still substantial.
    *   **Dependence on Segmentation:** The pipeline relies on accurate robotic arm segmentation, potentially limiting its robustness in scenarios with complex environments or occlusions.
    *   **Limited Discussion of Failure Cases:** While the paper highlights the successes of RoboSwap, a more detailed discussion of failure cases and limitations would strengthen the analysis.

*   **Potential Impact:** RoboSwap has the potential to significantly impact the field of robot learning by enabling more efficient data generation and facilitating cross-platform generalization. This could lead to more robust and adaptable robotic systems.

**Justification for Score:**

Based on the above analysis, the paper demonstrates a novel and significant contribution to the field of robot learning. The integration of GANs and diffusion models for unsupervised cross-embodiment robot arm swapping is a clever and effective approach. The strong experimental results and clear presentation further strengthen the paper's value. The weaknesses, primarily concerning computational cost and dependence on segmentation, do not overshadow the overall impact. The novel use of distortions during training to mimic GAN artifacts during testing is another notable design choice.

Score: 8

- **Score**: 8/10

### **[PhyBlock: A Progressive Benchmark for Physical Understanding and Planning via 3D Block Assembly](http://arxiv.org/abs/2506.08708v1)**
- **Summary**: Okay, I will provide a summary and a critical evaluation of the "PhyBlock: A Progressive Benchmark for Physical Understanding and Planning via 3D Block Assembly" paper.

**Summary:**

The paper introduces PhyBlock, a new benchmark designed to evaluate vision-language models (VLMs) in their ability to understand and reason about physics, specifically in the context of 3D block assembly tasks. PhyBlock consists of two main components: a hierarchical assembly planning task and a physical-understanding VQA task. The assembly task features a four-level cognitive hierarchy, progressing from basic perception to advanced spatial planning. The VQA task is designed to measure models' explicit understanding of physical concepts like object properties, spatial relationships, and scene dynamics.  The authors benchmarked 21 state-of-the-art VLMs and revealed limitations in high-level planning, spatial reasoning, and physical inference. The paper highlights persistent errors in spatial orientation and dependency reasoning, even with chain-of-thought prompting. The authors position PhyBlock as a unified testbed to bridge vision-language understanding and real-world physical problem-solving.

**Rigorous and Critical Evaluation:**

*   **Novelty:** The novelty of the paper is significant, albeit not revolutionary. Previous works often emphasized passive understanding of physical environments or basic manipulation tasks. PhyBlock introduces a combination of hierarchical planning and explicit physical reasoning VQA, designed to be progressive. The emphasis on 3D block assembly is itself a relatively intuitive and interpretable task setting that highlights various aspects of embodied reasoning. The use of a physics simulator with attention to realistic 3D scenes, and AOV-based evaluation are valuable contributions. However, some prior benchmarks explored similar concepts, even if less progressively, so the key novelty lies in the specific combination and thoroughness of the framework.

*   **Significance:** The paper's significance stems from its ability to identify key limitations in modern VLMs regarding physical understanding and planning. The benchmark exposes areas where VLMs still struggle, such as spatial dependency reasoning and long-horizon planning, even when prompted with chain-of-thought approaches. This helps direct research efforts towards developing models that can better incorporate physical priors. The comprehensive evaluation involving a large number of state-of-the-art models, both closed-source and open-source, makes the benchmark attractive and valuable to the community. Also, identifying that even Chain-of-Thought prompting helps only marginally hints at an even deeper need for true physical understanding and reasoning that VLMs can incorporate.

*   **Strengths:**

    *   **Well-Designed Benchmark:** The hierarchical structure and progressive difficulty levels are a major strength.
    *   **Comprehensive Evaluation:** Benchmarking a diverse set of VLMs reveals valuable insights into their capabilities and limitations.
    *   **Clear Error Analysis:** The categorization of error types (Euler errors, dependency errors, etc.) provides a useful diagnostic tool.
    *   **Physically Grounded:** Using a physics simulator adds realism and allows for evaluating models' ability to reason about dynamic interactions.
    *   **Clear Articulation of Limitations:** The paper acknowledges the limitations of the work and proposes concrete directions for future research (e.g., including VLA models, increasing viewpoint diversity).
*   **Weaknesses:**

    *   **Scope:** While the focus on 3D block assembly provides a structured environment, it may be considered a specialized setting. The generalizability of the findings to more complex, unstructured real-world scenarios could be a concern.
    *   **Evaluation Setting:** The current evaluation still operates largely within a simulation. While physically grounded, the translation of these results to real-world robotic manipulation might not be direct. More direct comparisons to real robot setups would strengthen the paper.
    *   **Limited Action Space:** The actions are relatively high-level and don't include low-level motor controls. Future extensions involving integration with robot control policies will make the benchmark more practical.
    *   **Lack of Solutions/Mitigation Strategies:** The paper provides little insight in how to combat the found limitations. Although acknowledging is the first step, suggestions on what to tackle in the future would enhance the paper.

*   **Potential Influence:** PhyBlock has the potential to become a valuable resource for the embodied AI and robotics community. By providing a standardized benchmark and a set of diagnostic tools, it can help drive progress in developing more physically aware and capable VLMs. The paper's clear articulation of limitations and future directions will likely stimulate further research in this area.

**Justification for Score:**

Considering the factors above, I assign the paper a score of **8**. The PhyBlock benchmark has considerable novelty and significance as a comprehensive and rigorous evaluation framework for physical reasoning and planning in VLMs. While it has some limitations in terms of scope, evaluation setting, and action granularity, the paper makes a valuable contribution by identifying key bottlenecks in current models and proposing concrete directions for future research. The open-source nature of the benchmark and the thorough evaluation of a large number of models increase its potential for impact within the community. The weaknesses, although limiting, are also opportunities for future work.

Score: 8

- **Score**: 8/10

### **[HiSin: Efficient High-Resolution Sinogram Inpainting via Resolution-Guided Progressive Inference](http://arxiv.org/abs/2506.08809v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "HiSin: Efficient High-Resolution Sinogram Inpainting via Resolution-Guided Progressive Inference":

**Summary:**

The paper introduces HiSin, a novel framework for efficient high-resolution sinogram inpainting, a crucial task in computed tomography (CT) reconstruction. HiSin addresses the significant memory and computational challenges posed by diffusion models when applied to high-resolution sinograms. It achieves efficiency through a resolution-guided progressive inference scheme, frequency-aware patch skipping, and structure-adaptive step allocation. The framework progressively extracts global structure at low resolution and then focuses high-resolution inference on small patches.  Frequency-aware patch skipping avoids redundant computations in areas with low signal content, and structure-adaptive step allocation tailors the number of denoising steps to the local signal complexity.  Experimental results demonstrate that HiSin reduces memory usage and inference time while maintaining high inpainting accuracy across various datasets, resolutions, and mask conditions. Critically, the method is presented as inference-time only, preserving compatibility with pre-trained models.

**Critical Evaluation:**

*   **Novelty:** The core novelty lies in the combination of three techniques specifically tailored for the characteristics of sinogram data. 1) the resolution-guided progressive inference allows to reduce peak memory usage and maintains long-range consistency. 2) frequency-aware patch skipping reduce computations in low-information regions based on structural properties unique to projection-domain measurements. 3) Structure-adaptive step allocation allows to adapt denoising effort to match signal complexity. While patch-based processing and multi-resolution techniques are not novel *per se* in image processing, their specific application, combination, and adaptation to the sinogram domain, along with a diffusion model, constitutes a significant contribution. The fact that this is achieved *at inference time* without architectural modification is a strong positive, increasing its practical applicability and adoption potential.
*   **Significance:** High-resolution sinogram inpainting is becoming increasingly important for industrial and medical applications of CT. The excessive memory footprint of diffusion models has been a barrier to their adoption in these contexts. HiSin addresses this barrier directly by enabling high-resolution inpainting on resource-constrained hardware.
*   **Strengths:**

    *   **Memory efficiency:**  The paper clearly demonstrates a significant reduction in peak memory usage, a critical factor for practical deployment.
    *   **Inference time improvement:** The combination of patch skipping and adaptive step allocation leads to noticeable speedups.
    *   **Preserved inpainting accuracy:**  The method maintains comparable or better inpainting quality compared to existing approaches.
    *   **Inference-time only changes:** This design is a huge advantage, meaning that existing models don't need to be retrained. The benefits of memory and time saving can be immediately realized.
    * **Generality:** The method isn't tied to a particular architecture. It should work well with other sinogram inpainting models and diffusion models in general.

*   **Weaknesses:**
    * FLOPs analysis reveals that HiSin needs higher computation than other baseline methods.
    *   **Limited discussion of limitations:** While the paper presents results over several datasets and resolutions, a more in-depth discussion of potential failure cases or edge cases where HiSin might underperform would be valuable. For example, what happens if frequency-aware patch skipping removes too much of the image?

*   **Impact:** HiSin has the potential to lower the barrier to entry for using diffusion models in high-resolution CT imaging, facilitating advancements in material science, medical diagnostics, and other fields. It also introduces new design principles for efficient diffusion inference that may be applicable to other domains beyond sinograms. The demonstrated results of low memory consumption and fast speed provide high practical impact.

**Justification for Score:**

The paper presents a well-engineered solution to a relevant problem, showing strong empirical results. The combination of techniques is tailored to the sinogram data format and the specific characteristics of diffusion models. The modular design, combined with its focus on inference-time applicability, contributes to the paper's novelty and increases the likelihood of its adoption. The only major weakness is an increased FLOP count than other baseline methods. While multi-resolution techniques aren't novel on their own, the complete package and its effective integration into diffusion models merit a high score.

**Score: 8**

- **Score**: 8/10

### **[Dialect Normalization using Large Language Models and Morphological Rules](http://arxiv.org/abs/2506.08907v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces a novel method for dialect normalization that combines rule-based, linguistically informed transformations with large language models (LLMs) using targeted few-shot prompting. The key innovation is that it doesn't require parallel dialectal-standard data for training.  The method is implemented and evaluated for Greek dialects, using a dataset of regional proverbs.  Human evaluation is used to assess the quality of normalization.  The normalized data is then used in downstream tasks, specifically text geocoding, to demonstrate the impact of normalization on subsequent analyses, revealing that earlier research may have been influenced more by orthographic artifacts than underlying semantic information.

**Critical Evaluation:**

*   **Novelty:** The core novelty lies in combining rule-based transformations with LLMs *without* relying on parallel data for fine-tuning. Previous approaches generally required parallel data. The targeted few-shot prompting using dialect-specific examples is also a key component that contributes to the overall effectiveness. The application of this approach to dialect normalization is novel, particularly for a relatively under-resourced scenario like Greek dialects where large parallel corpora are scarce.

*   **Significance:**  The significance is threefold:

    1.  **Practicality:** The approach is highly practical. The lack of requirement for parallel data makes it applicable to a much wider range of dialect normalization problems where creating or obtaining parallel corpora is difficult or impossible. This is a major advantage.

    2.  **Interpretability:** Combining rule-based methods with LLMs offers a degree of interpretability. While LLMs are typically black boxes, the initial rule-based transformations are transparent and can be inspected.

    3.  **Insightful Downstream Analysis:** The use of the normalized data in downstream tasks, specifically geocoding, is crucial. It highlights a significant problem in prior research: the over-reliance on superficial linguistic features rather than genuine semantic content.  This has broad implications for how dialectal data is analyzed.

*   **Strengths:**

    *   Clear problem definition and motivation.
    *   Well-defined methodology, with a good balance of rule-based and LLM-based components.
    *   Rigorous evaluation with human annotators and downstream tasks. The human evaluation includes metrics that guarantee agreement and reliability.
    *   Important insights into the limitations of previous approaches that relied heavily on orthographic features.
    *   Open-source code and data, enhancing reproducibility and encouraging further research.

*   **Weaknesses:**

    *   **Limited Linguistic Scope of RBN:** The rule-based component may only address a limited range of dialectal features.  More complex syntactic or semantic divergence might not be effectively handled by string replacements.
    *   **Generality beyond Greek:** While the approach is sound, the implementation is specific to Greek dialects. Generalizing the rule-based component for other languages would require significant effort.
    *   **Rare Vocabulary:** The paper acknowledges that the method struggles with rare dialectal words that lack clear standard-language cognates and are not well-represented in the LLM's training data. This is a fundamental limitation.
    *   **Evaluator Expertise:** The paper mentions that the evaluators may not have been experts on all of the Greek dialects being considered, which introduces a source of bias.

*   **Potential Influence:** The paper has the potential to influence future research on dialect normalization by providing a practical and effective method that doesn't depend on parallel data. It will likely spur more research into combining rule-based methods with LLMs for low-resource scenarios, and emphasize the importance of careful downstream analysis to avoid over-fitting to superficial features. The release of the normalized Greek proverb dataset is also a valuable contribution to the community.

*The potential broader impacts should also be considered, especially in light of the ethics statement. The paper clearly acknowledges the ethical issues surrounding normalization and notes that the intention is Natural Language Understanding (NLU) rather than Natural Language Generation (NLG).*

*Conclusion*

This is a strong paper that presents a well-executed and insightful approach to dialect normalization. The methodological novelty, strong empirical results, and insightful downstream analysis make it a valuable contribution to the field. While the approach has limitations (especially in handling rare vocabulary and broader generalizability), the benefits outweigh the drawbacks.

Score: 8. The paper represents a significant advance in dialect normalization techniques by reducing the dependence on parallel corpora. This offers high practicality and the potential for substantial impact. The identified weaknesses limit the score from being a 9 or 10, but the overall contribution is very strong.

- **Score**: 8/10

### **[What Limits Virtual Agent Application? OmniBench: A Scalable Multi-Dimensional Benchmark for Essential Virtual Agent Capabilities](http://arxiv.org/abs/2506.08933v1)**
- **Summary**: Here is a concise summary and critical evaluation of the paper "OmniBench: A Scalable Multi-Dimensional Benchmark for Essential Virtual Agent Capabilities":

**Summary:**

The paper introduces OmniBench, a novel benchmark for evaluating virtual agents. Addressing limitations of existing benchmarks like fixed task complexity, reliance on manual annotation, and lack of multi-dimensional evaluation, OmniBench offers a self-generating, cross-platform, graph-based framework. It automates task synthesis with controllable complexity by composing subtasks, using task intents to guide the process. It also introduces OmniEval, a multi-dimensional evaluation framework with subtask-level assessment, graph-based metrics (Coverage Rate and Logical Consistency), and comprehensive tests across ten agent capabilities. The authors generate a dataset of 36k graph-structured tasks across 20 scenarios with high human acceptance rate. The paper also demonstrates that training on their graph-structured data improves agent performance and reports evaluations of various open-source and closed-source models.

**Critical Evaluation:**

*   **Novelty:** The paper presents significant novelty. Automating the generation of diverse task scenarios and structured task graphs for virtual agents is a step beyond existing benchmarks with fixed tasks and substantial manual effort. The explicit incorporation of task intents to guide the task composition process is an important addition, helping to ensure semantic coherence.  The OmniEval framework with its graph-based metrics of Coverage Rate and Logical Consistency offers a more granular and nuanced assessment than existing benchmarks relying solely on success rates or action similarity.

*   **Significance:** The significance lies in several aspects:
    *   **Scalability:** OmniBench's automated pipeline drastically reduces the cost of generating diverse task scenarios, enabling more comprehensive agent evaluation and scalability for the field.
    *   **Controllable Complexity:** The five-dimensional task complexity definition facilitates targeted evaluation, addressing specific agent capabilities with graded difficulty.
    *   **Multi-dimensional Evaluation:** OmniEval provides a deeper understanding of agent performance, revealing capability gaps and providing directions for improvement that a simple success rate cannot.
    *   **Graph-Structured Tasks:** The emphasis on graph-structured tasks is crucial, as it more closely mirrors real-world complexities compared to sequential tasks, revealing limitations of current agents.
    *   **Practical Value:** The paper’s code and dataset release can significantly enhance the field of virtual agents by providing a robust tool for development and comparisons.

*   **Strengths:**
    *   Clear problem statement: The paper convincingly argues the shortcomings of existing benchmarks.
    *   Comprehensive approach: OmniBench addresses multiple limitations with a unified automated pipeline.
    *   Quantitative results: The paper provides extensive evaluation results, demonstrating OmniBench's benefits, comparing different models, and analyzing their performance across various capabilities. The detailed analysis in Section 5 (e.g., parameter tuning results, sensitivity to instruction order) is particularly valuable.
    *   Reproducibility: The inclusion of implementation details in the appendix, as well as the intent to release code and data, bolsters reproducibility.

*   **Weaknesses:**
    *   The metrics, though more comprehensive, can still be susceptible to some biases. There's a dependence on GPT-4 and other LLMs for task summaries and validation.  The reliance on annotators, even if professionally trained, injects some level of subjectivity.
    *   Some may question whether the 10 capabilities are truly a complete picture of what is required for a generalist virtual agent.
    *   Although a cross-platform benchmark is claimed, the case study focuses primarily on Windows. It would benefit from further elaboration and examples across mobile and web environments.

*   **Justification for the Score:**

    The paper represents a substantial advance in the field of virtual agent benchmarking. It convincingly addresses key limitations of existing approaches with a novel, well-engineered, and thoroughly evaluated solution. While relying on LLMs for some automation steps may introduce biases, the design choices largely mitigate those effects. The open release and impact that OmniBench would have on future agent training, evaluation and research cannot be overstated. That being said, its practical cross platform applicability is somewhat unaddressed and could influence its widespread adoption. Therefore, a score less than a 10 seems most appropriate.

Score: 8

- **Score**: 8/10

### **[AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions](http://arxiv.org/abs/2506.09038v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces "AbstentionBench," a large-scale benchmark designed to evaluate the abstention capabilities of Large Language Models (LLMs).  The core argument is that for LLMs to be reliably deployed, they need to know when *not* to answer a question, especially when faced with underspecified, ill-posed, or fundamentally unanswerable queries. The benchmark covers 20 diverse datasets, addressing scenarios like unknown answers, underspecification, false premises, and outdated information.  The authors evaluate 20 frontier LLMs and find that abstention is a significant, unsolved problem. Surprisingly, reasoning fine-tuning often *degrades* abstention performance. While system prompts can offer some boost, they don't fundamentally address the models' inability to reason about uncertainty.

**Critical Evaluation:**

*   **Novelty:** The primary novelty lies in the creation of a comprehensive and diverse benchmark specifically targeting LLM abstention across a wide range of real-world scenarios.  Prior work has looked at uncertainty in LLMs, but often in more isolated contexts like safety or factuality. This paper pulls together a wide array of existing datasets (and introduces some modified ones) into a single, unified framework.  The finding that reasoning fine-tuning *hurts* abstention is also a non-trivial and unexpected result.
*   **Significance:** The paper addresses a critical aspect of LLM reliability.  As LLMs are increasingly deployed in real-world applications (medical, legal, etc.), knowing when a model *shouldn't* answer becomes as important as knowing when it *can* answer correctly.  The benchmark provides a valuable tool for the community to systematically evaluate and improve this capability. The observation regarding the degradation of abstention with reasoning fine-tuning is particularly significant, as it raises questions about current training paradigms and reward structures.
*   **Strengths:**
    *   **Comprehensive Benchmark:** The AbstentionBench is well-motivated and appears to be carefully curated. The diverse datasets cover a broad spectrum of uncertainty scenarios.
    *   **Rigorous Evaluation:** The paper evaluates a significant number of state-of-the-art LLMs, including both open and closed models.
    *   **Surprising Findings:** The counter-intuitive finding that reasoning fine-tuning can degrade abstention performance is a valuable contribution that challenges existing assumptions and highlights potential limitations of current LLM training approaches.
    *   **Automatic Scoring:** The use of an LLM judge for automatic scoring is a good approach to handle the open-ended nature of LLM dialogue and ensures the scalability of the benchmark.
*   **Weaknesses:**
    *   **LLM Judge Dependence:** Reliance on an LLM judge introduces a level of subjectivity and potential bias in evaluation, despite the authors' efforts to validate its accuracy with human annotation. The quality of the benchmark fundamentally depends on the judge, and subtle biases in the judge could skew results.
    *   **Dataset Limitations:** While the benchmark is comprehensive, there's always a risk of dataset bias and coverage gaps. The authors acknowledge the open-endedness of dialogue and the limitations of only focusing on English datasets. Also, the reliance on existing datasets comes with potential limitations of dataset reuse.
    *   **Limited Solutions:** While the paper identifies the problem and provides a useful benchmark, it does not offer substantial solutions to improve LLM abstention. The system prompt experiments are a start, but the paper acknowledges their limitations. The paper mostly sets up the problem and demonstrates its existence, rather than providing potential avenues for fixing it.
*   **Potential Influence:**  AbstentionBench has the potential to become a widely used benchmark in the LLM research community, similar to how other benchmarks (e.g., GLUE, SuperGLUE) have driven progress in other areas.  It could inspire new research directions in LLM training, focusing on how to better represent and reason about uncertainty. The observation that reasoning fine-tuning negatively impacts abstention should prompt a re-evaluation of current training paradigms. However, its long-term impact will depend on whether the community can develop effective methods to improve LLM abstention.

**Overall:**

This is a valuable paper that identifies an important problem in LLM reliability and provides a well-designed benchmark for systematic evaluation. While the paper doesn't offer immediate solutions, it sets the stage for future research and has the potential to influence the development of more robust and trustworthy LLMs. The counter-intuitive findings regarding reasoning are very valuable. The main weakness is the reliance on an LLM judge for scoring.

Score: 8

- **Score**: 8/10

### **[Autoregressive Semantic Visual Reconstruction Helps VLMs Understand Better](http://arxiv.org/abs/2506.09040v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Autoregressive Semantic Visual Reconstruction Helps VLMs Understand Better":

**Summary:**

The paper introduces Autoregressive Semantic Visual Reconstruction (ASVR), a technique designed to improve the visual understanding capabilities of Large Vision-Language Models (LVLMs). The core idea is to incorporate autoregressive supervision on the *semantic* representation of images during training, in addition to the standard autoregressive supervision on textual outputs.  ASVR trains the model to predict the next discrete semantic token of the image's visual representation, guided by a pre-trained semantic visual tokenizer. The authors demonstrate that reconstructing the raw visual appearance of the image does not enhance performance.  Instead, they find that autoregressively reconstructing the semantic representation consistently leads to significant improvements across a range of multimodal understanding benchmarks, different LLM backbones, and varying data scales. They emphasize the importance of high-level semantic visual information in improving the visual understanding capabilities of LVLMs.

**Critical Evaluation:**

*   **Novelty:** The paper's novelty lies primarily in the specific way it applies autoregressive supervision to the visual modality in LVLMs. While autoregressive modeling and visual tokenization are not new concepts, ASVR's emphasis on reconstructing *semantic* visual tokens, rather than raw pixels or appearance features, is a distinguishing factor. The counterintuitive finding that appearance-based reconstruction can even degrade performance adds to the novelty.  The specific architecture and training procedure, incorporating continuous image features as input to reconstruct discrete semantic tokens, is also a non-trivial contribution.
*   **Significance:** The significance of the work is considerable. The results demonstrate that incorporating semantic visual supervision in the autoregressive training of LVLMs can lead to tangible improvements in multimodal understanding.  This addresses a key limitation of current LVLMs, which often underutilize visual information and rely heavily on language. The consistent gains across diverse benchmarks and model architectures suggest that ASVR is a general and effective technique. The improvement on challenging tasks like HallusionBench, a suite for detecting visual hallucination errors, is especially compelling. The approach addresses visual neglect of VLMs where most VLMs are text-centric in the manner that they only supervise the textual outputs. ASVR has the potential to become a new training paradigm for VLMs for better visual understanding. The study opens up a new way of training VLMs through auto-regression on visual features to allow models to first understand visual features and then combine the understanding with textual information.

*   **Strengths:**
    *   **Clear Motivation:** The paper clearly articulates the limitations of current LVLMs and the need for better visual understanding.
    *   **Well-Defined Method:** The ASVR technique is well-defined and relatively simple to implement.
    *   **Comprehensive Experiments:** The experiments are extensive and cover a wide range of benchmarks, data scales, and model architectures.  Ablation studies provide insights into the importance of different design choices.
    *   **Strong Results:** The results consistently demonstrate the effectiveness of ASVR.
    *   **Insightful Analysis:** The paper provides thoughtful analyses of the results, including comparisons between semantic and appearance-based reconstruction, and continuous vs. discrete visual features.
    *   **Well-written and presented:** The paper is written clearly with good figures and illustrations.
*   **Weaknesses:**
    *   **Reliance on Pre-trained Tokenizers:** ASVR relies on a pre-trained semantic visual tokenizer, which can be a bottleneck. The choice of tokenizer and its training data could significantly impact the performance of ASVR. Further investigation on the optimal types and qualities of semantic tokenizers for this application is necessary.
    *   **Limited Exploration of Architectures:** While the paper considers different LLM backbones, the architecture of the visual head itself is not extensively explored.
    *   **Limited Theory:** The paper lacks a rigorous theoretical justification for why semantic visual reconstruction works better than appearance reconstruction. While the intuition is presented, a more formal analysis would strengthen the work.

**Overall Justification:**

ASVR presents a novel and effective technique for improving visual understanding in LVLMs.  The empirical results are compelling, and the paper addresses a critical limitation of current models. The work is well-motivated, clearly presented, and thoroughly evaluated. While there are some weaknesses, particularly concerning the reliance on pre-trained tokenizers and lack of formal theoretical explanation, the contributions are significant enough to warrant a high score. ASVR has the potential to influence future research in multimodal learning and could become a standard technique for training LVLMs.

Score: 8

- **Score**: 8/10

### **[MagCache: Fast Video Generation with Magnitude-Aware Cache](http://arxiv.org/abs/2506.09045v1)**
- **Summary**: Here is a summary and critical evaluation of the paper:

**Summary:**

The paper introduces MagCache, a novel magnitude-aware caching strategy designed to accelerate video diffusion models. The key insight is a unified magnitude law observed across various models and prompts: the magnitude ratio of successive residual outputs decreases monotonically, with a rapid drop in the final steps.  MagCache leverages this insight to adaptively skip unimportant timesteps, using an error modeling mechanism and an adaptive caching strategy. Unlike existing methods that require extensive calibration, MagCache needs only a single sample for calibration.  Experiments on Open-Sora and Wan 2.1 show that MagCache achieves significant speedups (2.1x and 2.68x, respectively) while preserving visual fidelity, outperforming existing caching methods in LPIPS, SSIM, and PSNR.

**Critical Evaluation:**

*   **Novelty:** The core novelty of the paper lies in the discovery of the "magnitude law" and its application to adaptive caching. While caching techniques in diffusion models are not new, the authors present a robust, empirically-supported observation about the behavior of residual outputs, which provides a principled way to determine when timesteps can be skipped. The error modeling and adaptive caching strategies are built on this central insight. The need for only a single sample for calibration is also a significant practical advantage. The magnitude-aware approach is a significant refinement over uniform heuristics used in other caching techniques. This is a valuable insight that could be broadly applied to accelerate diffusion-based inference.

*   **Significance:** The significance of this work is substantial. Inference speed remains a major bottleneck for diffusion models, hindering their widespread adoption. MagCache offers a training-free acceleration strategy that significantly improves inference efficiency without sacrificing visual quality. The results convincingly demonstrate that MagCache outperforms existing methods on standard benchmarks, indicating its potential to improve the practicality of video generation models. The plug-and-play nature of the solution also makes it easier to integrate into existing workflows. The gains reported in terms of speed and quality are quite compelling.

*   **Strengths:**

    *   The magnitude law provides a robust, data-driven justification for timestep skipping, unlike heuristic-based methods.
    *   MagCache is training-free and requires minimal calibration.
    *   The paper provides strong empirical evidence of MagCache's effectiveness on multiple models (Open-Sora and Wan 2.1).
    *   The results demonstrate a significant improvement over existing caching methods in terms of both speed and visual quality.
    *   The ablation studies provide insights into the impact of key parameters on performance.

*   **Weaknesses:**

    *   Although the paper discusses the robustness of the method, it would be valuable to see results on an even wider range of video diffusion models and prompts.
    *   The assumption of preserving the first 20% of diffusion steps unchanged is somewhat ad-hoc and its necessity needs stronger justification.
    *   While experiments show the superiority of the multiplicative formulation in skip error computation, more theoretical justification might be beneficial.
    *   The lack of error bars, justified by resource constraints, makes a definitive assessment of statistical significance challenging.

*   **Potential Influence:** This paper is likely to have a significant influence on the field of diffusion model acceleration. The magnitude law could inspire further research into understanding and exploiting the internal dynamics of diffusion models for efficiency gains. MagCache provides a practical and effective solution for accelerating video generation, which could encourage wider adoption of diffusion models in real-world applications. Other researchers can potentially build upon this work by exploring different error modeling techniques, adaptive caching strategies, or application to other diffusion tasks.

*   **Rigorous Rationale:** The strengths outweigh the weaknesses. The novel discovery of the magnitude law, the practical benefits of MagCache, and strong experimental results make a solid contribution to the field. While minor limitations exist, they do not diminish the overall significance of the work.

Score: 8

- **Score**: 8/10

### **[VIKI-R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning](http://arxiv.org/abs/2506.09049v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces VIKI-Bench, a new hierarchical benchmark for evaluating embodied multi-agent cooperation, and VIKI-R, a two-stage learning framework. VIKI-Bench focuses on three structured levels: agent activation, task planning, and trajectory perception. The benchmark includes diverse robot embodiments, multi-view visual observations, and structured supervision signals. VIKI-R uses a vision-language model (VLM), fine-tuned using Chain-of-Thought demonstrations and reinforcement learning with multi-level reward signals. The paper demonstrates that VIKI-R outperforms baselines on VIKI-Bench, and reinforcement learning leads to the emergence of compositional cooperation patterns.

**Critical Evaluation:**

*   **Novelty:** The introduction of a hierarchical benchmark specifically designed for embodied multi-agent cooperation is a significant contribution. Existing benchmarks are often limited in scope or embodiment diversity. VIKI-Bench fills a gap by providing structured tasks and a multi-dimensional evaluation framework. The idea of structuring the task into three levels (agent activation, task planning, and trajectory perception) provides a useful breakdown of the problem. The structured format is valuable. The VIKI-R framework, while building upon existing VLM fine-tuning and reinforcement learning techniques, tailors these approaches to the specifics of the benchmark. Combining Chain-of-Thought demonstrations with reinforcement learning is not entirely novel, but the specific reward design is adapted well to the benchmark.

*   **Significance:**  The paper has the potential to significantly impact the field of embodied AI and multi-agent systems. It provides a standardized platform for evaluating and comparing different approaches to visual reasoning and cooperation. The benchmark’s structured format could facilitate more targeted research into specific challenges in embodied multi-agent systems. The results presented in the paper demonstrate that the proposed VIKI-R framework is effective and could serve as a baseline for future research. The focus on diverse robot embodiments is also important because many previous approaches consider only a single agent type.

*   **Strengths:**

    *   The VIKI-Bench benchmark offers a comprehensive and structured environment for evaluating embodied multi-agent cooperation.
    *   The VIKI-R framework demonstrates strong performance on the benchmark, indicating the effectiveness of the proposed approach.
    *   The paper includes extensive experimental results and ablation studies, providing valuable insights into the factors that influence performance.
    *   The framework emphasizes visual grounding.

*   **Weaknesses:**

    *   The environment is simulated. While RoboCasa provides a diverse set of tasks and layouts, real-world environments present additional challenges, such as sensor noise and dynamic changes.
    *   While diverse, the robot morphologies are still limited compared to the vast range of real-world robots.
    *   The complexity of the tasks could be increased. While the benchmark has three levels of visual reasoning, it still does not achieve robust performance.
    *   Further exploration into the interpretability of the agents’ reasoning processes would be beneficial.
    *   The ablation studies, while present, are not as thorough as they could be. A more complete study would help confirm the benefits of each component.

*   **Potential Influence:**

    *   VIKI-Bench could become a widely adopted benchmark in the embodied AI community.
    *   The VIKI-R framework could inspire the development of new and improved algorithms for multi-agent cooperation.
    *   The paper’s findings could guide future research on the role of visual reasoning in embodied AI.

*   **Justification for Score:** The paper introduces a valuable new benchmark and demonstrates a promising framework for embodied multi-agent cooperation. It has the potential to significantly advance the field. The strengths of the paper outweigh its weaknesses. It could become a go-to benchmark for multi-agent and visual reasoning in the robotic space.

Score: 8

- **Score**: 8/10

## Other Papers
### **[Repeton: Structured Bug Repair with ReAct-Guided Patch-and-Test Cycles](http://arxiv.org/abs/2506.08173v1)**
### **[LLM-BT: Back-Translation as a Framework for Terminology Standardization and Dynamic Semantic Embedding](http://arxiv.org/abs/2506.08174v1)**
### **[Unable to forget: Proactive lnterference Reveals Working Memory Limits in LLMs Beyond Context Length](http://arxiv.org/abs/2506.08184v1)**
### **[Surgeon Style Fingerprinting and Privacy Risk Quantification via Discrete Diffusion Models in a Vision-Language-Action Framework](http://arxiv.org/abs/2506.08185v1)**
### **[Extracting Information About Publication Venues Using Citation-Informed Transformers](http://arxiv.org/abs/2506.08199v1)**
### **[A Comprehensive Study of Decoder-Only LLMs for Text-to-Image Generation](http://arxiv.org/abs/2506.08210v1)**
### **["I Wrote, I Paused, I Rewrote" Teaching LLMs to Read Between the Lines of Student Writing](http://arxiv.org/abs/2506.08221v1)**
### **[Ensuring Reliability of Curated EHR-Derived Data: The Validation of Accuracy for LLM/ML-Extracted Information and Data (VALID) Framework](http://arxiv.org/abs/2506.08231v1)**
### **[Compound AI Systems Optimization: A Survey of Methods, Challenges, and Future Directions](http://arxiv.org/abs/2506.08234v1)**
### **[Can AI Validate Science? Benchmarking LLMs for Accurate Scientific Claim $\rightarrow$ Evidence Reasoning](http://arxiv.org/abs/2506.08235v1)**
### **[Temporalizing Confidence: Evaluation of Chain-of-Thought Reasoning with Signal Temporal Logic](http://arxiv.org/abs/2506.08243v1)**
### **[Highly Compressed Tokenizer Can Generate Without Training](http://arxiv.org/abs/2506.08257v1)**
### **[Automatic Generation of Inference Making Questions for Reading Comprehension Assessments](http://arxiv.org/abs/2506.08260v1)**
### **[Instruction-Tuned Video-Audio Models Elucidate Functional Specialization in the Brain](http://arxiv.org/abs/2506.08277v1)**
### **[Serendipitous Recommendation with Multimodal LLM](http://arxiv.org/abs/2506.08283v1)**
### **[From Debate to Equilibrium: Belief-Driven Multi-Agent LLM Reasoning via Bayesian Nash Equilibrium](http://arxiv.org/abs/2506.08292v1)**
### **[From Passive to Active Reasoning: Can Large Language Models Ask the Right Questions under Incomplete Information?](http://arxiv.org/abs/2506.08295v1)**
### **[Institutional Books 1.0: A 242B token dataset from Harvard Library's collections, refined for accuracy and usability](http://arxiv.org/abs/2506.08300v1)**
### **[Learnable Spatial-Temporal Positional Encoding for Link Prediction](http://arxiv.org/abs/2506.08309v1)**
### **[Understanding Software Engineering Agents Through the Lens of Traceability: An Empirical Study](http://arxiv.org/abs/2506.08311v1)**
### **[Why Masking Diffusion Works: Condition on the Jump Schedule for Improved Discrete Diffusion](http://arxiv.org/abs/2506.08316v1)**
### **[How Good LLM-Generated Password Policies Are?](http://arxiv.org/abs/2506.08320v1)**
### **[ORFS-agent: Tool-Using Agents for Chip Design Optimization](http://arxiv.org/abs/2506.08332v1)**
### **[A Simple Analysis of Discretization Error in Diffusion Models](http://arxiv.org/abs/2506.08337v1)**
### **[Wait, We Don't Need to "Wait"! Removing Thinking Tokens Improves Reasoning Efficiency](http://arxiv.org/abs/2506.08343v1)**
### **[Pureformer-VC: Non-parallel Voice Conversion with Pure Stylized Transformer Blocks and Triplet Discriminative Training](http://arxiv.org/abs/2506.08348v1)**
### **[Evaluating LLMs Across Multi-Cognitive Levels: From Medical Knowledge Mastery to Scenario-Based Problem Solving](http://arxiv.org/abs/2506.08349v1)**
### **[How Much To Guide: Revisiting Adaptive Guidance in Classifier-Free Guidance Text-to-Vision Diffusion Models](http://arxiv.org/abs/2506.08351v1)**
### **[Reinforcement Fine-Tuning for Reasoning towards Multi-Step Multi-Source Search in Large Language Models](http://arxiv.org/abs/2506.08352v1)**
### **[DEAL: Disentangling Transformer Head Activations for LLM Steering](http://arxiv.org/abs/2506.08359v1)**
### **[CC-RAG: Structured Multi-Hop Reasoning via Theme-Based Causal Graphs](http://arxiv.org/abs/2506.08364v1)**
### **[Mitigating Posterior Salience Attenuation in Long-Context LLMs with Positional Contrastive Decoding](http://arxiv.org/abs/2506.08371v1)**
### **[Draft-based Approximate Inference for LLMs](http://arxiv.org/abs/2506.08373v1)**
### **[EIFBENCH: Extremely Complex Instruction Following Benchmark for Large Language Models](http://arxiv.org/abs/2506.08375v1)**
### **[Reinforce LLM Reasoning through Multi-Agent Reflection](http://arxiv.org/abs/2506.08379v1)**
### **[SafeCoT: Improving VLM Safety with Minimal Reasoning](http://arxiv.org/abs/2506.08399v1)**
### **[mSTEB: Massively Multilingual Evaluation of LLMs on Speech and Text Tasks](http://arxiv.org/abs/2506.08400v1)**
### **[TACTIC: Translation Agents with Cognitive-Theoretic Interactive Collaboration](http://arxiv.org/abs/2506.08403v1)**
### **[Large Language Models Have Intrinsic Meta-Cognition, but Need a Good Lens](http://arxiv.org/abs/2506.08410v1)**
### **[Improved Scaling Laws in Linear Regression via Data Reuse](http://arxiv.org/abs/2506.08415v1)**
### **[Transforming Expert Knowledge into Scalable Ontology via Large Language Models](http://arxiv.org/abs/2506.08422v1)**
### **[Know-MRI: A Knowledge Mechanisms Revealer&Interpreter for Large Language Models](http://arxiv.org/abs/2506.08427v1)**
### **[Better Reasoning with Less Data: Enhancing VLMs Through Unified Modality Scoring](http://arxiv.org/abs/2506.08429v1)**
### **[CAF-I: A Collaborative Multi-Agent Framework for Enhanced Irony Detection with Large Language Models](http://arxiv.org/abs/2506.08430v1)**
### **[Low-resource domain adaptation while minimizing energy and hardware resource consumption](http://arxiv.org/abs/2506.08433v1)**
### **[Olica: Efficient Structured Pruning of Large Language Models without Retraining](http://arxiv.org/abs/2506.08436v1)**
### **[Forward and Backward Simulations for Partially Observable Probability](http://arxiv.org/abs/2506.08437v1)**
### **[SakugaFlow: A Stagewise Illustration Framework Emulating the Human Drawing Process and Providing Interactive Tutoring for Novice Drawing Skills](http://arxiv.org/abs/2506.08443v1)**
### **[A Survey on Large Language Models for Mathematical Reasoning](http://arxiv.org/abs/2506.08446v1)**
### **[A Review on Score-based Generative Models for Audio Applications](http://arxiv.org/abs/2506.08457v1)**
### **[Diffusion Models for Safety Validation of Autonomous Driving Systems](http://arxiv.org/abs/2506.08459v1)**
### **[Hybrid Reasoning for Perception, Explanation, and Autonomous Action in Manufacturing](http://arxiv.org/abs/2506.08462v1)**
### **[MAC: An Efficient Gradient Preconditioning using Mean Activation Approximated Curvature](http://arxiv.org/abs/2506.08464v1)**
### **[AsFT: Anchoring Safety During LLM Fine-Tuning Within Narrow Safety Basin](http://arxiv.org/abs/2506.08473v1)**
### **[Detecting Harmful Memes with Decoupled Understanding and Guided CoT Reasoning](http://arxiv.org/abs/2506.08477v1)**
### **[Re-Thinking the Automatic Evaluation of Image-Text Alignment in Text-to-Image Models](http://arxiv.org/abs/2506.08480v1)**
### **[RHealthTwin: Towards Responsible and Multimodal Digital Twins for Personalized Well-being](http://arxiv.org/abs/2506.08486v1)**
### **[CoQMoE: Co-Designed Quantization and Computation Orchestration for Mixture-of-Experts Vision Transformer on FPGA](http://arxiv.org/abs/2506.08496v1)**
### **[DRAGged into Conflicts: Detecting and Addressing Conflicting Sources in Search-Augmented LLMs](http://arxiv.org/abs/2506.08500v1)**
### **[MLVTG: Mamba-Based Feature Alignment and LLM-Driven Purification for Multi-Modal Video Temporal Grounding](http://arxiv.org/abs/2506.08512v1)**
### **[Teaching Physical Awareness to LLMs through Sounds](http://arxiv.org/abs/2506.08524v1)**
### **[LiftVSR: Lifting Image Diffusion to Video Super-Resolution via Hybrid Temporal Modeling with Only 4$\times$RTX 4090s](http://arxiv.org/abs/2506.08529v1)**
### **[DeepForm: Reasoning Large Language Model for Communication System Formulation](http://arxiv.org/abs/2506.08551v1)**
### **[Efficient Post-Training Refinement of Latent Reasoning in Large Language Models](http://arxiv.org/abs/2506.08552v1)**
### **[The Geometries of Truth Are Orthogonal Across Tasks](http://arxiv.org/abs/2506.08572v1)**
### **[CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmark of Large Language Models in Mental Health Counseling](http://arxiv.org/abs/2506.08584v1)**
### **[Diversity-Guided MLP Reduction for Efficient Large Vision Transformers](http://arxiv.org/abs/2506.08591v1)**
### **[Hateful Person or Hateful Model? Investigating the Role of Personas in Hate Speech Detection by Large Language Models](http://arxiv.org/abs/2506.08593v1)**
### **[Transformers Meet Hyperspectral Imaging: A Comprehensive Study of Models, Challenges and Open Problems](http://arxiv.org/abs/2506.08596v1)**
### **[WGLE:Backdoor-free and Multi-bit Black-box Watermarking for Graph Neural Networks](http://arxiv.org/abs/2506.08602v1)**
### **[Flow Matching Meets PDEs: A Unified Framework for Physics-Constrained Generation](http://arxiv.org/abs/2506.08604v1)**
### **[RE-oriented Model Development with LLM Support and Deduction-based Verification](http://arxiv.org/abs/2506.08606v1)**
### **[Data-Efficient Challenges in Visual Inductive Priors: A Retrospective](http://arxiv.org/abs/2506.08612v1)**
### **[Generalizing while preserving monotonicity in comparison-based preference learning models](http://arxiv.org/abs/2506.08616v1)**
### **[Leveraging LLMs to Evaluate Usefulness of Document](http://arxiv.org/abs/2506.08626v1)**
### **[ECMNet:Lightweight Semantic Segmentation with Efficient CNN-Mamba Network](http://arxiv.org/abs/2506.08629v1)**
### **[RoboSwap: A GAN-driven Video Diffusion Framework For Unsupervised Robot Arm Swapping](http://arxiv.org/abs/2506.08632v1)**
### **[Time Series Representations for Classification Lie Hidden in Pretrained Vision Transformers](http://arxiv.org/abs/2506.08641v1)**
### **[MEMETRON: Metaheuristic Mechanisms for Test-time Response Optimization of Large Language Models](http://arxiv.org/abs/2506.08643v1)**
### **[Summarization for Generative Relation Extraction in the Microbiome Domain](http://arxiv.org/abs/2506.08647v1)**
### **[JoFormer (Journey-based Transformer): Theory and Empirical Analysis on the Tiny Shakespeare Dataset](http://arxiv.org/abs/2506.08652v1)**
### **[Enhancing Reasoning Capabilities of Small Language Models with Blueprints and Prompt Template Search](http://arxiv.org/abs/2506.08669v1)**
### **[MAMBO: High-Resolution Generative Approach for Mammography Images](http://arxiv.org/abs/2506.08677v1)**
### **[Mitigating Reward Over-optimization in Direct Alignment Algorithms with Importance Sampling](http://arxiv.org/abs/2506.08681v1)**
### **[Brevity is the soul of sustainability: Characterizing LLM response lengths](http://arxiv.org/abs/2506.08686v1)**
### **[VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism](http://arxiv.org/abs/2506.08691v1)**
### **[On the Ethics of Using LLMs for Offensive Security](http://arxiv.org/abs/2506.08693v1)**
### **[Educators' Perceptions of Large Language Models as Tutors: Comparing Human and AI Tutors in a Blind Text-only Setting](http://arxiv.org/abs/2506.08702v1)**
### **[PhyBlock: A Progressive Benchmark for Physical Understanding and Planning via 3D Block Assembly](http://arxiv.org/abs/2506.08708v1)**
### **[ConfPO: Exploiting Policy Model Confidence for Critical Token Selection in Large Language Model Preference Optimization](http://arxiv.org/abs/2506.08712v1)**
### **[Explainable Compliance Detection with Multi-Hop Natural Language Inference on Assurance Case Structure](http://arxiv.org/abs/2506.08713v1)**
### **[Improved LLM Agents for Financial Document Question Answering](http://arxiv.org/abs/2506.08726v1)**
### **[Breaking the ICE: Exploring promises and challenges of benchmarks for Inference Carbon & Energy estimation for LLMs](http://arxiv.org/abs/2506.08727v1)**
### **[Unlocking the Potential of Large Language Models in the Nuclear Industry with Synthetic Data](http://arxiv.org/abs/2506.08750v1)**
### **[Enhancing Accuracy and Maintainability in Nuclear Plant Data Retrieval: A Function-Calling LLM Approach Over NL-to-SQL](http://arxiv.org/abs/2506.08757v1)**
### **[AraReasoner: Evaluating Reasoning-Based LLMs for Arabic NLP](http://arxiv.org/abs/2506.08768v1)**
### **[Paths to Causality: Finding Informative Subgraphs Within Knowledge Graphs for Knowledge-Based Causal Discovery](http://arxiv.org/abs/2506.08771v1)**
### **[Flow Diverse and Efficient: Learning Momentum Flow Matching via Stochastic Velocity Field Sampling](http://arxiv.org/abs/2506.08796v1)**
### **[Measuring Data Science Automation: A Survey of Evaluation Tools for AI Assistants and Agents](http://arxiv.org/abs/2506.08800v1)**
### **[HiSin: Efficient High-Resolution Sinogram Inpainting via Resolution-Guided Progressive Inference](http://arxiv.org/abs/2506.08809v1)**
### **[Video-CoT: A Comprehensive Dataset for Spatiotemporal Understanding of Videos Based on Chain-of-Thought](http://arxiv.org/abs/2506.08817v1)**
### **[FreqPolicy: Efficient Flow-based Visuomotor Policy via Frequency Consistency](http://arxiv.org/abs/2506.08822v1)**
### **[The impact of fine tuning in LLaMA on hallucinations for named entity extraction in legal documentation](http://arxiv.org/abs/2506.08827v1)**
### **[Design Patterns for Securing LLM Agents against Prompt Injections](http://arxiv.org/abs/2506.08837v1)**
### **[Filling in the Blanks: Applying Data Imputation in incomplete Water Metering Data](http://arxiv.org/abs/2506.08882v1)**
### **[From Legal Texts to Defeasible Deontic Logic via LLMs: A Study in Automated Semantic Analysis](http://arxiv.org/abs/2506.08899v1)**
### **[Dialect Normalization using Large Language Models and Morphological Rules](http://arxiv.org/abs/2506.08907v1)**
### **[SkipVAR: Accelerating Visual Autoregressive Modeling via Adaptive Frequency-Aware Skipping](http://arxiv.org/abs/2506.08908v1)**
### **[Inherently Faithful Attention Maps for Vision Transformers](http://arxiv.org/abs/2506.08915v1)**
### **[Quantifying Mix Network Privacy Erosion with Generative Models](http://arxiv.org/abs/2506.08918v1)**
### **[PropMEND: Hypernetworks for Knowledge Propagation in LLMs](http://arxiv.org/abs/2506.08920v1)**
### **[Socratic-MCTS: Test-Time Visual Reasoning by Asking the Right Questions](http://arxiv.org/abs/2506.08927v1)**
### **[What Limits Virtual Agent Application? OmniBench: A Scalable Multi-Dimensional Benchmark for Essential Virtual Agent Capabilities](http://arxiv.org/abs/2506.08933v1)**
### **[Can A Gamer Train A Mathematical Reasoning Model?](http://arxiv.org/abs/2506.08935v1)**
### **[FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation](http://arxiv.org/abs/2506.08938v1)**
### **[Can LLMs Ground when they (Don't) Know: A Study on Direct and Loaded Political Questions](http://arxiv.org/abs/2506.08952v1)**
### **[Cross-Spectral Body Recognition with Side Information Embedding: Benchmarks on LLCM and Analyzing Range-Induced Occlusions on IJB-MDF](http://arxiv.org/abs/2506.08953v1)**
### **[GFRIEND: Generative Few-shot Reward Inference through EfficieNt DPO](http://arxiv.org/abs/2506.08965v1)**
### **[ADAM: Autonomous Discovery and Annotation Model using LLMs for Context-Aware Annotations](http://arxiv.org/abs/2506.08968v1)**
### **[Atomic-to-Compositional Generalization for Mobile Agents with A New Benchmark and Scheduling System](http://arxiv.org/abs/2506.08972v1)**
### **[Propositional Logic for Probing Generalization in Neural Networks](http://arxiv.org/abs/2506.08978v1)**
### **[AdaDec: Uncertainty-Guided Adaptive Decoding for LLM-based Code Generation](http://arxiv.org/abs/2506.08980v1)**
### **[SwS: Self-aware Weakness-driven Problem Synthesis in Reinforcement Learning for LLM Reasoning](http://arxiv.org/abs/2506.08989v1)**
### **[Boosting Rust Unit Test Coverage through Hybrid Program Analysis and Large Language Models](http://arxiv.org/abs/2506.09002v1)**
### **[Learning to Reason Across Parallel Samples for LLM Reasoning](http://arxiv.org/abs/2506.09014v1)**
### **[SPEED-RL: Faster Training of Reasoning Models via Online Curriculum Learning](http://arxiv.org/abs/2506.09016v1)**
### **[Diffuse and Disperse: Image Generation with Representation Regularization](http://arxiv.org/abs/2506.09027v1)**
### **[Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning](http://arxiv.org/abs/2506.09033v1)**
### **[FZOO: Fast Zeroth-Order Optimizer for Fine-Tuning Large Language Models towards Adam-Scale Speed](http://arxiv.org/abs/2506.09034v1)**
### **[AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions](http://arxiv.org/abs/2506.09038v1)**
### **[Autoregressive Semantic Visual Reconstruction Helps VLMs Understand Better](http://arxiv.org/abs/2506.09040v1)**
### **[MagCache: Fast Video Generation with Magnitude-Aware Cache](http://arxiv.org/abs/2506.09045v1)**
### **[Agentic Neural Networks: Self-Evolving Multi-Agent Systems via Textual Backpropagation](http://arxiv.org/abs/2506.09046v1)**
### **[Understanding Task Vectors in In-Context Learning: Emergence, Functionality, and Limitations](http://arxiv.org/abs/2506.09048v1)**
### **[VIKI-R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning](http://arxiv.org/abs/2506.09049v1)**
