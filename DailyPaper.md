# The Latest Daily Papers - Date: 2025-09-08
## Highlight Papers
### **[Memorization $\neq$ Understanding: Do Large Language Models Have the Ability of Scenario Cognition?](http://arxiv.org/abs/2509.04866v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper investigates whether large language models (LLMs) possess genuine scenario cognition, defined as the ability to accurately link semantic scenario elements with their corresponding arguments in context. To assess this, the authors introduce a bi-perspective evaluation framework.  They create a novel scenario-based dataset consisting of fictional facts annotated with scenario elements.  LLMs are evaluated through: (1) their ability to answer scenario-related questions (model output perspective), and (2) probing their internal representations for encoded scenario element-argument associations (internal representation perspective). The experiments reveal that current LLMs primarily rely on superficial memorization and struggle to achieve robust semantic scenario cognition, even in simple scenarios.  The authors conclude that this limitation in semantic understanding may contribute to the generation of hallucinations.

**Critical Evaluation:**

*   **Novelty:** The paper introduces a novel evaluation framework specifically designed to probe LLMs' scenario cognition. The creation of a new scenario-based dataset with semantic annotations is a significant contribution. While prior work has explored knowledge memorization and factuality in LLMs, this paper offers a unique perspective by focusing on the *relational understanding* of scenario elements, a crucial aspect of semantic understanding often overlooked. This is more nuanced than the "reversal curse" type of problem considered in previous studies.

*   **Significance:** The findings expose critical limitations in LLMs' semantic understanding and highlight the gap between memorization and genuine cognition.  Demonstrating that LLMs struggle with even relatively simple scenario-based reasoning has significant implications for their use in applications requiring robust understanding and inference.  The suggestion of a link between poor scenario cognition and hallucination is an intriguing hypothesis worthy of further investigation. The bi-perspective evaluation method, combining model output analysis with probing, is also a significant and useful methodological innovation.

*   **Strengths:**
    *   **Clear Definition:** The paper clearly defines scenario cognition and motivates its importance.
    *   **Well-Designed Experiments:** The bi-perspective evaluation framework provides a comprehensive assessment of LLMs' abilities.  The dataset construction process is rigorous and considers diversity and quality.
    *   **Thorough Analysis:**  The paper provides a detailed analysis of the experimental results, including both quantitative metrics and a qualitative case study.
    *   **Addresses limitations in existing research:** The study goes beyond binary entity relations and focuses on a more complex understanding of scenario elements.

*   **Weaknesses:**
    *   **Synthetic Dataset:** The reliance on a synthetic dataset of fictional facts limits the ecological validity of the findings. While fictional data mitigates concerns about directly memorizing real-world facts, it may not fully capture the nuances and complexities of real-world scenarios.
    *   **Dataset Scale:**  While reasonable, the dataset size could be expanded to further strengthen the generalizability of the conclusions.
    *   **Probing Method Complexity:** While the authors experimented with various probing methods, the design of effective probes to fully capture the richness of internal representations remains an open challenge.

*   **Potential Influence:** The paper's findings are likely to stimulate further research in several areas:
    *   Development of new training techniques to improve LLMs' scenario cognition.
    *   Design of more robust evaluation frameworks for assessing semantic understanding.
    *   Investigation of the relationship between scenario cognition and hallucination.
    *   Exploration of alternative architectures or training objectives that explicitly promote relational reasoning.

*   **Score Justification:** The paper makes a clear and well-supported argument about the limitations of LLMs in a crucial area of semantic understanding. The methodology is sound, and the findings have significant implications for the development of more capable and reliable language models. While the reliance on a synthetic dataset and the limited scale are weaknesses, the strengths outweigh these limitations.

Score: 8

- **Score**: 8/10

### **[SparkUI-Parser: Enhancing GUI Perception with Robust Grounding and Parsing](http://arxiv.org/abs/2509.04908v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "SparkUI-Parser: Enhancing GUI Perception with Robust Grounding and Parsing":

**Summary:**

The paper introduces SparkUI-Parser, a novel end-to-end multimodal large language model (MLLM) framework designed to improve GUI (Graphical User Interface) perception. It addresses limitations of existing GUI-focused MLLMs, such as their discrete coordinate modeling that leads to lower grounding accuracy, inability to parse entire interfaces, and failure to reject non-existent elements. SparkUI-Parser employs a "route-then-predict" approach, using a token router to differentiate between text and visual grounding tokens. It features a coordinate decoder that performs continuous modeling of coordinates based on a pre-trained MLLM, a vision adapter to enhance GUI-specific localization, and a rejection mechanism using a modified Hungarian matching algorithm to identify and reject non-existent elements.  The authors also contribute ScreenParse, a new benchmark to evaluate structural GUI perception. The experiments demonstrate that SparkUI-Parser outperforms state-of-the-art methods on ScreenSpot, ScreenSpot-v2, CAGUI-Grounding, and ScreenParse benchmarks.

**Critical Evaluation:**

*   **Novelty:** The paper's novelty lies in several key aspects:
    *   **End-to-End Grounding and Parsing:** SparkUI-Parser is one of the first end-to-end MLLMs which simultaneously tackles GUI grounding and parsing.
    *   **Route-then-Predict Framework:** The "route-then-predict" architecture that separates text and visual grounding tokens. This decoupled approach with a coordinate decoder significantly improves grounding accuracy and inference speed, which is more efficient and accurate than previous MLLMs.
    *   **Rejection Mechanism:** Introduction of the rejection mechanism that handles non-existent elements.
    *   **New Benchmark:** The creation of the ScreenParse benchmark offers a much more fine-grained and challenging evaluation compared to existing datasets by requiring a comprehensive parsing of the entire interface.

*   **Significance:** The paper addresses critical limitations in GUI perception, which is important for advancing autonomous GUI agents. The improvements in grounding accuracy, parsing ability, and the ability to reject non-existent elements directly enhance the robustness and reliability of these agents. The improved inference speed is also beneficial for practical applications that require quick responses. Furthermore, ScreenParse provides a solid foundation for future research and enables more rigorous evaluation of GUI perception models.

*   **Strengths:**

    *   **Performance:** The experiments clearly demonstrate the superiority of SparkUI-Parser over existing methods across different benchmarks. The ablation studies thoroughly analyze the impact of each component in the proposed framework. The inference speed improvements is especially significant.
    *   **Technical Design:** The "route-then-predict" framework is well-designed and addresses the limitations of existing methods effectively. The coordinate decoder and rejection mechanism are novel contributions.
    *   **Evaluation Rigor:** The comprehensive evaluation on multiple benchmarks and the introduction of ScreenParse and related parsing metrics strengthens the paper.

*   **Weaknesses:**

    *   **Model Complexity:** The architecture involves multiple components (MLLM, token router, vision adapter, coordinate decoder, element matcher), which might increase complexity during implementation and training. While the paper provides sufficient details for implementation, a simplified version/analysis could broaden adoption.
    *   **Ablation Study Limitations:** While the ablation study provides valuable insights, there might be other interactions that could be further investigated.
    *   **Dataset Bias:** Although the dataset incorporates different domains, potential biases related to specific app designs or cultural aspects might exist.

*   **Potential Impact:** The paper has a high potential impact on the field of GUI agents, especially in human-computer interaction, robotics, and automation. It advances the development of more robust, accurate, and efficient GUI perception models, which could be crucial for creating intelligent assistants and autonomous systems that can seamlessly interact with various devices and applications.

**Justification of Score:**

SparkUI-Parser makes a valuable contribution to the field by introducing a novel framework that effectively addresses key limitations of existing GUI perception models. The technical design is well-reasoned, the evaluation is rigorous, and the results demonstrate significant performance improvements. The introduction of ScreenParse further enhances the contribution by offering a new and challenging benchmark for future research. The potential impact on the development of autonomous GUI agents is substantial.  While the model is complex and the analysis could be broadened to better understand dataset bias, the overall contribution and demonstrated results warrant a high score.

**Score: 8**
- **Score**: 8/10

### **[LUIVITON: Learned Universal Interoperable VIrtual Try-ON](http://arxiv.org/abs/2509.05030v1)**
- **Summary**: Here's a summary and a critical evaluation of the LUIVITON paper:

**Summary:**

The paper presents LUIVITON, a fully automated virtual try-on system that can drape complex, multi-layered clothing onto diverse and arbitrarily posed 3D humanoid characters. The system overcomes limitations of previous methods by using SMPL as a proxy representation to decouple clothing-to-body draping into two correspondence tasks: clothing-to-SMPL and body-to-SMPL. Clothing-to-SMPL fitting employs a geometric learning-based approach, while body-to-SMPL leverages a diffusion model with multi-view consistent appearance features and DINOv2 priors.  The system also supports fast clothing size customization and is demonstrated on a wide range of humanoid characters, including humans, robots, and stylized figures. The fitting results are shown to be high-quality without requiring human labor, even with 2D clothing.

**Critical Evaluation:**

The paper tackles a significant problem in computer graphics: automating virtual try-on for a wide range of characters and clothing styles. The key strength lies in the decoupling of the correspondence problem using SMPL as a bridge. This approach allows the system to handle complex geometries, non-manifold meshes, and stylized characters that previous methods struggled with.

**Novelty:**

*   **Universal Compatibility:** While previous methods often focus on specific body types (e.g., SMPL bodies) and clothing types (e.g., manifold meshes), LUIVITON demonstrably works on a much wider range of body shapes and garment complexities. This universality is a significant advance.
*   **Body-SMPL Correspondence Using Vision Foundation Models:** Leveraging multi-view consistent diffusion features and DINOv2 priors for body-to-SMPL registration is a novel approach.  This is crucial for generalizing to stylized characters where geometric cues alone are insufficient.
*   **Partial-to-Complete Correspondence Learning:** The clothing-to-SMPL correspondence learning, specifically with the dedicated dataset of 300 garments, is another contribution. The use of DiffusionNet is appropriate for this partial shape completion.
*   **System Integration:** The integration of these components into a functional and fast virtual try-on system with customization capabilities is valuable.

**Significance:**

*   **Practical Impact:** LUIVITON has significant potential for virtual try-on applications in e-commerce, gaming, and AR/VR. The ability to automate the process and handle diverse characters makes it more accessible than manual or semi-automatic methods.
*   **Technical Contribution:** The paper introduces novel techniques for addressing key challenges in correspondence estimation and registration, which can be applied to other problems in computer graphics and vision.
*   **Research Direction:**  The use of diffusion models and vision foundation models for body registration opens up new avenues for research in character modeling and animation.

**Weaknesses:**

*   **Limitations:** The paper acknowledges limitations related to highly non-humanoid shapes and hard materials. While understandable, these limitations should be addressed in future work. The dependence on rest-posed input clothing is also a limitation.
*   **Dataset:** While the creation of a dedicated clothing dataset is a strength, information about its size, variety, and potential biases should be explicitly stated.
*   **Evaluation Metrics and Baselines:** While comparisons are made with other draping methods, it might be beneficial to have a more comprehensive evaluation with a wider range of baselines and more targeted ablation studies to isolate the impact of different design choices.
*   **SMPL dependency:** The SMPL representation, though effective, may still be a bottleneck for handling extreme body shape variations.

**Justification for Score:**

Overall, the paper presents a solid contribution to the field of virtual try-on. It introduces novel techniques, addresses key challenges, and demonstrates promising results on a diverse range of characters and clothing styles. The system is reasonably practical and has the potential for real-world applications.  The limitations, while present, do not significantly detract from the overall contribution. Therefore, I am assigning a score of 8. This reflects the paper's notable novelty, significance, and potential impact while acknowledging some limitations and room for future improvements.

**Score: 8**

- **Score**: 8/10

### **[KVCompose: Efficient Structured KV Cache Compression with Composite Tokens](http://arxiv.org/abs/2509.05165v1)**
- **Summary**: Here is a summary and critical evaluation of the KVCompose paper:

**Summary:**

The paper introduces KVCompose, a new structured key-value (KV) cache compression framework for large language models (LLMs). It addresses the problem of the linearly increasing KV cache size with context length and model depth, which becomes a bottleneck in long-context inference. KVCompose uses attention scores to estimate token importance, independently selects tokens per attention head, and aligns them into composite tokens. A global allocation mechanism adapts retention budgets across layers. The result is a structured eviction strategy compatible with standard inference pipelines, reducing memory usage while preserving accuracy and outperforming existing methods.

**Critical Evaluation:**

*   **Novelty:** The core idea of using composite tokens is a significant contribution.  Instead of simply dropping entire tokens from all attention heads, KVCompose allows each head to retain its most important elements and then aligns these elements, creating "composite" tokens.  This approach is novel in its ability to retain more information per retained composite token compared to methods that simply drop entire original tokens. Adaptive budgeting is also a valuable addition.

*   **Significance:** KV cache compression is a critical problem for deploying LLMs, particularly for long context applications. The authors demonstrate empirical improvements in accuracy vs. compression compared to state-of-the-art structured methods like TOVA and PyramidKV.  The fact that the method is also compatible with existing inference pipelines (e.g., vLLM) without requiring custom CUDA kernels greatly enhances its practical significance. Being able to realize tangible savings and speed-ups directly is a major strength. The method attempts to address the trade-off of memory efficiency, model performance preservation, and computational efficiency. The adaptive budgeting adds another layer of optimization, leading to further gains.

*   **Strengths:**
    *   **Performance:** Strong empirical results that demonstrate improvements over existing structured methods on long-context reasoning tasks.
    *   **Compatibility:** The structured eviction strategy ensures compatibility with existing inference engines.
    *   **Practicality:** The method does not require complex CUDA kernels or custom optimization procedures, which makes it easier to deploy.
    *   **Adaptive Budgeting:** Adaptively allocating the retention budget across layers.

*   **Weaknesses:**
    *   **Computational Overhead:**  While the method aims for computational efficiency, the process of computing attention scores, aggregating, sorting, and creating composite tokens could introduce overhead, though presumably this cost is less than re-computation without any KV caching. The authors should provide details on this cost.
    *   **Limited Evaluation:** While the paper evaluates three LLMs and several tasks, a wider range of LLMs and benchmarks would strengthen the generalizability of the results.
    *   **Task-Specific Sensitivity:**  The choice of task set 'T' can influence the token importance scores, introducing sensitivity to the specific tasks.  The paper demonstrates task-agnostic and task-aware settings, but the potential bias introduced by the task set should be discussed in greater detail.

*   **Potential Influence:** The paper has the potential to influence the field by providing a practical and effective KV cache compression framework.  It encourages further research on more sophisticated token selection and aggregation techniques. Further studies could explore composite tokens for unstructured compression methods to investigate the possibility of better performance and computational trade-offs. Also, the idea of adaptive budget allocation could be further extended to heads within a layer as well as layers.

*   **Overall Assessment:**  KVCompose offers a solid advancement in KV cache compression. It balances effectiveness, ease of implementation, and compatibility. The composite token approach is novel and leads to empirically significant results. The weaknesses, while present, do not significantly detract from the overall contribution.

**Score: 8**

**Justification:** The paper presents a novel and significant improvement in KV cache compression with strong empirical results and practical considerations. While there is room for further evaluation and exploration of potential computational overheads, the approach provides a clear advancement over existing structured methods.

- **Score**: 8/10

### **[Crosscoding Through Time: Tracking Emergence & Consolidation Of Linguistic Representations Throughout LLM Pretraining](http://arxiv.org/abs/2509.05291v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper "Crosscoding Through Time: Tracking Emergence & Consolidation Of Linguistic Representations Throughout LLM Pretraining" introduces a novel method to analyze the evolution of linguistic representations within large language models (LLMs) during pretraining. The approach combines sparse crosscoders, which learn a joint feature space across multiple model checkpoints, with a new metric called Relative Indirect Effect (RELIE).  RELIE is designed to quantify the causal importance of individual features for task performance at different training stages. The authors demonstrate that this technique can effectively identify feature emergence, maintenance, and discontinuation during pretraining in LLMs like Pythia, BLOOM, and OLMo. They find evidence that LLMs gradually build higher-level abstractions, moving from token-specific representations to more universal linguistic concepts and that multilingual LLMs consolidate monolingual features to crosslingual ones through training.

**Critical Evaluation:**

*   **Novelty:** The core novelty lies in the combined use of sparse crosscoders and the RELIE metric to track the causal evolution of linguistic features across LLM pretraining checkpoints. While sparse crosscoders have been used before, applying them specifically to pretraining and developing RELIE to quantify feature importance across time is a valuable extension. The focus on syntactic concepts, such as subject-verb agreement, makes the investigation more targeted than just generic representation analysis.

*   **Significance:** The paper addresses a crucial gap in understanding how LLMs learn.  Traditional evaluation methods offer limited insight into *when* and *how* specific linguistic capabilities develop. The crosscoding/RELIE approach provides a way to peek inside the "black box" of pretraining and track the development of specific concept representations. This is significant because it could potentially lead to better pretraining strategies, more efficient use of resources, and improved control over the learned abilities of LLMs. The results help clarify the mechanisms by which a model internalizes particular linguistic concepts. Also, It offers a valuable framework for dissecting the complex processes underlying LLM training.

*   **Strengths:**
    *   **Clear Methodology:** The paper presents a well-defined methodology, detailing the crosscoder training, the RELIE metric, and the experimental setup.  This makes it easier to replicate the results and build upon the work.
    *   **Scalability & Architecture Agnostic:**  The authors emphasize the architecture-agnostic nature and scalability of their approach, testing it on different LLM families (Pythia, BLOOM, OLMo) with billion-parameter scales.
    *   **Qualitative and Quantitative Results:** The paper integrates both qualitative analyses (interpreting feature annotations and trajectories) and quantitative results (using RELIE to measure importance and identify phase transitions) to provide a more complete picture.
    *   **Well Grounded Claims:** The researchers connect dynamics detected with the proposed techniques to the language modeling abilities of models.
    *   **Careful Validation:** The experiments are controlled and ablated for validity.
    *   **Opens Future Directions:** Finally, the work is able to offer insights into open questions such as: "how do LMs handle agreement in languages with greater morphological complexity?".

*   **Weaknesses:**
    *   **Reliance on Probing Tasks:** The method relies on probing tasks (BLiMP, MultiBLiMP, CLAMS) to measure performance and guide feature selection. The choice of these tasks can influence the results, and the generalizability to other linguistic concepts might depend on the quality of the probing tasks.
    *   **Interpretability Bottleneck:**  While the RELIE metric helps identify important features, the interpretability of those features can still be a bottleneck.  Manual annotation and interpretation are required, which are time-consuming and subjective processes. The reliance on integrated gradients as an approximate causal intervention may also fail to accurately represent feature contributions.
    *   **Computational Cost:**  While the authors mention the computational cost is relatively short, training crosscoders, especially across many checkpoints and parameter scales, can still be resource-intensive. This might limit the ability to apply the approach to much larger models or more frequent checkpoints.
    *   **Checkpoint Selection Influence:** The reliance on checkpoint selection may not lead to the most optimal representations.
    *   **Annotation Process:** Despite manual checking of annotators (authors), annotation inherently includes subjective bias.

*   **Potential Impact:** The research opens several promising directions:

    *   **Informed Pretraining:** Understanding feature evolution can inform pretraining data selection, curriculum learning, and model architecture design.
    *   **Targeted Fine-tuning:** The approach could help identify the optimal checkpoint and features to fine-tune for specific downstream tasks, leading to more efficient learning.
    *   **Fairness and Bias Mitigation:** Analyzing the evolution of biased or unfair representations during pretraining could facilitate early detection and mitigation strategies.
    *   **Cross-Lingual Alignment:** The work offers insights into how to achieve better alignment of language representations.
    *   **Future research on the Evolution of Circuits:** The evolution of circuits and their relationship across time.
    *   **The Relationship between pretraining objective dynamics and learned features:** Avenues exist to link pretraining objective dynamics with learned features.

**Score: 8**

**Justification:**
The paper offers a significant contribution to the field of LLM interpretability by providing a novel and practical method for tracking the causal evolution of linguistic representations during pretraining. The approach is well-defined, scalable, and provides both qualitative and quantitative insights. While there are weaknesses related to the reliance on probing tasks and manual annotation, the potential impact on LLM development and future research directions is substantial.  A score of 8 reflects a strong contribution that builds upon existing techniques and opens new avenues for understanding how LLMs learn. The approach would be made stronger by including further explanation of edge cases and comparisons across the methods used to evaluate model representations, as well as further quantitative evaluation across model size and tasks.
- **Score**: 8/10

## Other Papers
### **[Fishing for Answers: Exploring One-shot vs. Iterative Retrieval Strategies for Retrieval Augmented Generation](http://arxiv.org/abs/2509.04820v1)**
### **[AFD-SLU: Adaptive Feature Distillation for Spoken Language Understanding](http://arxiv.org/abs/2509.04821v1)**
### **[TemporalFlowViz: Parameter-Aware Visual Analytics for Interpreting Scramjet Combustion Evolution](http://arxiv.org/abs/2509.04834v1)**
### **[A Knowledge-Driven Diffusion Policy for End-to-End Autonomous Driving Based on Expert Routing](http://arxiv.org/abs/2509.04853v1)**
### **[Memorization $\neq$ Understanding: Do Large Language Models Have the Ability of Scenario Cognition?](http://arxiv.org/abs/2509.04866v1)**
### **[Using LLMs for Multilingual Clinical Entity Linking to ICD-10](http://arxiv.org/abs/2509.04868v1)**
### **[OSC: Cognitive Orchestration through Dynamic Knowledge Alignment in Multi-Agent LLM Collaboration](http://arxiv.org/abs/2509.04876v1)**
### **[Integrating Large Language Models in Software Engineering Education: A Pilot Study through GitHub Repositories Mining](http://arxiv.org/abs/2509.04877v1)**
### **[L1RA: Dynamic Rank Assignment in LoRA Fine-Tuning](http://arxiv.org/abs/2509.04884v1)**
### **[PLaMo 2 Technical Report](http://arxiv.org/abs/2509.04897v1)**
### **[ACE-RL: Adaptive Constraint-Enhanced Reward for Long-form Generation Reinforcement Learning](http://arxiv.org/abs/2509.04903v1)**
### **[Revolution or Hype? Seeking the Limits of Large Models in Hardware Design](http://arxiv.org/abs/2509.04905v1)**
### **[SparkUI-Parser: Enhancing GUI Perception with Robust Grounding and Parsing](http://arxiv.org/abs/2509.04908v1)**
### **[Towards Ontology-Based Descriptions of Conversations with Qualitatively-Defined Concepts](http://arxiv.org/abs/2509.04926v1)**
### **[Internet 3.0: Architecture for a Web-of-Agents with it's Algorithm for Ranking Agents](http://arxiv.org/abs/2509.04979v1)**
### **[LLM Enabled Multi-Agent System for 6G Networks: Framework and Method of Dual-Loop Edge-Terminal Collaboration](http://arxiv.org/abs/2509.04993v1)**
### **[Do Large Language Models Need Intent? Revisiting Response Generation Strategies for Service Assistant](http://arxiv.org/abs/2509.05006v1)**
### **[LUIVITON: Learned Universal Interoperable VIrtual Try-ON](http://arxiv.org/abs/2509.05030v1)**
### **[Shared Autonomy through LLMs and Reinforcement Learning for Applications to Ship Hull Inspections](http://arxiv.org/abs/2509.05042v1)**
### **[Scale-interaction transformer: a hybrid cnn-transformer model for facial beauty prediction](http://arxiv.org/abs/2509.05078v1)**
### **[GenAI-based test case generation and execution in SDV platform](http://arxiv.org/abs/2509.05112v1)**
### **[KVCompose: Efficient Structured KV Cache Compression with Composite Tokens](http://arxiv.org/abs/2509.05165v1)**
### **[AI Agents for Web Testing: A Case Study in the Wild](http://arxiv.org/abs/2509.05197v1)**
### **[Triadic Fusion of Cognitive, Functional, and Causal Dimensions for Explainable LLMs: The TAXAL Framework](http://arxiv.org/abs/2509.05199v1)**
### **[Symbolic Graphics Programming with Large Language Models](http://arxiv.org/abs/2509.05208v1)**
### **[Hunyuan-MT Technical Report](http://arxiv.org/abs/2509.05209v1)**
### **[HoPE: Hyperbolic Rotary Positional Encoding for Stable Long-Range Dependency Modeling in Large Language Models](http://arxiv.org/abs/2509.05218v1)**
### **[Less is More Tokens: Efficient Math Reasoning via Difficulty-Aware Chain-of-Thought Distillation](http://arxiv.org/abs/2509.05226v1)**
### **[Scaling Performance of Large Language Model Pretraining](http://arxiv.org/abs/2509.05258v1)**
### **[SpikingBrain Technical Report: Spiking Brain-inspired Large Models](http://arxiv.org/abs/2509.05276v1)**
### **[Crosscoding Through Time: Tracking Emergence & Consolidation Of Linguistic Representations Throughout LLM Pretraining](http://arxiv.org/abs/2509.05291v1)**
