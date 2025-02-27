# The Latest Daily Papers - Date: 2025-02-27
## Highlight Papers
### **[KiRAG: Knowledge-Driven Iterative Retriever for Enhancing Retrieval-Augmented Generation](http://arxiv.org/abs/2502.18397v1)**
- **Summary**: Here's a concise summary and critical evaluation of the paper "KiRAG: Knowledge-Driven Iterative Retriever for Enhancing Retrieval-Augmented Generation":

**Summary:**

The paper introduces KiRAG, a novel iterative retrieval-augmented generation (iRAG) model designed to improve performance on multi-hop question answering (QA). KiRAG addresses two key challenges in existing iRAG models: susceptibility to irrelevant/inaccurate information and the inability to dynamically adapt to evolving information needs during multi-step reasoning. KiRAG overcomes these limitations by (1) decomposing documents into knowledge triples for more factually reliable retrieval and (2) employing a knowledge-driven iterative retrieval framework that dynamically identifies and retrieves information bridging existing knowledge gaps, creating a reasoning chain. Experiments on several multi-hop QA datasets demonstrate significant improvements in retrieval and QA performance compared to existing iRAG models.

**Critical Evaluation:**

**Strengths:**

*   **Addresses a well-defined problem:** Multi-hop QA is a challenging task for traditional RAG models, and the paper clearly articulates the limitations of existing iRAG approaches in this domain.
*   **Novel approach:**  The use of knowledge triples for iterative retrieval is a sensible and practical approach to enhance factual accuracy and focus the retrieval process.  The knowledge-driven iterative retrieval, dynamically adapting to evolving information needs, is a significant contribution. Building and refining a chain-of-thought through retrieved knowledge triples ensures context preservation during retrieval.
*   **Strong experimental results:** The paper provides comprehensive experimental results on several benchmarks, consistently demonstrating superior retrieval and QA performance compared to a range of competitive baselines, including other iterative RAG models and enhanced retrieval methods.  The ablation studies effectively isolate the contributions of each component of KiRAG.
*   **Rigorous analysis:** The paper includes detailed analysis of the model's behavior, including examination of retrieval performance at different reasoning steps and case studies illustrating the model's reasoning process. Further ablations quantify and analyze important design aspects such as K (number of retrieved documents) and N (candidate triples) during the chain construction, as well as comparisons among different LLMs as the constructor or reader.
*   **Generalizability:** The experiments demonstrating effective performance across datasets not used for training show good potential for generalization. The implementation is publicly accessible.
*  **Clear and well-organized:** The paper is well-written and easy to understand, clearly explaining the proposed approach and the experimental setup.

**Weaknesses:**

*   **Reliance on pre-computed knowledge triples:**  While pre-computing triples improves efficiency, it introduces an extra step in the pipeline and might not scale well to extremely large corpora. A fully end-to-end approach that dynamically extracts triples could be explored. However, its efficiency will come into question.
*   **Silver data creation:** The Reasoning Chain Aligner requires training. However, the data is only weakly labeled, so it may suffer in the ability to generalize. However, the experiments still present improvements, indicating good performance.
*   **Evaluation metric limitations:** While the paper uses standard QA metrics (EM and F1), these metrics don't fully capture the nuances of multi-hop reasoning. A more fine-grained analysis of the reasoning process, perhaps through human evaluation, could provide further insights. It would be worthwhile to analyze the failure cases, and how they can be improved.
*   **Marginal Novelty:** The idea of using Knowledge Graphs/Triples to improve QA is not entirely new. However, the iterative construction of a reasoning chain from these triples during retrieval is novel, as it effectively balances knowledge grounding and iterative reasoning, showing high impact within the LLM and QA community.

**Significance:**

KiRAG represents a significant advancement in iRAG models for multi-hop QA. By leveraging knowledge triples and dynamically adapting the retrieval process, it achieves superior performance and addresses key limitations of existing approaches. The paper is likely to have a strong influence on future research in this area. The use of document-grounded knowledge to improve the reasoning ability of LLMs during retrieval is a promising avenue for further research.

**Justification for Score:**

Considering the strengths and weaknesses, I give this paper a **Score: 8**.

The paper presents a well-executed solution to a clearly defined problem within the RAG space. The technical approach is novel and the experimental results are convincing. While the reliance on pre-computed triples and the reliance on existing LLMs as a constructor represents a potential limitation, the overall contribution is significant enough to warrant a high score. It represents a considerable refinement and improvement over existing iterative retrieval methods, and is likely to guide future work on incorporating knowledge graphs/triples within LLM applications.

- **Score**: 8/10

### **[Deep-Bench: Deep Learning Benchmark Dataset for Code Generation](http://arxiv.org/abs/2502.18726v1)**
- **Summary**: Here's a concise summary and a critical evaluation of the "Deep-Bench: Deep Learning Benchmark Dataset for Code Generation" paper:

**Summary:**

The paper introduces Deep-Bench, a new benchmark dataset for evaluating the performance of Large Language Models (LLMs) in generating Deep Learning (DL) code. Deep-Bench is designed to address the limitations of existing benchmarks like DS-1000, which primarily focus on small code snippets for pre/post-processing tasks. Deep-Bench provides a more comprehensive dataset covering the entire DL pipeline, including pre-processing, model construction, training, inference, and evaluation. It also categorizes DL problems based on phases, ML tasks (classification, regression, etc.), and input data types (tabular, image, text). The paper presents an initial evaluation of several state-of-the-art LLMs on Deep-Bench, revealing their challenges in generating accurate and executable DL code compared to more general code generation tasks. The authors also conduct a qualitative analysis of the types of bugs found in LLM-generated DL code.

**Critical Evaluation:**

**Novelty:** The paper's primary novelty lies in the creation of the Deep-Bench dataset itself. While other code generation benchmarks exist, Deep-Bench is specifically tailored to DL code generation and provides a more holistic view of the DL development lifecycle. The categorization of DL problems based on phases, tasks, and input data types is also a valuable contribution, allowing for a more granular analysis of LLM performance.

**Significance:** The significance of Deep-Bench stems from its potential to drive research and development in DL code generation. By providing a more challenging and comprehensive benchmark, it can help researchers identify the strengths and weaknesses of existing LLMs and develop new techniques for improving their ability to generate accurate and reliable DL code. The bug taxonomy also offers insights into the specific challenges that LLMs face in generating DL code compared to general-purpose code.

**Strengths:**

*   **Comprehensive coverage:** Deep-Bench covers a wide range of DL phases, tasks, and input data types, making it a more representative benchmark than existing datasets.
*   **Categorization:** The categorization of DL problems allows for a more detailed analysis of LLM performance and identification of specific areas for improvement.
*   **Bug taxonomy:** The bug taxonomy provides insights into the types of errors that LLMs make when generating DL code.
*   **Evaluation of SOTA LLMs:** The initial evaluation of several state-of-the-art LLMs on Deep-Bench provides a baseline for future research.
*   **Availability of data and resources:** The dataset and docker images available from this paper greatly increases the reproducibility and wide spread adoption of this work.

**Weaknesses:**

*   **Dataset size:** While Deep-Bench is more comprehensive than DS-1000, its size (520 instances) might be considered relatively small compared to some other code generation benchmarks. This could limit the statistical power of certain analyses.
*   **Data Selection & Bias:** The selection of code snippets solely from starred GitHub repos may introduce a bias towards specific libraries, coding styles, or problem types.
*   **Evaluation Metric:** While pass@k is a common metric, it primarily indicates executability, it may not capture code quality in its entirety (readability, efficiency).
*   **Prompt Engineering:** The intentional avoidance of advanced prompting may limit what is being measured to the general knowledge of the LLMs, without giving an indication of best possible performance for specific tasks.

**Influence:**

Deep-Bench has the potential to become a widely used benchmark for DL code generation, similar to how DS-1000 is used for general data science code generation. It can help accelerate research in this area by providing a standardized platform for evaluating and comparing different approaches. The bug taxonomy can also inform the development of new techniques for improving the accuracy and reliability of LLM-generated DL code. Over time, it could also incentivize the creation of better training datasets for LLMs focused on DL.

**Justification for Score:**

Deep-Bench is a valuable contribution to the field of DL code generation. The comprehensive coverage of the DL pipeline, the categorization of DL problems, and the bug taxonomy are all significant strengths. It will provide guidance on what prompts and general approaches are more accurate, and will help improve LLM code generation models for years to come. While the dataset size and GitHub repo data source might have certain limitations and biases, the benefits of Deep-Bench outweigh these drawbacks. It helps the DL LLM code generation community a much needed benchmark, moving the community forward, with reproducible data points. Overall, Deep-Bench represents a substantial step forward and is well worth the effort by the authors.

Score: 8

- **Score**: 8/10

### **[Learning to Generate Structured Output with Schema Reinforcement Learning](http://arxiv.org/abs/2502.18878v1)**
- **Summary**: Here's a concise summary and a critical evaluation of the provided paper:

**Summary:**

The paper introduces SchemaBench, a benchmark for evaluating large language models' (LLMs) ability to generate valid JSON outputs according to given schemas.  It identifies challenges in structure understanding, escaping, and following natural language descriptions within schemas. The paper proposes Schema Reinforcement Learning (SRL), a training method incorporating a fine-grained schema validator and "Thoughts of Structure" (ToS) to improve JSON generation. The authors demonstrate that SRL enhances model performance on both JSON generation and downstream tasks (BFCL), exceeding supervised fine-tuning baselines.

**Critical Evaluation:**

The paper addresses a relevant and important problem: enabling LLMs to reliably generate structured outputs, particularly JSON. While JSON is widely used, existing methods and benchmarks have limitations in dealing with complex schemas and nuanced constraints.

**Novelty:**

*   **Benchmark (SchemaBench):** The creation of a dataset of approximately 40,000 JSON schemas is a significant contribution. The paper presents a detailed description of the data collection and cleaning pipeline. This benchmark has the potential to serve as a valuable resource for the community, enabling more rigorous evaluation of LLMs' structured output generation capabilities.
*   **Schema Reinforcement Learning (SRL):** The use of reinforcement learning with a fine-grained schema validator is a novel approach. The idea of incorporating RL to address the sensitivity of JSON formatting and optimize for nuanced correctness is well-reasoned.  The combination of SRL with Thoughts of Structure (ToS) is a further innovation.
*   **Thoughts of Structure (ToS):** While inspired by Chain-of-Thought prompting, ToS adapts the reasoning process specifically to the generation of JSON structures. Training the model to explicitly outline the reasoning steps before producing JSON could improve its ability to handle complex schemas and dependencies.

**Significance:**

*   **Improved JSON Generation:** The experimental results demonstrate significant improvements in JSON generation accuracy using SRL, particularly when compared to supervised fine-tuning. The gains on complex schemas are especially noteworthy.
*   **Downstream Task Performance:** The positive results on BFCL indicate that the improvements in JSON generation translate to tangible benefits in downstream applications. This is a crucial finding, showing that the research is not simply optimizing for a specific metric but rather improving the usability of LLMs in real-world scenarios.
*   **Practical Implications:** Reliable JSON generation is essential for integrating LLMs with existing systems and APIs. By improving this capability, the paper makes LLMs more versatile and useful in a wider range of applications.

**Weaknesses:**

*   **Limited Analysis of ToS:** While ToS is presented as a key component, the paper could benefit from a more in-depth analysis of its effectiveness. Are there specific types of schemas or tasks where ToS is most beneficial?
*   **Benchmark Diversity:** While the 40K schema dataset is substantial, further information on the distribution of schema types and complexities could be provided. Are certain types of schemas over- or under-represented?
*   **Generalization:** While the paper shows improvements on BFCL, additional experiments on other downstream tasks would strengthen the claims about generalization.

**Justification:**

The paper is well-motivated, presents a clearly defined problem, and proposes a novel solution. The benchmark, SRL framework, and ToS component represent valuable contributions to the field. While some areas could benefit from further analysis, the paper demonstrates a significant advancement in enabling LLMs to generate valid and useful JSON outputs. The significant performance gains and the potential impact on downstream tasks justify a relatively high score.

**Score: 8**

- **Score**: 8/10

### **[GenTool: Enhancing Tool Generalization in Language Models through Zero-to-One and Weak-to-Strong Simulation](http://arxiv.org/abs/2502.18990v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "GenTool: Enhancing Tool Generalization in Language Models through Zero-to-One and Weak-to-Strong Simulation":

**Summary:**

The paper introduces GenTool, a novel training framework designed to improve the tool utilization capabilities of Large Language Models (LLMs), specifically focusing on generalization. GenTool tackles two key generalization challenges: zero-to-one generalization (adopting a tool when none were initially available) and weak-to-strong generalization (transitioning from a less effective to a more capable tool). The approach involves creating synthetic training data simulating these scenarios and using a two-stage fine-tuning process that first optimizes tool ranking and then refines tool selection. Experiments across various models (1B to 8B parameters) demonstrate significant improvements in tool-usage accuracy, surpassing even GPT-4o performance in certain aspects.  The paper also includes ablation studies and empirical analyses exploring factors that influence tool generalization.

**Critical Evaluation:**

**Novelty:**  The paper demonstrates several novel aspects:

*   **Structured Generalization Dimensions:**  The explicit framing of tool generalization as encompassing "zero-to-one" and "weak-to-strong" challenges provides a clear conceptual framework. This is more structured than simply aiming for general "tool usage improvement."
*   **Synthetic Data Generation Pipeline:**  The development of a high-quality synthetic dataset specifically designed to address these generalization dimensions is a significant contribution. The process of generating "weak tools" and creating queries that differentiate their capabilities from "strong tools" is well-motivated and executed.  The design of the data generation process using a powerful LLM (GPT-4o) combined with human verification adds to the quality of the dataset.
*   **Two-Stage Fine-Tuning:**  The fine-tuning strategy prioritizing tool ranking before selection is a worthwhile innovation. It allows the model to learn functional relationships and comparisons between tools rather than simply mapping queries to tools.

**Significance:**

*   **Addressing a Real-World Problem:** The paper directly addresses a critical limitation of LLMs in real-world applications, where the availability and capabilities of tools can change dynamically. Generalization to unseen tools and usage patterns is crucial for robust performance.
*   **Empirical Validation:** The comprehensive experiments across multiple models and generalization scenarios provide strong evidence of GenTool's effectiveness.  The comparisons with strong baselines, including GPT-4o, are compelling.
*   **In-depth Analysis:** The ablation studies and empirical analyses offer valuable insights into the factors affecting tool generalization. The investigation of the impact of related examples quantity and the contributions of different pair types (Zero-to-One vs Weak-to-Strong) deepen our understanding of these challenges.
*   **Limitations Acknowledged:** The paper acknowledges the limitations regarding model scale and the focus on single-query, single-tool scenarios. Acknowledging such limitations enhances the credibility of the research.

**Weaknesses:**

*   **Dependence on Synthetic Data:** While the synthetic data generation is a strength, it also represents a potential weakness. The effectiveness of GenTool is tied to the quality and representativeness of the synthetic data. Potential biases introduced by GPT-4o, despite the human verification stage, should be considered. It's not guaranteed this framework will transfer well to datasets outside of a simulated setting, and it would have been stronger if experiments were run on more diverse sources of data.
*   **Limited Model Scale Explored:** While experiments cover a range of models, exploring larger models (beyond 8B parameters) would further strengthen the results, especially given the data generation process relies on GPT-4o.

**Potential Influence:**

GenTool has the potential to significantly influence the development of more robust and adaptable LLM-based AI assistants. The framework and insights provided by this research can guide future efforts in tool learning and generalization, ultimately leading to more reliable and versatile AI systems.

The structured approach to both training and evaluation, along with the release of a high-quality synthetic dataset (presumably), makes this work reproducible and valuable to the research community.

Considering the novelty and impact, while acknowledging the limitations, I believe the paper deserves a strong score.

Score: 8
Rationale:
This paper has strong potential to meaningfully advance tool usage with LLMs through its targeted approach and data generation techniques. However, its dependence on high-quality synthetic data, and unproven transferability prevent it from achieving a higher score. The findings are particularly compelling due to clear improvements over SOTA models such as GPT-4o, and the paper provides valuable practical insight to improve tool generalization capabilities.

- **Score**: 8/10

### **[BIG-Bench Extra Hard](http://arxiv.org/abs/2502.19187v1)**
- **Summary**: Here's a summary and critical evaluation of the provided paper:

**Summary:**

The paper introduces BIG-Bench Extra Hard (BBEH), a new benchmark designed to evaluate the reasoning capabilities of Large Language Models (LLMs). BBEH addresses the saturation of existing benchmarks, particularly BIG-Bench Hard (BBH), where state-of-the-art models achieve near-perfect scores. BBEH replaces each task in BBH with a novel, more difficult counterpart probing similar reasoning skills but requiring more complex reasoning abilities such as multi-hop reasoning, learning on the fly, error detection in reasoning traces, long-context processing, and going against strong priors. The authors evaluate several LLMs on BBEH and find that even the best models achieve significantly lower accuracy than on BBH, demonstrating the increased difficulty. The paper also includes a failure analysis, highlighting areas where LLMs struggle. The BBEH dataset is publicly released.

**Critical Evaluation:**

The paper makes a valuable contribution to the field of LLM evaluation. The saturation of existing benchmarks is a well-recognized problem, hindering progress and making it difficult to meaningfully compare models. BBEH effectively addresses this by creating a more challenging benchmark that pushes the boundaries of LLM reasoning.

**Strengths:**

*   **Addresses a Critical Need:** The paper directly tackles the problem of benchmark saturation, which is crucial for continued progress in LLM research.
*   **Well-Designed Benchmark:** BBEH is carefully constructed, maintaining the diversity of reasoning skills present in BBH while significantly increasing the difficulty. The specific reasoning skills targeted (multi-hop reasoning, long-context, etc.) are relevant and important for real-world applications.
*   **Semi-Adversarial Approach:** The use of a semi-adversarial approach to create the benchmark by iterating on difficulty while evaluating against strong reference models is a sound methodology.
*   **Comprehensive Evaluation:** The paper provides a thorough evaluation of several state-of-the-art LLMs on BBEH, demonstrating its difficulty.
*   **Failure Analysis:** The inclusion of a failure analysis provides valuable insights into the limitations of current LLMs and guides future research directions.
*   **Publicly Released Dataset:** The public release of BBEH makes it accessible to the research community, fostering further investigation and development.

**Weaknesses:**

*   **Reference Model Bias:**  The semi-adversarial approach is a double-edged sword. While effective in creating a difficult benchmark, it introduces a potential bias toward the failure modes of the reference models used during construction.  Other types of reasoning biases that may be common to other model types would be missed. The authors acknowledge this limitation, but it is important to keep in mind when interpreting the results.
*   **Limited Ground Truth Verification?** The paper mentions verification of correctness but doesn't specify the degree to which humans were used to independently verify ground truth answers to the new challenging tasks. It relies somewhat on automated metrics. This is not a major weakness, but stronger human validation would bolster the dataset's reliability.
*   **Proxy Metrics for "Thinking":** While the use of output length as a proxy for the amount of "thinking" required is a reasonable heuristic, it is an indirect measure and might not always be accurate. More direct measures of reasoning complexity (e.g., step count in a formal logic system) would be valuable in future work.

**Novelty and Significance:**

The creation of BBEH is a novel and significant contribution. While other benchmarks exist, BBEH's specific focus on pushing the limits of general reasoning while maintaining diversity and avoiding simple mathematical or coding tasks sets it apart. The paper directly enables researchers to better understand the reasoning limitations of LLMs beyond the abilities captured in current popular benchmarks. The impact of this paper is highly likely to increase as more advanced reasoning benchmarks are released following this paper. The analysis and identified failure modes are also useful contributions.

**Justification for Score:**

Considering the strengths and weaknesses, I assign a score of **8**. BBEH is a significant advance in LLM evaluation, addressing a critical need and providing a valuable resource for the research community. The limitations regarding reference model bias and proxy metrics, while present, do not significantly diminish the overall impact of the paper. BBEH has the potential to shape the future direction of LLM research by driving the development of more robust and general reasoning capabilities.

Score: 8

- **Score**: 8/10

### **[Two Heads Are Better Than One: Dual-Model Verbal Reflection at Inference-Time](http://arxiv.org/abs/2502.19230v1)**
- **Summary**: Okay, here's a concise summary and critical evaluation of the provided paper, focusing on novelty, significance, and a justified score:

**Summary:**

The paper introduces a dual-model framework called DARS (Dual-model Reflective Scoring) to improve the reasoning and explainability of Large Language Models (LLMs) in Automated Student Answer Scoring (ASAS). It addresses two key challenges: 1) the lack of explicit feedback in preference optimization methods and 2) the limitations of single-model verbal reflection techniques. DARS uses a contrastive reflection synthesis pipeline to generate precise verbal reflection data and then trains specialized Reasoner and Critic models. The Reasoner model refines assessments based on the Critic's feedback, which integrates process and outcome reward modeling. The authors demonstrate that DARS outperforms traditional preference optimization methods and offers superior reasoning performance and transparency compared to single-model approaches. They show that a larger Critic is often more beneficial than a larger Reasoner and show generalization to unseen questions.

**Critical Evaluation:**

*   **Novelty:**

    *   The *contrastive reflection synthesis pipeline* to generate training data for error correction is a substantial contribution. It allows for the creation of detailed, trace-level reflections, which are often lacking in current approaches. The use of structured thought trees to identify mismatches in key element assessments leading to potential errors is innovative.
    *   The *dual-model architecture (Reasoner and Critic)* with specialized roles and independent training is a novel way to approach the self-reflection problem in LLMs. This is distinct from existing VRL techniques that rely on iterative refinement within a single model.
    *   The *integration of process and outcome reward modeling* in the Critic without relying on human labels (or "oracle" labels, which may not be accurate) for verification is valuable.
    *   The finding that scaling the *Critic model size has more influence on the performance than scaling the Reasoner model size* is a counter-intuitive, valuable empirical insight that can change the future direction.

*   **Significance:**

    *   The paper addresses a crucial problem in LLM research: improving reasoning and explainability in complex tasks. Automated Student Answer Scoring is a pertinent application area due to the stringent requirements for accurate and explainable assessments.
    *   The performance improvements over existing methods, along with the detailed empirical analysis (including ablation studies), demonstrate the effectiveness of the DARS framework.
    *   The framework makes VRL useful in scenarios where the models cannot be directly fine-tuned, a common situation for LLMs.
    *   Human evaluation confirmed that Critic-generated reflection provided actionable guidance that could be reliability followed by the reasoner.

*   **Strengths:**

    *   Well-defined problem and clear research questions.
    *   Methodologically sound, with a detailed description of the DARS framework and the contrastive reflection synthesis pipeline.
    *   Comprehensive experiments with multiple datasets, evaluation metrics, and ablation studies.
    *   The code and data will be released (at least, according to the paper) ensuring reproducibility.
    *   Insightful empirical findings, particularly the scaling law observation.
    *   The authors are rigorous in testing a variety of model scales and architectures.
    *   The study demonstrates the capability of the Critic to still provide useful feedback on novel or unseed data.

*   **Weaknesses:**

    *   The reliance on synthetic data generation could be a limitation. While the paper demonstrates the effectiveness of the approach, the quality of the synthetic data directly impacts the performance of the trained models. The authors do demonstrate generalizability of the synthetic data and the overall model to new datasets.
    *   The study primarily focuses on ASAS. While the principles may be applicable to other reasoning tasks, further research is needed to demonstrate broad generalizability.
    *   The increased training FLOPs involved in training both a Reasoner and a Critic, are a potential barrier to wider adoption. While future work should also consider better optimization of training approaches.

*   **Potential Influence:**

    *   The DARS framework could inspire new approaches to self-reflection in LLMs, particularly by focusing on specialized model architectures and targeted feedback mechanisms.
    *   The contrastive reflection synthesis pipeline provides a valuable technique for generating training data for error correction.
    *   The findings on scaling laws could guide future research on model architectures and training strategies for VRL.
    *   The code and data availability will enable other researchers to build upon and extend this work.

**Justification:**

The paper offers a novel and well-executed solution to improve LLM reasoning and explainability within the ASAS task. The contrastive reflection synthesis pipeline and the dual-model architecture are innovative contributions. The thorough experimental analysis and the insightful findings further enhance the paper's significance. There are some limitations regarding synthetic data reliance and generalizability, these are appropriately acknowledged by the authors and do not detract significantly from the overall contribution. The framework has the potential to influence future research on self-reflection and reasoning in LLMs.

Score: 8

- **Score**: 8/10

## Other Papers
### **[How Far are LLMs from Real Search? A Comprehensive Study on Efficiency, Completeness, and Inherent Capabilities](http://arxiv.org/abs/2502.18387v2)**
### **[Monte Carlo Temperature: a robust sampling strategy for LLM's uncertainty quantification methods](http://arxiv.org/abs/2502.18389v1)**
### **[KiRAG: Knowledge-Driven Iterative Retriever for Enhancing Retrieval-Augmented Generation](http://arxiv.org/abs/2502.18397v1)**
### **[OmniAlign-V: Towards Enhanced Alignment of MLLMs with Human Preference](http://arxiv.org/abs/2502.18411v1)**
### **[TextGames: Learning to Self-Play Text-Based Puzzle Games via Language Model Reasoning](http://arxiv.org/abs/2502.18431v1)**
### **[ToMCAT: Theory-of-Mind for Cooperative Agents in Teams via Multiagent Diffusion Policies](http://arxiv.org/abs/2502.18438v1)**
### **[MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning](http://arxiv.org/abs/2502.18439v1)**
### **[SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution](http://arxiv.org/abs/2502.18449v1)**
### **[FRIDA to the Rescue! Analyzing Synthetic Data Effectiveness in Object-Based Common Sense Reasoning for Disaster Response](http://arxiv.org/abs/2502.18452v1)**
### **[LLM-Based Design Pattern Detection](http://arxiv.org/abs/2502.18458v1)**
### **[FactReasoner: A Probabilistic Approach to Long-Form Factuality Assessment for Large Language Models](http://arxiv.org/abs/2502.18573v1)**
### **[Scalable Best-of-N Selection for Large Language Models via Self-Certainty](http://arxiv.org/abs/2502.18581v1)**
### **[Chain of Draft: Thinking Faster by Writing Less](http://arxiv.org/abs/2502.18600v1)**
### **[Toward Breaking Watermarks in Distortion-free Large Language Models](http://arxiv.org/abs/2502.18608v1)**
### **[Diffusion Models for conditional MRI generation](http://arxiv.org/abs/2502.18620v1)**
### **[PacQ: A SIMT Microarchitecture for Efficient Dataflow in Hyper-asymmetric GEMMs](http://arxiv.org/abs/2502.18627v1)**
### **[Steered Generation via Gradient Descent on Sparse Features](http://arxiv.org/abs/2502.18644v1)**
### **[Single- vs. Dual-Prompt Dialogue Generation with LLMs for Job Interviews in Human Resources](http://arxiv.org/abs/2502.18650v1)**
### **[Independent Mobility GPT (IDM-GPT): A Self-Supervised Multi-Agent Large Language Model Framework for Customized Traffic Mobility Analysis Using Machine Learning Models](http://arxiv.org/abs/2502.18652v1)**
### **[Discriminative Finetuning of Generative Large Language Models without Reward Models and Preference Data](http://arxiv.org/abs/2502.18679v1)**
### **[Comparing Native and Non-native English Speakers' Behaviors in Collaborative Writing through Visual Analytics](http://arxiv.org/abs/2502.18681v1)**
### **[Adaptive conditional latent diffusion maps beam loss to 2D phase space projections](http://arxiv.org/abs/2502.18684v1)**
### **[Policy-as-Prompt: Rethinking Content Moderation in the Age of Large Language Models](http://arxiv.org/abs/2502.18695v1)**
### **[MPO: An Efficient Post-Processing Framework for Mixing Diverse Preference Alignment](http://arxiv.org/abs/2502.18699v1)**
### **[A Cooperative Multi-Agent Framework for Zero-Shot Named Entity Recognition](http://arxiv.org/abs/2502.18702v1)**
### **[TrajLLM: A Modular LLM-Enhanced Agent-Based Framework for Realistic Human Trajectory Simulation](http://arxiv.org/abs/2502.18712v1)**
### **[Talking to the brain: Using Large Language Models as Proxies to Model Brain Semantic Representation](http://arxiv.org/abs/2502.18725v1)**
### **[Deep-Bench: Deep Learning Benchmark Dataset for Code Generation](http://arxiv.org/abs/2502.18726v1)**
### **[Random Forest-of-Thoughts: Uncertainty-aware Reasoning for Computational Social Science](http://arxiv.org/abs/2502.18729v1)**
### **[Cross-Modality Investigation on WESAD Stress Classification](http://arxiv.org/abs/2502.18733v1)**
### **[AI-Instruments: Embodying Prompts as Instruments to Abstract & Reflect Graphical Interface Commands as General-Purpose Tools](http://arxiv.org/abs/2502.18736v1)**
### **[Like Father, Like Son: Kinship-Aware Preference Mapping (KARMA) for Automatic Alignment in Large Language Models](http://arxiv.org/abs/2502.18744v1)**
### **[Automatic Prompt Optimization via Heuristic Search: A Survey](http://arxiv.org/abs/2502.18746v1)**
### **[Spectral-Enhanced Transformers: Leveraging Large-Scale Pretrained Models for Hyperspectral Object Tracking](http://arxiv.org/abs/2502.18748v1)**
### **[M-ANT: Efficient Low-bit Group Quantization for LLMs via Mathematically Adaptive Numerical Type](http://arxiv.org/abs/2502.18755v1)**
### **[Training Large Recommendation Models via Graph-Language Token Alignment](http://arxiv.org/abs/2502.18757v1)**
### **[CommGPT: A Graph and Retrieval-Augmented Multimodal Communication Foundation Model](http://arxiv.org/abs/2502.18763v1)**
### **[Reward Shaping to Mitigate Reward Hacking in RLHF](http://arxiv.org/abs/2502.18770v1)**
### **[Exploring Graph Tasks with Pure LLMs: A Comprehensive Benchmark and Investigation](http://arxiv.org/abs/2502.18771v1)**
### **[Plutus: Benchmarking Large Language Models in Low-Resource Greek Finance](http://arxiv.org/abs/2502.18772v1)**
### **[M2-omni: Advancing Omni-MLLM for Comprehensive Modality Support with Competitive Performance](http://arxiv.org/abs/2502.18778v1)**
### **[Towards Optimal Multi-draft Speculative Decoding](http://arxiv.org/abs/2502.18779v1)**
### **[Active Few-Shot Learning for Text Classification](http://arxiv.org/abs/2502.18782v1)**
### **[Seeing the Forest for the Trees: A Large Scale, Continuously Updating Meta-Analysis of Frontier LLMs](http://arxiv.org/abs/2502.18791v1)**
### **[SolEval: Benchmarking Large Language Models for Repository-level Solidity Code Generation](http://arxiv.org/abs/2502.18793v1)**
### **[Optimal Stochastic Trace Estimation in Generative Modeling](http://arxiv.org/abs/2502.18808v1)**
### **[Holistic Audit Dataset Generation for LLM Unlearning via Knowledge Graph Traversal and Redundancy Removal](http://arxiv.org/abs/2502.18810v1)**
### **[Judge as A Judge: Improving the Evaluation of Retrieval-Augmented Generation through the Judge-Consistency of Large Language Models](http://arxiv.org/abs/2502.18817v1)**
### **[CAMEx: Curvature-aware Merging of Experts](http://arxiv.org/abs/2502.18821v1)**
### **[Data-Efficient Multi-Agent Spatial Planning with LLMs](http://arxiv.org/abs/2502.18822v1)**
### **[Evidence-Driven Marker Extraction for Social Media Suicide Risk Detection](http://arxiv.org/abs/2502.18823v1)**
### **[Sentiment Analysis of Movie Reviews Using BERT](http://arxiv.org/abs/2502.18841v1)**
### **[Sliding Window Attention Training for Efficient Large Language Models](http://arxiv.org/abs/2502.18845v1)**
### **[A Causal Lens for Evaluating Faithfulness Metrics](http://arxiv.org/abs/2502.18848v1)**
### **[Marking Code Without Breaking It: Code Watermarking for Detecting LLM-Generated Code](http://arxiv.org/abs/2502.18851v1)**
### **[A Theoretical Perspective: How to Prevent Model Collapse in Self-consuming Training Loops](http://arxiv.org/abs/2502.18865v1)**
### **[Multi-LLM Collaborative Search for Complex Problem Solving](http://arxiv.org/abs/2502.18873v1)**
### **[Learning to Align Multi-Faceted Evaluation: A Unified and Robust Framework](http://arxiv.org/abs/2502.18874v1)**
### **[Learning to Generate Structured Output with Schema Reinforcement Learning](http://arxiv.org/abs/2502.18878v1)**
### **[From Hours to Minutes: Lossless Acceleration of Ultra Long Sequence Generation up to 100K Tokens](http://arxiv.org/abs/2502.18890v1)**
### **[An Empirical Study on Commit Message Generation using LLMs via In-Context Learning](http://arxiv.org/abs/2502.18904v1)**
### **[A Pipeline of Augmentation and Sequence Embedding for Classification of Imbalanced Network Traffic](http://arxiv.org/abs/2502.18909v1)**
### **[CLLoRA: An Approach to Measure the Effects of the Context Length for LLM Fine-Tuning](http://arxiv.org/abs/2502.18910v1)**
### **[END: Early Noise Dropping for Efficient and Effective Context Denoising](http://arxiv.org/abs/2502.18915v1)**
### **[ClassInvGen: Class Invariant Synthesis using Large Language Models](http://arxiv.org/abs/2502.18917v1)**
### **[Talking like Piping and Instrumentation Diagrams (P&IDs)](http://arxiv.org/abs/2502.18928v1)**
### **[JailBench: A Comprehensive Chinese Security Assessment Benchmark for Large Language Models](http://arxiv.org/abs/2502.18935v1)**
### **[Towards Label-Only Membership Inference Attack against Pre-trained Large Language Models](http://arxiv.org/abs/2502.18943v1)**
### **[Switching multiplicative watermark design against covert attacks](http://arxiv.org/abs/2502.18948v1)**
### **[DualSpec: Text-to-spatial-audio Generation via Dual-Spectrogram Guided Diffusion Model](http://arxiv.org/abs/2502.18952v1)**
### **[Know You First and Be You Better: Modeling Human-Like User Simulators via Implicit Profiles](http://arxiv.org/abs/2502.18968v1)**
### **[Low-Confidence Gold: Refining Low-Confidence Samples for Efficient Instruction Tuning](http://arxiv.org/abs/2502.18978v1)**
### **[PEToolLLM: Towards Personalized Tool Learning in Large Language Models](http://arxiv.org/abs/2502.18980v1)**
### **[GenTool: Enhancing Tool Generalization in Language Models through Zero-to-One and Weak-to-Strong Simulation](http://arxiv.org/abs/2502.18990v1)**
### **[OntologyRAG: Better and Faster Biomedical Code Mapping with Retrieval-Augmented Generation (RAG) Leveraging Ontology Knowledge Graphs and Large Language Models](http://arxiv.org/abs/2502.18992v1)**
### **[MEBench: Benchmarking Large Language Models for Cross-Document Multi-Entity Question Answering](http://arxiv.org/abs/2502.18993v1)**
### **[The Sharpness Disparity Principle in Transformers for Accelerating Language Model Pre-Training](http://arxiv.org/abs/2502.19002v1)**
### **[Binary Neural Networks for Large Language Model: A Survey](http://arxiv.org/abs/2502.19008v1)**
### **[Distilling Reinforcement Learning Algorithms for In-Context Model-Based Planning](http://arxiv.org/abs/2502.19009v1)**
### **[FungalZSL: Zero-Shot Fungal Classification with Image Captioning Using a Synthetic Data Approach](http://arxiv.org/abs/2502.19038v1)**
### **[Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs](http://arxiv.org/abs/2502.19041v1)**
### **[A Dual-Purpose Framework for Backdoor Defense and Backdoor Amplification in Diffusion Models](http://arxiv.org/abs/2502.19047v1)**
### **[MathClean: A Benchmark for Synthetic Mathematical Data Cleaning](http://arxiv.org/abs/2502.19058v1)**
### **[Can Large Language Models Outperform Non-Experts in Poetry Evaluation? A Comparative Study Using the Consensual Assessment Technique](http://arxiv.org/abs/2502.19064v1)**
### **[IndicEval-XL: Bridging Linguistic Diversity in Code Generation Across Indic Languages](http://arxiv.org/abs/2502.19067v1)**
### **[Sparse Brains are Also Adaptive Brains: Cognitive-Load-Aware Dynamic Activation for LLMs](http://arxiv.org/abs/2502.19078v1)**
### **[Nexus: A Lightweight and Scalable Multi-Agent Framework for Complex Tasks Automation](http://arxiv.org/abs/2502.19091v1)**
### **[LongEval: A Comprehensive Analysis of Long-Text Generation Through a Plan-based Paradigm](http://arxiv.org/abs/2502.19103v1)**
### **[The NeRF Signature: Codebook-Aided Watermarking for Neural Radiance Fields](http://arxiv.org/abs/2502.19125v1)**
### **[Self-Memory Alignment: Mitigating Factual Hallucinations with Generalized Improvement](http://arxiv.org/abs/2502.19127v1)**
### **[A Temporal Planning Framework for Multi-Agent Systems via LLM-Aided Knowledge Base Management](http://arxiv.org/abs/2502.19135v1)**
### **[Amulet: ReAlignment During Test Time for Personalized Preference Adaptation of LLMs](http://arxiv.org/abs/2502.19148v1)**
### **[Isolating Language-Coding from Problem-Solving: Benchmarking LLMs with PseudoEval](http://arxiv.org/abs/2502.19149v1)**
### **[RetinaRegen: A Hybrid Model for Readability and Detail Restoration in Fundus Images](http://arxiv.org/abs/2502.19153v1)**
### **[When Personalization Meets Reality: A Multi-Faceted Analysis of Personalized Preference Learning](http://arxiv.org/abs/2502.19158v1)**
### **[A Sliding Layer Merging Method for Efficient Depth-Wise Pruning in LLMs](http://arxiv.org/abs/2502.19159v1)**
### **[Detecting Linguistic Indicators for Stereotype Assessment with Large Language Models](http://arxiv.org/abs/2502.19160v1)**
### **[CodeIF: Benchmarking the Instruction-Following Capabilities of Large Language Models for Code Generation](http://arxiv.org/abs/2502.19166v1)**
### **[MEDDxAgent: A Unified Modular Agent Framework for Explainable Automatic Differential Diagnosis](http://arxiv.org/abs/2502.19175v1)**
### **[UQABench: Evaluating User Embedding for Prompting LLMs in Personalized Question Answering](http://arxiv.org/abs/2502.19178v1)**
### **[BIG-Bench Extra Hard](http://arxiv.org/abs/2502.19187v1)**
### **[Simulation of Language Evolution under Regulated Social Media Platforms: A Synergistic Approach of Large Language Models and Genetic Algorithms](http://arxiv.org/abs/2502.19193v1)**
### **[HDM: Hybrid Diffusion Model for Unified Image Anomaly Detection](http://arxiv.org/abs/2502.19200v1)**
### **[Bi'an: A Bilingual Benchmark and Model for Hallucination Detection in Retrieval-Augmented Generation](http://arxiv.org/abs/2502.19209v1)**
### **[Negation-Induced Forgetting in LLMs](http://arxiv.org/abs/2502.19211v1)**
### **[Two Heads Are Better Than One: Dual-Model Verbal Reflection at Inference-Time](http://arxiv.org/abs/2502.19230v1)**
### **[Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases](http://arxiv.org/abs/2502.19249v1)**
### **[ArtInsight: Enabling AI-Powered Artwork Engagement for Mixed Visual-Ability Families](http://arxiv.org/abs/2502.19263v1)**
### **[Efficient Federated Search for Retrieval-Augmented Generation](http://arxiv.org/abs/2502.19280v1)**
### **[Complex LLM Planning via Automated Heuristics Discovery](http://arxiv.org/abs/2502.19295v1)**
### **[Agent-centric Information Access](http://arxiv.org/abs/2502.19298v1)**
### **[Rethinking LLM Unlearning Objectives: A Gradient Perspective and Go Beyond](http://arxiv.org/abs/2502.19301v1)**
### **[Anomaly Detection in Complex Dynamical Systems: A Systematic Framework Using Embedding Theory and Physics-Inspired Consistency](http://arxiv.org/abs/2502.19307v1)**
### **[Shh, don't say that! Domain Certification in LLMs](http://arxiv.org/abs/2502.19320v1)**
### **[Agentic Reward Modeling: Integrating Human Preferences with Verifiable Correctness Signals for Reliable Reward Systems](http://arxiv.org/abs/2502.19328v1)**
### **[Evaluating LLMs and Pre-trained Models for Text Summarization Across Diverse Datasets](http://arxiv.org/abs/2502.19339v1)**
### **[Can Large Language Models Detect Errors in Long Chain-of-Thought Reasoning?](http://arxiv.org/abs/2502.19361v1)**
### **[DataMan: Data Manager for Pre-training Large Language Models](http://arxiv.org/abs/2502.19363v1)**
### **[TheoremExplainAgent: Towards Multimodal Explanations for LLM Theorem Understanding](http://arxiv.org/abs/2502.19400v1)**
### **[General Reasoning Requires Learning to Reason from the Get-go](http://arxiv.org/abs/2502.19402v1)**
### **[Learning Code-Edit Embedding to Model Student Debugging Behavior](http://arxiv.org/abs/2502.19407v1)**
### **[ImageChain: Advancing Sequential Image-to-Text Reasoning in Multimodal Large Language Models](http://arxiv.org/abs/2502.19409v1)**
### **[Less or More: Towards Glanceable Explanations for LLM Recommendations Using Ultra-Small Devices](http://arxiv.org/abs/2502.19410v1)**
### **[Code to Think, Think to Code: A Survey on Code-Enhanced Reasoning and Reasoning-Driven Code Intelligence in LLMs](http://arxiv.org/abs/2502.19411v1)**
### **[Norm Growth and Stability Challenges in Localized Sequential Knowledge Editing](http://arxiv.org/abs/2502.19416v1)**
