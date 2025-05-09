# The Latest Daily Papers - Date: 2025-05-09
## Highlight Papers
### **[The Aloe Family Recipe for Open and Specialized Healthcare LLMs](http://arxiv.org/abs/2505.04388v1)**
- **Summary**: Here's a concise summary and rigorous evaluation of the paper "The Aloe Family Recipe for Open and Specialized Healthcare LLMs":

**Summary:**

The paper introduces the Aloe family of open-source Large Language Models (LLMs) for healthcare, built on top of strong base models like Llama 3.1 and Qwen 2.5. It details a three-stage training methodology: (1) instruction tuning with supervised fine-tuning, (2) model merging using DARE-TIES to combine healthcare knowledge with general instruction-following abilities, and (3) model alignment using Direct Preference Optimization (DPO) to enhance safety and ethical performance, including resistance to jailbreaking attacks. The paper emphasizes data curation strategies, including synthetic Chain-of-Thought examples, and a robust evaluation methodology, encompassing open-ended, closed-ended, safety, and human assessments. The Aloe Beta models demonstrate competitive performance across various healthcare benchmarks and medical fields, often preferred by healthcare professionals. Crucially, the paper provides a healthcare-specific risk assessment to ensure responsible model release.

**Critical Evaluation:**

*   **Novelty:** The paper's novelty lies in several key aspects. The comprehensive three-stage training pipeline, optimized for healthcare LLMs, is well-defined. The use of model merging, specifically DARE-TIES, to combine specialized knowledge and general capabilities is valuable. The emphasis on rigorous safety alignment and the development of a healthcare-specific risk assessment are important contributions to responsible AI development. The combination of data preprocessing, model merging, and safety training shows holistic method.
*   **Significance:** The work contributes significantly to the open-source medical LLM field. It provides a recipe for creating high-performing, ethically aligned, and publicly available healthcare LLMs. The Aloe models demonstrate competitive performance compared to private alternatives, which can promote wider accessibility and scrutiny. The emphasis on safety, including jailbreaking resistance and risk assessment, addresses critical concerns in the deployment of AI in healthcare. It helps in data and training details, which contribute to reproducibility and further research in the field. The standardized evaluation offers new benchmarks to judge model's safety.
*   **Strengths:**
    *   **Comprehensive Methodology:** The paper outlines a detailed and reproducible training pipeline.
    *   **Strong Performance:** The Aloe models achieve competitive performance on healthcare benchmarks.
    *   **Safety Focus:** Emphasis on ethical alignment and jailbreaking resistance.
    *   **Open-Source Contribution:** Freely available models and datasets benefit the research community.
    *   **Risk Assessment:** Detailed analysis of potential risks and mitigation strategies.
*   **Weaknesses:**
    *   **Limited Ablation Studies:** Although the paper describes the complete architecture, it misses some key aspects. Limited analysis of the specific impact of each stage in the training pipeline or each type of data used for training. More ablation studies could have isolated the effect of individual components of the pipeline.

**Rationale for Score:**

The paper represents a strong contribution to the field. It provides a well-defined methodology, achieves competitive performance, addresses critical safety concerns, and offers valuable resources to the open-source community. While further exploration of individual component impact through ablation studies would have strengthened the analysis, the overall quality and potential influence of this work warrant a high score.

**Score: 8**

- **Score**: 8/10

### **[OBLIVIATE: Robust and Practical Machine Unlearning for Large Language Models](http://arxiv.org/abs/2505.04416v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "OBLIVIATE: Robust and Practical Machine Unlearning for Large Language Models":

**Summary:**

The paper introduces OBLIVIATE, a novel unlearning framework designed for large language models (LLMs).  The framework focuses on effectively removing targeted data while preserving both model utility (performance on retained tasks) and fluency (ability to generate coherent text).  OBLIVIATE uses a three-pronged loss function during fine-tuning: a masking loss to suppress the generation of unlearned data, a distillation loss to maintain performance on related data using teacher models, and a world-fact loss to retain general knowledge.  The framework is also designed to be efficient using low-rank adaptation (LoRA).  The authors conduct experiments across multiple datasets (Harry Potter, WMDP, TOFU) and evaluate the framework with various metrics including a new document-level memorization metric (DRMA), membership inference attacks (MIAs), and GPT-4o fluency scores. The results demonstrate that OBLIVIATE is effective in removing target data while maintaining model utility and fluency and is resistant to MIAs.

**Critical Evaluation:**

*   **Novelty:** The paper presents a well-structured framework with a clearly defined approach to LLM unlearning.  The main novelty lies in the combination of techniques:
    *   **Masked Loss:** The paper leverages a masked loss to aggressively target and remove specific knowledge. This contributes to enhanced unlearning, achieving strong deletion.
    *   **Distillation and World-Fact Losses:** By incorporating distillation and world-fact losses, the authors address the important issue of preventing catastrophic forgetting and preserving general knowledge, which is a common pitfall with aggressive unlearning methods.
    *   **Document-Level Memorization:** The introduction of the document-level RMA metric is a valuable contribution, as it provides a more holistic view of memorization compared to token-level analysis.
    *   **GPT-4o token identification:** The use of GPT-4o to find tokens can be considered a novel approach to addressing a hard problem with reasonable cost.

*   **Significance:** The research addresses a critical problem in the field of LLMs: the need for robust and practical unlearning mechanisms. The framework is significant for the following reasons:
    *   **Practicality:** The use of LoRA makes the framework computationally feasible for large models. The results also demonstrate a time efficiency, though more detailed comparisons with the individual cost breakdowns from other methods are desired.
    *   **Robustness:** The paper demonstrates effectiveness across multiple datasets and resistance to MIAs. This is important for real-world applications where security and privacy are paramount.
    *   **Comprehensive Evaluation:** The evaluation suite is comprehensive and includes metrics for forget quality, model utility, and fluency.
    *   **Addressing Ethical Concerns:** The work explicitly acknowledges and attempts to mitigate ethical concerns related to memorization, copyright infringement, and the generation of harmful content.

*   **Weaknesses:**
    *   **Reliance on GPT-4o:** The reliance on GPT-40 for identifying target tokens introduces some retrieval instability (which can lead to inconsistent extraction.)
    *   **Performance with smaller datasets:** The TOFU dataset exhibits limitations that may require further adaptation.
    *   **Limited Fluency Evaluation:** The fluency evaluation relies on GPT-40 scoring, which is not as reliable as human judgements. The presence of gibberish or blank outputs indicates some trade-offs with aggressive unlearning.
    *   **Comparisons:** More detailed timing analyses and comparisons with existing methods is desirable

*   **Potential Impact:** The OBLIVIATE framework has the potential to influence the development of more responsible and secure LLMs. It could be used by organizations to comply with data privacy regulations and mitigate the risk of generating harmful content. The introduction of DRMA could also lead to more sophisticated methods for assessing and controlling memorization in LLMs.

*   **Rigorous rationale for score:** The paper presents a well-designed and tested framework for unlearning in LLMs, addressing a vital problem in the field. While there are some weaknesses relating to the model's architecture and performance with smaller datasets, the combination of techniques, strong empirical results, and comprehensive evaluation justify a high score. The novel use of a masked loss combined with methods to preserve factual accuracy and utility makes the paper a clear advance, however, improvements in fluency and reliance on GPT-40 prevent the highest score.

Score: 8

- **Score**: 8/10

### **[TrajEvo: Designing Trajectory Prediction Heuristics via LLM-driven Evolution](http://arxiv.org/abs/2505.04480v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "TRAJEVO: Designing Trajectory Prediction Heuristics via LLM-driven Evolution":

**Summary:**

The paper introduces TRAJEVO, a framework that uses Large Language Models (LLMs) within an evolutionary algorithm to automatically design trajectory prediction heuristics.  Instead of relying on handcrafted rules or computationally expensive deep learning models, TRAJEVO iteratively generates, evaluates, and refines prediction heuristics based on trajectory data.  The system incorporates a Cross-Generation Elite Sampling strategy to maintain population diversity and a Statistics Feedback Loop to enable the LLM to analyze heuristic performance.  The authors demonstrate that TRAJEVO outperforms traditional heuristic methods and exhibits superior generalization on unseen datasets while maintaining computational efficiency and interpretability.

**Critical Evaluation:**

*   **Novelty:** The core novelty lies in the application of LLM-driven evolutionary algorithms to the specific problem of trajectory prediction heuristic design. While LLMs and evolutionary algorithms have been used in algorithmic design before, this paper focuses on a particularly important problem with a clear need for speed and interpretability, where current deep learning models have drawbacks. The two novel components of Cross-Generation Elite Sampling and the Statistics Feedback Loop enhance the evolutionary process and are not mere trivial applications of existing techniques.
*   **Significance:** Trajectory prediction is crucial in robotics and autonomous navigation, where real-time performance and safety are paramount.  Deep learning methods, while accurate in some scenarios, are often computationally expensive, lack explainability, and struggle with generalization. TRAJEVO offers a compelling alternative that addresses these shortcomings, providing a path toward automated design of fast, explainable, and generalizable heuristics. The demonstrated generalization ability on the SDD dataset is a significant result, indicating a potential for more robust deployment in real-world conditions.
*   **Strengths:**
    *   **Clear Problem Definition:** The paper clearly defines the problem and motivation, highlighting the limitations of existing methods.
    *   **Well-Defined Framework:** The TRAJEVO framework is well-described, with its core components (evolutionary loop, LLM integration, Cross-Generation Elite Sampling, and Statistics Feedback Loop) clearly explained.
    *   **Strong Experimental Results:**  The experimental results are compelling, demonstrating TRAJEVO's superior performance compared to heuristic baselines on the ETH-UCY datasets and its remarkable generalization on the SDD dataset. The speed comparisons are also a key selling point.
    *   **Explainability:** The framework produces human-readable and understandable Python code, increasing trust in the predictions of the trained heuristic.
*   **Weaknesses:**
    *   **In-Distribution Accuracy:** The paper acknowledges that TRAJEVO doesn't always achieve the absolute lowest error metrics compared to highly specialized deep learning models *within* the training distribution.  This suggests there's still room for improvement in accuracy, although the trade-off for speed and generalization is often worthwhile.
    *   **Input Data Complexity:** The current implementation primarily uses positional history as input. Incorporating richer sensor data (agent types, semantic maps, etc.) could further enhance the system's capabilities.
    *   **Downstream Task Performance:** The evaluation is based on standard trajectory prediction metrics. Direct optimization for task-specific objectives (e.g., collision avoidance in navigation) within a closed-loop system could provide more practical results.
*   **Influence:** If the research is reproduced and adopted, the impact of the paper could be significant, potentially changing how trajectory prediction models are developed, especially in resource-constrained environments and when interpretability is crucial. The framework provides a potential avenue to automate the tedious design cycle associated with traditional rule-based systems.

**Overall Assessment:**

The paper presents a novel and well-executed framework that addresses a significant challenge in trajectory prediction.  While there are areas for improvement, the combination of LLMs and evolutionary algorithms, the specific design choices for enhanced diversity and feedback, and the compelling experimental results warrant a high score. The paper's significance lies in providing a path toward automated design of fast, explainable, and generalizable heuristics, bridging the gap between handcrafted rules and complex neural networks.

Score: 8

- **Score**: 8/10

### **[ZeroSearch: Incentivize the Search Capability of LLMs without Searching](http://arxiv.org/abs/2505.04588v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces ZEROSEARCH, a novel reinforcement learning (RL) framework designed to improve the search capabilities of Large Language Models (LLMs) without relying on real-time interaction with external search engines. ZEROSEARCH addresses two key limitations of prior RL-based approaches: unpredictable document quality from live search engines and high API costs.  The approach involves: (1) Supervised fine-tuning an LLM to act as a retrieval module that can generate both relevant and noisy documents based on a query; and (2) Curriculum learning during RL training, where the quality of the generated documents is progressively degraded, forcing the policy model to reason more effectively in increasingly challenging retrieval scenarios.  Experiments demonstrate that ZEROSEARCH can incentivize search capabilities using a smaller LLM as the retrieval module, and larger retrieval modules can even outperform real search engines. The framework is shown to be generalizable across base and instruction-tuned models and compatible with various RL algorithms.

**Critical Evaluation:**

The paper presents a compelling approach to a significant problem in LLM research: improving access to external knowledge without incurring high costs or suffering from unreliable document quality. Here's a breakdown of the novelty, significance, strengths, and weaknesses:

*   **Novelty:** The core idea of simulating a search engine using a lightweight fine-tuned LLM and then using curriculum learning to train a policy model is relatively novel.  While RL for improving search is not new, the way they address the cost and quality issues using simulated search and curriculum learning is a substantial contribution. The approach of fine-tuning the LLM to be a controllable document generator (useful and noisy) is also interesting and facilitates the curriculum learning aspect.

*   **Significance:** RAG is a very important topic. Overcoming its limitations (specifically, cost and noise due to online searching) has direct real-world application. ZEROSEARCH directly contributes to making retrieval-augmented LLMs more practical. The results demonstrated are also impressive, showing that a simulated approach can match or even surpass the performance of live search.

*   **Strengths:**
    *   **Addresses a clear problem:** The paper directly tackles the practical challenges of API costs and document quality associated with RL-based search improvement for LLMs.
    *   **Well-designed approach:** The use of supervised fine-tuning for search simulation and curriculum learning provides a systematic and effective way to train the policy model.
    *   **Strong empirical results:** The experiments are thorough, comparing ZEROSEARCH against multiple baselines across several datasets and model types. The performance gains and generalization ability are well-demonstrated.
    *   **Scalability:** The approach has shown good results in scaling to very large LLMs with significant speed up.
    *   **Reproducibility/Practicality**: The framework appears relatively easy to implement and integrates well with established RL algorithms.

*   **Weaknesses:**
    *   **Reliance on SFT Data Quality:** The performance of the simulation LLM hinges on the quality of the supervised fine-tuning data, which needs to be carefully labeled (useful vs. noisy). Although they outlined how the SFT data is collected and labeled, its success is crucial to the entire approach.
    *   **Simulated vs. Real World:** While the experiments show strong performance, there's always a gap between simulated environments and the complexity of real-world web search. This needs to be acknowledged, although the results that match/exceed online search engines somewhat alleviate the concern.
    *   **Limited Discussion of Failure Cases**: While the paper demonstrates significant performance gains, it would benefit from a more detailed analysis of failure cases, providing insights into the limitations of the approach.
    *   **Potential resource cost:** The implementation is cheaper than querying the API but still depends on a high end GPU to conduct its training.

*   **Potential Influence:** The paper has the potential to influence future research in several ways:
    *   **Encouraging simulated environments for RL training:** The success of ZEROSEARCH could inspire more researchers to explore simulation-based approaches for training LLMs.
    *   **Curriculum learning for retrieval:** The curriculum learning strategy could be adopted and refined in other retrieval-augmented LLM settings.
    *   **Low-cost and controllable knowledge integration:** The framework offers a viable alternative to expensive and unpredictable API-based approaches.

**Overall:**

ZEROSEARCH offers a significant and practical advance in improving LLM search capabilities. The approach is well-designed, addresses a clear need, and demonstrates strong empirical results. The paper provides a strong case for simulation-based RL training, offering a compelling alternative to API-dependent methods. While some limitations exist (simulation realism, SFT data sensitivity), the overall contribution is substantial.

Score: 8. This score reflects the novelty of the approach, its significance in addressing key limitations of RAG, the rigor of the experimental evaluation, and the potential for influencing future research. The weaknesses mainly relate to potential implementation challenges in specific scenarios, rather than fundamental flaws in the design.
- **Score**: 8/10

### **[HiPerRAG: High-Performance Retrieval Augmented Generation for Scientific Insights](http://arxiv.org/abs/2505.04846v1)**
- **Summary**: Okay, I will provide a concise summary and a rigorous critical evaluation of the paper "HiPerRAG: High-Performance Retrieval Augmented Generation for Scientific Insights," including assigning a novelty and significance score with a thorough justification.

**Summary:**

The paper introduces HiPerRAG, a high-performance computing (HPC) powered Retrieval-Augmented Generation (RAG) workflow designed to index and retrieve knowledge from a large corpus of scientific articles (over 3.6 million). It addresses the challenges of scaling RAG to handle the exponential growth of scientific literature by focusing on efficient document parsing and accurate retrieval. The key components are: 1) Oreo, a high-throughput multimodal document parsing model; and 2) ColTrast, a query-aware encoder fine-tuning algorithm using contrastive learning and late interaction for enhanced retrieval accuracy. HiPerRAG demonstrates robust performance on scientific question answering benchmarks, achieving high accuracy on SciQ and PubMedQA, outperforming domain-specific models and commercial LLMs like GPT-4. The system is scaled on Polaris, Sunspot, and Frontier supercomputers to deliver million-document-scale RAG workflows, fostering interdisciplinary innovation.  The authors also introduce two new biomedical Q/A benchmarks for RAG.

**Rigorous and Critical Evaluation:**

*Strengths:*

*   **Addressing a Significant Problem:** The paper tackles a crucial challenge: the overwhelming volume of scientific literature and the need for effective knowledge access and synthesis. The use of RAG is a natural and well-motivated approach to improve the factuality of LLM outputs in this domain.
*   **Novel Components:** The introduction of Oreo and ColTrast presents potentially significant advancements. Oreo seems to offer a practical and efficient solution for parsing complex scientific documents, balancing speed and accuracy. ColTrast's combination of contrastive learning and late interaction is a well-reasoned approach to improve retrieval accuracy in a domain-specific context.
*   **Strong Experimental Results:** The paper demonstrates empirically that HiPerRAG achieves state-of-the-art performance on established scientific question answering benchmarks, and also on benchmarks that it created. The performance gains over existing models and even commercial LLMs are compelling.
*   **Scalability:**  The successful scaling of HiPerRAG to large HPC systems is a significant contribution. It showcases the practical feasibility of the proposed approach for handling massive scientific datasets.
*   **Well-Defined and Executed Experiments:** The experiments are clearly outlined, and the ablation studies (e.g., on ColTrast) provide valuable insights into the effectiveness of different components.
*   **New Datasets:** The creation and release of ProteinInteractionQA, ProteinFunctionQA, and BioSynthQPs represent a valuable contribution to the community. These datasets specifically address the needs of evaluating RAG systems in the biomedical domain.

*Weaknesses:*

*   **Limited Novelty in RAG Architecture:** While the components (Oreo and ColTrast) are novel, the overall RAG architecture is fairly standard. The paper integrates these components within a well-established framework. While not necessarily a weakness, this reduces the transformative novelty of the overall system.
*   **Dependency on Specific HPC Infrastructure:**  The system's design is heavily reliant on HPC infrastructure. While this enables impressive scaling, it potentially limits its accessibility for researchers without access to such resources. The paper could benefit from a discussion of potential adaptations for more resource-constrained environments.
*   **Limited ablation of generative model influence**: The results in Table 5 only contain two models: Mistral and Mixtral. While these are strong LLMs, the extent to which these impact the overall performance of the retrieval system is not rigorously tested. A wider variety of LLMs would help to better inform future architectural decisions.

*Significance:*

The paper has the potential to be highly significant within the scientific knowledge management and AI for science domains. HiPerRAG offers a concrete and effective solution for addressing the information overload problem faced by scientists. The proposed techniques could significantly accelerate scientific discovery by improving knowledge access and synthesis. The release of new benchmarks will also spur further research in this area. The HPC-enabled scaling is also a key point; large-scale RAG applications are necessary to manage the burgeoning scientific literature.

**Justification for Score:**

Considering the strengths and weaknesses, I assign a score of 8/10.

*   **A score of 9 or 10** would require more substantial architectural innovation in RAG itself or a demonstration of a truly transformative impact on scientific discovery.
*   **The score of 8** recognizes the practical significance of addressing the information overload problem in science, the well-executed implementation, the strong empirical results, the creation of new datasets, and the effective scaling of a RAG system on HPC resources. However, the limited architectural novelty in the overall RAG framework and the HPC dependency, as well as other weaknesses outlined, prevent it from reaching a higher score. While the components are novel and valuable, the overall contribution, while significant, falls short of being groundbreaking.

**Score: 8**

- **Score**: 8/10

### **[ConCISE: Confidence-guided Compression in Step-by-step Efficient Reasoning](http://arxiv.org/abs/2505.04881v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "ConCISE: Confidence-guided Compression in Step-by-step Efficient Reasoning":

**Summary:**

The paper introduces CONCISE, a novel framework for compressing reasoning chains generated by Large Reasoning Models (LRMs) while preserving task accuracy. It addresses the problem of verbose outputs in LRMs, which stem from redundant reflections due to two key patterns: *Confidence Deficit* (models reconsidering correct steps due to low confidence) and *Termination Delay* (models continuing reasoning after a confident answer is reached).  CONCISE integrates *Confidence Injection* to stabilize intermediate steps and *Early Stopping* to terminate reasoning when sufficient confidence is achieved.  The authors fine-tune LRMs using data generated by CONCISE via both SFT and SimPO, demonstrating significant reductions in output length (up to ~50% under SimPO) while maintaining or improving task accuracy across various reasoning benchmarks. The paper argues that CONCISE proactively suppresses redundant reflections, leading to more efficient and compact reasoning chains compared to existing methods that rely on post-hoc pruning or sampling-based selection.

**Critical Evaluation:**

*   **Novelty:** The paper's primary novelty lies in its confidence-guided perspective on reasoning chain compression. Existing methods often focus on correctness or brevity, whereas CONCISE explicitly models and addresses the *reasons* for redundancy, targeting the model's internal confidence. The identification of Confidence Deficit and Termination Delay as specific patterns contributing to verbosity is also a valuable contribution. The integration of Confidence Injection and Early Stopping, while conceptually straightforward, provides a practical and effective mechanism for proactive compression. In essence, it moves compression from a post-processing step (pruning) or data selection issue (sampling), to an inherent part of the inference process.

*   **Significance:** The significance of the paper stems from its potential to make LRMs more practical for resource-constrained environments and improve user experience. Reducing the verbosity of reasoning chains directly addresses key limitations of current LRMs, such as computational overhead and latency. Demonstrating robust performance across multiple reasoning benchmarks, along with successful generalization to out-of-domain tasks, strengthens the argument that CONCISE can be broadly applicable. The ablation studies effectively highlight the contributions of each component (Confidence Injection and Early Stopping).

*   **Strengths:**
    *   **Clear Problem Formulation:** The paper clearly defines the problem of verbose outputs in LRMs and identifies the underlying causes (Confidence Deficit and Termination Delay).
    *   **Principled Approach:**  CONCISE provides a principled way to mitigate redundancy by actively shaping the model's reasoning process based on confidence, rather than relying solely on post-hoc methods or data selection.
    *   **Empirical Validation:**  The paper presents comprehensive experimental results on multiple benchmarks, demonstrating the effectiveness of CONCISE in terms of both compression and accuracy.
    *   **Strong Ablation Study:** The Ablation study successfully shows that confidence injection and early stopping are each necessary for the best performance, highlighting that they each target different types of redundancy.
    *   **Effective Generalization:** The generalization to out-of-domain datasets, particularly GPQA\_diamond, showcases the robustness and adaptability of the framework.

*   **Weaknesses:**
    *   **Reliance on Hand-Crafted Confidence Phrases:** The Confidence Injection component relies on a manually curated pool of confidence phrases. While the authors conduct experiments to refine this pool, it introduces a potential source of bias and limits the framework's adaptability to different models or tasks. Developing a more automated way to generate or select confidence phrases could improve the framework.
    *   **Model Dependency:** The effectiveness of the Early Stopping mechanism and the optimal threshold (te) may be sensitive to the specific LRM architecture and scale.  The authors acknowledge this limitation, but further investigation into the framework's behavior across diverse LRMs would be beneficial.
    *   **Limited Intra-Step Compression:** The paper focuses primarily on reducing the number of steps in the reasoning chain but doesn't address the potential for compressing individual steps (intra-step compression). As the authors note, exploring techniques like TokenSkip alongside CONCISE could further improve compression ratios. While acknowledged in the limitations, this could significantly increase compression performance.

*   **Potential Impact:** The paper has the potential to influence future research in several ways:
    *   It encourages a shift in focus from post-hoc compression to proactive methods that shape the reasoning process itself.
    *   It highlights the importance of modeling and addressing the underlying causes of redundancy in LRMs.
    *   It provides a practical framework that can be used to improve the efficiency and usability of LRMs.

**Justification for Score:**

Despite some limitations, CONCISE presents a novel and significant contribution to the field of LRM compression. The confidence-guided perspective offers valuable insights into the origins of redundancy, and the proposed framework demonstrates impressive results across multiple benchmarks and settings. While the reliance on hand-crafted confidence phrases and potential model dependency are weaknesses, the paper's strengths outweigh these limitations. CONCISE addresses a critical problem in LRM research and provides a promising direction for future work.  Therefore, a score of 8 is justified because the paper presents significant novelty, strong empirical validation, and the potential for considerable impact on the field, while also acknowledging and discussing potential limitations.

**Score: 8**

- **Score**: 8/10

### **[Perception, Reason, Think, and Plan: A Survey on Large Multimodal Reasoning Models](http://arxiv.org/abs/2505.04921v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

This paper presents a survey on Large Multimodal Reasoning Models (LMRMs). It organizes existing research into a four-stage roadmap:
1.  Perception-Driven Modular Reasoning (task-specific modules).
2.  Language-Centric Short Reasoning (System-1, prompt-based).
3.  Language-Centric Long Reasoning (System-2, extended chains).
4.  Native Large Multimodal Reasoning Models (N-LMRMs).

The survey reviews models, architectures, learning methods, datasets, and benchmarks associated with each stage. It discusses challenges like omnimodal generalization, reasoning depth, and agentic behavior. The paper introduces the concept of N-LMRMs, focusing on native omnimodal perception, goal-driven cognition, unified representations, data synthesis, and learning from world experience. Finally, it identifies potential directions for advancing multimodal intelligence beyond current architectural constraints.

**Critical Evaluation:**

*   **Novelty:** The paper's strength lies in providing a comprehensive roadmap of the evolution of multimodal reasoning, connecting early modular designs with current LLM-based models and future N-LMRMs.  The four-stage organization is helpful in understanding the development path and the shift in design philosophies. While it builds upon existing research, the synthesis, structured organization, and the specific emphasis on reinforcement learning-enhanced reasoning in multimodal settings are valuable. The notion of "Native Large Multimodal Reasoning Models" is forward-looking, highlighting a potential paradigm shift rather than simply extrapolating from existing architectures.

*   **Significance:** The paper addresses a critical need for a coherent framework in the rapidly evolving field of multimodal reasoning. By categorizing and analyzing various approaches, it helps researchers understand the current landscape and identify gaps. The discussion of limitations (generalization, depth, agentic behavior) and the projection towards N-LMRMs provide a valuable direction for future research. The updated dataset and benchmark compilation is a helpful resource for the community.

*   **Strengths:**
    *   Comprehensive scope: Covers a wide range of models and techniques.
    *   Structured Organization: The four-stage roadmap is clear and insightful.
    *   Future-oriented:  Proposes the N-LMRM concept, anticipating trends.
    *   Up-to-date: includes recent advancements in the field.
    *   Helpful resources: Dataset and benchmark summary table.

*   **Weaknesses:**
    *   Somewhat descriptive:  While comprehensive, the survey is primarily descriptive. A more in-depth critical comparison of specific models within each stage (beyond a high-level categorization) could strengthen the analysis.
    *   N-LMRM concept is high-level: The N-LMRM section outlines general principles. More concrete architectural or training proposals would enhance its immediate impact.
    *   Limited quantitative analysis: More quantitative analysis of benchmark performance (beyond general mentions of "improvements") could further solidify the arguments.

*   **Potential Influence:** This survey has the potential to guide future research directions by clearly articulating the limitations of current LMRMs and providing a framework for developing more capable and adaptable reasoning systems. The emphasis on learning from world experience and creating more agentic models could be particularly influential.

**Justification for the score:**

I'm assigning a score of **8**.

*   **Rationale:** The paper offers a significant contribution by synthesizing a large body of work and providing a structured roadmap for understanding multimodal reasoning models. The forward-looking perspective on N-LMRMs is valuable. The limitations outlined above prevent it from achieving a higher score, particularly the high-level nature of N-LMRMs without a specific implementation/architecture proposal. However, it presents a strong foundation for understanding the direction of research, and a detailed analysis is carried out across these multiple reasoning models. Overall, the paper has the potential to significantly influence the field of multimodal reasoning.
Score: 8

- **Score**: 8/10

### **[Graffe: Graph Representation Learning via Diffusion Probabilistic Models](http://arxiv.org/abs/2505.04956v1)**
- **Summary**: Here is a summary and critical evaluation of the paper "Graffe: Graph Representation Learning via Diffusion Probabilistic Models":

**Summary:**

The paper introduces Graffe, a novel self-supervised graph representation learning framework based on diffusion probabilistic models (DPMs). Graffe addresses the challenge of adapting DPMs, primarily known for generative tasks, to the discriminative task of graph representation learning. The framework features a graph encoder that distills a source graph into a compact representation, which then conditions a diffusion decoder's denoising process. The authors provide theoretical justification by proving that the denoising objective implicitly maximizes conditional mutual information between the data and its representation, effectively following a "Diff-InfoMax" principle. Empirically, Graffe achieves competitive or state-of-the-art performance on node and graph classification tasks across various datasets.

**Critical Evaluation:**

*   **Novelty:** The paper's novelty lies in its principled integration of diffusion models into graph representation learning. While DPMs have shown promise in other domains (vision), their application to graphs is less explored. Graffe offers a dedicated framework addressing the specific challenges of graph data, such as its non-Euclidean nature and the need to extract meaningful representations. The theoretical analysis connecting the denoising objective to conditional mutual information and the Diff-InfoMax principle is also a significant contribution. The specific architecture Graffe uses, namely the GNN encoder with a tailored diffusion decoder for graphs, is also a novel design. This approach significantly moves past existing methods for graph representation learning.

*   **Significance:** The significance of this work stems from several factors:
    *   **Performance:** Graffe's strong empirical results on various benchmark datasets, including achieving state-of-the-art performance on many datasets, demonstrate its practical utility. It advances the state-of-the-art in graph representation learning.
    *   **Theoretical Foundation:** The theoretical analysis provides a solid foundation for understanding why DPMs can be effective for representation learning. This is crucial for guiding future research and development in this area. It offers insight into designing better graph representation learning models using principles from information theory.
    *   **Framework:** Graffe provides a flexible and adaptable framework that can be extended and modified for various graph-related tasks. The modular design makes it easier to experiment with different encoder architectures, diffusion decoders, and loss functions.
    *   **Bridging Generative and Discriminative Learning:** The work bridges the gap between generative and discriminative learning paradigms in the context of graph representation, leveraging the strengths of generative models for representation learning.

*   **Strengths:**
    *   Strong theoretical grounding and rigorous mathematical derivations.
    *   Comprehensive empirical evaluation on a diverse set of benchmark datasets.
    *   Clear and well-organized presentation.
    *   The Diff-InfoMax principle provides a novel perspective on the relationship between denoising objectives and mutual information maximization.

*   **Weaknesses:**
    *   While the experimental results are compelling, more detailed analysis into why Graffe performs particularly well on some datasets but not others would be useful. This requires analyzing properties specific to each graph dataset.
    *   The computational cost of training DPMs can be high, and it would be beneficial to discuss the trade-offs between performance and computational efficiency.
    *   More ablation studies dissecting key component choices would allow for a richer understanding of design trade-offs.
    *   The hyperparameter tuning seems rather involved, as discussed in the paper. More detail on how these choices affect performance would be beneficial.

*   **Potential Influence:** Graffe's clear theoretical framework and strong empirical results are likely to inspire further research in the use of DPMs for graph representation learning. The Diff-InfoMax principle may become a widely adopted concept in the field. The framework also has potential applications in areas such as drug discovery, social network analysis, and recommendation systems.

**Score: 8**

**Justification:**

Graffe represents a significant advancement in the field of graph representation learning by successfully integrating diffusion probabilistic models. The theoretical grounding, strong empirical performance, and flexible framework contribute substantially to the field. While the paper could benefit from more detailed analysis of specific dataset characteristics, and discussion of computational costs, its overall impact is considerable. The integration of DPMs, a relatively recent approach, for graph representation learning, is both novel and likely to impact the community. I would have assigned a 9 if the weaknesses identified had been more thoroughly analyzed. A score of 8 accurately reflects the significant contributions and the remaining opportunities for improvement.

- **Score**: 8/10

### **[ReAlign: Bilingual Text-to-Motion Generation via Step-Aware Reward-Guided Alignment](http://arxiv.org/abs/2505.04974v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "ReAlign: Bilingual Text-to-Motion Generation via Step-Aware Reward-Guided Alignment":

**Summary:**

The paper addresses the problem of bilingual text-to-motion generation, which aims to create 3D human motions from text descriptions in two languages. The authors identify two key challenges: the lack of bilingual motion-language datasets and the misalignment between text and motion distributions in diffusion models. To address these, they introduce BiHumanML3D, a new bilingual dataset; BiMD, a Bilingual Motion Diffusion model utilizing cross-lingual alignment; and ReAlign, a reward-guided sampling alignment method that incorporates a step-aware reward model to improve text-motion consistency and motion realism. The reward model combines a text-aligned module for semantic coherence and a motion-aligned module for realism, refining motions at each timestep to balance probability density and alignment. Experiments demonstrate that this approach improves both text-motion alignment and motion quality.

**Critical Evaluation:**

*   **Novelty:** The paper's novelty lies in several aspects:
    *   **Bilingual Dataset:** Creating BiHumanML3D addresses a significant gap in the availability of bilingual motion-language data. This is crucial for advancing research in this area.
    *   **Bilingual Motion Diffusion Model (BiMD):**  The use of cross-lingual alignment to create a unified bilingual model is a valuable contribution, moving away from training separate language-specific models. The strategy leverages shared semantic information across languages.
    *   **Reward-Guided Alignment (ReAlign):** The step-aware reward model is innovative. Explicitly accounting for noise variations across diffusion timesteps and integrating both text-alignment and motion-realism into a reward signal is a significant improvement over previous methods that primarily focus on fine-tuning motion quality. The plug-and-play nature of ReAlign is also a strong positive.

*   **Significance:** The work has the potential to be influential for several reasons:
    *   **Addresses a critical problem:** Bilingual text-to-motion generation broadens accessibility and applicability of motion synthesis to a global audience.
    *   **Provides a benchmark:**  BiHumanML3D will likely become a valuable resource for the research community, facilitating further progress in bilingual motion generation.
    *   **Improves alignment:** The ReAlign method demonstrably enhances text-motion coherence, addressing a fundamental challenge in diffusion-based motion generation. The gains observed by plugging this approach to existing methods underscore this.
    *   **Cross-lingual learning:** The model contributes to the broader area of cross-lingual learning and generative modeling.

*   **Strengths:**
    *   The problem is well-motivated, with clear explanations of the challenges involved.
    *   The proposed solutions are well-designed and technically sound.
    *   The experiments are comprehensive and provide strong evidence to support the claims made in the paper.
    *   The visual results (Figure 1 and S1) clearly show the improvements achieved by the proposed method.
    *   The ablation studies provide insights into the contribution of each component of the proposed framework.

*   **Weaknesses:**
    *   **Dependency on Pre-trained Models:** The model's reliance on pre-trained text encoders (OpenCLIP, XLM) and the diffusion model architecture is a point of consideration. Performance is intrinsically linked to the quality of these components. The dependency on a pre-trained diffusion model could limit how effectively the model samples *from* the generated space.
    *   **Generality of BiHumanML3D:** The paper does not deeply explore how well BiHumanML3D generalizes across different motion styles or complexities. The paper could benefit from exploring failure cases more fully.
    *   **Limited Language Scope:** While bilingual, the dataset only covers English and Chinese. Extending this to more languages would further increase its impact.

*   **Justification for Score:**

The paper makes significant contributions to the emerging field of bilingual text-to-motion generation. The introduction of the BiHumanML3D dataset fills a critical gap, while the BiMD model and ReAlign strategy represent valuable technical advances in cross-lingual semantic alignment and motion synthesis. The plug-and-play nature of ReAlign enhances its practical value. The limitations (dependency on pre-trained models and limited language scope) are acknowledged and do not diminish the core contributions. While further exploration of the limitations would bolster the paper, the strengths far outweigh the weaknesses. Therefore, a score of 8 reflects the paper's substantial novelty and potential for impact.

Score: 8

- **Score**: 8/10

### **[ChainMarks: Securing DNN Watermark with Cryptographic Chain](http://arxiv.org/abs/2505.04977v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "ChainMarks: Securing DNN Watermark with Cryptographic Chain":

**Summary:**

The paper "ChainMarks" proposes a novel dynamic DNN watermarking scheme designed to be more robust against watermark removal and ambiguity attacks than existing methods. The key idea is to introduce a cryptographic chain into the trigger inputs.  The trigger inputs are generated by repeatedly applying a hash function to a secret key. The target labels for these inputs are derived from the digital signature of the model owner. Watermark verification involves checking if the predicted labels for the generated trigger inputs match the target labels, using a two-phase Monte Carlo method to determine a decision threshold that accounts for the classification probabilities of the DNN. The paper claims improved robustness and security compared to existing methods, with better marginal utility (i.e., higher probability guarantee of watermark presence for a given level of accuracy).

**Critical Evaluation:**

*   **Novelty:** The introduction of a cryptographic chain to secure trigger inputs is a genuinely novel contribution. This approach effectively counters watermark ambiguity attacks that plague many existing methods. Using a two-phase Monte Carlo method to estimate the watermark presence threshold also presents a significant improvement over basic empirical methods, particularly in situations with small *p*-values.

*   **Significance:** Securing DNN models' intellectual property remains a crucial challenge as these models become more prevalent. The vulnerabilities of existing watermarking schemes necessitate more robust solutions. ChainMarks provides a significant step in this direction. The robustness and security against attacks are promising and address key weaknesses in current methodologies.

*   **Strengths:**
    *   **Robustness against Ambiguity Attacks:** The primary strength is the cryptographic chain that effectively prevents the optimization-based attacks needed to forge alternative watermarks.
    *   **Robustness against Removal Attacks:**  The use of "noise-like" trigger inputs which are far from the real input distributions helps to defend from attacks like fine-tuning.
    *   **Improved Threshold Estimation:** The two-phase Monte Carlo technique provides a more accurate means of deciding on watermark presence, particularly when low *p*-values are important for high security.
    *   **Extensive Evaluation:** The paper presents extensive experiments, comparing ChainMarks with four other watermarking schemes, against a range of attacks.

*   **Weaknesses:**
    *   **Assumptions about Access to Datasets:** The paper assumes the model owner has access to the original training dataset. This might not always be practical, especially when outsourcing model training.
    *   **Limited Generalizability Discussion:** Although the core concept is promising, the application to different DNN architectures (other than ResNets) or input data types (text, graphs) requires further investigation. The discussion of scalability to larger datasets (ImageNet) offers only high-level guidance.
    *  **Potential for Scalability Issues for Longer Chains:** While cryptographic chains offer a robust defence mechanism, the practical implementation of very long chains can face scalability issues, since they might necessitate a larger memory footprint for both embedding and verification.

*   **Potential Influence:** This paper has the potential to significantly influence the design of future DNN watermarking schemes. The introduction of cryptographic principles offers a new direction in securing watermarks, and the careful consideration of attack scenarios is valuable. The two-phase Monte Carlo estimation could become a standard technique for decision threshold calculations. It lays the ground work for future research on further integrating cryptographic techniques to increase security of the DNN models.

**Justification for Score:**

Considering the identified strengths and weaknesses, a score of 8 is appropriate. The novelty of the cryptographic chain provides a significant advancement in defending against watermark ambiguity attacks, a crucial problem in the field. The evaluation is comprehensive, demonstrating ChainMarks' superior performance. However, the assumptions about dataset access and limited generalizability (only images tested), along with potential scalability issues, prevent a higher score.
The paper has the potential to stimulate considerable follow-up research in more robust and secure DNN watermarking techniques.

**Score: 8**

- **Score**: 8/10

### **[SOAP: Style-Omniscient Animatable Portraits](http://arxiv.org/abs/2505.05022v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "SOAP: Style-Omniscient Animatable Portraits":

**Summary:**

The paper introduces SOAP, a novel framework for creating animatable 3D avatars from a single portrait image, regardless of the style (realistic, cartoon, anime, etc.). SOAP addresses the limitations of existing methods that are often style-specific or struggle with complex hairstyles and accessories. The method combines a multi-view diffusion model, trained on a large dataset of 3D heads with varying styles, with an adaptive optimization pipeline. This pipeline deforms a FLAME mesh to fit the input image, while preserving topology and rigging, thus enabling FACS-based animation, integration of eyeballs and teeth, and detailed representation of hairstyles and accessories.

**Critical Evaluation:**

*   **Novelty:** The core novelty lies in the *combination* of several techniques to achieve style-agnostic animatable 3D head reconstruction. While individual components like multi-view diffusion models and parametric head models exist, the integration is unique. The adaptive deformation and remeshing process to reconcile the output of the style-omniscient diffusion model with an animatable rig appear particularly novel. It is also the first that is truly style omniscient by design, learning and generalizing over the two extreme styles of human realism and non-realistic anime.
*   **Significance:** The ability to generate animatable avatars from a single image, across a wide range of styles, has significant implications for various applications like gaming, virtual reality, personalized avatars, and content creation. SOAP significantly lowers the barrier to entry for 3D avatar creation and opens up new possibilities for customization. Further, the focus on FACS-based animation and mesh-based output makes it compatible with existing animation pipelines, unlike many NeRF-based approaches. The release of the 24K 3D head dataset also represents a valuable contribution to the research community.
*   **Strengths:**
    *   **Style-agnostic Reconstruction:**  The framework demonstrably handles a broad range of styles, including realistic, cartoon, and anime, overcoming the limitations of style-specific approaches.
    *   **Animatability:**  The resulting avatars are fully animatable, with FACS-based animation, eye movements, and lip sync, making them useful for interactive applications.
    *   **Detailed Geometry:** SOAP captures complex hairstyles and accessories with good fidelity, addressing a common problem in single-view reconstruction.
    *   **Comprehensive Dataset:**  The publicly released dataset of 24k 3D heads is a significant resource for future research in the area.
*   **Weaknesses:**
    *   **Dependency on External Tools:** The framework relies on external tools for FLAME estimation, landmark detection, and head parsing, which may limit its robustness and performance on highly stylized inputs, as shown in the limitations section. It might suffer on inputs where these tools are unable to generate reliable initializations.
    *   **Resolution Limits:** The output resolution is limited by the diffusion model, potentially affecting the quality of fine details.  Increasing the output resolution could improve fidelity.
    *   **Failure Cases:** As with many ML based reconstruction techniques, SOAP has failure cases, resulting from wrong landmark initialization, errors in head parsing or when the initial FLAME estimation fails.
*   **Potential Influence:** The paper is likely to influence future research in single-view 3D head reconstruction and avatar generation. The style-agnostic approach and focus on animatability are valuable contributions. The combination of diffusion models with parametric head models and adaptive optimization could inspire new methods for combining generative and analytical techniques. The provided dataset will also prove to be quite important in the coming future.

Overall, the paper represents a significant advancement in single-view animatable 3D head reconstruction. The style-agnostic nature, detailed geometry capture, and full animatability are valuable contributions that address limitations of previous works. While the dependency on external tools and failure cases are present, the benefits outweigh the negatives.

**Score: 8.5**

**Rationale:** SOAP combines multiple known techniques, specifically diffusion models and parametric models with differentiable rendering, to achieve a novel solution to the problem of style omniscient 3D reconstruction. The adaptive deformation process is particularly interesting and contributes to this. Also, by releasing a dataset of 24k samples, the authors enable further progress to be made in the field. However, since it builds on previous work significantly and isn't fundamentally groundbreaking, it is not a 9.0 or above.

- **Score**: 8/10

### **[Towards Mitigating API Hallucination in Code Generated by LLMs with Hierarchical Dependency Aware](http://arxiv.org/abs/2505.05057v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper "Towards Mitigating API Hallucination in Code Generated by LLMs with Hierarchical Dependency Aware" introduces MARIN, a framework designed to reduce API hallucinations (incorrect API usage) in code generated by Large Language Models (LLMs). MARIN has two phases: (1) Hierarchical Dependency Mining, which analyzes local and global dependencies within a project to provide comprehensive context to the LLM; and (2) Dependency Constrained Decoding, which uses the identified dependencies to constrain the token generation process, ensuring the generated APIs align with project specifications. The authors also introduce a new benchmark, APIHulBench, and two new metrics, Micro Hallucination Number (MiHN) and Macro Hallucination Rate (MaHR), for evaluating API hallucination.  Experiments on several state-of-the-art LLMs demonstrate that MARIN significantly reduces API hallucinations compared to Retrieval-Augmented Generation (RAG) approaches. MARIN is also tested on Huawei's internal projects and proprietary LLMs, achieving similar improvements.

**Critical Evaluation:**

*   **Novelty:** The paper offers several novel contributions. The hierarchical dependency mining approach is a unique way to provide LLMs with a richer understanding of project context compared to existing RAG-based methods that primarily rely on code snippet retrieval.  The dependency-constrained decoding phase further enhances novelty by directly incorporating project specifications into the code generation process. The introduction of APIHulBench and the MiHN/MaHR metrics also addresses a gap in existing benchmarks, providing a more focused and realistic evaluation of API hallucination.

*   **Significance:** API hallucination is a significant issue that can hinder the adoption of LLMs in software development. The paper's demonstrated ability to reduce API hallucination has practical implications, improving the reliability and usability of LLM-generated code. The experiments are comprehensive, covering a range of open-source and proprietary LLMs, as well as both early and later-stage code development scenarios. The improvements are substantial and validated through statistical significance tests. The inclusion of results from an industrial setting (Huawei) strengthens the significance by showing the applicability to real-world projects. The released APIHulBench provides a valuable resource for future research in this area.

*   **Strengths:**
    *   Clear problem definition and motivation.
    *   Well-designed framework with a detailed explanation of each phase.
    *   Comprehensive experimental evaluation using diverse LLMs and benchmarks.
    *   Quantitative results demonstrating significant improvements over baselines.
    *   Evaluation in an industrial setting, increasing real-world relevance.
    *   Introduction of a new benchmark (APIHulBench) and metrics (MiHN and MaHR).

*   **Weaknesses:**
    *   The prompt template, while validated, might still benefit from broader community feedback for potential optimization.
    *   The evaluation is primarily focused on Java, which limits generalizability, though the authors suggest the approach is language-agnostic. Expanding the evaluation to other languages in future work is needed.
    *   While promising, the industrial experiments only involve one company (Huawei). More diverse industrial validation would further strengthen the results.
    *   The paper does not compare against recent methods, such as Documentation Augmented Generation (DAG).

*   **Impact:** The paper has the potential to significantly impact the field of LLM-based code generation. The MARIN framework provides a practical and effective approach for mitigating API hallucination, making LLM-generated code more reliable and usable. The APIHulBench benchmark will facilitate further research and development in this area.

**Justification of Score:**

Despite the minor weaknesses listed above, the paper presents a significant and novel contribution to the field of LLM-based code generation. The MARIN framework addresses a critical problem (API hallucination) with a well-designed and evaluated approach. The introduction of the APIHulBench benchmark and the strong empirical results further enhance the paper's value. Therefore, a score of 8 is warranted.

**Score: 8**

- **Score**: 8/10

### **[WaterDrum: Watermarking for Data-centric Unlearning Metric](http://arxiv.org/abs/2505.05064v1)**
- **Summary**: Here's a summary and evaluation of the provided research paper:

**Summary:**

The paper introduces WaterDrum, a novel data-centric unlearning metric for Large Language Models (LLMs). WaterDrum leverages robust text watermarking to address limitations of existing utility-centric unlearning metrics, particularly in scenarios where forget and retain sets contain semantically similar content, retraining is impractical, or model owners can manipulate the metric without actual unlearning. The paper defines clear desiderata for effective unlearning metrics, proposes WaterDrum based on the Waterfall watermarking framework, and introduces a new benchmark dataset, WaterDrum-Ax, to rigorously evaluate unlearning algorithms. The empirical results demonstrate WaterDrum's superior performance compared to existing metrics in satisfying the defined desiderata and its utility in benchmarking unlearning algorithms.

**Critical Evaluation:**

**Novelty:** The paper exhibits strong novelty in several aspects:

*   **Data-Centric Approach:** Moving away from utility-centric metrics to a data-centric approach using watermarking is a significant shift. This addresses the generalization limitations inherent in model performance-based evaluations.
*   **Practical Desiderata:** The paper's focus on practical desiderata such as feasibility, robustness to similar data, and resilience against adversarial model owners significantly strengthens its contribution, mirroring real-world concerns.
*   **WaterDrum-Ax Dataset:** The new benchmark dataset addresses key shortcomings in existing datasets, particularly in controlling the semantic similarity between forget and retain sets. This facilitates more rigorous and realistic unlearning evaluations.
*   **Application of watermarking** The paper is the first to develop a watermarking unlearning metric for LLMs in which the LLM owner can not influence the watermark.

**Significance:** The paper addresses a critical gap in the LLM unlearning literature by providing a more reliable and robust metric. The practical considerations and the new benchmark dataset have the potential to:

*   **Improve Unlearning Algorithms:**  A more accurate metric can guide the development of more effective unlearning algorithms.
*   **Increase Trust in LLMs:**  A reliable method to verify unlearning can increase trust in LLM deployments, especially in privacy-sensitive applications.
*   **Advance Research:** The WaterDrum-Ax dataset serves as a valuable resource for future research in LLM unlearning and evaluation.

**Strengths:**

*   **Clearly Defined Desiderata:** The paper lays out a clear, well-justified set of desiderata. This is crucial for rigorous evaluation.
*   **Robust Methodology:** The use of watermarking provides a counterfactual to measure the influence of data in the LLM outputs.
*   **Comprehensive Evaluation:** The empirical evaluation is comprehensive and demonstrates WaterDrum's advantages over existing metrics.
*   **Practical Relevance:** The consideration of real-world scenarios and adversarial model owners significantly enhances the practical relevance of the work.

**Weaknesses:**

*   **Reliance on Waterfall Framework:**  WaterDrum builds directly on the Waterfall watermarking framework. While Waterfall is a strong basis, the metric's performance is inherently tied to the robustness and limitations of that framework. It would be stronger if the watermark was more agnostic.
*   **Specific Experimental Setting:** While the WaterDrum-Ax dataset is a valuable contribution, the specific results are still tied to its characteristics and the selected LLM architectures. More diverse datasets and models would bolster the claims' generalizability.
*   **Limited Evaluation of Unlearning Algorithms:**  While the paper benchmarks some unlearning algorithms, the evaluation is relatively basic.  A more in-depth comparison, analyzing the trade-offs in unlearning effectiveness, utility preservation, and efficiency across a wider range of algorithms, would enhance the paper's impact.

**Justification of Score:**

The paper provides a significant advancement in LLM unlearning evaluation. The shift to a data-centric approach, consideration of practical challenges, and the introduction of the WaterDrum-Ax dataset are all valuable contributions. However, the reliance on the Waterfall framework and limited algorithm analysis prevent a higher score.

Score: 8

- **Score**: 8/10

### **[ItDPDM: Information-Theoretic Discrete Poisson Diffusion Model](http://arxiv.org/abs/2505.05082v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "ItDPDM: Information-Theoretic Discrete Poisson Diffusion Model":

**Summary:**

The paper introduces ItDPDM, a novel discrete diffusion model for generative modeling of discrete data. Unlike existing methods that either embed discrete data into continuous spaces or rely on variational lower bounds, ItDPDM operates directly in the discrete state-space using a Poisson diffusion process.  It leverages an information-theoretic approach, deriving a novel Poisson Reconstruction Loss (PRL) that has an exact relationship to the true negative log-likelihood.  This eliminates the need for approximate evidence lower bounds (ELBOs) used in other methods. The paper demonstrates the effectiveness of ItDPDM on symbolic music and image datasets, showing significant improvements in negative log-likelihood and faster convergence compared to baseline methods.

**Critical Evaluation:**

*   **Novelty:** The paper demonstrates a significant departure from existing diffusion models for discrete data. The use of a Poisson diffusion process, inspired by photon arrival models in sensors, is a novel approach for modeling discrete data generation. The derivation of the PRL and its exact relationship with the true negative log-likelihood is a substantial contribution. While the individual components (diffusion models, information-theoretic principles, Poisson distributions) are known, their combination in this specific way, tailored to the discrete data domain, constitutes a genuine innovation.

*   **Significance:** The significance lies in addressing the limitations of current methods in generative modeling of discrete data. Embedding into continuous spaces and using ELBOs are suboptimal, leading to training and inference discrepancies. ItDPDM overcomes these challenges, potentially leading to more efficient and accurate generative models for a wide range of discrete data types, including but not limited to images and symbolic music. Demonstrating superior performance on common benchmark datasets bolsters this claim. The faster convergence rate also has practical implications for reducing training time and computational resources.

*   **Strengths:**

    *   **Strong Theoretical Foundation:** The paper is grounded in information theory, providing a solid theoretical framework for the proposed model and loss function.
    *   **Clear Presentation:** The paper clearly explains the concepts and derivations, making it accessible to researchers in the field.
    *   **Empirical Validation:** The experiments demonstrate significant improvements over baselines on standard datasets, backing up the theoretical claims.
    *   **Addresses a Real Problem:** The paper directly tackles the challenges in generative modeling of discrete data.

*   **Weaknesses:**

    *   **Architecture Dependence:** The paper relies on established architectures like U-Nets and Transformer encoders for the denoiser. While this allows for a fair comparison, the results may be somewhat contingent on the effectiveness of these architectures. Further research could explore specifically designed denoiser architectures for the Poisson diffusion process.
    *   **Limited Dataset Variety:** While CIFAR-10 and LMD are standard benchmarks, demonstrating the model's effectiveness on a wider array of diverse discrete data types (e.g., text, graphs) would further strengthen the significance.
    *   **Computational Complexity Analysis:** While the paper mentions the computational cost, a more detailed analysis comparing it to existing discrete-state diffusion models would be beneficial.
    *   **No Image Samples:** While quantitative metrics like NLL are reported, the paper lacks image samples to demonstrate visual quality.

*   **Potential Influence:** This paper has the potential to significantly influence the field by providing a new, more principled, and effective approach to generative modeling of discrete data. The information-theoretic framework and exact loss function could inspire new research directions and improvements in discrete diffusion models. The code will also help move the research forward.

**Score: 8**

**Justification:**

The paper demonstrates significant novelty in its approach to discrete diffusion modeling. The use of Poisson diffusion and the PRL provides a strong theoretical and practical advantage over existing methods that approximate. The reported empirical results showing large improvements in NLL and faster convergence suggest the superiority of the approach and increase the paper's significance in the field. A score of 8 reflects these strengths and acknowledges that, while promising, the work could be enhanced with a wider array of experiments, visual results, and a more detailed computational cost analysis.

- **Score**: 8/10

### **[Reliably Bounding False Positives: A Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction](http://arxiv.org/abs/2505.05084v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Reliably Bounding False Positives: A Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction":

**Summary:**

The paper addresses the critical issue of high false positive rates (FPRs) in machine-generated text (MGT) detection.  While many existing detectors focus on maximizing detection accuracy, they often neglect the societal harm caused by falsely accusing human-written text of being machine-generated. The authors propose a zero-shot framework, Multiscaled Conformal Prediction (MCP), that leverages Conformal Prediction (CP) to reliably constrain the upper bound of the FPR.  MCP divides texts into different length intervals and calculates different nonconformity scores for each length, to account for a correlation between text length and nonconformity score. The paper also introduces RealDet, a new, large-scale, diverse dataset specifically designed for MGT detection that reflects a more realistic distribution of human and machine-written texts.  Experimental results demonstrate that MCP effectively constrains FPRs, improves detection performance (particularly at lower FPR thresholds), and enhances robustness against adversarial attacks.

**Critical Evaluation:**

*   **Novelty:**  The paper presents a novel application of conformal prediction to the MGT detection problem.  While CP is not a completely new technique, its use in this context, combined with the multiscaled approach, is a significant contribution.  The multiscaled quantile calculation is key innovation, addressing a previously overlooked problem related to the effect of text length on nonconformity scores. Introducing and leveraging the RealDet dataset is also a worthwhile contribution.

*   **Significance:** High FPRs are a major barrier to the practical deployment of MGT detectors.  If detectors frequently misclassify human-written content, their credibility and usefulness are severely diminished. By demonstrating that MCP can reliably bound FPRs without drastically sacrificing detection performance, the paper makes a significant step towards more trustworthy MGT detection systems. The framework's zero-shot nature is also beneficial, as it reduces the need for extensive training on task-specific datasets.
    The RealDet dataset is also a significant contribution, as it addresses the limitations of existing datasets in representing the diversity and complexity of real-world text. Datasets that lack domain and model diversity may not adequately capture the statistical characteristics of genuine human-written text. This dataset has the potential to advance further research in more robust detectors, thus increasing the significance.

*   **Strengths:**

    *   **Clear Problem Definition:**  The paper clearly articulates the importance of controlling FPRs in MGT detection and the limitations of existing approaches.
    *   **Well-Defined Solution:**  The MCP framework is well-defined and theoretically grounded in conformal prediction.
    *   **Comprehensive Evaluation:**  The experimental evaluation is extensive, covering multiple datasets, detectors, and adversarial attack scenarios.
    *   **Strong Empirical Results:**  The results convincingly demonstrate the effectiveness of MCP in constraining FPRs and improving detection performance.
    *   **Novel Dataset:** The introduction of RealDet addresses a crucial gap in existing MGT benchmarks.

*   **Weaknesses:**

    *   **Complexity:** While CP provides theoretical guarantees, the implementation and tuning of MCP (especially the multiscaled quantile calculation) can add some complexity.
    *   **Reliance on Baseline Detector:**  The performance of MCP is still dependent on the underlying baseline detector.  A poor baseline detector will likely limit the effectiveness of MCP.
    *   **Binning Strategy:** The binning strategy based on fixed-width intervals, despite being effective, seems somewhat arbitrary. As the authors state in the limitations, there may be more optimal strategies for selecting thresholds across different bin widths.
    *   **Generalizability:** Though it attempts to address it with RealDet, MGT is a moving target. Techniques that work well today might be easily bypassed by future LLMs.

*   **Potential Influence:** The paper has the potential to influence future research in MGT detection by shifting the focus towards more reliable and trustworthy systems. The MCP framework provides a valuable tool for developers to control FPRs, while the RealDet dataset provides a more realistic benchmark for evaluating detector performance.

**Score: 8**

**Rationale:** The paper presents a strong and well-evaluated solution to a critical problem in MGT detection, demonstrating clear improvements over existing approaches. The introduction of the RealDet dataset is also a significant contribution. While some aspects of the MCP framework, such as the binning strategy, could be further refined, the paper's novelty and potential impact on the field justify a score of 8. This is not a perfect 10 because the underlying reliance on the base model and relative complexity limit it from being truly groundbreaking. While it improves upon other techniques, it can be seen as a step in an incremental manner, rather than a transformative advancement.

- **Score**: 8/10

### **[Multi-agent Embodied AI: Advances and Future Directions](http://arxiv.org/abs/2505.05108v1)**
- **Summary**: Here is a summary and critical evaluation of the provided paper:

**Summary**

This paper presents a comprehensive survey of multi-agent embodied AI (MAEAI). It positions MAEAI as a crucial step beyond single-agent embodied AI, addressing real-world scenarios where multiple intelligent agents interact and collaborate in dynamic, open environments. The survey covers key concepts (embodied AI, multi-agent systems), fundamental methods (optimal control, reinforcement learning, hierarchical learning, imitation learning, generative models), and explores how these are applied in MAEAI. The paper examines both classical control and planning approaches, learning-based techniques, and the emerging role of generative models in MAEAI, encompassing aspects like task allocation, distributed decision-making, and human-AI coordination. A significant portion of the paper focuses on reviewing existing benchmarks tailored for evaluating MAEAI systems. The work concludes by outlining the challenges and future research directions, including theoretical frameworks, algorithmic development, efficiency in learning, the integration of large language models, general frameworks, and adaptation to open environments.

**Critical Evaluation**

*   **Novelty and Significance:** The paper fills a significant gap by providing a focused survey on MAEAI.  While there are surveys on general embodied AI and on multi-agent reinforcement learning (MARL), a comprehensive overview *specifically* on the integration of these two fields has been lacking. This focus is important because MAEAI poses unique challenges that are not adequately addressed by simply extending single-agent methods or applying general MARL techniques. This targeted survey could significantly help researchers by consolidating the state-of-the-art.

*   **Strengths:**
    *   **Comprehensive Coverage:** The paper provides a well-structured review, covering a wide range of relevant concepts, methods, and applications.
    *   **Clear Organization:** The organization is logical and facilitates understanding, progressing from foundational concepts to more specialized MAEAI topics.
    *   **Benchmark Emphasis:** The inclusion and discussion of specialized benchmarks are extremely valuable. This directs researchers toward evaluating their systems on appropriate and standardized tasks.
    *   **Forward-Looking:** The identified challenges and future directions provide a useful roadmap for researchers.
    *   **Up-to-date:** The survey includes recent developments and references, such as those leveraging LLMs, which is crucial given the rapid evolution of the field.

*   **Weaknesses:**
    *   **Limited Critical Analysis of Benchmarks:**  While the survey lists benchmarks, a more in-depth critical evaluation of their limitations (e.g., sim-to-real gap, narrow task focus, insufficient complexity) would strengthen the paper. This would provide more actionable insights for benchmark development.
    *   **Depth in Specific Techniques:**  While the breadth is commendable, the depth on individual algorithms or architectures is sometimes limited. The paper could delve more critically into the advantages and disadvantages of specific methods within MAEAI.
    *   **Missing Connection Between Challenges:** The survey does not provide a synthesis to link the discussed challenges. For example, adaptation to open environments poses a scalability problem from algorithmic design for efficient learning. This synthesis would increase the impact of the survey by enabling new research directions.
    *   **Limited discussion of Safety and Ethical Considerations:**  Considering the increased autonomy and impact of MAEAI systems on real-world environments, especially within collaborative contexts, the survey provides limited consideration for safety protocols, error handling, and societal implications. Including a thorough discussion of these considerations is highly advisable given the current trajectory of the field and would enhance the applicability of the document.

*   **Potential Influence:** The survey has strong potential to guide future research in MAEAI by: (1) clarifying the scope of the field, (2) highlighting important research directions, (3) promoting the use of relevant benchmarks, and (4) fostering a more integrated view of the intersection of embodied AI and multi-agent learning.

**Score: 8**

**Rationale:** The paper makes a strong contribution by offering a much-needed, focused survey on multi-agent embodied AI. Its comprehensiveness, clear structure, and emphasis on benchmarks are valuable assets. However, the limited critical analysis of benchmarks, relatively shallow dive into technical details, and discussion about safety, and a lack of a coherent narrative synthesizing the presented challenges hold it back from being truly exceptional. It will certainly be a valuable resource for the community, but with some improvements, it could become a landmark paper in the field.

- **Score**: 8/10

### **[Revealing Weaknesses in Text Watermarking Through Self-Information Rewrite Attacks](http://arxiv.org/abs/2505.05190v1)**
- **Summary**: Here's a summary and critical evaluation of the provided research paper:

**Summary:**

The paper introduces a novel attack, called Self-Information Rewrite Attack (SIRA), against text watermarking techniques in large language models (LLMs).  SIRA exploits a common design choice in watermarking: embedding signals in high-entropy tokens to maintain text quality. SIRA identifies these potential pattern tokens by calculating their self-information within the text context and then strategically rewrites the text to remove the watermark signal. The attack operates in a black-box setting, requiring no access to the watermarking algorithm, secret keys, or detector.  Experiments demonstrate that SIRA achieves near-perfect attack success rates across seven recent watermarking methods with minimal computational cost and can be easily transferred across different LLMs. The authors argue that their findings highlight a fundamental vulnerability in current watermarking approaches and call for more robust watermarking techniques.

**Critical Evaluation:**

*   **Novelty:** The paper's primary novelty lies in the strategic use of self-information to target specific tokens for rewriting, effectively turning a generic paraphrasing task into a more focused fill-in-the-blank problem. While previous paraphrasing attacks exist, SIRA demonstrates a significantly more efficient and successful approach. The identification and exploitation of high-entropy tokens as potential watermark carriers is a key insight.
*   **Significance:** The implications of this work are substantial. By exposing a common vulnerability across multiple watermarking algorithms, the paper underscores the need for a re-evaluation of current watermarking designs. The ease of implementation and transferability of SIRA pose a real threat to the effectiveness of existing watermarking strategies. The fact that it is a black box attack and works well against previous strong algorithms such as SIR significantly raises the practical importance of the paper. The low computational cost of the attack means that it can be readily deployed by an adversary.
*   **Strengths:**
    *   **Clear Problem Statement:** The paper clearly defines the problem and assumptions, specifically focusing on a black-box adversarial setting.
    *   **Well-Defined Methodology:** The SIRA attack is described in detail, with a clear algorithm presented in the pseudocode. The justifications for each step of the attack are well explained, most notably the use of high entropy tokens.
    *   **Comprehensive Evaluation:** The experiments are well-designed, comparing SIRA against several existing attacks and watermarking algorithms. The use of multiple LLMs as attack models as well as multiple text metrics demonstrates the attack is robust and doesn’t degrade text quality in the same way previous methods do.
    *   **Strong Results:** The experimental results are compelling, showing near-perfect attack success rates across a range of watermarking techniques with low computational cost. The paper is also well written and easy to follow.
    *   **Thorough analysis.** The analysis on the benefits of SIRA is quite thorough, and gives a lot of intuition as to why it works better than previous methods. The paper doesn't just give results but explains in detail as to why.
*   **Weaknesses:**
    *   **Limited Scope of Datasets:** While C4 is a common benchmark, it's a general-purpose dataset. It may be helpful to evaluate SIRA on more specific or specialized datasets to assess its effectiveness across various content types.
    *   **Theoretical Framework:** While Lemma H.1 helps justify their approach, there can be further theoretical analysis as to precisely why the attack works under what assumptions, e.g. how exactly does the watermarking shift the distribution of the LLM?
    *   **Limited Discussion of Defenses:** While the paper focuses on attack strategies, it could benefit from a more in-depth discussion of potential defenses or mitigations against SIRA. More analysis as to why it works over previous methods such as DIPPER is required.
    *   **Z-Score threshold.** While the z-score threshold helps mitigate the issue of detector performance, as stated in the paper there is a degree of arbitrariness here that could be improved on.

*   **Potential Influence:** This paper has the potential to significantly influence the field of text watermarking. It forces a critical re-evaluation of existing techniques and motivates the development of more robust and resilient watermarking strategies.

*   **Justification:** The paper presents a compelling and well-executed attack that reveals a critical vulnerability in current watermarking algorithms. While there are some limitations, the novelty of the SIRA approach, the significance of its findings, and the potential impact on the field of text watermarking justify the score. The potential for this paper to help improve current watermarking strategies and push LLMs towards improved safety is quite high.

Score: 8

- **Score**: 8/10

### **[Diffusion Model Quantization: A Review](http://arxiv.org/abs/2505.05215v1)**
- **Summary**: Here's a summary and critical evaluation of the provided paper:

**Summary:**

This paper presents a comprehensive survey of diffusion model quantization techniques, a crucial area for deploying these powerful generative models on resource-constrained devices. It systematically categorizes existing methods for both UNet-based and Diffusion Transformer (DiT) architectures, differentiating between Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT) approaches. The survey analyzes the challenges inherent in quantizing diffusion models, including those arising from the multi-step denoising process and model-specific architectures.  It benchmarks several open-source solutions across various image generation tasks and provides qualitative analyses of quantization artifacts. The paper concludes by outlining potential future research directions in the field.

**Critical Evaluation:**

*   **Novelty:**  The primary novelty of this paper lies in its comprehensive and structured overview of diffusion model quantization. While individual quantization techniques have been explored in prior works, this survey fills a gap by providing a consolidated and categorized view of the landscape. It identifies key challenges and classifies existing solutions based on their underlying principles.
    *   It offers the first in-depth analysis of the challenges inherent to quantizing these models specifically.
    *   Provides a comprehensive taxonomy of quantization solutions, categorized by approach (PTQ vs. QAT) and architecture.
    *   Offers a combined quantitative benchmark and qualitative error analysis.

*   **Significance:** Diffusion models are computationally intensive, hindering their widespread adoption on edge devices. Quantization is a key enabler for efficient deployment, making this survey highly relevant. The paper is significant because:

    *   It facilitates understanding of the trade-offs involved in different quantization techniques, aiding researchers and practitioners in selecting the most suitable approach for their specific needs.
    *   The thorough benchmarking provides a valuable resource for comparing the performance of different quantization methods.
    *   The identification of future research directions can stimulate further innovation in the field, especially towards addressing the challenges that persist in diffusion models.

*   **Strengths:**
    *   The taxonomy is well-defined and helpful for understanding the relationships between different quantization approaches.
    *   The benchmarking provides a rigorous quantitative comparison of various methods, covering three image generation tasks.
    *   The qualitative analysis of quantization artifacts offers valuable insights into the effects of different techniques.
    *   The identified research directions are insightful and relevant to the field's future development.
    *   The paper aims to offer an interdisciplinary perspective by introducing concepts that bridge CNN architectures and LLMs.

*   **Weaknesses:**
    *   The survey covers existing work primarily up to early 2025. The rapid pace of development in this field could mean the survey will need updates regularly.
    *   While the benchmarking is thorough, it is limited to open-source solutions. It could benefit from including some proprietary or more complex methods, even if only with summarized results.
    *   The experimental setting is not comprehensive enough, it is limited to ImageNet256x256 for conditional image generation.
    *   The qualitative analysis is somewhat subjective. While visually informative, it could benefit from more quantitative measures to support the observations.
    *   The conclusion of this survey may be considered premature as future studies for direction investigation and validation are still required.

*   **Potential Influence:** The paper has the potential to influence future research by:

    *   Providing a common framework for understanding and comparing diffusion model quantization techniques.
    *   Highlighting the key challenges that need to be addressed for further progress.
    *   Inspiring the development of new and more effective quantization methods.
    *   Guiding practitioners in selecting appropriate quantization methods for their applications.

* **Overall:** This survey is a solid contribution to the field of diffusion model compression and acceleration. It's a valuable resource for researchers looking to understand the landscape of diffusion model quantization, identify promising research directions, and compare the performance of different techniques.
Score: 8
Rationale:
The paper is a well-structured and comprehensive survey of diffusion model quantization. It fills a significant gap by providing a consolidated overview of various methods, their challenges, and their performance. While it suffers from some minor limitations regarding coverage and the need for updates due to the field's fast pace, its overall contribution and potential influence justify a score of 8. The combination of a comprehensive taxonomy, quantitative benchmarking, and qualitative analysis makes it a valuable resource for researchers and practitioners alike.

- **Score**: 8/10

### **[Latte: Transfering LLMs` Latent-level Knowledge for Few-shot Tabular Learning](http://arxiv.org/abs/2505.05237v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces "Latte," a novel framework for few-shot tabular learning that leverages latent-level knowledge from large language models (LLMs).  Unlike existing approaches that rely on test-time knowledge extraction or text-level knowledge, Latte performs training-time knowledge extraction, distilling the latent prior knowledge within LLMs to optimize a downstream tabular learning model.  Latte comprises a semantic-aware tabular encoder (which integrates feature semantics), and a knowledge adapter (which uses latent-level knowledge to weightedly fuse information from feature values). An unsupervised pre-training stage using unlabeled data is also incorporated.  The authors demonstrate Latte's superior performance compared to existing methods across various few-shot tabular learning benchmarks, highlighting its effectiveness and establishing it as a state-of-the-art approach.

**Critical Evaluation:**

*Novelty:* The core idea of using latent-level knowledge from LLMs *during training* to guide a downstream tabular model is a significant step beyond existing methods that either use LLMs at inference time (leading to latency issues) or rely on unreliable text-level knowledge (as in FeatLLM). The design of the knowledge adapter to bridge the semantic gap between LLMs and the tabular encoder is a key novel aspect. The inclusion of a semantic-aware encoder is also notable, as it addresses a known weakness of existing tabular learning methods.  The incorporation of unsupervised pre-training using pseudo-labels generated from clustering is a relatively standard technique, but its integration within the Latte framework enhances its practical utility.

*Significance:* Few-shot tabular learning is a practically important area because of the high cost of data annotation.  LLMs have shown promise but have limitations, which Latte attempts to address. The paper's results demonstrate tangible performance improvements over existing methods. This suggests that the approach has the potential to impact how few-shot tabular learning problems are approached.  The comprehensive ablation studies provide valuable insights into the contribution of each component. The efficiency gains due to reduced LLM calls are also significant, making the approach more deployable in resource-constrained settings.

*Strengths:*
*   Clear problem formulation and well-defined methodology.
*   Novel use of latent-level knowledge distillation.
*   Semantic-aware encoder specifically designed for tabular data.
*   Comprehensive experimental evaluation on multiple datasets.
*   Detailed ablation studies that validate the design choices.
*   Demonstrated efficiency improvements in terms of LLM calls.

*Weaknesses:*
*   While the use of pseudo-labels is common, the sensitivity to the clustering parameters and the impact of the quality of the pseudo-labels could be investigated further.
*   The paper focuses heavily on performance metrics. A deeper qualitative analysis of the knowledge transferred from the LLM and how it influences the downstream model's decision-making would further strengthen the paper. What specific relationships between features is it picking up?
*   The reliance on metadata (task and feature descriptions) is potentially limiting.  The performance of Latte is dependent on the quality and availability of this metadata. The paper could explore how the system could be made more robust to noisy or incomplete metadata.
*   The performance gains from simply increasing the size of the LLM aren't significant. This suggests there may be a diminishing returns effect. A better understanding of what the LLM contributes beyond a certain size is needed.

*Potential Influence:* Latte has the potential to become a widely adopted approach for few-shot tabular learning. The methodology can be applied to other tasks and domains by choosing the right tabular encoder. The paper might inspire more work in training-time distillation of knowledge from LLMs for structured data, including methods for more robust metadata handling and ways to better extract and represent relevant relationships to downstream models.

Score: 8

*Justification:* Latte presents a significant advance in few-shot tabular learning by strategically leveraging latent-level knowledge from LLMs during training. The novelty lies in the overall architecture and the knowledge adapter design. The comprehensive experimental results, efficiency gains, and ablation studies convincingly demonstrate the framework's effectiveness. The weaknesses primarily concern the reliance on high-quality metadata and the need for a deeper qualitative analysis. However, the paper's strengths significantly outweigh its weaknesses, and it is expected to have a strong influence on future research in this area.

- **Score**: 8/10

### **[ICon: In-Context Contribution for Automatic Data Selection](http://arxiv.org/abs/2505.05327v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces ICON (In-Context Contribution for Measurement), a novel data selection method for instruction tuning of large language models (LLMs). ICON leverages the implicit fine-tuning nature of in-context learning (ICL) to measure the contribution of individual data samples without relying on computationally expensive gradient-based methods or manually designed heuristics.  ICON comprises three components: an assessment set, a contribution score based on perplexity changes under ICL, and a selection paradigm trained with LoRA.  Experiments on LLaMA3.1-8B, Qwen2.5-3B, and LLaMA2-7B across various benchmarks demonstrate that models trained on ICON-selected subsets outperform those trained on full datasets, reducing training costs and even surpassing widely used data selection techniques. The paper also analyzes the characteristics of the selected high-contribution samples.

**Critical Evaluation:**

*   **Novelty:**  The core idea of using ICL as a proxy for measuring the contribution of training samples is relatively novel in the context of automated data selection for instruction tuning. Existing methods primarily rely on gradients or manually engineered features. Bypassing the need for explicit gradient computation through ICL, and training a separate selection classifier, does introduce a computationally efficient alternative.

*   **Significance:** The results convincingly demonstrate the effectiveness of ICON. Consistently outperforming models trained on the full dataset and several existing data selection methods suggests a significant improvement in training efficiency and final model performance. The ability to achieve superior results with significantly reduced computational resources (FLOPs) makes ICON a valuable tool. The analysis of high-contribution samples providing insights into the properties of effective training data further enhances the paper's impact.

*   **Strengths:**

    *   **Strong Empirical Results:**  The paper presents extensive experiments on multiple models and datasets, providing solid evidence for ICON's effectiveness.
    *   **Computational Efficiency:**  Avoiding gradient computation and leveraging LoRA for selection training makes ICON significantly more efficient than gradient-based approaches.
    *   **Addresses Limitations of Existing Methods:**  ICON directly measures sample contribution, mitigating the human inductive bias inherent in heuristic-based methods.
    *   **In-depth Analysis:** The paper goes beyond simply presenting results and provides valuable insights into the characteristics of high-contribution samples and the optimal data scale for instruction tuning.

*   **Weaknesses:**

    *   **Dependency on a Good Assessment Set:** The performance of ICON hinges on the quality and representativeness of the assessment set. Constructing this set could be challenging in practice and may introduce a degree of human bias, although the paper addresses this by sampling from multiple diverse sources.
    *   **Limited Generalization Analysis of the Selection classifier:** While the paper shows a robustness test transferring the selection paradigm to different datasets, it would be helpful to know what datasets are required to initialize it, or perform zero-shot transfer.
    *   **The LoRA selection paradigm performance:** While LoRA greatly increases the inference performance for finding the ICON samples, how does this performance impact the overall selection performance when a different base model and a different dataset is introduced?

*   **Potential Influence:** ICON has the potential to significantly influence the field of instruction tuning and LLM training by:

    *   Enabling more efficient use of training data.
    *   Facilitating the training of high-performance models with limited computational resources.
    *   Providing a framework for better understanding the properties of effective training data.

**Score:** 8

**Justification:**

ICON is a strong paper that introduces a novel and effective data selection method for instruction tuning. The use of ICL as a proxy for measuring sample contribution is a creative idea, and the empirical results convincingly demonstrate the benefits of ICON. While there are some limitations related to the dependency on the assessment set and potential for overreliance on a select range, its strengths in performance and computational efficiency make it a notable contribution that addresses significant challenges in the field and unlocks further avenues for improving dataset selection. Overall, I believe ICON is likely to have a substantial impact on the future of instruction tuning.

- **Score**: 8/10

### **[clem:todd: A Framework for the Systematic Benchmarking of LLM-Based Task-Oriented Dialogue System Realisations](http://arxiv.org/abs/2505.05445v1)**
- **Summary**: Here is a concise summary and critical evaluation of the paper "clem: todd: A Framework for the Systematic Benchmarking of LLM-Based Task-Oriented Dialogue System Realisations":

**Summary:**

The paper introduces clem: todd, a novel framework for systematically benchmarking LLM-based Task-Oriented Dialogue (TOD) systems. It adopts a self-play setup, pitting user simulators (Player A) against dialogue systems (Player B), orchestrated by a game master. This framework enables consistent evaluation across various user simulator and dialogue system combinations, facilitating detailed analysis of architectural choices, model sizes, and prompting strategies. The authors showcase clem: todd by re-evaluating existing TOD systems and integrating newly developed ones, providing insights into dialogue performance, task success, and computational cost. They address the limitations of existing TOD evaluation methods by offering a unified setup that mitigates inconsistent metrics and compute settings.

**Critical Evaluation:**

**Strengths:**

*   **Novelty:** The paper addresses a crucial gap in the field by providing a unified framework for evaluating LLM-based TOD systems. Existing evaluations are often fragmented, making cross-system comparisons difficult. Clem: todd offers a more standardized and systematic approach. The use of self-play for evaluating TOD systems is a valuable contribution.

*   **Significance:** Clem: todd enables more reliable and reproducible research in the field of TOD systems, by facilitating detailed benchmarking and providing insights into architectural choices and system interactions. This is particularly significant given the rapid evolution of LLMs and the increasing complexity of dialogue system architectures. The framework also allows exploration into areas of system robustness and generalizability which are often glossed over in individual system design papers. The tool has the potential to standardize evaluation in the field.

*   **Experimental Rigor:** The authors conduct extensive experiments, re-evaluating existing systems and integrating new ones, within their framework. The inclusion of both open-weight and closed-weight LLMs allows for a broad analysis of performance tradeoffs. The clear articulation of the experimental setup and evaluation metrics enhances the reproducibility of the results. The framework is built with integration of new systems and user simulators in mind, suggesting a reusable and adaptable tool for researchers to explore.

**Weaknesses:**

*   **Scope of Evaluation:** While clem: todd provides a valuable framework, the evaluation is largely confined to the MultiWOZ 2.2 dataset with a restriction to booking tasks. While filtering out irrelevant tasks creates a focused evaluation, it also impacts the breadth of use cases assessed in the paper. The new synthetic datasets expand the domain but also introduce the caveat of unrealistic and artificial settings. A broader evaluation across various datasets and task types would further strengthen the framework's generalizability.

*   **Complexity of Systems:** The framework includes both existing and new dialogue system configurations. The modularity of the framework may mean the underlying complexity to the systems or components being integrated, are not fully captured or addressed. This may lead to a situation where a simpler, monolithic system achieves a higher score than it may deserve.

*   **Reliance on LLMs for Evaluation:** The evaluation relies on LLMs (e.g., GPT-4o) for assessing aspects like dialogue quality and naturalness. While LLMs as judges are increasingly common, they are not without biases and limitations. Human evaluation, beyond the Turing test, for aspects like task success could enhance the reliability of the evaluation. The framework also assumes all goals can be assessed without human judgement.

*   **Limitations in the Self-Play Setup:** The reliance on adherence to the Tool Schema, while enforcing format compliance, can lead to premature termination of dialogues and underestimate the capabilities of certain models. The two-player interactions, while a starting point, may not fully capture the complexities of real-world dialogue scenarios. There also seems to be limited user customisability and control in current framework, limiting potential insights into agent behaviours to specific prompt and parameters.

**Justification for Score:**

The paper makes a significant contribution by introducing a much-needed framework for systematically benchmarking LLM-based TOD systems. While the current evaluation is somewhat limited in scope and there are inherent limitations to the self-play set-up, the clem: todd framework establishes a foundation for more reliable and reproducible research in the field. It also fosters research on system interactions and robustness. Considering the novelty, potential impact, and experimental rigor, but also taking into account the limitations in scope and dependence on LLMs for evaluation, a score of 8 is appropriate.

Score: 8

- **Score**: 8/10

### **[DiffusionSfM: Predicting Structure and Motion via Ray Origin and Endpoint Diffusion](http://arxiv.org/abs/2505.05473v1)**
- **Summary**: Here's a summary and critical evaluation of the "DiffusionSfM: Predicting Structure and Motion via Ray Origin and Endpoint Diffusion" paper:

**Summary:**

The paper introduces DiffusionSfM, a novel end-to-end multi-view structure-from-motion (SfM) approach that predicts 3D scene geometry and camera poses directly from multi-view images using a denoising diffusion model.  Instead of the traditional SfM pipeline that separates pairwise reasoning and global optimization, DiffusionSfM unifies these into a single framework by parameterizing scene geometry and cameras as pixel-wise ray origins and endpoints in a global coordinate frame.  The method addresses the challenges of training diffusion models with missing depth data and unbounded scene coordinates by introducing specialized techniques, including GT mask conditioning and parameterizing 3D points in projective space. Experiments on synthetic and real datasets demonstrate that DiffusionSfM outperforms classical and learning-based SfM approaches, while naturally modeling uncertainty.

**Critical Evaluation:**

*   **Novelty:** The paper presents a significant departure from the standard two-stage SfM pipeline. The key novelty lies in:

    *   **End-to-end learning via diffusion:** Directly predicting scene geometry and camera poses using a diffusion model without relying on pairwise matching and global optimization. This unification is a substantial shift.
    *   **Ray origin and endpoint parameterization:**  Representing scene geometry and cameras through pixel-wise ray origins and endpoints is a clever way to couple geometry and pose estimation, enabling joint reasoning. It also provides a more comprehensive representation than just camera poses (as in RayDiffusion [42]) or point clouds (as in DUSt3R [37]).
    *   **Training strategies for diffusion in SfM:** The paper introduces specific techniques to handle incomplete GT data (GT mask conditioning) and unbounded scene coordinates (homogeneous parameterization) for training diffusion models, making the method practical.
    *   **Sparse-to-dense training:** The proposed sparse-to-dense training approach helps to improve convergence and performance by pre-training a sparse model and using it to initialize a dense model. This tackles the computational challenges of training a high-resolution diffusion model from scratch.

*   **Significance:** The potential impact of DiffusionSfM on the field is considerable because:

    *   **Improved Accuracy:** The method demonstrates improved camera pose estimation compared to existing learning-based and classical approaches on standard datasets, suggesting a leap forward in SfM performance. In particular, the camera center accuracy improvement is noteworthy and appears directly linked to the explicit ray origin modeling.
    *   **Uncertainty Modeling:** Leveraging a diffusion model allows for inherent uncertainty modeling, which is important for applications requiring robust SfM in real-world scenarios. The ability to generate multiple plausible scene interpretations (as illustrated by the vase example) is valuable.
    *   **Scalability and Adaptability:** The end-to-end nature simplifies the SfM pipeline and offers potential for greater scalability and adaptation to different sensor modalities (e.g., incorporating LiDAR).
    *   **Alternative Representation:** Provides the community with a novel alternative representation for cameras and geometry, potentially spurring new research directions for multi-view geometry.

*   **Strengths:**

    *   **Comprehensive Evaluation:** Thorough experiments on various datasets.
    *   **Clear Problem Definition and Solutions:** The paper identifies and addresses key challenges related to training diffusion models for SfM.
    *   **Well-Written:** Easy to follow and understand the method's details.
    *   **Ablation Studies:**  Demonstrate the importance of different components of the method (GT mask, homogeneous coordinates).
    *   **Strong Performance:** Outperforms many existing methods for camera center estimation.

*   **Weaknesses:**

    *   **Computational Cost:** Diffusion models are inherently computationally expensive. The iterative denoising process, while offering uncertainty modeling benefits, significantly increases inference time compared to direct regression methods. However, the paper addresses this to some extent via early stopping.
    *   **Pixel-Space Diffusion Limitation:** The paper notes the potential limitation of using a pixel-space diffusion model. Moving to a latent space could improve efficiency and reduce noise in outputs, but is left as future work.
    *   **Still reliant on strong features:** Model uses DINOv2, this could mask other limitations in the approach as DINOv2 is extremely powerful on its own.
    *   **Reliance on pre-computed pairs**: The approach still requires pre-computed pairs of images to iteratively construct the multi-view dataset.
    *   **Limited ablation**: The performance of the sparse model is not clearly discussed. Further analysis would be very insightful.

*   **Potential Influence:** This work has the potential to significantly influence the field of SfM. The end-to-end diffusion approach, combined with the ray origin/endpoint parameterization, provides a strong foundation for future research on:

    *   **Integrating other sensor data:** Combining images with LiDAR or other depth sensors within the diffusion framework.
    *   **Learning expressive latent spaces:**  Developing more efficient latent space diffusion models for SfM.
    *   **Adaptive sampling strategies:**  Improving the scalability of the method by using more efficient sampling techniques for the diffusion process.
    *   **Application in other geometric vision tasks:** Extending the framework to tackle related geometric vision tasks, such as visual SLAM or 3D scene reconstruction.

*   **Score Justification:**

    The paper introduces a novel and significant approach to SfM that has the potential to reshape the field. The end-to-end diffusion approach is a notable departure from traditional methods. The strengths mentioned above are compelling. The weaknesses, while present, are adequately acknowledged and provide directions for future research. The improvement in performance metrics is substantial, and the uncertainty modeling is an added benefit. For these reasons, I assign a score of 8. This reflects the strong novelty, the potential impact, and the solid empirical validation, balanced by the computational cost and some limitations that present opportunities for future work.

**Score: 8**

- **Score**: 8/10

### **[3D Scene Generation: A Survey](http://arxiv.org/abs/2505.05474v1)**
- **Summary**: Here's a summary and critical evaluation of the provided paper:

**Summary:**

The paper presents a comprehensive survey of 3D scene generation techniques. It categorizes existing methods into four paradigms: procedural generation, neural 3D-based generation, image-based generation, and video-based generation. The survey analyzes the technical foundations, trade-offs, and representative results of each approach. It also reviews commonly used datasets, evaluation protocols, and downstream applications. The paper concludes by discussing key challenges in generation capacity, 3D representation, data and annotations, and evaluation. It also outlines promising future directions, including higher fidelity, physics-aware and interactive generation, and unified perception-generation models.  The paper maintains an up-to-date project page to track ongoing developments.

**Critical Evaluation:**

*   **Novelty:** The paper's novelty lies primarily in its comprehensive coverage and systematic organization of the rapidly evolving field of 3D scene generation. While individual components (e.g., GANs, NeRFs, diffusion models) are not novel, the holistic view and structured categorization across the four paradigms contribute meaningfully to the field. It addresses a gap in existing literature by providing a consolidated overview, whereas prior surveys focused on narrow subdomains or specific aspects of 3D or 4D generation. The inclusion of video-based generation and dynamic scene representations is a valuable addition that reflects recent trends.

*   **Significance:** The paper is significant for several reasons. First, it provides a valuable resource for researchers entering the field of 3D scene generation. It serves as a roadmap, guiding readers through the diverse approaches, datasets, and evaluation metrics. Second, it highlights the key challenges and future directions, which can stimulate future research and innovation. The identification of areas such as physics-aware generation and unified perception-generation models is particularly valuable. Third, it provides a structured taxonomy that is helpful for understanding the landscape and comparing different methods. The inclusion of downstream applications demonstrates the practical relevance of this area of research.

*   **Strengths:**

    *   **Comprehensive Coverage:** The survey covers a wide range of techniques, including both traditional and recent methods.
    *   **Systematic Organization:** The classification into four paradigms is clear and well-defined.
    *   **Balanced Analysis:** The paper provides a fair and balanced assessment of the strengths and weaknesses of each approach.
    *   **Future Directions:** The discussion of challenges and future directions is insightful and thought-provoking.
    *   **Up-to-date:** The presence of a continually updated project page enhances the long-term value.

*   **Weaknesses:**

    *   **Depth of Analysis:** While broad, the survey can sometimes lack in-depth analysis of the mathematical or algorithmic details of specific methods.
    *   **Subjectivity:** The categorization might be open to interpretation in some cases. While the chosen categorization provides a clear structure, some papers might fit into multiple categories.
    *   **Evaluation Metrics:** The discussion of evaluation metrics could be more critical, highlighting the limitations of current metrics and advocating for more robust evaluation strategies.

*   **Potential Influence:** This survey is likely to be highly influential in the 3D scene generation field. It provides a necessary overview, aiding both established researchers and newcomers. It should contribute towards a more unified understanding of the landscape and encourage research along the outlined future directions.

**Score: 8**

**Rationale:**

The paper's comprehensive scope and structured organization are its major strengths, warranting a high score. The detailed overview of datasets, evaluation metrics, and applications further enhances its value. The clear articulation of challenges and future research directions offers a helpful guide to the community. While the survey doesn't delve deeply into every technical detail, its breadth of coverage and its critical analysis of different approaches make it a significant contribution. The weaknesses regarding depth and the inherent subjectivity of categorization are relatively minor compared to the overall value provided. The constantly updated project page solidifies its importance and long-term impact.

- **Score**: 8/10

## Other Papers
### **[The Aloe Family Recipe for Open and Specialized Healthcare LLMs](http://arxiv.org/abs/2505.04388v1)**
### **[Large Means Left: Political Bias in Large Language Models Increases with Their Number of Parameters](http://arxiv.org/abs/2505.04393v1)**
### **[YABLoCo: Yet Another Benchmark for Long Context Code Generation](http://arxiv.org/abs/2505.04406v1)**
### **[OBLIVIATE: Robust and Practical Machine Unlearning for Large Language Models](http://arxiv.org/abs/2505.04416v1)**
### **[Localized Diffusion Models for High Dimensional Distributions Generation](http://arxiv.org/abs/2505.04417v1)**
### **[LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders](http://arxiv.org/abs/2505.04421v1)**
### **[Theoretical Guarantees for LT-TTD: A Unified Transformer-based Architecture for Two-Level Ranking Systems](http://arxiv.org/abs/2505.04434v1)**
### **[Towards Effectively Leveraging Execution Traces for Program Repair with Code LLMs](http://arxiv.org/abs/2505.04441v1)**
### **[M2Rec: Multi-scale Mamba for Efficient Sequential Recommendation](http://arxiv.org/abs/2505.04445v1)**
### **[Miipher-2: A Universal Speech Restoration Model for Million-Hour Scale Data Restoration](http://arxiv.org/abs/2505.04457v1)**
### **[Spectral and Temporal Denoising for Differentially Private Optimization](http://arxiv.org/abs/2505.04468v1)**
### **[TrajEvo: Designing Trajectory Prediction Heuristics via LLM-driven Evolution](http://arxiv.org/abs/2505.04480v1)**
### **[CAD-Llama: Leveraging Large Language Models for Computer-Aided Design Parametric 3D Model Generation](http://arxiv.org/abs/2505.04481v1)**
### **[Efficient Flow Matching using Latent Variables](http://arxiv.org/abs/2505.04486v1)**
### **[Defining and Quantifying Creative Behavior in Popular Image Generators](http://arxiv.org/abs/2505.04497v2)**
### **[Pangu Ultra MoE: How to Train Your Big MoE on Ascend NPUs](http://arxiv.org/abs/2505.04519v1)**
### **[Comparative Analysis of Carbon Footprint in Manual vs. LLM-Assisted Code Development](http://arxiv.org/abs/2505.04521v1)**
### **[Text2CT: Towards 3D CT Volume Generation from Free-text Descriptions Using Diffusion Model](http://arxiv.org/abs/2505.04522v1)**
### **[Fight Fire with Fire: Defending Against Malicious RL Fine-Tuning via Reward Neutralization](http://arxiv.org/abs/2505.04578v1)**
### **[SlideItRight: Using AI to Find Relevant Slides and Provide Feedback for Open-Ended Questions](http://arxiv.org/abs/2505.04584v1)**
### **[ZeroSearch: Incentivize the Search Capability of LLMs without Searching](http://arxiv.org/abs/2505.04588v1)**
### **[MonoCoP: Chain-of-Prediction for Monocular 3D Object Detection](http://arxiv.org/abs/2505.04594v2)**
### **[OmniGIRL: A Multilingual and Multimodal Benchmark for GitHub Issue Resolution](http://arxiv.org/abs/2505.04606v1)**
### **[Score Distillation Sampling for Audio: Source Separation, Synthesis, and Beyond](http://arxiv.org/abs/2505.04621v1)**
### **[Retrieval Augmented Generation Evaluation for Health Documents](http://arxiv.org/abs/2505.04680v1)**
### **[Lay-Your-Scene: Natural Scene Layout Generation with Diffusion Transformers](http://arxiv.org/abs/2505.04718v1)**
### **[SOAEsV2-7B/72B: Full-Pipeline Optimization for State-Owned Enterprise LLMs via Continual Pre-Training, Domain-Progressive SFT and Distillation-Enhanced Speculative Decoding](http://arxiv.org/abs/2505.04723v1)**
### **[QBD-RankedDataGen: Generating Custom Ranked Datasets for Improving Query-By-Document Search Using LLM-Reranking with Reduced Human Effort](http://arxiv.org/abs/2505.04732v1)**
### **[The Promise and Limits of LLMs in Constructing Proofs and Hints for Logic Problems in Intelligent Tutoring Systems](http://arxiv.org/abs/2505.04736v1)**
### **[Hyb-KAN ViT: Hybrid Kolmogorov-Arnold Networks Augmented Vision Transformer](http://arxiv.org/abs/2505.04740v1)**
### **[A Proposal for Evaluating the Operational Risk for ChatBots based on Large Language Models](http://arxiv.org/abs/2505.04784v1)**
### **[Safeguard-by-Development: A Privacy-Enhanced Development Paradigm for Multi-Agent Collaboration Systems](http://arxiv.org/abs/2505.04799v1)**
### **[Red Teaming the Mind of the Machine: A Systematic Evaluation of Prompt Injection and Jailbreak Vulnerabilities in LLMs](http://arxiv.org/abs/2505.04806v1)**
### **[Steerable Scene Generation with Post Training and Inference-Time Search](http://arxiv.org/abs/2505.04831v1)**
### **[Large Language Models are Autonomous Cyber Defenders](http://arxiv.org/abs/2505.04843v1)**
### **[Osiris: A Lightweight Open-Source Hallucination Detection System](http://arxiv.org/abs/2505.04844v1)**
### **[HiPerRAG: High-Performance Retrieval Augmented Generation for Scientific Insights](http://arxiv.org/abs/2505.04846v1)**
### **[CRAFT: Cultural Russian-Oriented Dataset Adaptation for Focused Text-to-Image Generation](http://arxiv.org/abs/2505.04851v1)**
### **[D-CODA: Diffusion for Coordinated Dual-Arm Data Augmentation](http://arxiv.org/abs/2505.04860v1)**
### **[From First Draft to Final Insight: A Multi-Agent Approach for Feedback Generation](http://arxiv.org/abs/2505.04869v1)**
### **[GroverGPT-2: Simulating Grover's Algorithm via Chain-of-Thought Reasoning and Quantum-Native Tokenization](http://arxiv.org/abs/2505.04880v1)**
### **[ConCISE: Confidence-guided Compression in Step-by-step Efficient Reasoning](http://arxiv.org/abs/2505.04881v1)**
### **[SpatialPrompting: Keyframe-driven Zero-Shot Spatial Reasoning with Off-the-Shelf Multimodal Large Language Models](http://arxiv.org/abs/2505.04911v1)**
### **[GlyphMastero: A Glyph Encoder for High-Fidelity Scene Text Editing](http://arxiv.org/abs/2505.04915v1)**
### **[Perception, Reason, Think, and Plan: A Survey on Large Multimodal Reasoning Models](http://arxiv.org/abs/2505.04921v1)**
### **[Accurate and Fast Channel Estimation for Fluid Antenna Systems with Diffusion Models](http://arxiv.org/abs/2505.04930v1)**
### **[Prompt-Based LLMs for Position Bias-Aware Reranking in Personalized Recommendations](http://arxiv.org/abs/2505.04948v1)**
### **[Position: Epistemic Artificial Intelligence is Essential for Machine Learning Models to Know When They Do Not Know](http://arxiv.org/abs/2505.04950v1)**
### **[Chain-of-Thought Tokens are Computer Program Variables](http://arxiv.org/abs/2505.04955v1)**
### **[Graffe: Graph Representation Learning via Diffusion Probabilistic Models](http://arxiv.org/abs/2505.04956v1)**
### **[Learning Item Representations Directly from Multimodal Features for Effective Recommendation](http://arxiv.org/abs/2505.04960v1)**
### **[DenseGrounding: Improving Dense Language-Vision Semantics for Ego-Centric 3D Visual Grounding](http://arxiv.org/abs/2505.04965v1)**
### **[ReAlign: Bilingual Text-to-Motion Generation via Step-Aware Reward-Guided Alignment](http://arxiv.org/abs/2505.04974v1)**
### **[ChainMarks: Securing DNN Watermark with Cryptographic Chain](http://arxiv.org/abs/2505.04977v1)**
### **[Latent Preference Coding: Aligning Large Language Models via Discrete Latent Codes](http://arxiv.org/abs/2505.04993v1)**
### **[Rethinking Invariance in In-context Learning](http://arxiv.org/abs/2505.04994v1)**
### **[Inter-Diffusion Generation Model of Speakers and Listeners for Effective Communication](http://arxiv.org/abs/2505.04996v1)**
### **[The Pitfalls of Growing Group Complexity: LLMs and Social Choice-Based Aggregation for Group Recommendations](http://arxiv.org/abs/2505.05016v1)**
### **[Scalable Multi-Stage Influence Function for Large Language Models via Eigenvalue-Corrected Kronecker-Factored Parameterization](http://arxiv.org/abs/2505.05017v1)**
### **[SOAP: Style-Omniscient Animatable Portraits](http://arxiv.org/abs/2505.05022v1)**
### **[LSRP: A Leader-Subordinate Retrieval Framework for Privacy-Preserving Cloud-Device Collaboration](http://arxiv.org/abs/2505.05031v1)**
### **[Divide-and-Conquer: Cold-Start Bundle Recommendation via Mixture of Diffusion Experts](http://arxiv.org/abs/2505.05035v1)**
### **[Towards Mitigating API Hallucination in Code Generated by LLMs with Hierarchical Dependency Aware](http://arxiv.org/abs/2505.05057v1)**
### **[CodeMixBench: Evaluating Large Language Models on Code Generation with Code-Mixed Prompts](http://arxiv.org/abs/2505.05063v1)**
### **[WaterDrum: Watermarking for Data-centric Unlearning Metric](http://arxiv.org/abs/2505.05064v1)**
### **[Performance Evaluation of Large Language Models in Bangla Consumer Health Query Summarization](http://arxiv.org/abs/2505.05070v1)**
### **[PIDiff: Image Customization for Personalized Identities with Diffusion Models](http://arxiv.org/abs/2505.05081v1)**
### **[ItDPDM: Information-Theoretic Discrete Poisson Diffusion Model](http://arxiv.org/abs/2505.05082v1)**
### **[Reliably Bounding False Positives: A Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction](http://arxiv.org/abs/2505.05084v1)**
### **[X-Driver: Explainable Autonomous Driving with Vision-Language Models](http://arxiv.org/abs/2505.05098v1)**
### **[MDE-Edit: Masked Dual-Editing for Multi-Object Image Editing via Diffusion Models](http://arxiv.org/abs/2505.05101v1)**
### **[A Weighted Byzantine Fault Tolerance Consensus Driven Trusted Multiple Large Language Models Network](http://arxiv.org/abs/2505.05103v1)**
### **[Multi-agent Embodied AI: Advances and Future Directions](http://arxiv.org/abs/2505.05108v1)**
### **[Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders](http://arxiv.org/abs/2505.05111v1)**
### **[MDAA-Diff: CT-Guided Multi-Dose Adaptive Attention Diffusion Model for PET Denoising](http://arxiv.org/abs/2505.05112v1)**
### **[Enhancing Text2Cypher with Schema Filtering](http://arxiv.org/abs/2505.05118v1)**
### **[Text2Cypher: Data Pruning using Hard Example Selection](http://arxiv.org/abs/2505.05122v1)**
### **[Research on Anomaly Detection Methods Based on Diffusion Models](http://arxiv.org/abs/2505.05137v1)**
### **[Overcoming Dimensional Factorization Limits in Discrete Diffusion Models through Quantum Joint Distribution Learning](http://arxiv.org/abs/2505.05151v1)**
### **[FedTDP: A Privacy-Preserving and Unified Framework for Trajectory Data Preparation via Federated Learning](http://arxiv.org/abs/2505.05155v1)**
### **[MARK: Memory Augmented Refinement of Knowledge](http://arxiv.org/abs/2505.05177v1)**
### **[Stochastic Variational Propagation: Local, Scalable and Efficient Alternative to Backpropagation](http://arxiv.org/abs/2505.05181v1)**
### **[Revealing Weaknesses in Text Watermarking Through Self-Information Rewrite Attacks](http://arxiv.org/abs/2505.05190v1)**
### **[EAM: Enhancing Anything with Diffusion Transformers for Blind Super-Resolution](http://arxiv.org/abs/2505.05209v1)**
### **[Diffusion Model Quantization: A Review](http://arxiv.org/abs/2505.05215v1)**
### **[Normalize Everything: A Preconditioned Magnitude-Preserving Architecture for Diffusion-Based Speech Enhancement](http://arxiv.org/abs/2505.05216v1)**
### **[QualBench: Benchmarking Chinese LLMs with Localized Professional Qualifications for Vertical Domain Evaluation](http://arxiv.org/abs/2505.05225v1)**
### **[ChemRxivQuest: A Curated Chemistry Question-Answer Database Extracted from ChemRxiv Preprints](http://arxiv.org/abs/2505.05232v1)**
### **[Latte: Transfering LLMs` Latent-level Knowledge for Few-shot Tabular Learning](http://arxiv.org/abs/2505.05237v1)**
### **[T-T: Table Transformer for Tagging-based Aspect Sentiment Triplet Extraction](http://arxiv.org/abs/2505.05271v1)**
### **[Software Development Life Cycle Perspective: A Survey of Benchmarks for CodeLLMs and Agents](http://arxiv.org/abs/2505.05283v1)**
### **[HEXGEN-TEXT2SQL: Optimizing LLM Inference Request Scheduling for Agentic Text-to-SQL Workflow](http://arxiv.org/abs/2505.05286v1)**
### **[Benchmarking Ophthalmology Foundation Models for Clinically Significant Age Macular Degeneration Detection](http://arxiv.org/abs/2505.05291v1)**
### **[Toward Reasonable Parrots: Why Large Language Models Should Argue with Us by Design](http://arxiv.org/abs/2505.05298v1)**
### **[ICon: In-Context Contribution for Automatic Data Selection](http://arxiv.org/abs/2505.05327v1)**
### **[Denoising Diffusion Probabilistic Models for Coastal Inundation Forecasting](http://arxiv.org/abs/2505.05381v1)**
### **[PillarMamba: Learning Local-Global Context for Roadside Point Cloud via Hybrid State Space Model](http://arxiv.org/abs/2505.05397v1)**
### **[Frame In, Frame Out: Do LLMs Generate More Biased News Headlines than Humans?](http://arxiv.org/abs/2505.05406v1)**
### **[Crosslingual Reasoning through Test-Time Scaling](http://arxiv.org/abs/2505.05408v1)**
### **[Hide & Seek: Transformer Symmetries Obscure Sharpness & Riemannian Geometry Finds It](http://arxiv.org/abs/2505.05409v1)**
### **[Reasoning Models Don't Always Say What They Think](http://arxiv.org/abs/2505.05410v1)**
### **[TokLIP: Marry Visual Tokens to CLIP for Multimodal Comprehension and Generation](http://arxiv.org/abs/2505.05422v1)**
### **[TransProQA: an LLM-based literary Translation evaluation metric with Professional Question Answering](http://arxiv.org/abs/2505.05423v1)**
### **[Ultra-FineWeb: Efficient Data Filtering and Verification for High-Quality LLM Training Data](http://arxiv.org/abs/2505.05427v1)**
### **[EcoAgent: An Efficient Edge-Cloud Collaborative Multi-Agent Framework for Mobile Automation](http://arxiv.org/abs/2505.05440v1)**
### **[clem:todd: A Framework for the Systematic Benchmarking of LLM-Based Task-Oriented Dialogue System Realisations](http://arxiv.org/abs/2505.05445v1)**
### **[Conversational Process Model Redesign](http://arxiv.org/abs/2505.05453v1)**
### **[UKElectionNarratives: A Dataset of Misleading Narratives Surrounding Recent UK General Elections](http://arxiv.org/abs/2505.05459v1)**
### **[Bring Reason to Vision: Understanding Perception and Reasoning through Model Merging](http://arxiv.org/abs/2505.05464v1)**
### **[ComPO: Preference Alignment via Comparison Oracles](http://arxiv.org/abs/2505.05465v1)**
### **[Mogao: An Omni Foundation Model for Interleaved Multi-Modal Generation](http://arxiv.org/abs/2505.05472v1)**
### **[DiffusionSfM: Predicting Structure and Motion via Ray Origin and Endpoint Diffusion](http://arxiv.org/abs/2505.05473v1)**
### **[3D Scene Generation: A Survey](http://arxiv.org/abs/2505.05474v1)**
### **[SVAD: From Single Image to 3D Avatar via Synthetic Data Generation with Video Diffusion and Data Augmentation](http://arxiv.org/abs/2505.05475v1)**
