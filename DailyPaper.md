# The Latest Daily Papers - Date: 2025-04-12
## Highlight Papers
### **[PR-Attack: Coordinated Prompt-RAG Attacks on Retrieval-Augmented Generation in Large Language Models via Bilevel Optimization](http://arxiv.org/abs/2504.07717v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper "PR-Attack: Coordinated Prompt-RAG Attacks on Retrieval-Augmented Generation in Large Language Models via Bilevel Optimization" introduces a novel attack method against Retrieval-Augmented Generation (RAG) systems used with Large Language Models (LLMs).  The attack, called PR-Attack, strategically injects a small number of poisoned texts into the RAG's knowledge base while simultaneously embedding a backdoor trigger within the prompt provided to the LLM.  When the trigger is activated (e.g., during sensitive periods like immediately after an earthquake), the LLM is forced to generate a pre-designed, malicious response to a targeted query, even if the correct information exists in the knowledge base. The authors formulate this attack as a bilevel optimization problem and propose an alternating optimization method to generate effective poisoned texts and triggers. Experiments across diverse LLMs and datasets are conducted to demonstrate that PR-Attack achieves high attack success rates with limited poisoned texts and improved stealthiness compared to existing attack methods.

**Critical Evaluation:**

* **Novelty:** The paper's core novelty lies in the coordinated attack paradigm.  Prior work has typically focused on either attacking the knowledge base of RAG systems *or* crafting adversarial prompts for LLMs.  By simultaneously manipulating both the knowledge base and the prompt, the authors achieve a more synergistic and potent attack. The formulation of the problem as a bilevel optimization and the proposed alternating optimization solution are also a novel contributions.

* **Significance:** The work highlights a critical vulnerability in RAG systems, especially when used in sensitive contexts.  The ability to trigger specific, malicious responses during critical periods (exploiting the Social Amplification of Risk Framework) is a particularly concerning finding. This has implications for the deployment of RAG systems in areas like news dissemination, crisis management, or financial advising where misinformation can have severe consequences.  The improved stealthiness is also significant; traditional defenses that rely on detecting consistently wrong answers are less effective against PR-Attack.

* **Strengths:**
    * **Well-defined problem:** The attack scenario and threat model are clearly articulated.
    * **Principled approach:** The formulation of the attack as a bilevel optimization problem is a strong point, allowing for a systematic and potentially more effective generation of poisoned texts and triggers. The alternating optimization method provides a solid framework for solving this complex problem.
    * **Empirical validation:**  The extensive experiments across various LLMs and datasets provide strong evidence for the effectiveness and robustness of PR-Attack.  The comparison against existing methods convincingly demonstrates the improvements in attack success rate and stealthiness.
    * **Clear writing and organization:** The paper is generally well-written and easy to follow.

* **Weaknesses:**
    * **Limited Real-World Evaluation:** While the experimental setup is robust, it primarily uses standard datasets. Evaluating the attack in a more realistic scenario, for example, by injecting poisoned texts into a live RAG system with user interactions, would further strengthen the paper's impact.
    * **Computational Cost:** The bilevel optimization approach could be computationally expensive, especially for large knowledge bases. While the paper analyzes the complexity, a more detailed discussion of the practical computational limitations would be beneficial. This needs further analysis in the context of real-world scalability.
    * **Defense Strategies:** The paper focuses primarily on the attack and does not extensively explore potential defense mechanisms. While the increased stealth makes detection harder, exploring possible defense strategies (even preliminary ones) would make the paper more impactful. The paper makes a good point of illustrating the significance of these kinds of attacks in real-world scenarios, like generating chaos after an earthquake in a sensitive period.

* **Overall Impact:** The paper provides a significant contribution to the field of LLM security by identifying and demonstrating a novel, effective, and stealthy attack against RAG systems. It highlights the importance of considering coordinated attacks and raises serious concerns about the deployment of RAG in sensitive applications. The formulation as a bilevel optimization problem opens doors for future research in developing more robust defenses based on similar optimization frameworks.

**Score: 8**

**Rationale:** The paper presents a genuinely novel and significant attack vector with strong empirical evidence. While the computational cost and a lack of comprehensive defense analysis are minor weaknesses, the paper's contributions significantly advance our understanding of the vulnerabilities of RAG systems and their potential for misuse. It also sets a promising direction for future research on both attack and defense strategies.

- **Score**: 8/10

### **[Zero-Shot Cross-Domain Code Search without Fine-Tuning](http://arxiv.org/abs/2504.07740v1)**
- **Summary**: Here is a summary and critical evaluation of the paper:

**Summary:**

The paper presents CodeBridge, a novel zero-shot, fine-tuning-free approach for cross-domain code search. It addresses the challenge of applying pre-trained language models (PLMs) to new code domains where data for fine-tuning is scarce.  CodeBridge decomposes the query-code matching process into two simpler tasks: query-comment matching and code-code matching.  It leverages Large Language Models (LLMs) with zero-shot prompting to generate comments for code snippets and code for queries, effectively augmenting the available information.  The approach combines three matching schemas – query-code, query-comment, and generated code-code – using a sampling-based fusion method for ranking search results.  Experimental results demonstrate that CodeBridge outperforms existing PLM-based code search methods and achieves results comparable to or better than RAPID, a data-hungry fine-tuning method.

**Critical Evaluation:**

*Novelty:* The paper's core novelty lies in the decomposition of the code search problem and the creative use of LLMs to bridge the domain gap without fine-tuning. While query-comment and code-code matching are not entirely new concepts in information retrieval, their synergistic combination within a zero-shot, cross-domain code search context and their augmentation via LLMs is a valuable contribution.  The sampling-based fusion strategy, although not fundamentally groundbreaking, is well-motivated and contributes to the overall effectiveness. The careful empirical analysis of the different matching strategies contributes significantly. The method is zero shot and free from any domain specific data.

*Significance:* The significance of this work stems from its ability to address a practical limitation of current PLM-based code search methods: the need for extensive fine-tuning in new domains.  CodeBridge offers a practical alternative, potentially enabling broader adoption of code search in resource-constrained settings. Outperforming data intensive methods like RAPID is a significant improvement.

*Strengths:*
    *   Well-defined problem and clear motivation.
    *   The decomposition strategy is logical and effective.
    *   Zero-shot nature makes it widely applicable.
    *   Thorough experimental evaluation comparing CodeBridge to strong baselines across multiple datasets.
    *   The ablation study and sensitivity analysis provide valuable insights into the approach's components and parameters.
    *   The analysis of different matching schemas and error modes is convincing and adds to the understanding of how and why CodeBridge works.
    *   Addresses data leakage issues.
    *   Includes a thorough computational analysis.

*Weaknesses:*
    * The sampling strategy could be improved.
    *   Relies on LLMs which are computationally expensive for large codebases.
    *   The improvement is significant but the approach may struggle with more complex or nuanced queries.
    *  The method could be improved with few shot domain knowledge.
    *  It would be more convincing to evaluate the strategy on code that is newly created and unseen.

*Potential Influence:*  This work has the potential to influence the development of more practical and adaptable code search tools.  The approach could inspire further research into zero-shot transfer learning for code and the exploitation of LLMs for data augmentation and domain adaptation.

**Score: 8**

*Justification:* The paper presents a novel and significant contribution to the field of code search. The zero-shot, fine-tuning-free approach addresses a real-world limitation of existing methods and provides a practical solution. While the individual components are not revolutionary, their integration and application to this problem are well-executed and lead to compelling results. The thorough experimental analysis adds to the credibility and impact of the work. The limited number of LLMs tested, along with LLMs being expensive to use, and reliance on the power of the LLM to generate decent code/comments restrict the paper from reaching the higher score. Overall, the paper makes a valuable contribution.

- **Score**: 8/10

### **[Efficient Tuning of Large Language Models for Knowledge-Grounded Dialogue Generation](http://arxiv.org/abs/2504.07754v1)**
- **Summary**: Here is a summary and critical evaluation of the paper "Efficient Tuning of Large Language Models for Knowledge-Grounded Dialogue Generation":

**Summary:**

The paper addresses the challenge of efficiently fine-tuning Large Language Models (LLMs) for knowledge-grounded dialogue generation. It proposes KEDiT, a method that consists of two phases:
1.  **Knowledge Compression:** An information bottleneck approach is used to compress retrieved knowledge into learnable parameters, retaining essential information while minimizing computational overhead. This is done using a BERT encoder and a Q-Former. An alignment loss is introduced to align these compressed vectors with the LLM's internal representations.
2.  **Knowledge Integration:** A lightweight knowledge-aware adapter (KA-Adapter) is inserted into the LLM architecture to integrate the compressed knowledge vectors during fine-tuning.  This adapter is a combination of knowledge aware attention mechanisms (KA-Attn) and knowledge-aware feed-forward networks (KA-FFN). It utilizes a gating mechanism to control the influence of the compressed knowledge.

The method requires fine-tuning less than 2% of the model parameters. The authors validate KEDiT on the Wizard of Wikipedia dataset and a new dataset they construct called PubMed-Dialog. Experimental results show that KEDiT outperforms baseline models in automatic evaluations, LLM-based evaluations, and human evaluations, particularly in generating contextually relevant and informative responses, and handling domain-specific knowledge.  The paper introduces a new dataset, PubMed-Dialog, to specifically evaluate models on up-to-date biomedical information.

**Critical Evaluation:**

*   **Novelty:** The combination of knowledge compression via an information bottleneck and lightweight adapter fine-tuning is a significant contribution. Existing work on retrieval augmented generation uses either large amounts of compute to train models end to end, or freezes weights using in-context learning. Compressing the knowledge, while maintaining key information, and integrating that with small, gated adapters is a promising approach.

*   **Significance:** Efficiently incorporating external knowledge into LLMs is critical for many real-world applications, especially in domains like medicine where information is constantly evolving. KEDiT offers a scalable and adaptable solution, potentially lowering the barrier to deploying knowledge-grounded dialogue systems. The introduction of the PubMed-Dialog dataset is also a valuable contribution, addressing the need for domain-specific benchmarks.

*   **Strengths:**
    *   The method is computationally efficient, addressing a key limitation of existing approaches.
    *   Experimental results are comprehensive across multiple datasets and evaluation metrics.
    *   Ablation studies demonstrate the importance of each component of KEDiT.
    *   The paper includes an analysis of generalization across different LLM architectures.
    *   The new PubMed-Dialog dataset is a valuable addition to the community.

*   **Weaknesses:**
    *   The performance gains on the Wizard of Wikipedia dataset are relatively modest compared to the PubMed-Dialog dataset. The authors attribute this to the fact that predefined gold knowledge is not readily available in the PubMed-Dialog set. However, more analysis is needed to fully understand the differences. KEDiT performs slightly lower than other models on the F1 and BLEU scores of the Wizard of Wikipedia unseen dataset.
    *   The paper could benefit from a more thorough comparison to other parameter-efficient fine-tuning methods specifically designed for knowledge incorporation (if any exist).
    *   The system's reliance on a retriever (TF-IDF in this case) means the quality of the generation is highly dependent on the relevance of the retrieved content. A deeper analysis of the relationship between retrieval accuracy and generation performance would be valuable.

*   **Potential Influence:** KEDiT has the potential to influence future research on knowledge-grounded dialogue generation by:

    *   Promoting the use of knowledge compression techniques to improve efficiency.
    *   Encouraging the development of lightweight adapter architectures for knowledge integration.
    *   Providing a strong baseline for future work on the PubMed-Dialog dataset.

* **Limitations**:
    * There is a high reliance on retrieved content quality and may be susceptible to issues when retrieval is inaccurate.
    * The proposed method is computationally efficient, but there could be efficiency limitations when there is a high volume of requests.
    * KEDiT may not capture nuances or biases in the PubMed-Dialog dataset that can have an influence in evaluation metrics.

**Justification of Score:**

KEDiT presents a novel and significant contribution to the field of knowledge-grounded dialogue generation. It addresses a critical need for efficient and adaptable methods to incorporate external knowledge into LLMs, and its comprehensive experimental evaluation demonstrates its effectiveness. It provides a viable alternative to end-to-end training for retrieval-augmented generation by leveraging small, efficient gated adapters. The new dataset, PubMed-Dialog, addresses a gap in the availability of domain-specific benchmarks. While there are some limitations in the performance gains on the Wizard of Wikipedia dataset and the lack of direct comparison to other knowledge-aware PEFT methods, the overall strengths of the paper outweigh these weaknesses.

Score: 8

- **Score**: 8/10

### **[2D-Curri-DPO: Two-Dimensional Curriculum Learning for Direct Preference Optimization](http://arxiv.org/abs/2504.07856v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces 2D-Curri-DPO, a novel framework for aligning large language models (LLMs) with human preferences using a two-dimensional curriculum learning approach.  The framework extends the existing Curriculum-DPO (Curri-DPO) by considering both Prompt Complexity (PC) and Pairwise Distinguishability (PD) when organizing the training data. The authors propose a method for quantifying PC based on the perplexity variance of a reference model's responses. They define a curriculum strategy space, incorporating strategies for traversing the 2D PC/PD grid, and use a KL-divergence-based adaptive mechanism to update the reference model during training, promoting stability. Experimental results demonstrate that 2D-Curri-DPO outperforms standard DPO and prior curriculum methods on several benchmarks, particularly in challenging scenarios.

**Critical Evaluation:**

*   **Novelty:** The primary novelty of the paper lies in the introduction of the two-dimensional curriculum, considering both prompt complexity (PC) *and* pairwise distinguishability (PD). While Curri-DPO previously focused solely on PD, the 2D approach acknowledges and addresses the complexity of the prompt itself as a critical factor in successful alignment.  The proposal to quantify Prompt Complexity based on response generation uncertainty is also a novel contribution. Defining and systematically investigating curriculum strategies within this 2D space adds further value. Finally, using a KL adaptive dynamic reference model is a useful and well designed stability mechanism.
*   **Significance:**  The paper has significant implications for LLM alignment. By demonstrating the importance of considering both prompt and response factors, it sheds light on limitations of existing alignment strategies.  The superior performance of 2D-Curri-DPO, particularly on demanding tasks, suggests that this approach can lead to more robust and reliable LLMs. The explicit strategy space provides a practical framework for adapting alignment procedures to specific tasks and data characteristics. The ablation studies and model behavior analyses strengthen the claims and offer valuable insights into the underlying mechanisms. The code availability helps with reproducibility.
*   **Strengths:**
    *   **Well-defined problem:** The paper clearly identifies and articulates the limitations of existing single-dimensional curriculum learning approaches.
    *   **Principled approach:** The proposed framework is grounded in sound theoretical motivations and methodological rigor.
    *   **Comprehensive experiments:**  The authors conduct thorough experiments on diverse benchmarks, including ablation studies and behavior analyses, to validate their claims.
    *   **Clear and insightful results:**  The results convincingly demonstrate the superior performance of 2D-Curri-DPO and provide valuable insights into the relative strengths of different curriculum strategies.
    *   **Robust design:**  The KL-divergence-based update mechanism promotes training stability.
*   **Weaknesses:**
    *   The method for PC quantification, while novel, relies on a reference model. The quality of this model will inevitably impact the PC measurement.  A more robust, model-agnostic approach could be beneficial.  It would be useful if the analysis section demonstrated the impact the selection of the reference model has on the performance.
    *   The reliance on GPT-4 scores as "ground truth" for evaluating model performance is a potential limitation. While GPT-4 is strong, its biases may influence outcomes. A more diverse set of evaluators or a focus on human evaluations would further strengthen the work.
    *   While the paper explores several curriculum strategies, the selection is somewhat limited. Investigating additional strategies and developing adaptive mechanisms for automatically selecting the best strategy would be a valuable extension.
    *   Some claims are not convincingly demonstrated. For example the justification that the reference signal is noisy when using an "easy" PD is not clearly illustrated, and the link to better reasoning capabilities is somewhat tenuous and anecdotal.
    *   The paper could improve by discussing the computational overhead of this method compared to single dimension curricula.
*   **Potential influence:** The paper is likely to have a significant influence on the field. It introduces a valuable new perspective on curriculum learning for LLM alignment, with demonstrable practical benefits. The insights into the interaction between prompt complexity and preference distinguishability will inform future research in this area.  The proposed framework also provides a template for developing more sophisticated alignment strategies.

**Score:** 8

**Justification:**  The paper presents a novel and well-executed extension to curriculum learning for LLM alignment.  The 2D curriculum, method for PC quantification, and experimental results are strong contributions. While there are some limitations regarding the reliance on GPT-4, and some unclear claims, the paper is novel, significant, and of high quality, well justifying the score.

- **Score**: 8/10

### **[Robust Hallucination Detection in LLMs via Adaptive Token Selection](http://arxiv.org/abs/2504.07863v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper "Robust Hallucination Detection in LLMs via Adaptive Token Selection" introduces HaMI, a novel approach for detecting hallucinations in Large Language Models (LLMs).  HaMI addresses the problem of inconsistent performance in existing hallucination detection methods, which often rely on predetermined tokens and struggle with free-form generations of varying lengths.  HaMI reformulates hallucination detection as a Multiple Instance Learning (MIL) problem, enabling adaptive selection and learning of critical tokens most indicative of hallucinations.  It also incorporates uncertainty measurements to enhance the representation space. The authors demonstrate through comprehensive experiments on four hallucination benchmarks that HaMI significantly outperforms existing state-of-the-art approaches.

**Critical Evaluation:**

* **Novelty:**  The key novelty of the paper lies in its MIL-based approach to hallucination detection, combined with adaptive token selection. This is a significant departure from existing methods that rely on fixed token positions or external resources. The joint optimization of token selection and hallucination detection is a strong contribution. The addition of uncertainty measurements is less novel in itself, as uncertainty has been explored before, but the way it's integrated into the representation space is a positive aspect.

* **Significance:** Hallucination detection is a crucial problem for the safe deployment of LLMs. HaMI presents a practical and effective solution that demonstrates significant improvements over current methods. The results on diverse datasets and with various LLMs (LLaMA-2 and Mistral) suggest good generalizability. The ablation studies provide useful insights into the contributions of different components of the HaMI framework.

* **Strengths:**
    * **Strong Performance:** The empirical results convincingly demonstrate HaMI's superiority.
    * **Adaptive Token Selection:** The MIL-based approach is well-motivated and effective.
    * **Comprehensive Evaluation:** The paper includes thorough experiments with multiple datasets, LLMs, and baselines.
    * **Ablation Studies:**  Detailed ablation studies provide insights into the importance of each component.
    * **Cross-Dataset Generalization:**  The cross-dataset experiments highlight the robustness of HaMI.
    * **Clear Presentation:**  The paper is well-written and easy to follow.

* **Weaknesses:**
    * **GPT-4 Reliance for Labeling:** The dependence on GPT-4 for generating ground truth labels has the inherent limitations of any LLM-based evaluation. Although mitigated by re-judging the positive answers, there's still a question of perfect "ground truth."
    * **Computational Cost (Minor):**  While the ATS module itself doesn't add significant cost, the overall complexity of the MIL-based approach is likely more computationally intensive than simpler methods.  This isn't addressed specifically in the paper, although the authors mention that one advantage is avoiding the computational cost of external LLMs when not incorporating the semantic consistency score.
    * **Lack of Detailed Analysis of Selected Tokens:**  The paper could benefit from a more in-depth analysis of the *types* of tokens that are adaptively selected. Are they predominantly nouns, verbs, or specific types of entities? Understanding *what* the model is focusing on would be very valuable.

* **Impact:** HaMI has the potential to significantly influence the field of hallucination detection by offering a more robust and adaptive solution.  The MIL framework provides a novel approach to address inherent limitations of previous methods, and could serve as inspiration for future research.

**Justification for Score:**

The paper makes a valuable contribution to hallucination detection through its novel MIL-based adaptive token selection framework. The empirical results are compelling, and the ablation studies provide useful insights. While the reliance on GPT-4 for evaluation and potential computational cost are minor limitations, the overall strengths of the paper outweigh its weaknesses.  The potential influence on the field is significant, making this a strong and important paper.

Score: 8

- **Score**: 8/10

### **[GLUS: Global-Local Reasoning Unified into A Single Large Language Model for Video Segmentation](http://arxiv.org/abs/2504.07962v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces GLUS, a novel framework for Referring Video Object Segmentation (RefVOS) that unifies global and local reasoning using a single multimodal large language model (MLLM).  GLUS addresses the dilemma faced by previous MLLM-based methods, which either focus on understanding key frames (global reasoning) or tracking objects on continuous frames (local reasoning). GLUS achieves this by dividing the input frames into sparse "context frames" for global information and a stream of continuous "query frames" for local object tracking. A pre-trained VOS memory bank is integrated to digest short-range and long-range temporal information. Object contrastive learning is used to distinguish hard false-positive objects, and a self-refined framework identifies crucial frames and performs propagation.  GLUS achieves state-of-the-art results for MLLMs on the MeViS and Ref-Youtube-VOS benchmarks.

**Critical Evaluation:**

*   **Novelty:** The paper presents a well-structured and innovative approach to RefVOS by unifying global and local reasoning within a single MLLM. The "context + query frames" approach is a simple yet effective way to address the "Ref" and "VOS" dilemma in MLLM-based RefVOS methods. The end-to-end integration of a pre-trained VOS memory bank into the MLLM is a significant contribution, as it simplifies the system and enhances temporal reasoning. The object contrastive loss and self-refined framework for key frame selection are also valuable additions.
*   **Significance:** The paper's significance lies in its ability to achieve state-of-the-art results on challenging RefVOS benchmarks while maintaining a relatively simple and efficient architecture.  Decoupling reliance on external VOS models by integrating a pre-trained VOS component directly into the MLLM is a valuable contribution. The code is open sourced which will also greatly help future research.
*   **Strengths:**
    *   Addresses a key limitation of existing MLLM-based RefVOS methods.
    *   The unified global-local reasoning approach is effective and well-motivated.
    *   The end-to-end training with a pre-trained VOS memory bank simplifies the system.
    *   Demonstrates strong performance on challenging benchmarks.
    *   Ablation studies provide insights into the effectiveness of each component.
*   **Weaknesses:**
    *   The improvement on the relatively easier dataset (Ref-Youtube-VOS) is not as pronounced as on MeViS. This suggests GLUS may be particularly effective in complex scenarios, but it would strengthen the argument to see more consistent gains across all datasets.
    *   The performance improvements, while significant, might be partly attributed to the extensive supervised fine-tuning and use of a pre-trained VOS model (SAM-2). More analysis of the MLLM component's isolated contribution would be helpful.
    *   The reliance on specific architectures (LISA-7B-v1, SAM-2) limits a fully broad assesment.

*   **Potential Influence:** The paper has the potential to influence future research in RefVOS and related fields, such as video understanding and multimodal learning. The unified global-local reasoning approach and the end-to-end integration of memory modules are valuable concepts that can be applied to other tasks.
* **Rigorous Rationale:** The paper presents a well-motivated, novel, and well-executed framework for RefVOS. It tackles a specific deficiency of prior art and presents a strong case, both quantitatively and qualitatively, for the GLUS method. While the model is not completely free of reliance on other pre-trained models, the way the system brings these components together and trains them makes a significant impact and contribution to the field.

**Score: 8**

- **Score**: 8/10

## Other Papers
### **[REANIMATOR: Reanimate Retrieval Test Collections with Extracted and Synthetic Resources](http://arxiv.org/abs/2504.07584v1)**
### **[Boosting Universal LLM Reward Design through the Heuristic Reward Observation Space Evolution](http://arxiv.org/abs/2504.07596v1)**
### **[VLM-R1: A Stable and Generalizable R1-style Large Vision-Language Model](http://arxiv.org/abs/2504.07615v1)**
### **[Beating Transformers using Synthetic Cognition](http://arxiv.org/abs/2504.07619v1)**
### **[ConceptFormer: Towards Efficient Use of Knowledge-Graph Embeddings in Large Language Models](http://arxiv.org/abs/2504.07624v1)**
### **[Agent That Debugs: Dynamic State-Guided Vulnerability Repair](http://arxiv.org/abs/2504.07634v1)**
### **[Enhancing Large Language Models through Neuro-Symbolic Integration and Ontological Reasoning](http://arxiv.org/abs/2504.07640v1)**
### **[On the Temporal Question-Answering Capabilities of Large Language Models Over Anonymized Data](http://arxiv.org/abs/2504.07646v1)**
### **[Unveiling the Impact of Multimodal Features on Chinese Spelling Correction: From Analysis to Design](http://arxiv.org/abs/2504.07661v1)**
### **[FMNV: A Dataset of Media-Published News Videos for Fake News Detection](http://arxiv.org/abs/2504.07687v1)**
### **[Proactive User Information Acquisition via Chats on User-Favored Topics](http://arxiv.org/abs/2504.07698v1)**
### **[PR-Attack: Coordinated Prompt-RAG Attacks on Retrieval-Augmented Generation in Large Language Models via Bilevel Optimization](http://arxiv.org/abs/2504.07717v1)**
### **[MRD-RAG: Enhancing Medical Diagnosis with Multi-Round Retrieval-Augmented Generation](http://arxiv.org/abs/2504.07724v1)**
### **[Automated Construction of a Knowledge Graph of Nuclear Fusion Energy for Effective Elicitation and Retrieval of Information](http://arxiv.org/abs/2504.07738v1)**
### **[Zero-Shot Cross-Domain Code Search without Fine-Tuning](http://arxiv.org/abs/2504.07740v1)**
### **[SF2T: Self-supervised Fragment Finetuning of Video-LLMs for Fine-Grained Understanding](http://arxiv.org/abs/2504.07745v1)**
### **[Virtual-mask Informed Prior for Sparse-view Dual-Energy CT Reconstruction](http://arxiv.org/abs/2504.07753v1)**
### **[Efficient Tuning of Large Language Models for Knowledge-Grounded Dialogue Generation](http://arxiv.org/abs/2504.07754v1)**
### **[Exploring a Patch-Wise Approach for Privacy-Preserving Fake ID Detection](http://arxiv.org/abs/2504.07761v1)**
### **[Fairness Mediator: Neutralize Stereotype Associations to Mitigate Bias in Large Language Models](http://arxiv.org/abs/2504.07787v1)**
### **[Breaking the Barriers: Video Vision Transformers for Word-Level Sign Language Recognition](http://arxiv.org/abs/2504.07792v1)**
### **[Revisiting Likelihood-Based Out-of-Distribution Detection by Modeling Representations](http://arxiv.org/abs/2504.07793v1)**
### **[Plan-and-Refine: Diverse and Comprehensive Retrieval-Augmented Generation](http://arxiv.org/abs/2504.07794v1)**
### **[FairEval: Evaluating Fairness in LLM-Based Recommendations with Personality Awareness](http://arxiv.org/abs/2504.07801v1)**
### **[A System for Comprehensive Assessment of RAG Frameworks](http://arxiv.org/abs/2504.07803v1)**
### **[Cluster-Driven Expert Pruning for Mixture-of-Experts Large Language Models](http://arxiv.org/abs/2504.07807v1)**
### **[Understanding Learner-LLM Chatbot Interactions and the Impact of Prompting Guidelines](http://arxiv.org/abs/2504.07840v1)**
### **[The KL3M Data Project: Copyright-Clean Training Resources for Large Language Models](http://arxiv.org/abs/2504.07854v1)**
### **[2D-Curri-DPO: Two-Dimensional Curriculum Learning for Direct Preference Optimization](http://arxiv.org/abs/2504.07856v1)**
### **[Robust Hallucination Detection in LLMs via Adaptive Token Selection](http://arxiv.org/abs/2504.07863v1)**
### **[Pangu Ultra: Pushing the Limits of Dense Large Language Models on Ascend NPUs](http://arxiv.org/abs/2504.07866v1)**
### **[Towards Sustainable Creativity Support: An Exploratory Study on Prompt Based Image Generation](http://arxiv.org/abs/2504.07879v1)**
### **[Benchmarking Adversarial Robustness to Bias Elicitation in Large Language Models: Scalable Automated Assessment with LLM-as-a-Judge](http://arxiv.org/abs/2504.07887v1)**
### **[DiverseFlow: Sample-Efficient Diverse Mode Coverage in Flows](http://arxiv.org/abs/2504.07894v1)**
### **[How do Large Language Models Understand Relevance? A Mechanistic Interpretability Perspective](http://arxiv.org/abs/2504.07898v1)**
### **[Redefining Machine Translation on Social Network Services with Large Language Models](http://arxiv.org/abs/2504.07901v1)**
### **[Porting an LLM based Application from ChatGPT to an On-Premise Environment](http://arxiv.org/abs/2504.07907v1)**
### **[GenEAva: Generating Cartoon Avatars with Fine-Grained Facial Expressions from Realistic Diffusion-based Faces](http://arxiv.org/abs/2504.07945v1)**
### **[VCR-Bench: A Comprehensive Evaluation Framework for Video Chain-of-Thought Reasoning](http://arxiv.org/abs/2504.07956v1)**
### **[MM-IFEngine: Towards Multimodal Instruction Following](http://arxiv.org/abs/2504.07957v1)**
### **[VisualCloze: A Universal Image Generation Framework via Visual In-Context Learning](http://arxiv.org/abs/2504.07960v1)**
### **[Geo4D: Leveraging Video Generators for Geometric 4D Scene Reconstruction](http://arxiv.org/abs/2504.07961v1)**
### **[GLUS: Global-Local Reasoning Unified into A Single Large Language Model for Video Segmentation](http://arxiv.org/abs/2504.07962v1)**
