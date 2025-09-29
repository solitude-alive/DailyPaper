# The Latest Daily Papers - Date: 2025-09-29
## Highlight Papers
### **[REMA: A Unified Reasoning Manifold Framework for Interpreting Large Language Model](http://arxiv.org/abs/2509.22518v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces REMA, a novel framework for interpreting and diagnosing reasoning failures in Large Language Models (LLMs) by analyzing the "Reasoning Manifold."  REMA posits that successful reasoning processes in LLMs reside within a low-dimensional, structured subspace of the activation space.  The framework quantifies the deviation of internal representations corresponding to erroneous reasoning from the manifold approximated by correct reasoning representations.  It then localizes the "divergence point" where the reasoning chain goes off-track. The paper presents extensive experiments on diverse LLMs and tasks, demonstrating the low-dimensional nature of the reasoning manifold, the separability between erroneous and correct reasoning representations, and the effectiveness of REMA in analyzing the origins of reasoning failures. The method is model-agnostic and provides a geometric interpretation of reasoning failures.

**Critical Evaluation:**

**Strengths:**

*   **Novelty:** The concept of the "Reasoning Manifold" and its use for diagnosing reasoning failures offers a fresh perspective on LLM interpretability. While the manifold hypothesis is established in machine learning, its specific application and adaptation to understanding *reasoning* in LLMs, coupled with the deviation-based analysis, is novel. The unification of different failure types under a geometric framework is a strong contribution.
*   **Methodological Rigor:** The paper presents a well-defined methodology with clear steps for calculating deviation distances and localizing divergence points. The inclusion of several datasets spanning different modalities and reasoning abilities enhances the generalizability of the findings.
*   **Extensive Experimentation:** The comprehensive experimental setup covering a range of models (including multimodal ones) and tasks strengthens the empirical validation of the framework. The analyses of intrinsic dimensionality and mutual information provide empirical evidence for the core assumptions. The ablation and sensitivity analyses demonstrate the robustness of the approach.
*   **Clarity of Presentation:** The paper is well-written and clearly explains the core concepts, methodology, and results. The figures and tables effectively visualize the findings.

**Weaknesses:**

*   **Approximation of the Manifold:** The reliance on the set of "correct" reasoning representations as an approximation of the true Reasoning Manifold has limitations. The quality of this approximation is directly influenced by the diversity and density of the correct samples. This is explicitly acknowledged, but the impact on the accuracy of failure localization needs further quantification. "Nearly correct" examples that are classified as incorrect may lead to skewed results
*   **Exact Match Metric:** The strict use of exact match to define correct vs. incorrect samples is a simplifying assumption. It does not capture nuances in reasoning where a model might be "partially correct" or provide a valid alternative answer. This can introduce noise into the analysis. While acknowledged, the implications aren't fully explored.
*   **Computational Cost:**  The nearest neighbor calculations, while standard, can be computationally expensive, especially for larger models and datasets.  The paper doesn't delve into the scalability of the framework or potential optimizations.
*   **Limited Interventional Validation:** The paper focuses on *diagnosis*. While insightful, it lacks direct interventional validation.  Can knowing the divergence point lead to effective interventions to correct the reasoning process?  The paper hints at future work in this direction, but it's a missing piece.

**Significance and Impact:**

The paper makes a significant contribution to the growing field of LLM interpretability by providing a quantifiable geometric approach to understanding reasoning failures. It connects abstract failures to measurable deviations in representations, paving the way for new diagnostic tools and potentially, strategies for improving model robustness.  The REMA framework has the potential to become a valuable tool for model developers seeking to understand and mitigate reasoning errors in LLMs. While the approach requires further refinement, it presents a compelling direction for future research.
Score: 8
**Rationale:**

The paper presents a genuinely novel framework with solid theoretical underpinnings and extensive empirical validation. While the weaknesses highlight areas for improvement, they do not diminish the significance of the contribution. The potential impact of REMA on the field of LLM interpretability justifies a high score. The model is limited to focusing solely on the point cloud of correct reasoning representation as the ground truth of reasoning manifold.

- **Score**: 8/10

### **[The Emergence of Altruism in Large-Language-Model Agents Society](http://arxiv.org/abs/2509.22537v1)**
- **Summary**: Here's a summary and critical evaluation of the provided paper:

**Summary:**

The paper investigates the emergence of altruism in large-scale societies of agents driven by Large Language Models (LLMs).  It addresses a gap in existing research that primarily focuses on cooperation in small-scale, task-oriented games, and often overlooks the spontaneous emergence of altruism (sacrificing self-interest for collective benefit) in more complex settings. The authors introduce a Schelling-variant urban migration model, creating a social dilemma where over 200 LLM agents must navigate a conflict between egoistic and altruistic goals. They discover two distinct archetypes: "Adaptive Egoists" (prioritizing self-interest but influenced by social norms) and "Altruistic Optimizers" (exhibiting an inherent altruistic logic). Furthermore, they introduce a Grounded Theory-inspired method for qualitatively analyzing agent reasoning. The study concludes that model selection for social simulation is not just about reasoning capabilities but also about choosing the intrinsic social action logic embodied by different LLMs. Adaptive Egoists are better for modeling complex human-like societies, while Altruistic Optimizers are more suited for modeling idealized pro-social actors or collective welfare scenarios.

**Critical Evaluation:**

**Novelty:** The paper's novelty lies in several aspects:

*   **Focus on Altruism in Large-Scale LLM Societies:**  The specific focus on *altruism*, rather than simple cooperation, in *large-scale* LLM-driven agent simulations is a crucial distinction. Prior work tends to concentrate on smaller interactions and game-theoretic scenarios.
*   **Schelling-Variant Model for Social Dilemma:** The adaptation of a Schelling model to create an explicit conflict between individual and system utility provides a well-defined framework for studying the emergence of social tendencies. This is a good design choice because of its established standing in social science.
*   **Identification of LLM Archetypes:**  The observation and categorization of LLMs into "Adaptive Egoists" and "Altruistic Optimizers" is a significant finding. This reveals intrinsic heterogeneity in LLMs' social behaviors.
*   **Grounded Theory-Inspired Qualitative Analysis:**  The integration of qualitative analysis, using an LLM as a "judge" and inspired by Grounded Theory, to understand agent reasoning adds depth and interpretability.
*   **Model Selection Guideline:** The paper argues for theory choice as the important consideration when making a decision about which LLM to use for a given simulation. This is very important for the field to consider as it matures.

**Significance:** The paper makes a significant contribution because:

*   **Foundational Understanding of LLM Social Tendencies:**  It moves beyond using LLMs as simple behavioral replicators and examines their underlying social action logics. This is essential for building more reliable and interpretable simulations.
*   **Implications for Model Selection:**  The study offers a practical guideline for selecting appropriate LLMs for different types of social simulations, recognizing that the choice reflects a theoretical commitment to a particular behavioral model.
*   **Opens New Research Avenues:**  The findings pave the way for further investigations into the factors shaping LLM social tendencies and for developing more sophisticated LLM-driven simulations of complex social phenomena.
*   **Well-Defined Methodology:** The paper presents a clear and reproducible methodology, combining quantitative metrics with qualitative analysis.

**Weaknesses:**

*   **Simplified Environment:** The Schelling-variant model, while useful, is a simplification of real-world urban dynamics. The homogeneity of agents, lack of migration costs, and idealized utility function might limit the generalizability of the findings.
*   **Limited LLM Selection:** While the selection covers various types of LLMs, the number of LLMs tested is relatively small. Examining a broader range of models could strengthen the conclusions. The choice of a mini model as a representative of "Adaptive Egoists" could also be scrutinized.
*   **Potential Bias in LLM-as-Judge:** The use of an LLM (Gemini-2.5-pro) for qualitative analysis introduces a potential bias, as its own social tendencies might influence the coding process. The authors acknowledge this, but it needs to be considered when interpreting the results. This may be something worth highlighting in the limitations.
*   **Limited to One Context** While the authors show robustness in the three GSD levels, the conclusion may be somewhat limited to the migration context. The conclusion can be strengthened with a diverse array of environmental context, such as resource distribution, and task assignment.

**Score:** 8

**Justification:**

The paper demonstrates significant novelty in its approach to studying altruism in LLM-driven agent societies. The identification of distinct LLM archetypes and the development of a Grounded Theory-inspired method for qualitative analysis represent valuable contributions. The implications for model selection and the opening of new research avenues further underscore its significance. While the study has some limitations, such as the simplified environment and limited LLM selection, the strengths significantly outweigh the weaknesses. The paper makes a well-reasoned argument and provides convincing evidence to support its conclusions, making it a valuable contribution to the field of computational social science and LLM research.

- **Score**: 8/10

### **[StepORLM: A Self-Evolving Framework With Generative Process Supervision For Operations Research Language Models](http://arxiv.org/abs/2509.22558v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "STEPORLM: A Self-Evolving Framework with Generative Process Supervision for Operations Research Language Models":

**Summary:**

The paper introduces STEPORLM, a novel framework for training Large Language Models (LLMs) to solve Operations Research (OR) problems.  It addresses two key limitations of existing approaches: the credit assignment problem in outcome-based reinforcement learning and the myopic nature of conventional discriminative process supervision. STEPORLM features a co-evolutionary loop between a policy model and a generative process reward model (GenPRM). The policy model generates solution trajectories, which are then evaluated by an external solver for outcome correctness and by the GenPRM for holistic process feedback. This dual-feedback signal is used to align the policy via Weighted Direct Preference Optimization (W-DPO) and refine the GenPRM.  The paper demonstrates that STEPORLM achieves state-of-the-art performance across several OR benchmarks and that the co-evolved GenPRM can be used as a universal inference-time verifier to improve the performance of other LLMs.

**Critical Evaluation:**

**Novelty:** The paper demonstrates solid novelty in its design. The core contribution lies in the introduction of *generative process supervision* paired with a *co-evolutionary training loop* for OR LLMs. This approach contrasts with previous work that relies on outcome-based rewards or step-wise, discriminative PRMs. The integration of solver-based verification and GenPRM-based process evaluation into a unified W-DPO objective and GenPRM refinement is also a novel combination. The concept of using the GenPRM as a universal inference-time verifier is a significant contribution beyond the basic architecture.

**Significance:**  The significance of the paper is substantial.  OR problem-solving is a high-value application area for LLMs, and STEPORLM demonstrates a clear improvement over existing methods.  The core idea of self-verification by the LLM itself and the ability to extract a universal verifier has significant implications on trust and safety of LLM responses, particularly in critical applications. The extensive empirical results, demonstrating SOTA performance on a diverse set of benchmarks against strong baselines (including GPT-40), further solidify the significance of the work.

**Strengths:**

*   **Addressing Key Limitations:**  The paper accurately identifies and tackles critical weaknesses in existing LLM-based OR problem-solving methods.
*   **Co-evolutionary Framework:** The co-evolutionary approach is a powerful mechanism for improving both the policy and reward models.
*   **GenPRM as Universal Verifier:** The idea of using the GenPRM for inference-time verification is highly valuable and has broader applicability.
*   **Strong Empirical Results:**  The paper provides compelling empirical evidence to support its claims, with state-of-the-art performance across a wide range of benchmarks.
*   **Reproducibility:** The authors release their code, model weights, and verifier weights, enhancing the reproducibility and impact of their work.

**Weaknesses:**

*   **Complexity:** The STEPORLM framework is somewhat complex, involving multiple components and training stages.  While well-motivated, the implementation details might present a barrier to entry for some researchers.
*   **Scalability of GenPRM:** While used in this case for OR, it could have been demonstrated on more complex and varied LLM outputs to showcase the capabilities of the universal verifier more broadly.

**Potential Influence:**

The paper has the potential to significantly influence the field of LLM-based OR problem-solving.  The STEPORLM framework provides a promising new direction for training LLMs to reason more reliably and accurately about complex problems. The idea of generative process supervision and co-evolutionary training can be applied to other domains beyond OR. The GenPRM could be a starting point for building more trustable and self-verifying LLMs.

**Justification of Score:**

I am assigning a score of 8 to this paper. It demonstrates substantial novelty and high significance in the specific context of LLM-based OR problem-solving. The co-evolutionary approach and the universal verifier concept are significant contributions. The empirical results are very strong, and the release of code/models will facilitate further research. The complexity of the approach and limited testing outside the OR domain are minor limitations. Overall, STEPORLM represents a valuable advancement in the field.

Score: 8

- **Score**: 8/10

### **[UniMIC: Token-Based Multimodal Interactive Coding for Human-AI Collaboration](http://arxiv.org/abs/2509.22570v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "UniMIC: Token-Based Multimodal Interactive Coding for Human-AI Collaboration":

**Summary:**

The paper introduces UniMIC, a novel framework for multimodal interactive coding designed to optimize human-AI collaboration. Recognizing that existing codecs are primarily optimized for unimodal, one-way communication, leading to performance degradation in interactive settings, UniMIC uses tokenized representations as the communication medium. This allows for efficient, low-bitrate transmission while maintaining compatibility with Large Multimodal Models (LMMs). The framework incorporates lightweight Transformer-based entropy models tailored for different scenarios (generic, masked, text-conditioned) to minimize inter-token redundancy and further enhance compression. The authors demonstrate UniMIC's effectiveness through experiments in text-to-image generation, text-guided inpainting/outpainting, and visual question answering, showcasing significant bitrate savings without compromising task performance.

**Critical Evaluation:**

*   **Novelty:**  The paper's core novelty lies in its shift towards a token-based approach for multimodal interactive coding. This departs from traditional pixel-based codecs and adapts communication paradigms to be natively compatible with the token-based representations used by LMMs. The use of specialized Transformer-based entropy models to further refine compression for different task scenarios adds another layer of innovation. While token-based approaches exist, their application to multimodal *interactive* coding, particularly considering both human and machine "receivers" in the communication loop, is a significant step forward.
*   **Significance:**  The significance of UniMIC is substantial because it directly addresses a growing bottleneck in human-AI collaboration: the inefficient communication of multimodal data. As LMMs become more integrated into interactive applications (e.g., generative design assistants, diagnostic tools), the ability to exchange information efficiently without sacrificing fidelity becomes crucial. UniMIC offers a promising solution to this problem. The reported gains in bitrate savings, coupled with maintained task performance, demonstrate its practical value.

*   **Strengths:**
    *   **Well-defined problem and clear motivation:** The paper clearly articulates the limitations of existing codecs for interactive multimodal communication and motivates the need for a new paradigm.
    *   **Novel approach:** The token-based transmission and scenario-specific entropy models provide a distinct departure from traditional methods.
    *   **Strong experimental validation:**  The paper presents a comprehensive set of experiments across diverse tasks, including thorough comparisons to established baselines (both traditional and generative codecs).
    *   **Quantifiable results:**  The results are presented with clear metrics and quantitative comparisons, providing strong evidence for the effectiveness of UniMIC.
    *   **Well-written and easy to understand:** The paper is clearly structured and explained, making the concepts accessible.

*   **Weaknesses:**
    *   **Dependence on tokenizers:** The performance is inherently limited by the quality of the underlying tokenizers (e.g., MagViT-v2 for images, BPE for text). While the paper addresses this in the experiments, the framework is ultimately constrained by the capabilities of these components.  Improvements in tokenization could lead to further gains.
    *   **Complexity of Entropy Models:** While the Transformer-based entropy models are lightweight, they still add complexity to the system and require training.  The adaptation of these models to new LMM backbones might require some effort. The paper touches on the need for potential retraining/adaptation in the conclusion.
    *   **Limited discussion of limitations:** While the paper demonstrates effectiveness, a deeper discussion of potential limitations would strengthen it.  For example, under what conditions *might* UniMIC not perform as well, or where might the gains be marginal? (e.g., scenarios where the edge device has abundant bandwidth or when the data is inherently not compressible).

*   **Potential Influence:** The paper has the potential to significantly influence the field of multimodal communication. It establishes a new direction for designing codecs that are AI-native and optimized for interactive settings. UniMIC could inspire further research into token-based compression techniques, adaptive entropy models, and architectures for human-AI collaboration.  The shift towards token-based methods could also drive the development of more efficient tokenizers designed specifically for compression.

**Justification for Score:**

Given the novelty, significance, well-defined problem, strong experimental validation, and potential influence of UniMIC, I am assigning a score of 8.

*I am deducting points because, while addressing an important bottleneck and presenting a solid and novel approach with excellent experimental validation, the limitations could be explored in more depth and are dependent on external factors like the quality of the chosen tokenizer and the design/training requirements of the lighter Transformer entropy modules.*

Score: 8

- **Score**: 8/10

### **[SPARK: Synergistic Policy And Reward Co-Evolving Framework](http://arxiv.org/abs/2509.22624v1)**
- **Summary**: Here's a concise summary and rigorous evaluation of the provided paper:

**Summary:**

The paper introduces SPARK (Synergistic Policy And Reward Co-Evolving FrameworK), a novel reinforcement learning (RL) framework for Large Language Models (LLMs) and Large Vision-Language Models (LVLMs).  SPARK builds upon RL with Verifiable Rewards (RLVR) and addresses limitations of RL from Human Feedback (RLHF) which can be costly, and RLVR which discards potentially valuable data. SPARK recycles rollouts and correctness data from RLVR to train the model itself as a generative reward model. This unified framework enables a co-evolving feedback loop: improved reward accuracy yields better policy gradients, leading to higher-quality rollouts that further refine the reward model. SPARK demonstrates substantial performance gains on various reasoning, reward, and general benchmarks compared to baselines.

**Critical Evaluation:**

*   **Novelty:** The core novelty of SPARK lies in its synergistic, co-evolutionary approach to policy and reward modeling. Unlike existing RL approaches that treat policy and reward as separate entities (RLHF with a standalone reward model) or discard valuable data (RLVR), SPARK internalizes the reward model within the policy model and leverages RLVR rollouts for auxiliary reward training.  This on-policy approach, while not entirely unprecedented in the broader RL literature, is a significant contribution in the LLM/LVLM alignment space.  The method is novel in how it recycles RLVR data, and co-evolves both the reward and policy. Also the framework enables for test-time scaling without external reward models.

*   **Significance:** SPARK offers several advantages that make it significant:
    *   **Data Efficiency:** SPARK removes the need for expensive human preference data or separate reward model training, as the signals are already available from RLVR training. This greatly reduces data annotation costs and compute expenses.
    *   **Stability:** The on-policy and co-evolving nature of SPARK contributes to training stability by reducing reward-policy mismatch issues.
    *   **Performance Gains:** The empirical results demonstrate significant improvements on a range of reasoning, reward, and general benchmarks for both LLMs and LVLMs, suggesting the effectiveness and generalizability of the proposed framework.
    *   **Unified Development:** Enables RL training and test-time scaling within a single framework, saving GPU memory and reducing communication overhead.

*   **Strengths:**
    *   Clear Problem Definition: The paper clearly identifies the limitations of existing RL-based alignment methods for LLMs/LVLMs.
    *   Elegant Solution: SPARK provides a conceptually clean and technically sound solution to address the identified challenges.
    *   Strong Empirical Validation: The paper presents compelling empirical evidence demonstrating the effectiveness of SPARK on various benchmarks.
    *   Generalizability: Demonstrated across multiple LLM/LVLM architectures and scales.
    *   Practical Benefits:  Data efficiency and reduction in development costs.

*   **Weaknesses:**
    *   Reliance on RLVR: SPARK builds on RLVR, inherently limiting its applicability to tasks where verifiable rewards are easily available. While the paper argues that SPARK can extend alignment to tasks *beyond* strictly verifiable domains through self-reflection, the foundation remains RLVR.
    *   Coarse-Grained Objectives: The pointwise, pairwise, and reflection objectives, while effective, are relatively simple. Exploring more sophisticated or adaptive reward learning objectives could potentially further enhance performance.
    *   Limited Ablation Studies: While some ablation studies are included, deeper investigation of the contribution of various reward objectives (pointwise, pairwise, reflection) would be beneficial.

*   **Potential Influence:** SPARK has the potential to influence the field by promoting more efficient and stable RL-based alignment methods.  Its data efficiency and co-evolutionary approach could make it a valuable tool for training and deploying LLMs/LVLMs in various applications. The concept of internalizing the reward model is a particularly promising direction.

**Justification for Score:**

Given the above assessment, I assign the paper a score of **8**.

The paper makes a novel and significant contribution to the field of LLM/LVLM alignment. While it builds upon existing techniques (RLVR), it elegantly addresses some of the key limitations and proposes a more efficient and stable framework. The empirical results are strong, demonstrating substantial performance gains on relevant benchmarks. The unified framework and data efficiency offer clear practical advantages. However, the reliance on RLVR as a foundation somewhat limits its scope. Furthermore, there is room for future research in exploring more sophisticated reward learning objectives and deeper ablation studies.  Overall, SPARK is a well-executed and impactful work with the potential to influence future research in this area.

Score: 8

- **Score**: 8/10

### **[RefAM: Attention Magnets for Zero-Shot Referral Segmentation](http://arxiv.org/abs/2509.22650v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "REFAM: Attention Magnets for Zero-Shot Referral Segmentation":

**Summary:**

The paper introduces REFAM, a novel training-free approach to zero-shot referral segmentation that leverages diffusion transformers (DiTs).  The key idea is to exploit the attention mechanisms within DiTs for grounding referring expressions (natural language descriptions of regions) in images and videos. The approach addresses the issue of "attention sinks," where certain tokens (especially stop words) attract disproportionately high attention, hindering accurate localization.  REFAM introduces the concept of "attention magnets"—appending additional stop words to the referring expression—to redistribute background attention and improve the sharpness and accuracy of grounding maps.  These augmented stop words are then filtered out before aggregating the attention maps.  REFAM also identifies and addresses global attention sinks (GAS), showing they can be redirected to auxiliary tokens.  The method combines cross-attention maps, GAS handling, and attention redistribution.  The authors demonstrate state-of-the-art zero-shot performance on various referring image and video segmentation benchmarks without requiring fine-tuning or additional components.

**Critical Evaluation:**

*   **Novelty:** The paper introduces several novel ideas.
    *   **Attention Magnets:** The use of stop words to redistribute attention and improve grounding maps is a clever and relatively simple technique. While the general idea of attention manipulation isn't entirely new, its specific application in this context, especially within DiTs and without fine-tuning, is innovative.
    *   **Analysis of GAS in DiTs:** The detailed analysis of global attention sinks in diffusion transformers, particularly their impact on vision-language grounding and their relationship to semantic structure, is a valuable contribution.  While attention sinks are known in language models, this paper extends this observation and provides a solution to mitigate them in DiTs.
    *   **Training-Free Approach:** The training-free nature of REFAM is a significant advantage.  It makes the method readily applicable and eliminates the need for task-specific training data or architectural modifications.

*   **Significance:** The paper has the potential to be influential because:
    *   **Strong Performance:** The method achieves state-of-the-art results in zero-shot referral segmentation. This is a competitive field, and the improved performance demonstrates the effectiveness of the proposed techniques.
    *   **Simplicity and Generality:** The method is relatively simple to implement and can be applied to both image and video segmentation. This makes it attractive to researchers and practitioners.
    *   **Reliance on Foundation Models:** The work successfully leverages the power of pre-trained diffusion models without fine-tuning. It represents a shift towards more efficient use of large-scale models for downstream tasks.

*   **Strengths:**
    *   **Clear and Well-Written:** The paper is clearly written and well-organized, making it easy to understand the method and its contributions.
    *   **Thorough Evaluation:** The authors provide comprehensive experimental results on a variety of benchmarks, comparing REFAM to strong baselines. The ablation studies provide further insight into the importance of each component.
    *   **Insightful Analysis:** The analysis of attention sinks and the rationale behind the attention magnet approach are insightful.

*   **Weaknesses:**
    *   **Reliance on LLM Captions:** The method benefits from high-quality captions, sometimes generated by LLMs which are not transparent. This dependency makes it less fully independent and could introduce biases (although this is also a common and accepted practice).
    *   **SAM2 Limitations**: They use the SAM2 as segmentation model but discuss its limitations, especially those pertaining to under-segmentation, which may require a more advanced point-sampling strategy.
    *   **Limited Exploration of Alternative Magnets**: The experiments use stop words and one color term as attention magnets. A broader exploration of other types of tokens or learned magnets might yield further improvements.
    *   **Incremental improvements**: While the individual techniques may be considered somewhat incremental, the combination contributes to a novel and effective framework.

**Overall:**

The paper offers a novel, significant, and well-validated approach to zero-shot referral segmentation. The attention magnet concept, the analysis of GAS in DiTs, and the training-free nature of the method represent key contributions. The paper is generally well-written and the experiments demonstrate the efficacy of the approach. It addresses an important practical problem (efficiently adapting large generative models to downstream tasks) and provides a new approach leveraging attention manipulation.

**Score: 8**

**Rationale:** The paper makes solid contributions to the field by demonstrating state-of-the-art zero-shot referral segmentation without fine-tuning using diffusion transformers. The approach leverages attention manipulation techniques based on an insightful analysis of attention sinks. Despite the use of foundational models, some limitations on LLM caption dependency and the under-segmentation issue require more work. Still, REFAM will likely have a significant impact on the field by offering a readily adaptable training-free framework.

- **Score**: 8/10

## Other Papers
### **[Group Critical-token Policy Optimization for Autoregressive Image Generation](http://arxiv.org/abs/2509.22485v1)**
### **[Where MLLMs Attend and What They Rely On: Explaining Autoregressive Token Generation](http://arxiv.org/abs/2509.22496v1)**
### **[Estimating the Empowerment of Language Model Agents](http://arxiv.org/abs/2509.22504v1)**
### **[Representing LLMs in Prompt Semantic Task Space](http://arxiv.org/abs/2509.22506v1)**
### **[We Think, Therefore We Align LLMs to Helpful, Harmless and Honest Before They Go Wrong](http://arxiv.org/abs/2509.22510v1)**
### **[AxLLM: accelerator architecture for large language models with computation reuse capability](http://arxiv.org/abs/2509.22512v1)**
### **[REMA: A Unified Reasoning Manifold Framework for Interpreting Large Language Model](http://arxiv.org/abs/2509.22518v1)**
### **[Boosting Pointer Analysis With Large Language Model-Enhanced Allocation Function Detection](http://arxiv.org/abs/2509.22530v1)**
### **[InfiR2: A Comprehensive FP8 Training Recipe for Reasoning-Enhanced Language Models](http://arxiv.org/abs/2509.22536v1)**
### **[The Emergence of Altruism in Large-Language-Model Agents Society](http://arxiv.org/abs/2509.22537v1)**
### **[HyCoVAD: A Hybrid SSL-LLM Model for Complex Video Anomaly Detection](http://arxiv.org/abs/2509.22544v1)**
### **[JanusVLN: Decoupling Semantics and Spatiality with Dual Implicit Memory for Vision-Language Navigation](http://arxiv.org/abs/2509.22548v1)**
### **[Linear Causal Representation Learning by Topological Ordering, Pruning, and Disentanglement](http://arxiv.org/abs/2509.22553v1)**
### **[StepORLM: A Self-Evolving Framework With Generative Process Supervision For Operations Research Language Models](http://arxiv.org/abs/2509.22558v1)**
### **[Retrieval-Augmented Guardrails for AI-Drafted Patient-Portal Messages: Error Taxonomy Construction and Large-Scale Evaluation](http://arxiv.org/abs/2509.22565v1)**
### **[UniMIC: Token-Based Multimodal Interactive Coding for Human-AI Collaboration](http://arxiv.org/abs/2509.22570v1)**
### **[Dynamic Experts Search: Enhancing Reasoning in Mixture-of-Experts LLMs at Test Time](http://arxiv.org/abs/2509.22572v1)**
### **[ArabJobs: A Multinational Corpus of Arabic Job Ads](http://arxiv.org/abs/2509.22589v1)**
### **[Transport Based Mean Flows for Generative Modeling](http://arxiv.org/abs/2509.22592v1)**
### **[Benefits and Pitfalls of Reinforcement Learning for Language Model Planning: A Theoretical Perspective](http://arxiv.org/abs/2509.22613v1)**
### **[SPARK: Synergistic Policy And Reward Co-Evolving Framework](http://arxiv.org/abs/2509.22624v1)**
### **[CCNeXt: An Effective Self-Supervised Stereo Depth Estimation Approach](http://arxiv.org/abs/2509.22627v1)**
### **[UML-CoT: Structured Reasoning and Planning with Unified Modeling Language for Robotic Room Cleaning](http://arxiv.org/abs/2509.22628v1)**
### **[Towards Efficient Online Exploration for Reinforcement Learning with Human Feedback](http://arxiv.org/abs/2509.22633v1)**
### **[Training-Free Synthetic Data Generation with Dual IP-Adapter Guidance](http://arxiv.org/abs/2509.22635v1)**
### **[Scale-Wise VAR is Secretly Discrete Diffusion](http://arxiv.org/abs/2509.22636v1)**
### **[Language Models Can Learn from Verbal Feedback Without Scalar Rewards](http://arxiv.org/abs/2509.22638v1)**
### **[WebGen-Agent: Enhancing Interactive Website Generation with Multi-Level Feedback and Step-Level Reinforcement Learning](http://arxiv.org/abs/2509.22644v1)**
### **[RefAM: Attention Magnets for Zero-Shot Referral Segmentation](http://arxiv.org/abs/2509.22650v1)**
### **[VoiceAssistant-Eval: Benchmarking AI Assistants across Listening, Speaking, and Viewing](http://arxiv.org/abs/2509.22651v1)**
### **[Pixel Motion Diffusion is What We Need for Robot Control](http://arxiv.org/abs/2509.22652v1)**
