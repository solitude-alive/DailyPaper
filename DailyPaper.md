# The Latest Daily Papers - Date: 2025-05-10
## Highlight Papers
### **[Efficient Flow Matching using Latent Variables](http://arxiv.org/abs/2505.04486v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Efficient Flow Matching using Latent Variables":

**Summary:**

The paper introduces Latent-CFM, a novel approach to improve flow matching generative models by explicitly incorporating data structure using pre-trained deep latent variable models (LVMs). Latent-CFM leverages LVMs to learn a lower-dimensional latent space representation of the data, which is then used to condition the flow matching process. This aims to address the inefficiency of standard flow matching models when dealing with high-dimensional data residing on lower-dimensional manifolds. The authors demonstrate that Latent-CFM improves generation quality, reduces training time, and enables conditional image generation based on latent features, outperforming state-of-the-art flow matching methods on synthetic, image benchmark datasets (MNIST, CIFAR10), and a 2D Darcy flow dataset.

**Critical Evaluation:**

*   **Novelty:** The paper's novelty lies in its effective combination of flow matching with pre-trained deep latent variable models.  While conditioning flow matching on latent spaces has been explored before, this work differentiates itself by:

    *   Utilizing *pre-trained* LVMs, potentially leading to more stable training and efficient extraction of relevant data features. This helps sidestep the expensive joint training of VAE and flow matching models (like VRFM) reducing computational overhead.
    *   Specifically targeting scenarios where high-dimensional data resides on a lower-dimensional manifold, a common characteristic of real-world data, allowing for more efficient flows.
    *   Demonstrating improved performance and training efficiency compared to competing flow matching approaches on several benchmark datasets.
    *   Highlighting Latent-CFM’s ability to generate physically accurate results.

*   **Significance:** The work addresses a crucial limitation of standard flow matching – the inability to efficiently model complex data structures. By leveraging LVMs, Latent-CFM enhances generation quality and reduces training costs, making flow matching more practical for real-world applications, especially in scientific domains where underlying physical constraints often induce manifold structures. The ability to generate samples conditioned on latent features also opens up new possibilities for controlled generation and interpretability. The observed improvements in physical plausibility of generated data, demonstrated through the Darcy flow experiment, are significant for scientific computing applications.

*   **Strengths:**

    *   **Clear problem definition:** The paper clearly articulates the limitations of standard flow matching.
    *   **Well-motivated approach:** The use of LVMs is logically justified.
    *   **Effective method:** The Latent-CFM approach demonstrates tangible improvements in generation quality, training efficiency, and conditional generation capabilities.
    *   **Comprehensive experiments:** The experiments cover a range of datasets and provide convincing evidence of the method's effectiveness, including both synthetic and real-world datasets.
    *   **Strong results:** The results are convincingly demonstrated using appropriate evaluation metrics (FID, Wasserstein-2).

*   **Weaknesses:**

    *   **Dependency on pre-trained models:** The approach relies on the availability of suitable pre-trained LVMs. While transfer learning is common, the performance may be limited by the quality of the pre-trained model. It would be beneficial to show the method is robust to different LVM architectures or performance of the LVM.
    *   **Limited theoretical analysis:** While the proposition of upper bound is included, it would be strengthened by further analysis, perhaps exploring the properties of the latent space and its relationship to the flow matching process.
    *   **Parameter tuning:** There is some discussion of hyperparameter settings for Latent-CFM, but a more thorough sensitivity analysis would be valuable.

*   **Potential Impact:** Latent-CFM has the potential to impact generative modeling by making flow matching more applicable and efficient for a broader range of datasets, especially in scientific domains. Its ability to generate physically plausible data and offer interpretable latent representations could significantly benefit scientific applications.

**Overall:**

The paper presents a solid contribution to the field of generative modeling. Latent-CFM offers a practical and effective approach to improve flow matching by explicitly modeling data structure, addressing a key limitation of existing methods. The experimental results are convincing, and the potential impact is significant, particularly for scientific applications. However, the reliance on pre-trained LVMs is a potential constraint that warrants further exploration.

**Score: 8**

- **Score**: 8/10

### **[ZeroSearch: Incentivize the Search Capability of LLMs without Searching](http://arxiv.org/abs/2505.04588v1)**
- **Summary**: Here's a summary and critical evaluation of the ZEROSEARCH paper:

**Summary:**

The paper introduces ZEROSEARCH, a novel reinforcement learning (RL) framework designed to improve the search capabilities of large language models (LLMs) without requiring interaction with real-world search engines.  The framework tackles the challenges of uncontrolled document quality and high API costs typically associated with RL training on live search engines. ZEROSEARCH leverages a lightweight supervised fine-tuning (SFT) stage to transform an LLM into a retrieval module capable of generating both relevant and noisy documents. A curriculum-based rollout strategy is then employed during RL training, incrementally degrading the quality of the generated documents to progressively challenge the model's reasoning ability.  Experiments demonstrate that ZEROSEARCH effectively enhances the search capabilities of LLMs, even using smaller models (3B) as retrieval modules, and scaling up to performance comparable to or even surpassing real search engines. The framework is also shown to be generalizable across different model architectures (base vs. instruction-tuned) and compatible with various RL algorithms.

**Critical Evaluation:**

*   **Novelty:** The core idea of simulating a search engine using an LLM is a clever way to circumvent the limitations of real-world search engine interaction. The use of curriculum learning to progressively introduce noise into the retrieval process is a valuable contribution. While the individual components (SFT, RL, curriculum learning) are not entirely novel, their combination within the ZEROSEARCH framework, specifically tailored to the problem of LLM search, constitutes a significant advance.
*   **Significance:**  The paper addresses a critical issue in LLM research: how to equip models with the ability to access and reason with external information effectively and efficiently. ZEROSEARCH tackles the major bottlenecks of cost and instability associated with existing RL-based search approaches.  The performance results are compelling, demonstrating that the simulated search engine can achieve comparable or even superior performance to real search engines.  This has the potential to democratize research in this area by removing the barrier of high API costs. The generalizability across different model types and RL algorithms further enhances its value.
*   **Strengths:**

    *   **Addresses a practical problem:**  The high cost and variability of real-world search APIs pose a significant barrier to research. ZEROSEARCH offers a viable and scalable solution.
    *   **Well-designed framework:** The combination of SFT and curriculum learning is thoughtfully designed to train robust search capabilities.
    *   **Strong empirical results:** The experiments are thorough and demonstrate consistent improvements over baseline methods.
    *   **Generalizability:**  The framework works with different model architectures and RL algorithms.
    *   **Potential for impact:** Could significantly lower the barrier to entry for research on LLMs with search capabilities.
*   **Weaknesses:**

    *   **Dependency on GPU resources:** While cheaper than API calls, deploying the simulation LLM still requires substantial GPU resources, which is a limitation for some researchers. The paper acknowledges this.
    *   **Complexity:** The framework involves multiple stages (SFT, RL, curriculum learning), which adds to the complexity of implementation and tuning.
    *   **Evaluation metrics:** While EM is a standard metric, a more nuanced evaluation of the quality of the retrieved documents and their impact on the final answer would strengthen the paper.
    *   **Limited ablation studies:** More ablation studies on the different components of the framework (e.g., the specific reward function) would provide further insights.
*   **Potential Influence:**  ZEROSEARCH has the potential to become a widely adopted framework for training LLMs with search capabilities due to its scalability, cost-effectiveness, and generalizability. It could spur further research into simulated environments for RL training.

**Justification of Score:**

I am assigning a score of **8**. The paper presents a novel and well-designed framework that tackles a significant problem in the field. The empirical results are compelling, and the potential for impact is high. While there are some limitations, the strengths of the paper outweigh the weaknesses. The work makes a substantial contribution to the area of LLMs with search capabilities, offering a practical and scalable alternative to existing methods.
Score: 8

- **Score**: 8/10

### **[PrimitiveAnything: Human-Crafted 3D Primitive Assembly Generation with Auto-Regressive Transformer](http://arxiv.org/abs/2505.04622v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper "PrimitiveAnything: Human-Crafted 3D Primitive Assembly Generation with Auto-Regressive Transformer" introduces a novel framework for decomposing complex 3D shapes into simpler geometric primitives. It frames this task as a sequence generation problem, leveraging a transformer-based architecture trained on a large dataset of human-crafted 3D primitive assemblies. The framework includes a shape-conditioned primitive transformer for autoregressive generation and an ambiguity-free parameterization scheme for multiple primitive types. The authors demonstrate that their method can generate high-quality primitive assemblies that align better with human perception than existing geometric optimization or category-specific learning-based methods. It benefits various 3D applications and shows potential for enabling primitive-based user-generated content.

**Critical Evaluation:**

* **Novelty:** The paper's primary novelty lies in reframing the problem of 3D shape abstraction as a sequence generation task, specifically using an autoregressive transformer model. While the use of transformers for 3D generation is not entirely new (MeshAnything being a notable predecessor), applying it to primitive assembly and learning directly from human-crafted abstractions is a significant departure from traditional geometric fitting or direct regression approaches. The proposed ambiguity-free parameterization scheme is another novel contribution that improves the robustness and accuracy of the learning process.

* **Significance:** The work holds significant potential for various applications in 3D computer vision and graphics. Primitive-based representations offer advantages in semantic understanding, robotic manipulation, scene understanding, and interactive modeling systems. A model that learns from human intuition and generalizes well across diverse shape categories has the potential to simplify complex 3D content creation pipelines and enable new forms of user interaction. The potential for creating lightweight 3D models suitable for real-time multiplayer game environments is particularly noteworthy.

* **Strengths:**
    * **Human-centric approach:**  Learning from human-crafted abstractions is a key strength, allowing the model to capture intuitive shape decomposition logic rather than just optimizing for geometric fidelity.
    * **Generalization:**  The ability to generalize across diverse shape categories, unlike previous category-specific methods, is a significant advantage.
    * **Technical Design:** The carefully designed ambiguity-free parameterization scheme and the cascaded primitive decoder contribute to the robustness and performance of the model.
    * **Comprehensive Evaluation:**  The paper includes extensive quantitative and qualitative comparisons against state-of-the-art methods, demonstrating the superiority of the proposed approach. The user study adds further credibility by validating the alignment with human perception.
    * **Ablation Studies:** Provides strong insight and proves the effectiveness of the individual components.
* **Weaknesses:**
    * **Primitive limitations:**  The current implementation is limited to three primitive types (cuboids, elliptical cylinders, and ellipsoids). While the modular design allows for the addition of new primitive types, the expressiveness of the representation is still constrained.
    * **Out-of-Distribution Performance:** As the authors acknowledge, the method struggles with objects containing topological structures rarely seen in training.
    * **Annotation Diversity:** The variety of annotation styles could also introduce some inconsistencies in the training data, potentially affecting the model's performance.
    * **Lack of texture:** Only focus on geometry and not appearance modelling.

* **Potential Influence:** The paper has the potential to influence future research in 3D shape abstraction, content generation, and scene understanding. The sequence generation approach could inspire new methods for representing and manipulating 3D scenes. The focus on human-centric abstractions could lead to more intuitive and user-friendly 3D modeling tools.

**Score:** 8

**Justification:**

The paper presents a significant and novel approach to 3D shape abstraction with a clear potential impact. The framework's ability to learn from human-crafted abstractions and generalize across shape categories represents a notable advance over existing methods. The technical design is well-reasoned and the experimental evaluation is thorough. While the limitations regarding primitive types and out-of-distribution performance exist, these are clearly acknowledged and offer directions for future research. The novelty of the autoregressive approach to shape abstraction, combined with the solid experimental results and clear potential for real-world applications, warrants a high score. However, the current limitations prevent it from achieving a score in the 9-10 range.

- **Score**: 8/10

### **[Hyb-KAN ViT: Hybrid Kolmogorov-Arnold Networks Augmented Vision Transformer](http://arxiv.org/abs/2505.04740v1)**
- **Summary**: Here's a summary and critical evaluation of the Hyb-KAN ViT paper:

**Summary:**

The paper introduces Hyb-KAN ViT, a novel vision transformer architecture that replaces traditional MLPs with hybrid Kolmogorov-Arnold Networks (KANs).  Hyb-KAN ViT incorporates two key modules: Efficient-KAN (Eff-KAN), which uses spline functions for efficient computation, and Wavelet-KAN (Wav-KAN), which leverages wavelet transforms for multi-resolution feature extraction. The paper explores different configurations of these modules within the ViT encoder layers and classification heads.  The results on ImageNet-1K, COCO, and ADE20K show state-of-the-art performance, suggesting the effectiveness of wavelet-driven spectral priors and spline-based efficiency. The authors present ablation studies to validate these findings and demonstrate a balanced approach to parameter efficiency and multi-scale representation in vision architectures. The paper highlights the potential of this approach to address computational bottlenecks in ViTs while enhancing feature encoding and representation learning.

**Critical Evaluation:**

**Novelty:**

The primary novelty lies in the *specific combination* of existing techniques into a hybrid ViT architecture.

*   **KANs and Wavelets:** The use of KANs in vision transformers is not entirely new, as the paper itself acknowledges related works that introduce KANs and Wavelet-KANs. However, the *integration of both Efficient-KANs (spline-based) and Wavelet-KANs within the same ViT architecture* is a significant contribution. This hybrid approach allows for combining the advantages of both methods: spline-based efficiency and wavelet-based multi-resolution analysis.

*   **Modular Architecture & Exploration:** The paper's modular design is also a key novelty. Systematically replacing MLPs with altered Wav-KAN modules in both encoder layers and classification heads enables flexibility to experiment with different configurations, and enables a detailed examination of how Wav-KAN enhances feature encoding and representation learning within ViTs.
The paper makes a claim on overcoming limitations of MLPs and existing ViT variants. However, the existing ViT variants already focus on hierarchical attention, hybrid architectures, or parameter-efficient designs. The introduction of Wav-KAN in conjunction with KAN modules achieves greater flexibility compared to prior work.

**Significance:**

*   **Performance Gains:** The state-of-the-art results on benchmark datasets (ImageNet-1K, COCO, ADE20K) clearly demonstrate the potential of Hyb-KAN ViT to improve performance compared to traditional ViTs and other existing architectures. This is significant because it shows that the proposed approach can lead to better accuracy in various vision tasks.

*   **Computational Efficiency:** The paper addresses the computational limitations of original ViTs by introducing Eff-KAN, which is designed for GPU-friendly matrix multiplications and optimized spline computations. This is significant because it makes the architecture more practical for real-world applications. However, the analysis of GFLOPs in the KAN framework needs to be improved.

*   **Multi-Scale Representation:** Wav-KAN leverages wavelet transforms to extract both high-frequency and low-frequency components of input data, providing a more robust feature representation. This is significant because it allows the model to capture complex data patterns more effectively.

*   **Edge Awareness and Smoothing:** the framework leverages KAN's mathematical properties and enables early detection of edges while suppressing high frequency noise. This capability is essential for enhanced segmentation and detection.

**Weaknesses:**

*   **Limited Comparison and Baselines:** Comparing against a broader range of SOTA ViT architectures would strengthen the claims of superiority.

*   **Clarity on Computational Complexity:** While the paper claims better computational efficiency, a more detailed analysis of the computational complexity of Eff-KAN and Wav-KAN compared to standard MLPs in ViTs would be beneficial. GFLOPs alone don't tell the full story. Factors like memory access patterns and GPU kernel efficiency need to be considered.

*   **Detailed Implementation Details:** While the authors adhere to reproducibility standards, further details on the kernel optimizations, wavelet implementation, and hyperparameter tuning strategy would be valuable.

*   **Scaling Challenges:** Although the paper introduces the concept of how the framework can be scaled, further performance would be needed to assess the accuracy and efficiency.

**Potential Influence:**

*   **Hybrid Architectures:** The Hyb-KAN ViT approach could inspire future research on combining different types of neural network modules within vision transformers to achieve better performance and efficiency.
*   **Wavelet-Based Vision:** The use of wavelet transforms for feature extraction in vision transformers could lead to new approaches for capturing multi-scale information and improving the robustness of these models.
*   **KANs in Computer Vision:** This work strengthens the argument for KANs as a viable alternative to MLPs in ViTs.

**Justification of Score:**

The paper presents a significant advancement in ViT architecture by introducing a hybrid KAN-based approach. The use of both spline-based and wavelet-based KANs, along with the modular architecture, allows for achieving SOTA results on benchmark datasets. However, there are some weaknesses. Therefore:

**Score: 8**

The paper combines existing techniques in a novel way, demonstrating clear performance gains with a solid architectural design. The significance of the paper is also increased due to the improvements in computational efficiency. Although there are a few points to be improved, the paper presents a clear and effective application of both spline based and wavelet based functions to the MLP problem of Vision Transformers. The paper introduces a robust and effective approach to integrating hybrid Kolmogorov-Arnold Networks in ViTs.

- **Score**: 8/10

### **[Perception, Reason, Think, and Plan: A Survey on Large Multimodal Reasoning Models](http://arxiv.org/abs/2505.04921v1)**
- **Summary**: Here's a summary and critical evaluation of the provided paper, aiming to provide a rigorous rationale for the assigned score:

**Summary:**

The paper is a comprehensive survey on Large Multimodal Reasoning Models (LMRMs). It structures the field around a four-stage developmental roadmap:
1.  **Perception-Driven Modular Reasoning**: Early efforts relying on task-specific modules for representation, alignment, and fusion.
2.  **Language-Centric Short Reasoning (System 1)**: Emergence of prompt-based and structured reasoning in multimodal LLMs, emphasizing surface-level understanding.
3.  **Language-Centric Long Reasoning (System 2)**: Exploration of extended reasoning chains and reinforcement learning to enable long-horizon thinking and planning.
4.  **Native LMRMs (N-LMRMs)**: A prospective paradigm shift where reasoning is natively integrated, emphasizing omnimodal perception, agentic behavior, and adaptive planning.

The survey identifies limitations in current models regarding omnimodal generalization, reasoning depth, and agentic behaviors. It highlights emerging trends like instruction tuning, reinforcement learning, and the development of unified representations and benchmark construction as key to future progress. It offers a taxonomy of models, datasets, and benchmarks, and projects future directions for N-LMRMs.

**Critical Evaluation:**

*   **Novelty:** The paper's novelty lies in its structured approach to surveying the evolution of LMRMs, organizing them into distinct stages characterized by design philosophies and capabilities. While previous surveys may have touched on aspects of MLLMs or reasoning, this paper offers a more holistic and chronologically organized view of the entire roadmap, filling a gap in the need for a coherent framework. Additionally, the introduction of "Native LMRMs" is forward-looking and offers a clear goal for future research.
*   **Significance:**  The significance stems from its comprehensive nature and the synthesis of trends. It addresses limitations of current models and contextualizes a vast amount of literature (540+ publications). This provides researchers with a valuable resource for understanding the landscape and identifying promising areas for future work.  The identification of two transformative capabilities 1) Multimodal Agentic Reasoning and 2) Omni-Modal Understanding and Generative Reasoning are also important directions for the community.  The reorganisation of existing datasets and benchmarks into clear categories also enhances the usefulness of the review for the community.

*   **Strengths:**
    *   **Comprehensive Scope:** The breadth of coverage across different models, datasets, and benchmarks is impressive.
    *   **Clear Structure:**  The four-stage roadmap provides a compelling narrative for the field's evolution.
    *   **Forward-Looking Perspective:**  The discussion of N-LMRMs and future research directions is valuable for guiding future development.
    *   **Practical Utility:** The reorganized dataset and benchmark information is extremely useful for researchers.
    *   **Timeliness:** The survey appears to be up-to-date (given the inclusion of papers to 2025.04) which further strengthens the significance and insights of the review.

*   **Weaknesses:**
    *   **Limited Technical Depth on N-LMRMs**: As the N-LMRMs paradigm is prospective, the section is primarily conceptual and less grounded in concrete technical details compared to the earlier stages.  This is understandable given the early stage of development.
    *   **Potential for Over-Simplification:**  Any attempt to categorize a complex field risks oversimplification. While the stages are generally well-defined, some overlap and nuances within each stage might be missed.
    *   **Subjectivity in Category Definitions:** The assignment of models to specific stages could be subjective, requiring further refinement. However, the categories used are generally well explained and representative of the field.

*   **Potential Influence:** The paper has a high potential to influence the field by:
    *   **Providing a shared understanding:**  The roadmap and taxonomy will help researchers contextualize their work and understand the relationships between different approaches.
    *   **Identifying key challenges:**  The highlighting of limitations and future research directions will focus attention on critical areas.
    *   **Facilitating collaboration:**  The survey will make it easier for researchers to identify relevant work and collaborators.

**Justification for Score:**

Given its comprehensive nature, novel structuring of the field, clear articulation of current limitations, and forward-looking perspective, this survey makes a valuable contribution. While the prospective nature of the N-LMRMs section presents some limitation, the survey provides clear potential future direction for the area. The thorough taxonomy and dataset re-organization contribute to the paper being a valuable resource for researchers. Therefore, the rating is justified as:

Score: 8

- **Score**: 8/10

### **[Graffe: Graph Representation Learning via Diffusion Probabilistic Models](http://arxiv.org/abs/2505.04956v1)**
- **Summary**: This paper introduces Graffe, a self-supervised graph representation learning framework based on diffusion probabilistic models (DPMs). Graffe employs a graph encoder to distill a source graph into a compact representation, which then conditions a diffusion decoder during its denoising process. The paper theoretically demonstrates that the denoising objective implicitly maximizes the conditional mutual information between the data and its representation. Empirically, Graffe achieves competitive results in node and graph classification tasks, attaining state-of-the-art performance on several real-world datasets.

**Rigorous and Critical Evaluation:**

*   **Novelty:** The paper's novelty lies in its application of diffusion models to graph representation learning and its theoretical analysis of the denoising objective within this context. While DPMs have been explored for visual semantics, their application to graph data, particularly with a strong theoretical grounding linking it to InfoMax principles, is a significant contribution. The Diff-InfoMax principle, extending the standard InfoMax, and its application in graph representation learning appears to be a key original idea.

*   **Significance:** The paper demonstrates that DPMs, typically associated with generative tasks, can be effectively adapted for discriminative representation learning in graphs. The theoretical analysis provides a strong justification for why this approach works and offers insights into the role of information content in the denoising process. The empirical results validate the effectiveness of Graffe, showcasing its competitive or state-of-the-art performance on a range of graph learning tasks. This suggests that leveraging DPMs opens promising new avenues for research in graph representation learning. The rigorous proofs provided give solid foundations to the findings.

*   **Strengths:**
    *   **Strong Theoretical Foundation:** The paper offers a solid theoretical justification for its approach, linking the denoising objective to conditional mutual information and introducing the Diff-InfoMax principle. The theorem proves the negative log likelihood of the denoising score as a lower bound on the conditional mutual information, which is crucial to give a solid theoretical backing to their methodology.
    *   **Extensive Empirical Validation:** The paper presents comprehensive experimental results on a diverse set of datasets, demonstrating the effectiveness of Graffe in both node and graph classification tasks. This wide range of test settings greatly strengthens confidence in the findings.
    *   **Clear and Well-Structured Presentation:** The paper is well-written and organized, making it easy to understand the proposed approach and its theoretical underpinnings.

*   **Weaknesses:**
    *   **Computational Cost:** Although not explicitly mentioned, DPMs can be computationally expensive to train, potentially limiting the scalability of Graffe to very large graphs. This should have been mentioned and explored.
    *   **Encoder Choice:** While the paper justifies the use of GAT and GIN as encoders, there is limited exploration of alternative encoder architectures and their impact on performance. It could have mentioned and explored a different encoder.
    *   **Limited Ablation Analysis:** While the ablation studies assess the impact of masking and decoder choices, a more comprehensive ablation analysis could further elucidate the contribution of different components of Graffe.

*   **Potential Influence:** Graffe's novel combination of DPMs and graph representation learning is likely to influence future research in this area. The Diff-InfoMax principle could provide a theoretical foundation for other self-supervised learning approaches. The code release will also facilitate adoption and further exploration of this method.

Despite some minor limitations, the paper presents a significant contribution to the field of graph representation learning. The novel application of diffusion models, the rigorous theoretical analysis, and the compelling empirical results make this a highly valuable contribution.

Score: 8

- **Score**: 8/10

### **[ReAlign: Bilingual Text-to-Motion Generation via Step-Aware Reward-Guided Alignment](http://arxiv.org/abs/2505.04974v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper "ReAlign: Bilingual Text-to-Motion Generation via Step-Aware Reward-Guided Alignment" introduces a new approach to generating 3D human motions from bilingual text descriptions (English and Chinese). The paper addresses two key challenges: the lack of bilingual text-motion datasets and the misalignment between text and motion distributions in diffusion models.  To tackle these issues, the authors: 1) introduce BiHumanML3D, a new bilingual human motion dataset; 2) propose BiMD, a Bilingual Motion Diffusion model that leverages cross-lingual alignment to capture semantics; and 3) develop ReAlign, a Reward-guided sampling Alignment method that comprises a step-aware reward model to assess alignment quality during sampling, guiding the diffusion process toward an optimally aligned distribution. The step-aware reward combines text-aligned and motion-aligned modules, refining noisy motions at each timestep. Experiments demonstrate improvements in text-motion alignment and motion quality compared to existing methods.

**Critical Evaluation:**

*   **Novelty:** The paper has several novel contributions:
    *   **BiHumanML3D Dataset:** Creating a bilingual text-to-motion dataset is itself a significant contribution given the scarcity of such resources. This opens up new avenues for research in cross-lingual motion generation.
    *   **BiMD Model:** Developing a bilingual motion diffusion model that leverages cross-lingual alignment is a logical and effective approach. Using cross-lingual embeddings is not novel in itself, but its specific application in text-to-motion generation with a bilingual context is new.
    *   **ReAlign:** The reward-guided alignment strategy, particularly the step-aware reward model, is a key innovation. Decomposing the reward into text-aligned and motion-aligned components and integrating timestep information is an important improvement over simply fine-tuning the diffusion model using reinforcement learning. The "plug-and-play" nature of ReAlign is valuable.

*   **Significance:**
    *   **Addressing a Gap:** The paper addresses a significant gap in the field of text-to-motion generation, which has primarily focused on monolingual English datasets.
    *   **Improved Alignment:** The ReAlign method demonstrates a clear improvement in text-motion alignment, which is a persistent challenge in diffusion-based models. This translates to more coherent and semantically accurate motions.
    *   **Potential Impact:** The paper has the potential to impact cross-linguistic applications in areas such as gaming, filmmaking, and robotics, making motion synthesis more accessible and inclusive.
    *   **Strong Empirical Results:** The experiments are comprehensive, demonstrating the effectiveness of the proposed method on both the new BiHumanML3D dataset and existing datasets. Ablation studies provide insights into the contribution of each component.

*   **Strengths:**
    *   Clear problem definition and motivation.
    *   Well-designed bilingual dataset construction pipeline.
    *   Innovative and effective ReAlign method.
    *   Comprehensive experiments and analysis.
    *   Plug-and-play approach, allowing for easy integration with existing methods.

*   **Weaknesses:**
    *   Dependence on CLIP and Pre-trained Models: Like many current approaches, it relies heavily on pre-trained language models like CLIP, potentially inheriting biases and limitations. While mitigating some effects via finetuning, inherent constraints may apply.
    *   Computational Cost: While the paper addresses it well, applying diffusion models can be slow. Though addressed via timestep-aware adjustments, practical applications may require further refinement, like MotionLCM.

*   **Potential Influence:** The paper is likely to stimulate further research in bilingual and multilingual text-to-motion generation. The ReAlign method could be adapted for other modalities and generative tasks. The BiHumanML3D dataset will be a valuable resource for the community.

*   **Justification of Score:** While the individual components are built upon existing research, the paper's combination of a new bilingual dataset, cross-lingual aligned diffusion model, and step-aware reward-guided alignment makes a noteworthy contribution to the field. It addresses an under-explored problem with a well-designed solution and strong empirical validation.

**Score: 8**

- **Score**: 8/10

### **[Scalable Multi-Stage Influence Function for Large Language Models via Eigenvalue-Corrected Kronecker-Factored Parameterization](http://arxiv.org/abs/2505.05017v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper addresses the challenge of attributing predictions made by fine-tuned large language models (LLMs) back to their original pre-training data. It proposes a multi-stage influence function framework that traces the influence of pre-training data on the downstream predictions of fine-tuned LLMs, even when the fine-tuning process significantly modifies the model (e.g., changes the output layer). To improve scalability, the authors employ Eigenvalue-corrected Kronecker-Factored (EK-FAC) parameterization to approximate inverse Hessian-Vector Products (iHVPs) and use semantic similarity heuristics to narrow down the candidate training samples. They conduct experiments to demonstrate the scalability of the EK-FAC approximation, the effectiveness of the multi-stage influence function, and its applicability to a publicly available instruction-tuned LLM, showing insights into the LLM's generation based on pre-training data.

**Critical Evaluation:**

* **Novelty:** The key novelty lies in combining multi-stage influence functions with EK-FAC parameterization for scaling to large language models under full-parameter fine-tuning scenarios.  While multi-stage influence functions and EK-FAC have been used previously, their combination in this specific context and scale represents a significant improvement. Specifically, it extends the original IF framework and addresses limitations of prior approaches like frozen encoders stacked with linear classifiers by addressing the issue of output domain mismatch and accommodating the full-parameter tuning paradigm prevalent in LLMs.
* **Significance:** Understanding the influence of pre-training data is crucial for building trustworthy AI systems and providing interpretability for LLMs. The ability to trace predictions back to the original data enables users to better understand the model's behavior, identify potential biases, and verify the grounding of the model's outputs.  The paper offers a practical approach for achieving this, addressing a significant bottleneck in existing influence function methods by improving scalability to billion-scale models. The paper presents a qualitative use case using the developed multi-stage influence function on the Dolly-v2-3b LLM to qualitatively explain its generations based on pre-training data.
* **Strengths:**
    * **Scalability:** The EK-FAC approximation significantly reduces the computational cost of iHVP calculations, enabling the analysis of much larger models than previously possible.
    * **Practicality:**  The combination of EK-FAC and semantic-similarity-based candidate selection provides a practical way to apply influence functions to real-world LLMs.
    * **Empirical Validation:** The experiments provide strong evidence of the scalability of the EK-FAC approximation and the effectiveness of the multi-stage influence function, showing it outperforms simpler methods.
    * **Case Study:** The qualitative case study with Dolly-v2-3b demonstrates the interpretive power of the method, providing concrete examples of how the multi-stage influence function can be used to understand model behavior.
* **Weaknesses:**
    * **Approximations:** The use of EK-FAC and semantic similarity introduces approximations that may reduce the accuracy of the influence estimates.  While the experiments show that EK-FAC provides a good trade-off between accuracy and efficiency, the impact of these approximations on the interpretability of the results should be considered. There is an observed performance gap between the multi-stage IF and simpler methods like GDP with large damping terms. This underscores the necessity for careful selection of the damping term in order to attain the best performance from the IFs.
    * **Decoder-Only Architecture:** The current implementation is limited to decoder-only transformer architectures, which limits its applicability to other types of models (e.g., encoder-decoder models or diffusion models).
    * **Limited Component Analysis**: Analysis is limited to backbones only (MLP and MHA). Excluding other components could potentially limit improvement of influence estimates,
* **Potential Influence:** The paper is likely to have a significant influence on the field of LLM interpretability and trustworthiness.  It provides a practical and scalable method for understanding the behavior of these models, enabling users to better understand their biases and verify the grounding of their outputs.  The approach could be used to develop more robust and reliable LLMs, and could also be used to identify and mitigate potential risks associated with these models.

**Justification of Score:**

I assign a score of **8**. The paper provides a significant contribution to the field by addressing a crucial problem in LLM interpretability and trustworthiness: tracing predictions back to pre-training data. It introduces a novel combination of techniques that significantly improves the scalability and practicality of influence function methods.  The paper is well-written, the experiments are thorough, and the results are compelling. While the method relies on approximations and is currently limited to decoder-only models, these limitations do not detract from the paper's overall contribution. Furthermore, it highlights an area ripe for subsequent exploration and refinement.

Score: 8

- **Score**: 8/10

### **[Divide-and-Conquer: Cold-Start Bundle Recommendation via Mixture of Diffusion Experts](http://arxiv.org/abs/2505.05035v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper "Divide-and-Conquer: Cold-Start Bundle Recommendation via Mixture of Diffusion Experts" addresses the challenging problem of cold-start bundle recommendation, where new bundles lack sufficient interaction data. The authors propose a novel framework called MoDiffE (Mixture of Diffusion Experts). It leverages a divide-and-conquer strategy: first, it divides the problem into smaller, manageable sub-problems by considering the bundle-level and item-level interactions separately. Second, it conquers each sub-problem using diffusion models to generate representations for cold-start bundles and items, effectively learning to denoise representations without relying on existing embeddings. Finally, it combines the results using a cold-aware hierarchical Mixture of Experts (MoE) architecture, adaptively weighting the contributions of different views (bundle-level and item-level) and experts. The framework uses a multi-stage training pipeline and incorporates a cold-start gating augmentation method to improve training the MoE. The authors empirically validate MoDiffE on three real-world datasets, demonstrating significant performance gains over existing cold-start bundle recommendation methods.

**Critical Evaluation:**

**Novelty:**

The paper has several novel aspects:

1.  **Formalizing Dual-Level Multi-View Complexity:** The paper explicitly acknowledges and formalizes the dual-level (bundle/item) multi-view interaction problem within cold-start bundle recommendation.  This is a significant contribution. Previous approaches often implicitly addressed these complexities without making them explicit.

2.  **Diffusion Models for Cold Start:** The use of diffusion models to generate representations directly, bypassing the need for feature embeddings for cold bundles and items, is a compelling idea.  It addresses the inherent limitation of feature-based methods, which struggle with missing data.

3.  **Cold-Aware Hierarchical MoE:** The MoE architecture, particularly with the cold-start gating augmentation, aims to adaptively handle bundles with varying degrees of cold start. This architecture is key for stable aggregation.

4.  **Divide-and-Conquer Approach:** The divide-and-conquer strategy simplifies what could be an overwhelmingly complex problem. It contributes by providing a structured solution process.

**Significance:**

*   **Improved Performance:** The empirical results clearly demonstrate that MoDiffE outperforms existing state-of-the-art methods in cold-start and all-bundle scenarios.
*   **Practical Applicability:**  The improved performance in the cold-start setting is crucial for real-world applications, as it allows newly introduced bundles to gain visibility and provide relevant recommendations early on.
*   **Framework Adaptability:**  The modularity of MoDiffE allows for the framework to be tailored with different implementations (e.g. different prior-embedding models or different diffusion solver techniques).

**Strengths:**

*   **Well-Defined Problem:** The paper clearly defines the cold-start bundle recommendation problem, emphasizing its dual-level, multi-view complexity.
*   **Novel and Well-Justified Approach:** The proposed MoDiffE framework is innovative and well-motivated. The use of diffusion models, MoE, and the divide-and-conquer strategy are all thoughtfully explained and justified.
*   **Comprehensive Experiments:** The experimental evaluation is extensive, using multiple real-world datasets and comparing against a range of baselines.
*   **Detailed Ablation Study:** The ablation study effectively demonstrates the contribution of each component of MoDiffE.
*   **Insightful Analysis:** The visualizations and case studies provide valuable insights into how MoDiffE works.

**Weaknesses:**

*   **Computational Complexity:** The use of diffusion models can be computationally expensive, which might be a concern for large-scale recommendation systems. Although they use a fast inference solver, the complexity may still be high compared to traditional methods. This is a potential limitation.
*   **Parameter Sensitivity:** Diffusion models have many hyperparameters to be tuned. The process of tuning hyperparameters can be costly.
*   **Black Box Nature:**  While the case study is helpful, diffusion models can be perceived as black boxes.  Further investigation into *why* specific bundled are handled particularly well and others are less so would strengthen the paper. This would help understand model failure modes, which are particularly important with limited data.

**Justification for Score:**

The paper makes a significant contribution to the field of bundle recommendation, particularly in the challenging area of cold start. The explicit acknowledgement of the dual-level multi-view complexity and the innovative use of diffusion models sets this paper apart from prior work.  The experimental results are convincing, and the ablation studies provide valuable insights.  While the computational complexity of diffusion models is a potential limitation, the demonstrated performance gains and the well-defined problem warrant a high score.

**Score: 8.5**

The formalization of the problem and the innovative solution are substantial contributions, and the results are well-supported. The computational concerns are a valid consideration for real-world scaling, but do not detract from the paper's overall significance and impact. A slightly higher score could be warranted if the black box nature of the diffusion model was addressed with explanation methodologies for individual bundle recommendations.

- **Score**: 8/10

### **[WaterDrum: Watermarking for Data-centric Unlearning Metric](http://arxiv.org/abs/2505.05064v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces WaterDrum, a novel data-centric metric for evaluating LLM unlearning based on robust text watermarking. It addresses the limitations of existing utility-centric metrics, which often fail in real-world scenarios where the forget and retain sets have semantically similar content, retraining is impractical, or model owners can manipulate the metric without actual unlearning. The paper also presents WaterDrum-Ax, a new benchmark dataset designed to evaluate unlearning algorithms under these challenging conditions, including data from multiple parties and varying degrees of data similarity. Through experiments, the authors demonstrate that WaterDrum outperforms existing metrics in satisfying several important desiderata for an effective, practical, and resilient unlearning metric.

**Critical Evaluation:**

*   **Novelty:** The paper's primary novelty lies in proposing a data-centric unlearning metric that overcomes limitations of utility-centric approaches. Shifting focus from model performance to directly tracking the presence of data through watermarking is a valuable contribution. The WaterDrum-Ax dataset is also a novel contribution, directly designed to address the limitations in existing datasets for the problem of LLM unlearning. However, building upon the Waterfall framework of Lau et al. (2024) does decrease the innovation slightly.

*   **Significance:** The paper's significance stems from addressing critical issues in LLM unlearning evaluation. By providing a metric and dataset that are robust to real-world challenges (semantic similarity, lack of retraining access, and potential for manipulation), the authors enable more accurate and reliable assessment of unlearning algorithms. This contribution is particularly relevant given the increasing concerns about privacy, copyright, and the ethical implications of LLMs. The focus on query-access, which is practically relevant in many modern LLM applications, is another strength.

*   **Strengths:**
    *   Clear and well-defined desiderata for unlearning metrics.
    *   A practical and resilient data-centric approach based on watermarking.
    *   New benchmark dataset specifically designed to address existing limitations.
    *   Empirical validation demonstrating WaterDrum's superiority.
    *   Demonstration of potential weaknesses of commonly used utility-centric metrics.

*   **Weaknesses:**
    *   The watermarking approach adds complexity to the training data preparation.
    *   The effectiveness of WaterDrum relies on the robustness of the underlying watermarking scheme. Future work could test this more rigorously.
    *   While the authors demonstrate resilience to one specific threat model, further analysis of other potential attacks and defenses is warranted.
    *   While WaterDrum provides a more data-centric perspective, it is important to note that watermarking, by its very nature, adds additional complexity to both the model training and evaluation pipeline.

*   **Impact:** The paper has the potential to significantly influence the field of LLM unlearning. The data-centric perspective and the benchmark dataset could facilitate the development of more effective and practical unlearning algorithms.

**Justification for Score:**

The paper makes a strong contribution to the field of LLM unlearning by identifying limitations in current unlearning metrics and presenting a new approach and dataset to address them. WaterDrum offers a more realistic and practical evaluation framework, which is crucial for advancing the development of effective unlearning techniques. However, there are some weaknesses due to the fact that it builds on previous works.

Score: 8

- **Score**: 8/10

### **[MDAA-Diff: CT-Guided Multi-Dose Adaptive Attention Diffusion Model for PET Denoising](http://arxiv.org/abs/2505.05112v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces MDAA-Diff, a CT-Guided Multi-Dose Adaptive Attention Diffusion Model, designed for denoising low-dose PET (LPET) images. The model addresses the limitations of existing methods by considering both inter-patient variability in dose response and the complementary anatomical information available from CT images. MDAA-Diff incorporates a CT-Guided High-frequency Wavelet Attention (HWA) module to extract anatomical boundary features and a Dose-Adaptive Attention (DAA) module to dynamically integrate dose levels into the denoising process. Experiments on 18F-FDG and 68Ga-FAPI datasets demonstrate superior denoising performance compared to state-of-the-art approaches, particularly in preserving diagnostic quality under reduced-dose conditions.

**Critical Evaluation:**

*   **Novelty:** The paper exhibits several novel aspects.
    *   The integration of CT-guided anatomical information using a High-frequency Wavelet Attention (HWA) module is a significant contribution. Existing methods often overlook the valuable spatial constraints offered by CT imaging in PET denoising. The separation of high-frequency components to enhance edge details is clever.
    *   The Dose-Adaptive Attention (DAA) module addresses a crucial gap in previous research by explicitly modeling dose-dependent relationships. This dynamic integration of dose levels into the attention mechanism provides a unique way to adapt the denoising process to varying acquisition parameters.
    *   The combination of diffusion models with multi-modal fusion and dose adaptation shows an improvement, however, there exist prior works that also combine these.

*   **Significance:**
    *   The work has significant potential to reduce radiation exposure in PET imaging by enabling accurate image reconstruction from lower doses. This is especially beneficial for sensitive populations like children and pregnant women.
    *   The proposed approach addresses a practical challenge in clinical PET imaging, where variations in patient physiology, metabolism, and tracer uptake can affect dose response. By explicitly modeling these dose-dependent relationships, MDAA-Diff offers a more robust solution for LPET denoising.
    *   The reported results show substantial improvements in both quantitative metrics (PSNR and SSIM) and qualitative visual assessments, demonstrating the practical value of the proposed method. The ablation studies provide strong evidence for the effectiveness of both the HWA and DAA modules.
    *   The comparison with several state-of-the-art methods strengthens the credibility of the findings and underscores the advancements achieved by MDAA-Diff.
    *   The code is publicly available for better verification.

*   **Weaknesses:**
    *   While the results look promising, the paper could benefit from a more in-depth analysis of the computational complexity of the proposed approach, especially the wavelet transform and attention mechanisms. This information is crucial for evaluating the practical feasibility of MDAA-Diff in clinical settings.
    *   The paper could also benefit from demonstrating the generalizability of MDAA-Diff to other PET tracers or imaging modalities. The current evaluation is limited to 18F-FDG and 68Ga-FAPI.
    *   The method improves the diffusion based denoising by including CT image features and dose adaptation, however, the improvement over the SOTA is not dramatic.

**Justification for Score:**

The paper presents a novel and significant contribution to the field of low-dose PET imaging. The integration of CT-guided information and dose-adaptive mechanisms within a diffusion model framework is innovative and addresses practical challenges in clinical PET. The results demonstrate promising improvements in denoising performance, and the ablation studies support the effectiveness of the proposed modules. However, the relatively modest improvements over SOTA as well as the points on computational complexity and the need for further generalization warrant a slight reduction of the score.

**Score: 8**

- **Score**: 8/10

### **[FedTDP: A Privacy-Preserving and Unified Framework for Trajectory Data Preparation via Federated Learning](http://arxiv.org/abs/2505.05155v1)**
- **Summary**: Here's a summary and critical evaluation of the FedTDP paper:

**Summary:**

The paper "FedTDP: A Privacy-Preserving and Unified Framework for Trajectory Data Preparation via Federated Learning" addresses two key limitations in trajectory data preparation (TDP):  lack of privacy protection in federated settings and the absence of generalized models applicable across diverse TDP scenarios. The authors propose FedTDP, a framework leveraging Large Language Models (LLMs) for TDP in federated environments.  FedTDP incorporates a trajectory privacy autoencoder (TPA) for secure data transmission, a trajectory knowledge enhancer (TKE) to improve model learning of TDP-related knowledge, and federated parallel optimization (FPO) to enhance training efficiency. Experimental results on real datasets and TDP tasks are presented to demonstrate FedTDP's superiority over existing methods.

**Critical Evaluation:**

*   **Novelty:** The paper's primary novelty lies in combining federated learning, LLMs, and specialized components (TPA, TKE, FPO) to address TDP in a privacy-preserving and generalizable manner. While individual components (federated learning for privacy, LLMs for generalizability) are not novel *per se*, their integration within a unified framework specifically tailored for trajectory data preparation represents a significant contribution.  The TPA, TKE, and FPO modules are also novel designs tailored for the federated trajectory data preparation scenario. Existing methods for privacy either degrade performance or are unsuitable for the federated setting. The TKE module addresses a critical gap: how to adapt LLMs to understand trajectory data specifics. And the FPO design looks to significantly improve training efficiency. The introduction of the Small Language Model (SLM) is also a useful pragmatic approach.

*   **Significance:** Trajectory data is increasingly important for applications like traffic management and urban planning. Ensuring data quality and privacy in federated settings is crucial for realizing the potential of these applications.  The proposed FedTDP framework holds the potential to significantly improve the quality and accessibility of trajectory data for various analytical tasks while adhering to privacy regulations. The modular design further allows for easier adaptation to different datasets or TDP tasks, making it practical for real-world scenarios. The thorough experimental evaluation across diverse datasets and TDP tasks strengthens the paper's claims and demonstrates the effectiveness of the FedTDP framework. Also the strong experimental results showing improvements over baselines show the value of this approach. The experiments are performed on a realistic distributed hardware setup to further enhance the experimental results.

*   **Strengths:**
    *   Well-defined problem and clear motivation.
    *   Novel integration of federated learning, LLMs, and specialized modules for TDP.
    *   Detailed explanation of the FedTDP framework and its components.
    *   Thorough experimental evaluation on real datasets and diverse TDP tasks.
    *   Demonstrated improvements over existing methods.
    *   Addresses a practical and important problem in the field.

*   **Weaknesses:**
    *   The complexity of the framework might limit its adoption.  A simplified version or a more automated deployment process would enhance usability.
    *   The paper could benefit from a more in-depth analysis of the computational overhead associated with TPA, TKE, and FPO. While FPO aims to improve efficiency, a clearer understanding of the trade-offs is needed.
    *   There's a need for exploration of the limits of "trajectory data privacy". How much can you infer from embeddings, even with the additional security methods in this paper?

*   **Potential Influence:** The paper has the potential to influence research in several ways:
    *   It could inspire further work on privacy-preserving TDP methods.
    *   It demonstrates the feasibility of using LLMs for TDP and could stimulate research on adapting LLMs for other data preparation tasks.
    *   It could lead to the development of more generalizable and efficient TDP frameworks.
    *   Provides a practical approach for deploying LLMs to solve real-world problems within federated environments

**Score: 8**

**Justification:**

The paper presents a novel and significant contribution to the field of trajectory data preparation. The integration of federated learning and LLMs, along with the specialized components, addresses a critical gap in the existing literature. The thorough experimental evaluation provides strong evidence of the framework's effectiveness. The weaknesses (complexity, computational overhead) are areas for future research, but do not diminish the core contribution of the paper. It is likely to influence future research in this area by spurring work on privacy-preserving and generalizable TDP methods.

- **Score**: 8/10

### **[Revealing Weaknesses in Text Watermarking Through Self-Information Rewrite Attacks](http://arxiv.org/abs/2505.05190v1)**
- **Summary**: Here's a concise summary and critical evaluation of the provided paper:

**Summary:**

The paper introduces a novel paraphrasing attack, called Self-Information Rewrite Attack (SIRA), to evaluate the robustness of text watermarking techniques used in Large Language Models (LLMs). SIRA exploits a vulnerability in existing watermarking algorithms that embed statistical signals in high-entropy tokens. The attack identifies potential watermark tokens by calculating self-information and then selectively rewrites them. The authors demonstrate that SIRA achieves high success rates (near 100%) on several recent watermarking methods, using significantly less computational resources compared to existing paraphrasing attack methods. The paper also highlights the transferability of the proposed attack across different LLMs and its ability to operate in a black-box setting, making it a practical and potent tool for evaluating watermark robustness.

**Critical Evaluation:**

*   **Novelty:** The core novelty lies in the identification and exploitation of a specific vulnerability in LLM watermarking schemes that rely on high-entropy tokens. Leveraging self-information to target these tokens for rewriting provides a new and efficient attack strategy compared to existing brute-force paraphrasing attacks. The targeted approach based on self-information is significantly less explored than other attack strategies. This is a key strength.

*   **Significance:** The paper’s significance lies in its ability to expose a widespread vulnerability in current watermarking algorithms and propose a simple, efficient, and transferable attack method. SIRA's efficiency, coupled with its black-box nature, make it a practical tool for rigorously evaluating watermark robustness. The results suggest that existing watermarking techniques are less robust than previously thought. This work emphasizes the urgency for developing more robust and adaptive watermarking strategies. The paper also provides a theoretical basis for the success of the attack, which adds credibility.

*   **Strengths:**
    *   Clear problem definition and well-defined attack strategy.
    *   Comprehensive experimental evaluation demonstrating high attack success rates across multiple watermarking methods.
    *   Detailed cost analysis showing SIRA's computational efficiency compared to existing methods.
    *   Effective visualization and text quality analysis to illustrate the attack's impact and preservation of semantic content.
    *   Theoretical justification provided for the attack method.
    *   Strong results on OpenGen and other watermarking schemes (adaptive/waterfall).

*   **Weaknesses:**
    *   While the paper demonstrates high success rates, further investigation into the types of watermark algorithms that are most vulnerable to SIRA would be beneficial.
    *   The paper focuses primarily on black-box attacks. While this is a strength in terms of practicality, exploring potential defenses or mitigations against SIRA could improve the comprehensiveness of the work. The investigation of mitigation strategies is important in future research.

*   **Potential Influence:** The paper's findings are likely to significantly influence the development of more robust watermarking techniques for LLMs. SIRA serves as a valuable benchmark for evaluating the effectiveness of future watermarking methods, pushing the field towards more resilient and secure solutions. The focus on self-information is also likely to influence future research on watermark detection and removal, leading to more sophisticated and targeted approaches. The work might encourage a shift in the design philosophy of watermarking.

*   **Rigorous Rationale:** The score is justified by the clear identification of a vulnerability, the practical and efficient attack proposed, and the comprehensive experimental validation across a range of existing watermarking methods and datasets. The black-box nature and high transferability adds significantly to the significance and the fact the vulnerability is found in multiple schemes. The theoretical basis adds weight to the empirical results. The main weakness is the lack of investigation into defenses.

Score: 8

- **Score**: 8/10

### **[Diffusion Model Quantization: A Review](http://arxiv.org/abs/2505.05215v1)**
- **Summary**: Here's a summary and critical evaluation of the provided survey paper:

**Summary:**

The paper presents a comprehensive survey of recent advancements in diffusion model quantization, a technique crucial for deploying these models on resource-constrained devices. It begins by outlining the background and preliminary concepts related to diffusion models and quantization, highlighting the unique challenges encountered when quantizing diffusion models due to their multi-step denoising process and complex architectures (U-Nets and Diffusion Transformers). The survey proposes a taxonomy of quantization algorithms, classifying them based on model architecture (U-Net vs. DiT) and quantization strategy (PTQ vs. QAT). It provides a detailed analysis of various techniques, including calibration strategy customization, temporal dynamic quantization, error correction mechanisms, and methods for preserving text-image consistency.  The survey also includes benchmarking experiments on class-conditional, unconditional, and text-guided image generation tasks, providing both quantitative results and qualitative analyses of quantization artifacts like color bias and loss of detail. Finally, the paper concludes with a discussion of future research directions.

**Critical Evaluation:**

*   **Novelty:** While the paper doesn't introduce a novel algorithm or method, its novelty lies in being the first comprehensive survey specifically focusing on the quantization of *diffusion models*. Previous surveys have largely focused on model quantization in general or more specific architectures like CNNs or large language models.  The survey fills a gap by systematically organizing and analyzing the rapidly growing literature in this specific subfield. The identified challenges particular to diffusion models (multi-step issues, time-step embedding problems) are valuable. The proposed taxonomy, while based on existing categories (PTQ, QAT), is tailored to the unique techniques used in diffusion model quantization, making it helpful for researchers in the field.
*   **Significance:** The significance of the survey is substantial because it addresses a critical need in the diffusion model community. Diffusion models, while powerful, are computationally expensive. Quantization is essential for their widespread deployment on edge devices and in real-time applications. By consolidating the current state-of-the-art, identifying key challenges, and providing a structured analysis of existing solutions, the survey accelerates research progress in this area. The quantitative benchmarking and qualitative analysis further enhance the survey's practical value, offering insights into the trade-offs between different quantization techniques.  The clear delineation of future research directions further solidifies its role as a guide for the field.
*   **Strengths:**
    *   **Comprehensiveness:** The survey appears to cover a wide range of relevant papers, including very recent work (up to March 2025, as mentioned in the survey).
    *   **Clarity and Organization:** The taxonomy is well-defined, and the explanations of different quantization techniques are clear and concise.
    *   **Balanced Analysis:** The inclusion of both quantitative and qualitative analyses strengthens the evaluation.  The discussion of visual artifacts and trajectory-level effects provides a more nuanced understanding of quantization impact.
    *   **Identification of Key Challenges:** The paper explicitly identifies challenges that are particular to diffusion models, which adds depth.
*   **Weaknesses:**
    *   **Limited Critical Assessment of Individual Techniques:** While the survey covers many methods, it could benefit from a more in-depth critical comparison of the individual techniques' strengths and weaknesses *beyond* just the benchmarking results. A discussion about computational complexity, hardware requirements for specific methods, and ease of implementation would be valuable.
    *   **Dependency on Empirical Results:** The survey relies heavily on experimental evaluations, which is appropriate, but more in-depth theoretical analysis explaining *why* certain methods work better than others would be beneficial.  For example, diving more deeply into the math behind how the different loss functions minimize the specific types of error in quantized models may strengthen the discussions.
    *   **Potential for Overlap with General Quantization Surveys:** Since many of the core quantization methods discussed (e.g., LSQ, SmoothQuant) are not specific to diffusion models, there's a risk of some overlap with broader quantization survey literature. However, the survey's focus on the adaptations and specific challenges related to diffusion models mitigates this concern to a good degree.
    *   **Future Directions are broad:** While future directions are identified, they remain broad. A more specific call for research could increase their value to the community.

**Justification for Score:**

The survey provides a valuable service to the diffusion model community by consolidating and analyzing the growing body of work on quantization. It correctly identifies the unique challenges in this field and offers a helpful taxonomy for understanding different approaches. The inclusion of benchmark results and qualitative analyses strengthens its practical value. While the survey could benefit from a more in-depth critical comparison of individual methods and a theoretical discussion on experimental evidence, its comprehensiveness, clarity, and timeliness make it a significant contribution. The comprehensive, dedicated focus on the unique challenges is the key reason for a high (though not perfect) score.

Score: 8

- **Score**: 8/10

### **[Software Development Life Cycle Perspective: A Survey of Benchmarks for CodeLLMs and Agents](http://arxiv.org/abs/2505.05283v1)**
- **Summary**: Here's a summary and evaluation of the paper:

**Summary:**

The paper provides a comprehensive survey of benchmarks for CodeLLMs and agents, viewed from the perspective of the Software Development Life Cycle (SDLC). It analyzes 181 benchmarks from 461 papers, categorizing them according to the different phases of the SDLC (Requirements Engineering, Design, Software Development, Testing, and Maintenance). The authors find an imbalance in coverage, with a significant focus on the software development phase and limited attention to the earlier phases like Requirements Engineering and Design.  They also analyze benchmark usage, programming languages, and the evolution of benchmarks towards more realistic scenarios. Finally, they highlight the limitations of existing benchmarks and propose future directions for research, including standardized benchmarks, cross-phase evaluations, non-functional requirements, multi-modal benchmarks, multi-agent systems, and code-centric evaluation frameworks.

**Critical Evaluation:**

**Strengths:**

*   **Comprehensive Scope:** The paper provides a valuable service to the community by systematically organizing and categorizing a large number of benchmarks.  The SDLC lens is a useful framework for understanding the strengths and weaknesses of the current evaluation landscape.
*   **Clear Identification of Gaps:** The paper convincingly demonstrates the imbalance in benchmark coverage across different SDLC phases. Highlighting the neglect of Requirements Engineering and Design phases is a crucial observation.
*   **Forward-Looking Recommendations:** The proposed future directions are well-reasoned and address the identified limitations. Suggestions like focusing on non-functional requirements, multi-modal inputs, and human-model collaboration are particularly relevant.
*   **Well-Structured and Readable:** The paper is logically organized and clearly written, making it accessible to a broad audience.
*   **Reproducibility is well articulated:** The inclusion of all papers and search terms makes it easy to reproduce and validate.

**Weaknesses:**

*   **Limited Depth of Analysis for Individual Benchmarks:** While the paper covers a large number of benchmarks, the individual descriptions of each benchmark are relatively brief. A more in-depth analysis of the individual tasks and metrics within each benchmark could have been helpful.
*   **Potential for Subjectivity in Categorization:** Classifying benchmarks into specific SDLC phases can be subjective, as some benchmarks may touch upon multiple phases. More detailed explanations or examples for the categorization decisions would strengthen the argument.

**Novelty and Significance:**

The novelty lies in the systematic application of the SDLC framework to analyze the landscape of CodeLLM benchmarks. This approach is significant because it highlights the practical implications of evaluation practices and offers a structured roadmap for future benchmark development. The paper successfully identifies the areas where research is lacking and suggests directions that align with the evolving needs of real-world software engineering. The comprehensive analysis and forward-looking recommendations make it a valuable resource for researchers and practitioners in the field.
The novelty and contribution are high, but some points lack specific detail (e.g. categorization decisions).
Overall a well constructed and valuable survey with a new take on existing benchmarks.

Score: 8

- **Score**: 8/10

### **[Denoising Diffusion Probabilistic Models for Coastal Inundation Forecasting](http://arxiv.org/abs/2505.05381v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Denoising Diffusion Probabilistic Models for Coastal Inundation Forecasting":

**Summary:**

The paper introduces DIFF-FLOOD, a novel probabilistic spatiotemporal forecasting method for coastal inundation based on denoising diffusion models. DIFF-FLOOD predicts inundation levels by considering both spatial context (neighboring inundation levels and digital elevation data) and temporal context (historical inundation data and other relevant covariates like tidal cycles). The model uses convolutional neural networks and a cross-attention mechanism to capture the complex spatiotemporal dynamics in the data.  The authors trained and tested DIFF-FLOOD on data from the Eastern Shore of Virginia and showed that it outperforms existing forecasting methods in terms of prediction performance and scalability. The paper also demonstrates the model's utility in answering scenario-based queries of interest to policymakers.

**Critical Evaluation:**

*   **Novelty:**  The application of diffusion models to coastal inundation forecasting is a novel approach.  While diffusion models have been used for time-series forecasting and spatiotemporal prediction, the specific adaptation and optimization for high-resolution coastal inundation with the integration of both spatial and temporal context appears to be a significant contribution. The proposed model and its demonstrated ability to provide probabilistic forecasts at a fine spatial scale, suitable for decision making are strengths.

*   **Significance:** Coastal inundation forecasting is a practically important problem. Improved forecasting methods can lead to better preparedness and mitigation strategies, reducing potential damage to communities.  The demonstrated performance of DIFF-FLOOD in comparison to existing methods highlights its potential to improve real-world forecasting capabilities. The ability of the model to address questions such as "What is the probability that the flooding level in an area A will be above d units within the next T hours?" shows the real application of the model.

*   **Strengths:**

    *   **Strong Performance:** The experimental results convincingly demonstrate the superiority of DIFF-FLOOD over several baseline methods, including state-of-the-art diffusion-based models for spatiotemporal forecasting (DiffSTG).
    *   **Scalability:** The paper demonstrates that DIFF-FLOOD scales better to large, high-resolution datasets than competing methods, which is crucial for practical applications.
    *   **Ablation Study:** The ablation study effectively isolates the contributions of different components (elevation data, temporal covariates) to the overall performance of the model.
    *   **Practical Applicability:**  The discussion of inundation scenario queries and their relevance to policymakers demonstrates the potential for real-world impact.
    *   **Clarity and Completeness:** The paper is well-written and provides sufficient details about the model architecture, training procedure, and experimental setup.
*   **Weaknesses:**

    *   **Dataset Limitations:**  While the Tidewatch dataset is a good choice, it has its own limitations. Further evaluation on diverse datasets from different coastal regions with varying environmental conditions would strengthen the generalizability of the findings.
    *   **Complexity:** Diffusion models are computationally expensive to train. The paper could benefit from a more in-depth discussion of the computational resources required and strategies for optimizing training time.
    *   **Comparison Details**: It could be helpful to know more details of parameters of comparison of different model training details.

*   **Potential Influence:** This paper has the potential to influence the field by demonstrating the effectiveness of diffusion models for coastal inundation forecasting. It could spur further research into applying these models to other environmental forecasting problems and developing more efficient diffusion model architectures for spatiotemporal data. The model and its approach could be used for several real world decision-making.

**Justification of Score:**

The paper presents a novel and significant contribution to the field of coastal inundation forecasting. The DIFF-FLOOD model demonstrably outperforms existing methods in both prediction accuracy and scalability, addressing a crucial practical challenge. The inclusion of the ability to answer scenario based queries adds to the usefulness of the model. While there are limitations regarding dataset diversity and computational cost, the paper is well-executed and the results are compelling.

Score: 8

- **Score**: 8/10

### **[Frame In, Frame Out: Do LLMs Generate More Biased News Headlines than Humans?](http://arxiv.org/abs/2505.05406v1)**
- **Summary**: Here's a summary and critical evaluation of the provided paper:

**Summary:**

The paper investigates whether Large Language Models (LLMs) generate more biased news headlines (framed content) compared to humans. The authors perform a comparative analysis across various LLM families (both out-of-the-box and fine-tuned) using the XSUM dataset, detecting framing through the same method in Pastorino et al. (2024). The key findings are: LLMs generally frame information more than humans, particularly in politically and socially sensitive contexts; Fine-tuning helps reduce framing tendencies; Larger models demonstrate a higher propensity for framing; and framing rates vary depending on the topic and model architecture. They also find a weak correlation between the framing tendency and length of the text.

**Critical Evaluation:**

*   **Novelty:** The paper addresses a crucial and timely question: the potential for LLMs to amplify or introduce framing biases in news generation. While framing analysis itself is not new, applying it to a diverse set of modern LLMs and comparing them directly to human baselines is a valuable contribution. The finding that LLMs are consistently more prone to framing than humans, even in supposedly neutral topics, is a novel and concerning result. The paper also presents novelty in considering different models and fine-tuning techniques, in addition to exploring framing patterns for various topics.

*   **Significance:** The findings have significant implications for the responsible development and deployment of LLMs in news and content creation. If LLMs inherently tend to frame information more than humans, it raises serious concerns about the potential for these models to shape public opinion in unintended or even harmful ways. This has direct relevance to AI ethics, journalism, and media studies. The research provides empirical evidence to support the growing concern about bias in AI-generated content, encouraging further investigation into mitigation strategies. It also pushes the evaluation metrics of LLMs beyond just accuracy and fluency and brings attention to fairness and bias.

*   **Strengths:**
    *   **Comprehensive Analysis:** The paper analyzes a wide range of LLM architectures, including both out-of-the-box and fine-tuned models.
    *   **Clear Methodology:** The methodology is well-defined, building upon existing techniques for framing detection (Pastorino et al., 2024). It makes the choice of prompting scheme very clear.
    *   **Empirical Evidence:** The paper provides empirical evidence to support its claims, using statistical tests (t-tests, Pearson correlation) to validate the significance of its findings.
    *   **Timeliness:** The topic is highly relevant given the increasing adoption of LLMs for content creation.

*   **Weaknesses:**
    *   **Dataset Limitations:** The XSUM dataset, while a good benchmark for summarization, focuses on BBC news articles and may not fully represent the breadth of news content or the nuances of different framing techniques used across various media outlets.
    *   **Framing Detection Method:** The study depends on the jury-based framing detection method, in Pastorino et al. (2024). While this is the best performing approach currently, framing detection itself is still an evolving area of research, and improvements in detection methods could impact the results.
    *   **Limited Mitigation Exploration:** While the paper identifies the problem, it does not delve deeply into specific mitigation strategies or techniques to reduce framing bias in LLMs, though the results on fine-tuning offer a starting point.
    *   **Simple Keyword-based Topic Analysis:** More sophisticated topic modeling methods could provide a better understanding of the relationship between topics and framing tendencies.

*   **Potential Influence:** The paper is likely to influence the field by raising awareness of the issue of framing bias in LLMs and prompting further research into:
    *   Developing better evaluation metrics that incorporate fairness and bias assessments.
    *   Exploring mitigation techniques to reduce framing bias in LLMs.
    *   Investigating the impact of different training datasets and model architectures on framing tendencies.
    *   Creating more comprehensive and diverse datasets for framing analysis.

**Overall:**

This is a significant contribution to the field. The paper provides compelling evidence that LLMs are more prone to framing than humans, highlighting a critical issue that needs to be addressed. The findings will likely stimulate further research and development in this area, driving the creation of more responsible and unbiased AI systems for content generation. While the analysis could be strengthened with more advanced topic modeling, framing detection, and dataset diversification, the paper's novel insights and clear presentation merit a high score.

Score: 8

- **Score**: 8/10

### **[TransProQA: an LLM-based literary Translation evaluation metric with Professional Question Answering](http://arxiv.org/abs/2505.05423v1)**
- **Summary**: Here's a summary and critical evaluation of the TRANProQA paper:

**Summary:**

The paper introduces TRANSPROQA, a novel Large Language Model (LLM)-based, reference-free, Question Answering (QA) framework specifically designed to evaluate literary translations.  It addresses the limitations of existing MT evaluation metrics which prioritize mechanical accuracy over artistic and cultural nuances critical in literary works.  TRANSPROQA integrates insights from professional literary translators by focusing on elements like literary devices, cultural understanding, and authorial voice through a curated question set.  The paper demonstrates that TRANSPROQA outperforms current metrics, including fine-tuned versions of XCOMET-XL, in correlation with human judgments and adequacy assessments. The framework also exhibits robust performance with open-source models, making it an accessible tool for literary translation evaluation.  The incorporation of professional translator insights as weights further enhances the framework's accuracy, approaching human-level evaluation performance.

**Critical Evaluation:**

**Strengths:**

*   **Addresses a Significant Problem:** The paper tackles a critical gap in MT evaluation, which is the lack of metrics tailored for the subjective and nuanced qualities of literary translation.
*   **Novel Approach:** The QA-based framework leveraging professional translator insights is a novel approach and offers a more holistic evaluation compared to traditional metrics focused on lexical overlap or semantic similarity.
*   **Strong Empirical Results:** The paper presents robust empirical evidence demonstrating that TRANSPROQA outperforms state-of-the-art metrics, including fine-tuned versions, in both correlation with human judgments and adequacy assessments.
*   **Translator-Centric:** The inclusion of literary translator insights in question selection and scoring adds significant value and moves the evaluation closer to human expertise.
*   **Accessibility:** Demonstrated effectiveness with open-source LLMs makes the tool accessible to a wider audience, especially in contexts where local processing is required due to copyright or ethical considerations.
*   **Reproducibility:** The paper mentions that the code and datasets are publicly available for further research.

**Weaknesses:**

*   **Limited Language Coverage:** The evaluation primarily focuses on high-resource language pairs, limiting the generalizability of the findings to low-resource languages.
*   **Paragraph-Level Evaluation:**  The evaluation is conducted at the paragraph level, which may not capture elements that are relevant only when considering larger portions of text (e.g., coherence, character development)
*   **Question-Level Instructions are less effective:**  QuestionStep shows a reduction in translation effectiveness, indicating a challenge for LLMs to incorporate instructions at a finer level.
*   **Dependence on LLM Performance:** The metric relies on the ability of LLMs to accurately answer the evaluation questions. While the framework aims to mitigate this through question selection, it still introduces a potential bias or limitation.
*   **Survey response:** The survey response is the key for weighting, a larger group will likely lead to even better result.

**Novelty and Significance:**

The primary novelty lies in the design of an LLM-based metric that is specifically geared toward evaluating literary translation quality, taking into account insights from expert human translators. By moving away from traditional methods, TRANSPROQA has the potential to address critical shortcomings of existing approaches. By focusing on elements such as cultural nuances and literary devices, the paper has the ability to offer a new way to approach machine translation evaluation and potentially improve LLM's capabilities when working with artistic content. The significance is enhanced by demonstrating that even accessible, open-source models can be effective with this framework.

**Influence:**

The paper has the potential to influence future research in MT evaluation by:

*   Encouraging the development of more specialized metrics for creative domains.
*   Highlighting the importance of incorporating expert human knowledge in automated evaluation systems.
*   Providing a framework for evaluating the performance of LLMs on literary translation tasks.
*   Raising awareness of the potential biases in existing evaluation metrics.

**Justification for Score:**

I assign a score of **8**.  The paper presents a significant contribution to the field by addressing a clear need for specialized evaluation metrics in literary translation. The novelty of the approach is solid, and the empirical results convincingly demonstrate the superiority of TRANSPROQA over existing methods.  The strengths of the paper, particularly the translator-centric design and empirical validation, are considerable. The reliance on LLMs also brings certain limitations, but are mitigated by the question selection process and the demonstrated ability to work well with open source models.

Score: 8

- **Score**: 8/10

### **[Mogao: An Omni Foundation Model for Interleaved Multi-Modal Generation](http://arxiv.org/abs/2505.05472v1)**
- **Summary**: Okay, I will provide a summary and critical evaluation of the provided paper, based on the text available.

**Summary:**

The paper introduces Mogao, a unified framework for interleaved multi-modal generation. Mogao aims to improve upon existing models which are often limited to single-modal generation conditioned on multiple modalities. The core idea is to enable interleaved generation of text and images through a causal approach. This involves several architectural innovations: a deep-fusion design, dual vision encoders (VAE and ViT), interleaved rotary position embeddings, and multi-modal classifier-free guidance.  The model combines autoregressive text generation with diffusion-based image synthesis. The paper also introduces a large-scale interleaved multi-modal dataset and an efficient training strategy.  Experiments demonstrate state-of-the-art performance in multi-modal understanding and text-to-image generation, as well as emergent capabilities in zero-shot image editing and compositional generation. The authors position Mogao as a practical omni-modal foundation model, paving the way for future unified multi-modal systems.

**Critical Evaluation:**

*   **Novelty:** The novelty lies primarily in the architectural combination and the interleaved training approach. While individual components like deep fusion, dual encoders, and rotary embeddings are not entirely new, their integration within this specific framework and for this specific task (interleaved generation) seems to be the key contribution. The dual CFG mechanism to address image repetition in interleaved generation also appears novel. The creation of a large, interleaved multi-modal dataset is another contribution, albeit dependent on in-house resources.

*   **Significance:** The significance of the work is considerable, if the claimed results hold up under scrutiny. The ability to generate coherent interleaved text and images has practical implications for content creation, interactive AI systems, and other applications. By demonstrating that such a model can be trained effectively and achieve state-of-the-art performance, the authors are pushing the boundaries of multi-modal AI. The architectural choices (deep fusion with decoupled layers) and training strategies (efficient complete teacher forcing) also have the potential to influence future research.

*   **Strengths:**
    *   Clear problem definition: The paper clearly identifies the limitations of existing multi-modal models.
    *   Comprehensive approach: The paper addresses both architectural and training challenges.
    *   Strong empirical results: The paper claims state-of-the-art performance on multiple benchmarks. The human evaluation also suggests a notable improvement in subjective quality compared to existing solutions.
    *   In-depth ablation studies: The ablation studies justify the design choices made.
    *   Demonstrates compelling emergent capabilities: Zero-shot image editing and composition are valuable emergent abilities.

*   **Weaknesses:**
    *   Dependency on proprietary dataset: The use of an in-house dataset, while understandable, makes it difficult for other researchers to directly replicate the results or compare against the same training data.
    *   Incremental novelty: While the *combination* of techniques is novel, the individual components are largely based on existing research. The degree of engineering effort and hyperparameter tuning required to make them work together is not fully clear from the paper.
    *   Limited details on the training data creation process: More detail on the nature of the created dataset and how it was curated would be helpful. For example, what sources were used, how was filtering and cleaning done, etc.?
    *   Potential for bias: The reliance on web data raises concerns about potential biases in the generated content. This is not addressed in the paper.

*   **Justification of Score:**
    The paper presents a solid contribution to the field of multi-modal AI. The architecture and training techniques are well-motivated, and the experimental results are promising. While the novelty of individual components may be incremental, the overall system represents a significant advance in interleaved multi-modal generation. The key weakness is the dependence on a proprietary dataset, limiting reproducibility. The paper also does not fully address potential risks relating to the potential for bias. Because of these weaknesses, the paper does not merit a score in the 9-10 range.

Score: 8

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
### **[PrimitiveAnything: Human-Crafted 3D Primitive Assembly Generation with Auto-Regressive Transformer](http://arxiv.org/abs/2505.04622v1)**
### **[EchoInk-R1: Exploring Audio-Visual Reasoning in Multimodal LLMs via Reinforcement Learning](http://arxiv.org/abs/2505.04623v1)**
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
