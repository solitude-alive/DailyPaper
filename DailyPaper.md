# The Latest Daily Papers - Date: 2025-08-03
## Highlight Papers
### **[MoGA: 3D Generative Avatar Prior for Monocular Gaussian Avatar Reconstruction](http://arxiv.org/abs/2507.23597v1)**
- **Summary**: Here's a summary and critical evaluation of the MoGA paper:

**Summary:**

The paper introduces MoGA, a novel method for reconstructing high-fidelity 3D Gaussian avatars from a single monocular image. The core idea is to combine the strengths of a generative 3D avatar model (learned prior) with the ability of multi-view diffusion models to hallucinate unseen views. MoGA leverages the generative avatar model for initialization, regularization, and pose optimization when fitting it to synthetic views produced by the diffusion model.  The generative model utilizes a 2D Gaussian Splatting anchored to a parametric body template, and includes a deformation module for pose control. By integrating the generative prior, the method aims to overcome the limitations of relying solely on sparse and inconsistent synthetic images from multi-view diffusion. Experiments show superior results compared to state-of-the-art techniques, particularly in handling challenging poses, clothing styles, and self-occlusion in in-the-wild images. The resulting Gaussian avatars are inherently animatable.

**Critical Evaluation:**

*   **Novelty:** The primary novelty lies in the synergistic combination of a 3D generative avatar prior with a 2D multi-view diffusion model for single-image avatar reconstruction.  While both generative 3D models and multi-view diffusion have been used previously, their integration within an optimization-based framework specifically for Gaussian avatar reconstruction is a significant contribution. Using Gaussian Splatting for this task, while not entirely unique, is well-justified by its rendering efficiency. Model inversion is a standard procedure, but its application in this context is nontrivial and leads to improved performance. The authors introduce specific losses and optimization strategies tailored to Gaussian avatar fitting.

*   **Significance:** The problem of single-view 3D avatar reconstruction is important for various applications (AR/VR, gaming, etc.).  MoGA addresses key limitations of existing methods, such as 3D inconsistency, unrealistic artifacts, and poor generalization to diverse poses and clothing.  The improved handling of self-occlusion is particularly valuable.  The end-to-end nature of the method, resulting in animatable avatars without post-processing, adds to its practical significance. The demonstrated ability to work on in-the-wild images is a crucial step towards real-world usability.

*   **Strengths:**
    *   Well-motivated approach addressing critical limitations of existing methods.
    *   Elegant combination of generative prior and multi-view diffusion.
    *   Clear description of the method and its components.
    *   Strong experimental results, both quantitatively and qualitatively.
    *   Good generalization to in-the-wild images.
    *   Demonstrated animatability of the reconstructed avatars.

*   **Weaknesses:**
    *   Relies on a pre-trained multi-view diffusion model, inheriting its limitations (e.g., potential biases in the training data, computational cost).
    *   While Gaussian Splatting provides rendering efficiency, the representation itself might not be as editable or controllable as other alternatives (e.g., mesh-based models).
    *   The quantitative evaluation relies on standard metrics, but more specialized metrics (e.g., for assessing the quality of the reconstructed geometry in occluded regions) could strengthen the analysis.
    *   While the paper mentions optimizing pose parameters with a photometric loss, it doesn't provide extensive analysis of the robustness of pose estimation.

*   **Potential Impact:**
    MoGA has the potential to significantly impact the field of 3D human avatar reconstruction.  Its ability to generate high-quality, animatable avatars from single images could lead to wider adoption of avatar-based technologies. The framework provides a strong foundation for future research, such as exploring more advanced generative priors, improving the robustness of pose estimation, and developing more efficient rendering techniques. The project's code and models being publicly available will likely increase its adoption and impact.

*   **Justification for the Score:**
The paper presents a well-executed and novel approach that addresses important limitations in single-view avatar reconstruction. The experimental results are convincing, demonstrating a significant improvement over state-of-the-art methods. While the reliance on pre-trained models and the use of Gaussian Splatting are not without limitations, the overall contribution is substantial. The ability to generate high-fidelity, animatable avatars from single in-the-wild images makes the technique valuable for many real-world applications.

**Score: 8**

- **Score**: 8/10

### **[TweakLLM: A Routing Architecture for Dynamic Tailoring of Cached Responses](http://arxiv.org/abs/2507.23674v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "TweakLLM: A Routing Architecture for Dynamic Tailoring of Cached Responses":

**Summary:**

The paper introduces TweakLLM, a novel caching architecture designed to improve the efficiency of Large Language Model (LLM) deployments. TweakLLM tackles the challenges of traditional semantic caching, which often struggles with precision due to the sensitivity of LLM responses to small changes in user queries.  The architecture consists of two tiers: (1) a semantic cache lookup that uses a robust embedding model to retrieve candidate cached responses and (2) a lightweight LLM that dynamically adapts these cached responses to the specific nuances of the incoming prompt. The authors conduct extensive experiments using real-world datasets, including user studies and multi-agent LLM debates, demonstrating that TweakLLM maintains response quality comparable to frontier models while significantly improving cache effectiveness, leading to cost and latency reductions.

**Critical Evaluation:**

**Novelty:**

The novelty of TweakLLM lies in its two-tiered approach to caching.  While semantic caching is a well-established concept, the combination of a semantic lookup with a dynamic refinement step using a smaller LLM is a distinctive contribution. This architecture addresses the limitations of solely relying on semantic similarity, which can lead to both false positives (irrelevant responses) and false negatives (missed caching opportunities).  The idea of using a smaller, cost-effective LLM to tailor cached responses to better match the current prompt demonstrates an innovative approach to optimizing LLM deployments.  The integration of LLM routing and semantic caching is well-executed.

**Significance:**

The significance of this work is substantial.  As LLMs become increasingly prevalent, the cost of inference becomes a major concern.  TweakLLM offers a practical solution to reduce these costs without sacrificing response quality, making LLMs more accessible and sustainable for high-volume applications.  The user studies and LLM-as-evaluator experiments provide strong evidence supporting the effectiveness of TweakLLM. The quantitative analysis of cost savings based on real-world datasets further strengthens the paper's impact. The open-sourcing of their framework and evaluation pipeline is also a significant contribution, enabling further research and adoption of their approach. The paper also correctly identifies and addresses a crucial challenge in LLM caching, namely precision, where even minor input variations may have a significant impact.

**Strengths:**

*   **Well-Defined Problem:** The paper clearly identifies and articulates the limitations of existing semantic caching approaches.
*   **Novel Architecture:** TweakLLM provides an innovative architecture that addresses these limitations in a resource-efficient manner.
*   **Comprehensive Evaluation:** The authors conduct extensive experiments using a variety of datasets and evaluation metrics, including user studies and LLM-as-evaluator debates.
*   **Strong Results:** The results demonstrate that TweakLLM maintains response quality while significantly improving cache effectiveness.
*   **Practical Implications:** The paper offers practical guidance for parameter tuning and highlights the potential for significant cost savings.
*   **Open-Source:**  The availability of the code and evaluation pipeline promotes reproducibility and further research.
*   **Real-world applicability:** The study on datasets like LMSYS-Chat-1M strengthens the real-world use of the approach.

**Weaknesses:**

*   **Limited Conversation Evaluation:** The evaluation primarily focuses on single-turn interactions. The authors acknowledge that evaluating TweakLLM's performance on multi-turn conversations is a future direction.
*   **Specificity of LLM Choice:** The paper is specific to GPT-4o and Llama 3.1. The results need to be evaluated on different LLM pairs.
*   **Generalisability of parameters:** More insight could have been provided on how to effectively tune the critical hyperparameter; the cosine similarity threshold. It is stated that lower thresholds are required for lower-quality modification LLMs; however, specific thresholds for model-to-model pairings could strengthen this.
*   **Cache Management Strategy:** The current append-only caching strategy is simplistic. More sophisticated cache eviction policies could further improve performance, as highlighted by the authors.
*   **Reliance on GPT-4:** The evaluation using LLM-as-evaluators relies heavily on GPT-4o. Diversifying the referee LLMs could further strengthen the findings.

**Overall:**

TweakLLM represents a significant advance in LLM caching techniques. It offers a practical and effective solution to reduce inference costs without sacrificing response quality. The paper's novelty, comprehensive evaluation, and practical implications justify a high score. While some limitations exist, the authors have clearly identified areas for future research.

**Score: 8**

**Rationale:**

The score of 8 reflects the significant novelty and impact of TweakLLM. While the paper isn't a "groundbreaking" theoretical advance (which would warrant a 9 or 10), it provides a solid engineering solution with practical implications. The comprehensive evaluation and strong results, coupled with the availability of the code, make this paper a valuable contribution to the field. The weaknesses are primarily related to the scope of the evaluation and potential for further improvement, rather than fundamental flaws in the approach. TweakLLM has the potential to significantly influence how LLMs are deployed and scaled in real-world applications.

- **Score**: 8/10

### **[DiffuMatch: Category-Agnostic Spectral Diffusion Priors for Robust Non-rigid Shape Matching](http://arxiv.org/abs/2507.23715v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper "DiffuMatch: Category-Agnostic Spectral Diffusion Priors for Robust Non-rigid Shape Matching" addresses the challenge of non-rigid shape matching, a core problem in geometry processing.  It introduces a novel approach that replaces traditional axiomatic regularizations used in deep functional map methods with data-driven priors learned from a large collection of registered shapes.  The key idea is to train a spectral diffusion model on functional maps derived from human shape registrations and then distill this learned knowledge into a mask that serves as a regularizer during functional map computation on new, potentially unseen, shape categories. The authors demonstrate that this learned regularization can outperform axiomatic approaches and generalize well to diverse shape categories like humanoids and animals, even though trained primarily on human data.  A novel distillation strategy within the spectral domain is presented as the key technical contribution.

**Critical Evaluation:**

* **Novelty:** The novelty lies in several aspects:
    * **Data-driven Replacement of Axiomatic Regularization:**  Moving away from hand-crafted regularizations (like Laplacian commutativity) entirely and replacing them with priors learned from data is a significant shift.  This allows the model to capture complex structural properties of functional maps that might be difficult to express axiomatically.
    * **Spectral Diffusion Model for Functional Maps:** Applying score-based generative models to the *spectral domain* of functional maps is a non-trivial adaptation. It necessitates careful consideration of how the diffusion process interacts with the properties of functional maps.
    * **Distillation Strategy:** The proposed distillation strategy from the spectral diffusion model to a sparsity-promoting mask is a crucial element.  This process enables the transfer of learned structural information to the task of matching new shapes.

* **Significance:**
    * **Category-Agnostic Generalization:**  The ability to generalize shape matching to unseen categories is a major step forward.  Previous deep functional map methods often suffered from domain-specific limitations. The paper highlights the generalization capabilities.
    * **Robustness:**  The paper claims robustness as another benefit, likely stemming from learning structural priors directly from data.
    * **Potential Impact:** The approach has the potential to influence future research in shape matching by demonstrating the power of learned priors and paving the way for more flexible and adaptable matching algorithms.
    * **Clear Demonstrations:** The paper provides clear visualizations of the mask distillation, as well as texture transfer results.
* **Strengths:**
    * The core idea is well-motivated and theoretically sound.
    * The experimental results demonstrate impressive performance and generalization ability.
    * The ablation studies provide insights into the contribution of each component of the method.
    * The paper is well-written and easy to follow.
* **Weaknesses:**
    * **Limited Ablation Depth:** While the ablation study provides useful insights, a more detailed analysis exploring different choices for the feature extractor or the architecture of the diffusion model could be more useful.
    * **Computational Cost:** The current implementation requires considerable computational resources and time, especially during the optimization phase. While the authors note the performance of feature extractors with random weights, this might not be sufficient.
    * **Limited Dataset Diversity:** The diffusion model is trained on human shapes with limited diversity, which might affect performance on highly dissimilar shapes.
    * **Partial Shapes:** The paper discusses the limited capabilities in partial matching results. The authors could benefit from delving deeper into the implications of this limitation.

* **Overall:** The paper presents a well-executed and potentially transformative approach to non-rigid shape matching. By replacing axiomatic regularizations with learned priors, the method achieves impressive generalization and robustness. While there are certain limitations that future research should address, the paper's core contribution is significant and has the potential to influence the direction of the field.

**Score: 8**

**Justification:**  A score of 8 reflects the paper's solid novelty, strong performance, and clear potential for future impact. The move towards fully data-driven shape matching and the success in category-agnostic generalization are significant contributions. While computational cost and limited dataset diversity during diffusion model training present limitations, they do not detract from the paper's overall significance. A score of 8 reflects that the paper is an excellent contribution, but has room for future improvements in the implementation or methodology.

- **Score**: 8/10

### **[Seed-Prover: Deep and Broad Reasoning for Automated Theorem Proving](http://arxiv.org/abs/2507.23726v1)**
- **Summary**: Here's a concise summary and critical evaluation of the paper:

**Summary:**

The paper introduces Seed-Prover, a novel lemma-style whole-proof reasoning model for automated theorem proving in Lean.  It addresses the challenges of theorem proving, particularly in contrast to problems solvable with natural language reasoning, through formal verification of proofs.  Seed-Prover iteratively refines proofs based on Lean feedback, proved lemmas, and self-summarization.  It incorporates a three-tiered inference strategy for deep and broad reasoning and utilizes a geometry reasoning engine, Seed-Geometry, to handle geometry problems. The system achieves state-of-the-art performance on IMO problems, MiniF2F, and PutnamBench.

**Critical Evaluation:**

*   **Novelty:**  The paper demonstrates several noteworthy innovations.  The lemma-style proving approach is a significant departure from previous whole-proof generation methods, which directly generate the complete proof from scratch. This lemma-style approach mimics how mathematicians solve problems by breaking them down into smaller, manageable parts. Also, the iterative refinement is well-motivated. Combining a neural network with formal verification in this way adds a meaningful amount of automation with assurances the result is valid. The three-tiered inference strategy is practical and useful for allocating resources appropriately. The novel Seed-Geometry component to assist with the proving process is a valuable addition.
*   **Significance:** The results are impressive. Achieving state-of-the-art results on several standard benchmarks (IMO problems, MiniF2F, PutnamBench) signals a genuine advance. The authors present a complex integration of multiple technologies. The ability to prove 5 out of 6 IMO 2025 problems represents a considerable advancement in automated theorem proving. The work also contributes to developing improved geometry engines.

**Strengths:**

*   Strong empirical results across diverse benchmarks.
*   Well-motivated design choices and clear explanations of the system's components.
*   Demonstration of practical application to a high-profile competition (IMO).
*   Clear comparison against prior art and highlighting of improvements.
*   Address the limitation of previous works that lack support for geometry.
*   Provide a comprehensive overview and access through open-source resources.

**Weaknesses:**

*   The description of the training process could be more detailed. More specifics on the data generation process for geometry problems would be helpful.
*   A more in-depth analysis of the types of problems Seed-Prover still struggles with would provide valuable insights. What are the limitations of the system? A thorough study of failure cases will provide directions for future research.

**Justification:**

The paper makes a solid contribution to automated theorem proving by demonstrating a novel architecture and achieving significant performance gains. The modular design of Seed-Prover allows for future improvements in each component (e.g., better lemma proposal, more efficient geometry engine). The clear empirical results and analysis of the approach make this an impactful work that will likely influence future research in this area.

Score: 8

- **Score**: 8/10

### **[CoT-Self-Instruct: Building high-quality synthetic prompts for reasoning and non-reasoning tasks](http://arxiv.org/abs/2507.23751v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces CoT-Self-Instruct, a novel method for generating high-quality synthetic training data for Large Language Models (LLMs). This approach instructs LLMs to first reason and plan using Chain-of-Thought (CoT) on seed tasks, and then generate new synthetic prompts.  The generated data is subsequently filtered using automatic metrics, such as Answer-Consistency for verifiable reasoning tasks and Rejecting Instruction Preferences (RIP) for non-verifiable tasks.  The paper demonstrates that this method significantly outperforms existing training datasets and self-instruction techniques in both verifiable reasoning (MATH500, AMC23, AIME24, GPQA-Diamond) and non-verifiable instruction following (AlpacaEval 2.0, Arena-Hard) tasks.

**Critical Evaluation:**

*   **Novelty:** The key novelty lies in the combination of CoT for generating prompts and the use of automatic filtering techniques (Answer-Consistency and RIP) to ensure high data quality.  Self-Instruct has been explored before, but the addition of CoT reasoning during *prompt generation*, combined with automatic self-filtering represents a meaningful improvement.  It builds on previous work on Self-Instruct, Evol-Instruct, and using reward models for data selection. The specific application of CoT *at the prompt generation stage* to improve synthetic data quality appears to be a strong contribution. The Answer Consistency metric tailored to reasoning tasks and the adoption of RIP are also valuable contributions.

*   **Significance:** The significance is substantial. High-quality training data is a bottleneck for LLM development.  The CoT-Self-Instruct method offers a way to generate effective training data synthetically, reducing the reliance on expensive and potentially biased human-annotated data.  The demonstrated improvements on challenging benchmarks provide strong empirical evidence of the effectiveness of the approach. Furthermore, it provides insights into how to improve models through the use of specific filtering mechanisms tailored to the type of task. The method can easily be used in different contexts with various architectures, making it a powerful tool to fine-tune existing models and accelerate research in the field.

*   **Strengths:**
    *   **Strong empirical results:** The paper presents comprehensive experiments demonstrating the superiority of CoT-Self-Instruct across various tasks and benchmarks. The performance gains compared to existing datasets and self-instruction methods are significant and compelling.
    *   **Clear and well-defined methodology:** The paper clearly explains the CoT-Self-Instruct method, including the synthetic prompt generation and curation steps.  The use of Answer-Consistency and RIP for filtering is well-motivated and explained.
    *   **Effective data filtering techniques:**  The Answer-Consistency metric is well-tailored to verifiable reasoning tasks. The utilization of RIP for non-verifiable prompts effectively leverages reward models to identify and filter lower-quality data.
    *   **Adaptability:** The method is adaptable, as show by variations of existing methods and applying them to different tasks.
    *   **Reproducibility:** The paper includes enough experimental details and supplementary materials.

*   **Weaknesses:**
    *   **Dependency on Seed Data:** Like other Self-Instruct variants, the method is inherently dependent on the quality of the initial seed data.  While the paper mentions this, a more detailed analysis of the impact of different seed data sets would strengthen the work. How is the sensitivity of the methodology to the starting prompts? Are there any characteristics of seed prompts that would lead to an increase in data quality?
    *   **Computational cost:**  CoT-Self-Instruct requires running LLMs multiple times (for prompt generation, answering, and potentially scoring with reward models). While this is less costly than human annotation, the computational cost is still significant and should be carefully considered.
    *   **Potential for bias amplification:** Synthetic data generation methods can potentially amplify biases present in the base LLM used for generation. Addressing this potential issue more explicitly would be valuable.
    *   **Black box filter metrics:** While these filter metrics improve upon existing methods, they are not as robust as would be hoped for, since they are, in some ways, black boxes. In other words, the reasons why the methods fail is difficult to explain and reason about.

*   **Potential Impact:** The approach has the potential to accelerate progress in both reasoning and instruction-following capabilities of LLMs.  By providing a more efficient and effective way to generate high-quality training data, CoT-Self-Instruct could help improve the performance of models on challenging tasks and reduce the reliance on human-annotated data. Furthermore, it suggests that synthetic data generation is actually a viable method to improving existing LLMs.

**Score: 8**

**Rationale:**

The paper presents a novel and significant contribution to the field of LLM training. The CoT-Self-Instruct method demonstrably improves the quality of synthetic training data and leads to substantial performance gains on various benchmarks. The weaknesses, while present, are not severe enough to detract significantly from the overall value of the work. The paper is well-written, the experiments are well-designed, and the results are compelling. It offers a valuable tool for researchers and practitioners working on LLM development. While not groundbreaking, the pragmatic improvement to the data quality through COT-instruction and a combination of novel and existing metrics justifies a strong score.

- **Score**: 8/10

### **[Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis](http://arxiv.org/abs/2507.23785v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces a novel framework, GVFDiffusion, for generating high-fidelity dynamic 3D (4D) content from single video inputs.  The approach addresses the challenges of 4D content generation by introducing a Direct 4DMesh-to-GS Variation Field VAE. This VAE encodes canonical Gaussian Splats (GS) and their temporal variations from 3D animation data into a compact latent space, circumventing costly per-instance fitting.  A Gaussian Variation Field diffusion model, conditioned on input videos and canonical GS, is then trained to generate the 4D content.  The model, trained on a curated dataset of animatable 3D objects, demonstrates superior generation quality and generalization to in-the-wild video inputs.

**Critical Evaluation:**

*Novelty:*

The paper exhibits significant novelty in several aspects. The Direct 4DMesh-to-GS Variation Field VAE is a novel component that addresses a critical bottleneck in 4D generative modeling – the expensive and computationally intensive process of fitting dynamic Gaussian Splattings for each training instance. The mesh-guided loss function that aligns the motion of Gaussian points with the corresponding mesh vertices is also a novel component that enables effective motion encoding. Furthermore, leveraging the Diffusion Transformer (DiT) architecture and augmenting it with temporal self-attention layers specifically for 4D generation showcases a thoughtful adaptation of existing techniques. The combination of these components results in a unique and potentially impactful pipeline.

*Significance:*

The significance of the paper lies in its ability to generate high-quality 4D content from single video inputs. This capability opens up new possibilities in various fields, including animation, game development, and virtual reality. The demonstrated generalization to in-the-wild video inputs makes the framework practically relevant. The paper also addresses a crucial gap in the existing literature, which has primarily focused on static 3D generation or video generation, with comparatively less progress in true 4D generative modeling. By achieving efficient and high-quality 4D content creation, this work lowers the barrier to generating dynamic 3D scenes.

*Strengths:*

*   **Novel Architecture:** The combination of the VAE and diffusion model with tailored components for 4D content is a significant strength.
*   **Efficient Training:** The paper addresses a major bottleneck in 4D generation through its VAE.
*   **Strong Results:** The quantitative and qualitative results demonstrate a clear improvement over existing methods. The results on "in-the-wild" videos are particularly compelling.
*   **Clear Writing:** The paper is well-written and clearly explains the proposed method and its components.

*Weaknesses:*

*   **Reliance on Synthetic Data:** While the paper demonstrates generalization to in-the-wild videos, the model is still primarily trained on synthetic data. This could potentially limit its performance on more complex real-world scenarios.
*   **Limited Discussion of Failure Cases:** Although a failure case is presented in the supplementary material, a more comprehensive analysis of limitations would strengthen the paper. Discussing the types of motions or object characteristics that the model struggles with would be beneficial.
*   **Azimuth Alignment:** The paper mentions a post-processing step to address orientation mis-alignment. However, this suggests that the conditioning on canonical GS and input video are not inherently well aligned, which may be a weakness in the core approach.

*Potential Impact:*

This paper has a high potential for impact. By making 4D content generation more accessible and efficient, it could spur further research in this area and enable new applications.

*Justification for Score:*

I'm assigning a score of 8. The paper presents a novel and significant contribution to the field of 4D generative modeling. The architecture is well-designed, the results are compelling, and the paper is clearly written. The approach addresses a critical bottleneck in the field. The identified weaknesses (reliance on synthetic data, a lack of a comprehensive failure case analysis, and reliance on post-processing for azimuth alignment) prevent it from being a 9 or 10, but the overall contribution is substantial and merits a high score.

Score: 8

- **Score**: 8/10

## Other Papers
### **[Beyond Gloss: A Hand-Centric Framework for Gloss-Free Sign Language Translation](http://arxiv.org/abs/2507.23575v1)**
### **[DiffLoRA: Differential Low-Rank Adapters for Large Language Models](http://arxiv.org/abs/2507.23588v1)**
### **[Can LLM-Reasoning Models Replace Classical Planning? A Benchmark Study](http://arxiv.org/abs/2507.23589v1)**
### **[MoGA: 3D Generative Avatar Prior for Monocular Gaussian Avatar Reconstruction](http://arxiv.org/abs/2507.23597v1)**
### **[Medical Image De-Identification Benchmark Challenge](http://arxiv.org/abs/2507.23608v1)**
### **[LLM-Based Identification of Infostealer Infection Vectors from Screenshots: The Case of Aurora](http://arxiv.org/abs/2507.23611v1)**
### **[DivControl: Knowledge Diversion for Controllable Image Generation](http://arxiv.org/abs/2507.23620v1)**
### **[MemoCue: Empowering LLM-Based Agents for Human Memory Recall via Strategy-Guided Querying](http://arxiv.org/abs/2507.23633v1)**
### **[Adaptively Distilled ControlNet: Accelerated Training and Superior Sampling for Medical Image Synthesis](http://arxiv.org/abs/2507.23652v1)**
### **[Arabic Hate Speech Identification and Masking in Social Media using Deep Learning Models and Pre-trained Models Fine-tuning](http://arxiv.org/abs/2507.23661v1)**
### **[TweakLLM: A Routing Architecture for Dynamic Tailoring of Cached Responses](http://arxiv.org/abs/2507.23674v1)**
### **[I2V-GS: Infrastructure-to-Vehicle View Transformation with Gaussian Splatting for Autonomous Driving Data Generation](http://arxiv.org/abs/2507.23683v1)**
### **[UniLDiff: Unlocking the Power of Diffusion Priors for All-in-One Image Restoration](http://arxiv.org/abs/2507.23685v1)**
### **[A survey of multi-agent geosimulation methodologies: from ABM to LLM](http://arxiv.org/abs/2507.23694v1)**
### **[DiffuMatch: Category-Agnostic Spectral Diffusion Priors for Robust Non-rigid Shape Matching](http://arxiv.org/abs/2507.23715v1)**
### **[Seed-Prover: Deep and Broad Reasoning for Automated Theorem Proving](http://arxiv.org/abs/2507.23726v1)**
### **[Rule2Text: Natural Language Explanation of Logical Rules in Knowledge Graphs](http://arxiv.org/abs/2507.23740v1)**
### **[CoT-Self-Instruct: Building high-quality synthetic prompts for reasoning and non-reasoning tasks](http://arxiv.org/abs/2507.23751v1)**
### **[SimuRA: Towards General Goal-Oriented Agent via Simulative Reasoning Architecture with LLM-Based World Model](http://arxiv.org/abs/2507.23773v1)**
### **[Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis](http://arxiv.org/abs/2507.23785v1)**
