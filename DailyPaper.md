# The Latest Daily Papers - Date: 2025-02-12
## Highlight Papers
### **[Generalizable automated ischaemic stroke lesion segmentation with vision transformers](http://arxiv.org/abs/2502.06939v1)**
- **Summary**: This paper presents a high-performance automated ischemic stroke lesion segmentation tool using vision transformers (specifically, SWIN-UNETR).  The authors address limitations of existing U-Net based models by leveraging a large, multi-site dataset (3563 annotated lesions), incorporating data augmentation techniques, and developing algorithmic enhancements.  Crucially, they introduce a novel evaluation framework that goes beyond standard metrics (Dice, Hausdorff Distance) to assess model fidelity, equity across demographics and lesion subtypes, anatomical precision, and robustness to instrumental variability.  Their results show that the SWIN-UNETR models outperform a U-Net baseline, exhibiting greater generalizability and resilience to noise. The inclusion of control images in training (SWIN-UNETR+Ctr) further improves performance, particularly in reducing false positives.  The paper emphasizes the importance of comprehensive evaluation for clinical translation, advocating for a move beyond mean performance metrics to a more nuanced assessment of model behavior across various factors.


**Rigorous and Critical Evaluation:**

This paper makes a significant contribution to the field of medical image analysis, specifically in ischemic stroke lesion segmentation.  The use of vision transformers represents a methodological advancement over the prevalent U-Net architectures, and the large, multi-site dataset strengthens the generalizability of the results. The proposed evaluation framework is a major strength, addressing a critical gap in the field's ability to assess the real-world applicability of segmentation models.  The detailed analysis of performance across anatomical locations, lesion morphologies, and noise levels provides valuable insights into model strengths and limitations.  The open-source availability of the code further enhances the paper's impact.

However, some weaknesses exist. The manual curation of labels, although necessary given the scale of the dataset, introduces potential bias and limits the claim of completely objective ground truth.  The reliance on specific preprocessing steps (SPM12) might limit reproducibility for researchers using different pipelines.  While the paper critiques existing evaluation methods, a more formal comparison with other state-of-the-art methods on established benchmarks (like ISLES) would strengthen the claim of achieving "state-of-the-art" results.


Considering both the strengths and weaknesses, the paper represents a substantial advancement in the field. The novel evaluation framework is particularly impactful, potentially influencing future research by setting a higher standard for evaluating segmentation models.  The demonstrated superior performance of SWIN-UNETR and the insights gained from the comprehensive evaluation justify a high score.

Score: 9

- **Score**: 9/10

### **[Rethinking Fine-Tuning when Scaling Test-Time Compute: Limiting Confidence Improves Mathematical Reasoning](http://arxiv.org/abs/2502.07154v1)**
- **Summary**: This paper investigates the alignment between large language model (LLM) training and test-time compute strategies.  Focusing on the "pass@N" strategy (selecting the best answer from N samples), the authors demonstrate a surprising misalignment: standard cross-entropy (CE) loss fine-tuning can lead to decreased pass@N accuracy with more training epochs, especially for larger N.  They attribute this to model overconfidence induced by CE loss, which hinders the benefits of increased test-time compute.  To address this, they propose Direct Coverage Optimization (DCO), a modified loss function that directly optimizes pass@N accuracy.  Experiments on mathematical reasoning benchmarks (MATH and MiniF2F) and theorem proving show DCO consistently improves performance over CE loss, especially when N is large.  A step-wise variant of DCO is also introduced and shown to improve theorem proving by controlling the exploration breadth during test-time search.  Finally, an approximate DCO (DCOa) is successfully applied to the Chain-of-Thought (CoT) setting.  The core finding is that co-designing training and test-time strategies is crucial for maximizing LLM performance when scaling test-time compute.


**Rigorous and Critical Evaluation:**

The paper presents a valuable and novel contribution to the field of LLM training and evaluation.  The identification of the misalignment between standard CE loss fine-tuning and the pass@N test-time strategy is a significant finding, challenging the common practice of decoupling training and testing. The theoretical analysis explaining this misalignment through the lens of overconfidence, supported by empirical evidence, is a major strength.  The proposed DCO loss function provides a principled approach to address this issue, and its variations (step-wise DCO and DCOa) demonstrate its adaptability to different reasoning tasks and training paradigms.  The experiments are comprehensive, covering various benchmarks and test-time strategies, strengthening the paper's claims.

However, the paper has some weaknesses. The simplicity of the pass@N strategy, while allowing for clear theoretical analysis, might limit the generalizability of the findings to more complex test-time algorithms.  The computational cost of DCO, particularly its approximate variant, needs further discussion and potential optimization strategies.  Furthermore, while the theoretical lemmas provide valuable insights, they rely on certain assumptions (well-calibrated policies) which might not always hold in practice.

Despite these minor weaknesses, the paper's impact on the field is substantial.  It highlights a critical oversight in current LLM development and proposes a novel solution that addresses a previously unknown limitation. The results are compelling and suggest a paradigm shift towards a more integrated approach to training and testing LLMs.  This work is likely to inspire further research on co-designing training and test-time strategies for a broader range of LLMs and tasks.


Score: 9

- **Score**: 9/10

### **[JBShield: Defending Large Language Models from Jailbreak Attacks through Activated Concept Analysis and Manipulation](http://arxiv.org/abs/2502.07557v1)**
- **Summary**: This paper, "JBShield: Defending Large Language Models from Jailbreak Attacks through Activated Concept Analysis and Manipulation," proposes a novel defense mechanism against jailbreak attacks on LLMs.  The authors hypothesize that jailbreaks succeed by activating "jailbreak concepts" in the LLM's hidden representations, overriding the activation of "toxic concepts" that would normally trigger safety mechanisms.  They introduce JBSHIELD, a two-component framework: JBSHIELD-D for detection and JBSHIELD-M for mitigation. JBSHIELD-D identifies jailbreaks by detecting the simultaneous activation of both toxic and jailbreak concepts using a concept extraction algorithm based on singular value decomposition (SVD) of hidden layer representations.  JBSHIELD-M then mitigates the attack by enhancing the toxic concept and weakening the jailbreak concept in the LLM's hidden representations.  Extensive experiments across five LLMs and nine jailbreak attacks demonstrate high detection accuracy (average F1-score of 0.94) and a significant reduction in attack success rate (from 61% to 2%).  The method requires minimal calibration data (30 prompts).

**Critical Evaluation and Score:**

This paper presents a significant advancement in LLM security.  The core strength lies in its insightful analysis of jailbreak mechanisms, moving beyond surface-level pattern recognition to a deeper understanding of the underlying conceptual activations within the model.  The proposed JBSHIELD framework is innovative in its dual detection and mitigation approach, operating directly on the LLM's internal representations rather than relying on external filters or modifications. The impressive experimental results, showing high accuracy and a dramatic reduction in attack success rates across diverse LLMs and attack types, strongly support the effectiveness of the approach. The minimal calibration data requirement further enhances its practicality.

However, some weaknesses exist. The reliance on access to internal LLM representations limits its applicability to closed-source models.  The effectiveness might depend on the specific LLM architecture and the generalizability to completely novel jailbreak techniques remains to be fully demonstrated, although the transferability experiments offer a positive indication.  Furthermore, a more detailed discussion of the computational cost of the SVD operation at scale would strengthen the paper.  The ablation study could be extended to explore the sensitivity to the choice of layer for concept extraction.


Despite these limitations, the paper's novel approach, compelling experimental results, and potential for significant impact on the field of LLM security justify a high score.


Score: 9

- **Score**: 9/10

### **[SymGPT: Auditing Smart Contracts via Combining Symbolic Execution with Large Language Models](http://arxiv.org/abs/2502.07644v1)**
- **Summary**: SymGPT is a novel tool for auditing smart contracts' compliance with Ethereum Request for Comment (ERC) standards.  It combines Large Language Models (LLMs) for natural language understanding of ERC rules with symbolic execution for formal verification of Solidity code.  The paper details an empirical study of 132 rules from three widely-used ERC standards, identifying key insights into rule content, security implications, and linguistic patterns. SymGPT leverages these insights, using an LLM to translate ERC rules into an EBNF grammar, then synthesizes constraints for symbolic execution to detect violations.  Evaluation on 4,000 real-world contracts reveals 5,783 violations, including 1,375 with clear attack paths.  Benchmarks against six automated techniques and a human auditing service demonstrate SymGPT's superior performance and cost-effectiveness.  The paper also shows SymGPT's generalizability to ERCs beyond those initially studied.


**Rigorous and Critical Evaluation:**

SymGPT presents a significant advancement in smart contract auditing. The combination of LLMs and symbolic execution is a powerful approach, addressing the limitations of existing methods that rely solely on either program analysis or LLMs.  The empirical study of ERC rules provides valuable context and informs the design of SymGPT, making the methodology well-justified. The use of an EBNF grammar as an intermediate representation is a clever strategy for mitigating LLM hallucinations and improving accuracy.  The detailed explanation of the symbolic execution process, including constraint generation and handling of diverse implementations, is commendable. The extensive evaluation, using both a large dataset and a ground-truth dataset, strengthens the paper's claims.  The comparison with existing techniques clearly demonstrates SymGPT's superiority in terms of both accuracy and efficiency.  The demonstration of generalizability to a previously unstudied ERC further solidifies the tool's potential impact.

However, some weaknesses exist.  The reliance on an LLM introduces inherent uncertainties, although the paper mitigates this risk to a considerable degree.  The false positives, primarily stemming from LLM errors in rule extraction, represent a limitation that needs further refinement. While the paper suggests a simple mitigation strategy, a more robust solution would be desirable.  The handling of assembly code and external calls remains an area for improvement.  Finally, while the cost-effectiveness is highlighted, a more detailed breakdown of the computational resources consumed by SymGPT would be beneficial.

Despite these minor shortcomings, the overall contribution is substantial.  SymGPT offers a promising approach to automating a crucial and complex task, significantly improving the security and reliability of smart contracts.  The paper's findings and the tool itself are likely to influence the development of future smart contract auditing techniques.


Score: 9

- **Score**: 9/10

### **[CausalGeD: Blending Causality and Diffusion for Spatial Gene Expression Generation](http://arxiv.org/abs/2502.07751v1)**
- **Summary**: CausalGeD is a novel framework for integrating single-cell RNA sequencing (scRNA-seq) and spatial transcriptomics (ST) data.  It addresses the limitations of existing methods by explicitly incorporating causal relationships between genes using a combined diffusion and autoregressive model.  The core innovation is a Causality Aware Transformer (CAT) module that learns these relationships without predefined regulatory networks.  Evaluated across ten diverse tissue datasets, CausalGeD significantly outperforms state-of-the-art baselines in terms of Pearson correlation, structural similarity, RMSE, and JS divergence, achieving improvements of 5-32%.  Ablation studies demonstrate the effectiveness of the individual components of the model, particularly the AR step decay strategy mimicking biological regulatory hierarchies.  The improved accuracy translates into more reliable biological insights, such as better identification of spatial patterns in tumor microenvironments and developmental gene regulation.  A limitation is the requirement for ST genes to be a subset of those in the scRNA-seq data.


**Rigorous and Critical Evaluation:**

The paper presents a significant advancement in the field of spatial transcriptomics data integration.  The key strength lies in its novel approach of incorporating causal relationships between genes, a factor often overlooked in existing methods. The CAT module cleverly addresses the challenge of learning these relationships without requiring prior knowledge of gene regulatory networks. The extensive experimental validation across ten diverse datasets, including a robust set of evaluation metrics and ablation studies, provides strong evidence for the effectiveness of the proposed method.  The visualizations (UMAP and hierarchical clustering) effectively demonstrate the improved accuracy and preservation of both global and local structural features.  The discussion of biological implications adds practical value, showcasing the potential of CausalGeD to enhance biological understanding.

However, some critical points need consideration:

* **Comparability of Baselines:** While the paper compares CausalGeD against several state-of-the-art methods, a more detailed analysis of the implementation details and hyperparameter settings used for each baseline would strengthen the comparison.  Slight variations in preprocessing or training could influence the results.
* **Generalizability:**  The limitation regarding the overlap of genes between scRNA-seq and ST data is a notable constraint.  The paper mentions this as future work, but its absence currently limits the broad applicability of CausalGeD.
* **Biological Interpretation Depth:** While the paper mentions biological implications, a deeper dive into the specific biological insights gained from CausalGeD's improved predictions would further enhance its impact.  For instance, could specific causal relationships learned by the model be validated through existing biological knowledge or experimental data?

Despite these minor weaknesses, the core novelty and significant performance improvements of CausalGeD justify a high score. The method's ability to learn complex gene regulatory relationships directly from data, without relying on external knowledge, has considerable potential to accelerate research in spatial transcriptomics and related fields.


Score: 9

- **Score**: 9/10

## Other Papers
### **[Rationalization Models for Text-to-SQL](http://arxiv.org/abs/2502.06759v1)**
### **[History-Guided Video Diffusion](http://arxiv.org/abs/2502.06764v1)**
### **[Exploiting Sparsity for Long Context Inference: Million Token Contexts on Commodity GPUs](http://arxiv.org/abs/2502.06766v1)**
### **[Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions](http://arxiv.org/abs/2502.06768v1)**
### **[Lumina-Video: Efficient and Flexible Video Generation with Multi-scale Next-DiT](http://arxiv.org/abs/2502.06782v1)**
### **[DeepCrossAttention: Supercharging Transformer Residual Connections](http://arxiv.org/abs/2502.06785v1)**
### **[Generalizable automated ischaemic stroke lesion segmentation with vision transformers](http://arxiv.org/abs/2502.06939v1)**
### **[GAS: Generative Avatar Synthesis from a Single Image](http://arxiv.org/abs/2502.06957v1)**
### **[Model Diffusion for Certifiable Few-shot Transfer Learning](http://arxiv.org/abs/2502.06970v1)**
### **[Position: Episodic Memory is the Missing Piece for Long-Term LLM Agents](http://arxiv.org/abs/2502.06975v1)**
### **[Investigating the Zone of Proximal Development of Language Models for In-Context Learning](http://arxiv.org/abs/2502.06990v1)**
### **[Conditional diffusion model with spatial attention and latent embedding for medical image segmentation](http://arxiv.org/abs/2502.06997v1)**
### **[Outsourced diffusion sampling: Efficient posterior inference in latent spaces of generative models](http://arxiv.org/abs/2502.06999v1)**
### **[From Image to Video: An Empirical Study of Diffusion Representations](http://arxiv.org/abs/2502.07001v1)**
### **[Demystifying Singular Defects in Large Language Models](http://arxiv.org/abs/2502.07004v1)**
### **[Finding Words Associated with DIF: Predicting Differential Item Functioning using LLMs and Explainable AI](http://arxiv.org/abs/2502.07017v1)**
### **[AIMS.au: A Dataset for the Analysis of Modern Slavery Countermeasures in Corporate Statements](http://arxiv.org/abs/2502.07022v1)**
### **[Automated Consistency Analysis of LLMs](http://arxiv.org/abs/2502.07036v1)**
### **[Scalable and Ethical Insider Threat Detection through Data Synthesis and Analysis by LLMs](http://arxiv.org/abs/2502.07045v1)**
### **[SnipGen: A Mining Repository Framework for Evaluating LLMs for Code](http://arxiv.org/abs/2502.07046v1)**
### **[Large Language Models in Software Security: A Survey of Vulnerability Detection Techniques and Insights](http://arxiv.org/abs/2502.07049v1)**
### **[Tokenization Standards for Linguistic Integrity: Turkish as a Benchmark](http://arxiv.org/abs/2502.07057v1)**
### **[Using Contextually Aligned Online Reviews to Measure LLMs' Performance Disparities Across Language Varieties](http://arxiv.org/abs/2502.07058v1)**
### **[Specializing Large Language Models to Simulate Survey Response Distributions for Global Populations](http://arxiv.org/abs/2502.07068v1)**
### **[IRepair: An Intent-Aware Approach to Repair Data-Driven Errors in Large Language Models](http://arxiv.org/abs/2502.07072v1)**
### **[Multi-turn Evaluation of Anthropomorphic Behaviours in Large Language Models](http://arxiv.org/abs/2502.07077v1)**
### **[Evaluating the Systematic Reasoning Abilities of Large Language Models through Graph Coloring](http://arxiv.org/abs/2502.07087v1)**
### **[Generative Distribution Prediction: A Unified Approach to Multimodal Learning](http://arxiv.org/abs/2502.07090v1)**
### **[Likelihood-Free Estimation for Spatiotemporal Hawkes processes with missing data and application to predictive policing](http://arxiv.org/abs/2502.07111v1)**
### **[Is Long Range Sequential Modeling Necessary For Colorectal Tumor Segmentation?](http://arxiv.org/abs/2502.07120v1)**
### **[Cardiverse: Harnessing LLMs for Novel Card Game Prototyping](http://arxiv.org/abs/2502.07128v1)**
### **[Language-TPP: Integrating Temporal Point Processes with Language Models for Event Analysis](http://arxiv.org/abs/2502.07139v1)**
### **[Ask Patients with Patience: Enabling LLMs for Human-Centric Medical Dialogue with Grounded Reasoning](http://arxiv.org/abs/2502.07143v1)**
### **[Rethinking Fine-Tuning when Scaling Test-Time Compute: Limiting Confidence Improves Mathematical Reasoning](http://arxiv.org/abs/2502.07154v1)**
### **[HDCompression: Hybrid-Diffusion Image Compression for Ultra-Low Bitrates](http://arxiv.org/abs/2502.07160v1)**
### **[A Survey on Mamba Architecture for Vision Applications](http://arxiv.org/abs/2502.07161v1)**
### **[Does Training on Synthetic Data Make Models Less Robust?](http://arxiv.org/abs/2502.07164v1)**
### **[Refine Knowledge of Large Language Models via Adaptive Contrastive Learning](http://arxiv.org/abs/2502.07184v1)**
### **[A Large-Scale Benchmark for Vietnamese Sentence Paraphrases](http://arxiv.org/abs/2502.07188v1)**
### **[Bag of Tricks for Inference-time Computation of LLM Reasoning](http://arxiv.org/abs/2502.07191v1)**
### **[Provably Efficient RLHF Pipeline: A Unified View from Contextual Bandits](http://arxiv.org/abs/2502.07193v1)**
### **[Monte Carlo Tree Diffusion for System 2 Planning](http://arxiv.org/abs/2502.07202v1)**
### **[Improve the Training Efficiency of DRL for Wireless Communication Resource Allocation: The Role of Generative Diffusion Models](http://arxiv.org/abs/2502.07211v1)**
### **[LUNAR: LLM Unlearning via Neural Activation Redirection](http://arxiv.org/abs/2502.07218v1)**
### **[MLLM4PUE: Toward Universal Embeddings in Computational Pathology through Multimodal LLMs](http://arxiv.org/abs/2502.07221v1)**
### **[A Memory Efficient Randomized Subspace Optimization Method for Training Large Language Models](http://arxiv.org/abs/2502.07222v1)**
### **[CAT: Contrastive Adversarial Training for Evaluating the Robustness of Protective Perturbations in Latent Diffusion Models](http://arxiv.org/abs/2502.07225v1)**
### **[Linear Transformers as VAR Models: Aligning Autoregressive Attention Mechanisms with Autoregressive Forecasting](http://arxiv.org/abs/2502.07244v1)**
### **[When More is Less: Understanding Chain-of-Thought Length in LLMs](http://arxiv.org/abs/2502.07266v1)**
### **[GENERator: A Long-Context Generative Genomic Foundation Model](http://arxiv.org/abs/2502.07272v1)**
### **[Articulate That Object Part (ATOP): 3D Part Articulation from Text and Motion Personalization](http://arxiv.org/abs/2502.07278v1)**
### **[Exploratory Diffusion Policy for Unsupervised Reinforcement Learning](http://arxiv.org/abs/2502.07279v1)**
### **[Small Language Model Makes an Effective Long Text Extractor](http://arxiv.org/abs/2502.07286v1)**
### **[Investigating Creativity in Humans and Generative AI Through Circles Exercises](http://arxiv.org/abs/2502.07292v1)**
### **[Generation of Drug-Induced Cardiac Reactions towards Virtual Clinical Trials](http://arxiv.org/abs/2502.07297v1)**
### **[TRAVEL: Training-Free Retrieval and Alignment for Vision-and-Language Navigation](http://arxiv.org/abs/2502.07306v1)**
### **[Prompt-Based Document Modifications In Ranking Competitions](http://arxiv.org/abs/2502.07315v1)**
### **[CodeI/O: Condensing Reasoning Patterns via Code Input-Output Prediction](http://arxiv.org/abs/2502.07316v1)**
### **[MEMIT-Merge: Addressing MEMIT's Key-Value Conflicts in Same-Subject Batch Editing for LLMs](http://arxiv.org/abs/2502.07322v1)**
### **[Semantic to Structure: Learning Structural Representations for Infringement Detection](http://arxiv.org/abs/2502.07323v1)**
### **[Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filtering](http://arxiv.org/abs/2502.07340v1)**
### **[BenchMAX: A Comprehensive Multilingual Evaluation Suite for Large Language Models](http://arxiv.org/abs/2502.07346v1)**
### **[KABB: Knowledge-Aware Bayesian Bandits for Dynamic Expert Coordination in Multi-Agent Systems](http://arxiv.org/abs/2502.07350v1)**
### **[Bridging the Evaluation Gap: Leveraging Large Language Models for Topic Model Evaluation](http://arxiv.org/abs/2502.07352v1)**
### **[LongReD: Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Distillation](http://arxiv.org/abs/2502.07365v1)**
### **[LLMs Can Easily Learn to Reason from Demonstrations Structure, not content, is what matters!](http://arxiv.org/abs/2502.07374v1)**
### **[Spatial Degradation-Aware and Temporal Consistent Diffusion Model for Compressed Video Super-Resolution](http://arxiv.org/abs/2502.07381v1)**
### **[On Iterative Evaluation and Enhancement of Code Quality Using GPT-4o](http://arxiv.org/abs/2502.07399v1)**
### **[EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering](http://arxiv.org/abs/2502.07411v1)**
### **[Entity Linking using LLMs for Automated Product Carbon Footprint Estimation](http://arxiv.org/abs/2502.07418v1)**
### **[RomanLens: Latent Romanization and its role in Multilinguality in LLMs](http://arxiv.org/abs/2502.07424v1)**
### **[Optimizing Knowledge Distillation in Transformers: Enabling Multi-Head Attention without Alignment Barriers](http://arxiv.org/abs/2502.07436v1)**
### **[Forget What You Know about LLMs Evaluations - LLMs are Like a Chameleon](http://arxiv.org/abs/2502.07445v1)**
### **[RusCode: Russian Cultural Code Benchmark for Text-to-Image Generation](http://arxiv.org/abs/2502.07455v1)**
### **[PerCul: A Story-Driven Cultural Evaluation of LLMs in Persian](http://arxiv.org/abs/2502.07459v1)**
### **[Logarithmic Regret for Online KL-Regularized Reinforcement Learning](http://arxiv.org/abs/2502.07460v1)**
### **[Less is More: Masking Elements in Image Condition Features Avoids Content Leakages in Style Transfer Diffusion Models](http://arxiv.org/abs/2502.07466v1)**
### **[Improving Adaptive Moment Optimization via Preconditioner Diagonalization](http://arxiv.org/abs/2502.07488v1)**
### **[Mask-Enhanced Autoregressive Prediction: Pay Less Attention to Learn More](http://arxiv.org/abs/2502.07490v1)**
### **[LLM-Sketch: Enhancing Network Sketches with LLM](http://arxiv.org/abs/2502.07495v1)**
### **[Unified Graph Networks (UGN): A Deep Neural Framework for Solving Graph Problems](http://arxiv.org/abs/2502.07500v1)**
### **[The Devil is in the Prompts: De-Identification Traces Enhance Memorization Risks in Synthetic Chest X-Ray Generation](http://arxiv.org/abs/2502.07516v1)**
### **[Exoplanet Transit Candidate Identification in TESS Full-Frame Images via a Transformer-Based Algorithm](http://arxiv.org/abs/2502.07542v1)**
### **[Grammar Control in Dialogue Response Generation for Language Learning Chatbots](http://arxiv.org/abs/2502.07544v1)**
### **[Attention Learning is Needed to Efficiently Learn Parity Function](http://arxiv.org/abs/2502.07553v1)**
### **[O1 Embedder: Let Retrievers Think Before Action](http://arxiv.org/abs/2502.07555v1)**
### **[SketchFlex: Facilitating Spatial-Semantic Coherence in Text-to-Image Generation with Region-Based Sketches](http://arxiv.org/abs/2502.07556v1)**
### **[JBShield: Defending Large Language Models from Jailbreak Attacks through Activated Concept Analysis and Manipulation](http://arxiv.org/abs/2502.07557v1)**
### **[PIM Is All You Need: A CXL-Enabled GPU-Free System for Large Language Model Inference](http://arxiv.org/abs/2502.07578v1)**
### **[Single-Step Consistent Diffusion Samplers](http://arxiv.org/abs/2502.07579v1)**
### **[Generative Modeling with Bayesian Sample Inference](http://arxiv.org/abs/2502.07580v1)**
### **[DSV: Exploiting Dynamic Sparsity to Accelerate Large-Scale Video DiT Training](http://arxiv.org/abs/2502.07590v1)**
### **[Towards spatial computing: recent advances in multimodal natural interaction for XR headsets](http://arxiv.org/abs/2502.07598v1)**
### **[Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models](http://arxiv.org/abs/2502.07601v1)**
### **[Beyond Prompting: Time2Lang -- Bridging Time-Series Foundation Models and Large Language Models for Health Sensing](http://arxiv.org/abs/2502.07608v1)**
### **[Tractable Transformers for Flexible Conditional Generation](http://arxiv.org/abs/2502.07616v1)**
### **[Exploring Mobile Touch Interaction with Large Language Models](http://arxiv.org/abs/2502.07629v1)**
### **[Consistency Training with Physical Constraints](http://arxiv.org/abs/2502.07636v1)**
### **[FoQA: A Faroese Question-Answering Dataset](http://arxiv.org/abs/2502.07642v1)**
### **[SymGPT: Auditing Smart Contracts via Combining Symbolic Execution with Large Language Models](http://arxiv.org/abs/2502.07644v1)**
### **[Large Language Models as Proxies for Theories of Human Linguistic Cognition](http://arxiv.org/abs/2502.07687v1)**
### **[A Framework for LLM-powered Design Assistants](http://arxiv.org/abs/2502.07698v1)**
### **[Magic 1-For-1: Generating One Minute Video Clips within One Minute](http://arxiv.org/abs/2502.07701v1)**
### **[Verifying LLM-Generated Code in the Context of Software Verification with Ada/SPARK](http://arxiv.org/abs/2502.07728v1)**
### **[Economics of Sourcing Human Data](http://arxiv.org/abs/2502.07732v1)**
### **[WHODUNIT: Evaluation benchmark for culprit detection in mystery stories](http://arxiv.org/abs/2502.07747v1)**
### **[CausalGeD: Blending Causality and Diffusion for Spatial Gene Expression Generation](http://arxiv.org/abs/2502.07751v1)**
### **[Towards Efficient Optimizer Design for LLM via Structured Fisher Approximation with a Low-Rank Extension](http://arxiv.org/abs/2502.07752v1)**
### **[Direct Ascent Synthesis: Revealing Hidden Generative Capabilities in Discriminative Models](http://arxiv.org/abs/2502.07753v1)**
### **[Scalable Fingerprinting of Large Language Models](http://arxiv.org/abs/2502.07760v1)**
### **[Great Power Brings Great Responsibility: Personalizing Conversational AI for Diverse Problem-Solvers](http://arxiv.org/abs/2502.07763v1)**
