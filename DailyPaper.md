# The Latest Daily Papers - Date: 2025-04-28
## Highlight Papers
### **[DCT-Shield: A Robust Frequency Domain Defense against Malicious Image Editing](http://arxiv.org/abs/2504.17894v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "DCT-Shield: A Robust Frequency Domain Defense against Malicious Image Editing":

**Summary:**

The paper "DCT-Shield" introduces a novel defense mechanism against malicious image editing attacks that leverage diffusion models. Instead of adding adversarial noise in the pixel space (as done by previous methods), DCT-Shield operates in the frequency domain by modifying the Discrete Cosine Transform (DCT) coefficients of the image. This approach aims to create adversarial perturbations that are less perceptible to human observers while still disrupting the image editing capabilities of diffusion models.  The authors also explicitly incorporate the JPEG compression pipeline into their optimization, making the defense more robust against JPEG purification techniques.  They present several variants of DCT-Shield tailored for specific attack scenarios like inpainting and for achieving optimal imperceptibility. The authors claim improved performance compared to existing methods in terms of both imperceptibility and robustness.

**Critical Evaluation:**

* **Novelty:** The key novelty of this paper is shifting the adversarial defense from the pixel domain to the DCT domain. This is a significant departure from previous work. Existing methods have generally focused on adding noise directly to the pixel values, which often leads to noticeable artifacts. By operating in the frequency domain and leveraging the JPEG pipeline, DCT-Shield creates perturbations that are more subtle and robust to compression. The explicit incorporation of the JPEG pipeline into the adversarial training process is a strong point. Further, the variants created to target specific edit types contribute to its novelty.

* **Significance:** The paper addresses a very relevant problem: the vulnerability of images to malicious editing enabled by powerful diffusion models. The demonstrated robustness against JPEG compression, a common image processing technique, is a crucial practical consideration. The reduced number of parameters (compared to U-Net based approaches) also adds to the practical significance as it facilitates faster and more efficient training. The paper's findings have the potential to influence how images are protected in the future, especially considering the widespread use of JPEG and the increasing sophistication of AI-powered image manipulation tools. DCT-Shield makes images resilient to edits after they have been heavily compressed, thus preventing malicious agents from subtly manipulating content.

* **Strengths:**
    *   **Strong Empirical Results:** The paper provides extensive experimental results across different datasets and editing tasks, demonstrating the effectiveness of DCT-Shield compared to existing baselines. Qualitative results reinforce the argument about improved imperceptibility and protection against edits.
    *   **JPEG Robustness:** The comprehensive analysis of robustness against JPEG compression is a major strength. The paper explicitly demonstrates that their method is less susceptible to JPEG purification techniques, which addresses a significant limitation of previous pixel-space defenses.
    *   **Practical Considerations:** The method is relatively parameter-efficient and computationally cheaper than some existing defenses (particularly those relying on diffusion-based attacks), making it more practical for real-world deployment.
    *   **Adaptability:** The development of different DCT-Shield variants for specific scenarios (inpainting, low-quality JPEG) showcases the adaptability and flexibility of the proposed framework.

* **Weaknesses:**
    *   **Limited Scope of Attacks:** While the paper demonstrates effectiveness against several editing tasks, it is important to note the scope of attack. They consider attacks via latent diffusion models only, with the goal of disrupting the editing task. The paper could have added other types of attacks like watermark attacks, and image replacement attacks.
    *   **Dependency on VAE:** The framework uses a VAE (Variational Autoencoder) for the loss computation. Therefore, the attack would be dependent on the VAE architecture used. It is possible that using a different VAE may yield different results.
    *   **Adversarial Transferability:** The paper doesn't explore the transferability of the generated adversarial examples to different diffusion models or architectures.  It's important to assess whether the protection provided by DCT-Shield is specific to the VAE and diffusion model used during training or if it generalizes to other models.

* **Clarity and Presentation:** The paper is well-written and clearly presents the proposed method, experimental setup, and results. The inclusion of visualizations and ablation studies helps to understand the different aspects of DCT-Shield.

**Justification for Score:**

Considering the novelty, significance, strengths, and weaknesses, a score of 8 is appropriate.  The move to the DCT domain, the explicit JPEG robustness, and the practical considerations contribute to a strong overall contribution. While the paper could have benefited from a more comprehensive analysis of attack scenarios and transferability, the core idea is sound and the results are compelling. The explicit optimization of the loss function by JPEG pipeline is a novel contribution. DCT-shield creates an attack model that targets specific type of edits, making it significantly more powerful than prior methods.

**Score: 8**

- **Score**: 8/10

### **[Diffusion-Driven Universal Model Inversion Attack for Face Recognition](http://arxiv.org/abs/2504.18015v1)**
- **Summary**: The paper introduces DiffUMI, a novel training-free diffusion-driven universal model inversion attack for face recognition systems. DiffUMI leverages a pretrained diffusion model to reconstruct facial images from embeddings, eliminating the need for target-specific generator training. The method operates within a fixed framework and seamlessly adapts to diverse target identities and models. The paper also introduces a novel application of out-of-domain detection (OODD) using model inversion to distinguish non-face inputs from face inputs based on embeddings.

**Critical Evaluation:**

**Novelty:** The paper presents a significant advance by applying diffusion models to model inversion in a training-free and universal manner. Existing model inversion attacks often require training specific generators for each target model, making them computationally expensive. DiffUMI overcomes this limitation by utilizing a fixed, pretrained diffusion model. The introduction of OODD using model inversion is also novel and demonstrates a unique application of this technique.

**Significance:** The paper has substantial significance due to the privacy risks associated with face recognition technology. Demonstrating that privacy-preserving techniques based on embeddings are vulnerable to universal model inversion attacks highlights the importance of developing stronger defenses. The high success rates achieved by DiffUMI in reconstructing facial images from embeddings raise serious concerns about the adequacy of existing privacy measures. The OODD application provides a new approach to detect potentially malicious inputs, further enhancing the paper's significance.

**Strengths:**
*   **Training-Free and Universal:** The most significant strength is the elimination of target-specific training, making the attack scalable and applicable to various face recognition models.
*   **High Reconstruction Quality:** DiffUMI achieves impressive reconstruction accuracy, recovering facial identities with a high degree of fidelity.
*   **Novel OODD Application:** The introduction of OODD based on model inversion demonstrates a unique application with practical implications for security.
*   **Comprehensive Evaluation:** The paper includes a thorough evaluation on multiple datasets and models, comparing DiffUMI to existing benchmark attacks.

**Weaknesses:**
*   **Computational Cost:** While the paper highlights the computational efficiency gains relative to training-dependent attacks, the black-box version of DiffUMI remains computationally expensive for high-resolution generation due to adversarial manipulation, limiting its practical applicability. The paper could provide greater detail and justification on the trade-off between attack fidelity, computational cost, and query efficiency.
*   **Black-box Attack:** The paper relies heavily on white box attacks. How effective is black box in more realistic settings. How many queries and the associated costs required.
*   **Potential Overfitting:** Although the paper addresses the risk of overfitting to the target model, it could provide a more in-depth analysis of the potential limitations and challenges associated with adversarial manipulation in latent space. How well does the result generalizes to new models?
*   **Limited Evaluation of Defenses:** While the paper discusses potential defenses against DiffUMI, it doesn't thoroughly evaluate the effectiveness of these defenses. More analysis could be provided on how to incorporate defense methods within the evaluation.
*   **Reproducibility:** While the source code is to be released the paper lacks detailed experimental setup details in the main body and relies heavily on the appendixes for implementation details that could hinder reproducibility.

**Potential Influence:** The paper is likely to have a significant influence on the field by motivating further research into stronger defenses against model inversion attacks and exploring the applications of diffusion models in privacy and security. The OODD application provides a new direction for research in detecting malicious inputs to face recognition systems.

**Score: 8**

**Justification:** The paper presents a valuable contribution to the field by introducing a training-free universal model inversion attack and a novel application of OODD. The strengths of the paper outweigh its weaknesses, and the work has the potential to stimulate further research in developing stronger defenses against model inversion attacks. However, the computational cost of the black-box attack and the need for further evaluation of defenses limit the overall impact of the paper.

- **Score**: 8/10

### **[Enhancing Privacy-Utility Trade-offs to Mitigate Memorization in Diffusion Models](http://arxiv.org/abs/2504.18032v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper addresses the problem of memorization in text-to-image diffusion models, which can lead to copyright infringement and privacy violations. The authors propose a novel method called PRSS (Prompt Re-anchoring and Semantic Search) to mitigate this issue. PRSS refines the classifier-free guidance (CFG) approach by integrating prompt re-anchoring (PR) to enhance privacy and semantic prompt search (SS) to maintain utility (text-image alignment).  PR helps by steering the image generation away from memorized examples, while SS finds alternative prompts that are semantically similar to the original prompt but less likely to cause memorization. The paper presents experimental results demonstrating that PRSS consistently improves the privacy-utility trade-off compared to existing methods.

**Critical Evaluation:**

* **Novelty:** The paper introduces a novel combination of techniques (prompt re-anchoring and semantic prompt search) to address the memorization problem.  Existing methods typically rely heavily on prompt engineering, which can significantly degrade utility (text alignment). The novelty lies in the synergistic use of PR and SS.  PR focuses on privacy by anchoring on the original prompt, while SS attempts to maintain utility through semantic alternatives. The idea of splitting responsibilities between privacy and utility enhancement is innovative.
* **Significance:** The memorization problem in diffusion models is a significant concern with real-world legal and ethical implications. A method that effectively balances privacy and utility is highly valuable. The paper provides a practical and efficient solution that can be applied to pre-trained diffusion models without requiring retraining or fine-tuning. This adaptability is a major strength. The improvements in privacy-utility trade-off shown through comprehensive experiments highlight the significance of the work. The authors adequately addressed the related issues of training and inference time costs with this approach.
* **Strengths:**
    *   The PRSS method offers a well-reasoned approach to balancing the privacy-utility trade-off.
    *   The paper provides a clear explanation of the motivations, methodology, and experimental setup.
    *   Extensive experiments and ablation studies validate the effectiveness of each component of PRSS.
    *   The method is efficient and adaptable to existing diffusion models.
    *   The geometric analysis using diagrams effectively illustrates the impact of each technique.
    * The detection scheme is simple and can be easily adaptable to most recent methods that tackle the privacy issues of text to image generation.
* **Weaknesses:**
    *   While the paper argues against the necessity of retraining models, the integration of semantic prompt search does rely on GPT-4, requiring an API call, which could introduce latency. While the authors explicitly state the cost to be minimal and acceptable, the dependence on an external LLM is an implementation detail that future works should aim to negate.
    *   The reliance on LLM-generated alternative prompts could potentially introduce bias.  While semantic similarity is enforced, subtle differences in meaning could still affect the generated images in unpredictable ways. A rigorous analysis is needed to investigate the potential negative impacts of biases.
    *  Although improvements in safety are notable, there's room for deeper investigation on the generalizability and robustness on more challenging memorization datasets.

**Justification of Score:**

The paper presents a valuable contribution to mitigating memorization in diffusion models by proposing a novel, efficient, and adaptable method. While the dependence on external LLMs introduces concerns related to biases and is a weakness, the paper's rigorous experimentation and clear exposition of its core concepts strongly support its claims.

Score: 8.5

- **Score**: 8/10

### **[RAG LLMs are Not Safer: A Safety Analysis of Retrieval-Augmented Generation for Large Language Models](http://arxiv.org/abs/2504.18041v1)**
- **Summary**: Here's a summary and critical evaluation of the provided paper:

**Summary:**

The paper investigates the safety implications of using Retrieval-Augmented Generation (RAG) with Large Language Models (LLMs). Contrary to the intuitive hypothesis that RAG improves safety by grounding responses in a controlled corpus, the authors find that RAG can actually *decrease* the safety of LLMs. Through extensive experiments with eleven popular LLMs, they demonstrate that RAG can lead to a higher percentage of unsafe responses, even when using safe models and safe documents. They explore factors contributing to this phenomenon, including the LLM's inherent safety, the safety of retrieved documents, and the LLM's ability to perform RAG tasks effectively.  They further show that existing red-teaming methods designed for standard LLMs are less effective in RAG settings, highlighting the need for specialized safety research and red-teaming techniques tailored to RAG LLMs.

**Critical Evaluation:**

*   **Novelty:** The paper addresses a crucial but relatively underexplored area: the safety of RAG-based LLMs *beyond* corpus poisoning attacks.  While existing work focuses on injecting harmful content into the knowledge base, this paper examines the safety implications of RAG with a *controlled*, supposedly safe, corpus. The finding that RAG can decrease safety, even with safe components, is a novel and counterintuitive result. It identifies previously unrecognized vulnerabilities specific to RAG systems. The work shows that current methods that fine-tune, evaluate, and red-team LLMs often operate in a non-RAG setting.

*   **Significance:** The paper has significant implications for the development and deployment of LLM-based applications. RAG is a widely used framework for improving the factual accuracy and relevance of LLM outputs. The finding that RAG can compromise safety raises serious concerns and underscores the importance of considering safety implications throughout the development of LLM applications.  The fact that standard red-teaming techniques are inadequate for RAG also highlights the need for new evaluation methods.

*   **Strengths:**
    *   **Extensive Experiments:**  The paper presents a large-scale evaluation involving multiple LLMs and a substantial dataset of harmful questions. This provides strong empirical support for its claims.
    *   **Detailed Analysis:** The authors conduct a thorough investigation of the factors contributing to the safety degradation in RAG, including the LLM's safety, document safety, and RAG task performance.
    *   **Practical Implications:** The paper identifies concrete challenges and suggests directions for future research, including the need for RAG-specific safety fine-tuning, red-teaming methods, and understanding the mechanisms behind unsafe RAG generations.
    *   **Clear Problem Definition:** The paper succinctly and convincingly identifies an emergent problem for a commonly used method in AI, RAG, for which prior solutions do not address.

*   **Weaknesses:**
    *   **Limited Corpus Scope:** The paper uses English Wikipedia as the corpus, and how this relates to other corpora is not fully clear. This could limit the generalizability of the findings. Also, the paper notes that safety of the RAG system can be impacted by whether it also draws on internal knowledge, or purely the retrieved documents. While the study attempts to control for this factor in its experiments, there is a potential weakness in understanding the interplay of knowledge sources in the RAG context.
    *   **Simple Retrieval Method:** The paper uses BM25 for retrieval. While BM25 is a strong baseline, more advanced retrieval methods might yield different results.
    *   **Reliance on LLM as Safety Judge:** The reliance on Llama Guard 2 as the safety judge has limitations, given that it is another LLM and thus imperfect. Using more robust and diverse safety evaluation methods would strengthen the results. While the paper discusses this limitation and uses a secondary model as a second judge, further efforts may be required.

*   **Impact:** The paper has the potential to significantly influence the field of LLM safety and RAG research. It may lead to the development of safer RAG architectures, new red-teaming techniques, and a better understanding of the interplay between retrieval and generation in LLMs.

**Justification for Score:**

The paper presents a novel and important finding with significant practical implications. The extensive experiments and detailed analysis add to the strengths of the work, but the limitations regarding corpus scope, retrieval method, and reliance on LLM-based safety judgements slightly temper its contribution. Based on these considerations, I believe a score of 8 is warranted. The work reveals a blindspot in safety for RAG systems that will drive the development of future solutions.

**Score: 8**

- **Score**: 8/10

### **[DREAM: Disentangling Risks to Enhance Safety Alignment in Multimodal Large Language Models](http://arxiv.org/abs/2504.18053v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "DREAM: Disentangling Risks to Enhance Safety Alignment in Multimodal Large Language Models":

**Summary:**

The paper introduces DREAM, a novel training-time approach aimed at enhancing the safety alignment of Multimodal Large Language Models (MLLMs). The core idea revolves around "disentangling risks" within multimodal inputs. First, they perform a detailed analysis of how different combinations of safe and unsafe image-text pairs can lead to complex risk scenarios.  They then propose Multimodal Risk Disentanglement (MRD) to systematically analyze multimodal inputs and identify potential risks. Based on the insights from MRD, they develop DREAM, a two-part framework: Risk-aware Fine-tuning (SFT) and Risk-aware Preference Optimization (iterative RLAIF). SFT internalizes the MRD capability, and RLAIF further improves safety. Experiments demonstrate that DREAM enhances safety without significantly affecting the model's performance on normal tasks, achieving a substantial improvement compared to GPT-4V in a safe&effective score.

**Critical Evaluation:**

*   **Novelty:** The paper's novelty lies in its structured approach to risk disentanglement in MLLMs. While previous works have addressed safety concerns in MLLMs, the explicit focus on analyzing and disentangling different risk combinations within multimodal inputs is a unique contribution. Moreover, leveraging this risk understanding within a training framework (DREAM) based on SFT and RLAIF is also novel. The use of MRD for automated feedback collection is another notable aspect.

*   **Significance:** The significance stems from the growing importance of safety in MLLMs. As these models become more prevalent in real-world applications, addressing safety concerns is crucial. The paper's results, showing improved safety with minimal impact on performance, are promising. The framework's ability to generalize across multiple benchmarks and models is also a significant strength. The study also highlights the problem that simple prompts may not effectively stimulate the risk-awareness capabilities of MLLMs, and provides a way to deal with it.

*   **Strengths:**

    *   **Well-defined problem:** The paper clearly articulates the challenges of safety alignment in MLLMs, particularly the complexity introduced by multimodal inputs.
    *   **Thorough analysis:** The risk combination analysis provides valuable insights into the different types of risks that MLLMs face.
    *   **Comprehensive framework:** DREAM offers a structured and effective approach to enhancing safety alignment, combining SFT and RLAIF.
    *   **Strong experimental results:** The experiments demonstrate the effectiveness of DREAM on various benchmarks and models.

*   **Weaknesses:**

    *   **Reliance on teacher model:** The data synthesis step in risk-aware fine-tuning relies heavily on the capabilities of the teacher model. The quality of data is also a concern.
    *   **Limited exploration of modalities:** The work primarily focuses on image and text modalities. The approach's applicability to other modalities (e.g., audio, video) is not fully explored.
    *   **Potential Verbose responses:** MLLMs trained with DREAM may generate verbose responses that explicitly mention safety concerns.

*   **Potential Influence:** The paper has the potential to influence the field by:

    *   Encouraging more structured approaches to risk analysis in MLLMs.
    *   Providing a practical framework (DREAM) for enhancing safety alignment.
    *   Highlighting the importance of considering different risk combinations during training.
    *   Inspiring further research into automated feedback mechanisms for RLAIF.

*   **Caveats:** It's crucial to consider the following caveats when assessing the paper's influence:

    *   The long-term impact of the proposed framework will depend on its adoption by other researchers and practitioners.
    *   Further research is needed to address the limitations, particularly the reliance on a powerful teacher model and the limited exploration of modalities.

*   **Overall, this paper makes a valuable contribution to the field of MLLM safety by providing a structured approach to risk disentanglement and a comprehensive training framework for enhancing safety alignment. The experimental results are compelling, and the paper has the potential to influence future research in this area.**

Score: 8

*Rationale:*
The paper demonstrates significant novelty and addresses a critical concern in multimodal AI. The structured risk analysis and the DREAM framework offer a clear improvement in safety. The paper provides clear evidence and sound methodology that it achieves it's results. The drawbacks are mainly that it still relies on a teacher model for its framework and that it only touches on a couple modalities. While this framework has potential to be further developed, it has shown sufficient results.

- **Score**: 8/10

### **[PropRAG: Guiding Retrieval with Beam Search over Proposition Paths](http://arxiv.org/abs/2504.18070v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "PropRAG: Guiding Retrieval with Beam Search over Proposition Paths":

**Summary:**

The paper introduces PropRAG, a novel Retrieval Augmented Generation (RAG) framework designed to enhance reasoning and address limitations of standard RAG approaches in capturing the interconnectedness of knowledge. PropRAG achieves this through two key innovations: (1) Utilizing contextually rich propositions (extracted offline using an LLM) as knowledge units, moving beyond the context loss associated with triple-based Knowledge Graphs (KGs). (2) Employing an LLM-free online beam search algorithm to explicitly discover and score paths of interconnected propositions, mimicking multi-step reasoning chains.  Crucially, the online retrieval operates without LLM inference, reducing latency, cost and potential inconsistencies. The framework is evaluated on multi-hop Question Answering (QA) datasets, demonstrating state-of-the-art zero-shot Recall@5 and F1 scores compared to previous methods, including HippoRAG 2.

**Critical Evaluation:**

*   **Novelty:** The paper demonstrates innovation on several fronts. The central novelty lies in the combination of pre-extracted propositions with beam search for path discovery in RAG. While both propositions and beam search have been used separately in the NLP field, their integration as a non-parametric, LLM-free online retrieval strategy is novel. The shift from node-centric ranking (as in HippoRAG) to evaluating entire reasoning chains using beam search is also a significant conceptual advance. The use of propositions to overcome context loss compared to triple based KGs is valuable.
*   **Significance:** The paper addresses a crucial challenge in RAG: enabling more complex reasoning over interconnected knowledge. The gains in performance on multi-hop QA tasks demonstrate the practical significance of PropRAG's approach. The LLM-free online retrieval is also a significant advantage, reducing computational costs and potential inconsistencies. The zero shot performance indicates practical applicability of the system.
*   **Strengths:**
    *   Clear problem formulation and well-defined contributions.
    *   Comprehensive evaluation across multiple challenging datasets.
    *   Significant performance improvements over strong baselines, including the state-of-the-art HippoRAG 2.
    *   Well-motivated design choices and insightful ablation studies that highlight the contribution of the components.
    *   A compelling argument for a shift towards explicit, algorithmic modeling of reasoning paths in RAG.

*   **Weaknesses:**
    *   The system relies on high-quality offline proposition extraction using an LLM. While the online retrieval is LLM-free, the overall performance hinges on the effectiveness of the proposition extraction stage. The robustness of the system to errors or imperfections in proposition extraction is not fully explored.
    *   While it mentions computational cost considerations, a more detailed analysis of the computational overhead of beam search relative to simpler retrieval methods would be beneficial.
    *   The parameter tuning (beam width, path length, etc.) is dataset-specific. It could be more generalizable, or at least an exploration of a single parameter setting is needed.
*   **Potential Influence:** The paper has the potential to significantly influence the direction of RAG research. It encourages the development of more sophisticated retrieval strategies that go beyond simple vector similarity and incorporate explicit reasoning mechanisms. The focus on non-parametric, LLM-free online retrieval also offers a valuable alternative to approaches that rely on costly and potentially inconsistent online LLM calls. It can potentially inform future RAG designs that aim for both efficiency and reasoning capabilities.

**Justification for Score:**

PropRAG introduces a novel and effective framework that significantly advances the state-of-the-art in multi-hop QA. The combination of contextually rich propositions and LLM-free beam search for path discovery represents a significant step forward in enabling more complex reasoning capabilities in RAG systems. While the reliance on offline proposition extraction and the computational cost of beam search are limitations, the paper presents a well-designed and thoroughly evaluated solution with the potential to significantly influence future research in the field. Given the solid innovation, substantial performance improvements, and potential influence on future RAG architectures, I assign a score of 8.

**Score: 8**

- **Score**: 8/10

### **[Automating Function-Level TARA for Automotive Full-Lifecycle Security](http://arxiv.org/abs/2504.18083v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Automating Function-Level TARA for Automotive Full-Lifecycle Security":

**Summary:**

The paper introduces DefenseWeaver, a novel system that automates function-level Threat Analysis and Risk Assessment (TARA) for automotive security. Unlike existing approaches that rely on static threat libraries, DefenseWeaver leverages component-specific details and Large Language Models (LLMs) to dynamically generate attack trees and assess risk levels. It uses an extended OpenXSAM++ format to represent automotive configurations, a multi-agent LLM framework for attack method inference, and incorporates LoRA fine-tuning and RAG for adaptation to evolving threats and diverse standards. The system's effectiveness is demonstrated through deployment in real automotive security projects and its adaptability to other domains like UAVs and marine systems.  It is shown to outperform human experts in attack tree generation and improve cybersecurity efficiency and scalability, generating over 8,200 attack trees within cybersecurity platforms.

**Rigorous and Critical Evaluation:**

The paper presents a significant and well-executed contribution to the field of automotive cybersecurity. The key novelty lies in the application of LLMs to automate function-level TARA, a task traditionally reliant on manual effort and static threat libraries. The following points justify the assessment:

*   **Strengths:**

    *   **Automation of Function-Level TARA:**  This is the most compelling aspect. The paper addresses a critical gap by automating TARA at the function level, which is vital for comprehensive risk management, especially in the face of increasing system complexity and supply chain vulnerabilities.
    *   **Dynamic Threat Analysis:**  The use of LLMs, LoRA, and RAG allows the system to adapt to evolving threats and different standards, providing a more robust and up-to-date assessment compared to static, rule-based systems.
    *   **Comprehensive Approach:**  DefenseWeaver not only generates attack trees but also assesses risk levels, providing a more complete TARA solution.
    *   **Real-World Validation:** Deployment in real automotive projects, identification of vulnerabilities (later confirmed by penetration testing), and integration into industry cybersecurity platforms strengthens the paper's claims and highlights its practical value.
    *   **Cross-Domain Adaptability:** Demonstrating its applicability to UAVs and marine systems showcases the generalizability of the approach.
    *   **Performance Against Human Experts:**  Quantifiable improvements over manual attack tree generation further support the effectiveness of DefenseWeaver.

*   **Weaknesses:**

    *   **Reliance on LLMs:** The system's performance hinges on the capabilities of the underlying LLMs, which can be susceptible to biases and may require significant computational resources.  The paper only references GPT-4; a more thorough investigation across different LLMs would strengthen the work.
    *   **Threat Scenario Dependency:** While the system automates many aspects of TARA, it still requires users to define the initial threat scenarios.  The paper acknowledges this, indicating that full automation would require integration of STRIDE model but this remains future work.
    *   **Limited Open Source Availability:** Lack of full open-source code limits the wider research community's ability to validate and extend the work, although efforts have been made to compensate through releasing parts of the code.

*   **Significance:**

    *   **Improved Efficiency:**Automating TARA reduces the processing time significantly, addresses the cybersecurity skills shortage, and improves scalability.
    *   **Enhanced Security:** The system's ability to identify previously unknown attack paths and non-intuitive vulnerabilities strengthens automotive security.
    *   **Regulatory Compliance:** The system supports compliance with automotive cybersecurity regulations (WP29 R155), which is a critical factor for OEMs and suppliers.

*   **Overall Impact:**

    The paper's impact could be considerable. Automated and adaptive TARA solutions like DefenseWeaver have the potential to significantly improve automotive cybersecurity practices, mitigate risks associated with complex vehicle systems, and enhance regulatory compliance. The cross-domain applicability further broadens its relevance.

**Justification for Score:**

The paper addresses a critical and timely problem with an innovative solution grounded in solid engineering and experimental validation. While it has minor weaknesses related to reliance on LLMs and threat scenario dependency, its demonstrated strengths and potential impact outweigh those concerns.

Score: 8

- **Score**: 8/10

### **[Disentangle Identity, Cooperate Emotion: Correlation-Aware Emotional Talking Portrait Generation](http://arxiv.org/abs/2504.18087v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces DICE-Talk, a novel framework for generating emotional talking head videos using diffusion models. The approach addresses three key limitations of existing methods: insufficient utilization of audio's emotional cues, identity leakage in emotion representations, and isolated learning of emotion correlations.  DICE-Talk employs a disentangled emotion embedder that models audio-visual cues through cross-modal attention, representing emotions as identity-agnostic Gaussian distributions. It incorporates a correlation-enhanced emotion conditioning module with learnable emotion banks to capture inter-emotion relationships. Finally, it uses an emotion discrimination objective to enforce consistency during the diffusion process.  Experiments on MEAD and HDTF datasets demonstrate superior emotion accuracy compared to state-of-the-art methods while maintaining competitive lip-sync performance. Qualitative results and user studies support the generation of identity-preserving portraits with rich, correlated emotional expressions.

**Critical Evaluation:**

*   **Novelty:** The paper introduces several novel components. The disentangled emotion embedder using cross-modal attention and Gaussian distribution representation is a notable contribution. The correlation-enhanced emotion conditioning module with learnable emotion banks is also a novel way to explicitly model inter-emotion relationships, which is generally overlooked in the previous researches. The emotion discrimination objective used during the diffusion process is a decent contribution, which allows the model to generate more precise emotions.

*   **Significance:** The paper addresses a significant problem in the field of talking head generation: the lack of emotionally expressive portraits. Existing methods often produce "emotionally flat" outputs, limiting their applicability in areas like mental health support. By disentangling identity from emotion and modeling emotion correlations, DICE-Talk improves the realism and expressiveness of generated videos. This has practical implications for creating more engaging and lifelike digital humans. The quantitative and qualitative results are convincing, showing the superiority of DICE-Talk over existing methods in emotion accuracy and identity preservation. The user study further validates the subjective improvements in emotional expression and video smoothness. Code availability at GitHub facilitates reproducibility and further research. The paper fills a gap in the literature by directly addressing the generation of emotionally expressive talking heads using diffusion models, and the results have the potential to significantly advance the field.

*   **Strengths:**
    *   Clear problem definition and motivation.
    *   Well-designed framework with novel components (disentangled embedder, correlation-enhanced conditioning).
    *   Comprehensive experiments on standard datasets.
    *   Both quantitative and qualitative evaluation, including user studies.
    *   Demonstrated ability to generalize to unseen identities.
    *   Code availability.

*   **Weaknesses:**
    *   While the method shows improvement in emotion accuracy, there is a trade-off in the quality metrics for the videos/portraits themselves (FID, FVD). The trade-off could be discussed and studied in more detail.
    *   The improvement on HDTF datasets might be limited by the lack of obvious emotions in the training dataset.
    *   While the paper focuses on discrete emotions, future work could explore generating more subtle and nuanced emotional expressions.
    *   Ablation study could be more comprehensive, for example, ablating each part of the emotion conditioning module.

**Justification for Score:**

The paper makes several contributions to the field of talking head generation. The disentanglement strategy is particularly valuable, as is the consideration of cross-modal information and inter-emotion relationships. The quantitative and qualitative results sufficiently support the claims made in the paper. While there are some weaknesses regarding the trade-off between emotion accuracy and video quality, and the limitation that the training dataset in one experiment can be limited in terms of emotions and expressions, they do not detract significantly from the overall impact.

Score: 8

- **Score**: 8/10

### **[STP4D: Spatio-Temporal-Prompt Consistent Modeling for Text-to-4D Gaussian Splatting](http://arxiv.org/abs/2504.18318v1)**
- **Summary**: Here is a summary and evaluation of the paper:

**Summary:**

The paper introduces STP4D, a novel approach for generating high-quality text-to-4D content using Gaussian Splatting. The core idea is to integrate comprehensive spatio-temporal-prompt consistency modeling within a unified framework.  STP4D employs three key modules: Time-varying Prompt Embedding (TPE), Geometric Information Enhancement (GIE), and Temporal Extension Deformation (TED). TPE ensures prompt alignment by embedding accurate text features into Gaussians for each frame. GIE uses a GroupFormer and K-Planes to extract and enhance geometric information. TED extends content from anchor frames to actual frames using cross-attention.  STP4D also leverages the Diffusion model framework, enabling rapid inference and fine-grained 4D representation. Experimental results demonstrate that STP4D generates high-fidelity 4D content efficiently (around 4.6s per asset), outperforming existing methods in both quality and speed. The paper includes quantitative evaluations on the Diffusion4D dataset, qualitative visual comparisons, and ablation studies to validate the effectiveness of the proposed modules.

**Novelty and Significance:**

The paper's novelty lies in the following aspects:

1.  **Comprehensive Spatio-Temporal-Prompt Modeling:** STP4D is among the first approaches to explicitly address and integrate all three aspects – spatial consistency, temporal consistency, and prompt alignment – in a unified framework for text-to-4D generation. While prior works may focus on one or two aspects, STP4D aims for holistic consistency, which is a significant step forward.

2.  **Diffusion Model for 4D Gaussian Splatting Generation:** The paper is among the first to directly generate 4D Gaussian Splatting assets using a Diffusion model. This allows fine-grained 4D content representation with fast rendering and inference. Existing methods often rely on distilling knowledge from pretrained 2D/3D diffusion models, which can be suboptimal.

3.  **Efficient Generation:** The extremely fast generation time (4.6s per asset) is a significant practical contribution. This efficiency makes the method more accessible for real-time applications.

4.  **Modular Design:** The TPE, GIE, and TED modules provide a structured and interpretable approach to the problem. This modularity facilitates future research and improvements in each aspect of the generation process.

**Strengths:**

*   The paper presents a clear and well-structured explanation of the method.
*   The experimental results are compelling, demonstrating state-of-the-art performance in both quantitative metrics and qualitative comparisons.
*   The ablation studies thoroughly validate the contribution of each module.
*   The approach offers a good balance between quality and efficiency.
*   The user study reinforces the subjective superiority of the method.

**Weaknesses:**

*   While the method achieves fast generation times, it does require training a custom model, which can be computationally expensive (11 hours on 4x RTX 3090 GPUs).
*   The limitations section mentions the method struggles with complex scenes due to limited training data and a fixed number of Gaussians.
*   The impact of the design choices for KPlanes (specifically using explicit radiance fields instead of other alternatives) is not deeply explored.
* The experimental analysis should also explore scenarios when text prompts are potentially contradictory or ambiguous to investigate the robustness of STP4D

**Significance and Potential Influence:**

STP4D has the potential to influence the field of text-to-4D generation in several ways:

*   It sets a new benchmark for spatio-temporal-prompt consistency.
*   It motivates the exploration of Diffusion models for direct 4D Gaussian Splatting generation.
*   The efficient generation capabilities open up new avenues for real-time applications.
*   The modular design encourages further research into specific aspects of the generation pipeline.
*   The approach could be extended and adapted to other 4D representation formats.

**Score:** 8

**Rationale:**

The paper makes a significant contribution to the field by addressing a critical challenge: spatio-temporal-prompt consistency in text-to-4D generation. The method demonstrates clear advantages over existing techniques in both quality and efficiency, and its modular design facilitates future research. The use of Diffusion models for direct 4D Gaussian Splatting generation represents a substantial advancement. While there are some limitations, such as the requirement for training data and limited performance on complex scenes, the strengths of the paper far outweigh the weaknesses. The detailed analysis with ablation studies provides strong evidence supporting the effectiveness of the proposed approach. This positions the work to significantly influence future research and applications in this domain.

- **Score**: 8/10

### **[Unsupervised Visual Chain-of-Thought Reasoning via Preference Optimization](http://arxiv.org/abs/2504.18397v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Unsupervised Visual Chain-of-Thought Reasoning via Preference Optimization":

**Summary:**

This paper introduces Unsupervised Visual Chain-of-Thought (UV-COT), a novel framework for image-level reasoning.  Unlike existing visual CoT methods that rely on supervised fine-tuning with labeled bounding boxes, UV-COT uses a preference optimization approach. The key idea is to train a Multimodal Large Language Model (MLLM) to select preferred bounding boxes (indicating key image regions) over dis-preferred ones, without requiring explicit bounding box annotations. The process involves an automatic data generation pipeline: a target MLLM proposes bounding boxes and answers the question based on them, and another, more robust MLLM evaluates the quality of these answers. The rankings from the evaluator are used to train the target MLLM using a modified version of Direct Preference Optimization (DPO), encouraging the model to favor regions that lead to better answers.  Experiments on several datasets demonstrate the superiority of UV-COT compared to text-based CoT and supervised visual CoT methods, also exhibiting strong generalization on unseen datasets.

**Critical Evaluation:**

* **Novelty:** The paper presents a significant step forward by eliminating the need for human-annotated bounding boxes in visual CoT reasoning. Using preference optimization is not entirely new in LLMs, but adapting it to the *visual* domain and specifically for learning visual CoT is a novel and well-executed idea. The automatic data generation pipeline is a valuable contribution, as it allows for scalable training.  The Score-DPO method, which incorporates the *degree* of preference, is a further refinement of standard DPO and is also a meaningful contribution. Prior work has often focused on the *order* of preference, not the magnitude. The idea of training the MLLM by teaching it to emulate how humans visually perceive, identify key regions, and reason based on them, makes this paper an important contribution.

* **Significance:** The paper addresses a key limitation of current visual CoT approaches: the reliance on expensive and time-consuming labeled data.  UV-COT offers a more scalable and potentially generalizable approach. The empirical results are compelling, showing improved performance over both text-based and supervised visual CoT methods. Demonstrating good zero-shot performance is also a significant finding and highlights the ability of the approach to generalize to new tasks. The impact lies in democratizing visual reasoning in MLLMs, making it accessible without massive annotated datasets. The paper includes zero-shot analysis which is critical to test the robustness of the algorithm.

* **Strengths:**
    * **Well-defined problem and clear motivation:**  The paper clearly articulates the limitations of existing methods and motivates the need for unsupervised visual CoT.
    * **Technically sound approach:** The proposed framework is well-designed, combining data generation, a modified DPO loss, and an iterative learning strategy.
    * **Strong experimental results:** The paper provides extensive experimental results on multiple datasets, demonstrating the effectiveness and generalization ability of the proposed method. The ablation studies provide insights into the importance of various components of the framework. It's not just an incremental advancement, but a significant improvement.

* **Weaknesses:**
    * **Reliance on an Evaluator MLLM:**  The framework relies on a more robust MLLM (the evaluator) for data generation.  The performance of UV-COT is clearly tied to the quality of this evaluator.  While the authors provide some analysis of using a self-evaluated UV-COT, the performance drop is noticeable.
    * **Computational Cost:** Although UV-COT avoids human labeling, the iterative data generation and training process is computationally expensive, as is acknowledged in the paper. This factor has to be taken into consideration when applying the algorithm.
    * **Potential Biases in the Data Generation Pipeline:** The data generation pipeline depends on the evaluator and target models. Bias present in these models are likely to be amplified during the data generation and training.

* **Potential Influence:** The paper has the potential to influence the field of visual reasoning in MLLMs by providing a more scalable and generalizable alternative to supervised fine-tuning.  The framework and insights gained could be used to develop new unsupervised or self-supervised methods for visual reasoning and other complex vision-language tasks. The impact of this paper lies in democratizing visual reasoning in MLLMs, making it accessible without massive annotated datasets.

**Justification for Score:**

This paper has a strong positive impact, but is limited to computational cost and the need for an "evaluator" MLLM. However, given the significant novelty in the overall framework, the solid results, the demonstration of generalization, and the potential impact in the field, a high score is justified.

**Score: 8**

- **Score**: 8/10

### **[BitNet v2: Native 4-bit Activations with Hadamard Transformation for 1-bit LLMs](http://arxiv.org/abs/2504.18415v1)**
- **Summary**: Here's a summary and critical evaluation of the BitNet v2 paper:

**Summary:**

The paper introduces BitNet v2, a novel framework for enabling native 4-bit activation quantization for 1-bit Large Language Models (LLMs). The core innovation is the H-BitLinear layer, which incorporates an online Hadamard transformation before activation quantization in attention output and FFN down projection layers. This transformation reshapes the sharp, outlier-prone activation distributions into more Gaussian-like forms, making them more suitable for low-bit representation. The paper demonstrates that BitNet v2, when trained with 8-bit activations, matches the performance of BitNet b1.58 and can then be fine-tuned for native 4-bit activation use, offering significant improvements in memory footprint and computational cost for batched inference.

**Critical Evaluation:**

*   **Novelty:** The use of a Hadamard transform *before* quantization to improve the distribution of activations for low-bit LLMs is a genuinely novel contribution. Prior work attempted to deal with outlier activations, but not by proactively shaping the distribution in this manner. This pre-processing step distinguishes BitNet v2.
*   **Significance:** The ability to train 1-bit LLMs with native 4-bit activations has considerable practical significance. It directly addresses a bottleneck in deploying these models efficiently on hardware designed for low-bit computations. The reduction in memory footprint and computational cost translates directly to faster inference and improved energy efficiency.

**Strengths:**

*   **Clear Problem Definition:** The paper clearly articulates the problem of activation outliers hindering low-bit quantization.
*   **Technical Innovation:** The H-BitLinear layer is a simple yet effective solution to the problem.
*   **Experimental Validation:** The paper provides extensive experimental results across various model sizes (400M to 7B parameters) and benchmarks, demonstrating the effectiveness of BitNet v2. Ablation studies isolate the impact of the Hadamard transformation.
*   **Practical Impact:** The focus on batched inference efficiency is important for real-world deployment scenarios. The paper also explores different quantization strategies for K, Q, and V projections, showcasing its adaptability.

**Weaknesses:**

*   **Complexity of Hadamard Transform:** The use of the Hadamard Transform could potentially introduce computational overhead which may negate some of the gains depending on the specific hardware implementation. While the paper uses fast-hadamard-transform it may not always be a practical trade off in certain scenarios.
*   **Limited Ablation:** While some ablation studies are conducted, more investigation into the parameters of the H-BitLinear layer could be beneficial.
*   **Dependence on Existing Architectures:** The paper builds upon the LLaMA architecture. A broader exploration across different architectures might further demonstrate the generalizability of the H-BitLinear technique.
*   **Limited Exploration of Alternative Transformation Techniques:** While the Hadamard transformation is effective, the paper does not explicitly compare it to other potential transformations which could be investigated.

**Impact:**

The paper's potential impact is high. Lowering the activation bit-width while maintaining performance has significant implications for efficient LLM deployment.  It paves the way for more efficient hardware utilization and reduced inference costs, especially in edge computing scenarios. Follow-up research will likely explore variations of the H-BitLinear layer and its application to other model architectures.

**Justification for Score:**

The paper offers a significant contribution with its H-BitLinear layer and the ability to train BitNet v2 with native 4-bit activations while matching BitNet b1.58 performance. This addresses a critical bottleneck in deploying low-bit LLMs, significantly improving inference efficiency. While the Hadamard Transform complexity, needs further investigation, and the ablation study is rather limited this work advances the field and creates new opportunities for research.

Score: 8

- **Score**: 8/10

### **[Eval3D: Interpretable and Fine-grained Evaluation for 3D Generation](http://arxiv.org/abs/2504.18509v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Eval3D: Interpretable and Fine-grained Evaluation for 3D Generation":

**Summary**

The paper introduces Eval3D, a new evaluation tool designed for assessing the quality of generated 3D assets.  The tool emphasizes interpretability and fine-grained analysis, moving beyond reliance on black-box methods like large language models (LLMs). Eval3D operates by measuring the consistency across diverse foundation models and tools.  It evaluates 3D generations along five key dimensions: geometric consistency (alignment between surface normals and image-based normals), semantic consistency (stability of semantics across viewpoints), structural consistency (plausibility of the overall structure from multiple views), text-3D alignment (fidelity to the input text or image), and aesthetic appeal. The system provides pixel-level inconsistency measurements which are then used for accurate 3D spatial localization of artifacts. The paper presents a new benchmark dataset with scene graphs, text prompts, SDXL images, and human preferences and uses Eval3D to evaluate state-of-the-art 3D generation models, highlighting their limitations.

**Critical Evaluation**

*   **Novelty:** The paper offers a significantly more granular and interpretable approach to 3D generation evaluation compared to existing metrics. The core idea of leveraging consistency across multiple foundation models as a proxy for quality is clever and provides a pathway for better understanding the strengths and weaknesses of various 3D generation methods. The use of Depth Anything, DINov2, and Stable Zero123 to evaluate generated geometry and consistency are clever choices. Moreover, the metric's spatial localization of artifacts, providing clues to failure modes of current generation models, is a distinct advantage. The human preference annotations are also significantly expanded compared to prior work.
*   **Significance:** The problem of evaluating 3D generation is critical for advancing the field. While visual appeal is subjective, the paper tackles objective metrics like geometric and semantic consistency which are fundamental to creating useful 3D assets. The insights gained from using Eval3D can guide the development of improved 3D generation architectures and training methods. The curated dataset can become a valuable resource for the community.
*   **Strengths:**

    *   **Interpretability:** The system produces interpretable results, pinpointing specific issues like Janus faces or texture-geometry misalignment.
    *   **Fine-grained analysis:** It provides pixel-level feedback on geometric and semantic inconsistencies.
    *   **Comprehensive evaluation:** It covers multiple important dimensions of 3D asset quality.
    *   **Novel Approach**: Eval3D proposes a novel approach for evaluating the 3D quality.
    *   **Extensive Evaluation:** The work does more than a handful of evaluations, where 3D generative models are evaluated with different parameters.
*   **Weaknesses:**

    *   **Reliance on Foundation Models:**  The accuracy of Eval3D is inherently tied to the quality of the underlying foundation models (e.g., Depth Anything, DINOv2). Errors or biases in these models will propagate into the evaluation results. While they acknowledge this limitation, it is a core dependency of the system.
    *   **Metric Selection:** The selection of threshold parameters (e.g., the `norm` threshold for geometric consistency) seems somewhat arbitrary and based on a hold-out set. A more rigorous justification or sensitivity analysis could strengthen the paper.
    *   **Computational Cost:** While providing detailed results, using multiple foundation models could be computationally expensive for comprehensive and large-scale evaluation. The run-time experiments would have benefited the reader to evaluate feasibility.
    *   **Scope of Validation**: Eval3D has shown promising results with many 3D datasets but is lacking tests on out-of-distribution data (e.g. different styles, complex scenes, etc.).
*   **Potential Impact:** Eval3D has the potential to become a standard evaluation tool in the 3D generation community. Its interpretable results and fine-grained analysis can accelerate research by providing valuable insights into the strengths and weaknesses of different approaches. However, it also raises interesting ethical concerns, e.g., employment within creative fields, that researchers can keep in mind.

**Justification of Score:**

The paper presents a solid contribution to the field of 3D generation by addressing a critical need for more reliable and interpretable evaluation metrics. The approach is novel, the analysis is thorough, and the potential impact is significant. Despite the limitations tied to the reliance on foundation models, the paper provides a valuable step forward. Therefore, a score of 8 is justified. It's not a groundbreaking, paradigm-shifting paper, but it addresses a fundamental problem with a well-designed solution and holds considerable promise for advancing the field.

**Score: 8**

- **Score**: 8/10

## Other Papers
### **[Towards Machine-Generated Code for the Resolution of User Intentions](http://arxiv.org/abs/2504.17531v1)**
### **[Auditing the Ethical Logic of Generative AI Models](http://arxiv.org/abs/2504.17544v1)**
### **[A Comprehensive Survey of Knowledge-Based Vision Question Answering Systems: The Lifecycle of Knowledge in Visual Reasoning Task](http://arxiv.org/abs/2504.17547v1)**
### **[HalluLens: LLM Hallucination Benchmark](http://arxiv.org/abs/2504.17550v1)**
### **[DeepDistill: Enhancing LLM Reasoning Capabilities via Large-Scale Difficulty-Graded Data Training](http://arxiv.org/abs/2504.17565v2)**
### **[A Multi-Agent, Laxity-Based Aggregation Strategy for Cost-Effective Electric Vehicle Charging and Local Transformer Overload Prevention](http://arxiv.org/abs/2504.17575v1)**
### **[L3: DIMM-PIM Integrated Architecture and Coordination for Scalable Long-Context LLM Inference](http://arxiv.org/abs/2504.17584v1)**
### **[Beyond Labels: Zero-Shot Diabetic Foot Ulcer Wound Segmentation with Self-attention Diffusion Models and the Potential for Text-Guided Customization](http://arxiv.org/abs/2504.17628v1)**
### **[polyGen: A Learning Framework for Atomic-level Polymer Structure Generation](http://arxiv.org/abs/2504.17656v1)**
### **[Evaluating Grounded Reasoning by Code-Assisted Large Language Models for Mathematics](http://arxiv.org/abs/2504.17665v1)**
### **[Towards a HIPAA Compliant Agentic AI System in Healthcare](http://arxiv.org/abs/2504.17669v1)**
### **[Cross-region Model Training with Communication-Computation Overlapping and Delay Compensation](http://arxiv.org/abs/2504.17672v1)**
### **[Energy Considerations of Large Language Model Inference and Efficiency Optimizations](http://arxiv.org/abs/2504.17674v1)**
### **[INSIGHT: Bridging the Student-Teacher Gap in Times of Large Language Models](http://arxiv.org/abs/2504.17677v1)**
### **[Ensemble Bayesian Inference: Leveraging Small Language Models to Achieve LLM-level Accuracy in Profile Matching Tasks](http://arxiv.org/abs/2504.17685v1)**
### **[Generative Fields: Uncovering Hierarchical Feature Control for StyleGAN via Inverted Receptive Fields](http://arxiv.org/abs/2504.17712v1)**
### **[Multilingual Performance Biases of Large Language Models in Education](http://arxiv.org/abs/2504.17720v1)**
### **[Towards Robust LLMs: an Adversarial Robustness Measurement Framework](http://arxiv.org/abs/2504.17723v1)**
### **[Conversational Assistants to support Heart Failure Patients: comparing a Neurosymbolic Architecture with ChatGPT](http://arxiv.org/abs/2504.17753v1)**
### **[Replay to Remember: Retaining Domain Knowledge in Streaming Language Models](http://arxiv.org/abs/2504.17780v1)**
### **[The Role of Open-Source LLMs in Shaping the Future of GeoAI](http://arxiv.org/abs/2504.17833v1)**
### **[Do We Need Transformers to Play FPS Video Games?](http://arxiv.org/abs/2504.17891v1)**
### **[DCT-Shield: A Robust Frequency Domain Defense against Malicious Image Editing](http://arxiv.org/abs/2504.17894v1)**
### **[Toward a Human-Centered Evaluation Framework for Trustworthy LLM-Powered GUI Agents](http://arxiv.org/abs/2504.17934v1)**
### **[Masked strategies for images with small objects](http://arxiv.org/abs/2504.17935v1)**
### **[Evaluating Machine Expertise: How Graduate Students Develop Frameworks for Assessing GenAI Content](http://arxiv.org/abs/2504.17964v1)**
### **[LLM Agent Swarm for Hypothesis-Driven Drug Discovery](http://arxiv.org/abs/2504.17967v1)**
### **[Cluster-Aware Attacks on Graph Watermarks](http://arxiv.org/abs/2504.17971v1)**
### **[Optimism, Expectation, or Sarcasm? Multi-Class Hope Speech Detection in Spanish and English](http://arxiv.org/abs/2504.17974v1)**
### **[Back to Fundamentals: Low-Level Visual Features Guided Progressive Token Pruning](http://arxiv.org/abs/2504.17996v1)**
### **[Streaming, Fast and Slow: Cognitive Load-Aware Streaming for Efficient LLM Serving](http://arxiv.org/abs/2504.17999v1)**
### **[Diffusion-Driven Universal Model Inversion Attack for Face Recognition](http://arxiv.org/abs/2504.18015v1)**
### **[Enhancing Privacy-Utility Trade-offs to Mitigate Memorization in Diffusion Models](http://arxiv.org/abs/2504.18032v1)**
### **[RAG LLMs are Not Safer: A Safety Analysis of Retrieval-Augmented Generation for Large Language Models](http://arxiv.org/abs/2504.18041v1)**
### **[A BERT-Style Self-Supervised Learning CNN for Disease Identification from Retinal Images](http://arxiv.org/abs/2504.18049v1)**
### **[Validating Network Protocol Parsers with Traceable RFC Document Interpretation](http://arxiv.org/abs/2504.18050v1)**
### **[DREAM: Disentangling Risks to Enhance Safety Alignment in Multimodal Large Language Models](http://arxiv.org/abs/2504.18053v1)**
### **[POET: Prompt Offset Tuning for Continual Human Action Adaptation](http://arxiv.org/abs/2504.18059v1)**
### **[LLM-Guided Open RAN: Empowering Hierarchical RAN Intelligent Control](http://arxiv.org/abs/2504.18062v1)**
### **[PropRAG: Guiding Retrieval with Beam Search over Proposition Paths](http://arxiv.org/abs/2504.18070v1)**
### **[Stabilizing Reasoning in Medical LLMs with Continued Pretraining and Reasoning Preference Optimization](http://arxiv.org/abs/2504.18080v1)**
### **[Automating Function-Level TARA for Automotive Full-Lifecycle Security](http://arxiv.org/abs/2504.18083v1)**
### **[Random-Set Large Language Models](http://arxiv.org/abs/2504.18085v1)**
### **[Disentangle Identity, Cooperate Emotion: Correlation-Aware Emotional Talking Portrait Generation](http://arxiv.org/abs/2504.18087v1)**
### **[Application and Optimization of Large Models Based on Prompt Tuning for Fact-Check-Worthiness Estimation](http://arxiv.org/abs/2504.18104v1)**
### **[Think, Prune, Train, Improve: Scaling Reasoning without Scaling Models](http://arxiv.org/abs/2504.18116v1)**
### **[NoEsis: Differentially Private Knowledge Transfer in Modular LLM Adaptation](http://arxiv.org/abs/2504.18147v1)**
### **[Leveraging Decoder Architectures for Learned Sparse Retrieval](http://arxiv.org/abs/2504.18151v1)**
### **[Optimizing Multi-Round Enhanced Training in Diffusion Models for Improved Preference Understanding](http://arxiv.org/abs/2504.18204v1)**
### **[Efficient Single-Pass Training for Multi-Turn Reasoning](http://arxiv.org/abs/2504.18246v1)**
### **[MAGI: Multi-Agent Guided Interview for Psychiatric Assessment](http://arxiv.org/abs/2504.18260v1)**
### **[TextTIGER: Text-based Intelligent Generation with Entity Prompt Refinement for Text-to-Image Generation](http://arxiv.org/abs/2504.18269v1)**
### **[Artificial Intelligence health advice accuracy varies across languages and contexts](http://arxiv.org/abs/2504.18310v1)**
### **[Towards Adaptive Software Agents for Debugging](http://arxiv.org/abs/2504.18316v1)**
### **[STP4D: Spatio-Temporal-Prompt Consistent Modeling for Text-to-4D Gaussian Splatting](http://arxiv.org/abs/2504.18318v1)**
### **[SSD-Poser: Avatar Pose Estimation with State Space Duality from Sparse Observations](http://arxiv.org/abs/2504.18332v1)**
### **[Comparing Uncertainty Measurement and Mitigation Methods for Large Language Models: A Systematic Review](http://arxiv.org/abs/2504.18346v1)**
### **[Revisiting Data Auditing in Large Vision-Language Models](http://arxiv.org/abs/2504.18349v1)**
### **[Testing Individual Fairness in Graph Neural Networks](http://arxiv.org/abs/2504.18353v1)**
### **[ThreMoLIA: Threat Modeling of Large Language Model-Integrated Applications](http://arxiv.org/abs/2504.18369v1)**
### **[Auto-SLURP: A Benchmark Dataset for Evaluating Multi-Agent Frameworks in Smart Personal Assistant](http://arxiv.org/abs/2504.18373v1)**
### **[Pushing the boundary on Natural Language Inference](http://arxiv.org/abs/2504.18376v1)**
### **[Bridge the Domains: Large Language Models Enhanced Cross-domain Sequential Recommendation](http://arxiv.org/abs/2504.18383v1)**
### **[Fast Autoregressive Models for Continuous Latent Generation](http://arxiv.org/abs/2504.18391v1)**
### **[Unsupervised Visual Chain-of-Thought Reasoning via Preference Optimization](http://arxiv.org/abs/2504.18397v1)**
### **[HepatoGEN: Generating Hepatobiliary Phase MRI with Perceptual and Adversarial Models](http://arxiv.org/abs/2504.18405v1)**
### **[HRScene: How Far Are VLMs from Effective High-Resolution Image Understanding?](http://arxiv.org/abs/2504.18406v1)**
### **[An Empirical Study of Evaluating Long-form Question Answering](http://arxiv.org/abs/2504.18413v1)**
### **[BitNet v2: Native 4-bit Activations with Hadamard Transformation for 1-bit LLMs](http://arxiv.org/abs/2504.18415v1)**
### **[LLMpatronous: Harnessing the Power of LLMs For Vulnerability Detection](http://arxiv.org/abs/2504.18423v1)**
### **[Reason Like a Radiologist: Chain-of-Thought and Reinforcement Learning for Verifiable Report Generation](http://arxiv.org/abs/2504.18453v1)**
### **[Investigating Co-Constructive Behavior of Large Language Models in Explanation Dialogues](http://arxiv.org/abs/2504.18483v1)**
### **[Eval3D: Interpretable and Fine-grained Evaluation for 3D Generation](http://arxiv.org/abs/2504.18509v1)**
### **[TRACE Back from the Future: A Probabilistic Reasoning Approach to Controllable Language Generation](http://arxiv.org/abs/2504.18535v1)**
