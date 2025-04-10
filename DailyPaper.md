# The Latest Daily Papers - Date: 2025-04-10
## Highlight Papers
### **[Navigating the Rabbit Hole: Emergent Biases in LLM-Generated Attack Narratives Targeting Mental Health Groups](http://arxiv.org/abs/2504.06160v2)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper "Navigating the Rabbit Hole: Emergent Biases in LLM-Generated Attack Narratives Targeting Mental Health Groups" investigates how Large Language Models (LLMs) can generate harmful and stigmatizing narratives about mental health groups even when not explicitly prompted to do so. The authors make three key contributions: (1) they explicitly evaluate LLM-generated attacks on mental health groups; (2) they use a network-based framework to analyze the propagation of biases; and (3) they assess the degree of stigmatization emerging from these attacks.  They find that mental health entities are disproportionately targeted and occupy central positions within attack narrative networks.  Their stigmatization analysis shows increased labeling and discriminatory components directed towards mental health-related targets as generative chains progress, suggesting that LLMs structurally heighten harmful discourse. The paper concludes by highlighting the need for mitigation strategies to address these emergent biases.

**Critical Evaluation:**

*   **Novelty:** The paper's novelty stems from its specific focus on *unprovoked* targeting of mental health groups in LLM-generated narratives. While research has addressed biases in LLMs concerning other demographics, the specific examination of mental health within a generative, chain-like context is relatively new. The use of a network approach to analyze the propagation of bias, combined with a stigmatization component analysis, adds another layer of methodological innovation. This shifts bias evaluation beyond static sentence analyses into a networked understanding of harm.

*   **Significance:** The paper is significant because it uncovers a critical blind spot in LLM safety evaluations. Current safeguards often focus on preventing directly prompted harmful content. This work demonstrates that LLMs can autonomously generate harmful content, suggesting implicit biases baked into the model architecture itself and magnified over generative sequences. This has implications for how LLMs are used in clinical decision-making, mental health support tools, and content moderation, as these applications can unintentionally reinforce stigma.  Moreover, it contributes to the broader discussion on algorithmic harm and the need for proactive interventions rather than reactive moderation.

*   **Strengths:**
    *   **Clear Research Questions:** The research questions are well-defined and address a timely and important issue.
    *   **Methodological Rigor:** The paper combines quantitative network analysis with qualitative insights from stigmatization theory, providing a rich and nuanced understanding of the problem.
    *   **Strong Empirical Evidence:** The findings are supported by a robust analysis of a large-scale dataset of LLM-generated outputs.
    *   **Practical Implications:**  The paper highlights the need for better LLM safety mechanisms and offers insights for designing interventions to prevent the spread of harmful narratives.

*   **Weaknesses:**
    *   **Model Specificity:** The analysis is primarily based on a single LLM, Mistral-7B, which raises concerns about generalizability. While the authors discuss the potential for wider applicability, further studies with other LLMs are needed to confirm the findings.
    *   **Framework Dependence:** The reliance on the "Toxicity Rabbit Hole" framework, while providing a structured approach, might also limit the exploration of other potential pathways for emergent bias. The results may be conditioned by this particular method of iteratively generating toxic content.
    *   **Annotation Limitations:** While the authors describe their methodology for the extraction of entities and their classification, the reliance on an LLM to measure stigmatization components is prone to errors of its own.
    *  **Limited counterfactuals**: A limited set of counterfactual interventions are provided, in order to illustrate the potential for mitigation, and thus, limited empirical validation.

*   **Potential Influence:** The paper has the potential to influence future research on LLM safety and bias mitigation. It calls for a shift from reactive moderation to proactive, trajectory-aware interventions. It encourages interdisciplinary collaborations between computer scientists, sociologists, and mental health professionals to address these complex issues. It can also inform the development of ethical guidelines for LLM development and deployment.

**Justification of Score:**

The paper is a valuable contribution to the field of LLM safety and ethics. While some limitations exist regarding generalizability and framework dependence, the novelty of the approach, the significance of the findings, and the practical implications warrant a high score. The study provides empirical evidence of the complex ways in which LLMs can generate harmful content, even without explicit prompting. The combination of network science and social science theories is particularly effective in providing insight into how biases propagate. I give it a score of **8** because it identifies a vital issue and provides a well-reasoned exploration of it, although future research should address the limitations mentioned above to enhance its generalizability.

**Score: 8**
- **Score**: 8/10

### **[TxGemma: Efficient and Agentic LLMs for Therapeutics](http://arxiv.org/abs/2504.06196v1)**
- **Summary**: Here's a summary and critical evaluation of the TxGemma paper:

**Summary:**

The paper introduces TxGemma, a suite of efficient, generalist Large Language Models (LLMs) designed for therapeutic development. Fine-tuned from Gemma-2 on a comprehensive dataset of small molecules, proteins, nucleic acids, diseases, and cell lines, TxGemma demonstrates superior or comparable performance on 66 therapeutic development tasks compared to state-of-the-art generalist and specialist models.  Key features include:

*   Efficient generalist performance using 2B, 9B, and 27B parameter models.
*   Explainable and interactive capabilities through TxGemma-Chat, allowing for natural language interaction and mechanistic reasoning.
*   Agentic orchestration of therapeutic development workflows via Agentic-Tx, an agent system powered by Gemini 2.5 that reasons, acts, and manages diverse workflows.
*   Open model release to enable innovative research and adaptation by the broader scientific community.

The authors evaluate TxGemma against existing models on a range of predictive and generative tasks and showcase the capabilities of Agentic-Tx on reasoning-intensive benchmarks.

**Critical Evaluation:**

*   **Novelty:**  The paper presents several novel elements. While LLMs have been explored in various scientific domains, TxGemma represents a significant advancement by creating a *generalist* LLM specifically for therapeutics that achieves competitive performance compared to specialized models and generalist models like Tx-LLM. TxGemma-Chat's ability to provide reasoning and explanations is also a significant step forward. The agentic framework leveraging Gemini 2.5 is novel in the therapeutic domain, demonstrating a path toward more automated and sophisticated research workflows.

*   **Significance:** The potential impact of TxGemma on the pharmaceutical industry is considerable. By providing accurate predictions, enabling interactive reasoning, and automating complex workflows, TxGemma has the potential to significantly reduce attrition rates and accelerate drug development timelines. The open release of TxGemma is particularly important, as it empowers researchers to adapt and validate the models on their own data, facilitating broader adoption and innovation.

*   **Strengths:**
    *   **Comprehensive Evaluation:** The authors conduct a thorough evaluation of TxGemma across a wide range of therapeutic development tasks, providing strong evidence for its effectiveness.
    *   **Clear Presentation:** The paper is well-written and clearly articulates the motivations, methods, and results of the research.
    *   **Open Release:**  The decision to release TxGemma as an open model is commendable and will significantly accelerate progress in the field.

*   **Weaknesses:**
    *   **In-silico validation only:** The authors acknowledge that the performance of TxGemma has not yet been validated in real-world, wet-lab experiments. Prospective validation is a crucial next step to confirm its practical utility.
    *   **Reasoning depth limitations:**  While TxGemma-Chat offers reasoning capabilities, the explanations provided may be limited and require careful interpretation. Future research should focus on improving the reliability and comprehensiveness of these explanations.
    *   **Reliance on TDC:** The model is primarily trained and evaluated on the Therapeutics Data Commons (TDC). While TDC is a valuable resource, it may not fully represent the diversity of real-world therapeutic data.

*   **Potential Influence:** TxGemma has the potential to significantly influence the field of therapeutic AI by:
    *   Accelerating drug discovery and development.
    *   Enabling more efficient and transparent research workflows.
    *   Empowering researchers to leverage the power of LLMs for therapeutic applications.
    *   Providing a foundation for future advancements in the field.

**Rigorous Rationale for Score:**

I've assigned a score of **8** for the TxGemma paper. Here's a detailed justification:

*   **High Impact (3 points):**  The potential impact of TxGemma is significant.  The pharmaceutical industry faces substantial challenges, and TxGemma offers a promising approach to address these challenges by accelerating drug discovery and development while reducing costs and risks. The introduction of TxGemma-Chat for interactive reasoning has the potential to transform how the field is approached.
*   **Substantial Novelty (3 points):** The combination of a therapeutic-specific generalist LLM, explainable AI capabilities, and an agentic framework represents a notable advancement.  While other LLMs have been applied to scientific domains, TxGemma's focus on therapeutics and its open release differentiate it from prior work.
*   **Solid Evaluation (2 points):**  The authors provide a thorough evaluation of TxGemma across a diverse range of therapeutic tasks, demonstrating its superior or comparable performance to existing models. However, the lack of wet-lab validation represents a limitation.

The paper provides a novel and significant contribution to the field, but also recognizes potential limitations that should be addressed in future work.

Score: 8

- **Score**: 8/10

### **[LExT: Towards Evaluating Trustworthiness of Natural Language Explanations](http://arxiv.org/abs/2504.06227v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary**

The paper "LEXT: Towards Evaluating Trustworthiness of Natural Language Explanations" introduces a framework for evaluating the trustworthiness of natural language explanations generated by Large Language Models (LLMs), particularly in high-stakes domains like healthcare. The framework centers around two core dimensions: *Plausibility* (how reasonable and robust an explanation appears to humans) and *Faithfulness* (how accurately an explanation reflects the model's actual reasoning). It proposes a Language Explanation Trustworthiness (LExT) score that balances these dimensions. The authors apply the framework to assess different LLMs (domain-specific and general-purpose) on medical datasets, revealing variations in their ability to produce trustworthy explanations. The paper also presents several novel metrics, including Context Relevancy, QAG Score, Counterfactual Stability, and Contextual Faithfulness.

**Critical Evaluation**

*   **Novelty:** The paper's novelty lies in its holistic, quantitative framework for assessing the trustworthiness of LLM-generated explanations. While individual metrics like plausibility and faithfulness have been explored, the LExT framework integrates them and introduces new ones (e.g., contextual faithfulness, counterfactual stability).  This is a valuable contribution because the increasing reliance on LLMs necessitates robust evaluation methods that go beyond simple accuracy metrics.  The paper acknowledges and builds upon existing work (e.g., OpenXAI), but carves out its unique space by offering a more integrated and application-oriented (healthcare) approach.

*   **Significance:** The paper addresses a crucial and timely problem: the need to evaluate the reliability and transparency of LLM explanations. The potential impact of unreliable explanations in sensitive areas like healthcare is significant.  The work highlights that existing NLG metrics are insufficient, which is a key finding. Demonstrating the variations in trustworthiness among different LLMs, including how general-purpose models sometimes outperform domain-specific ones in faithfulness, has significant practical implications for model selection and development. The paper contributes to making LLMs safer and more reliable for high-stakes decision-making.  The specific focus on the healthcare domain is a strength, increasing the practical relevance of the work. The paper provides a methodology (and code) for other researchers to use and adapt. The code is publicly available which makes the research immediately useful. The limitations of individual models such as Meditron is discussed and this further supports the need for holistic evaluations.

*   **Strengths:**

    *   Clear and well-defined framework.
    *   Comprehensive set of evaluation metrics.
    *   Application to a relevant high-stakes domain (healthcare).
    *   Comparison of domain-specific and general-purpose LLMs.
    *   Publicly available code, enabling reproducibility.
    *   Addresses an urgent need for evaluation frameworks in the context of rising usage of LLMs
*   **Weaknesses:**

    *   The framework relies heavily on BERT embeddings and the results are affected by the quality of BERT embeddings.
    *   While the LExT score attempts to balance Plausibility and Faithfulness, the specific weighting and the form of the disagreement penalty function (g(x)) could be further explored and justified.  The authors state the function ensures the metric derives into a harmonic mean but a further rigorous justification of why this should be used, as opposed to other methods of weighting, is required.
    *   The generalizability of the framework to *other* high-stakes domains besides healthcare could be discussed more explicitly. While the metrics are designed to be dataset-agnostic, tailoring them to the specifics of, for example, legal or financial contexts may require some adjustments.

*   **Potential Influence:** This paper is likely to influence future research on evaluating LLM explanations, especially in high-stakes application areas. It provides a practical, quantifiable framework that can be used to compare different models and guide the development of more trustworthy LLMs. The framework can also contribute to building more transparent and accountable AI systems, addressing regulatory requirements (e.g., the EU AI Act). The paper can further motivate research in the area of explainable artificial intelligence and potentially influence practitioners when selecting their models for use in real-world applications.

**Score: 8**

**Rationale:**

The paper offers a substantial contribution to the field by providing a practical and comprehensive framework for evaluating the trustworthiness of LLM explanations. While the framework may benefit from further refinement in terms of weighting and generalizability, the core concepts, the novel metrics, the application to healthcare, and public code release make this a valuable and influential piece of work. The paper tackles an important aspect of LLMs in high-stakes applications and provides a foundation for improvements and adoption by practitioners and researchers. The paper is not "groundbreaking" or revolutionary enough to warrant a 9 or 10, but its strong methodology and relevance to a significant problem makes a score of 8 appropriate.
- **Score**: 8/10

### **[Defending LLM Watermarking Against Spoofing Attacks with Contrastive Representation Learning](http://arxiv.org/abs/2504.06575v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper addresses the vulnerability of existing LLM watermarking techniques to "spoofing attacks," where malicious actors alter the meaning of watermarked text while preserving the watermark, leading to potential misattribution of harmful content.  The authors identify two key challenges in defending against spoofing: (1) the need for watermarks to be both sensitive to semantic-distorting changes and insensitive to semantic-preserving edits, and (2) the difficulty of detecting global semantic shifts using local, auto-regressive watermarking schemes. To overcome these challenges, they propose a semantic-aware watermarking algorithm that post-hoc embeds watermarks into a given text while preserving its original meaning.  Their approach uses a contrastively trained "semantic mapping model" to guide the generation of a green-red token list, making it sensitive to semantic distortions and robust to meaning-preserving edits. Experiments on standard benchmarks demonstrate strong robustness against removal attacks and improved security against spoofing attacks (sentiment reversal and toxic content injection) while maintaining watermark detectability.

**Critical Evaluation:**

*   **Novelty:** The paper presents a significant and novel perspective on LLM watermarking by explicitly focusing on security against spoofing attacks. While existing research has addressed robustness against removal, the vulnerability to adversarial semantic manipulation has been largely understudied. The introduction of a contrastively trained semantic mapping model specifically designed to differentiate between permissible and impermissible text alterations is a valuable contribution.  The idea of a globally conditioned watermark is also novel. However, the authors draw some inspiration from existing post-hoc watermarking techniques, particularly Liu & Bu (2024), which could be viewed as incremental improvement rather than groundbreaking innovation.

*   **Significance:** The problem addressed is highly significant in the context of responsible AI.  If watermarks can be spoofed, the entire mechanism of using watermarks to identify the source of generated content is undermined.  The authors' solution has the potential to improve the trustworthiness and reliability of LLM watermarking, making it a more viable tool for combating misinformation and harmful content. The experimental results, demonstrating improvements in security against spoofing without sacrificing other key criteria, add to the significance.  The ablation studies that isolate the contributions of global conditioning and contrastive learning provide compelling evidence for the utility of the proposed techniques.

*   **Strengths:**

    *   **Clear Problem Definition:**  The paper clearly articulates the problem of spoofing attacks and the associated challenges.
    *   **Well-Motivated Approach:** The design choices of the semantic mapping model and contrastive training are logically justified and aligned with the identified challenges.
    *   **Comprehensive Evaluation:**  The experiments cover a range of attack scenarios and compare the proposed method against several strong baselines.
    *   **Ablation Studies:**  The ablation studies effectively demonstrate the importance of the key components of the proposed approach.
    *   **Code Availability:** The availability of code enhances the reproducibility and impact of the work.

*   **Weaknesses:**

    *   **Computational Cost:** The contrastive training process might be computationally expensive. While the paper provides detailed implementation, it lacks a discussion of the overall training and inference time.
    *   **Reliance on Pre-trained Models:**  The initial embedding model might still have biases or shortcomings that could affect the performance of the watermarking algorithm, even with contrastive fine-tuning. The paper does not thoroughly discuss potential limitations of leveraging pre-trained language models.
    *   **Limited Attack Scenarios:** While the paper covers sentiment reversal and toxic content injection, other types of spoofing attacks could exist.

*   **Potential Influence:** The paper is likely to stimulate further research on security-focused watermarking techniques. The contrastive learning approach could be adapted and extended to defend against a broader range of adversarial attacks.

**Justification for Score:**

The paper tackles an important and under-explored problem in LLM watermarking with a well-motivated and empirically supported approach. The use of a contrastively trained semantic mapping model to address the challenges of spoofing attacks is a significant contribution. While the method builds upon existing techniques, the novelty and potential impact are considerable.  The weaknesses, such as computational cost, are manageable and do not detract significantly from the overall value of the work.

Score: 8

- **Score**: 8/10

### **[Probability Density Geodesics in Image Diffusion Latent Space](http://arxiv.org/abs/2504.06675v1)**
- **Summary**: Here's a summary and critical evaluation of the provided research paper:

**Summary:**

The paper introduces a method for computing geodesics within the latent space of diffusion models. It leverages the inherent probability density estimated by these models to define a Riemannian manifold, where paths through high-density (probable) regions are considered "shorter" than those traversing low-density areas.  The authors present algorithms for solving initial and boundary value problems (IVP/BVP) associated with these geodesics, including techniques to address challenges specific to diffusion models (e.g., score estimation in low-density regions, conditional densities). The paper demonstrates applications in training-free image sequence interpolation and extrapolation, showing competitive performance compared to existing methods.  It also conducts experiments to analyze the properties of video clips in diffusion latent space, exploring how closely they approximate geodesics.

**Critical Evaluation:**

*   **Novelty:** The core idea of computing geodesics weighted by probability density in the latent space of a diffusion model is a significant contribution. This is not simply applying a known technique but formulating a novel framework for exploring and manipulating diffusion model latent spaces. The approaches to handling the specific challenges of high dimensionality and noisy score estimation are valuable. The application to training-free video manipulation is a clever demonstration of the concept. The paper also makes theoretical contributions by presenting algorithms for solving the IVP/BVP.
*   **Significance:** The potential impact of this work is substantial. The ability to find "shortest" paths between images in latent space opens up possibilities for controllable image generation and manipulation. The training-free aspect is particularly appealing as it allows leveraging pre-trained models without the need for fine-tuning. The theoretical framework is also a great way to understand the interpolation and extrapolation within diffusion models. The geodesic analysis of existing video clips offers insights into the structure of the learned latent space.
*   **Strengths:**
    *   Strong theoretical foundation and rigorous derivations.
    *   Addresses practical challenges in applying geodesic computations to diffusion models.
    *   Demonstrates competitive performance in image interpolation/extrapolation without training.
    *   Provides insightful analysis of video clips in latent space.
    *   Code availability enhances reproducibility and further research.
*   **Weaknesses:**
    *   The evaluation metrics used are subjective and don't always capture interpolation quality (smoothness, directness and realism) fully.
    *   The method struggles with large appearance changes or semantic gaps between input images.
    *   The image extrapolation performance seems unreliable and there is a need for a more robust technique.
    *   Score estimation in low-density regions of the latent space can be problematic leading to poor results.
*   **Potential Influence:** This work has the potential to influence several areas:
    *   Diffusion model manipulation and control: Geodesics can provide a principled way to navigate the latent space.
    *   Video editing and generation: Training-free interpolation and extrapolation techniques are highly desirable.
    *   Understanding latent spaces: Analyzing geodesic paths can reveal insights into the structure and properties of latent spaces.

**Justification:**

The paper presents a novel formulation with strong theoretical backing and compelling applications. While there are limitations, the work introduces a valuable framework for exploring and manipulating diffusion model latent spaces. The practical approaches and the training-free aspect makes the method potentially more applicable. The potential influence of this paper is significant, which is why I assign it a rating higher than the average paper.

Score: 8

- **Score**: 8/10

### **[Compass Control: Multi Object Orientation Control for Text-to-Image Generation](http://arxiv.org/abs/2504.06752v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "Compass Control: Multi Object Orientation Control for Text-to-Image Generation":

**Summary:**

The paper introduces "Compass Control," a novel approach to explicitly control the 3D orientation of multiple objects within text-to-image (T2I) diffusion models.  The method leverages "compass tokens" alongside text tokens, conditioned by a lightweight encoder network that takes object orientation as input.  To address issues with diffuse attention, a "Coupled Attention Localization (CALL)" mechanism constrains cross-attention maps between compass tokens and corresponding object regions.  The method is trained on synthetic data but demonstrates strong generalization to unseen objects and complex scenes. Further, it is extended with personalized objects, achieving state-of-the-art orientation control, and text alignment as evidenced through evaluations and user study.

**Critical Evaluation:**

**Novelty:**

The paper presents a genuinely novel approach. While prior work has explored 3D control in T2I generation, it's typically been limited to camera viewpoint, coarse object placement, or required detailed 3D information like bounding boxes or multi-view images. Compass Control's strength lies in:

*   **Explicit Object-Centric Control:** It offers explicit control over the 3D orientation of *individual* objects within a scene, a feature not readily available in existing methods.
*   **Lightweight and Flexible Conditioning:** The approach doesn't require dense 3D information or multi-view data. The use of compass tokens and a lightweight encoder makes it efficient and easy to integrate into existing T2I pipelines.
*   **Attention Localization:** The CALL mechanism effectively mitigates the issue of diffused attention, a critical problem when introducing new tokens or conditioning signals in T2I models. This is a valuable technical contribution.
*   **Personalization and Generalization:** Demonstrating control over personalized object orientation within diverse scene scenarios significantly amplifies the value and practical applicability of the method.
*   **Multi-Object Scene:** Although other papers do personalization, few control multiple objects in the scene with complex orientations.

**Significance:**

The significance of the work stems from its potential to enhance the creative control and precision offered by T2I models. Specific strengths include:

*   **Improved User Interface:**  The method replaces iterative prompting with a direct, controllable input mechanism for orientation, streamlining the creative process.
*   **Enhanced Scene Composition:** Enables the creation of complex and visually coherent scenes with specific object arrangements, expanding the artistic possibilities offered by T2I.
*   **Practical Applications:** Object orientation control has implications for various applications, including scene understanding, design, and content creation.
*   **Generalization across objects:** This paper generalizes over multiple objects in a single scene with precise orientation whereas other methods provide camera control only.

**Weaknesses and Limitations:**

*   **Synthetic Training Data:** Training on synthetic data introduces a domain gap that can potentially limit performance on real-world images. Although ControlNet augmentations help, this issue remains. The paper acknowledges this by showcasing results that control orientations in personalized images.
*   **Occlusion and Overlap:** The method's limitations with occluded or overlapping objects represent a practical constraint. As acknowledged, these problems are typical for any bounding box based technique.
*   **Simplistic Orientation Representation:** A single-angle rotation around the up-axis, while practical for many objects, is insufficient for modeling the full 3D pose of complex, non-rigid objects like humans.
*   **Dependency on Object Detectors:** The method depends on a detector such as GroundingDINO to be accurate which can create some challenges.

**Justification for Score:**

The paper contributes substantially to the T2I generation literature.  While the method has limitations, the proposed Compass Control is a novel, effective, and flexible solution for object-centric orientation control. The CALL mechanism is a valuable technical contribution, and the results demonstrate significant generalization capabilities. This method unlocks new levels of precision and artistic control in T2I generation.

*It is recognized that reliance on synthetic data represents a trade-off for precise control and the ability to train a lightweight, generalizable model.* A potential refinement would entail exploring techniques for unsupervised or self-supervised training to mitigate the synthetic data gap, with appropriate incorporation of regularization strategies to promote enhanced robustness.

**Score: 8.5**

- **Score**: 8/10

### **[A Meaningful Perturbation Metric for Evaluating Explainability Methods](http://arxiv.org/abs/2504.06800v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper introduces a novel metric for evaluating the faithfulness and plausibility of explainability methods for Deep Neural Networks (DNNs).  It addresses the limitations of existing perturbation-based evaluation methods, which often produce out-of-distribution (OOD) modifications to input images, leading to unreliable results. The core idea is to use Stable Diffusion (SD) inpainting to selectively perturb high-relevance pixels of an image, guiding the modification towards the class with the second-highest classification score.  By controlling the perturbation and ensuring that it results in plausible alternatives, the metric aims to better reflect the actual reasoning process of the DNN. The paper demonstrates through extensive experiments that the proposed metric generates meaningful rankings across various models and attribution methods, and aligns better with human preferences compared to existing metrics.  A weighting function is also introduced to ensure the inpainting directly relates to changing predictions.

**Critical Evaluation:**

* **Novelty:** The use of generative models like Stable Diffusion for evaluating explainability methods is a noteworthy innovation.  Existing methods often rely on simplistic perturbations (blurring, deletion) that are not realistic and can trigger unintended OOD behavior. The idea of inpainting towards a specific, plausible alternative target (top-2 class) is also a significant improvement.  The weighting function also introduces a novel aspect by comparing the relevance maps before and after inpainting which validates the perturbation procedure itself. The combination of these ideas leads to improved correlation with human preferences for explanations.

* **Significance:** Explainability is a crucial factor in building trust and adoption of DNNs. Improving the evaluation of explainability methods directly contributes to this goal. If widely adopted, this new metric could become a standard tool for comparing and refining attribution techniques. The ability to produce more realistic perturbations is not just about better rankings but also about generating insights into DNN behavior under conditions closer to natural data distributions. Furthermore, its superior correlation with human preference means that a more practically useful standard metric has been introduced.

* **Strengths:**
    * **Addresses a clear problem:** The limitations of existing perturbation-based metrics are well-defined and illustrated through examples.
    * **Technically sound:** The use of Stable Diffusion inpainting is well-motivated and implemented.
    * **Extensive experiments:** The paper presents a comprehensive experimental evaluation across various models, datasets, and attribution methods.
    * **Human evaluation:** The user study provides strong evidence that the proposed metric aligns better with human intuition.
    * **Novel weighting function:** A new approach is provided to analyze the perturbations caused by the inpainting by weighting the changes in the original and modified relevance maps.

* **Weaknesses:**
    * **Computational cost:** While the paper claims to reduce the computational burden compared to fully inpainting-based attribution, inpainting with Stable Diffusion is still relatively expensive, which might limit the scalability of the metric. The claim that our metric "reduces this overhead to just 10 iterations" doesn't make sense and requires more context.
    * **Dependency on Stable Diffusion:** The metric's effectiveness is directly tied to the capabilities and biases of the Stable Diffusion model. If the SD model struggles with certain classes or exhibits undesirable artifacts, it could affect the evaluation results.
    * **Limited to top-2 class:** The approach relies on inpainting the image towards the second highest predicted class. While plausible, this limits the evaluation to specific scenarios and might not be generalizable to more complex explanations.
   *  The reliance of pixel-based explanations is questionable because high-level representations in DNNs often don't correspond to individual pixels.

* **Potential Influence:** The paper has the potential to significantly influence the field of explainable AI by providing a more robust and human-aligned metric for evaluating attribution methods.  It could drive the development of new explainability techniques that are not only faithful to the model but also intuitively understandable to humans. By being more computationally expensive than previous metrics, there is risk the proposed method will not be adopted as a standard metric, or, for that matter, not be used very frequently.

**Justification for Score:**

The paper introduces a genuinely novel and significant improvement over existing evaluation metrics for explainability methods. The use of generative models to produce more realistic perturbations is a substantial step forward, and the user study provides compelling evidence that the metric aligns better with human perception. While the computational cost and dependency on Stable Diffusion are limitations, the benefits outweigh these drawbacks. Given the potential for this metric to become a standard tool in the field and drive the development of more interpretable DNNs, I believe a score of 8 is justified.

**Score: 8**

- **Score**: 8/10

### **[MovSAM: A Single-image Moving Object Segmentation Framework Based on Deep Thinking](http://arxiv.org/abs/2504.06863v1)**
- **Summary**: Okay, I will provide a concise summary and a rigorous critical evaluation, including a justified novelty/significance score, for the provided paper.

**Summary:**

The paper introduces MovSAM, a novel framework for segmenting moving objects from single images.  Unlike existing methods that rely on multi-frame image sequences and temporal information, MovSAM leverages a Multimodal Large Language Model (MLLM) enhanced with Chain-of-Thought (CoT) prompting for deep scene understanding and reasoning about motion. The framework fuses information from a pre-trained Segment Anything Model (SAM) and a Vision-Language Model (VLM) and employs a deep thinking refinement loop to iteratively improve segmentation results by considering scene context and inter-object relationships. The approach is designed to work in scenarios where multi-frame methods fail (e.g., camera failures, pedestrian gaming). Experimental results on various moving object segmentation (MOS) benchmarks demonstrate state-of-the-art performance, even surpassing some multi-frame methods despite using only single images. The authors also present results showing the effectiveness of the approach in real-world driving scenarios.

**Rigorous and Critical Evaluation:**

*   **Novelty:**  The core novelty lies in the *integration* of existing components (MLLM, SAM, VLM) into a deep thinking framework specifically tailored for *single-image* moving object segmentation.  While the individual components themselves are not new, the way they are combined and orchestrated, along with the iterative refinement loop using CoT prompting, is a genuinely novel contribution. Specifically the deep thinking framework addresses limitations of simply combining large models (i.e. model hallucinations or flawed reasoning). The real-world application in autonomous driving and robot perception, is also promising. The ability to segment from a single image to deal with sensor failure or incomplete data provides practical value.
*   **Significance:**  The significance stems from addressing a critical gap in the field.  Most MOS methods require multi-frame data, which is not always available or reliable. The performance of MovSAM, achieving state-of-the-art results with *only* single images and even exceeding the performance of methods that require multi-frame, makes this a significant contribution. It opens up new possibilities for applications like motion intention prediction and handling camera frame drops in autonomous driving and robotics. The demonstrated ability to handle occlusions and motion blur through reasoning is also very valuable. However, the paper focuses on a rather specific application. The impact might be narrower than general image segmentation.

*   **Strengths:**

    *   **Clear problem definition:** The paper clearly identifies the limitations of existing multi-frame MOS methods.
    *   **Well-defined approach:**  The MovSAM framework is clearly explained, with detailed descriptions of each component and the iterative refinement process.
    *   **Strong experimental results:** The quantitative results on multiple datasets demonstrate state-of-the-art performance.
    *   **Real-world validation:** The real-world driving scenarios provide compelling evidence of the practical applicability of MovSAM.
    *   **Ablation Studies:**  The ablation studies systematically evaluate the contributions of different components of the framework.
*   **Weaknesses:**

    *   **Reliance on Large Models:**  The approach relies heavily on the performance of pre-trained large models (MLLM, SAM, VLM). While this is becoming increasingly common, it also makes the framework computationally expensive and potentially difficult to deploy in resource-constrained environments.
    *   **Limited generalization analysis:** Further study is needed to evaluate the generalizability of the framework to significantly different types of scenes, especially those outside of the common autonomous driving and robotics contexts.
    *   **Inference Time:** The paper admits a longer inference time (0.3 seconds). This is a point of concern, particularly for real-time applications.
    *   **Lack of theoretical guarantee:** The iterative refinement process in the deep thinking loop relies on convergence. Although experiments prove the results, there is a lack of a theoretical guarantee that the MovSAM can converge within a certain time.

*   **Potential Influence:**  MovSAM has the potential to influence the field by:

    *   Inspiring new research on single-image MOS methods.
    *   Encouraging the development of more efficient and robust deep learning models for scene understanding and motion reasoning.
    *   Enabling new applications of MOS in autonomous driving, robotics, and other areas where multi-frame data is limited or unreliable.

**Overall Assessment:**

Despite the dependence on large pre-trained models and some limitations in generalizability analysis and runtime, MovSAM represents a significant advancement in moving object segmentation. The novel combination of MLLM, SAM, and VLM, along with the deep thinking refinement loop, enables state-of-the-art performance on single images, addressing a critical gap in the field. The real-world validation and thorough ablation studies further strengthen the paper's claims.

**Score: 8**

**Rationale:**  The paper demonstrates a clear, novel contribution with strong empirical results. The state-of-the-art performance on single-image MOS benchmarks is very impressive. However, the dependence on large models, runtime considerations, and the limited analysis of very diverse scenarios prevent the paper from achieving a higher score. While there are weaknesses, the strengths significantly outweigh them, making this a strong contribution to the field. The real-world applications is a significant component which drives up the value, even though there are performance considerations which lower it.

- **Score**: 8/10

### **[MedSegFactory: Text-Guided Generation of Medical Image-Mask Pairs](http://arxiv.org/abs/2504.06897v1)**
- **Summary**: Okay, I will provide a summary of the paper, a critical evaluation of its novelty and significance, and a justified score.

**Summary:**

The paper "MedSegFactory: Text-Guided Generation of Medical Image-Mask Pairs" introduces a novel framework for synthesizing paired medical images and segmentation masks directly from user-defined text prompts. The core innovation is a dual-stream diffusion model, where one stream generates the medical image and the other generates the corresponding segmentation mask. A key component is the Joint Cross-Attention (JCA) mechanism, which enables bidirectional conditioning between the image and mask streams to ensure precise alignment and semantic consistency. The authors demonstrate the effectiveness of MedSegFactory through extensive experiments, showing its ability to generate high-quality, diverse, and consistent image-mask pairs that can be used to improve the performance of downstream medical image segmentation tasks. The method addresses data scarcity and regulatory constraints in medical imaging by providing an accessible way to create synthetic training data.

**Critical Evaluation:**

**Novelty:**

The paper presents a novel approach to medical image synthesis by directly generating paired images and masks from text prompts. This is a significant advancement over existing methods that rely on unconditional generation, mask-conditional generation, or require manual annotation. The introduction of the Joint Cross-Attention mechanism is also a novel contribution that enables more precise control over the relationship between the generated image and mask.

**Significance:**

The potential impact of this work is considerable. It addresses a major bottleneck in medical image analysis: the scarcity of labeled data. By allowing users to generate realistic and accurately segmented medical images from text prompts, MedSegFactory could significantly reduce the cost and time associated with training deep learning models for various medical imaging tasks (e.g., segmentation, detection, classification). It circumvents ethical and regulatory restrictions that prevent data sharing.  This has the potential to be a game-changer in areas where data annotation is particularly challenging, enabling researchers and clinicians to develop and deploy AI-powered tools more easily. The demonstrated improvement in downstream segmentation tasks further highlights its practical value.

**Strengths:**

*   **Novel Approach:** The text-to-image-mask generation approach using a dual-stream diffusion model is novel and effective.
*   **Joint Cross-Attention (JCA):** The JCA mechanism is a key innovation that addresses the crucial challenge of ensuring semantic consistency and alignment between generated images and masks. The ablation study confirms its importance.
*   **Comprehensive Experiments:** The paper includes thorough experimental validation, comparing MedSegFactory against strong baselines on a variety of medical image datasets and segmentation tasks. Qualitative and quantitative results demonstrate the superiority of the proposed method.
*   **Addresses a Key Challenge:** The work directly addresses the significant issue of data scarcity and high annotation costs in medical imaging.
*   **Practical Applications:** The generated image-mask pairs can be readily used to augment existing datasets and improve the performance of downstream segmentation models.

**Weaknesses:**

*   **Limited 3D Capabilities:** The current framework is designed primarily for 2D medical images. While 2D slices can augment training, fully extending the model to 3D volumetric data would be beneficial.
*   **Target-Free Generation Challenges:** The paper acknowledges limitations in generating images without specific segmentation targets, leading to blurred boundaries. This is a valid concern and points to an area for future research.
*   **Dependency on Text Prompts:** The quality of the generated images and masks is dependent on the quality of the text prompts. While the paper demonstrates that the model works well with relatively simple prompts, more complex or ambiguous prompts might lead to less satisfactory results.
*   **Computational Cost:** Diffusion models, in general, are computationally intensive to train. This might limit the accessibility of MedSegFactory to researchers with limited computational resources, although inference is less demanding.

**Justification of Score:**

MedSegFactory represents a significant advance in medical image synthesis. It provides a versatile and accessible way to generate realistic and accurately segmented medical images from text prompts, addressing a crucial problem in the field. The proposed JCA mechanism is a key innovation, and the extensive experimental results support the effectiveness of the approach. While there are some limitations, such as the limited 3D capabilities and the potential dependency on the quality of text prompts, the overall contribution is substantial. MedSegFactory has the potential to significantly impact medical image analysis by reducing the cost and time associated with data annotation and enabling the development of more robust and accurate AI-powered tools.

Score: 8.5

- **Score**: 8/10

### **[DeCoMa: Detecting and Purifying Code Dataset Watermarks through Dual Channel Code Abstraction](http://arxiv.org/abs/2504.07002v1)**
- **Summary**: Here's a summary and critical evaluation of the DeCoMa paper:

**Summary:**

The paper "DeCoMa: Detecting and Purifying Code Dataset Watermarks through Dual Channel Code Abstraction" presents a novel technique to detect and remove stealthy watermarks embedded in code datasets.  The approach, named DeCoMa, leverages a dual-channel code abstraction method that considers both the natural language and formal (executable logic) aspects of code.  It generalizes code samples into standardized templates, identifies outlier associations representing potential watermarks, and then purifies the dataset by removing samples containing the detected watermarks. The paper evaluates DeCoMa against various code watermarking techniques and tasks, demonstrating its effectiveness in detecting and removing watermarks with minimal impact on model performance.

**Critical Evaluation:**

*   **Novelty:**  The paper's primary novelty lies in its dual-channel code abstraction approach specifically designed to attack code watermarks.  While previous works have explored watermarking and backdoor detection in code, DeCoMa is the first to explicitly leverage the dual nature of code (natural and formal channels) for watermark detection and purification. This dual-channel approach helps to overcome challenges created by semantic-preserving transformations and flexible watermark embedding strategies used in code watermarks. It improves upon previous backdoor and watermark detection methods by effectively identifying hidden outlier associations, which existing techniques have difficulty distinguishing due to their minimal impact on the model. The idea of abstracting the code using tree-sitter to identify these anomalies, is novel.

*   **Significance:** The paper addresses a critical and growing concern: protecting valuable code datasets from unauthorized use.  The increasing reliance on large code datasets for training neural code models makes them attractive targets for misappropriation. DeCoMa provides a practical and effective means to detect and remove watermarks, safeguarding the intellectual property of dataset creators. Experimental results demonstrate effectiveness on a variety of code datasets, code completion, code summarization, and code search tasks, with detection recall of 100%. Further, it attacks code watermarks with embedding rates as low as 0.1%, while maintaining performance after purifying the dataset. Given the cost of LLM models to remove watermarks and obfuscate data, the faster time (31x-130x) to remove watermarks makes this method more effective.

*   **Strengths:**
    *   **Clear Problem Definition:**  The paper clearly articulates the problem of code dataset misappropriation and the limitations of existing watermark detection techniques in the code domain.
    *   **Sound Methodology:**  The dual-channel code abstraction approach is well-motivated and rigorously explained.
    *   **Extensive Evaluation:** The experimental evaluation is comprehensive, covering various watermarking techniques, code intelligence tasks, and parameter settings.
    *   **Significant Performance Improvement:** DeCoMa significantly outperforms existing baseline methods in watermark detection effectiveness and efficiency.
    *   **Practicality:** It requires no model training and significantly less time compared to other methods, making it a viable solution.

*   **Weaknesses:**

    *   **Dependency on Bare Datasets:** The approach relies on a small bare dataset to accurately identify watermarks. While the authors demonstrate robustness to different distributions of bare datasets, it still introduces a potential limitation.
    *   **Abstraction limitations:** The abstraction may not cover all possible obfuscation techniques and may require future research into other abstractions.
    *   **Watermark strength limitations:** The outlier model makes it stronger at attacking specific watermarks, but it may miss more complex or nuanced signatures.

*   **Potential Influence:** This paper has the potential to significantly influence the field of code dataset protection. DeCoMa offers a practical and effective solution, providing a baseline for future research and development in code watermarking techniques.

Score: 8

**Rationale:**

A score of 8 reflects the paper's significant contributions to a relevant and challenging problem. DeCoMa introduces a novel approach, is thoroughly evaluated, and achieves substantial improvements over existing methods. While there are limitations related to the dependence on clean data and model parameters, the strengths outweigh the weaknesses, making this a valuable contribution to the field. The code dataset protection space has a high barrier to entry, given the intellectual property rights issues, so the DeCoMa method and the source code shared within the paper are particularly valuable to the community.
- **Score**: 8/10

### **[A Unified Agentic Framework for Evaluating Conditional Image Generation](http://arxiv.org/abs/2504.07046v1)**
- **Summary**: Here's a summary and critical evaluation of the provided paper:

**Summary:**

The paper introduces CIGEVAL, a unified agentic framework for evaluating conditional image generation tasks.  CIGEVAL leverages large multimodal models (LMMs) as its core, incorporates a multi-functional toolbox (Grounding, Difference, Highlight, Scene Graph), and establishes a fine-grained evaluation framework.  The framework decomposes evaluation into sub-tasks, selects appropriate tools, and analyzes tool outputs.  The authors also synthesize evaluation trajectories for fine-tuning smaller LMMs (7B models), enabling them to autonomously perform nuanced analyses.  Experiments across seven conditional image generation tasks demonstrate that CIGEVAL (using GPT-40) achieves a high correlation with human assessments.  Furthermore, fine-tuned open-source 7B LMMs within CIGEVAL outperform the previous GPT-40-based state-of-the-art method.  Case studies highlight CIGEVAL's capability in identifying subtle issues related to subject consistency and adherence to control guidance.

**Critical Evaluation:**

*   **Novelty:** The paper presents a novel approach by framing image evaluation as an agentic task.  Integrating a toolbox *with* an LMM, rather than solely relying on the LMM's perceptual abilities, is a crucial element of the novelty.  The fine-grained evaluation framework and the trajectory synthesis for fine-tuning smaller models further contribute to the innovation.  The key is that CIGEVAL isn't just using GPT-4 to score images; it's designing a system that uses tools and a process to *arrive* at that score. This agentic approach is a significant step beyond simple LMM-based scoring.

*   **Significance:** The significance lies in addressing the challenges of developing task-agnostic, reliable, and explainable evaluation metrics for conditional image generation.  Current metrics often fall short in terms of explainability, human alignment, and the ability to generalize across diverse tasks. CIGEVAL's higher correlation with human ratings (compared to existing metrics like VIEScore) and its ability to surpass prior GPT-4 based methods by using fine-tuned open-source models signifies a meaningful advancement. By being able to make use of lower powered models after some fine-tuning, CIGEVAL brings this evaluation strategy to the wider research community which is significant. The case studies further support the claim that CIGEVAL can identify subtle flaws, suggesting a move towards human-level reliability.

*   **Strengths:**

    *   **Agentic Framework:**  The agentic framework is a well-defined and promising approach.
    *   **Toolbox Integration:**  The inclusion of image analysis and editing tools enhances the LMM's capabilities and overcomes some of its limitations.
    *   **Fine-grained Evaluation:**  The task decomposition and sub-question approach enable more nuanced analyses.
    *   **Trajectory Synthesis and Fine-tuning:** This allows for making smaller models more performant.
    *   **Strong Experimental Results:** The experiments across multiple tasks and with different LMMs demonstrate the effectiveness and robustness of CIGEVAL.
    *  **Comprehensive Benchmarking:**  The evaluation on ImagenHub provides a solid foundation.

*   **Weaknesses:**

    *   **Closed-source dependency (GPT-4):** While the authors address this by fine-tuning open-source models, the initial trajectory generation relies on GPT-4. The quality of those synthetic examples may be hard to replicate with open-source alternatives, and thus may be a hard ceiling to break.
    *   **Scope:**  The evaluation focuses on semantic consistency and leaves perceptual quality for future research. A more holistic evaluation encompassing both aspects would be ideal.
    *   **Dataset limited:** Only ImagenHub is used. The training data and experiments being limited to this is a concern, and may be too focused on just those types of images and flaws.
    *  **Black-box still:** Even with the additional tools, the actual internal reasoning of the model may still be a black-box beyond the tooling itself, making it still hard to understand *why* the model reached a certain conclusion.

*   **Potential Influence:**  CIGEVAL has the potential to significantly impact the field of conditional image generation by providing a more reliable, explainable, and unified evaluation method.  The agentic framework could inspire new approaches to automated evaluation in other areas of AI.  The accessibility afforded by the finetuned open source models is particularly important.

**Justification for Score:**

The paper introduces a genuinely novel approach to image generation evaluation, backed by strong experimental results. The agentic approach combined with tool integration and a clear trajectory for fine-tuning smaller models addresses many of the limitations of existing evaluation metrics. While the initial dependence on GPT-4 and the limited scope of the evaluation are valid concerns, the strengths of the paper far outweigh its weaknesses. It represents a clear advancement in the field and offers a compelling direction for future research. The ability to fine-tune smaller models and achieve better-than-GPT4 results makes this important work.

Score: 8

- **Score**: 8/10

## Other Papers
### **[Diffusion Based Ambiguous Image Segmentation](http://arxiv.org/abs/2504.05977v1)**
### **[An Empirical Study of GPT-4o Image Generation Capabilities](http://arxiv.org/abs/2504.05979v1)**
### **[NativQA Framework: Enabling LLMs with Native, Local, and Everyday Knowledge](http://arxiv.org/abs/2504.05995v1)**
### **[Optuna vs Code Llama: Are LLMs a New Paradigm for Hyperparameter Tuning?](http://arxiv.org/abs/2504.06006v1)**
### **[Llama-3-Nanda-10B-Chat: An Open Generative Large Language Model for Hindi](http://arxiv.org/abs/2504.06011v1)**
### **[CamContextI2V: Context-aware Controllable Video Generation](http://arxiv.org/abs/2504.06022v1)**
### **[OSDM-MReg: Multimodal Image Registration based One Step Diffusion Model](http://arxiv.org/abs/2504.06027v1)**
### **[Multi-Sense Embeddings for Language Models and Knowledge Distillation](http://arxiv.org/abs/2504.06036v1)**
### **[Explainable AI for building energy retrofitting under data scarcity](http://arxiv.org/abs/2504.06055v1)**
### **[Knowledge Graph Completion with Relation-Aware Anchor Enhancement](http://arxiv.org/abs/2504.06129v1)**
### **[QGen Studio: An Adaptive Question-Answer Generation, Training and Evaluation Platform](http://arxiv.org/abs/2504.06136v1)**
### **[A Training-Free Style-aligned Image Generation with Scale-wise Autoregressive Model](http://arxiv.org/abs/2504.06144v1)**
### **[V-MAGE: A Game Evaluation Framework for Assessing Visual-Centric Capabilities in Multimodal Large Language Models](http://arxiv.org/abs/2504.06148v1)**
### **[Navigating the Rabbit Hole: Emergent Biases in LLM-Generated Attack Narratives Targeting Mental Health Groups](http://arxiv.org/abs/2504.06160v2)**
### **[Assessing how hyperparameters impact Large Language Models' sarcasm detection performance](http://arxiv.org/abs/2504.06166v1)**
### **[TxGemma: Efficient and Agentic LLMs for Therapeutics](http://arxiv.org/abs/2504.06196v1)**
### **[From 128K to 4M: Efficient Training of Ultra-Long Context Large Language Models](http://arxiv.org/abs/2504.06214v1)**
### **[Encoder-Decoder Gemma: Improving the Quality-Efficiency Trade-Off via Adaptation](http://arxiv.org/abs/2504.06225v1)**
### **[LExT: Towards Evaluating Trustworthiness of Natural Language Explanations](http://arxiv.org/abs/2504.06227v1)**
### **[HiFlow: Training-free High-Resolution Image Generation with Flow-Aligned Guidance](http://arxiv.org/abs/2504.06232v1)**
### **[Transfer between Modalities with MetaQueries](http://arxiv.org/abs/2504.06256v1)**
### **[FEABench: Evaluating Language Models on Multiphysics Reasoning Ability](http://arxiv.org/abs/2504.06260v1)**
### **[Hogwild! Inference: Parallel LLM Generation via Concurrent Attention](http://arxiv.org/abs/2504.06261v2)**
### **[A Geometric-Aware Perspective and Beyond: Hybrid Quantum-Classical Machine Learning Methods](http://arxiv.org/abs/2504.06328v1)**
### **[Multihead self-attention in cortico-thalamic circuits](http://arxiv.org/abs/2504.06354v1)**
### **[Query Understanding in LLM-based Conversational Information Seeking](http://arxiv.org/abs/2504.06356v1)**
### **[Unifying Autoregressive and Diffusion-Based Sequence Generation](http://arxiv.org/abs/2504.06416v1)**
### **[Releasing Differentially Private Event Logs Using Generative Models](http://arxiv.org/abs/2504.06418v1)**
### **[S'MoRE: Structural Mixture of Residual Experts for LLM Fine-tuning](http://arxiv.org/abs/2504.06426v1)**
### **[D-Feat Occlusions: Diffusion Features for Robustness to Partial Visual Occlusions in Object Recognition](http://arxiv.org/abs/2504.06432v1)**
### **[Human Trust in AI Search: A Large-Scale Experiment](http://arxiv.org/abs/2504.06435v1)**
### **[Language-Dependent Political Bias in AI: A Study of ChatGPT and Gemini](http://arxiv.org/abs/2504.06436v1)**
### **[Don't Let It Hallucinate: Premise Verification via Retrieval-Augmented Logical Reasoning](http://arxiv.org/abs/2504.06438v1)**
### **[Can you Finetune your Binoculars? Embedding Text Watermarks into the Weights of Large Language Models](http://arxiv.org/abs/2504.06446v1)**
### **[Can LLMs Simulate Personas with Reversed Performance? A Benchmark for Counterfactual Instruction Following](http://arxiv.org/abs/2504.06460v1)**
### **[Mind the Gap: Evaluating Vision Systems in Small Data Applications](http://arxiv.org/abs/2504.06486v1)**
### **[Towards Holistic Prompt Craft](http://arxiv.org/abs/2504.06496v1)**
### **[Lugha-Llama: Adapting Large Language Models for African Languages](http://arxiv.org/abs/2504.06536v1)**
### **[DiffusionCom: Structure-Aware Multimodal Diffusion Model for Multimodal Knowledge Graph Completion](http://arxiv.org/abs/2504.06543v1)**
### **[NeedleInATable: Exploring Long-Context Capability of Large Language Models towards Long-Structured Tables](http://arxiv.org/abs/2504.06560v1)**
### **[Diffusion Factor Models: Generating High-Dimensional Returns with Factor Structure](http://arxiv.org/abs/2504.06566v1)**
### **[Defending LLM Watermarking Against Spoofing Attacks with Contrastive Representation Learning](http://arxiv.org/abs/2504.06575v1)**
### **[Bypassing Safety Guardrails in LLMs Using Humor](http://arxiv.org/abs/2504.06577v1)**
### **[Right Prediction, Wrong Reasoning: Uncovering LLM Misalignment in RA Disease Diagnosis](http://arxiv.org/abs/2504.06581v1)**
### **[Automated Business Process Analysis: An LLM-Based Approach to Value Assessment](http://arxiv.org/abs/2504.06600v1)**
### **[Benchmarking Multimodal CoT Reward Model Stepwise by Visual Program](http://arxiv.org/abs/2504.06606v1)**
### **[Rethinking LayerNorm in Image Restoration Transformers](http://arxiv.org/abs/2504.06629v1)**
### **[The Method for Storing Patterns in Neural Networks-Memorization and Recall of QR code Patterns-](http://arxiv.org/abs/2504.06631v1)**
### **[PosterMaker: Towards High-Quality Product Poster Generation with Accurate Text Rendering](http://arxiv.org/abs/2504.06632v1)**
### **[BBQRec: Behavior-Bind Quantization for Multi-Modal Sequential Recommendation](http://arxiv.org/abs/2504.06636v1)**
### **[SCI-Reason: A Dataset with Chain-of-Thought Rationales for Complex Multimodal Reasoning in Academic Areas](http://arxiv.org/abs/2504.06637v1)**
### **[ThoughtProbe: Classifier-Guided Thought Space Exploration Leveraging LLM Intrinsic Reasoning](http://arxiv.org/abs/2504.06650v1)**
### **[A Neuro-inspired Interpretation of Unlearning in Large Language Models through Sample-level Unlearning Difficulty](http://arxiv.org/abs/2504.06658v1)**
### **[Bridging the Gap Between Preference Alignment and Machine Unlearning](http://arxiv.org/abs/2504.06659v1)**
### **[SEE: Continual Fine-tuning with Sequential Ensemble of Experts](http://arxiv.org/abs/2504.06664v1)**
### **[Patch Matters: Training-free Fine-grained Image Caption Enhancement via Local Perception](http://arxiv.org/abs/2504.06666v1)**
### **[RAGME: Retrieval Augmented Video Generation for Enhanced Motion Realism](http://arxiv.org/abs/2504.06672v1)**
### **[Probability Density Geodesics in Image Diffusion Latent Space](http://arxiv.org/abs/2504.06675v1)**
### **[CAT: Circular-Convolutional Attention for Sub-Quadratic Transformers](http://arxiv.org/abs/2504.06704v1)**
### **[EDIT: Enhancing Vision Transformers by Mitigating Attention Sink through an Encoder-Decoder Architecture](http://arxiv.org/abs/2504.06738v1)**
### **[Compass Control: Multi Object Orientation Control for Text-to-Image Generation](http://arxiv.org/abs/2504.06752v1)**
### **[FamilyTool: A Multi-hop Personalized Tool Use Benchmark](http://arxiv.org/abs/2504.06766v1)**
### **[DIMA: DIffusing Motion Artifacts for unsupervised correction in brain MRI images](http://arxiv.org/abs/2504.06767v1)**
### **[CHIME: A Compressive Framework for Holistic Interest Modeling](http://arxiv.org/abs/2504.06780v1)**
### **[Zero-Shot Image-Based Large Language Model Approach to Road Pavement Monitoring](http://arxiv.org/abs/2504.06785v1)**
### **[A Meaningful Perturbation Metric for Evaluating Explainability Methods](http://arxiv.org/abs/2504.06800v1)**
### **[DyDiT++: Dynamic Diffusion Transformers for Efficient Visual Generation](http://arxiv.org/abs/2504.06803v1)**
### **[Open Problems and a Hypothetical Path Forward in LLM Knowledge Paradigms](http://arxiv.org/abs/2504.06823v1)**
### **[LVC: A Lightweight Compression Framework for Enhancing VLMs in Long Video Understanding](http://arxiv.org/abs/2504.06835v1)**
### **[Integrating Cognitive Processing Signals into Language Models: A Review of Advances, Applications and Future Directions](http://arxiv.org/abs/2504.06843v1)**
### **[CasTex: Cascaded Text-to-Texture Synthesis via Explicit Texture Maps and Physically-Based Shading](http://arxiv.org/abs/2504.06856v1)**
### **[EIDT-V: Exploiting Intersections in Diffusion Trajectories for Model-Agnostic, Zero-Shot, Training-Free Text-to-Video Generation](http://arxiv.org/abs/2504.06861v1)**
### **[MovSAM: A Single-image Moving Object Segmentation Framework Based on Deep Thinking](http://arxiv.org/abs/2504.06863v1)**
### **[MedSegFactory: Text-Guided Generation of Medical Image-Mask Pairs](http://arxiv.org/abs/2504.06897v1)**
### **[Data Augmentation for Fake Reviews Detection in Multiple Languages and Multiple Domains](http://arxiv.org/abs/2504.06917v1)**
### **[FeedbackEval: A Benchmark for Evaluating Large Language Models in Feedback-Driven Code Repair Tasks](http://arxiv.org/abs/2504.06939v1)**
### **[Review of Case-Based Reasoning for LLM Agents: Theoretical Foundations, Architectural Components, and Cognitive Integration](http://arxiv.org/abs/2504.06943v1)**
### **[RuOpinionNE-2024: Extraction of Opinion Tuples from Russian News Texts](http://arxiv.org/abs/2504.06947v1)**
### **[PathSegDiff: Pathology Segmentation using Diffusion model representations](http://arxiv.org/abs/2504.06950v1)**
### **[VideoChat-R1: Enhancing Spatio-Temporal Perception via Reinforcement Fine-Tuning](http://arxiv.org/abs/2504.06958v1)**
### **[Towards LLMs Robustness to Changes in Prompt Format Styles](http://arxiv.org/abs/2504.06969v1)**
### **[DeCoMa: Detecting and Purifying Code Dataset Watermarks through Dual Channel Code Abstraction](http://arxiv.org/abs/2504.07002v1)**
### **[Latent Diffusion U-Net Representations Contain Positional Embeddings and Anomalies](http://arxiv.org/abs/2504.07008v1)**
### **[LLM-IFT: LLM-Powered Information Flow Tracking for Secure Hardware](http://arxiv.org/abs/2504.07015v1)**
### **[Evaluating Retrieval Augmented Generative Models for Document Queries in Transportation Safety](http://arxiv.org/abs/2504.07022v1)**
### **[A Unified Agentic Framework for Evaluating Conditional Image Generation](http://arxiv.org/abs/2504.07046v1)**
### **[To Backtrack or Not to Backtrack: When Sequential Search Limits Model Reasoning](http://arxiv.org/abs/2504.07052v1)**
### **[TASTE: Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling](http://arxiv.org/abs/2504.07053v1)**
### **[A Survey on Personalized and Pluralistic Preference Alignment in Large Language Models](http://arxiv.org/abs/2504.07070v1)**
### **[DeduCE: Deductive Consistency as a Framework to Evaluate LLM Reasoning](http://arxiv.org/abs/2504.07080v1)**
### **[KG-LLM-Bench: A Scalable Benchmark for Evaluating LLM Reasoning on Textualized Knowledge Graphs](http://arxiv.org/abs/2504.07087v1)**
### **[OmniCaptioner: One Captioner to Rule Them All](http://arxiv.org/abs/2504.07089v1)**
### **[Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning](http://arxiv.org/abs/2504.07097v1)**
