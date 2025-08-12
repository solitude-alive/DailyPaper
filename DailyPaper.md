# The Latest Daily Papers - Date: 2025-08-12
## Highlight Papers
### **[DiTVR: Zero-Shot Diffusion Transformer for Video Restoration](http://arxiv.org/abs/2508.07811v1)**
- **Summary**: Here's a summary and critical evaluation of the DiTVR paper:

**Summary**

The paper introduces DiTVR, a novel zero-shot video restoration framework designed to improve temporal consistency and detail preservation in tasks like super-resolution, denoising, and deblurring. DiTVR combines a Diffusion Transformer (DiT) architecture with three key components: a spatiotemporal neighbor selection cache (STNC) to efficiently manage tokens, trajectory-aware attention to align features along motion paths, and a flow-guided diffusion sampler to integrate optical flow guidance during the reverse diffusion process. By targeting enhancements to layers most sensitive to temporal dynamics, DiTVR aims to overcome the limitations of U-Net based methods and naive flow-warping approaches. The paper demonstrates state-of-the-art performance on video restoration benchmarks, showing superior temporal consistency and detail preservation while remaining robust to flow noise and occlusions.

**Critical Evaluation**

*   **Novelty:** The paper's novelty lies in the combination of several techniques to address the challenges of zero-shot video restoration. While each component might have precursors in the literature, their integration within a DiT framework, specifically designed for zero-shot video restoration, is a significant contribution. The layer-wise analysis to identify 'vital layers' and then focusing computational resources on those layers is a good engineering approach. The trajectory aware attention that focuses on motion alignment is also a strong contribution. The flow-guided diffusion sampler is another significant component that contributes to the overall improved performance.

*   **Significance:** The paper addresses a critical problem in video restoration: maintaining temporal consistency without task-specific fine-tuning.  The zero-shot capability makes it applicable to a broader range of real-world scenarios.  The performance gains demonstrated on standard benchmarks are compelling. The clear articulation of the limitations of existing methods, particularly U-Net-based approaches and the naive application of optical flow, strengthens the paper's significance. Given that Video Restoration can be used in a wide variety of applications and the limitations of existing methods, this method tackles the challenges effectively.

*   **Strengths:**
    *   Strong empirical results: The paper provides comprehensive experimental results on multiple datasets.
    *   Clear problem definition: The paper clearly defines the problem and limitations of existing methods.
    *   Well-motivated approach: The proposed method is well-motivated, with clear explanations of each component and their contributions.
    *   Ablation Study: The ablation studies isolate the impact of each component.

*   **Weaknesses:**
    *   While the DiT architecture is a strong improvement over the traditional U-Net architecture, there is limited discussion on the computational complexity.
    *   The paper focuses on performance metrics and visual quality but could benefit from a more detailed analysis of failure cases. What types of videos or degradations does the method still struggle with?

*   **Potential Impact:** DiTVR has the potential to influence future research in video restoration. It may lead to the development of more efficient and robust zero-shot methods that can handle a wide range of degradations. The trajectory-aware attention and flow-guided sampling techniques could be adopted in other video processing tasks.

*   **Justification for Score:** While the paper builds upon existing research, the novel integration of techniques within a DiT framework, coupled with the strong empirical results, makes a significant contribution to the field of zero-shot video restoration. The approach addresses limitations of previous methods, providing a more robust and effective solution. Given the limited discussion on computational complexity and focus of the study on synthetic benchmarks, the score will not be a 9 or 10.

Score: 7

- **Score**: 9/10

### **[Progressive Bird's Eye View Perception for Safety-Critical Autonomous Driving: A Comprehensive Survey](http://arxiv.org/abs/2508.07560v1)**
- **Summary**: Here's a summary and critical evaluation of the provided research paper:

**Summary:**

The paper presents a comprehensive survey of Bird's-Eye-View (BEV) perception techniques for safety-critical autonomous driving. It categorizes existing approaches into three progressive stages: single-modality vehicle-side perception, multi-modality vehicle-side perception, and multi-agent collaborative perception. The survey analyzes state-of-the-art frameworks, implementation strategies, and public datasets relevant to BEV perception. It identifies key challenges including open-world deployment, handling large-scale unlabeled data, managing sensor degradation, and minimizing inter-agent communication latency. Finally, it outlines future research directions, such as integration with end-to-end autonomous driving systems, embodied intelligence, and large language models. An open-source repository is included with methods and benchmark datasets.

**Critical Evaluation:**

*   **Novelty:**  The paper's primary contribution lies in its comprehensive and systematic categorization of BEV perception techniques from a *safety-critical perspective.* While other surveys exist, they often focus on specific modalities (e.g., camera-only) or collaboration paradigms.  This survey is the first to explicitly frame the analysis within a safety and robustness context, which is a very important and timely perspective. It offers a structured view of the evolution of BEV perception, highlighting how different techniques address specific safety challenges. The open-source repository adds practical value for the community.

*   **Significance:** BEV perception is undeniably a foundational element in modern autonomous driving. Enhancing safety and robustness of such systems is a crucial goal. A thorough understanding of existing methods, their limitations, and the challenges hindering real-world deployment is paramount. The paper provides a strong baseline for researchers and practitioners, facilitating a more focused and informed exploration of solutions. The coverage of multi-agent collaborative perception, and its framing in V2X context is particularly valuable, highlighting a key area for future development. The analysis of available datasets and the provision of benchmarks will enable more consistent and comparable evaluations, which is beneficial to the whole field.

*   **Strengths:**

    *   **Comprehensive Coverage:** The survey covers a wide range of BEV perception techniques, encompassing single-modality, multi-modality, and collaborative approaches.
    *   **Clear Categorization:** The progressive stage-based categorization (SafeBEV 1.0, 2.0, 3.0) offers a structured and easily understandable view of the field.
    *   **Safety-Critical Focus:** Framing the discussion within a safety and robustness context is relevant and timely.
    *   **Dataset Analysis:** The in-depth review of public datasets, their limitations, and support for safety validation is helpful.
    *   **Open-Source Contribution:** The provision of an open-source repository enhances accessibility and reproducibility.

*   **Weaknesses:**

    *   **Limited In-Depth Technical Analysis:** While the survey provides a comprehensive overview, it could benefit from a more in-depth technical analysis of the algorithms.  For instance, more mathematical explanations of the methods. However this is understandable, since this is a survey paper.
    *   **Rapidly Evolving Field:** The field of BEV perception is constantly evolving, so it runs the risk of becoming outdated relatively quickly.  The authors should clearly indicate the cut-off date for the literature review.
    *   **Limited Focus on Computational Efficiency:**  While safety is the focus, a stronger discussion of the computational cost and real-time feasibility of different approaches could be valuable, particularly for safety-critical applications. The emphasis on E2E learning could be increased, since this area is heavily related to reducing latency.

*   **Potential Influence:** The paper is likely to be highly influential due to its clear categorization, safety-critical perspective, and comprehensive dataset analysis. It will serve as a valuable resource for researchers and engineers working on autonomous driving and related fields. The open-source repository will further enhance its impact by facilitating the adoption and comparison of existing techniques.

**Justification of Score:**

The paper presents a significant and timely contribution to the field of autonomous driving by providing a comprehensive and well-structured survey of BEV perception techniques from a safety-critical perspective. While some weaknesses exist, the strengths of the paper, particularly its clear categorization, dataset analysis, and open-source repository, outweigh these limitations. The authors address the most urgent topics related to BEV, namely, safety and robustness. The categorization provides a clear overview and allows for easy searching of methods.

Score: 8

- **Score**: 8/10

### **[Beyond Single: A Data Selection Principle for LLM Alignment via Fine-Grained Preference Signals](http://arxiv.org/abs/2508.07638v1)**
- **Summary**: Okay, I will provide a concise summary and a rigorous, critical evaluation of the paper based on the provided text.

**Summary:**

The paper addresses the challenge of aligning Large Language Models (LLMs) using fine-grained, aspect-specific preference data. Existing methods like DPO struggle with noise and conflicts inherent in aggregated fine-grained preference datasets. The authors propose a data-centric approach, deriving the Direct Multi-Preference Optimization (DMPO) objective and identifying a Preference Divergence (PD) term to quantify inter-aspect preference conflicts.  They then formulate a theoretically grounded data selection principle based on this PD, advocating for selecting high-consensus data (identified by negative PD values) for DPO training. They offer practical methods for PD estimation and length bias mitigation.  Empirical results on the UltraFeedback dataset demonstrate improved performance compared to holistic preference baselines and even oracle baselines, while also boosting training efficiency. The core idea is to filter out conflicting and low-value data and train with high-quality, high-consensus examples.

**Critical Evaluation:**

*   **Novelty:** The paper's primary novelty lies in its data-centric perspective for LLM alignment with fine-grained preferences.  While fine-grained preference alignment is explored in previous works, the explicit formulation of DMPO objective, identification of PD for conflict quantification, and utilization of PD as a data selection criterion is a novel and insightful combination. The theoretical justification for using the most negative PD terms for data selection also contributes to the novelty. The approach of creating a smaller proxy reward model and applying cross pseudo rewards to facilitate calculation adds to the novelty.

*   **Significance:** The paper addresses a crucial practical problem: how to effectively use readily available, but noisy and conflicting, fine-grained preference data for aligning LLMs.  The potential impact is significant, as this approach could lead to more robust, efficient, and scalable LLM alignment, particularly in scenarios where obtaining high-quality holistic preference data is challenging or cost-prohibitive. The empirical results demonstrating improvements over strong baselines and oracle performance, combined with improved training efficiency, strongly support the paper's claims. By using a selection budget, authors significantly accelerate training and curation of data.

*   **Strengths:**
    *   **Theoretical Foundation:** The paper provides a solid theoretical grounding for its data selection principle, deriving the DMPO objective and proving loss bounds in the selection problem.
    *   **Practical Approach:** The authors bridge the gap between theory and practice by introducing methods for PD term estimation and length bias mitigation, making their approach readily applicable.
    *   **Empirical Validation:** Extensive experiments on a widely used dataset (UltraFeedback) with controlled conflict levels provide compelling evidence for the effectiveness of their method.
    *   **Clear Problem Statement and Solution:** The paper clearly identifies a problem, thoroughly analyzes the current solution and proposes a working framework for LLM alignment with multiple sub-preferences.
    *   **Excellent Results:** The simple yet effective strategy achieves significant improvement against both the standard holistic preference and a stronger oracle baseline, all while boosting training efficiency and obviating the need for intractable holistic preference annotating.
    *   **Rigorous and insightful Analysis:** Author's analysis of preference signals through the proposed theoretical framework is insightful and allows for significant performance improvement.

*   **Weaknesses:**
    *   **Dependency on Proxy Model:** While the proxy model for PD term estimation addresses practical challenges, it introduces another potential source of error.  The reliance on the smaller model could be better discussed or analyzed to understand its effect on accuracy and selection.
    *   **Limited Generalizability:** The experiments are conducted on a specific dataset (UltraFeedback) with a fixed set of fine-grained preference criteria. More experiments and datasets would be needed to assess generalizability to other LLM alignment scenarios.
    *   **Scope:** The study focuses on the data selection component. It could benefit from a discussion of how to integrate the PD selection principle with alternative optimization algorithms beyond DPO.
    *   **Limited datasets:** The approach is constrained by the limited availability of public feedback datasets that offer multiple fine-grained preferences.

*   **Potential Influence:** The paper has the potential to influence research in LLM alignment by shifting focus towards data-centric approaches and leveraging fine-grained preferences more effectively. The proposed DMPO derivation and the identification and use of the Preference Divergence (PD) metric can be adopted to quantify inter-aspect preference conflicts for other datasets. The core insight of selecting based on high-consensus data points might be adopted in other domains.

**Justification for Score:**

Considering the above points, I believe a score of **8** is appropriate. The paper exhibits considerable novelty and potential significance, addressing a relevant problem with a theoretically sound and empirically validated approach.  The improvements over strong baselines are compelling. However, the reliance on a proxy model and the limited generalizability due to dataset specificity prevent it from achieving a higher score. Future work should address these limitations and explore broader applications of the proposed principles.

**Score: 8**

- **Score**: 8/10

### **[LoSemB: Logic-Guided Semantic Bridging for Inductive Tool Retrieval](http://arxiv.org/abs/2508.07690v1)**
- **Summary**: Here's a summary and critical evaluation of the LoSemB paper:

**Summary:**

The paper addresses the problem of inductive tool retrieval for Large Language Models (LLMs). Traditional tool retrieval methods, which involve selecting relevant tools from a repository for LLMs to use, often struggle when faced with "unseen" tools (i.e., tools not present during the model's training). The paper identifies two key challenges: (1) a large distribution shift between seen and unseen tools and (2) the vulnerability of similarity-based retrieval methods which are sensitive to the quality of representations and can lead to errors when tools have similar descriptions but different functionalities.

To overcome these issues, the authors propose LoSemB (Logic-Guided Semantic Bridging), a novel framework that leverages logical information extracted from a knowledge graph of tools and instructions. LoSemB employs a logic-based embedding alignment module to mitigate the distribution shift and a relational augmented retrieval mechanism to improve retrieval accuracy.  Experimental results demonstrate that LoSemB performs significantly better than existing methods in inductive settings while maintaining competitive performance in transductive settings.

**Critical Evaluation:**

*   **Novelty:** The key novelty of this paper lies in its focus on *inductive* tool retrieval and its utilization of logical information derived from tool co-occurrence and instruction-tool interaction patterns to address the challenges of unseen tools. Most prior work assumes a transductive setting. The idea of drawing inspiration from human cognitive processes (leveraging prior experience to understand new tools) is well-motivated. The architecture, particularly the logic-based embedding alignment, is a novel contribution.
*   **Significance:** The paper tackles a practically important problem. As LLMs are increasingly used in real-world scenarios, their ability to adapt to dynamically changing tool repositories is crucial. Addressing the limitations of current methods in handling unseen tools is a significant step toward making tool-augmented LLMs more robust and versatile. By making it easier to add new tools to LLMs, the paper has the potential to make tool-augmented LLMs more versatile and adaptable. The performance improvements shown in the inductive setting are substantial, demonstrating the effectiveness of the proposed approach.
*   **Strengths:**
    *   Clearly defines and motivates the problem of inductive tool retrieval.
    *   Identifies and analyzes the limitations of existing retrieval methods in the context of unseen tools.
    *   Proposes a novel and well-designed framework (LoSemB) with two key components.
    *   Provides thorough experimental results demonstrating the effectiveness of LoSemB in both inductive and transductive settings.
    *   Includes ablation studies to analyze the contribution of each component of LoSemB.
    *   Provides a case study to illustrate the benefits of LoSemB in practical scenarios.
*   **Weaknesses:**
    *   The reliance on constructing a logical graph might be a limitation, as it requires curating tool-instruction interactions. While the paper shows good results on ToolBench, the performance might depend on the quality and completeness of the graph data. The robustness of the approach to noisy or incomplete logical graphs could be discussed further.

*   **Potential Influence:** The paper has the potential to influence the direction of research in tool learning and LLM augmentation. By highlighting the importance of inductive tool retrieval and demonstrating a practical solution, it could encourage researchers to focus on developing more robust and adaptable tool retrieval methods. The proposed architecture, particularly the logic-based embedding alignment, could serve as a basis for future research in this area. The concepts may be generalized to other retrieval-augmented generation (RAG) tasks.

*   **Justification for Score:** Given the novelty of the problem formulation (inductive tool retrieval), the well-motivated and well-engineered solution, and the substantial experimental results demonstrating improved performance, I assign a score of 8. The strengths of the paper outweigh the weaknesses, and it offers a significant contribution to the field. While the reliance on a logical graph is a potential limitation, it is addressed through the proposed architecture and empirical results.

**Score: 8**

- **Score**: 8/10

### **[Generative Video Matting](http://arxiv.org/abs/2508.07905v1)**
- **Summary**: **Summary of "Generative Video Matting":** The paper addresses the challenges in video matting caused by limited availability of high-quality ground-truth data. Traditional datasets provide imperfect annotations, which hinder the generalization of video matting methods in real-world scenarios. The authors propose two main strategies: first, they underline the importance of large-scale pre-training with diverse synthetic and pseudo-labeled datasets. They introduce a scalable synthetic data generation pipeline that yields about 200 video clips featuring diverse human bodies and fine-grained hairs for fine-tuning. Second, they present a novel video matting approach that leverages pre-trained video diffusion models to enhance the inter-frame consistency and mitigate domain gaps. Their model offers enhanced temporal consistency and demonstrates superior performance in comprehensive evaluations against three benchmark datasets, illustrating strong generalization in various real-world contexts. The accompanying code is available for further exploration. **Critical Evaluation:** The paper exhibits significant novelty by addressing a prevalent issue in video matting: the reliance on imperfect ground-truth data, which limits the applicability of many existing methods. The dual approach—enhancing pre-training with synthetic data and utilizing diffusion models for temporal consistency—offers a progressive solution that diverges from conventional frame-by-frame processing. The introduction of a scalable synthetic data generation pipeline is particularly noteworthy, as it demonstrates a thorough understanding of the importance of diverse training data. Strengths: 1. **Innovative Approach**: The integration of generative modeling with video matting represents a meaningful advancement in the field. 2. **Temporal Consistency**: Addressing temporal coherence within video processing is a critical enhancement over earlier methods that often ignore temporal dynamics. 3. **Comprehensive Evaluation**: The empirical results showcase a rigorous assessment across benchmarks, which bolsters the claims of improved performance. Weaknesses: 1. **Synthetic Data Limitations**: While synthetic data generation is a strength, it can still pose challenges in terms of the realism and variability of scenarios compared to real-world data. 2. **Generalization Evidence**: Although the paper claims improved generalization, the real-world applicability may need further testing across a broader array of diverse datasets that reflect natural variations and occlusions. 3. **Computationally Intensive**: The reliance on large-scale pre-training and sophisticated models may require substantial computational resources, which may not be accessible to all researchers in the field. Overall, the paper contributes a noteworthy solution to a well-defined problem within video matting. Its innovative methodology and strong quantitative backing suggest it will encourage further research and development in video processing techniques. Therefore, the score awarded reflects the paper's substantial contribution and potential influence on future studies in the domain. **Score: 8**
- **Score**: 8/10

### **[Large Language Models for Subjective Language Understanding: A Survey](http://arxiv.org/abs/2508.07959v1)**
- **Summary**: Okay, I will provide a concise summary and critical evaluation of the provided paper abstract and partial OCR text.

**Summary:**

This paper presents a survey on the application of Large Language Models (LLMs) to Subjective Language Understanding (SLU) tasks. It defines SLU as encompassing various tasks related to interpreting or generating content that conveys personal feelings, opinions, or figurative meanings, such as sentiment analysis, emotion recognition, sarcasm detection, humor understanding, stance detection, metaphor recognition, intent detection, and aesthetics identification. The survey reviews recent advancements in using LLMs for these tasks, examining datasets, methods, and challenges. It emphasizes the significance of LLMs' capabilities in contextual understanding and reasoning for these nuanced tasks, while also acknowledging their limitations, such as potential biases and difficulties in handling complex linguistic phenomena. It also discusses challenges and potential directions for future research in this intersection of LLMs and SLU.

**Critical Evaluation:**

*   **Novelty:** The paper's main novelty lies in providing a comprehensive overview of the rapidly evolving field of applying LLMs to SLU. While individual tasks like sentiment analysis and sarcasm detection have been studied extensively, this survey appears to be the *first* to synthesize recent advancements across the *entire* spectrum of subjective language tasks within the context of the LLM era. The unified framework proposed to cover these tasks is important. The critical assessment of how effective LLMs are in tasks previously considered challenging is valuable.

*   **Significance:** The survey is significant for several reasons:

    *   **Consolidation of Knowledge:** It synthesizes a large body of recent research (over 200 papers) into a coherent overview, making it easier for researchers and practitioners to grasp the current state of the field.
    *   **Identification of Trends and Challenges:** By examining the commonalities and differences across tasks, the survey identifies key trends (e.g., prompt engineering, multi-task learning) and challenges (e.g., bias, data limitations, handling context).
    *   **Guidance for Future Research:** The survey explicitly outlines open issues and suggests directions for future research, which can help to focus and accelerate progress in the field. The discussion of how to develop standard benchmarks to span the spectrum of subjective tasks is crucial.
    *   **Impact on Affective Computing and NLP:** The discussion of LLMs’ capabilities and limitations for tasks involving nuances in languages can help accelerate improvements in a wide variety of applications.
*   **Strengths:**

    *   **Broad Scope:** The survey covers a wide range of subjective language tasks, including topics that are often not covered in other surveys (e.g., metaphor recognition, aesthetics identification).
    *   **Emphasis on Understanding:** The survey focuses on *understanding* subjective language rather than just generating it, providing valuable insights into the cognitive aspects of these tasks.
    *   **Comprehensive Review:** The survey provides detailed descriptions of datasets, methods, and challenges for each task.
*   **Weaknesses:**

    *   **Timeliness:** Given the rapid pace of advancements in LLMs, the survey might become outdated relatively quickly. Regular updates or extensions would be necessary to maintain its relevance.  However, it is difficult to avoid this in a quickly evolving field.
    *   **Depth vs. Breadth:** While the survey covers a wide range of tasks, the depth of analysis for each task might be limited due to space constraints. More in-depth comparisons of specific methods or datasets could be beneficial. The full survey will contain more depth.

**Justification for Score:**

I assign a score of **8** to this paper. The paper provides a timely and valuable synthesis of a rapidly evolving field. The unified treatment of subjective language tasks within the LLM era, the identification of key trends and challenges, and the guidance for future research make it a significant contribution. While the paper might be limited by its timeliness and depth (likely unavoidable in a survey), its broad scope, emphasis on understanding, and comprehensive review of the literature outweigh these limitations.

Score: 8

- **Score**: 8/10

### **[Learned Regularization for Microwave Tomography](http://arxiv.org/abs/2508.08114v1)**
- **Summary**: Here's a summary and critical evaluation of the paper "LEARNED REGULARIZATION FOR MICROWAVE TOMOGRAPHY":

**Summary:**

This paper proposes a novel hybrid framework called Single-Step Diffusion Regularization (SSD-Reg) for microwave tomography (MWT). The framework integrates a physics-informed approach with learned priors from diffusion models to improve the accuracy, stability, and robustness of image reconstruction.  SSD-Reg combines a Fréchet-differentiable forward model for stable optimization with a lightweight plug-and-play (PnP) diffusion-based regularization to guide solutions towards anatomically plausible outcomes without paired training data. The method is compared against several existing reconstruction techniques on both synthetic and real-world datasets, demonstrating its superior performance in terms of reconstruction quality, robustness to noise, and computational efficiency. The method uses only a single diffusion step, reducing the computational burden of traditional diffusion model integrations.

**Critical Evaluation:**

* **Novelty:** The novelty of this work lies in the specific way it integrates diffusion models into the MWT reconstruction process. While using diffusion models for inverse problems is not entirely new, applying a single-step diffusion regularizer within a physics-informed iterative optimization framework, specifically using a Fréchet-differentiable forward model for MWT, presents a unique approach.  The key is that this avoids needing paired training data while achieving better reconstruction quality than other unsupervised methods. The "single-step" nature of the diffusion regularization is also novel, providing a computationally efficient way to inject learned priors.
* **Significance:**  The significance of this work stems from its potential to address key challenges in MWT: nonlinearity and ill-posedness. MWT has promising clinical applications (e.g., breast cancer screening), but these challenges limit its widespread adoption. SSD-Reg offers a practical solution by combining the strengths of physical modeling (stability and accuracy) with the generative power of diffusion models (anatomical plausibility and fine-grained structural detail). The improved robustness to noise is also significant, making it more suitable for real-world applications. It successfully uses unsupervised learning in a setting often requiring supervised learning, which is critical because generating the necessary data for supervised MWT is difficult and costly.

**Strengths:**

* **Strong Performance:**  The experimental results demonstrate that SSD-Reg consistently outperforms existing methods across various datasets and noise levels. The quantitative metrics (SSIM, PSNR, LPIPS) and visual results support this claim.
* **Computational Efficiency:**  The single-step diffusion regularization significantly reduces the computational cost compared to other diffusion-based methods, making it more practical for real-world applications. The use of a differentiable forward model enables efficient gradient-based optimization.
* **Robustness:** The method exhibits excellent robustness to noise and high contrast, surpassing other methods and thus making this method suitable for many types of scenarios. The inclusion of random flips is also a smart technique.
* **Unsupervised Learning:**  The method leverages unsupervised learning, avoiding the need for large paired training datasets. This is particularly important in MWT, where acquiring such datasets is difficult.
* **Well-Written and Comprehensive:** The paper is well-written, clearly explaining the methodology and providing comprehensive experimental results.

**Weaknesses:**

* **Limited Task-Specific Prior Incorporation:** The paper acknowledges that the method currently provides primarily geometric regularization without strong task-specific anatomical priors. While it improves reconstruction quality, incorporating anatomical information from other modalities or prior knowledge could further enhance the results. The paper notes this is a future research direction.
* **Computational Complexity Trade-offs:** The paper argues that SSD is computationally efficient, which, relative to other diffusion-based reconstruction methods, is accurate. Relative to the simplest iterative methods, however, it will likely be more computationally expensive. The key is that the single-step diffusion is significantly cheaper than the full sampling process.
* **Black Box Nature:** Diffusion models are inherently black boxes. While this work tries to inject more of the physics through the differentiable model, it remains that there is limited interpretability of the single-step denoising process.

**Potential Influence:**

This paper has the potential to significantly influence the field of MWT reconstruction by providing a practical and effective framework for addressing the challenges of nonlinearity and ill-posedness. The combination of physics-informed modeling with learned priors offers a promising direction for future research. Furthermore, the framework is modular, allowing for further improvements and extensions, such as incorporating anatomical information and extending it to 3D imaging. The demonstrated success of unsupervised diffusion models in this context could spur further research into their application to other inverse problems in medical imaging.

**Score:** 8

**Justification:**

I assign a score of 8 because the paper presents a novel and significant contribution to the field of MWT reconstruction. The SSD-Reg framework effectively addresses key challenges in MWT by combining physics-informed modeling with learned priors from diffusion models. While there is room for improvement in terms of incorporating more task-specific prior information and full interpretability, the method demonstrates strong performance, robustness, and computational efficiency, making it a potentially valuable tool for real-world MWT applications. The demonstration of successful unsupervised learning in this setting also sets it apart from other approaches. The relatively small score reduction stems from some of the limitations as discussed such as the lack of task-specific priors.

- **Score**: 8/10

### **[FantasyStyle: Controllable Stylized Distillation for 3D Gaussian Splatting](http://arxiv.org/abs/2508.08136v1)**
- **Summary**: Here's a summary and critical evaluation of the paper:

**Summary:**

The paper "FantasyStyle: Controllable Stylized Distillation for 3D Gaussian Splatting" presents a novel approach to style transfer for 3D Gaussian Splatting (3DGS) scenes. It addresses two key challenges in existing methods: multi-view inconsistency and content leakage from style images.  The proposed method, FantasyStyle, relies entirely on diffusion model distillation, incorporating two core components: (1) Multi-View Frequency Consistency (MVFC) to enhance cross-view consistency by selectively reducing low-frequency components in the latent space, and (2) Controllable Stylized Distillation (CSD), which uses negative guidance to suppress content leakage from the style image and remove reconstruction term from the objective.  The authors demonstrate through qualitative and quantitative experiments that FantasyStyle outperforms existing state-of-the-art methods in terms of style transfer quality, content preservation, and visual realism. The method is the first to rely solely on diffusion model distillation in the 3DGS style transfer domain.

**Critical Evaluation:**

*   **Novelty:** The paper introduces several novel elements. The application of frequency-domain analysis and filtering to the latent space of a diffusion model for multi-view consistency is a well-justified and effective approach.  The controllable stylized distillation, particularly the use of negative guidance and the removal of the reconstruction term, is also innovative and directly addresses a specific problem in style transfer. The claim of being the *first* 3DGS style transfer method to rely solely on diffusion distillation is also significant.

*   **Significance:** The work addresses a crucial gap in 3DGS style transfer.  Existing methods often suffer from artifacts, inconsistencies, and content leakage, limiting their practical applications. FantasyStyle's improvements in these areas make it a valuable contribution to the field. The performance improvements in visual quality and faithfulness to the source style are convincing and the method bridges the gap from 2D diffusion based style transfer to 3D scenes. By introducing the MVFC and CSD framework, the authors provide a new avenue for research in 3D style transfer.

*   **Strengths:**
    *   **Clear problem definition:** The paper clearly articulates the challenges of multi-view inconsistency and content leakage.
    *   **Well-motivated approach:** The proposed solutions are logically derived from an analysis of the problem.
    *   **Comprehensive experiments:** The qualitative and quantitative evaluations demonstrate the effectiveness of the proposed method. The comparison with existing methods is thorough.
    *   **Strong results:** The visual results presented clearly show the superior performance of FantasyStyle.
    *   **Clarity and Presentation:** The paper is well-written and easy to understand, with clear explanations of the methods and experimental setup.

*   **Weaknesses:**
    *   **Computational Cost:** The method is acknowledged to be computationally expensive. Further work could explore strategies for improving efficiency. While employing smaller diffusion models is suggested, the actual impact on performance in terms of quality and computational cost needs to be empirically validated.
    *   **Limited Style Control:** While the negative guidance improves content preservation, the degree of user control over specific style attributes could be further enhanced. A way of intuitively controlling the intensity of certain style transfer aspects could be beneficial.
    *   **Domain Specificity:** The work focuses primarily on 3DGS. While the core concepts may be transferable to other 3D representations, this is not explicitly addressed.

*   **Potential Impact:** The paper has the potential to significantly impact the field of 3D style transfer.  It offers a more robust and visually appealing approach than existing methods and provides a foundation for future research in diffusion-based techniques for 3D content generation and editing. The flexible extension of 2D techniques to 3D domains is a clear advantage.

**Justification for the Score:**

The paper represents a significant advancement in 3DGS style transfer. The proposed method effectively addresses key limitations of existing approaches. While the computational cost and limited style control are drawbacks, the strong results and novel approach justify a high score. The clear presentation and potential impact further contribute to its value.

**Score: 8**

- **Score**: 8/10

### **[From Natural Language to Solver-Ready Power System Optimization: An LLM-Assisted, Validation-in-the-Loop Framework](http://arxiv.org/abs/2508.08147v1)**
- **Summary**: Here's a summary and critical evaluation of the provided research paper:

**Summary:**

The paper introduces a novel approach to power system optimization by leveraging Large Language Models (LLMs) to translate natural language descriptions of optimization scenarios into solver-ready mathematical formulations. Instead of directly using LLMs to generate solutions (which the authors argue is unreliable), they propose an LLM-assisted agent that converts natural language inputs into compact, solver-compatible mixed-integer programs (MIPs). The framework incorporates domain-aware prompting, schema validation, iterative repair mechanisms, and integrates with off-the-shelf optimization solvers like Gurobi. The authors demonstrate the effectiveness of their method using the unit commitment (UC) problem as a case study, showing that the agent produces optimal or near-optimal schedules. They further explore solution enhancement techniques such as GNN-guided branching and LLM-based cut separator configuration to reduce solve times.

**Critical Evaluation:**

*   **Novelty:** The core novelty lies in the architecture of the LLM-assisted agent specifically designed for translating natural language into formally verifiable and solvable optimization models. While LLMs have been used in power systems and optimization before, the specific approach of using them primarily for formulation *generation* followed by rigorous validation and repair represents a significant departure from methods that directly produce solutions or focus solely on acceleration via ML surrogates. The combination of LLMs for high-level translation, schema validation for ensuring mathematical consistency, and standard solvers for numerical rigor addresses a key limitation of pure LLM-based approaches. The integration of GNN-guided branching adds a layer of efficiency without sacrificing optimality guarantees, demonstrating a well-rounded contribution. The modular approach of solution enhancement is a significant improvement over LLM techniques alone.

*   **Significance:** The significance of this work stems from its potential to bridge the gap between high-level operational policies and executable mathematical models. This is particularly important in the power system domain, where translating evolving policies into accurate and solver-compatible models is often a bottleneck. By automating this process, the proposed framework can enable faster iteration, improve accessibility for non-experts, and enhance the overall decision-making process. Moreover, the validation-in-the-loop approach addresses the inherent limitations of LLMs in numerical precision and constraint handling, paving the way for more reliable and trustworthy AI applications in critical infrastructure.

*   **Strengths:**

    *   Clear and well-defined problem statement.
    *   Novel architecture combining LLMs with traditional optimization solvers and a validation layer.
    *   Rigorous validation and repair mechanisms to ensure solution feasibility.
    *   Demonstration of effectiveness on the unit commitment problem.
    *   Exploration of solution enhancement techniques (GNN branching, LLM-based cuts) for improved efficiency.
    *   Clear experimental setup and results.
    *   Well-written and organized paper.

*   **Weaknesses:**

    *   The case study focuses primarily on the unit commitment problem. While UC is a relevant and representative problem, expanding the evaluation to other power system optimization problems (e.g., economic dispatch, optimal power flow) would further strengthen the claims of generalizability.
    *   While the prompts are shown, a more detailed description of the engineering and thought process behind them could benefit the reader.
    *   While the paper highlights the benefits of using commercial solvers, a more detailed comparison against other open-source solvers could be included.

*   **Potential Influence:**

    The paper has the potential to influence the field by promoting a more principled and reliable approach to integrating AI in power system optimization. The focus on formulation generation, validation, and iterative repair can serve as a blueprint for developing other AI-assisted tools for translating high-level descriptions into executable models in various domains. The GNN-guided branching and LLM-based cuts further showcase a way for classical solvers to be augmented with LLM strategies.

**Score:** 8

**Rationale:**

The paper presents a novel and significant contribution to the field of power system optimization. The integration of LLMs for formulation generation, coupled with rigorous validation and iterative repair, addresses a critical need for bridging the gap between natural language policies and solver-ready models. The results on the unit commitment problem are promising, and the exploration of solution enhancement techniques further enhances the value of the work. The paper is well-written, organized, and clearly demonstrates the effectiveness of the proposed approach.

However, the score is not a 9 or 10 due to the limited scope of the case study and the potential for further improvements in first-pass formulation accuracy and efficiency. Expanding the evaluation to other power system optimization problems and providing a more detailed description of the prompts would further strengthen the claims of generalizability and impact. While valuable, the described enhancements are not a perfect replacement for domain expertise.

- **Score**: 8/10

### **[SAEMark: Multi-bit LLM Watermarking with Inference-Time Scaling](http://arxiv.org/abs/2508.08211v1)**
- **Summary**: Here's a summary and evaluation of the paper:

**Summary:**

The paper introduces SAEMARK, a novel framework for multi-bit watermarking of LLM-generated text. Unlike existing methods that often require white-box access to the model, manipulate logits directly, or degrade text quality, SAEMARK operates post-hoc, embedding personalized messages by selecting from multiple LLM-generated candidate outputs.  It leverages a deterministic feature extractor (specifically, sparse autoencoders - SAEs) to compute a scalar statistic for each candidate unit (e.g., sentence), normalizing the statistic and comparing it to a target value derived from a watermark key.  The candidate closest to the target value is selected, ensuring the embedded message. The framework is designed to be model-agnostic (working with API access only), language-agnostic, domain-agnostic, and comes with theoretical guarantees concerning watermark success probability and computational budget.  Experiments across multiple datasets demonstrate high detection accuracy and text quality, exceeding existing single-bit and multi-bit watermarking techniques.

**Critical Evaluation:**

*   **Novelty:** The paper's core contribution lies in its approach to watermarking through *selection* rather than *modification* of the LLM output. This post-hoc, inference-time selection significantly sidesteps major limitations of existing watermarking methods, specifically:

    *   **Black-box compatibility:**  It avoids requiring direct access to model logits, making it usable with API-based LLMs.
    *   **Text quality preservation:** Since it selects from natural LLM outputs, text quality degradation is minimized.
    *   **Multi-bit capability:** The use of deterministic feature extraction and target-guided selection enables encoding of multiple bits.
    *   **The use of SAE's as a feature representation:** SAEMARK doesn't necessarily *introduce* the SAE itself, but *effectively* uses SAE features to represent the distribution of patterns from LLM generation and *selects* the most relevant sentence given the target distribution pattern from the key.
*   **Significance:** Watermarking is a crucial aspect of responsible AI development and deployment. The ability to attribute AI-generated content is essential for preventing misinformation, copyright infringement, and content laundering. SAEMARK's advantages make it potentially more practical and scalable than existing techniques.  It directly addresses a critical need for robust attribution in a world increasingly saturated with LLM-generated content. The general framework is further enabled through a detailed statistical analysis and feature representations derived from SAE's, all available for use with an API to an LLM.

*   **Strengths:**

    *   **Strong Theoretical Foundations:** The paper provides clear theoretical guarantees linking success probability to computational resources. The theoretical arguments are persuasive and adds to the robustness of the approach.
    *   **Extensive Empirical Validation:** Experiments across diverse datasets (English text, Chinese text, Python code, instruction following) bolster claims of robustness and generalizability. The comparisons against existing methods are convincing.
    *   **Clear Problem Definition and Solution:**  The paper clearly defines the problem of multi-bit watermarking with API access and presents a well-articulated solution with SAEMARK.
    *   **Addressing limitations with proposed implementations:** SAEMARK also addresses problems with practical applications. For example, the CheckAlignment algorithm's eliminate spurious statistical matches that would otherwise compromise detection accuracy. Second, background feature masking ensures FCS calculations focus on discriminative semantic patterns rather than ubiquitous surface features.

*   **Weaknesses:**

    *   **Dependence on Feature Extractor Quality:** The reliance on the quality of the feature extractor (SAE in this case) is a potential limitation.  While the authors demonstrate good performance with SAEs, the performance of SAEMARK could degrade with a less effective feature representation. The paper does address this, stating "we only apply SAEs on the Anchor LLM and require only access to the output texts from the base LLM."
    *   **Potential for Adversarial Attacks:** While the paper does address adversarial attacks such as paraphrase attacks and other types of attacks. Additional work focusing on adversarial robustness, specifically targeting the feature selection and target generation aspects, would be beneficial.
    *   **Reliance on Multiple Candidates**: The performance and quality will degrade based on the number of candidates *N*. This also introduces a computational element since many LLM services will charge per token. The authors have also addressed this practical concern in their design choices by "backing out" unnecessary candidates.

*   **Overall Score Rationale:** The paper presents a novel and significant contribution to the field of LLM watermarking. SAEMARK's black-box compatibility, text quality preservation, multi-bit capability, and theoretical guarantees make it a potentially impactful solution for content attribution. While the dependence on feature extractor quality and potential for adversarial attacks are concerns, the strengths of the approach outweigh these limitations. Given the importance of watermarking and the promise of SAEMARK, I am assigning a high score.

Score: 8

- **Score**: 8/10

### **[LL3M: Large Language 3D Modelers](http://arxiv.org/abs/2508.08228v1)**
- **Summary**: Here's a summary and critical evaluation of the LL3M paper:

**Summary:**

The paper introduces LL3M, a novel multi-agent system leveraging large language models (LLMs) to generate 3D assets in Blender by writing interpretable Python code.  Instead of relying on training data derived directly from 3D models, LL3M treats 3D shape generation as a code-writing task.  The system coordinates specialized LLM agents for planning, retrieving relevant code snippets, writing, debugging, and refining the Blender scripts. A key aspect is the generation of human-readable, well-documented code utilizing Blender's sophisticated constructs (B-meshes, modifiers, shaders).  This facilitates co-creative loops with both automatic self-critiquing and user-driven refinements.  The use of a retrieval-augmented generation (RAG) knowledge base, BlenderRAG, is critical, providing agents with examples and documentation to ensure code correctness and enables advanced modeling operations. Experiments demonstrate LL3M's effectiveness across diverse shapes, materials, and styles.

**Critical Evaluation:**

*   **Novelty:** The core novelty lies in the shift from representation-centric 3D generative modeling to a code-centric approach. This is a significant departure from methods that learn directly from 3D data or try to predict mesh geometry directly.  The multi-agent architecture is also a valuable contribution, breaking down the complex task of 3D modeling into manageable subtasks handled by specialized LLMs. The integration of BlenderRAG is a key enabler for generating complex, executable code, and represents a significant advancement over prior methods which often produced fragile or incomplete code. The iterative refinement loop with visual feedback and user interaction is also a strong point, enabling a co-creative modeling experience.
    Previous works exist using LLMs for generating specific parts of Blender scenes or materials. LL3M's ability to generate the entire 3D object in a cohesive and controllable fashion, guided by interpretable code, is novel.

*   **Significance/Impact:** The paper has potential to significantly impact the field by making 3D content creation more accessible and editable. The interpretable code output allows artists to easily modify and customize the generated assets, overcoming the limitations of "black box" generative models. The iterative refinement and co-creation capabilities provide a powerful tool for artists and designers. The ability to generate diverse assets without category-specific training data further enhances its practical value. This could lead to more efficient workflows and democratize 3D asset creation. Furthermore, the successful demonstration of code as a generative medium for 3D content could inspire new research directions in both generative modeling and LLM applications.

*   **Strengths:**

    *   The code-centric approach fosters editability and interpretability.
    *   The multi-agent architecture is well-designed and addresses the complexity of 3D modeling.
    *   The BlenderRAG integration significantly enhances code quality and functionality.
    *   The iterative refinement loop enables co-creation and precise control.
    *   Comprehensive experiments demonstrate versatility and quality across various asset categories.
    *   The project provides access to a Blender code-writing system with a code-edit option, enhancing transparency and control.
*   **Weaknesses:**

    *   The reliance on external VLMs for visual feedback introduces potential dependencies and limitations tied to the performance of those models. The accuracy of visual feedback affects the auto-refinement process.
    *   Computational cost: While user edits are fast, the initial asset generation can be time-consuming.
    *   Hallucinations: The paper admits there is limited correction on the LLM's hallucinations, so there are potential issues.

*   **Justification for Score:**

LL3M presents a compelling and innovative approach to 3D asset generation by focusing on code as a generative medium. The shift away from purely data-driven methods provides substantial benefits in terms of editability, interpretability, and control.  The multi-agent architecture and BlenderRAG integration are crucial components that contribute to the system's robustness and effectiveness. While some limitations exist in terms of reliance on external VLMs and computation cost, the benefits far outweigh these drawbacks. The potential impact on the field is significant, promising to democratize 3D content creation and inspire new research directions. The open-sourced model will accelerate progress in the area.

Score: 8.5

- **Score**: 8/10

## Other Papers
### **[Splat4D: Diffusion-Enhanced 4D Gaussian Splatting for Temporally and Spatially Consistent Content Creation](http://arxiv.org/abs/2508.07557v1)**
### **[Progressive Bird's Eye View Perception for Safety-Critical Autonomous Driving: A Comprehensive Survey](http://arxiv.org/abs/2508.07560v1)**
### **[Phoenix: A Novel Context-Aware Voice-Powered Math Equation Workspace and Editor](http://arxiv.org/abs/2508.07576v1)**
### **[Exploiting Layer Normalization Fine-tuning in Visual Transformer Foundation Models for Classification](http://arxiv.org/abs/2508.07577v1)**
### **[Towards Comprehensible Recommendation with Large Language Model Fine-tuning](http://arxiv.org/abs/2508.07595v1)**
### **[Keyword-Centric Prompting for One-Shot Event Detection with Self-Generated Rationale Enhancements](http://arxiv.org/abs/2508.07598v1)**
### **[HGMF: A Hierarchical Gaussian Mixture Framework for Scalable Tool Invocation within the Model Context Protocol](http://arxiv.org/abs/2508.07602v1)**
### **[LaVieID: Local Autoregressive Diffusion Transformers for Identity-Preserving Video Creation](http://arxiv.org/abs/2508.07603v1)**
### **[In-situ Value-aligned Human-Robot Interactions with Physical Constraints](http://arxiv.org/abs/2508.07606v1)**
### **[X2Edit: Revisiting Arbitrary-Instruction Image Editing through Self-Constructed Data and Task-Aware Representation Learning](http://arxiv.org/abs/2508.07607v1)**
### **[Klear-Reasoner: Advancing Reasoning Capability via Gradient-Preserving Clipping Policy Optimization](http://arxiv.org/abs/2508.07629v1)**
### **[Beyond Single: A Data Selection Principle for LLM Alignment via Fine-Grained Preference Signals](http://arxiv.org/abs/2508.07638v1)**
### **[Multi-Turn Jailbreaks Are Simpler Than They Seem](http://arxiv.org/abs/2508.07646v1)**
### **[LaRender: Training-Free Occlusion Control in Image Generation via Latent Rendering](http://arxiv.org/abs/2508.07647v1)**
### **[GraphCoT-VLA: A 3D Spatial-Aware Reasoning Vision-Language-Action Model for Robotic Manipulation with Ambiguous Instructions](http://arxiv.org/abs/2508.07650v1)**
### **[Understanding Users' Privacy Perceptions Towards LLM's RAG-based Memory](http://arxiv.org/abs/2508.07664v1)**
### **[1-2-3 Check: Enhancing Contextual Privacy in LLM via Multi-Agent Reasoning](http://arxiv.org/abs/2508.07667v1)**
### **[Semantic Caching for Low-Cost LLM Serving: From Offline Learning to Online Adaptation](http://arxiv.org/abs/2508.07675v1)**
### **[DiffVC-OSD: One-Step Diffusion-based Perceptual Neural Video Compression Framework](http://arxiv.org/abs/2508.07682v1)**
### **[LoSemB: Logic-Guided Semantic Bridging for Inductive Tool Retrieval](http://arxiv.org/abs/2508.07690v1)**
### **[Semantic-Enhanced Time-Series Forecasting via Large Language Models](http://arxiv.org/abs/2508.07697v1)**
### **[Make Your MoVe: Make Your 3D Contents by Adapting Multi-View Diffusion Models to External Editing](http://arxiv.org/abs/2508.07700v1)**
### **[What am I missing here?: Evaluating Large Language Models for Masked Sentence Prediction](http://arxiv.org/abs/2508.07702v1)**
### **[Training-Free ANN-to-SNN Conversion for High-Performance Spiking Transformer](http://arxiv.org/abs/2508.07710v1)**
### **[DoorDet: Semi-Automated Multi-Class Door Detection Dataset via Object Detection and Large Language Models](http://arxiv.org/abs/2508.07714v1)**
### **[Separation and Collaboration: Two-Level Routing Grouped Mixture-of-Experts for Multi-Domain Continual Learning](http://arxiv.org/abs/2508.07738v1)**
### **[Symmetry-Aware Transformer Training for Automated Planning](http://arxiv.org/abs/2508.07743v1)**
### **[Grouped Speculative Decoding for Autoregressive Image Generation](http://arxiv.org/abs/2508.07747v1)**
### **[Exploring Causal Effect of Social Bias on Faithfulness Hallucinations in Large Language Models](http://arxiv.org/abs/2508.07753v1)**
### **[Comparison Reveals Commonality: Customized Image Generation through Contrastive Inversion](http://arxiv.org/abs/2508.07755v1)**
### **[Correspondence as Video: Test-Time Adaption on SAM2 for Reference Segmentation in the Wild](http://arxiv.org/abs/2508.07759v1)**
### **[Sea-Undistort: A Dataset for Through-Water Image Restoration in High Resolution Airborne Bathymetric Mapping](http://arxiv.org/abs/2508.07760v1)**
### **[UniSVG: A Unified Dataset for Vector Graphic Understanding and Generation with Multimodal Large Language Models](http://arxiv.org/abs/2508.07766v1)**
### **[Pareto Multi-Objective Alignment for Language Models](http://arxiv.org/abs/2508.07768v1)**
### **[Dream4D: Lifting Camera-Controlled I2V towards Spatiotemporally Consistent 4D Generation](http://arxiv.org/abs/2508.07769v1)**
### **[AgentWorld: An Interactive Simulation Platform for Scene Construction and Mobile Robotic Manipulation](http://arxiv.org/abs/2508.07770v1)**
### **[Grove MoE: Towards Efficient and Superior MoE LLMs with Adjugate Experts](http://arxiv.org/abs/2508.07785v1)**
### **[Pose-RFT: Enhancing MLLMs for 3D Pose Generation via Hybrid Action Reinforcement Fine-Tuning](http://arxiv.org/abs/2508.07804v1)**
### **[Can You Trick the Grader? Adversarial Persuasion of LLM Judges](http://arxiv.org/abs/2508.07805v1)**
### **[EvoCoT: Overcoming the Exploration Bottleneck in Reinforcement Learning](http://arxiv.org/abs/2508.07809v1)**
### **[DiTVR: Zero-Shot Diffusion Transformer for Video Restoration](http://arxiv.org/abs/2508.07811v1)**
### **[Segmenting and Understanding: Region-aware Semantic Attention for Fine-grained Image Quality Assessment with Large Language Models](http://arxiv.org/abs/2508.07818v1)**
### **[Evaluating Large Language Models as Expert Annotators](http://arxiv.org/abs/2508.07827v1)**
### **[Large Language Models for Czech Aspect-Based Sentiment Analysis](http://arxiv.org/abs/2508.07860v1)**
### **[Tailored Emotional LLM-Supporter: Enhancing Cultural Sensitivity](http://arxiv.org/abs/2508.07902v1)**
### **[Diffusing the Blind Spot: Uterine MRI Synthesis with Diffusion Models](http://arxiv.org/abs/2508.07903v1)**
### **[Generative Video Matting](http://arxiv.org/abs/2508.07905v1)**
### **[RSVLM-QA: A Benchmark Dataset for Remote Sensing Vision Language Model-based Question Answering](http://arxiv.org/abs/2508.07918v1)**
### **[Score Augmentation for Diffusion Models](http://arxiv.org/abs/2508.07926v1)**
### **[\(X\)-evolve: Solution space evolution powered by large language models](http://arxiv.org/abs/2508.07932v1)**
### **[Careful Queries, Credible Results: Teaching RAG Models Advanced Web Search Tools with Reinforcement Learning](http://arxiv.org/abs/2508.07956v1)**
### **[Large Language Models for Subjective Language Understanding: A Survey](http://arxiv.org/abs/2508.07959v1)**
### **[WeChat-YATT: A Simple, Scalable and Balanced RLHF Trainer](http://arxiv.org/abs/2508.07970v1)**
### **[Adaptive Multiple Access and Service Placement for Generative Diffusion Models](http://arxiv.org/abs/2508.07978v1)**
### **[The Escalator Problem: Identifying Implicit Motion Blindness in AI for Accessibility](http://arxiv.org/abs/2508.07989v1)**
### **[WideSearch: Benchmarking Agentic Broad Info-Seeking](http://arxiv.org/abs/2508.07999v1)**
### **[Progressive Depth Up-scaling via Optimal Transport](http://arxiv.org/abs/2508.08011v1)**
### **[EchoAid: Enhancing Livestream Shopping Accessibility for the DHH Community](http://arxiv.org/abs/2508.08020v1)**
### **[Robust Anomaly Detection in O-RAN: Leveraging LLMs against Data Manipulation Attacks](http://arxiv.org/abs/2508.08029v1)**
### **[Audio-Thinker: Guiding Audio Language Model When and How to Think via Reinforcement Learning](http://arxiv.org/abs/2508.08039v1)**
### **[On Understanding of the Dynamics of Model Capacity in Continual Learning](http://arxiv.org/abs/2508.08052v1)**
### **[AdaptFlow: Adaptive Workflow Optimization via Meta-Learning](http://arxiv.org/abs/2508.08053v1)**
### **[Investigating the Design Space of Visual Grounding in Multimodal Large Language Model](http://arxiv.org/abs/2508.08066v1)**
### **[Heterogeneity in Entity Matching: A Survey and Experimental Analysis](http://arxiv.org/abs/2508.08076v1)**
### **[Matrix-3D: Omnidirectional Explorable 3D World Generation](http://arxiv.org/abs/2508.08086v1)**
### **[MDD-Net: Multimodal Depression Detection through Mutual Transformer](http://arxiv.org/abs/2508.08093v1)**
### **[Dual Information Speech Language Models for Emotional Conversations](http://arxiv.org/abs/2508.08095v1)**
### **[Assessing LLM Text Detection in Educational Contexts: Does Human Contribution Affect Detection?](http://arxiv.org/abs/2508.08096v1)**
### **[TBAC-UniImage: Unified Understanding and Generation by Ladder-Side Diffusion Tuning](http://arxiv.org/abs/2508.08098v1)**
### **[Learned Regularization for Microwave Tomography](http://arxiv.org/abs/2508.08114v1)**
### **[TeamMedAgents: Enhancing Medical Decision-Making of LLMs Through Structured Teamwork](http://arxiv.org/abs/2508.08115v1)**
### **[Optimal Transport Regularization for Speech Text Alignment in Spoken Language Models](http://arxiv.org/abs/2508.08131v1)**
### **[FantasyStyle: Controllable Stylized Distillation for 3D Gaussian Splatting](http://arxiv.org/abs/2508.08136v1)**
### **[Can LLMs Detect Their Confabulations? Estimating Reliability in Uncertainty-Aware Language Models](http://arxiv.org/abs/2508.08139v1)**
### **[Data-Efficient Biomedical In-Context Learning: A Diversity-Enhanced Submodular Perspective](http://arxiv.org/abs/2508.08140v1)**
### **[An effective potential for generative modelling with active matter](http://arxiv.org/abs/2508.08146v1)**
### **[From Natural Language to Solver-Ready Power System Optimization: An LLM-Assisted, Validation-in-the-Loop Framework](http://arxiv.org/abs/2508.08147v1)**
### **[REX-RAG: Reasoning Exploration with Policy Correction in Retrieval-Augmented Generation](http://arxiv.org/abs/2508.08149v1)**
### **[PyVeritas: On Verifying Python via LLM-Based Transpilation and Bounded Model Checking for C](http://arxiv.org/abs/2508.08171v1)**
### **[CD-TVD: Contrastive Diffusion for 3D Super-Resolution with Scarce High-Resolution Time-Varying Data](http://arxiv.org/abs/2508.08173v1)**
### **[MedReasoner: Reinforcement Learning Drives Reasoning Grounding from Clinical Thought to Pixel-Level Precision](http://arxiv.org/abs/2508.08177v1)**
### **[THAT: Token-wise High-frequency Augmentation Transformer for Hyperspectral Pansharpening](http://arxiv.org/abs/2508.08183v1)**
### **[Reinforcement Learning in Vision: A Survey](http://arxiv.org/abs/2508.08189v1)**
### **[Efficient Speculative Decoding for Llama at Scale: Challenges and Solutions](http://arxiv.org/abs/2508.08192v1)**
### **[Street-Level AI: Are Large Language Models Ready for Real-World Judgments?](http://arxiv.org/abs/2508.08193v1)**
### **[Human-Alignment and Calibration of Inference-Time Uncertainty in Large Language Models](http://arxiv.org/abs/2508.08204v1)**
### **[SAEMark: Multi-bit LLM Watermarking with Inference-Time Scaling](http://arxiv.org/abs/2508.08211v1)**
### **[Learning User Preferences for Image Generation Model](http://arxiv.org/abs/2508.08220v1)**
### **[Multi-head Transformers Provably Learn Symbolic Multi-step Reasoning via Gradient Descent](http://arxiv.org/abs/2508.08222v1)**
### **[Capabilities of GPT-5 on Multimodal Medical Reasoning](http://arxiv.org/abs/2508.08224v1)**
### **[OMGSR: You Only Need One Mid-timestep Guidance for Real-World Image Super-Resolution](http://arxiv.org/abs/2508.08227v1)**
### **[LL3M: Large Language 3D Modelers](http://arxiv.org/abs/2508.08228v1)**
### **[ODYSSEY: Open-World Quadrupeds Exploration and Manipulation for Long-Horizon Tasks](http://arxiv.org/abs/2508.08240v1)**
### **[Bringing Everyone to the Table: An Experimental Study of LLM-Facilitated Group Decision Making](http://arxiv.org/abs/2508.08242v1)**
### **[StableAvatar: Infinite-Length Audio-Driven Avatar Video Generation](http://arxiv.org/abs/2508.08248v1)**
