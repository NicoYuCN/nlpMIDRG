*This is a natural language processing (NLP) project for medical imaging diagnositic report generation (MIDRG).*

**Background**
<br />
Medical imaging description and disease diagnosis are vitally important yet time-consuming. Automatic medical imaging diagnostic report generation (MIDRG) from medical image description can reduce clinicians’ workload and improve their routine efficiency. As a cost-effective approach, fine-tuning of pre-trained large language models (LLMs) becomes indispensable for downstream applications. However, semantic inconsistency of sentence embedding has been massively witnessed from undesirable repetitions or unnaturalness in text generation.

**Our work**
<br />
To address the underlying issue of anisotropic distribution of token representation, in this study, a contrastive learning penalized cross-entropy (CLpCE) objective function is implemented firstly to enhance the semantic consistency and accuracy of token representation by guiding the fine-tuning procedure towards a specific task. Further, to improve the diversity of text summarization and to prevent sampling from unreliable tail of token distributions, a diversity contrastive
search (DCS) decoding method is designed for restricting the report generation derived from a probable candidate set with maintained semantic coherence. Based on the LLM of pre-trained GPT-2, the proposed CLpCE with DCS decoding framework is validated on 30,000 desensitized text samples from the “Medical Imaging Diagnosis Report Generation” track of 2023 Global Artificial Intelligence Technology Innovation Competition. Using four kinds of metrics evaluated from n-gram word matching, semantic relevance and content similarity, extensive experiments reveal that the proposed framework achieves improvement of semantic coherence and diversity on the MIDRG task. The phenomenon of dull or repetitive text generation is common when fine-tuning pre-trained LLMs for natural language processing applications.

**Our contribution**
<br />
This study might shed some light on relieving this issue by developing comprehensive strategies to enhance semantic coherence, accuracy and diversity of sentence embedding. In summary, the novelty comes from (1) An objective function CLpCE is designed for balancing both unsupervised and supervised learning in the model fine-tuning stage to enhance the consistency of feature representation of sentence embedding. (2) A novel decoding method DCS is proposed to improve the representation diversity and to relieve anisotropic distributions of token generation with maintained quality of text summarization. (3) The effectiveness of the CLpCEwDCS decoding framework is verified, and competitive performance and better diversity are observed on the MIDRG task.
