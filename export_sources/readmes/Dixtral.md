---
library_name: transformers
tags:
- speech
- automatic-speech-recognition
- speech-language-model
- target-speaker-asr
- multi-talker
- speaker-diarization
- meeting-transcription
- Dixtral
- Voxtral
- DiCoW
- BUT-FIT
pipeline_tag: automatic-speech-recognition
license: apache-2.0
base_model: mistralai/Voxtral-Mini-3B-2507
datasets:
- microsoft/NOTSOFAR
- edinburghcstr/ami
---

# 🧠 Dixtral — BUT-FIT Diarization-Conditioned Voxtral for Target-Speaker ASR

This repository hosts **Dixtral**, developed by [BUT Speech@FIT](https://github.com/BUTSpeechFIT). 
**Dixtral** couples the **Voxtral-Mini-3B** spoken-language model with the **DiCoW** diarization-conditioned encoder, giving the LLM target-speaker awareness in multi-talker audio.

This checkpoint is tuned for **target-speaker / multi-talker transcription (TS-ASR)** of conversational and meeting recordings. For spoken question answering, use [**Dixtral_QA**](https://huggingface.co/BUT-FIT/Dixtral_QA) instead.

## 🛠️ Model Usage

```python
from transformers import AutoModel, AutoProcessor

MODEL_NAME = "BUT-FIT/Dixtral"
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
```

➡️ For full inference pipelines (diarization → FDDT masks → generation), see the
[**Dixtral GitHub repository**](https://github.com/BUTSpeechFIT/Dixtral).

---

## 📦 Model Details

* **Base Model:** [Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)
* **Encoder:** DiCoW v3 large
* **Training Datasets:**
  * [NOTSOFAR-1](https://github.com/microsoft/NOTSOFAR1-Challenge)
  * [AMI Meeting Corpus](http://groups.inf.ed.ac.uk/ami/corpus/)
  * [LibriMix / LibriSpeechMix](https://github.com/JorisCos/LibriMix)


---

## 📬 Contact

📧 **Email:** [ipoloka@fit.vut.cz](mailto:ipoloka@fit.vut.cz)
🏢 **Affiliation:** [BUT Speech@FIT](https://github.com/BUTSpeechFIT), Brno University of Technology
🔗 **GitHub:** [BUTSpeechFIT](https://github.com/BUTSpeechFIT)
