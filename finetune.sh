# #!/bin/bash
# Fine-tune fatigue model
python finetune.py \
  --target fatigue \
  --train_data "C:\Users\SIU856561805\HealthLLMFinal\HealthLLMV1\data\AWFB_fatigue_train_all.json" \
  --output_dir "C:\Users\SIU856561805\HealthLLMFinal\HealthLLMV1\output\fatigue_model"
# Fine-tune stress model
python finetune.py \
  --target stress \
  --train_data "C:\Users\SIU856561805\HealthLLMFinal\HealthLLMV1\data\AWFB_stress_train_all.json" \
  --output_dir "C:\Users\SIU856561805\HealthLLMFinal\HealthLLMV1\output\stress_model"

# Fine-tune readiness model
python finetune.py \
  --target readiness \
  --train_data "C:\Users\SIU856561805\HealthLLMFinal\HealthLLMV1\data\AWFB_readiness_train_all.json" \
  --output_dir "C:\Users\SIU856561805\HealthLLMFinal\HealthLLMV1\output\readiness_model"

# Fine-tune sleep quality model
python finetune.py \
  --target sleep_quality \
  --train_data "C:\Users\SIU856561805\HealthLLMFinal\HealthLLMV1\data\AWFB_sleep_quality_train_all.json" \
  --output_dir "C:\Users\SIU856561805\HealthLLMFinal\HealthLLMV1\output\sleep_quality_model"