# Large Language Models for Human Activity Prediction Using Structured Data

This application uses Large Language Models (LLMs) to predict various health metrics (fatigue, stress, readiness, and sleep quality) based on wearable sensor data. It supports multiple models including fine-tuned models, OpenAI GPT-3.5, and Google Gemini.

## Features

- **Multiple Health Predictions:**
  - Fatigue Level (1-5)
  - Stress Level (1-5)
  - Readiness Level (1-5)
  - Sleep Quality (1-5)

- **Supported Models:**
  - Fine-tuned Models (based on TinyLlama)
  - OpenAI GPT-3.5
  - Google Gemini

- **Inference Modes:**
  - Zero-shot: Direct predictions
  - Few-shot: Uses examples for better context
  - Few-shot with Chain of Thought (CoT): Detailed reasoning
  - Few-shot with Self-Checking CoT: Enhanced accuracy

## Setup

1. **Create a Python Environment:**
   ```bash
   conda create -n healthllm python>=3.9
   conda activate healthllm
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   - Copy `.env.example` to `.env`
   - Add your API keys:
     ```
     OPENAI_API_KEY=your-openai-key
     GOOGLE_API_KEY=your-google-key
     BASE_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
     DEVICE=cuda  # or mps for Mac, defaults to cpu
     ```
## Datasets

   1.PMData: https://datasets.simula.no/pmdata/
   2.AW_FB: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ZS2Z2J
   
## Usage

### 1. Generate Training Data
```bash
python gen_dataset.py --samples 1000 --data-dir data
```
This will create training datasets for all prediction types in the specified directory.

### 2. Fine-tune Models
```bash
bash finetune.sh
```
This script fine-tunes models for each prediction type using the generated datasets.

### 3. Run the Application
```bash
streamlit run app.py
```

The web interface provides:
- Model selection (Fine-tuned, OpenAI, Gemini)
- Prediction type selection
- Inference mode selection
- Input field for sensor data

### Example Input Format
```
The recent 10-days sensor readings show: [Steps]: 8000 steps/day, [Burned Calories]: 1800 calories/day, [Heart Rate]: 75 beats/min, [SleepMinutes]: 420 minutes, [Mood]: 4 out of 5
```

## Directory Structure

- `app.py`: Main Streamlit application
- `llm_providers.py`: LLM implementations
- `gen_dataset.py`: Dataset generation script
- `finetune.py`: Model fine-tuning script
- `data/`: Training datasets
- `output/`: Fine-tuned models
- `test_questions.md`: Sample test cases
- `prompts/`: Jinja templates for model prompts
  - `base.j2`: Base template for all predictions
  - `few_shot.j2`: Template with examples
  - `few_shot_cot.j2`: Template with chain of thought reasoning
  - `few_shot_cot_sc.j2`: Template with self-checking

## Customizing Prompts

The application uses Jinja2 templates for all model prompts, located in the `prompts/` directory. You can customize these templates to modify how the models interact with users:

1. **Base Template (`base.j2`)**: 
   - Contains the fundamental instruction for health predictions
   - Used by all other templates

2. **Few-Shot Template (`few_shot.j2`)**:
   - Includes example predictions
   - Helps model understand the task better

3. **Chain of Thought Template (`few_shot_cot.j2`)**:
   - Demonstrates step-by-step reasoning
   - Improves prediction accuracy

4. **Self-Checking Template (`few_shot_cot_sc.j2`)**:
   - Includes verification steps
   - Ensures comprehensive analysis

Variables available in templates:
- `{{ target_name }}`: Type of prediction (e.g., "fatigue level")
- `{{ min_val }}`: Minimum value in range (1)
- `{{ max_val }}`: Maximum value in range (5)
- `{{ instruction }}`: User's input query

## Testing

The repository includes `test_questions.md` with 40 test cases (10 for each prediction type) to validate the model's performance across different scenarios.

## Model Performance

The fine-tuned models are optimized for health predictions and can achieve comparable performance to larger models like GPT-3.5 and Gemini-Pro. The models consider multiple factors:
- Physical activity (steps, calories)
- Cardiovascular health (heart rate)
- Sleep patterns
- Mental state (mood)

## Troubleshooting

1. **Model Loading Issues:**
   - Ensure sufficient RAM/VRAM
   - Check DEVICE setting in .env
   - Verify model paths in output directory

2. **API Errors:**
   - Verify API keys in .env
   - Check internet connection
   - Ensure API quota availability

3. **Performance Issues:**
   - Try different inference modes
   - Adjust input data quality
   - Check system resources

## License

This project is licensed under the terms of the LICENSE.md file.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{kim2024healthllm,
      title={Health-LLM: Large Language Models for Health Prediction via Wearable Sensor Data}, 
      author={Yubin Kim and Xuhai Xu and Daniel McDuff and Cynthia Breazeal and Hae Won Park},
      year={2024},
      eprint={2401.06866},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

