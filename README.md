# Dissociating Answerability from Extraction in LLM
Question Answering over Radiology Reports

Code for evaluating large language models on [RadQA](https://physionet.org/content/radqa/1.0.0/), a question-answering dataset over MIMIC-III radiology reports. Answering a RadQA question takes two decisions: whether the report contains an answer, and which span gives it. Alongside SQuAD v2 F1, the evaluation reports the false-answer rate (FAR, the share of unanswerable questions a model answers) and the over-abstention rate (OAR, the share of answerable questions it declines). It also splits any change in F1 between conditions into an extraction term and an answerability term.

The code runs 13 LLMs zero-shot and with three cumulative demonstration sets (an answerable Findings question, then an answerable Impression question, then an unanswerable question), and then scores the stored outputs.

## Setup

1. Clone the repo.

   ```bash
   git clone <repo_url>
   cd rad-qa-llm-eval
   ```

2. Create and activate a virtual environment, then install the requirements.

   ```bash
   python3 -m venv radqa_venv
   source radqa_venv/bin/activate
   python -m pip install -r requirements.txt
   ```

3. Download the data from [PhysioNet](https://physionet.org/content/radqa/1.0.0/) and place it in the `data` directory, so that the split files sit at `data/radqa/1.0.0/` (`train.json`, `dev.json`, `test.json`). RadQA is distributed under the PhysioNet credentialed-access license, so the data is not included in this repo.

4. Set the model configuration in `config.yaml` before running inference. For Bedrock models, set the AWS credentials profile (`bedrock.credentials_profile_name`) and, for each model under `bedrock.models`, its `model_id`, `region_name`, optional `base_url` and `model_config` (e.g. `max_tokens`). Gemini models go under `vertex_gcp.models`. The file lists all 13 models as commented-out templates; uncomment the ones to run.

## 1. Inference

Model calls are made from the notebooks in `notebooks/`, using the models set in `config.yaml` (step 4 above).

| Notebook | Models |
|---|---|
| `bedrock_driver.ipynb` | AWS Bedrock models (Claude, DeepSeek, GPT-OSS, LLaMA 4, Mistral), as listed under `bedrock.models` in `config.yaml` |
| `gemini-driver.ipynb` | Gemini models on Vertex AI |

Every notebook calls the model with structured output (tool calling) under the schema in `src/response_model.py`: `answer_extracted`, `answerable`, `reasoning`. Set `K_SHOT` (0–3) in the notebook. Demonstrations are drawn once from the train split with a fixed seed (`src/few_shot.py`) and reused for every model and test question. Prompts are in `src/prompts/`.

Outputs are written as chunk files to `results-{k}shot/<model_id>/`. `notebooks/results-consolidator.ipynb` merges them into one `test.json` per model and condition.

## 2. Evaluation

Run from the project root, in this order. All outputs go to `results/`.

```bash
# Question-level scores (token F1, EM, RadGraph F1, answerability) and per-model tests
python scripts/run_question_level_eval.py            # add --no-radgraph to skip RadGraph F1

# Decomposition, consensus gate, clustered CIs, FAR tests, selectivity, human agreement
python scripts/run_answer_decision_analyses.py       # or --only decompose gaterobust ...

# Condition-level Wilcoxon over model means
python scripts/run_shot_significance_tests.py

# Error categories, and RadGraph validation of the lexical categories
python scripts/run_statistical_analyses.py --analysis error
python scripts/run_statistical_analyses.py --analysis taxonomy

# Figures (PNG, written to figures/)
python scripts/make_figures.py
```

`run_answer_decision_analyses.py --help` lists every analysis and its output file. The `human` analysis loads RadGraph and takes about 20 minutes on CPU.

## Layout

```
config.yaml                     model and provider settings for inference
notebooks/                      inference drivers and the output consolidator
src/data_loader/                RadQA loading (flattened to one row per question x gold answer)
src/few_shot.py                 demonstration sampling and rendering
src/prompts/                    per-provider prompts
src/providers/                  LangChain model wrappers and structured-output calls
src/response_model.py           output schema
src/eval/radqa_metrics_enhanced.py  SQuAD v2 F1/EM, RadGraph F1, answerability metrics, McNemar
src/eval/question_level.py      question-level scoring (one prediction per question, max over golds)
scripts/                        evaluation and analysis entry points
```

## Scoring notes

- **Unit:** the question. The test split has 614 questions over 104 reports, 154 of them unanswerable. The data loader flattens 280 doubly annotated questions into two rows each (894 rows). Scoring takes the first prediction per question and the maximum over the annotators' answers, following SQuAD v2.
- **Answerability:** an answerable/unanswerable mismatch scores 0, and a correct abstention scores 1.
- **Inference:** confidence intervals are bootstrap intervals clustered by report.
