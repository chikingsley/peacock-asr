# Audio LLM

Fine-tune and inference audio+text models, starting with Qwen 2 7B Audio.

## Data Preparation
Install requirements and log in to huggingface:
```
uv venv
uv pip install -r requirements.txt
uv run huggingface-cli login
uv python create-bird-dataset.py
```

## Fine-tuning
Run the `Trelis_Qwen_Audio_Fine_Tuning.ipynb` notebook. This notebook is 50 MB because of outputs, so you can use `Trelis_Qwen_Audio_Fine_Tuning_NO_OUTPUTS.ipynb` for faster upload/download.

## Inference
For inference, vLLM is recommended. See the [one-click-llms](https://github.com/TrelisResearch/one-click-llms) repo for more details.

You can also inference, for testing, in Google Colab using [this link]().