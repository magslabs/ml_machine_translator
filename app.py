import pandas as pd
from ollama import * 
import os
from time import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Map models identified
models_mapping = {
    'DeepSeek': 'deepseek-r1:8b',
    'LLaMA3': 'llama3.1:8b',
    'Gemma3': 'gemma3:4b'
}

# Load Datasets
dataset_01 = pd.read_csv("./static/datasets/english-ceb-bible-prompt.csv")
dataset_01_file_name = os.path.splitext(os.path.basename('./static/datasets/english-ceb-bible-prompt.csv'))[0]

dataset_02 = pd.read_csv("./static/datasets/english-ceb-bible-prompt-edited-to-cebuano-to-english.csv")
dataset_02_file_name = os.path.splitext(os.path.basename('./static/datasets/english-ceb-bible-prompt-edited-to-cebuano-to-english.csv'))[0]

dataset_03 = pd.read_csv("./static/datasets/cebuano_to_english_refactored_prompts_per_model.csv")
dataset_03_file_name = os.path.splitext(os.path.basename('./static/datasets/cebuano_to_english_refactored_prompts_per_model.csv'))[0]

sample_size = 1000

# sample_data = dataset_01.head(sample_size).reset_index(drop=True)
# sample_data = dataset_02.head(sample_size).reset_index(drop=True)
sample_data = dataset_03.head(sample_size).reset_index(drop=True)

# Set Ollama Client
client = Client(host="http://ollama.local")

# Check if model is loaded -> this makes sure the model is loaded before testing
def is_model_loaded(model_name: str) -> bool:
    models = client.list()["models"]
    for model in models:
        if model.get("model") == model_name and model.get("loaded", False):
            return True
    return False

# Prompt the model -> parameters: model_key, prompt
# Prompt the model with the given prompt and return the response
def prompt_model(model_key: str, prompt: str):
    response = client.generate(
        model=models_mapping[model_key],
        prompt=prompt,
        stream=False,
    )
    return response["response"]

def test_model(model_key: str, sample_data: pd.DataFrame, output_file: str):
    model_id = models_mapping[model_key]

    if not is_model_loaded(model_id):
        try:
            client.generate(model=model_id, prompt="Hello", stream=False)
        except Exception as e:
            print(f"    Could not warm up model: {e}")
            return
    
    results = [None] * len(sample_data)

    def process_row(row_num, row):
        prompt_col = f'Prompt_{model_key}' if f'Prompt_{model_key}' in row else 'prompt'
        prompt = row[prompt_col]
        expected_output = row.get('english', '')

        print(f"  Processing row {row_num + 1}/{sample_size}...")

        try:
            response = prompt_model(model_key, prompt)
            print("     Translation completed")
            return row_num, {
                'Prompt': prompt,
                'Expected Output': expected_output,
                'Model Response': response
            }
        except Exception as e:
            print(f"    Error: {e}")
            return row_num, {
                'Prompt': prompt,
                'Expected Output': expected_output,
                'Model Response': f"ERROR: {str(e)}"
            }

    max_threads = min(16, multiprocessing.cpu_count())
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [
            executor.submit(process_row, row_num, row)
            for row_num, row in sample_data.iterrows()
        ]
        for future in as_completed(futures):
            row_num, result = future.result()
            results[row_num] = result

    results_df = pd.DataFrame(results)

    # Write results to Excel
    if os.path.exists(output_file):
        with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            results_df.to_excel(writer, sheet_name=f"{model_key}_results", index=False)
    else:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name=f"{model_key}_results", index=False)

    print(f"\nAll testing for {model_key} completed. Results saved to {output_file}")
    

if __name__ == "__main__":
    output_dir = "./static/results/" + dataset_02_file_name
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/translation_results.xlsx"
        
    model_to_test = 'DeepSeek'
    # model_to_test = 'LLaMA3'
    # model_to_test = 'Gemma3'
    
    print(f"Testing model: {model_to_test}")
    test_model(model_to_test, sample_data, output_file)
