#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import fire
import datasets
import numpy as np
import re

def transform_question_format(item):
    # Skip if "you" or "You" is in the question
    if "you" in item['question'].lower():
        return [item]  # Return list with single item
    
    # Get the completion with highest pass_rate
    inferences = item['inferences']
    best_completion = max(inferences, key=lambda x: x['pass_rate'])['completion']
    
    # Extract function definition using split
    func_def = best_completion.split(':\n')[0].strip()
    if not func_def.startswith('def '):
        return [item]  # Return list with single item
    
    # Remove 'def ' prefix to get function name and parameters
    func_def = func_def[4:]  # Remove 'def '
    
    # Create new item with transformed question
    new_item = item.copy()
    new_item['question'] = f"""Please complete the following problem:
```
def {func_def}:
    \"\"\"
    {item['question']}
    \"\"\"
```
"""
    return [item, new_item]  # Return list with both original and transformed items

def main(
    dataset_path="TIGER-Lab/AceCode-89K",
    output_path="./data/acecode_89k.json",
    difficulty="hard",  # Options: "easy", "medium", "hard"
):
    dataset = datasets.load_dataset(dataset_path, split="train")   
    print(f"Loaded {len(dataset)} examples")
    
    if difficulty in ["easy", "medium", "hard"]:
        consider_models = ["qwen_coder_2.5", 'llama3_instruct']
        def get_accs(item):
            inference = item['inferences']
            inference = [x for x in inference if x['model_name'] in consider_models]
            accs = [x['pass_rate'] for x in inference]
            item['accs'] = accs
            return item
        dataset = dataset.map(get_accs, desc="Getting accs", num_proc=4)
        
        # Calculate average accuracies
        avg_accs = [np.mean(x) for x in dataset['accs']]
        std_accs = [np.std(x) for x in dataset['accs']]
        
        # Define thresholds for different difficulty levels
        if difficulty == "hard":
            # Keep bottom 25% by average accuracy
            split_acc = np.percentile(avg_accs, 20)
            dataset = dataset.filter(lambda x: np.mean(x['accs']) <= split_acc, desc="Filtering low average examples")
            # Then keep top 50% by standard deviation
            split_std = np.percentile(std_accs, 40)
            dataset = dataset.filter(lambda x: np.std(x['accs']) >= split_std, desc="Filtering high std examples")
        elif difficulty == "medium":
            # Keep middle 50% by average accuracy
            lower_acc = np.percentile(avg_accs, 25)
            upper_acc = np.percentile(avg_accs, 75)
            dataset = dataset.filter(
                lambda x: lower_acc <= np.mean(x['accs']) <= upper_acc,
                desc="Filtering medium average examples"
            )
        elif difficulty == "easy":
            # Keep top 25% by average accuracy
            split_acc = np.percentile(avg_accs, 75)
            dataset = dataset.filter(lambda x: np.mean(x['accs']) >= split_acc, desc="Filtering high average examples")
        
        print(f"Kept {len(dataset)} examples for {difficulty} difficulty")
        for i in range(min(3, len(dataset))):
            print(dataset[i]['accs'])
        
        dataset = dataset.remove_columns("accs")

    # Transform questions that don't contain "you" and keep both versions
    transformed_items = []
    for item in dataset:
        transformed_items.extend(transform_question_format(item))
    
    # Create new dataset from the list of items
    dataset = datasets.Dataset.from_list(transformed_items)
    print('Total examples:', len(dataset))
    
    dataset = dataset.remove_columns("inferences")
    dataset.to_json(output_path)
    print(f"Successfully saved to {output_path}")
    
if __name__ == "__main__":
    fire.Fire(main)
    
    
"""
python process_data.py --dataset_path "TIGER-Lab/AceCode-89K" --output_path "./acecode_hard.jsonl" --difficulty "hard"
"""