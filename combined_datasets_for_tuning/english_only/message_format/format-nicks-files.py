import json
import pandas as pd

# Read the JSONL files
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Load the datasets
test_data = read_jsonl('/Users/jgilhuly/Downloads/test_message_prompt_en_no-dedup_2025-02-13 (2).jsonl')
val_data = read_jsonl('/Users/jgilhuly/Downloads/validation_message_prompt_en_no-dedup_2025-02-13 (1).jsonl')
train_data = read_jsonl('/Users/jgilhuly/Downloads/train_message_prompt_en_no-dedup_2025-02-13 (2).jsonl')

# Convert to dataframes for easier deduplication
def extract_fields(data):
    rows = []
    for item in data:
        if 'messages' in item:  # Message format
            user_msg = next(m for m in item['messages'] if m['role'] == 'user')
            content = user_msg['content']
        else:  # Prompt format
            content = item['prompt']
            
        # Extract input, reference, output from the content
        # This assumes consistent formatting in the prompt template
        input_start = content.find("Input: ") + 7
        reference_start = content.find("Reference: ") + 11
        output_start = content.find("Output: ") + 8
        
        input_text = content[input_start:content.find("Reference:")].strip()
        reference_text = content[reference_start:content.find("Output:")].strip()
        output_text = content[output_start:].strip()
        
        rows.append({
            'input': input_text,
            'reference': reference_text, 
            'output': output_text,
            'original': item
        })
    return pd.DataFrame(rows)

test_df = extract_fields(test_data)
val_df = extract_fields(val_data)
train_df = extract_fields(train_data)

# Get unique combinations across all datasets
all_combinations = pd.concat([
    test_df[['input', 'reference', 'output']],
    val_df[['input', 'reference', 'output']],
    train_df[['input', 'reference', 'output']]
]).drop_duplicates()

# Filter each dataset to keep only first occurrence
test_dedup = test_df.drop_duplicates(subset=['input', 'reference', 'output'])
val_dedup = val_df.drop_duplicates(subset=['input', 'reference', 'output'])
train_dedup = train_df.drop_duplicates(subset=['input', 'reference', 'output'])

# Write deduplicated data back to JSONL
def write_jsonl(data, filename):
    with open(filename, 'w') as f:
        for item in data['original']:
            json.dump(item, f)
            f.write('\n')

write_jsonl(test_dedup, 'test_dedup_nick.jsonl')
write_jsonl(val_dedup, 'validation_dedup_nick.jsonl') 
write_jsonl(train_dedup, 'train_dedup_nick.jsonl')

print(f"Original counts: Test={len(test_data)}, Val={len(val_data)}, Train={len(train_data)}")
print(f"Deduplicated counts: Test={len(test_dedup)}, Val={len(val_dedup)}, Train={len(train_dedup)}")
