import json
import os

def convert_niah_jsonl(input_jsonl_path_list, output_path_train, output_path_valid):
    """
    Convert a NIAH-style JSONL file into a formatted JSON file with the desired structure.
    The output is saved as 'formatted_output.json' in the same directory as the input.
    """
    formatted_data_train = []
    formatted_data_val = []

    for input_jsonl_path in input_jsonl_path_list: 
        print (f"Processing file: {input_jsonl_path}" )
        
        idx=0
        with open(input_jsonl_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                instruction = "<|begin_of_text|>" + entry["input"]
                outputs = entry["outputs"]
                answer_prefix = entry["answer_prefix"]

                instruction += answer_prefix  + " "
                # output_text = answer_prefix + "\n" + "\n".join(outputs) + "<|eot_id|>"
                indexed_outputs = [f"{uuid}" for i, uuid in enumerate(outputs)]  
                output_text = "\n".join(indexed_outputs) + "<|eot_id|>"

                # if idx == 1: 
                #     break 
                if idx < 1470:
                    formatted_data_train.append({
                        "instruction": instruction,
                        "input": "",
                        "output": output_text
                    })
                else: 
                    print (f'at idx {idx}, adding to validation set')
                    formatted_data_val.append({
                        "instruction": instruction,
                        "input": "",
                        "output": output_text
                    })

                idx+=1 

    os.makedirs (os.path.dirname (output_path_train), exist_ok=True)  
    os.makedirs (os.path.dirname (output_path_valid), exist_ok=True)  
    
    with open(output_path_train, "w") as out_file:
        json.dump(formatted_data_train, out_file, indent=2)
    with open(output_path_valid, "w") as out_file:
        json.dump(formatted_data_val, out_file, indent=2)

    print(f"Converted formatted_data_train {len(formatted_data_train)} entries.")
    print(f"Saved to {output_path_train}")
    print(f"Converted formatted_data_val {len(formatted_data_val)} entries.")
    print(f"Saved to {output_path_valid}")

data_list = ["/data/original_8/8192/data/niah_multivalue/validation.jsonl",
            "/data/original_8/12288/data/niah_multivalue/validation.jsonl",
            "/data/original_8/16384/data/niah_multivalue/validation.jsonl", 
            "/data/original_16/8192/data/niah_multivalue/validation.jsonl",
            "/data/original_16/12288/data/niah_multivalue/validation.jsonl",
            "/data/original_16/16384/data/niah_multivalue/validation.jsonl", 
            "/data/original_32/8192/data/niah_multivalue/validation.jsonl",
            "/data/original_32/12288/data/niah_multivalue/validation.jsonl",
            "/data/original_32/16384/data/niah_multivalue/validation.jsonl"]
output_path_train = "/data/train.json" 
output_path_valid = "/data/valid.json" 

convert_niah_jsonl(data_list, output_path_train, output_path_valid )
