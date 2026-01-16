#!/usr/bin/env python3
"""
Extract code from results JSONL files by task index and model name.

Usage:
    python extract_code.py <task_idx> <model_name>
    
Example:
    python extract_code.py 0 "openai/gpt-5.2"
    python extract_code.py 2 "openai/gpt-4"
"""

import json
import os
import sys


def extract_code(task_idx: int, model_name: str, output_dir: str = "outputs") -> str:
    """
    Extract code from results file for a specific task and model.
    
    Args:
        task_idx: The task index (idx field in JSON)
        model_name: The model name (e.g., "openai/gpt-5.2")
        output_dir: Directory where results files are stored
        
    Returns:
        The extracted code as a string, or None if not found
    """
    # Convert model name to safe filename format
    model_safe_name = model_name.replace('/', '_').replace('\\', '_')
    results_file = os.path.join(output_dir, f"results_{model_safe_name}.jsonl")
    
    if not os.path.exists(results_file):
        return None
    
    # Search through the JSONL file for matching task index
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
                
                # Check if this is the task we're looking for
                if record.get('idx') == task_idx:
                    # Try to get code from different possible fields
                    code = record.get('code')
                    if code:
                        return code
                    
                    # Fallback: try to extract from generated_text if code field doesn't exist
                    generated_text = record.get('generated_text', '')
                    if generated_text:
                        # Try to extract code from markdown blocks
                        import re
                        patterns = [
                            r'```(?:cpp|c\+\+|c)?\s*\n(.*?)```',
                            r'```(?:cpp|c\+\+|c)?\s*(.*?)```',
                        ]
                        for pattern in patterns:
                            matches = re.findall(pattern, generated_text, re.DOTALL)
                            if matches:
                                return matches[0].strip()
                        
                        # If no code block found, return the full generated_text
                        return generated_text
                    
                    return None
                    
            except json.JSONDecodeError:
                continue
    
    return None


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) != 3:
        print("Usage: python extract_code.py <task_idx> <model_name>")
        print('\nExample:')
        print('  python extract_code.py 0 "openai/gpt-5.2"')
        print('  python extract_code.py 2 "openai/gpt-4"')
        sys.exit(1)
    
    try:
        task_idx = int(sys.argv[1])
    except ValueError:
        print(f"Error: Task index must be an integer, got: {sys.argv[1]}")
        sys.exit(1)
    
    model_name = sys.argv[2]
    
    # Extract code
    code = extract_code(task_idx, model_name)
    
    if code is None:
        print(f"No code found for task index {task_idx} and model {model_name}")
        sys.exit(1)
    
    # Print the code
    print(code)


if __name__ == "__main__":
    main()
