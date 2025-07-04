# main.py
import argparse

def process_file(file_path):
    """Process a single input file."""
    print(f"Processing file: {file_path}")
    with open(file_path, 'r') as f:
        data = f.read()
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process one input file (--file1 OR --file2).")
    
    # Create a mutually exclusive group
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file1", help="Path to first input file")
    group.add_argument("--file2", help="Path to second input file")

    args = parser.parse_args()

    # Determine which file was provided
    file_path = args.file1 if args.file1 else args.file2
    process_file(file_path)