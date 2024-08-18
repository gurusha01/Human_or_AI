import sys
from generate_review import *

def main():
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 5:
        print("Usage: script.py <paper_address> <paper_number> <language_model> <prompt_template_name>")
        sys.exit(1)

    # Assign the command-line arguments to variables
    paper_address = sys.argv[1]
    paper_number = sys.argv[2]
    language_model = sys.argv[3]
    prompt_template_name = sys.argv[4]

    # Call the generate_review function with the arguments
    generate_review(paper_address, paper_number, language_model, prompt_template_name)

if __name__ == "__main__":
    main()

# python AI_generation/main.py data/acl_2017/dev 37 gpt-35-turbo paper_prompt_1
