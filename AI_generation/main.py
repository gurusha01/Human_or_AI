import sys
from generate_review import *

def main():
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 6:
        print("Usage: script.py <paper_address> <paper_number> <language_model> <prompt_template_name>")
        sys.exit(1)

    # Assign the command-line arguments to variables
    paper_address = sys.argv[1]
    paper_number = sys.argv[2]
    language_model = sys.argv[3]
    prompt_template_name = sys.argv[4]
    level = sys.argv[5]

    # Call the generate_review function with the arguments
    generate_review(paper_address, paper_number, language_model, prompt_template_name, level)

if __name__ == "__main__":
    paper_address = "data/acl_2017/train"
    paper_numbers = ["12", "16", "18", "19", "21", "26", "31", "56", "66", "67"]
    language_model = "gpt-4o-mini"
    levels = [1,3,4,5]
    for paper_number in paper_numbers[5:]:
        for level in levels:
            prompt_template_name = f'level_{level}_prompt'
            generate_review(paper_address, paper_number, language_model, prompt_template_name, level)
            print(f'{paper_number}, {level}')



    # main()

# python AI_generation/main.py data/acl_2017/dev 37 gpt-35-turbo paper_prompt_1
