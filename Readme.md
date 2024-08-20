

# Project Overview

This project involves understanding, preprocessing, and analyzing conference paper data, along with generating AI-based reviews using LLMs (Large Language Models). The repository is structured to facilitate easy access to data, AI review generation, and a WebUI for interacting with the results.

## Data Structure and Preprocessing

All data is located in the `./data` directory, organized by conference. Each conference directory contains three subdirectories: `dev`, `test`, and `train`. 

Within each of these subdirectories:
- The `papers.json` file lists all the paper numbers within that folder. This file is essential for accessing specific papers from other directories.
- Additional subdirectories include `parsed_pdfs`, `pdfs`, `reviews`, and `reviews_llm`. 

**Note:** The `reviews_llm` folder is not present by default. It is created when you run the code to generate LLM reviews.

## Generating AI Reviews

The AI review generation code resides in the `./AI_generation` directory. This directory contains a `prompts.yaml` file, which includes templates used to prompt the LLM for generating reviews.

### Customizing the LLM Call

To tailor the review generation to your needs, you may need to modify the `llm_call` function in the `utils.py` file. 

- If using Azure OpenAI, ensure you set the following environment variables:
  - `OPENAI_API_KEY`
  - `OPENAI_ENDPOINT`
  - `OPENAI_VERSION`

### Generating Reviews

To generate AI reviews, use the following command:

```bash
python AI_generation/main.py conference_root_directory paper_number llm_name prompt_template_name
```

Replace the placeholders with the appropriate values for your use case.

## Accessing the WebUI

To interact with the data and AI-generated reviews via the WebUI:

1. From the root directory, start a local server by running the following command:

   ```bash
   python -m http.server 8000
   ```

2. Open your web browser and navigate to [http://localhost:8000/](http://localhost:8000/).

3. In the WebUI:
   - Select the conference name.
   - Click the 'Load Papers' button.
   - Select the desired paper number and click "Load Reviews" to view the reviews.

4. To view the paper's PDF:
   - Click on the "PAPER" tab located at the right end of the interface.

5. Finally, select the appropriate answers and view your score.

### NOTE: as of now the web UI contains code to display ony dev folder papers.