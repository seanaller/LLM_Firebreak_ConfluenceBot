# LLM_Firebreak

## Description

Repository to store the large language model (LLM) firebreak project of creating an LLM (using LangChain) to query documentation.
Initially the scope was to link directly to our confluence pages. 

However, in the interests of time and security, this proof of concept was created using local copies of Word Documents (exported confluence pages).

## Setup

To use this project, you will require an **Open AI API Key** that is defined in a `.env` file.

1. Clone the repository.
2. Ensure the environment pre-requisits are met (see the `requirements` file).
3. Create a `.env` file specifiying your Open AI API key. Note that this key should be defined as a string, variable named `OPEN_API_KEY`. see example below in README.
4. Add any documentation you want processed as part of the model into the `data/` folder. Note: this currently only supports Word Documents (`.docx`).
5. Run the Streamlit app on your local machine.

Note: You can alter the `temperature` of the LLM using the slider in the app. 

## Running the Streamit App and Documentation ChatBot

To run the Documentation Bot you will need to use Streamlit in local hosting mode.

Navigate to the cloned repository folder and (in terminal) run:

```bash
streamlit run confluence_bot.py
```

You should then be presented with a URL to your internally hosted Streamlit app.

## Example .env file
  
```python
OPEN_API_KEY='XXXXXXXXXX'
```
