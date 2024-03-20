## Building Advanced Retreival Augemented Generation based LLM Applications of Questions and Answers System

### Overview

Retrieval Augmented Generation (RAG) is a methodology that combines elements of information retrieval and natural language generation to improve the quality and relevance of generated text, particularly in the context of complex language tasks like question-answering, summarization, and text completion.

- **Blog post**: https://medium.com/@alextakele16/rag-based-llm-applications-for-contract-advisory-31bc77b81e72
  
- **GitHub repository**: https://github.com/alextakele/Advanced_RAG-LLM_Question-Answer
  
- **Interactive notebook**: https://github.com/alextakele/Advanced_RAG-LLM_Question-Answer/RAG_Q&A.ipynb
  
In this github code, we will learn how to:

- Develop a retrieval augmented generation (RAG) based LLM application from scratch.
- Scale the major components (load, chunk, embed, index, Retriev,Generate, and Evaluate etc.) in my application.
- Evaluate different configurations of my application to optimize for both per-component (ex. retrieval_score) and overall performance (quality_score).
- Implement LLM hybrid routing approach to bridge the gap b/w OSS and closed LLMs.
- Serve the application in a highly scalable and available manner.
<br>
![image](https://github.com/alextakele/Advanced_RAG-LLM_Q-A/assets/67500303/9c4e2608-540f-4f5e-83c9-89c96bb873c8)

## Setup
### API keys
I am using [OpenAI](https://platform.openai.com/docs/models/) to access ChatGPT models like `gpt-3.5-turbo`, `gpt-4`, etc. 


#### List of Tech Stacks I used and you should read more before using this github code.

[LLM: OpenAI GPT-3.5-turb](https://www.smashingmagazine.com/2024/01/guide-retrieval-augmented-generation-language-models/)
[Langchain : Build LLM applications](https://www.analyticsvidhya.com/blog/2023/07/building-llm-powered-applications-with-langchain/)

[RAGAS: Evaluation Framework](https://docs.ragas.io/en/latest/howtos/integrations/llamaindex.html)

[FAISS: Vector store](https://faiss.ai/index.html)

[text-embedding-ada-002: OpenAIEmbeddings](https://platform.openai.com/docs/guides/embeddings)

[RecursiveCharacterTextSplitter: Text Splitter](https://api.python.langchain.com/en/latest/text_splitter/langchain.text_splitter.RecursiveCharacterTextSplitter.html)
### Repository
```bash
git clone https://github.com/alextakele/Advanced_RAG-LLM_Question-Answer
```
### Installation 
```bash
git clone https://github.com/ray-project/llm-applications.git .
```
### Environment

Then set up the environment correctly by specifying the values in your `.env` file,
and installing the dependencies:

```bash
pip install --user -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$PWD
pre-commit install
pre-commit autoupdate
```
### Credentials
```bash
touch .env
# Add environment variables to .env
OPENAI_API_BASE="https://api.openai.com/v1"
OPENAI_API_KEY=""  # https://platform.openai.com/account/api-keys
source .env
```
# Hi, I am the author of this project! 👋

## Authors Github
- Alexander  Mengesha
- [Github](https://www.github.com/alextakele)
- [Linkedin](https://www.linkedin.com/in/alextakele)

# Acknowledgements
I am deeply thankful to the team at 10 Academy (https://10academy.org/) for their essential and constructive guidance throughout the intensive training program focused on generative AI, Web3, and data engineering.

I appreciate their crucial role in providing supportive tutoring, valuable assistance, insightful comments, constructive criticisms, and professional advice consistently from the inception of the training until the culmination of all eleven projects. Their timely contributions have been instrumental in my accomplishments during this period of professional development.




