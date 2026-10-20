# Agentic AI Prompt Optimization Engine

LangChain + Streamlit app that iteratively evaluates and refines user prompts.

## How the agent loop works
1. **Evaluator** (LLM + Pydantic parser) scores the prompt on clarity, specificity, context, constraints, output format.
2. **Refiner** rewrites the prompt using the critique.
3. Repeat until score >= target or max iterations; the best-scoring version is returned.
4. Optionally generates answers from the original vs. optimized prompt for comparison.

## Run
```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add GROQ_API_KEY (free key at console.groq.com)
streamlit run app.py
```
For local models: install Ollama, run `ollama pull llama3.2`, pick "Ollama (local)" in the sidebar.
