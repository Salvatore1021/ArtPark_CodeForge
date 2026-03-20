# Models Directory

This directory is for storing local model files, embeddings, and checkpoints.

## Current Model Architecture

AURORA's intelligence comes from three model layers:

### 1. LLM Skill Extraction (Primary)
- **Model**: `gpt-4o-mini` (via OpenAI API)
- **Task**: Structured JSON extraction of skills from unstructured resume/JD text
- **Why**: Superior entity extraction vs regex; handles abbreviations, context, proficiency signals
- **Config**: Set `OPENAI_MODEL` in `.env` to swap models

### 2. Regex + Taxonomy Fallback (No API key needed)
- **Model**: Rule-based pattern matching over `data/skill_taxonomy.json`
- **Task**: Alias resolution and skill normalization without API dependency
- **Why**: Zero-latency, zero-cost fallback; works offline for demos

### 3. Bayesian Knowledge Tracing (BKT)
- **Model**: Hidden Markov Model with 4 parameters (p_init, p_learn, p_slip, p_guess)
- **Location**: `backend/knowledge_tracing.py`
- **Task**: Estimate per-skill mastery probability from performance history
- **Why**: Principled probabilistic model; original implementation; proven in ITS literature (Corbett & Anderson 1994)

## Swapping the LLM

To use a different LLM, update `.env`:

```bash
# Anthropic Claude via OpenAI-compatible proxy
OPENAI_API_KEY=your-anthropic-key
OPENAI_MODEL=claude-3-haiku-20240307
OPENAI_BASE_URL=https://api.anthropic.com/v1

# Local Ollama (Mistral/Llama3)
OPENAI_API_KEY=ollama
OPENAI_MODEL=mistral
OPENAI_BASE_URL=http://localhost:11434/v1
```

## Adding Embedding-Based Skill Matching (Future Enhancement)

To improve skill matching beyond exact/alias lookup, you can add a sentence embedding model:

```python
# Example: all-MiniLM-L6-v2 from sentence-transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
# Save embeddings for all catalog skills once, then cosine-match at inference
```

Place any `.bin`, `.pt`, or `.pkl` files here. They are gitignored by default.

## Fitting BKT Parameters from Real Data

If you collect performance logs from learners, you can fit per-skill BKT parameters
using maximum likelihood estimation. A starter EM fitting loop is available in
`notebooks/02_bkt_parameter_fitting.ipynb` (to be created).

**Reference:**  
Corbett, A. T., & Anderson, J. R. (1994). Knowledge Tracing: Modeling the Acquisition of Procedural Knowledge. *User Modeling and User-Adapted Interaction*, 4(4), 253–278.
