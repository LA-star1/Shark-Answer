# Shark Answer

A-Level CIE exam answer generation system with subject-specific pipelines, multi-model verification, and bilingual output.

## Architecture

```
shark_answer/
├── app.py                    # FastAPI backend (main entry)
├── config.py                 # Configuration, subject routing, model tiers
├── providers/                # AI model provider modules
│   ├── base.py               # BaseProvider interface
│   ├── claude_provider.py    # Anthropic Claude
│   ├── openai_provider.py    # OpenAI GPT-4o
│   ├── gemini_provider.py    # Google Gemini
│   ├── openai_compat_provider.py  # DeepSeek, Qwen, Grok, MiniMax, Kimi
│   └── registry.py           # Provider registry + parallel calling
├── pipelines/                # Subject-specific answer pipelines
│   ├── pipeline_a_science.py # Physics, Chemistry, Biology, Math
│   ├── pipeline_b_essay.py   # Economics, Bio essays
│   ├── pipeline_c_cs.py      # Computer Science
│   ├── pipeline_d_practical.py # Physics practical prediction (skeleton)
│   └── router.py             # Question → pipeline routing
├── modules/                  # Shared modules
│   ├── explanation.py        # Answer explanation generation
│   └── examiner_profile.py   # Examiner preference profiles
├── knowledge_base/           # Mark scheme + examiner report storage
│   └── store.py
└── utils/
    ├── cost_tracker.py       # Token usage and cost tracking
    ├── math_verifier.py      # Sympy/numerical verification
    └── image_extractor.py    # Exam paper photo → questions
```

## Pipelines

| Pipeline | Subjects | Primary Models | Process |
|----------|----------|---------------|---------|
| A | Physics, Chemistry, Biology, Math, Further Math | Claude, GPT-4o, DeepSeek | 3 models solve → verify numerically → debate if disagreement → 5 method variants |
| B | Economics, Biology essays | Claude, GPT-4o, Gemini, DeepSeek, Qwen | 5 unique angles → full drafts → quality gate → humanize with writing personalities |
| C | Computer Science | Claude, GPT-4o, DeepSeek | Solve → execute code to verify → CIE pseudocode formatting → algorithm variants |
| D | Physics Practical | Claude, GPT-4o | Equipment extraction → cross-reference past papers → predict experiments (skeleton) |

## Setup

```bash
# 1. Clone and enter directory
cd shark-answer

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env with your API keys (minimum: ANTHROPIC_API_KEY or OPENAI_API_KEY)

# 5. Run the server
python run.py
```

Server starts at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

## API Endpoints

### `POST /api/solve` — Main endpoint
Upload exam paper photos and get A/A* answers.

```bash
curl -X POST http://localhost:8000/api/solve \
  -F "images=@exam_page1.jpg" \
  -F "images=@exam_page2.jpg" \
  -F "subject=physics" \
  -F "language=en" \
  -F "max_versions=3"
```

### `POST /api/practical/predict` — Pipeline D
Predict physics practical from equipment list.

### `GET /api/cost` — Cost tracking
View current session cost breakdown by model and subject.

### `GET /api/examiner/profiles` — Examiner profiles
List and manage examiner marking tendency profiles.

### `GET /api/health` — Health check

## Testing

```bash
# Unit tests (no API calls needed)
pytest test_quick.py -v

# Live tests with real API calls
pytest test_quick.py -v -m live

# Test a specific pipeline
pytest test_quick.py -v -k "pipeline_a"
```

## Configuration

### Subject → Pipeline Routing
Edit `config.py` `SUBJECT_PIPELINE_MAP` to change routing.

### Model Assignments
Edit `config.py` `PIPELINE_MODEL_CONFIG` to change which models are used per pipeline tier.

### Examiner Profiles
Create via API or edit `data/examiner_profiles/profiles.json`. Profiles adjust answer tone, depth, and style per examiner tendency.

### Bilingual Output
Set `language=zh` in API calls for Chinese output. Default is English.

## Cost Tracking
Every API call logs token usage and estimated cost. View via `GET /api/cost` or in server logs. Set `COST_BUDGET_WARNING_USD` in `.env` to get warnings.
