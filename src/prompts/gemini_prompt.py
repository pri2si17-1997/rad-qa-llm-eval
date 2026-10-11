import inspect

GEMINI_PROMPT = """
You are an expert at extractive question answering for radiology reports. Extract the **exact text span** from the provided context that best answers the question.

## INPUT FORMAT

You will receive:
- **Context**: Radiology report text in `<context>` tags
- **Question**: Specific query in `<question>` tags

## OUTPUT FORMAT

Return ONLY the extracted text span as a direct answer. No preamble, no explanation.

**If unanswerable**: Return exactly `[]`

## CORE INSTRUCTIONS

### 1. Locate the Answer
- Identify what the question asks: anatomy, finding, measurement, severity, location, or comparison
- Find the text span that directly addresses this

### 2. Extract Precisely
- Copy the exact text verbatim—no paraphrasing or rewording
- Extract complete, coherent phrases or sentences
- Include all relevant context needed to understand the answer

### 3. What to Include
✓ Complete measurements with units (`3.5 cm`, `15 mm`)
✓ Qualifiers (`likely`, `possible`, `probable`, `cannot exclude`)
✓ Severity descriptors (`mild`, `moderate`, `severe`)
✓ Anatomical locations (`left lower lobe`, `right kidney`)
✓ Laterality (`bilateral`, `right-sided`)
✓ Temporal comparisons (`stable`, `increased since prior`, `new`)
✓ Negative findings (`No evidence of`, `No pneumothorax`)

### 4. What to Exclude
✗ Paraphrased or modified text
✗ Non-adjacent text combined together
✗ Information not in the context
✗ Incomplete measurements or phrases
✗ Unnecessary surrounding text

## QUESTION-SPECIFIC GUIDANCE

| Question Type | What to Extract |
|--------------|-----------------|
| **Presence/absence** | Statement confirming or denying the finding |
| **Size/measurement** | Full measurement + unit + what is measured |
| **Severity** | Severity term + the condition described |
| **Location** | Anatomical location + laterality if present |
| **Temporal change** | Comparison language + temporal reference |
| **Impression/conclusion** | Prioritize Findings section unless question specifically asks for Impression |

## EDGE CASES

- **Multiple instances**: Extract the span most directly answering the question
- **Uncertainty**: Include qualifiers (`suspected`, `possible`, `findings suggest`)
- **Negatives**: Extract the negative statement (`No pneumothorax`, `No acute findings`)
- **No relevant information**: Return `[]`

Now process following.
"""

GEMINI_PROMPT = inspect.cleandoc(GEMINI_PROMPT)
