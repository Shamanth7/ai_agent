# System Roles & Temperature Dynamics

Overview of contextual framing via system prompts and token sampling mechanics covered in `period2`.

---

## System Roles as Contextual Framing

The `system` role sets the social and functional relationship between the user and the agent. The exact same user input produces completely divergent outputs depending on the system context.

For instance, the statement `"I love you"` processed under:
- A **romantic partner** persona yields reciprocal emotional validation.
- A **corporate enterprise manager** persona triggers professional boundary enforcement and workplace communication protocols.

In software architecture, system prompts define specialized operational personas:
- **Lead System Architect**: Evaluates trade-offs, scalability bottlenecks, latency, and fault tolerance.
- **Frontend Developer**: Specializes in component reusability, DOM rendering lifecycle, and UI responsiveness.
- **Security Auditor**: Scans for injection vectors, privilege escalation, and zero-trust violations.

---

## Temperature Dynamics & Sampling Mechanics

The `temperature` hyperparameter ($T$) scales the raw logits ($z_i$) produced by the model's final linear projection layer prior to computing token probabilities via the softmax function:

$$P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

# System Roles & Temperature Control (`sys_temp.py`)

This script demonstrates how to shape an LLM's persona and output constraints using the `system` message role, as well as how to adjust response creativity using the `temperature` hyperparameter via the Groq SDK.

## Complete Code

```python
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("where is your api key")

client = Groq(api_key=my_api_key)

model = "openai/gpt-oss-120b"
role = "user"
prompt = "Suggest a name for my clothing company"

# SYSTEM: Defines the persona and strict output constraints
message_system = {
    "role": "system",
    "content": "You are a brand manager who suggests name for my company. name should be in one word."
}

# USER: The actual request submitted to the model
message = {
    "role": role,
    "content": prompt
}

messages = [message_system, message]

# Temperature controls randomness/creativity (range: [0, 2])
response = client.chat.completions.create(model=model, messages=messages, temperature=1)
# print(response)

print("#######################################")

answer = response.choices[0].message.content
print(answer)
```

## Step-by-Step Code Explanation

1. **Environment & Client Initialization**
   * `load_dotenv()` loads environment variables from your local `.env` file, and `os.getenv("GROQ_API_KEY")` retrieves the secret API key.
   * `if not my_api_key:` checks that the key exists before making any API call, raising a `ValueError` immediately if it is missing.
   * `client = Groq(api_key=my_api_key)` initializes the Groq SDK client.

2. **Defining the `system` Role (`message_system`)**
   * Unlike a standard user prompt, the `"system"` role sets the behavioral rules, persona, and formatting constraints for the entire conversation.
   * Here, it instructs the model to act as a **brand manager** and enforces a strict constraint: *"name should be in one word."*

3. **Constructing the `messages` Array**
   * `message` holds the `"user"` request (`"Suggest a name for my clothing company"`).
   * `messages = [message_system, message]` combines both dictionaries in sequence so the model reads its system instructions before processing the user's request.

4. **Controlling Randomness with `temperature`**
   * In `client.chat.completions.create(..., temperature=1)`, the `temperature` parameter (valid range `[0, 2]`) controls the creativity and variance of token selection:
     * **Lower values (e.g., `0.0` – `0.3`)**: Deterministic, safe, and repetitive outputs.
     * **Balanced value (`1.0`)**: Standard creative sampling—ideal for brainstorming brand names while still following the one-word rule.
     * **Higher values (e.g., `1.5` – `2.0`)**: Highly unpredictable or experimental outputs.

5. **Extracting the Response**
   * `response.choices[0].message.content` extracts only the generated text reply from the API response object and prints it to the console.

## How to Run

From inside the `hebdomad/period2` directory:

```bash
uv run python sys_temp.py

'''or best for all python program'''
python -m sys_temp
```

