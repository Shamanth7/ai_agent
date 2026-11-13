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