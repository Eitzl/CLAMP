# CLAMP — project instructions

## Design philosophy

Two things to hold in mind on every phase of this project, from the project owner directly:

> Make things as simple as possible, but no simpler. (Einstein's razor)

And, from a critique of AI-agent-driven open source contributions ([r/ClaudeCode](https://www.reddit.com/r/ClaudeCode/comments/1rgqv2z/please_stop_spamming_oss_projects_with_useless/)): don't generate complexity, abstraction, or churn just because an agent is capable of producing it. More code, more configurability, more statistical rigor, and more custom infrastructure are not virtues by themselves — they're costs, to be paid only when the problem actually requires them.

Concretely, for this project:

- **Prefer established external tools over custom-built pipeline components.** Before designing or implementing a custom clustering/partitioning/alignment step, ask whether an existing, validated tool (e.g. MMseqs2 for sequence clustering/homology partitioning) already does the job. Reimplementing something a purpose-built tool already does well is exactly the kind of unnecessary complexity to avoid.
- **Favor the simplest architecture that satisfies the actual requirement.** When a design doc proposes an extra layer, a shared module, or a bespoke statistical procedure, it should have a concrete justification tied to a real requirement — "the original authors did it this way" or "it might be useful later" is not sufficient justification on its own if a simpler version satisfies the same requirement.
- **Don't over-invest in rigor the project doesn't need.** If compute cost isn't a real constraint (this project can use Colab/the NRP cluster for expensive runs), don't add complexity purely to save compute. Conversely, don't add elaborate statistical machinery (multi-seed aggregation policies, paired bootstraps, decision-gate logic, etc.) to a step that could instead be a straightforward, judgment-based decision — reserve that rigor for places where the decision is genuinely close or its correctness is genuinely load-bearing.
- **When in doubt, ask whether the added complexity is paying for something real**, and say so explicitly in the design doc if it is. If it isn't, cut it.
