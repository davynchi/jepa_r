# Project instructions

- Use `uv` as the default Python package and environment installer. Install Python libraries with `uv` unless the user explicitly asks for another tool.
- Never launch an experiment based on an ambiguous reference such as "that one" or
  "the same". Ask the user to disambiguate instead of making a quick inference.
- Before launching any experiment, state the exact dataset, model, sampling/weighting
  method, richness functional, important hyperparameters, server, and GPU, then wait
  for the user's confirmation.

## Skill routing

When the user's request matches an available skill, invoke it. When in doubt, invoke the skill.

Key routing rules:

- Product ideas and brainstorming: `/office-hours`
- Strategy and scope: `/plan-ceo-review`
- Architecture: `/plan-eng-review`
- Design system and plan review: `/design-consultation` or `/plan-design-review`
- Full review pipeline: `/autoplan`
- Bugs and errors: `/investigate`
- QA and site behavior: `/qa` or `/qa-only`
- Code review and diff checks: `/review`
- Visual polish: `/design-review`
- Ship, deploy, or pull request: `/ship` or `/land-and-deploy`
- Save progress: `/context-save`
- Resume context: `/context-restore`
- Backlog-ready spec or issue: `/spec`
