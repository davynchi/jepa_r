# Project instructions

- Use `uv` as the default Python package and environment installer. Install Python libraries with `uv` unless the user explicitly asks for another tool.

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
