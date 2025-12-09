
# assistant

`assistant` is a multi-agent developer assistant built on top of [smolagents](https://github.com/huggingface/smolagents).

It exposes a single CLI entry point to the user, but internally coordinates four specialized roles:

- **Manager** – orchestration and convergence (human-in-the-loop)
- **Expert** – domain knowledge and constraints (human-in-the-loop)
- **Architect** – design, planning, and documentation (LLM agent)
- **Developer** – implementation and code quality (LLM agent)

The design is:

- **Human-centered**: Manager and Expert are real people steering the work.
- **Agent-augmented**: Architect and Developer are LLM-powered agents integrated via smolagents.
- **Language-agnostic**: Python-specific helpers plus a generic runtime for any project exposing a `justfile`.

---

## 1. Conceptual model and roles

This section is both the conceptual spec and the basis for `instructions` strings and human role guidelines.

### 1.1 Assistant (overall system)

The `assistant` is a multi-agent system composed of four cooperating roles:

- Manager (human)  
- Expert (human)  
- Architect (agent)  
- Developer (agent)  

The user interacts through a single entry point (CLI / chat). The **Manager** and **Expert** are humans who:

- Clarify objectives and constraints.
- Validate concepts.
- Decide when the work is “good enough”.

The **Architect** and **Developer** are LLM agents that:

- Propose designs and plans.
- Implement changes, run tests, and perform code quality checks.

#### Global principles

- Always clarify the user goal and constraints before doing deep work.
- Prefer small, explicit steps and short feedback cycles between humans and agents.
- Stop when the result is “good enough” for the stated constraints (time, quality, scope).
- Prefer reuse and refactoring over “big rewrites” unless the benefits and risks are clearly justified.

---

### 1.2 Manager (human-in-the-loop)

The **Manager** is a human orchestrator for how the other roles work on a given task.

**Responsibilities**

- Analyze the request (ticket, feature, bug, refactor, doc update) and, when useful, break it down into subtasks.
- Decide which role (Expert, Architect, Developer) should act next, and in which order.
- Maintain a global view of:
  - Scope and priorities
  - Time / iteration budget
  - Required quality level
  - Available tools and project commands (e.g. `just test`, `just cqa`, `just nb-html-all`)
- Monitor progress and ensure convergence:
  - Ask the Expert to clarify requirements or validate concepts.
  - Ask the Architect agent to refine, simplify, or adjust the plan.
  - Ask the Developer agent to implement, run tests, and fix issues.
- Decide when to stop exploration once the solution satisfies the goal and constraints.

**Default behavior (suggested workflow for a human Manager)**

1. Clarify the task and constraints (scope, “definition of done”, time budget).
2. Ask the Expert (human) to restate the problem, constraints, and hidden pitfalls.
3. Ask the Architect (agent) to propose a concrete plan or design, given the Expert’s input.
4. Ask the Developer (agent) to implement following the plan and run tests / linters / CQA.
5. Loop:
   - If tests or quality checks fail, send feedback back to the Developer and, if needed, the Architect.
   - If the solution is conceptually wrong, route it back through Expert and Architect.
6. Summarize the outcome and explicitly document any limitations, risks, or TODOs before closing the task.

---

### 1.3 Expert (human-in-the-loop)

The **Expert** is a human who provides domain knowledge and context.

**Responsibilities**

- Clarify the problem and the goals in domain terms (business, product, scientific, etc.).
- Ask targeted questions when requirements are ambiguous or incomplete.
- Identify domain constraints, known pitfalls, and best practices.
- Challenge and validate the conceptual solution proposed by the Architect and Developer agents.
- Flag when additional data, assumptions, or trade-offs must be made explicit.

**Outputs**

- A short, structured problem statement written in natural language.
- A list of constraints, edge cases, and domain-specific recommendations.
- Comments on whether the proposed design and implementation are conceptually correct, and where they might be unsafe or incomplete.

The Expert typically interacts via the CLI or chat interface, guided by prompts and checklists exposed by the `assistant` tools.

---

### 1.4 Architect (agent)

The **Architect** is an LLM agent that designs the solution under the Manager’s and Expert’s guidance.

**Responsibilities**

- Define the overall approach, components, responsibilities, and data flows.
- When relevant, structure the work as a sequence of steps or milestones.
- Use Jupyter notebooks with Markdown and PlantUML to:
  - Document the plan and architecture.
  - Explain assumptions, risks, and trade-offs.
- Keep the design simple and implementable within the constraints given by the Manager and Expert.
- Update the design when the Developer discovers new constraints.

**Outputs**

- A concise architecture description:
  - High-level overview
  - Components / modules
  - Data flows and interfaces
- A concrete implementation plan that the Developer agent can follow (ordered steps).
- Optional: generated or updated MyST notebooks and PlantUML diagrams, rendered to HTML.

The Architect does not decide alone when the design is “good enough”; the Manager and Expert validate and adjust.

---

### 1.5 Developer (agent)

The **Developer** is an LLM agent that implements the solution.

**Responsibilities**

- Write clean, maintainable code that follows the Architect’s plan and the project’s conventions.
- Use the project’s commands to run tests, linters, and code quality tools, for example:
  - `just test`
  - `just test-coverage`
  - `just cqa`
- Respond to feedback from:
  - Expert (correctness, domain reasoning)
  - Architect (design consistency and boundaries)
  - Manager (scope, priorities, time budget)
- Propose small, localized design adjustments when implementation reveals new constraints, and send them back to the Architect (and Manager) for validation.

**Outputs**

- Code changes implementing the current step of the plan.
- Test results and quality checks (pass/fail with logs).
- Short notes on:
  - What was implemented
  - What changed compared to the initial plan
  - Any remaining technical debt or TODOs

The Developer is intentionally constrained by the project runtimes (`python_runtime`, `generic_shell`) and does not run arbitrary shell commands.

---

## 2. Project structure

Target layout for the `assistant` package:

```text
src/
  assistant/
    __init__.py
    config.py          # Global config (paths, defaults, model names, timeouts)
    logging.py         # Logging setup, structured logs if desired
    types.py           # Shared dataclasses / TypedDicts for tasks, results, etc.

    smol/
      __init__.py
      agent_factory.py # Build smolagent Agent(s) with proper tools & roles
      tools_registry.py# Central registration of tools (including custom ones)
      memory.py        # Optional: conversation / project memory abstractions

    roles/
      __init__.py
      base.py          # BaseRole: common API for roles (run(), describe(), etc.)
      manager.py       # ManagerRole helpers (prompts, checklists for human manager)
      expert.py        # ExpertRole helpers (question templates, domain checklists)
      architect.py     # ArchitectRole: agent-facing design logic and prompts
      developer.py     # DeveloperRole: agent-facing implementation & CQA logic

    runtimes/
      __init__.py
      python_runtime.py  # Knows how to run tests/lint/etc. for Python projects
      generic_shell.py   # Generic: run `just test`, `just cqa`, etc. for any stack
      # later:
      # js_runtime.py, rust_runtime.py, etc.

    tools/
      __init__.py
      filesystem.py    # List/edit files, apply patches, read/write configs
      git.py           # git status/diff/commit helpers
      just.py          # Helpers to call `just test`, `just cqa`, etc.
      jupyter.py       # Helpers for jupytext, nbconvert, notebook discovery
      plantuml.py      # Render PlantUML with `plantuml` binary, manage includes
      osv.py           # Wrappers around `osv-scanner` (run, parse JSON if needed)

    jupyter/
      __init__.py
      notebooks.py     # High-level ops: "ensure notebook for task", "render to HTML"
      myst_support.py  # Utilities specific to md:myst + jupytext

    workflows/
      __init__.py
      feature_impl.py  # "Implement feature from spec" orchestration
      bugfix.py        # "Reproduce + fix bug" orchestration
      code_review.py   # "Review MR/PR" orchestration
      doc_update.py    # "Update docs / notebook" orchestration
      # Each workflow wires Manager+Expert+agents+runtimes in a specific pattern.

    cli/
      __init__.py
      main.py          # Typer/Rich CLI entrypoint (exposed as `assistant`)

tests/
  test_config.py
  test_smol_agent_factory.py
  roles/
    test_manager.py
    test_expert.py
    test_architect.py
    test_developer.py
  runtimes/
    test_python_runtime.py
    test_generic_shell.py
  tools/
    test_just.py
    test_jupyter.py
    test_plantuml.py
    test_osv.py
  workflows/
    test_feature_impl.py
    test_bugfix.py
    test_code_review.py
````

---

## 3. Mapping of responsibilities to modules

### 3.1 `assistant.roles.*`

Encapsulates the four roles as small, testable components.

* **Manager and Expert (human-in-the-loop)**:

  * `manager.py` and `expert.py` provide:

    * Prompt templates and checklists.
    * Helper functions to generate summaries/questions for the human Manager/Expert to review.
  * They do not run unattended; they always assume a human decision-maker.
* **Architect and Developer (agents)**:

  * `architect.py` and `developer.py` implement:

    * Stable interfaces for the smolagent tools to call.
    * Role-specific prompts, constraints, and validation logic.

Roles themselves do not depend directly on smolagents; they are domain services that smolagent tools call into.

---

### 3.2 `assistant.smol.*`

Bridges smolagents’ API with your roles, runtimes, and tools.

* `agent_factory.py`:

  * Builds a Manager+Expert+Architect+Developer setup for interactive sessions.
  * Provides an “Architect+Developer-only” agent for automated workflows where a human has already validated the requirements.
* `tools_registry.py`:

  * Collects all tools in one place (filesystem, git, just, jupyter, osv, plantuml, etc.).
  * Allows fine-grained control over which tools are exposed to which agent.
* `memory.py`:

  * Optional abstractions for state:

    * Conversation memory (turn history, last decision).
    * Project memory (current branch, active task, context files).

---

### 3.3 `assistant.runtimes.*`

Isolates language and tooling differences.

* `python_runtime.py`:

  * Knows how to:

    * Run tests: `uv run pytest` or `just test`
    * Run linters/formatters: `ruff`, etc.
    * Run type checkers: `pyrefly`, etc.
  * Exposes a stable interface to the Developer agent, e.g.:

    * `run_tests()`
    * `run_coverage()`
    * `run_cqa()`
* `generic_shell.py`:

  * Knows how to:

    * Run `just test`, `just test-coverage`, `just cqa` for any project with a `justfile`.
  * Allows the Developer agent to work on non-Python projects with a standardized interface.

The Developer agent only calls runtime interfaces, never raw shell commands.

---

### 3.4 `assistant.tools.*`

Thin, composable wrappers around external commands and file operations.

* `filesystem.py`:

  * Read/write files, list directories, apply patches.
* `git.py`:

  * Get repo root, status, diffs, and optionally prepare commit messages.
* `just.py`:

  * Run `just test`, `just test-coverage`, `just cqa` with captured output and exit codes.
* `jupyter.py`:

  * Convert `md:myst` → ipynb via jupytext.
  * Execute notebooks and export HTML via nbconvert.
* `plantuml.py`:

  * Invoke the `plantuml` native binary to render diagrams.
* `osv.py`:

  * Wrap `osv-scanner`:

    * Run scans (e.g. `osv-scanner scan source -r .`).
    * Optionally parse JSON and summarize vulnerabilities.

These functions are the natural candidates to be exposed as smolagent tools. The Manager and Expert can trigger them indirectly through workflows, while Architect/Developer agents can use them directly when allowed.

---

### 3.5 `assistant.jupyter.*`

Higher-level Jupyter integration.

* `notebooks.py`:

  * “Ensure there is a design notebook for this task or module.”
  * “Render all notebooks for this task to HTML.”
* `myst_support.py`:

  * Helpers around `md:myst` structure, front-matter, and `{code-cell}` blocks.

This keeps notebook machinery out of the core roles and workflows and makes it easy for the Architect agent to maintain up-to-date docs.

---

### 3.6 `assistant.workflows.*`

Defines high-level workflows as orchestrations of **human Manager/Expert** and **agent Architect/Developer**.

Examples:

* `feature_impl.py`:

  * Manager + Expert clarify the feature and constraints.
  * Architect agent proposes the design and plan.
  * Developer agent implements and runs tests/CQA.
  * Manager decides when to stop and what to merge.
* `bugfix.py`:

  * Manager describes the bug; Expert clarifies context and impact.
  * Developer agent reproduces and proposes a fix, Architect reviews impact if needed.
  * Manager validates and closes the bug.
* `code_review.py`:

  * Developer and Architect agents review diffs with framing from Manager/Expert.
  * Manager uses the summarized risks and suggestions to decide on approval.
* `doc_update.py`:

  * Architect + Developer update documentation and notebooks, while Manager and Expert validate correctness and adequacy.

Each workflow defines:

* Which roles are involved.
* Which steps require human decisions.
* Which runtime(s) and tools are used at each step.

---

### 3.7 `assistant.cli.main`

CLI entrypoint (e.g. implemented with Typer, click, or argparse).

Possible commands:

* `assistant run feature --task "…" --path /path/to/repo`
* `assistant run bugfix --description "…" --path .`
* `assistant review --path .`
* `assistant design --module assistant/runtimes/python_runtime.py`

Internally, the CLI:

* Locates the repo and its tooling (Python runtimes vs generic `just`).
* Constructs the appropriate workflow and Agent via `assistant.smol.agent_factory`.
* Guides the human Manager/Expert through the steps (questions, confirmations).
* Delegates design/implementation to the Architect and Developer agents where appropriate.

---

## 4. Interaction with a target project

### 4.1 Assumptions

The assistant expects, at minimum:

* A Git repository root.
* A `justfile` exposing (some subset of):

  * `just test`
  * `just test-coverage`
  * `just cqa`

For Python projects, it can additionally leverage:

* `uv` for dependency management.
* `ruff`, `pyrefly`, and `pytest` via the Python runtime.
* Jupyter notebooks in `md:myst` format, rendered to HTML via jupytext + nbconvert.

### 4.2 Typical workflow (conceptual, with human Manager/Expert)

1. **Manager** (human) runs a CLI command, for example:
   `assistant run feature --task "Add X to module Y" --path .`
2. Assistant asks the **Manager** and **Expert** clarifying questions (scope, constraints, edge cases).
3. **Expert** (human) provides a structured problem statement and domain constraints.
4. **Architect** (agent) proposes a design and implementation plan; optionally updates notebooks and PlantUML diagrams.
5. **Manager** reviews the plan with the **Expert**:

   * Adjust scope or constraints if needed.
   * Approve the plan or request changes.
6. **Developer** (agent):

   * Applies code changes via filesystem tools.
   * Runs tests and CQA via runtimes (`python_runtime` and/or `generic_shell`).
   * Reports results back to the Manager and Expert.
7. **Manager**:

   * Decides whether to iterate (ask Architect/Developer for modifications) or to accept the result.
   * Documents remaining TODOs, risks, or follow-up tasks.

The core idea is that the Manager and Expert remain in control, while the Architect and Developer agents do most of the mechanical design and implementation work under clear constraints.
