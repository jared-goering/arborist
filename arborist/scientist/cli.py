"""CLI commands for the scientist layer."""

from __future__ import annotations

import json
import sys

import click

from arborist.scientist.journal import Journal


@click.group()
def scientist_cli() -> None:
    """Scientist — autonomous research agent for Arborist."""
    pass


@scientist_cli.command("run")
@click.option("--problem", "-p", required=True, help="Research problem description")
@click.option("--baseline", "-b", required=True, type=click.Path(exists=True), help="Path to baseline Python script")
@click.option("--metric", "-m", default="f1_macro", help="Metric to optimize")
@click.option("--budget", default=200, type=int, help="Total experiment budget")
@click.option("--max-rounds", default=10, type=int, help="Max observe-hypothesize-execute cycles")
@click.option("--model", default="openrouter/anthropic/claude-sonnet-4-6", help="Scientist LLM model")
@click.option("--mutator-model", default="openrouter/anthropic/claude-haiku-4-5", help="Mutator LLM model")
@click.option("--db", default="./scientist.db", help="Database path")
@click.option("--journal-dir", default="./scientist_journals", help="Journal storage directory")
@click.option("--memory-url", default=None, help="Ultramemory URL (default: http://localhost:8642)")
@click.option("--human-in-the-loop", is_flag=True, help="Pause between rounds for approval")
@click.option("--concurrency", default=3, type=int, help="Parallel experiments")
@click.option("--verbose/--quiet", default=True, help="Verbose logging")
def run_scientist(
    problem: str,
    baseline: str,
    metric: str,
    budget: int,
    max_rounds: int,
    model: str,
    mutator_model: str,
    db: str,
    journal_dir: str,
    memory_url: str | None,
    human_in_the_loop: bool,
    concurrency: int,
    verbose: bool,
) -> None:
    """Run an autonomous research session.

    The scientist observes model performance, generates hypotheses,
    designs experiments, executes them via Arborist tree search,
    and analyzes results in a loop.
    """
    from arborist.executors.shell import ShellExecutor
    from arborist.scientist.scientist import Scientist

    # Build executor from baseline script
    executor = ShellExecutor(
        command=f"python {baseline} {{config_path}}",
        timeout=300,
    )

    sci = Scientist(
        problem=problem,
        executor=executor,
        score=metric,
        metric_name=metric,
        max_rounds=max_rounds,
        total_budget=budget,
        model=model,
        mutator_model=mutator_model,
        db_path=db,
        journal_dir=journal_dir,
        memory_url=memory_url or "http://localhost:8642",
        human_in_the_loop=human_in_the_loop,
        verbose=verbose,
    )

    click.echo(f"Starting scientist session: {sci.session_id}")
    click.echo(f"Problem: {problem}")
    click.echo(f"Baseline: {baseline}")
    click.echo(f"Budget: {budget} experiments, {max_rounds} rounds")
    click.echo("")

    result = sci.run()

    click.echo("")
    click.echo(result.summary())
    click.echo(f"\nSession ID: {result.session_id}")
    click.echo(f"View journal: arborist scientist journal --session {result.session_id}")


@scientist_cli.command("resume")
@click.option("--session", "-s", required=True, help="Session ID to resume")
@click.option("--journal-dir", default="./scientist_journals", help="Journal storage directory")
def resume_session(session: str, journal_dir: str) -> None:
    """Resume a scientist session.

    Note: Full resume requires the Python API to restore the executor.
    This command shows how to resume programmatically.
    """
    journal = Journal(session_id=session, journal_dir=journal_dir, memory_url=None)
    entries = journal.entries

    if not entries:
        click.echo(f"No journal entries found for session {session}")
        sys.exit(1)

    click.echo(f"Session {session}: {len(entries)} journal entries")
    click.echo("")
    click.echo("To resume this session, use the Python API:")
    click.echo("")
    click.echo("  from arborist.scientist import Scientist")
    click.echo(f"  sci = Scientist(")
    click.echo(f"      problem='...',")
    click.echo(f"      executor=my_executor,")
    click.echo(f"      score='metric_name',")
    click.echo(f"  )")
    click.echo(f"  # Session state is preserved in the journal")
    click.echo(f"  result = sci.run()")


@scientist_cli.command("journal")
@click.option("--session", "-s", required=True, help="Session ID")
@click.option("--journal-dir", default="./scientist_journals", help="Journal storage directory")
@click.option("--format", "-f", "fmt", type=click.Choice(["text", "json"]), default="text")
def show_journal(session: str, journal_dir: str, fmt: str) -> None:
    """View the research journal for a session."""
    journal = Journal(session_id=session, journal_dir=journal_dir, memory_url=None)
    entries = journal.entries

    if not entries:
        click.echo(f"No journal entries found for session {session}")
        return

    if fmt == "json":
        data = [e.to_dict() for e in entries]
        click.echo(json.dumps(data, indent=2))
    else:
        click.echo(f"Research Journal — Session {session}")
        click.echo(f"{'='*60}")
        click.echo("")

        outcomes = journal.get_hypothesis_outcomes()
        if outcomes:
            confirmed = sum(1 for o in outcomes.values() if o == "CONFIRMED")
            rejected = sum(1 for o in outcomes.values() if o == "REJECTED")
            inconclusive = sum(1 for o in outcomes.values() if o == "INCONCLUSIVE")
            click.echo(
                f"Hypotheses: {len(outcomes)} tested "
                f"({confirmed} confirmed, {rejected} rejected, "
                f"{inconclusive} inconclusive)"
            )
            click.echo("")

        for entry in entries:
            prefix = entry.entry_type.upper().ljust(12)
            hid = f"[{entry.hypothesis_id}] " if entry.hypothesis_id else ""
            outcome_str = f" -> {entry.outcome}" if entry.outcome else ""

            click.echo(f"  {entry.timestamp[:19]}  {prefix} {hid}{outcome_str}")
            click.echo(f"    {entry.content}")

            if entry.score_before is not None and entry.score_after is not None:
                delta = entry.score_after - entry.score_before
                click.echo(
                    f"    Score: {entry.score_before:.4f} -> "
                    f"{entry.score_after:.4f} ({delta:+.4f})"
                )
            click.echo("")

        dead_ends = journal.get_dead_ends()
        if dead_ends:
            click.echo("Dead Ends:")
            for de in dead_ends:
                click.echo(f"  - {de}")


@scientist_cli.command("hypotheses")
@click.option("--session", "-s", required=True, help="Session ID")
@click.option("--journal-dir", default="./scientist_journals", help="Journal storage directory")
@click.option("--status", type=click.Choice(["all", "confirmed", "rejected", "inconclusive"]), default="all")
def show_hypotheses(session: str, journal_dir: str, status: str) -> None:
    """View hypothesis history for a session."""
    journal = Journal(session_id=session, journal_dir=journal_dir, memory_url=None)
    entries = journal.entries
    outcomes = journal.get_hypothesis_outcomes()

    # Get hypothesis entries
    hyp_entries = [e for e in entries if e.entry_type == "hypothesis"]

    if not hyp_entries:
        click.echo(f"No hypotheses found for session {session}")
        return

    click.echo(f"Hypotheses — Session {session}")
    click.echo(f"{'='*60}")
    click.echo("")

    for entry in hyp_entries:
        hid = entry.hypothesis_id or "?"
        outcome = outcomes.get(hid, "PENDING")

        # Filter by status
        if status != "all" and outcome.lower() != status:
            continue

        symbol = {
            "CONFIRMED": "+", "REJECTED": "-",
            "INCONCLUSIVE": "?", "PENDING": " ",
        }.get(outcome, "?")

        category = entry.tags[0] if entry.tags else "?"

        click.echo(f"  [{symbol}] {hid}: {entry.content}")
        click.echo(f"      Category: {category}  |  Outcome: {outcome}")
        if entry.metadata.get("rationale"):
            click.echo(f"      Rationale: {entry.metadata['rationale']}")
        click.echo("")
