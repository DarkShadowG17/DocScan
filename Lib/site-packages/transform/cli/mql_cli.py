import json
from pathlib import Path

import click
from click.shell_completion import ShellComplete, BashComplete, FishComplete, ZshComplete
import shellingham

import time
import textwrap
import pandas as pd
from halo import Halo
from rich.console import Console  # type: ignore
from typing import Dict, List, Optional
from datetime import datetime
from packaging.version import parse

from log_symbols import LogSymbols
from termcolor import colored
from update_checker import UpdateChecker

from ..pretty_pandas import prettify

from ..interfaces import DWValidationType

from .cli_context import CLIContext
from .cli_utils import (
    metrics_and_dimensions_options,
    query_options,
    config_validation_options,
    async_options,
    no_active_mql_servers,
    validate_query_args,
    enable_debug_log_file,
    get_query_logs_url_for_id,
    query_observability_options,
    exception_handler,
    start_end_time_options,
)
from ..models import (
    CacheMode,
    PercentChange,
    PrintStyle,
    Query,
    HealthReportStatus,
    DEFAULT_LIMIT,
    TimeGranularity,
)
from .. import PACKAGE_NAME, __version__
from ..constants import CLI_UPDATE_LAST_CHECKED, MAX_TIME_BEFORE_UPDATE, TRANSFORM_PROD_APP
from ..utils import get_cli_config_path

pass_config = click.make_pass_decorator(CLIContext, ensure=True)

MAX_LIST_OBJECT_ELEMENTS = 5

console = Console()


@click.group()
@click.option("-v", "--verbose", is_flag=True)
@click.option("-y", "--skip-confirm", is_flag=True)
@click.option("--debug-log-file", is_flag=True)
@pass_config
def cli(config: CLIContext, verbose: bool, skip_confirm: bool, debug_log_file: bool) -> None:  # noqa: D
    config.verbose = verbose
    config.skip_confirm = skip_confirm

    if debug_log_file:
        enable_debug_log_file()
    check_for_update = False
    last_updated = config.cli_state.get_config_value(CLI_UPDATE_LAST_CHECKED)
    if last_updated is None or last_updated == "":
        check_for_update = True
    else:
        try:
            check_for_update = (datetime.now().timestamp() - float(last_updated)) > MAX_TIME_BEFORE_UPDATE
        except ValueError:
            pass
    if check_for_update:
        checker = UpdateChecker()
        result = checker.check(PACKAGE_NAME, __version__)
        config.cli_state.set_config_value(CLI_UPDATE_LAST_CHECKED, str(datetime.now().timestamp()))
        # result is None when an update was not found or a failure occurred
        if result:
            # Note: As the CLI and API stabilize, we can be a little less aggressive about forcing upgrade
            # For now, we should do so to minimize the likelihood of out of date CLIs causing unexpected behavior
            click.secho(
                "Warning: A new version of the MQL CLI is available.",
                bold=True,
                fg="red",
            )

            click.echo(
                f"Please update to version {result.available_version}, released {result.release_date} by running:\n\t$ pip install --upgrade {PACKAGE_NAME}",
            )
        else:
            click.echo(f"Using the latest version of {PACKAGE_NAME} ({__version__}).")


@cli.command()
@pass_config
def contact(config: CLIContext) -> None:
    """Instructions for how to contact Transform for help."""
    click.echo(
        "🛎  We're here to help. Contact a friendly Transformer at support@transformdata.io or over a shared Slack if your org has one set up.",
    )


@cli.command()
@click.option(
    "-k",
    "--api-key",
    type=str,
    help="Your Transform API key used for authentication",
)
@click.option(
    "-o",
    "--mql-override",
    type=str,
    help="Override the regular MQL server URL",
    default=None,
)
@click.option(
    "-r",
    "--reset",
    type=bool,
    help="Reset the mql setup.",
    is_flag=True,
    default=False,
)
@click.option(
    "--remove-override",
    type=bool,
    help="Remove override on the regular MQL server URL",
    is_flag=True,
    default=False,
)
@click.option(
    "--unpin-model",
    type=bool,
    help="Remove pinned model ID and revert to the primary model for your organization",
    is_flag=True,
    default=False,
)
@pass_config
@exception_handler
def setup(
    config: CLIContext,
    api_key: Optional[str],
    mql_override: Optional[str],
    remove_override: bool,
    reset: bool,
    unpin_model: bool,
) -> None:
    """Guides user through CLI setup."""

    if api_key:
        config.api_key = api_key

    # Allow user to override their mql server via a flag
    if (remove_override or reset) and config.mql_server_url_config_override:
        config.mql_server_url_config_override = ""

    # Allow user to override their mql server via a flag
    if mql_override:
        config.mql_server_url_config_override = mql_override

    if unpin_model:
        config.unpin_model()

    if mql_override or remove_override or api_key:
        exit()

    # If the user doesn't yet have auth creds, wants to override existing cred, or token is expired
    if not config.just_authenticated:
        if not config.is_authenticated or (
            not config.just_authenticated
            and click.confirm(
                f"We've found existing credentials for {config.user.user_name} within the {config.org.name} organization.\nWould you like to provide new credentials, thus resetting the existing credentials?"
            )
        ):
            config.prompt_authentication()

    # If the user has an existing MQL server override, let them know and ask if they want it removed
    if config.mql_server_url_config_override and click.confirm(
        config.mql_server_url_status + "\nWould you like to remove the MQL server URL override shown above?"
    ):
        config.mql_server_url_config_override = ""

    # Allow user to override their mql server via a flag
    if mql_override:
        config.mql_server_url_config_override = mql_override

    if config.just_authenticated:
        config.cli_state.reset()


@cli.command()
@pass_config
@exception_handler
def identify(config: CLIContext) -> None:
    """Identify the currently authenticated user."""
    config.identify()


@cli.command()
@pass_config
def version(config: CLIContext) -> None:
    """Print the current version of the MQL CLI."""
    click.echo(__version__)


@cli.command()
@pass_config
@exception_handler
@click.option("--force", "-f", is_flag=True, default=False)
def drop_cache(config: CLIContext, force: bool) -> None:
    """Drop the MQL cache. Only necessary if there is evidence cache corruption."""
    drop_confirm_msg = (
        click.style("WARNING", fg="yellow") + ": This can be an expensive operation.\nDo you want to drop the cache?"
    )
    if not force and not click.confirm(
        drop_confirm_msg,
        abort=False,
    ):
        click.echo("❌ Aborted dropping the MQL cache.")
        exit()

    spinner = Halo(text="Initiating drop-cache query... this can take a little while", spinner="dots")
    spinner.start()

    resp = config.mql.drop_cache()
    if not resp:
        raise click.ClickException("‼️ Failed to drop the cache.")
    spinner.succeed("💥 Successfully dropped MQL cache.")


@cli.command()
@pass_config
@exception_handler
@click.option(
    "--metric-name",
    required=False,
    type=str,
    help="Invalidate caches related to this metric",
)
@click.option("--skip-confirm", "-s", is_flag=True, default=False)
def invalidate_caches(config: CLIContext, metric_name: Optional[str] = None, skip_confirm: bool = False) -> None:
    """Invalidates caches

    Invalidates caches related to the provided metric_name, if no metric is passed invalidate all caches.
    """
    cache_msg = f"caches related to {metric_name}" if metric_name else "all caches"
    confirm_msg = (
        click.style("WARNING", fg="yellow")
        + f": Once invalidated, caches will be rebuilt from sources automatically, but until then, queries will be significantly slower.\nDo you want to continue to invalidate {cache_msg}?"
    )
    if not skip_confirm and not click.confirm(
        confirm_msg,
        abort=False,
    ):
        click.echo(f"❌ Abort invalidate {cache_msg} command.")
        exit()

    spinner = Halo(text="Initiating cache invalidation query...", spinner="dots")
    spinner.start()

    resp = config.mql.invalidate_cache_for_metric(metric_name) if metric_name else config.mql.invalidate_all_caches()
    if not resp:
        raise click.ClickException("‼️ Failed to invalidate caches.")
    spinner.succeed(f"💥 Successfully invalidated {cache_msg}.")


@cli.command()
@pass_config
@exception_handler
def ping(config: CLIContext) -> None:
    """Perform basic HTTP health check against configured MQL server."""
    tic = time.perf_counter()
    resp = config.mql.ping()
    click.echo(
        f"🏓 Received HTTP {resp.status_code} code from MQL in {time.perf_counter() - tic:0.4f} seconds. \n {resp.text}"
    )


@cli.command()
@pass_config
@exception_handler
def list_servers(config: CLIContext) -> None:
    """Lists available MQL servers."""
    servers = config.mql.list_servers()
    if len(servers) == 0:
        no_active_mql_servers(config.org.name)
        exit()

    click.echo(f"🖨  We've found {len(servers)} MQL servers for the {config.org.name} organization. ⭐️ - Primary.")

    for s in servers:
        click.echo(f"• {'⭐️ ' if s.is_org_default else ''}{click.style(s.name, bold=True, fg='yellow')}: {s.url}")


@cli.command()
@pass_config
@exception_handler
def health_report(config: CLIContext) -> None:
    """Completes a health check on MQL servers."""
    spinner = Halo(text="Checking health of MQL servers...", spinner="dots")
    spinner.start()
    servers = config.mql.health_report()
    if len(servers) == 0:
        spinner.fail()
        no_active_mql_servers(config.org.name)
        exit()
    spinner.succeed("Successfully built health report!")

    click.echo(
        f"🏥 Health Report for {len(servers)} MQL Servers at {config.org.name}",
    )
    for s in servers:
        if s.status == HealthReportStatus.FAIL:
            click.echo(
                f"• ❌ {click.style(s.name, bold=True, fg=('red'))}:  Unable to connect to MQL server at url {s.url}."
            )
        else:
            click.echo(f"• {click.style(s.name, bold=True, fg='yellow')}: {s.url} running commit {s.version}")
            for h in s.servers:
                if h.status == HealthReportStatus.SUCCESS.value:
                    click.echo(f"  • ✅ {click.style(h.name, bold=True, fg=('green'))}: No Errors")
                else:
                    click.echo(f"  • ❌ {click.style(h.name, bold=True, fg=('red'))}:  {h.error_message}")


@cli.command()
@async_options
@metrics_and_dimensions_options
@query_options
@pass_config
@click.pass_context
@click.option("--as-table", required=False, type=str, help="To write a table in your data warehouse")
@click.option(
    "--csv",
    type=click.File("wb"),
    required=False,
    help="Provide filepath for dataframe output to csv",
)
@click.option(
    "--explain",
    is_flag=True,
    required=False,
    default=False,
    help="In the query output, show the query that was executed against the data warehouse",
)
@click.option(
    "--decimals",
    required=False,
    default=2,
    help="Choose the number of decimal places to round for the numerical values",
)
@query_observability_options
@exception_handler
def query(
    ctx: click.core.Context,
    config: CLIContext,
    detach: bool,
    timeout: int,
    metrics: List[str],
    dimensions: List[str],
    where: Optional[str] = None,
    time_constraint: Optional[str] = None,
    time_granularity: Optional[TimeGranularity] = None,
    time_comparison: Optional[PercentChange] = None,
    order: Optional[List[str]] = None,
    limit: Optional[str] = None,
    cache_mode: Optional[CacheMode] = None,
    config_dir: Optional[str] = None,
    force_commit: bool = False,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    as_table: Optional[str] = None,
    csv: Optional[click.utils.LazyFile] = None,
    explain: bool = False,
    decimals: int = 2,
    web: bool = False,
    debug: bool = False,
    allow_dynamic_cache: bool = True,
    style: Optional[PrintStyle] = None,
    trim: Optional[bool] = None,
) -> None:
    """Create a new MQL query, polls for completion and assembles a DataFrame from the response."""
    validate_query_args(limit)

    # Note: calling config.resolve_query_model_key may initiate a dialog if the user has a pinned model
    # Purposefully pulling this out before the spinner
    model_key = config.resolve_query_model_key(config_dir, force_commit)

    start = time.time()
    spinner = Halo(text="Initiating query…", spinner="dots")
    spinner.start()
    query_id = config.mql.create_query(
        model_key_id=model_key.id if model_key else None,
        metrics=metrics,
        dimensions=dimensions,
        where=where,
        time_constraint=time_constraint,
        time_granularity=time_granularity.value if time_granularity else None,
        time_comparison=time_comparison.value if time_comparison else None,
        order=order,
        limit=limit or DEFAULT_LIMIT,
        cache_mode=cache_mode.value if cache_mode else None,
        as_table=as_table,
        allow_dynamic_cache=allow_dynamic_cache,
        start_time=start_time,
        end_time=end_time,
        trim=trim,
    ).query_id

    spinner.succeed(f"Query initialized: {query_id}")
    if detach:
        return
    elif explain:
        sql = config.mql_client.explain_query_sql(query_id)
        if debug:
            ctx.invoke(stream_query_logs, query_id=query_id)
        click.echo(f"\n🔎 SQL executed for successful query (remove --explain to see data): \n{sql}")
        exit()

    spinner.start("Retrieving results")

    df: Optional[pd.DataFrame] = None
    try:
        df = config.mql.get_query_dataframe(query_id, timeout)
        final_message = f"Success 🦄 - query completed after {time.time() - start:.2f} seconds"
        final_symbol = LogSymbols.SUCCESS
    except Exception as e:
        final_message = str(e)
        final_symbol = LogSymbols.ERROR

    if debug:
        ctx.invoke(stream_query_logs, query_id=query_id)

    spinner.stop_and_persist(symbol=final_symbol.value, text=final_message)

    warnings = config.mql.get_query_status(query_id).warnings
    for warning in warnings:
        click.echo(click.style("WARNING: ", fg="yellow") + warning)

    # Show the data if returned successfully
    if df is not None:
        if df.empty:
            click.echo("Successful MQL query returned an empty result set.")
        elif csv is not None:
            df.to_csv(csv, index=False)
            click.echo(f"Successfully written query output to {csv.name}")
        else:
            if style == PrintStyle.PRETTY:
                # see pretty_pandas.py for df printing logic
                pretty_df = prettify(
                    df=df,
                    metrics=metrics,
                    dimensions=dimensions,
                    time_constraint=time_constraint,
                    time_granularity=time_granularity,
                    time_comparison=time_comparison,
                    order=order,
                    start_time=start_time,
                    end_time=end_time,
                )

                # must use rich print to print rich table as opposed to click.echo
                console.print(pretty_df, justify="center")
            else:
                df_to_print = None
                if parse(pd.__version__) >= parse("1.1.0"):
                    df_to_print = df.to_markdown(index=False, floatfmt=f".{decimals}f")
                else:
                    df_to_print = df.to_string(index=False, float_format=lambda x: format(x, f".{decimals}f"))

                if style == PrintStyle.STANDARD:
                    # must use rich print to print rich table as opposed to click.echo
                    console.print(df_to_print, justify="center")
                else:
                    click.echo(df_to_print)

    if web:
        click.launch(get_query_logs_url_for_id(query_id))

    if final_symbol == LogSymbols.ERROR:
        exit(1)


@cli.command()
@click.option(
    "--materialization-name",
    required=True,
    type=str,
    help="Name of materialization to drop",
)
@start_end_time_options
@async_options
@query_observability_options
@pass_config
@click.pass_context
@exception_handler
def drop_materialization(
    ctx: click.core.Context,
    config: CLIContext,
    detach: bool,
    timeout: int,
    materialization_name: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    config_dir: Optional[str] = None,
    web: bool = False,
    debug: bool = False,
) -> None:
    """***NEW*** Create a new MQL drop materialization query, polls for completion"""
    # Note: calling config.resolve_query_model_key may initiate a dialog if the user has a pinned model
    # Purposefully pulling this out before the spinner
    model_key = config.resolve_query_model_key(config_dir)

    if not config.skip_confirm and start_time is None:
        if not click.confirm(
            "You haven't provided a start_time. This means we will drop the materialization from the beginning of time."
            "This may be expensive. Are you sure you want to continue?"
        ):
            click.echo("Exiting")
            return

    start = time.time()
    spinner = Halo(text="Initiating drop materialization query…", spinner="dots")
    spinner.start()

    query_id = config.mql.drop_materialization(
        model_key_id=model_key.id if model_key else None,
        materialization_name=materialization_name,
        start_time=start_time,
        end_time=end_time,
    ).query_id

    spinner.succeed(f"Query initialized: {query_id}")
    if detach:
        return

    spinner.start("Waiting for query")
    try:
        config.mql.poll_for_query_completion(query_id, timeout)
        final_message = f"Success 🦄 - drop materialization query completed after {time.time() - start:.2f} seconds."
        final_symbol = LogSymbols.SUCCESS
    except Exception as e:
        final_message = str(e)
        final_symbol = LogSymbols.ERROR

    if debug:
        ctx.invoke(stream_query_logs, query_id=query_id)

    spinner.stop_and_persist(symbol=final_symbol.value, text=final_message)

    if web:
        click.launch(get_query_logs_url_for_id(query_id))

    if final_symbol == LogSymbols.ERROR:
        exit(1)


@cli.command()
@click.option(
    "--materialization-name",
    required=True,
    type=str,
    help="Name of materialization to materialize",
)
@start_end_time_options
@click.option(
    "--output-table",
    required=False,
    type=str,
    help="Write materialized result to specified table of format '<schema>.<table_name>'",
)
@click.option("--force", "-f", is_flag=True, default=False)
@async_options
@query_observability_options
@pass_config
@click.pass_context
@exception_handler
def materialize(
    ctx: click.core.Context,
    config: CLIContext,
    detach: bool,
    timeout: int,
    materialization_name: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    config_dir: Optional[str] = None,
    output_table: Optional[str] = None,
    force: bool = False,
    web: bool = False,
    debug: bool = False,
) -> None:
    """Create a new MQL materialization query, polls for completion and returns materialized table id"""
    # Note: calling config.resolve_query_model_key may initiate a dialog if the user has a pinned model
    # Purposefully pulling this out before the spinner
    model_key = config.resolve_query_model_key(config_dir)

    if not config.skip_confirm and start_time is None:
        if not click.confirm(
            "You haven't provided a start_time. This means we will materialize from the beginning of time. This may be expensive. Are you sure you want to continue?"
        ):
            click.echo("Exiting")
            return

    start = time.time()
    spinner = Halo(text="Initiating materialization query…", spinner="dots")
    spinner.start()

    query_id = config.mql.create_materialization(
        model_key_id=model_key.id if model_key else None,
        materialization_name=materialization_name,
        start_time=start_time,
        end_time=end_time,
        output_table=output_table,
        force=force,
    ).query_id

    spinner.succeed(f"Query initialized: {query_id}")
    if detach:
        return

    spinner.start("Waiting for query")
    try:
        schema, table = config.mql.get_materialization_result(query_id, timeout)
        materialized_at = f"Materialized table: {schema}.{table}" if schema is not None else ""
        final_message = (
            f"Success 🦄 - materialize query completed after {time.time() - start:.2f} seconds." f"{materialized_at}"
        )
        final_symbol = LogSymbols.SUCCESS
    except Exception as e:
        final_message = str(e)
        final_symbol = LogSymbols.ERROR

    if debug:
        ctx.invoke(stream_query_logs, query_id=query_id)

    spinner.stop_and_persist(symbol=final_symbol.value, text=final_message)

    if web:
        click.launch(get_query_logs_url_for_id(query_id))

    if final_symbol == LogSymbols.ERROR:
        exit(1)


@cli.command()
@pass_config
@exception_handler
@click.option("--search", required=False, type=str, help="Filter available materializations by this search term")
def list_materializations(config: CLIContext, search: Optional[str] = None) -> None:
    """List the materializations for the Organization with their available metrics and dimensions."""

    spinner = Halo(text="Looking for all available materializations...", spinner="dots")
    spinner.start()

    model_key = config.resolve_query_model_key()
    materializations = config.mql.list_materializations(model_key.id if model_key else None)
    if not materializations:
        spinner.fail("List of materializations unavailable.")
        exit()

    filter_msg = ""
    if search is not None:
        count = len(materializations)
        materializations = [m for m in materializations if search.lower() in m.name.lower()]
        filter_msg = f" matching `{search}`, of a total of {count} available"

    spinner.succeed(
        f"🌱 We've found {len(materializations)} materializations for the {config.org.name} organization{filter_msg}."
    )
    click.echo(
        'The list below shows materializations in the format of "materialization: list of available metrics, then dimensions"'
    )
    for m in materializations:
        dimensions = sorted(m.dimensions)
        metrics = sorted(m.metrics)
        click.echo(
            f"• {click.style(m.name, bold=True, fg='green')}:"
            + (f"\nMetrics: {', '.join(metrics[:MAX_LIST_OBJECT_ELEMENTS])}")
            + (
                f" and {len(metrics) - MAX_LIST_OBJECT_ELEMENTS} more"
                if len(metrics) > MAX_LIST_OBJECT_ELEMENTS
                else ""
            )
            + (f"\nDimensions: {', '.join(dimensions[:MAX_LIST_OBJECT_ELEMENTS])}")
            + (
                f" and {len(dimensions) - MAX_LIST_OBJECT_ELEMENTS} more"
                if len(dimensions) > MAX_LIST_OBJECT_ELEMENTS
                else ""
            )
            + (f"\ndestination table: {m.destination_table or m.name}")
        )


@cli.command()
@pass_config
@click.option("--metric-name", required=True, type=str, help="Metric that is associated with the dimension")
@click.option("--dimension-name", required=True, type=str, help="Dimension to query")
@click.option("--page-number", required=False, type=int, hidden=True)
@click.option("--page-size", required=False, type=int, hidden=True)
@exception_handler
def get_dimension_values(
    config: CLIContext,
    dimension_name: str,
    metric_name: str,
    page_number: Optional[int] = None,
    page_size: Optional[int] = None,
) -> None:
    """List all dimension values that are queryable through materialized tables."""

    spinner = Halo(
        text=f"Retrieving dimension values for dimension {dimension_name} of metric {metric_name}...", spinner="dots"
    )
    spinner.start()

    model_key = config.resolve_query_model_key()
    try:
        dim_vals = sorted(
            config.mql.get_dimension_values(
                metric_name,
                dimension_name,
                model_key.id if model_key else None,
                page_number=page_number,
                page_size=page_size,
            )
        )
    except Exception as e:
        spinner.fail()
        click.echo(
            textwrap.dedent(
                f"""\
                ❌ Failed to query dimension values for dimension {dimension_name} of metric {metric_name}.
                💡 Ensure that this query can be done via pre-materialized results by creating a
                       materialization that includes the desired metric and dimension

                    ERROR: {str(e)}
                """
            )
        )
        exit(1)

    spinner.succeed(
        f"🌱 We've found {len(dim_vals)} dimension values for dimension {dimension_name} of metric {metric_name}."
    )
    for dim_val in dim_vals:
        click.echo(f"• {click.style(dim_val, bold=True, fg='green')}")


@cli.command()
@pass_config
@click.option("--search", required=False, type=str, help="Filter available metrics by this search term")
@click.option("--show-all-dims", is_flag=True, default=False, help="Show all dimensions associated with a metric.")
@exception_handler
def list_metrics(config: CLIContext, show_all_dims: bool = False, search: Optional[str] = None) -> None:
    """List the metrics for the Organization with their available dimensions.

    Automatically truncates long lists of dimensions, pass --show-all-dims to see all.
    """

    # Note: calling config.resolve_query_model_key may initiate a dialog if the user has a pinned model
    # Purposefully pulling this out before the spinner

    spinner = Halo(text="Looking for all available metrics...", spinner="dots")
    spinner.start()

    model_key = config.resolve_query_model_key()
    metrics = sorted(config.mql.list_metrics(model_key.id if model_key else None).values(), key=lambda m: m.name)
    if not metrics:
        spinner.fail("List of metrics unavailable.")

    filter_msg = ""
    if search is not None:
        num_org_metrics = len(metrics)
        metrics = [m for m in metrics if search.lower() in m.name.lower()]
        filter_msg = f" matching `{search}`, of a total of {num_org_metrics} available"

    spinner.succeed(f"🌱 We've found {len(metrics)} metrics for the {config.org.name} organization{filter_msg}.")
    click.echo('The list below shows metrics in the format of "metric_name: list of available dimensions"')
    num_dims_to_show = MAX_LIST_OBJECT_ELEMENTS
    for m in metrics:
        # sort dimensions by whether they're local first(if / then global else local) then the dim name
        dimensions = sorted(map(lambda d: d.name, filter(lambda d: "/" not in d.name, m.dimensions))) + sorted(
            map(lambda d: d.name, filter(lambda d: "/" in d.name, m.dimensions))
        )
        if show_all_dims:
            num_dims_to_show = len(dimensions)
        click.echo(
            f"• {click.style(m.name, bold=True, fg='green')}: {', '.join(dimensions[:num_dims_to_show])}"
            + (f" and {len(dimensions) - num_dims_to_show} more" if len(dimensions) > num_dims_to_show else "")
        )


@cli.command()
@pass_config
@click.option("--metric-names", required=False, help="Metrics to filter dimensions by (intersection)")
@exception_handler
def list_dimensions(config: CLIContext, metric_names: Optional[str] = None) -> None:
    """List all unique dimensions for the Organization."""
    metric_names_input: Optional[List[str]] = None
    if metric_names:
        metric_names_input = [m.strip() for m in metric_names.split(",")]

    spinner = Halo(
        text="Looking for all available dimensions...",
        spinner="dots",
    )
    spinner.start()

    model_key = config.resolve_query_model_key()
    dimensions = sorted(
        config.mql.list_dimensions(
            metric_names=metric_names_input, model_key_id=model_key.id if model_key else None
        ).values(),
        key=lambda d: d.name,
    )
    if not dimensions:
        spinner.fail("List of dimensions unavailable.")

    spinner.succeed(f"🌱 We've found {len(dimensions)} unique dimensions for the {config.org.name} organization.")
    for d in dimensions:
        click.echo(f"• {click.style(d.name, bold=True, fg='green')}")


@cli.command()
@pass_config
@click.option("--query-id", required=True, type=str, help="Query ID to stream logs for")
@exception_handler
def stream_query_logs(
    config: CLIContext,
    query_id: str,
) -> None:
    """Retrieve queries from mql server"""
    line_number = 0
    query_status_resp = None
    while query_status_resp is None or not query_status_resp.is_complete:
        logs, line_count = config.mql_client.get_logs_by_line(query_id, line_number)
        line_number += line_count
        query_status_resp = config.mql_client.get_query_status(query_id)
        color_map: dict = {"INFO": "green", "ERROR": "red", "WARNING": "yellow"}
        if line_count > 0:
            lines: List[dict] = [json.loads(line) for line in logs.split("\n")[0:-1]]
            formatted_lines = [
                f'[{click.style(str(line.get("level")), bold=True, fg=color_map.get(line.get("level"), "green"))}]'
                f'[{line.get("asctime")}]: {line.get("message")}'
                for line in lines
            ]
            click.echo("\n".join(formatted_lines))
        time.sleep(1)

    if query_status_resp is not None and query_status_resp.error:
        click.echo("Error message: " + query_status_resp.error)


@cli.command()
@click.option("--active-only", type=bool, default=False, help="Return active queries only")
@click.option("--limit", required=False, type=int, help="Limit the number of queries retrieved: syntax is --limit 100")
@pass_config
@exception_handler
def list_queries(
    config: CLIContext,
    active_only: bool,
    limit: Optional[int] = None,
) -> None:
    """Retrieve queries from mql server"""

    active_query_str = "active" if active_only else "non-active"
    spinner = Halo(text=f"Looking for all {active_query_str} queries...", spinner="dots")
    spinner.start()

    queries = config.mql.list_queries(active_only=active_only, limit=limit)

    spinner.succeed(f"🌱 We've found {len(queries)} {active_query_str} queries. Grouped by branch-commit,")

    # group queries by branch-commit
    grouped_queries: Dict[str, List[Query]] = {}
    for q in queries:
        branch_commit = f"{q.branch}-{q.commit}"
        if branch_commit not in grouped_queries:
            grouped_queries[branch_commit] = []
        grouped_queries[branch_commit].append(q)

    for group in grouped_queries:
        click.echo(click.style(group, bold=True, fg=("green")))
        for q in grouped_queries[group]:
            click.echo(f"• query_id={q.id}, status={q.status.value}")


def _run_dw_validations(config: CLIContext, type: DWValidationType, model_id: Optional[int] = None) -> List[str]:
    """Helper handles the calling of data warehouse issue generating functions"""

    spinner = Halo(
        text=f"Validating {type.name.lower()} elements on model {model_id} against data warehouse...", spinner="dots"
    )
    spinner.start()
    issues = config.mql.get_data_warehouse_issues(type=type, model_key_id=model_id)
    if issues is not None and len(issues) == 0:
        spinner.succeed(
            f"🎉 Finished validating {type.name.lower()} elements on model {model_id} against data warehouse, no issues found"
        )
    else:
        spinner.fail(
            f"😬 Issues found when validating {type.name.lower()} elements on model {model_id} against data warehouse"
        )
    return issues


def _data_warehouse_validations_runner(config: CLIContext, model_id: Optional[int] = None) -> bool:
    """Helper which calls the individual data warehouse validations to run and prints collected issues"""
    # the only reason we do this is so that the model id is available in terminal output
    if model_id is None and config.current_model is not None:
        model_id = config.current_model.id

    issues: List[str] = []
    issues += _run_dw_validations(config=config, type=DWValidationType.DATA_SOURCE, model_id=model_id)
    issues += _run_dw_validations(config=config, type=DWValidationType.DIMENSION, model_id=model_id)
    issues += _run_dw_validations(config=config, type=DWValidationType.METRIC, model_id=model_id)

    print(colored("\n".join(issues), "red"))
    return len(issues) == 0


@cli.command()
@click.option("--model-id", type=int, required=False, help="Specific model id to run")
@pass_config
@exception_handler
def data_warehouse_validations(config: CLIContext, model_id: Optional[int] = None) -> None:
    """Run data warehouse validations for a model, defaults to current model"""
    _data_warehouse_validations_runner(config=config, model_id=model_id)


@cli.command()
@config_validation_options
@click.option(
    "--skip-dw",
    is_flag=True,
    default=False,
    help="If specified, skips the data warehouse validations",
)
@pass_config
@exception_handler
def validate_configs(config: CLIContext, config_dir: str, skip_dw: bool = False) -> None:
    """Validate yaml configs found in specified config directory"""
    spinner = Halo(text=f"Running parsing and semantic validations for configs in '{config_dir}'...", spinner="dots")
    spinner.start()

    # Currently if this fails, an exception is thrown, and the calling function handles it with a decorator
    model = config.mql.validate_configs(config_dir=config_dir)
    spinner.succeed(
        f"🦄 Successfully validated parsing and semantics of configs for commit ({model.commit}) in repo ({model.repository}), on branch ({model.branch})"
    )

    if not skip_dw:
        _data_warehouse_validations_runner(config=config, model_id=model.id)


@cli.command()
@config_validation_options
@click.option("--pin", type=bool, required=False, help="Whether to pin the model or not")
@click.option("--force-primary", is_flag=True, default=None, help="Make this model current")
@click.option("--dev-ui", is_flag=True, default=False, help="See the model in the Transform app")
@click.option(
    "--skip-dw",
    is_flag=True,
    default=False,
    help="If specified, skips the data warehouse validations",
)
@pass_config
@click.pass_context
def commit_configs(
    ctx: click.core.Context,
    config: CLIContext,
    config_dir: str,
    pin: Optional[bool] = None,
    force_primary: Optional[bool] = None,
    dev_ui: bool = False,
    skip_dw: bool = False,
) -> None:
    """Commit yaml configs found in specified config directory"""
    spinner = Halo(text=f"Running parsing and semantic validations for configs in '{config_dir}'...", spinner="dots")
    spinner.start()

    # Currently if this fails, an exception is thrown, and the calling function handles it with a decorator
    model = config.mql.commit_configs(config_dir=config_dir)
    spinner.succeed(
        f"🦄 Successfully validated parsing and semantics of configs for commit ({model.commit}) in repo ({model.repository}), on branch ({model.branch})"
    )

    if not skip_dw:
        success = _data_warehouse_validations_runner(config=config, model_id=model.id)
        if not success:
            return

    if force_primary:
        promote_spinner = Halo(text="Promoting commited model to be primary", spinner="dots")
        promote_spinner.start()
        config.mql.promote_model(model)
        promote_spinner.succeed(f"🚀 Success! Model {model.id} is now the primary model for your org")

    if pin is None:
        if click.confirm("📌 Would you like to pin this model commit for future MQL queries?"):
            pin = True

    if pin:
        ctx.invoke(pin_model, model_id=model.id)
    else:
        click.echo(
            textwrap.dedent(
                f"""\
                💡 Pin these configs in the future:

                    mql pin-model --model-id {model.id}
                """
            )
        )
    dev_ui_link = f"{TRANSFORM_PROD_APP}/?model-id={model.id}"
    if dev_ui:
        click.launch(dev_ui_link)
    else:
        click.echo(
            textwrap.dedent(
                f"""\
                📊 Test the model in the Transform App:

                    {dev_ui_link}
                """
            )
        )


@cli.command()
@click.option("--model-id", type=int, required=True, help="Model id to pin for local queries")
@pass_config
@exception_handler
def pin_model(config: CLIContext, model_id: int) -> None:
    """Pin a model id from configs that are already committed to the MQL Server"""
    try:
        config.pinned_model_id = model_id
    except Exception:
        click.echo(
            textwrap.dedent(
                f"""\
                ❌ Failed to pin model {model_id}. This may not be a valid model id.
                💡 Commit and pin new configs:

                    mql commit-configs --config-dir <path> --pin true

                """
            )
        )
        exit(1)

    click.echo(
        textwrap.dedent(
            f"""\
            ✅ Successfully pinned model {model_id}.
            💡 Unpin this model in the future:

                mql unpin-model

            """
        )
    )


@cli.command()
@pass_config
@exception_handler
def unpin_model(config: CLIContext) -> None:
    """Unpin a model id"""
    config.unpin_model()
    click.echo(
        textwrap.dedent(
            """\
                ✅ Successfully unpinned model and reverted to default configs.
                💡 Commit and pin new configs:

                    mql commit-configs --config-dir <path> --pin true

            """
        )
    )


@cli.command()
@pass_config
@exception_handler
def latest_mql_image(config: CLIContext) -> None:
    """Outputs the latest MQL server image details"""

    server_image = config.mql.latest_mql_image()
    click.echo(
        textwrap.dedent(
            f"""\
                ✅ Successfully retrieved the latest MQL server image,
                    Service Name: {server_image.service_name}
                    Version Hash: {server_image.version_hash}
                    Download URL: {server_image.download_url}

            """
        )
    )


_shell_properties = {
    "bash": (".bashrc", BashComplete),
    "zsh": (".zshrc", ZshComplete),
    "fish": ("", FishComplete),
}


@cli.command()
@pass_config
@exception_handler
def install_completion(config: CLIContext) -> None:
    """Install command completion for the MQL CLI"""
    try:
        shell_name, _ = shellingham.detect_shell()
    except shellingham.ShellDetectionFailure:
        click.echo("Not running in a shell. Re-run `mql install completion` in a shell to install completion.")
    if shell_name not in _shell_properties.keys():
        click.echo(f"Completion for {shell_name} not yet supported.")
        return

    shell_config_file, complete_cls = _shell_properties[shell_name]
    shell_complete: ShellComplete = complete_cls(None, {}, "mql", "_MQL_COMPLETE")
    try:
        completion_script = shell_complete.source()
    except RuntimeError as e:
        click.echo(e)
        return

    if shell_name == "fish":
        fish_config_path = Path.home() / ".config" / "fish" / "completions"
        fish_config_path.mkdir(parents=True, exist_ok=True)
        script_path = fish_config_path / "mql.fish"
        with script_path.open("w") as file:
            file.write(completion_script)
        click.echo("Added the completion script to ~/.config/fish/completions/mql.fish")
        click.echo("To finish installation, restart your terminal.")
        return

    transform_config = get_cli_config_path()
    transform_config.mkdir(parents=True, exist_ok=True)
    script_path = transform_config / f"{shell_name}_completion.sh"
    with script_path.open("w") as file:
        file.write(completion_script)

    shell_config_path = Path.home() / shell_config_file

    if not shell_config_path.is_file():
        click.echo(f"Error: {shell_config_file} not found in home directory.")
        return

    config_line = f"source {str(script_path.absolute())}"

    already_installed = False
    with shell_config_path.open() as file:
        for line in file.readlines():
            if config_line in line:
                already_installed = True
                break
    if already_installed:
        click.echo(f"Completion script already in {shell_config_file}.")
        return
    if not click.confirm(
        f"To finish the install, we will add the completion script to {shell_config_file}. Would you like to continue?"
    ):
        click.echo(
            f"You can finish the installation manually by adding `{config_line}` to {shell_config_file} and restarting your shell."
        )
        return
    with shell_config_path.open("a") as file:
        file.write(f"\n{config_line}\n")
        click.echo(f"Added completion script to {shell_config_file}!")
        click.echo("To finish installation and enable completion, restart your terminal.")


if __name__ == "__main__":
    cli()
