import pandas as pd
from typing import Optional, List, Union
from rich.box import MINIMAL, SIMPLE, SIMPLE_HEAD, SQUARE  # type: ignore
from rich.columns import Columns  # type: ignore
from rich.console import Console  # type: ignore
from rich.measure import Measurement  # type: ignore
from rich.table import Table  # type: ignore

from .models import (
    PercentChange,
    TimeGranularity,
)

console = Console()

COLORS = ["cyan", "magenta", "red", "green", "blue", "purple"]


class DataFramePrettify:
    """Create animated and pretty Pandas DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        The data you want to prettify
    metrics : List[str], optional
        List of metrics queried
    dimensions : List[str], optional
        List of dimensions queried
    time_constraint : str, optional
        Time constraint
    time_granularity : TimeGranularity, optional
        Time granularity
    time_comparison : TimeComparison, optional
        Time comparison
    order : List[str], optional
        Order by operations
    start_time : str, optional
        Start time
    end_time : str, optional
        End time
    """

    def __init__(  # noqa: D
        self,
        df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        dimensions: Optional[List[str]] = None,
        time_constraint: Optional[str] = None,
        time_granularity: Optional[TimeGranularity] = None,
        time_comparison: Optional[PercentChange] = None,
        order: Optional[List[str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)

        # For printing out query config
        self.metrics = metrics
        self.dimensions = dimensions
        self.time_constraint = time_constraint
        self.time_granularity = time_granularity
        self.time_comparison = time_comparison
        self.order = order
        self.start_time = start_time
        self.end_time = end_time

        self.table = Table(show_footer=False)
        self.table_centered = Columns((self.table,), align="center", expand=True)
        self.num_colors = len(COLORS)

        self.columns = self.df.columns
        self.rows = self.df.values

        self.num_rows, self.num_cols = self.df.shape

    # following methods borrowed from https://github.com/khuyentran1401/rich-dataframe/blob/master/rich_dataframe/rich_dataframe.py
    # ----------
    def _add_columns(self) -> None:
        for col in self.columns:
            self.table.add_column(str(col))

    def _add_rows(self) -> None:
        for row in self.rows:
            row = [str(item) for item in row]
            self.table.add_row(*list(row))

    def _move_text_to_right(self) -> None:
        for i in range(len(self.table.columns)):
            self.table.columns[i].justify = "right"

    def _add_random_color(self) -> None:
        for i in range(len(self.table.columns)):
            self.table.columns[i].header_style = COLORS[i % self.num_colors]

    def _add_style(self) -> None:
        for i in range(len(self.table.columns)):
            self.table.columns[i].style = "bold " + COLORS[i % self.num_colors]

    def _adjust_box(self) -> None:
        for box in [SIMPLE_HEAD, SIMPLE, MINIMAL, SQUARE]:
            self.table.box = box

    def _dim_row(self) -> None:
        self.table.row_styles = ["none", "dim"]

    def _adjust_border_color(self) -> None:
        self.table.border_style = "bright_yellow"

    def _change_width(self) -> None:
        original_width = Measurement.get(console, self.table).maximum
        width_ranges = [
            [original_width, console.width, 2],
            [console.width, original_width, -2],
            [original_width, 90, -2],
            [90, original_width + 1, 2],
        ]

        for width_range in width_ranges:
            for width in range(*width_range):
                self.table.width = width

            self.table.width = None

    # ----------

    def _add_caption(self) -> None:
        self.table.caption = f"[bold magenta not dim] {self.num_rows} rows[/bold magenta not dim] and [bold green not dim]{self.num_cols} columns[/bold green not dim] are shown above.\n"
        if self.metrics:
            self.table.caption += f"""{"Metrics" if len(self.metrics) > 1 else "Metric"} queried: {self.metrics}\n"""
        if self.dimensions:
            self.table.caption += (
                f"""{"Dimensions" if len(self.dimensions) > 1 else "dimension"}: {self.dimensions}\n"""
            )
        if self.time_constraint:
            self.table.caption += f"Time constraint: {self.time_constraint}\n"
        if self.time_granularity:
            self.table.caption += f"Time granularity: {self.time_granularity}\n"
        if self.time_comparison:
            self.table.caption += f"Time comparison: {self.time_comparison}\n"
        if self.start_time or self.end_time:
            self.table.caption += f"""Between start_time of {self.start_time if self.start_time else "-inf"} and end_time of {self.end_time if self.end_time else "inf"}\n"""
        if self.order:
            self.table.caption += f"Order: {list(self.order)}\n"

    def generate(self) -> Table:
        """Pretty Table Maker"""
        self._add_columns()
        self._add_rows()
        self._move_text_to_right()
        self._add_random_color()
        self._add_style()
        self._adjust_border_color()
        self._add_caption()
        return self.table


def prettify(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    dimensions: Optional[List[str]] = None,
    time_constraint: Optional[str] = None,
    time_granularity: Optional[TimeGranularity] = None,
    time_comparison: Optional[PercentChange] = None,
    order: Optional[List[str]] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> Union[Table, pd.DataFrame]:  # noqa: D
    """Create animated and pretty Pandas DataFrame

    Args:
        df (pd.DataFrame): Dataframe with query results
        metrics (Optional[List[str]], optional): List of metrics queried. Defaults to None.
        dimensions (Optional[List[str]], optional): List of dimensions. Defaults to None.
        time_constraint (Optional[str], optional): Time constraint on query. Defaults to None.
        time_granularity (Optional[TimeGranularity], optional): Time granularity on query. Defaults to None.
        time_comparison (Optional[PercentChange], optional): Time comparison on query. Defaults to None.
        order (Optional[List[str]], optional): Any orderings on query. Defaults to None.
        start_time (Optional[str], optional): Start time. Defaults to None.
        end_time (Optional[str], optional): End time. Defaults to None.
    """
    if isinstance(df, pd.DataFrame):
        pretty_df = DataFramePrettify(
            df,
            metrics,
            dimensions,
            time_constraint,
            time_granularity,
            time_comparison,
            order,
            start_time,
            end_time,
        ).generate()
        return pretty_df
    else:
        return df
