import pandas as pd
import arrow

from typing import List, Optional, Any, Dict
from enum import Enum
from dataclasses import dataclass, field

INF = "inf"
NEGATIVE_ONE = "-1"
DEFAULT_LIMIT = "100"


@dataclass(frozen=True)
class ModelKey:
    """A model key. Represents a given point in time for the Transform configs."""

    id: int
    organization_id: int
    repository: str
    branch: str
    commit: str
    created_at_ts: str
    is_current: bool

    @property
    def created_at(self) -> str:
        """Converts a system timestamp to a human readable string, ex: 'an hour ago'"""
        return arrow.get(self.created_at_ts).humanize()

    @classmethod
    def from_gql_dict(cls, _input: Dict[str, str]) -> "ModelKey":  # noqa: D
        return cls(
            id=int(_input["id"]),
            organization_id=int(_input["organizationId"]),
            repository=_input["gitRepo"],
            branch=_input["gitBranch"],
            commit=_input["gitCommit"],
            created_at_ts=_input["createdAt"],
            is_current=bool(_input["isCurrent"]),
        )

    @classmethod
    def from_snake_dict(cls, _input: Dict[str, str]) -> "ModelKey":  # noqa: D
        return cls(
            id=int(_input["id"]),
            organization_id=int(_input["organization_id"]),
            repository=_input["repository"],
            branch=_input["branch"],
            commit=_input["commit"],
            created_at_ts=_input["created_at"],
            is_current=bool(_input["is_current"]),
        )


@dataclass(frozen=True)
class Organization:
    """Lightweight object to represent a Transform Organization."""

    id: int
    name: str
    primary_config_repo: str
    primary_config_branch: str

    @classmethod
    def from_gql_dict(cls, _input: Dict[str, str]) -> "Organization":  # noqa: D
        return cls(
            id=int(_input["id"]),
            name=_input["name"],
            primary_config_repo=_input["primaryConfigRepo"],
            primary_config_branch=_input["primaryConfigBranch"],
        )


@dataclass(frozen=True)
class User:
    """Lightweight object to represent a Transform User."""

    id: int
    user_name: str
    email: str
    mql_server_url: str

    @classmethod
    def from_gql_dict(cls, _input: Dict[str, str]) -> "User":  # noqa: D
        return cls(
            id=int(_input["id"]),
            user_name=_input["userName"],
            email=_input["email"],
            mql_server_url=_input["mqlServerUrl"],
        )


@dataclass(frozen=True)
class UserState:
    """Aggregation of the core models necessary for most operations"""

    user: User
    organization: Organization
    current_model: Optional[ModelKey]


@dataclass(frozen=True)
class Materialization:
    """Object to represent a Metric."""

    name: str
    metrics: List[str]
    dimensions: List[str]
    destination_table: Optional[str]

    @classmethod
    def from_gql_dict(cls, _input: Dict[str, Any]) -> "Materialization":  # type: ignore [misc] # noqa: D
        destination_table = None
        if "destination_table" in _input:
            destination_table = _input["destination_table"]
        return cls(
            name=_input["name"],
            metrics=_input["metrics"],
            dimensions=_input["dimensions"],
            destination_table=destination_table,
        )


@dataclass(frozen=True)
class UserOwner:
    """Object to represent a user owner."""

    user_name: str
    is_org_admin: bool
    email: str

    @classmethod
    def from_gql_dict(cls, _input: Dict[str, Any]) -> "UserOwner":  # type: ignore [misc] # noqa: D
        return cls(
            user_name=_input["userName"],
            is_org_admin=_input["isOrgAdmin"],
            email=_input["email"],
        )


@dataclass(frozen=True)
class TeamOwner:
    """Object to represent a team owner."""

    name: str
    users: List[UserOwner]

    @classmethod
    def from_gql_dict(cls, _input: Dict[str, Any]) -> "TeamOwner":  # type: ignore [misc] # noqa: D
        return cls(
            name=_input["name"],
            users=[UserOwner.from_gql_dict(u) for u in _input["memberUsers"]],
        )


class MetricType(Enum):
    """Metric type Enum."""

    MEASURE_PROXY = "measure_proxy"
    RATIO = "ratio"
    EXPR = "expr"
    CUMULATIVE = "cumulative"


@dataclass
class MetricTypeParams:
    """Object to represent the type params of a Metric."""

    numerator: Optional[str]
    denominator: Optional[str]
    expr: Optional[str]
    window: Optional[str]

    @classmethod
    def from_gql_dict(cls, _input: Dict[str, Any]) -> "MetricTypeParams":  # type: ignore [misc] # noqa: D
        return cls(
            numerator=_input["numerator"],
            denominator=_input["denominator"],
            expr=_input["expr"],
            window=_input["window"],
        )


@dataclass
class Metric:
    """Object to represent a Metric."""

    name: str
    measures: List[str]
    metric_type: MetricType
    type_params: MetricTypeParams
    constraint: Optional[str]
    dimensions: List["Dimension"]
    team_owners: Optional[List[TeamOwner]] = None
    user_owners: Optional[List[UserOwner]] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    tier: Optional[int] = None
    value_format: Optional[str] = None
    is_additive: Optional[bool] = None

    @classmethod
    def from_gql_dict(cls, _input: Dict[str, Any]) -> "Metric":  # type: ignore [misc] # noqa: D
        return cls(
            name=_input["name"],
            measures=_input["measures"],
            metric_type=MetricType(_input["type"].lower()),
            type_params=MetricTypeParams.from_gql_dict(_input["typeParams"]),
            constraint=_input["constraint"],
            dimensions=list({d["name"]: Dimension.from_gql_dict(d) for d in _input["dimensionObjects"]}.values()),
        )

    def input_metadata(self, metadata: Dict[str, Any]) -> None:  # type: ignore[misc]
        """Input additional backend metadata."""
        self.description = metadata["description"]
        self.display_name = metadata["displayName"]
        self.tier = metadata["tier"]
        self.value_format = metadata["valueFormat"] if "valueFormat" in metadata else None
        self.is_additive = metadata["isAdditive"] if "isAdditive" in metadata else None
        self.team_owners = [TeamOwner.from_gql_dict(t["team"]) for t in metadata["teamOwners"]]
        self.user_owners = [UserOwner.from_gql_dict(u["user"]) for u in metadata["userOwners"]]


class TimeGranularity(Enum):
    """For time dimensions, the smallest possible difference between to time values.

    Needed for calculating adjacency when merging 2 different time ranges.
    """

    # Names are used in parameters to DATE_TRUC, so don't change them.
    # Values are used to convert user supplied string to enums.
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class PrintStyle(Enum):
    """For use in choosing which style/format to print query results"""

    PLAIN = "plain"
    STANDARD = "standard"
    PRETTY = "pretty"


class MqlQueryStatus(Enum):
    """The status of queries submitted for execution in the query manager."""

    # Created and waiting to run.
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESSFUL = "SUCCESSFUL"
    # Handled exception prevented a query from being successful.
    FAILED = "FAILED"
    # Unhandled exception prevented a query from being successful.
    UNHANDLED_EXCEPTION = "UNHANDLED_EXCEPTION"
    # The given query ID is not known to exist - possibly expired.
    UNKNOWN = "UNKNOWN"
    TIMEOUT = "TIMEOUT"


class MqlQueryResultSource(Enum):
    """The source of a query's result"""

    DYNAMIC_CACHE = "DYNAMIC_CACHE"
    DW_MATERIALIZATION = "DW_MATERIALIZATION"
    FAST_CACHE = "FAST_CACHE"
    METRICFLOW = "METRICFLOW"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NOT_SPECIFIED = "NOT_SPECIFIED"
    DIMENSION_VALUES_CACHE = "DIMENSION_VALUES_CACHE"
    MATERIALIZED_QUERY_CACHE = "MATERIALIZED_QUERY_CACHE"


@dataclass(frozen=True)
class MqlQueryResultSeries:  # noqa: D
    value: float
    delta: float
    pct_change: float


@dataclass(frozen=True)
class MqlQueryStatusResp:  # noqa: D
    query_id: str
    status: MqlQueryStatus
    error: Optional[str]
    sql: Optional[str]

    chart_value_min: Optional[float]
    chart_value_max: Optional[float]
    result: Optional[List[MqlQueryResultSeries]]
    result_source: Optional[MqlQueryResultSource]
    result_primary_time_granularity: Optional[TimeGranularity]
    warnings: List[str]

    @property
    def is_complete(self) -> bool:
        """These statuses indicate the Query has completed execution"""
        return self.status in [
            None,
            MqlQueryStatus.SUCCESSFUL,
            MqlQueryStatus.FAILED,
            MqlQueryStatus.UNHANDLED_EXCEPTION,
        ]

    @property
    def is_successful(self) -> bool:
        """Indicate whether the Query has completed successfully"""
        return self.status == MqlQueryStatus.SUCCESSFUL

    @property
    def is_failed(self) -> bool:
        """Indicate whether the Query has completed successfully"""
        return self.status in [MqlQueryStatus.FAILED, MqlQueryStatus.UNHANDLED_EXCEPTION]


@dataclass(frozen=True)
class QueryResult:
    """Object for query results with a cursor"""

    df: pd.DataFrame
    cursor: Optional[int] = None


class CacheMode(Enum):
    """Different CacheMode options."""

    READ = "r"  # use cached datasets but do not write new datasets to cache
    READWRITE = "rw"  # use cached datasets and write new datasets to cache
    WRITE = "w"  # ignore cached datatsets and write new datasets to cache
    IGNORE = "i"  # ignore cache completely, only use configuration files

    @classmethod
    def from_string(cls, string_value: str) -> "CacheMode":  # noqa: D
        if string_value == "wr":  # convenience rewire
            string_value = "rw"
        return cls(string_value)

    @classmethod
    def default(cls) -> "CacheMode":  # noqa: D
        return CacheMode.READWRITE


@dataclass(frozen=True)
class CompletedQueryScalarValues:
    """Scalar values returned by a completed query"""

    query_id: str
    chart_value_min: float
    chart_value_max: float
    result_primary_time_granularity: TimeGranularity


@dataclass(frozen=True)
class MQLServer:
    """Object to represent a Org MQL Server."""

    id: int
    name: str
    url: str
    is_org_default: bool

    @classmethod
    def from_gql_dict(cls, _input: Dict[str, str]) -> "MQLServer":  # noqa: D
        return cls(
            id=int(_input["id"]),
            name=_input["name"],
            url=_input["url"],
            is_org_default=bool(_input["isOrgDefault"]),
        )


class HealthReportStatus(Enum):
    """Statuses for Health Reports."""

    SUCCESS = "SUCCESS"  # MQL Server responded as healthy
    FAIL = "FAIL"  # MQL Server responded as unhealthy


@dataclass(frozen=True)
class HealthReport:
    """Object to represent a MQL Server Health Report."""

    name: str
    status: str
    error_message: str

    @classmethod
    def from_gql_dict(cls, _input: Dict[str, str]) -> "HealthReport":  # noqa: D
        return cls(
            name=_input["name"],
            status=_input["status"],
            error_message=_input["errorMessage"],
        )


@dataclass(frozen=True)
class ServerHealthReport:
    """Object to represent the health of a data warehouse."""

    name: str
    status: str
    url: str
    version: str = ""
    error_message: str = ""
    servers: List[HealthReport] = field(default_factory=list)


@dataclass(frozen=True)
class MqlMaterializeResp:
    """Response object received from MQL from a successful Materialize request."""

    schema: Optional[str]
    table: Optional[str]
    query_id: str

    @property
    def fully_qualified_name(self) -> str:  # noqa: D
        return f"{self.schema}.{self.table}"


@dataclass(frozen=True)
class MQLServerImage:
    """Object to represent a MQL Server Image."""

    service_name: str
    version_hash: str
    download_url: str

    @classmethod
    def from_gql_dict(cls, _input: Dict[str, str]) -> "MQLServerImage":  # noqa: D
        return cls(
            service_name=_input["serviceName"],
            version_hash=_input["versionHash"],
            download_url=_input["downloadUrl"],
        )


class PercentChange(Enum):
    """Used to calculate the percent change between the given granularity."""

    DOD = "dod"  # day-over-day
    WOW = "wow"  # week-over-week
    MOM = "mom"  # month-over-month
    QOQ = "qoq"  # quarter-over-quarter
    YOY = "yoy"  # year-over-year


class DimensionType(Enum):
    """Dimension type Enums."""

    CATEGORICAL = "categorical"
    TIME = "time"


@dataclass(frozen=True)
class Dimension:
    """Object to represent a Dimension object."""

    name: str
    identifier_name: str
    identifier_names: List[str]
    dimension_type: DimensionType
    is_primary_time: bool
    time_granularity: Optional[TimeGranularity]
    cardinality: Optional[int]

    @classmethod
    def from_gql_dict(cls, _input: Dict[str, Any]) -> "Dimension":  # type: ignore[misc] # noqa: D
        return cls(
            name=_input["name"],
            identifier_name=_input["identifierName"],
            identifier_names=_input["identifierNames"],
            dimension_type=DimensionType(_input["type"].lower()),
            is_primary_time=_input["isPrimaryTime"],
            time_granularity=TimeGranularity(_input["timeGranularity"].lower()) if _input["timeGranularity"] else None,
            cardinality=_input["cardinality"],
        )


@dataclass(frozen=True)
class Query:
    """Object to represent a Query object."""

    id: int
    branch: int
    commit: str
    metrics: List[str]
    dimensions: List[str]
    status: MqlQueryStatus

    @classmethod
    def from_gql_dict(cls, _input: Dict[str, Any]) -> "Query":  # type: ignore[misc] # noqa: D
        return cls(
            id=_input["id"],
            branch=_input["modelKey"]["branch"],
            commit=_input["modelKey"]["commit"],
            metrics=_input["metrics"],
            dimensions=_input["dimensions"],
            status=MqlQueryStatus(_input["status"]),
        )
