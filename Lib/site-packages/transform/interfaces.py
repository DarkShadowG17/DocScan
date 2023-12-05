from enum import Enum
import logging
import functools
import time
import base64
import json
import pandas as pd
import requests
import textwrap

from typing import List, Dict, Optional, Tuple, Any

from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport

from transform_tools.validate import (
    validate_configs as tt_validate_configs,
    commit_configs as tt_commit_configs,
    promote_model,
    ERROR_RESPONSE_PREFIX,
)
from .constants import DEFAULT_QUERY_TIMEOUT, MAX_PAGE_SIZE, TRANSFORM_PROD_API
from .exceptions import QueryRuntimeException
from .utils import resolve_local_commit_info
from .helpers import intersect_dict
from .models import (
    MqlQueryResultSeries,
    MqlQueryResultSource,
    MqlQueryStatus,
    MqlQueryStatusResp,
    QueryResult,
    CacheMode,
    Dimension,
    Organization,
    User,
    Materialization,
    ModelKey,
    Metric,
    MQLServer,
    MQLServerImage,
    PercentChange,
    Query,
    HealthReport,
    UserState,
    TimeGranularity,
)

logger = logging.getLogger(__name__)


class DWValidationType(Enum):  # noqa: D
    ALL = "allIssues"
    DATA_SOURCE = "dataSourceIssues"
    DIMENSION = "dimensionIssues"
    METRIC = "metricIssues"


class BaseInterface:
    """Base implementation for our various GraphQL APIs we connect to.

    Assumes that authorization and GraphQL query pattern are the same across Backend, and MQL APIs
    """

    def __init__(self, auth_header: Dict[str, str], rest_api_url: str, use_async: bool = False) -> None:  # noqa: D
        # FIXME: use_async is available, but since we had to downgrade to GQL 2, it makes
        # no difference to the underlying transport. We should probably come back to this
        # later and find out an elegant solution to async execution.

        self.rest_api_url = rest_api_url
        self.auth_header = auth_header

        self.gql_client = Client(
            transport=RequestsHTTPTransport(
                url=self.graphql_api_url, retries=2, headers=self.auth_header, timeout=90
            ),  # type: ignore
            fetch_schema_from_transport=False,
        )

    def execute(self, query: Any, verbose: bool = False, variable_values: Dict[str, Any] = None) -> Dict[str, Any]:  # type: ignore[misc]
        """Error handling for GQL Clients"""
        start_time = time.time()
        try:
            response = self.gql_client.execute(query, variable_values=variable_values)
        except requests.exceptions.ConnectionError:
            raise Exception(
                textwrap.dedent(
                    """\
                    Transform could not connect to the MQL Server.

                    Check that you are currently connected to the internet or that the MQL Server is up using:

                            mql health-report
                    """
                )
            )
        except Exception as e:
            # It's possible for e.args to be an empty tuple, so we have to
            # guard against that
            exception_response = ""
            if len(e.args) > 0:
                exception_response = str(e.args[0])

            if "Authentication hook unauthorized" in exception_response:
                raise Exception(
                    textwrap.dedent(
                        """\
                        Transform could not authenticate the set API Key.

                        A new API Key can be created at:

                            https://app.transformdata.io/api_keys

                        and then set using the following command:

                            mql setup -k <api-key>
                            OR instantiate with MQLClient(api_key=<api-key>)
                        """
                    )
                )
            else:
                raise e
        logger.debug(f"Query for {' '.join(response.keys())} took {(time.time() - start_time):.2f}s")
        return response

    @property
    def graphql_api_url(self) -> str:  # noqa: D
        return f"{self.rest_api_url.strip('/')}/graphql"


class BackendInterface(BaseInterface):  # noqa: D
    def __init__(  # noqa: D
        self, auth_header: Dict[str, str], rest_api_url: str = TRANSFORM_PROD_API, use_async: bool = False
    ) -> None:
        super(BackendInterface, self).__init__(auth_header, rest_api_url, use_async)

    def validate_configs(self, config_dir: str) -> ModelKey:
        """Finds and validates yaml configs in the provided directory"""
        repo, branch, commit = resolve_local_commit_info(config_dir)
        try:
            response = tt_validate_configs(
                auth_header=self.auth_header,
                repo=repo,
                branch=branch,
                commit=commit,
                config_dir=config_dir,
                api_url=self.rest_api_url,
            )
            return ModelKey.from_snake_dict(response.json()["model"])
        except Exception as e:
            error_content = str(e).split(ERROR_RESPONSE_PREFIX)
            error_msg = str(e)
            if len(error_content) == 2:
                error_response = json.loads(error_content[1])["error"]
                error_msg = f"🙈 {error_response['error_type']}: {error_response['message']}"
            raise Exception(error_msg) from None

    def commit_configs(self, config_dir: str, is_validation: bool = False) -> ModelKey:
        """Finds, validates, and commits yaml configs in the provided directory"""
        repo, branch, commit = resolve_local_commit_info(config_dir)
        try:
            response = tt_commit_configs(
                auth_header=self.auth_header,
                repo=repo,
                branch=branch,
                commit=commit,
                config_dir=config_dir,
                is_validation=is_validation,
                api_url=self.rest_api_url,
            )
            return ModelKey.from_snake_dict(response.json()["model"])
        except Exception as e:
            error_content = str(e).split(ERROR_RESPONSE_PREFIX)
            error_msg = str(e)
            if len(error_content) == 2:
                error_response = json.loads(error_content[1])["error"]
                error_msg = f"🙈 {error_response['error_type']}: {error_response['message']}"
            raise Exception(error_msg) from None

    def promote_model_with_key(self, model_key: ModelKey) -> None:
        """Promotes a given model key to be the current/primary model for the organization"""
        promote_model(
            auth_header=self.auth_header,
            repo=model_key.repository,
            branch=model_key.branch,
            commit=model_key.commit,
            api_url=self.rest_api_url,
        )

    def get_my_org(self) -> Organization:  # noqa: D
        """Queries for "my org" (the current user's org)"""
        query = gql(
            """
            query MyOrgQuery {
                myOrganization {
                    id
                    name
                    createdAt
                    primaryConfigRepo
                    primaryConfigBranch
                }
            }
            """
        )

        return Organization.from_gql_dict(self.execute(query)["myOrganization"])

    def get_me(self) -> User:  # noqa: D
        """Queries for "me" (the current user)"""
        query = gql(
            """
            query MeQuery {
                myUser {
                    id
                    userName
                    email
                    mqlServerUrl
                }
            }
            """
        )

        return User.from_gql_dict(self.execute(query)["myUser"])

    def get_latest_model_key(self) -> "ModelKey":
        """Queries for latest model key in Web API GQL"""
        query = gql(
            """
            query LatestModelKeyQuery {
                myOrganization {
                    currentModel {
                        id
                        organizationId
                        gitBranch
                        gitCommit
                        gitRepo
                        createdAt
                        isCurrent
                    }
                }
            }
            """
        )
        return ModelKey.from_gql_dict(self.execute(query)["myOrganization"]["currentModel"][0])

    def get_model_key(self, model_id: int) -> "ModelKey":
        """Queries for a given model id"""
        query = gql(
            """
            query ModelKeyQuery($modelId: ID) {
                myOrganization {
                    models(id: $modelId){
                        id
                        organizationId
                        gitBranch
                        gitCommit
                        gitRepo
                        createdAt
                        isCurrent
                    }
                }
            }
            """
        )
        return ModelKey.from_gql_dict(
            self.execute(query, variable_values={"modelId": model_id})["myOrganization"]["models"][0]
        )

    def get_current_model_key(self) -> "ModelKey":
        """Queries for the user's org's current model key"""
        query = gql(
            """
            query CurrentModelKeyQuery {
                myOrganization {
                    currentModel {
                        id
                        organizationId
                        gitBranch
                        gitCommit
                        gitRepo
                        createdAt
                        isCurrent
                    }
                }
            }
            """
        )
        return ModelKey.from_gql_dict(self.execute(query)["myOrganization"]["currentModel"][0])

    def get_org_mql_servers(self) -> List[MQLServer]:  # noqa: D
        query = gql(
            """
            query MqlServersQuery ($pageSize: Int!){
                myUser {
                    organization {
                        mqlServers (pageSize: $pageSize) {
                            id
                            name
                            url
                            isOrgDefault
                        }
                    }
                }
            }
            """
        )

        return [
            MQLServer.from_gql_dict(s)
            for s in self.execute(query, variable_values={"pageSize": MAX_PAGE_SIZE})["myUser"]["organization"][
                "mqlServers"
            ]
        ]

    def get_user_state(self) -> UserState:
        """Fetch all the core models for the User/Org/current model in one simple query"""
        query = gql(
            """
            query MeQuery {
                myUser {
                    id
                    userName
                    email
                    mqlServerUrl
                    organization {
                        id
                        name
                        createdAt
                        primaryConfigRepo
                        primaryConfigBranch
                        currentModel {
                            id
                            organizationId
                            gitBranch
                            gitCommit
                            gitRepo
                            createdAt
                            isCurrent
                        }
                    }
                }
            }
            """
        )

        resp = self.execute(query)
        user = User.from_gql_dict(resp["myUser"])
        org = Organization.from_gql_dict(resp["myUser"]["organization"])

        model: Optional[ModelKey] = None
        if len(resp["myUser"]["organization"]["currentModel"]):
            model = ModelKey.from_gql_dict(resp["myUser"]["organization"]["currentModel"][0])

        return UserState(user=user, organization=org, current_model=model)

    def get_latest_mql_server_image(self) -> MQLServerImage:
        """Fetch the latest mql server image."""
        query = gql(
            """
            query MqlLatestServerQuery {
                latestMqlServer {
                    serviceName
                    versionHash
                    downloadUrl
                }
            }
            """
        )
        return MQLServerImage.from_gql_dict(self.execute(query)["latestMqlServer"])

    def get_metrics_metadata(self, model_key_id: Optional[int] = None) -> List[Dict[str, Any]]:  # type: ignore[misc]
        """Fetch all org metrics' metadata."""
        query = gql(
            """
            query MetricsMetadataQuery ($pageSize: Int!, $modelId: Int){
                myOrganization {
                    metrics (pageSize: $pageSize, modelId: $modelId){
                        name
                        description
                        displayName
                        tier
                        userOwners{
                            user {
                                userName
                                isOrgAdmin
                                email
                            }
                        }
                        teamOwners{
                            team {
                                name
                                memberUsers {
                                    userName
                                    isOrgAdmin
                                    email
                                }
                            }
                        }
                    }
                }
            }
            """
        )
        return self.execute(query, variable_values={"pageSize": MAX_PAGE_SIZE, "modelId": model_key_id},)[
            "myOrganization"
        ]["metrics"]

    def get_metric_metadata_by_name(self, metric_name: str, model_key_id: Optional[int] = None) -> Dict[str, Any]:  # type: ignore[misc]
        """Fetch a single org metric metadata."""
        query = gql(
            """
            query MetricMetadataQuery ($metricName: String!, $modelId: Int) {
                myOrganization {
                    metric (name: $metricName, modelId: $modelId){
                        name
                        description
                        displayName
                        tier
                        valueFormat
                        isAdditive
                        userOwners{
                            user {
                                userName
                                isOrgAdmin
                                email
                            }
                        }
                        teamOwners{
                            team {
                                name
                                memberUsers {
                                    userName
                                    isOrgAdmin
                                    email
                                }
                            }
                        }
                    }
                }
            }
            """
        )
        return self.execute(query, variable_values={"metricName": metric_name, "modelId": model_key_id},)[
            "myOrganization"
        ]["metric"]


class MQLInterface(BaseInterface):  # noqa: D
    def __init__(self, auth_header: Dict[str, str], query_server_url: str, use_async: bool = False) -> None:  # noqa: D
        super(MQLInterface, self).__init__(auth_header, query_server_url, use_async)

    def __common_query_input_params(  # type: ignore [misc]
        self,
        metrics: List[str],
        dimensions: List[str],
        model_key: Optional[ModelKey] = None,
        where_constraint: Optional[str] = None,
        time_constraint: Optional[str] = None,
        time_granularity: Optional[TimeGranularity] = None,
        time_comparison: Optional[PercentChange] = None,
        order: Optional[List[str]] = None,
        limit: Optional[str] = None,
        cache_mode: Optional[CacheMode] = None,
        as_table: Optional[str] = None,
        time_series: bool = True,
        result_format: Optional[str] = None,
        allow_dynamic_cache: bool = True,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        trim: Optional[bool] = None,
    ) -> Dict[str, Any]:
        return {
            "modelKey": {
                "organization": model_key.organization_id,
                "repo": model_key.repository,
                "branch": model_key.branch,
                "commit": model_key.commit,
            }
            if model_key
            else None,
            "metrics": metrics,
            "groupBy": dimensions,
            "whereConstraint": where_constraint,
            "timeConstraint": time_constraint,
            "timeGranularity": time_granularity.name if time_granularity else None,
            "pctChange": time_comparison.name if time_comparison else None,
            "order": order,
            "limit": limit,
            "cacheMode": (CacheMode.default() if cache_mode is None else cache_mode).name,
            "asTable": as_table,
            "timeSeries": time_series,
            "resultFormat": result_format,
            "allowDynamicCache": allow_dynamic_cache,
            "startTime": start_time,
            "endTime": end_time,
            "trim": trim,
        }

    def drop_materialization(  # noqa: D
        self,
        materialization_name: str,
        start_time: Optional[str],
        end_time: Optional[str],
        model_key: Optional[ModelKey] = None,
    ) -> str:
        """Implements drop materialization mutation in MQL/GQL"""
        query = gql(
            """
            mutation CreateMqlDropMaterializationMutation(
                $materializationName: String!
                $startTime: String
                $endTime: String
                $modelKey: ModelKeyInput
            ) {
                createMqlDropMaterialization(
                    input: {
                        modelKey: $modelKey,
                        materializationName: $materializationName,
                        startTime: $startTime,
                        endTime: $endTime,
                    }
                ) {
                    id
                }
            }
            """
        )

        return self.execute(
            query,
            variable_values={
                "modelKey": {
                    "organization": model_key.organization_id,
                    "repo": model_key.repository,
                    "branch": model_key.branch,
                    "commit": model_key.commit,
                }
                if model_key
                else None,
                "materializationName": materialization_name,
                "startTime": start_time,
                "endTime": end_time,
            },
        )["createMqlDropMaterialization"]["id"]

    def create_materialization(  # noqa: D
        self,
        materialization_name: str,
        start_time: Optional[str],
        end_time: Optional[str],
        model_key: Optional[ModelKey] = None,
        output_table: Optional[str] = None,
        force: bool = False,
    ) -> str:
        """Implements Materialize (new) mutation in MQL/GQL"""
        # Note: we omit the outputTable var if arg is not passed to maintain backwards compatibility with servers
        # that don't recognize it (ie if they're not on 0e7448e8b6a75484f806c32c7f74e3aafbf11b7d or later)
        query = gql(
            """
            mutation CreateMqlMaterializationNewMutation(
                $materializationName: String!
                $startTime: String
                $endTime: String
                $modelKey: ModelKeyInput
                $outputTable: String
                $force: Boolean
            ) {
                createMqlMaterializationNew(
                    input: {
                        modelKey: $modelKey,
                        materializationName: $materializationName,
                        startTime: $startTime,
                        endTime: $endTime,
                        outputTable: $outputTable
                        force: $force
                    }
                ) {
                    id
                }
            }
            """
        )

        return self.execute(
            query,
            variable_values={
                "modelKey": {
                    "organization": model_key.organization_id,
                    "repo": model_key.repository,
                    "branch": model_key.branch,
                    "commit": model_key.commit,
                }
                if model_key
                else None,
                "materializationName": materialization_name,
                "startTime": start_time,
                "endTime": end_time,
                "outputTable": output_table,
                "force": force,
            },
        )["createMqlMaterializationNew"]["id"]

    def get_materializations(self, model_key: Optional[ModelKey] = None) -> List[Materialization]:  # noqa: D
        query = gql(
            """
            query MaterializationsList($modelKey: ModelKeyInput) {
                materializations(modelKey: $modelKey) {
                    name
                    metrics
                    dimensions
                    destinationTable
                }
            }
            """
        )

        return [
            Materialization.from_gql_dict(m)
            for m in self.execute(
                query,
                variable_values={
                    "modelKey": {
                        "organization": model_key.organization_id,
                        "repo": model_key.repository,
                        "branch": model_key.branch,
                        "commit": model_key.commit,
                    }
                    if model_key
                    else None
                },
            )["materializations"]
        ]

    def get_dimension_values(
        self,
        metric_name: str,
        dimension_name: str,
        allow_dynamic_cache: bool = True,
        model_key: Optional[ModelKey] = None,
        page_number: Optional[int] = None,
        page_size: Optional[int] = None,
    ) -> List[str]:
        """Retrieve all dimension values of a given [metric,dimension].

        NOTE: dynamic caching is disabled (ie., only allowed to retrieve dimension values through materialization)
        """
        query = gql(
            """
            query DimensionValueQuery($dimensionName: String!, $metricName: String!, $allowDynamicCache: Boolean, $modelKey: ModelKeyInput, $pageNumber: Int, $pageSize: Int) {
                metricByName(modelKey: $modelKey, name: $metricName) {
                    dimensionValues(dimensionName: $dimensionName, allowDynamicCache: $allowDynamicCache, pageNumber: $pageNumber, pageSize: $pageSize)
                }
            }
            """
        )

        return self.execute(
            query,
            variable_values={
                "modelKey": {
                    "organization": model_key.organization_id,
                    "repo": model_key.repository,
                    "branch": model_key.branch,
                    "commit": model_key.commit,
                }
                if model_key
                else None,
                "pageNumber": page_number,
                "pageSize": page_size,
                "dimensionName": dimension_name,
                "metricName": metric_name,
                "allowDynamicCache": allow_dynamic_cache,
            },
        )["metricByName"]["dimensionValues"]

    def get_metrics(self, model_key: Optional[ModelKey] = None) -> List[Metric]:  # noqa: D
        query = gql(
            """
            query MetricList($modelKey: ModelKeyInput) {
                metrics(modelKey: $modelKey) {
                    name
                    measures
                    type
                    typeParams {
                        numerator
                        denominator
                        expr
                        window
                    }
                    constraint
                    dimensionObjects(modelKey: $modelKey) {
                        name
                        identifierName
                        identifierNames
                        type
                        isPrimaryTime
                        timeGranularity
                        cardinality
                    }
                }
            }
            """
        )
        return [
            Metric.from_gql_dict(m)
            for m in self.execute(
                query,
                variable_values={
                    "modelKey": {
                        "organization": model_key.organization_id,
                        "repo": model_key.repository,
                        "branch": model_key.branch,
                        "commit": model_key.commit,
                    }
                    if model_key
                    else None
                },
            )["metrics"]
        ]

    def get_metric_by_name(self, metric_name: str, model_key: Optional[ModelKey] = None) -> Metric:  # noqa: D
        query = gql(
            """
            query MetricQuery($modelKey: ModelKeyInput, $metricName: String!) {
                metricByName(modelKey: $modelKey, name: $metricName) {
                    name
                    measures
                    type
                    typeParams {
                        numerator
                        denominator
                        expr
                        window
                    }
                    constraint
                    dimensionObjects(modelKey: $modelKey) {
                        name
                        identifierName
                        identifierNames
                        type
                        isPrimaryTime
                        timeGranularity
                        cardinality
                    }
                }
            }
            """
        )
        return Metric.from_gql_dict(
            self.execute(
                query,
                variable_values={
                    "modelKey": {
                        "organization": model_key.organization_id,
                        "repo": model_key.repository,
                        "branch": model_key.branch,
                        "commit": model_key.commit,
                    }
                    if model_key
                    else None,
                    "metricName": metric_name,
                },
            )["metricByName"]
        )

    def get_dimensions(
        self,
        metric_names: Optional[List[str]] = None,
        model_key: Optional[ModelKey] = None,
    ) -> List[Dimension]:
        """Get a list of all unique Dimension objects."""
        variable_values: Dict[str, Any] = {  # type: ignore[misc]
            "modelKey": {
                "organization": model_key.organization_id,
                "repo": model_key.repository,
                "branch": model_key.branch,
                "commit": model_key.commit,
            }
            if model_key
            else None,
        }
        if metric_names:
            dims = []
            # TODO: add a gql query in MQS to get a list of metrics given a list of metric names
            for metric_name in metric_names:
                query = gql(
                    """
                        query MetricDimensionQuery($modelKey: ModelKeyInput, $metricName: String!) {
                            metricByName(modelKey: $modelKey, name: $metricName) {
                                dimensionObjects(modelKey: $modelKey) {
                                    name
                                    identifierName
                                    identifierNames
                                    type
                                    isPrimaryTime
                                    timeGranularity
                                    cardinality
                                }
                            }
                        }
                    """
                )
                variable_values["metricName"] = metric_name
                resp = self.execute(query, variable_values=variable_values)["metricByName"]["dimensionObjects"]
                dims.append({dim["name"]: dim for dim in resp})
                dims = [functools.reduce(lambda x, y: intersect_dict(x, y), dims)]
                if len(dims) == 0:
                    # No intersection, early return
                    return []
            dims = list(dims[0].values())
        else:
            query = gql(
                """
                    query DimensionsQuery($modelKey: ModelKeyInput) {
                        dimensions(modelKey: $modelKey) {
                            name
                            identifierName
                            identifierNames
                            type
                            isPrimaryTime
                            timeGranularity
                            cardinality
                        }
                    }
                """
            )
            dims = self.execute(query, variable_values=variable_values)["dimensions"]

        return [Dimension.from_gql_dict(dim) for dim in dims]

    def get_data_warehouse_validation_issues(
        self, dw_validation_type: DWValidationType, model_key: Optional[ModelKey] = None
    ) -> List[str]:
        """Get data warehouse validation issues associated with the dw_validation_type for the model"""
        query = gql(
            f"""
            query ValidateModel{dw_validation_type.name}($modelKey: ModelKeyInput) {{
                validations(modelKey: $modelKey) {{
                    {dw_validation_type.value}
                }}
            }}
            """
        )

        # TODO: If this starts timing out we should move to a createMqlQuery approach
        return self.execute(
            query,
            variable_values={
                "modelKey": {
                    "organization": model_key.organization_id,
                    "repo": model_key.repository,
                    "branch": model_key.branch,
                    "commit": model_key.commit,
                }
                if model_key
                else None
            },
        )["validations"][dw_validation_type.value]

    def create_query(  # noqa: D
        self,
        metrics: List[str],
        dimensions: List[str],
        model_key: Optional[ModelKey] = None,
        where_constraint_str: Optional[str] = None,
        time_constraint: Optional[str] = None,
        time_granularity: Optional[TimeGranularity] = None,
        time_comparison: Optional[PercentChange] = None,
        order: Optional[List[str]] = None,
        limit: Optional[str] = None,
        cache_mode: Optional[CacheMode] = None,
        as_table: Optional[str] = None,
        allow_dynamic_cache: bool = True,
        result_format: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        trim: Optional[bool] = None,
    ) -> str:
        """Implements CreateMqlQuery mutation in MQL/GQL"""
        # TODO: Find a fix to being able to show cli query graph on server logs
        time_series = False
        query = gql(
            """
            mutation CreateMqlQueryMutation(
                $modelKey: ModelKeyInput,
                $metrics: [String!],
                $groupBy: [String!],
                $whereConstraint: String,
                $timeConstraint: String,
                $timeGranularity: TimeGranularity,
                $order: [String!],
                $limit: LimitInput,
                $cacheMode: CacheMode,
                $asTable: String,
                $timeSeries: Boolean,
                $resultFormat: ResultFormat,
                $pctChange: PercentChange,
                $allowDynamicCache: Boolean,
                $startTime: String,
                $endTime: String,
                $trim: Boolean,
            ) {
                createMqlQuery(
                    input: {
                        modelKey: $modelKey,
                        metrics: $metrics,
                        groupBy: $groupBy,
                        whereConstraint: $whereConstraint,
                        timeConstraint: $timeConstraint,
                        timeGranularity: $timeGranularity,
                        order: $order,
                        limit: $limit,
                        cacheMode: $cacheMode,
                        asTable: $asTable,
                        addTimeSeries: $timeSeries,
                        resultFormat: $resultFormat,
                        pctChange: $pctChange,
                        allowDynamicCache: $allowDynamicCache
                        startTime: $startTime,
                        endTime: $endTime
                        trimIncompletePeriods: $trim
                    }
                ) {
                    id
                }
            }
            """
        )

        params = self.__common_query_input_params(
            model_key=model_key,
            metrics=metrics,
            dimensions=dimensions,
            where_constraint=where_constraint_str,
            time_constraint=time_constraint,
            time_granularity=time_granularity,
            time_comparison=time_comparison,
            order=order,
            limit=limit,
            cache_mode=cache_mode,
            as_table=as_table,
            time_series=time_series,
            result_format=result_format,
            allow_dynamic_cache=allow_dynamic_cache,
            start_time=start_time,
            end_time=end_time,
            trim=trim,
        )

        return self.execute(query, variable_values=params)["createMqlQuery"]["id"]

    def query_latest_metric_change(self, metric_name: str) -> str:
        """Create a query for the latest change in the specified metric"""
        query = gql(
            """
            mutation CreateLatestMetricChangeQuery($input: QueryLatestMetricChangeInput!) {
                queryLatestMetricChange(input: $input) {
                    id
                }
            }
            """
        )

        params = {"input": {"metricName": metric_name}}

        return self.execute(query, variable_values=params)["queryLatestMetricChange"]["id"]

    def drop_cache(self) -> bool:  # noqa: D
        """Drops the entire MQL cache. Only do this is the cache is somehow corrupt."""
        query = gql(
            """
            mutation {
                dropCache(confirm: "yes") {
                    success
                }
            }
            """
        )

        return bool(self.execute(query)["dropCache"]["success"])

    def ping(self) -> requests.Response:  # noqa: D
        """Calls the MQL REST API for Health"""
        return requests.get(f"{self.rest_api_url}/health", headers=self.auth_header, timeout=10)

    def get_version(self) -> str:  # noqa: D
        query = gql(
            """
            query GetVersion {
                version
            }
            """
        )
        return self.execute(query)["version"]

    def get_health_report(self) -> Tuple[str, List[HealthReport]]:  # noqa: D
        """Calls the MQL Server GQL API for Health"""
        query = gql(
            """
            query GetHealthReport {
                version
                healthReport {
                    name
                    status
                    errorMessage
                }
            }
            """
        )

        query_result = self.execute(query)

        return query_result["version"], [HealthReport.from_gql_dict(h) for h in query_result["healthReport"]]

    def _get_materialization_result(self, query_id: str) -> Tuple[str, str]:
        """Retrieves query status given a query_id"""
        query = gql(
            """
            query GetMqlQueryResultsStatus($queryId: ID!) {
                mqlQuery(id: $queryId) {
                    resultTableSchema,
                    resultTableName
                }
            }
            """
        )

        query_result = self.execute(query, variable_values={"queryId": query_id})["mqlQuery"]
        return query_result["resultTableSchema"], query_result["resultTableName"]

    def explain_query_sql(self, query_id: str) -> str:
        """Retrieves the SQL generated by the MQL server for this query"""
        query = gql(
            """
            query SourceQuery($queryId: ID!) {
                sourceQuery(id: $queryId)
            }
            """
        )

        return self.execute(query, variable_values={"queryId": query_id})["sourceQuery"]

    def get_query_status(self, query_id: str) -> MqlQueryStatusResp:
        """Retrieves query status given a query_id"""
        query = gql(
            """
            query GetMqlQueryResultsStatus($queryId: ID!) {
                mqlQuery(id: $queryId) {
                    status
                    error
                    sql
                    resultSource
                    resultPrimaryTimeGranularity
                    result {
                        value
                        pctChange
                        delta
                    }
                    chartValueMin
                    chartValueMax
                    warnings
                }
            }
            """
        )

        q = self.execute(query, variable_values={"queryId": query_id})["mqlQuery"]

        result = None
        if q["result"] is not None:
            result = [
                MqlQueryResultSeries(value=v["value"], delta=v["delta"], pct_change=v["pctChange"]) for v in q["result"]
            ]

        return MqlQueryStatusResp(
            query_id=query_id,
            status=MqlQueryStatus[q["status"]],
            error=q["error"] if q["error"] else None,
            sql=q["sql"],
            result=result,
            result_source=(None if q["resultSource"] is None else MqlQueryResultSource(q["resultSource"])),
            result_primary_time_granularity=(
                None
                if q["resultPrimaryTimeGranularity"] is None
                else TimeGranularity(q["resultPrimaryTimeGranularity"].lower())
            ),
            chart_value_min=q["chartValueMin"],
            chart_value_max=q["chartValueMax"],
            warnings=q["warnings"],
        )

    def get_queries(self, active_only: bool, limit: Optional[int]) -> List[Query]:
        """Retrieves query status given a query_id"""
        query = gql(
            """
            query GetMqlQueries($activeOnly: Boolean, $limit: Int) {
                queries(activeOnly: $activeOnly, limit:$limit) {
                    id,
                    modelKey {
                        branch,
                        commit,
                    },
                    metrics,
                    dimensions,
                    status,
                }
            }
            """
        )
        return [
            Query.from_gql_dict(q)
            for q in self.execute(query, variable_values={"activeOnly": active_only, "limit": limit})["queries"]
        ]

    def get_logs_by_line(self, query_id: str, from_line: int, max_lines: int = 0) -> Tuple[str, int]:
        """Retrieves logs by log-file line number"""
        query = gql(
            """
            query GetLogLines($queryId: ID!, $fromLine: Int, $maxLines: Int) {
                mqlQuery(id: $queryId) {
                    logsByLine(fromLine: $fromLine, maxLines: $maxLines),
                }
            }
            """
        )
        variable_values = {"queryId": query_id, "fromLine": from_line, "maxLines": max_lines}
        logs = self.execute(query, variable_values=variable_values)["mqlQuery"]["logsByLine"]
        return logs, len(logs.split("\n"))

    def get_query_page_as_df(self, query_id: str, cursor: Optional[int] = None) -> QueryResult:
        """Retrieves a single page of a query's results and converts to a DataFrame"""
        cursor = cursor or 0
        query = gql(
            """
            query GetMqlQueryResultsTabular($queryId: ID!, $cursor: Int) {
                mqlQuery(id: $queryId) {
                    resultTabular(orient: TABLE, cursor: $cursor) {
                        nextCursor
                        data
                    }
                }
            }
            """
        )

        query_result = self.execute(query, variable_values={"queryId": query_id, "cursor": cursor})["mqlQuery"]
        tabular = query_result["resultTabular"]
        df = pd.read_json(
            base64.b64decode(tabular["data"].encode()).decode(),
            orient="table",
        )
        return QueryResult(
            df=df,
            cursor=tabular["nextCursor"],
        )

    def poll_for_query_completion(self, query_id: str, timeout: Optional[int] = None) -> MqlQueryStatusResp:
        """Poll for query completion, displaying some progress indication."""
        timeout = timeout if timeout is not None else DEFAULT_QUERY_TIMEOUT
        start = time.time()

        # Poll until the query is complete or we hit the timeout (timeout of 0 indicate no timeout)
        poll_interval = 0.1
        while timeout == 0 or time.time() < start + timeout:
            query_result = self.get_query_status(query_id)
            if query_result.is_complete:
                return query_result
            time.sleep(poll_interval)

            # poll interval starts very low and gets longer until we hit a maximum of 5s
            # allows us to resolve fast queries quickly without hammering mql for long queries
            poll_interval = min(5, poll_interval * 1.5)

        raise QueryRuntimeException(
            query_id,
            f"Timeout reached waiting for query {query_id} to complete after {timeout} seconds. Please see --timeout option to override.",
        )

    def get_materialization_result(self, query_id: str, timeout: Optional[int] = None) -> Tuple[str, str]:
        """Retrieves materialized table resulting from successful materialization query"""
        resp = self.poll_for_query_completion(query_id, timeout=timeout)
        if not resp.is_successful:
            raise QueryRuntimeException.from_query_response(resp)

        return self._get_materialization_result(query_id)

    def get_query_dataframe(self, query_id: str, timeout: Optional[int] = None) -> pd.DataFrame:
        """Retrieves query results in Pandas DataFrame format

        Includes logic for:
        1. Polling for completion
        2. Paging through results
        3. Converting JSON to a Pandas DataFrame
        """
        resp = self.poll_for_query_completion(query_id, timeout=timeout)
        if not resp.is_successful:
            raise QueryRuntimeException.from_query_response(resp)

        cursor: Optional[int] = 0
        df = pd.DataFrame()

        while cursor is not None:
            query_result = self.get_query_page_as_df(query_id, cursor)
            cursor = query_result.cursor
            df = pd.concat([df, query_result.df], ignore_index=True)

        return df

    def invalidate_all_caches(self) -> bool:
        """Invalidates all caches in the server."""
        query = gql(
            """
            mutation InvalidateAllCachesMutation{
                invalidateAllCaches{
                    success
                }
            }
            """
        )

        return bool(self.execute(query)["invalidateAllCaches"]["success"])

    def invalidate_cache_for_metric(self, metric_name: str) -> bool:  # noqa: D
        """Invalidates all caches in the server."""
        query = gql(
            """
            mutation InvalidateCacheForMetricMutation(
                $metricName: String!
            ){
                invalidateCacheForMetric(metricName: $metricName){
                    success
                }
            }
            """
        )

        variable_values = {"metricName": metric_name}
        return bool(self.execute(query, variable_values=variable_values)["invalidateCacheForMetric"]["success"])
