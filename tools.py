import re
from enum import Enum
from pathlib import Path

import pandas as pd
import yaml
from google.cloud import bigquery
from pydantic import BaseModel, Field


def get_highlevel_tables_information():
    tables_info = {}
    tables_md = Path("tables").glob("*.md")

    for table_file in tables_md:
        # Skip _index.md file if it exists
        if table_file.name == "_index.md":
            continue

        table_name = table_file.stem.upper()
        tables_info[table_name] = {}

        content = table_file.read_text()

        # Extract Table source
        source_match = re.search(r"\*\*Table source:\*\*(.*?)(?:\n\n|\*\*)", content, re.DOTALL)
        tables_info[table_name]["Table source"] = source_match.group(1).strip()

        purpose_match = re.search(r"\*\*Table purpose:\*\*(.*?)(?:\n\n|\*\*)", content, re.DOTALL)
        tables_info[table_name]["Table purpose"] = purpose_match.group(1).strip()

        rows_match = re.search(r"\*\*Number of rows:\*\*(.*?)(?:\n\n|\*\*)", content, re.DOTALL)
        tables_info[table_name]["Number of rows"] = rows_match.group(1).strip()

        # Extract Links to
        links_section = re.search(r"\*\*Links to:\*\*(.*?)(?:\n\n\#|\Z)", content, re.DOTALL)
        if links_section:
            tables_info[table_name]["Links to:"] = links_section.group(1).strip()

        summary_match = re.search(r"\# Brief summary\n\n(.*?)(?:\n\n\#|\Z)", content, re.DOTALL)
        if summary_match:
            tables_info[table_name]["Brief summary"] = summary_match.group(1).strip()

        considerations_match = re.search(r"\# Important considerations\n\n(.*?)(?:\n\n\#|\Z)", content, re.DOTALL)
        if considerations_match:
            tables_info[table_name]["Important considerations"] = considerations_match.group(1).strip()

    return yaml.dump(tables_info)


class FindRelevantTablesInput(BaseModel):
    user_request: str = Field(
        description="A detailed description of the user's request. These instructions will be used to figure out which mimic-iii tables are needed"
    )


def find_relevant_tables(inputs: FindRelevantTablesInput) -> str:
    """This tool is used to find the relevant tables for the user's request.
    It will use the user's request to figure out which mimic-iii tables are needed.
    You can call this tool multiple times, and mention the tables that worked or not in the previous calls.
    """
    return str(["admissions", "patients", "diagnoses_icd"])


class Table(Enum):
    ICUSTAYS = "icustays"
    D_CPT = "d_cpt"
    MICROBIOLOGYEVENTS = "microbiologyevents"
    CPTEVENTS = "cptevents"
    PATIENTS = "patients"
    INPUTEVENTS_CV = "inputevents_cv"
    OUTPUTEVENTS = "outputevents"
    PROCEDUREEVENTS_MV = "procedureevents_mv"
    LABEVENTS = "labevents"
    DRGCODES = "drgcodes"
    D_ICD_PROCEDURES = "d_icd_procedures"
    D_ITEMS = "d_items"
    CALLOUT = "callout"
    PRESCRIPTIONS = "prescriptions"
    TRANSFERS = "transfers"
    D_ICD_DIAGNOSES = "d_icd_diagnoses"
    DATETIMEEVENTS = "datetimeevents"
    PROCEDURES_ICD = "procedures_icd"
    SERVICES = "services"
    CHARTEVENTS = "chartevents"
    CAREGIVERS = "caregivers"
    DIAGNOSES_ICD = "diagnoses_icd"
    NOTEEVENTS = "noteevents"
    INPUTEVENTS_MV = "inputevents_mv"
    D_LABITEMS = "d_labitems"
    ADMISSIONS = "admissions"


class GetTableSchemaAndDescriptionInput(BaseModel):
    table_name: Table = Field(description="mimic-iii table name")


def get_table_schema_and_description(inputs: GetTableSchemaAndDescriptionInput) -> str:
    """Get schema and description of a mimic-iii table."""
    path = (Path("tables") / inputs.table_name.value.lower()).with_suffix(".md")
    return path.read_text()


class QueryDBInput(BaseModel):
    query: str = Field(description="bigquery sql query to execute. Always limit the result to max 10 rows.")


def query_db(inputs: QueryDBInput, client: bigquery.Client) -> str:
    """Query the mimic-iii bigquery database. Will return only first 10 rows."""
    query_job = client.query(inputs.query)
    df = query_job.to_dataframe()
    df = df.applymap(lambda x: str(x) if isinstance(x, pd.Timestamp) else x)
    return yaml.dump(df.head(10).to_dict(orient="records"))
