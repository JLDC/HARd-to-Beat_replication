import pandas as pd
import wrds

from typing import Optional


def get_credentials(file: str = "credentials") -> dict[str, str]:
    credentials = {}
    with open(file) as f:
        credentials["username"] = f.readline().strip()
        credentials["password"] = f.readline().strip()
    return credentials


def get_connector(credentials: Optional[dict] = None) -> wrds.Connection:
    if credentials is None:
        return wrds.Connection()
    else:
        return wrds.Connection(
            wrds_username=credentials["username"], wrds_password=credentials["password"]
        )


def get_timestring(
    t: pd.Timestamp, tz: str = "America/New_York", format: str = "%H:%M"
) -> str:
    return t.tz_convert(tz).strftime(format)


def get_crsp_naming_table(conn: wrds.Connection) -> pd.DataFrame:
    return conn.get_table(library="crsp", table="stocknames")
