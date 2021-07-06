
import sqlite3
import pandas as pd


class Data():
    """
    A class to model querying and returning tables from an sqlite3
    database.
    """

    # relative path from scripts -> data
    SOURCE: str = "..\\data\\source.db3"
    conn: sqlite3.Connection = None

    def _create_connection() -> None:
        """
        create a database connection to the SQLite database
        specified by the db_file
        """
        try:
            Data.conn = sqlite3.connect(Data.SOURCE)
        except sqlite3.Error as e:
            print(e)

    def _close_connecion() -> None:
        """
        closes the active connection
        """
        if Data.conn is not None:
            Data.conn.close()

    def get(tbl: str) -> pd.DataFrame:
        """
        :return: tbl from the source database as a pd.DataFrame
        """
        Data._create_connection()
        df: pd.DataFrame = pd.read_sql_query("SELECT * from " + tbl, Data.conn)
        Data._close_connecion()
        return df
