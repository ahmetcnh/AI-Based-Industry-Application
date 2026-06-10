import structlog

logger = structlog.get_logger(__name__)

_driver = None


def get_driver(uri: str, user: str, password: str):
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase
        _driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Neo4j driver initialized", uri=uri)
    return _driver


def close_driver():
    global _driver
    if _driver:
        _driver.close()
        _driver = None


def is_available(uri: str, user: str, password: str) -> bool:
    try:
        drv = get_driver(uri, user, password)
        with drv.session() as s:
            s.run("RETURN 1")
        return True
    except Exception:
        return False
