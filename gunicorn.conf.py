"""Gunicorn configuration for NBA Fundamentals V2."""

bind = "127.0.0.1:5050"
workers = 1
threads = 4
timeout = 120
accesslog = "-"
errorlog = "-"
# Do NOT use preload_app — bootstrap starts background threads
# (injury monitor) that cause deadlocks when the master forks workers.
preload_app = False


def post_fork(server, worker):
    """Run bootstrap in the worker process after fork."""
    from src.bootstrap import bootstrap
    bootstrap()
    worker.log.info("Worker %s: bootstrap complete", worker.pid)
