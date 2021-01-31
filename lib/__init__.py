import logging
import sys

from appmetrics import metrics


def setup_logging():
    for _ in ("colormath.color_conversions", "colormath.color_objects"):
        logging.getLogger(_).setLevel(logging.CRITICAL)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
                        datefmt='%Y-%m-%d %H:%M:%S')


def gauge(prefix, metric):
    name = "%s.%s" % (prefix, metric)
    if name not in metrics.REGISTRY:
        return metrics.new_gauge(name)
    else:
        return metrics.metric(name)


def get_metric_prefix(obj, metric_prefix):
    if metric_prefix is None:
        return "%s_%s" % (obj.__class__.__name__, id(obj))
    else:
        return metric_prefix
