import logging

def logger(name):
    l = logging.getLogger(name)
    l.setLevel(logging.INFO)
    return l