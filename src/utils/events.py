"""Expose the `events` collection in the tranY database.

The meaning of events and their associated data is listed here.

TrainingIterationStart:
    iter (int): iteration count

TrainingIterationEnd:
    loss (float): loss on training set
"""
from datetime import datetime

from pymongo import MongoClient

client = MongoClient()
db = client.tranY
events = db.events


def add_event(document):
    document["timestamp"] = datetime.now()
    events.insert_one(document)
