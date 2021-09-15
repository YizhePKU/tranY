"""Expose the `events` collection in the tranY database.

TrainingIterationStart:
    iter (int): iteration count

TrainingIterationEnd:
    loss (float): loss on training set
"""

from pymongo import MongoClient

client = MongoClient()
db = client.tranY
events = db.events
