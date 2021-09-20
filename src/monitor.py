"""Monitor events and print them for debugging purposes."""

from utils.events import events

for change in events.watch():
    data = change["fullDocument"]
    _timestamp = data["_timestamp"]
    _event = data["_event"]
    print(
        _timestamp.strftime("%H:%M:%S"),
        _event,
        {k: v for k, v in data.items() if not k.startswith("_")},
    )
