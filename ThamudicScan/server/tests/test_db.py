def test_session_round_trip(database):
    session_id = database.create_session()
    database.save_result(session_id, {"source": "x", "text": "𐪀"})
    session = database.get_session(session_id)
    assert session["id"] == session_id
    assert database.list_results(session_id)[0]["text"] == "𐪀"


def test_progress_events_are_monotonic(database):
    session_id = database.create_session()
    database.add_event(session_id, {"type": "started", "progress": 0})
    database.add_event(session_id, {"type": "completed", "progress": 100})
    events = database.list_events(session_id)
    assert [event["sequence"] for event in events] == [1, 2]
