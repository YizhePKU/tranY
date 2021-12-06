from data.conala_v2 import (
    process_intent,
    process_snippet,
    unprocess_snippet,
)


def test_process_example():
    intent = "zip two lists `[1, 2]` and `[3, 4]` into one list named `foo`"
    snippet = "foo = [1, 2] + [3, 4]"

    intent1, slots = process_intent(intent)
    assert intent1 == "zip two lists str_0 and str_1 into one list named var_0"
    assert len(slots) == 3

    snippet1 = process_snippet(snippet, slots)
    assert snippet1 == "var_0 = str_0 + str_1"

    snippet2 = unprocess_snippet(snippet1, slots)
    assert snippet2 == snippet
