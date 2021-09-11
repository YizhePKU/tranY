import asdl.parser
import asdl.action


def test_cardinality():
    grammar = asdl.parser.parse("src/asdl/Python.asdl")
    cardinality = asdl.action.extract_cardinality(grammar)
    assert cardinality["Module"]["body"] == "multiple"
    assert cardinality["ClassDef"]["name"] == "single"
    assert cardinality["keyword"]["arg"] == "optional"
    assert cardinality["arguments"]["defaults"] == "multiple"
