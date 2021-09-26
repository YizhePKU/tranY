import re
import ast
import nltk
# TODO: May stuck due to network reasons. Please download manully from http://www.nltk.org/nltk_data/
# See StackOverflow: https://stackoverflow.com/questions/4867197/failed-loading-english-pickle-with-nltk-data-load
# move to ~/nltk_data/tokenizers/ unzip
import itertools


QUOTED_TOKEN_RE = re.compile(r"(?P<quote>''|[`'\"])(?P<string>.*?)(?P=quote)")


def preprocess_example(rewritten_intent, snippet):
    """
    Replace str and var.
    Example:
        preprocess_example("Fastest Way 'to' `drop` Duplicated", "print(drop / b)")
        {'canonical_intent': 'Fastest Way str_0 var_0 Duplicated',
        'intent_tokens': ['fastest', 'way', 'str_0', 'var_0', 'duplicated'],
        'slot_map': {'str_0': {'value': 'to', 'quote': "'", 'type': 'str'},
        'var_0': {'value': 'drop', 'quote': '`', 'type': 'var'}},
        'canonical_snippet': 'print(var_0 / b)'}
    """
    canonical_intent, slot_map = canonicalize_intent(rewritten_intent)
    canonical_snippet = canonicalize_code(snippet, slot_map)
    intent_tokens = tokenize_intent(canonical_intent)
    decanonical_snippet = decanonicalize_code(canonical_snippet, slot_map)

    # test reconstructor recovers the original code
    reconstructed_snippet = ast.unparse(ast.parse(snippet)).strip()
    reconstructed_decanonical_snippet = ast.unparse(ast.parse(decanonical_snippet)).strip()
    assert compare_ast(ast.parse(reconstructed_snippet), ast.parse(reconstructed_decanonical_snippet))
    
    return {'canonical_intent': canonical_intent,
            'intent_tokens': intent_tokens,
            'slot_map': slot_map,
            'canonical_snippet': canonical_snippet}


def tokenize_intent(intent):
    lower_intent = intent.lower()
    tokens = nltk.word_tokenize(lower_intent)
    return tokens


def infer_slot_type(quote, value):
    if quote == '`' and value.isidentifier():
        return 'var'
    return 'str'

def canonicalize_intent(intent):
    """
    Input: str of natural language
    """
    marked_token_matches = QUOTED_TOKEN_RE.findall(intent)
    slot_map = dict()
    var_id = 0
    str_id = 0
    for match in marked_token_matches:
        quote = match[0]
        value = match[1]
        quoted_value = quote + value + quote

        slot_type = infer_slot_type(quote, value)

        if slot_type == 'var':
            slot_name = 'var_%d' % var_id
            var_id += 1
            slot_type = 'var'
        else:
            slot_name = 'str_%d' % str_id
            str_id += 1
            slot_type = 'str'

        intent = intent.replace(quoted_value, slot_name)
        slot_map[slot_name] = {'value': value.strip().encode().decode('unicode_escape', 'ignore'),
                               'quote': quote,
                               'type': slot_type}
    return intent, slot_map



def replace_identifiers_in_ast(py_ast, identifier2slot):
    for node in ast.walk(py_ast):
        for k, v in list(vars(node).items()):
            if k in ('lineno', 'col_offset', 'ctx'):
                continue
            if isinstance(v, str):
                if v in identifier2slot:
                    slot_name = identifier2slot[v]
                    setattr(node, k, slot_name)


def canonicalize_code(code, slot_map):
    string2slot = {x['value']: slot_name for slot_name, x in list(slot_map.items())}
    py_ast = ast.parse(code)
    replace_identifiers_in_ast(py_ast, string2slot)
    canonical_code = ast.unparse(py_ast).strip()

    # the following code handles the special case that
    # a list/dict/set mentioned in the intent, like
    # Intent: zip two lists `[1, 2]` and `[3, 4]` into a list of two tuples containing elements at the same index in each list
    # Code: zip([1, 2], [3, 4])

    entries_that_are_lists = [slot_name for slot_name, val in slot_map.items() if is_enumerable_str(val['value'])]
    if entries_that_are_lists:
        for slot_name in entries_that_are_lists:
            list_repr = slot_map[slot_name]['value']
            #if list_repr[0] == '[' and list_repr[-1] == ']':
            first_token = list_repr[0]  # e.g. `[`
            last_token = list_repr[-1]  # e.g., `]`
            fake_list = first_token + slot_name + last_token
            slot_map[fake_list] = slot_map[slot_name]
            # else:
            #     fake_list = slot_name
            canonical_code = canonical_code.replace(list_repr, fake_list)
    return canonical_code


def is_enumerable_str(identifier_value):
    """
    Test if the quoted identifier value is a list
    """
    return len(identifier_value) > 2 and identifier_value[0] in ('{', '(', '[') and identifier_value[-1] in ('}', ']', ')')


def decanonicalize_code(code, slot_map):
    for slot_name, slot_val in slot_map.items():
        if is_enumerable_str(slot_name):
            code = code.replace(slot_name, slot_val['value'])

    slot2string = {x[0]: x[1]['value'] for x in list(slot_map.items())}
    py_ast = ast.parse(code)
    replace_identifiers_in_ast(py_ast, slot2string)
    raw_code = ast.unparse(py_ast).strip()
    return raw_code


def compare_ast(node1, node2):
    if not isinstance(node1, str):
        if type(node1) is not type(node2):
            return False
    if isinstance(node1, ast.AST):
        for k, v in list(vars(node1).items()):
            if k in ('lineno', 'col_offset', 'ctx'):
                continue
            if not compare_ast(v, getattr(node2, k)):
                return False
        return True
    elif isinstance(node1, list):
        return all(itertools.starmap(compare_ast, zip(node1, node2)))
    else:
        return node1 == node2