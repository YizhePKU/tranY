def transform_mr(mr, fn):
    '''Build a new MR by walking the original MR, calling `fn` on each node.
    
    The original MR is left unchanged.

    `fn` should take exactly one argument, an MR node. Its return value will be used
    to replace the node. Unlike `ast.NodeTransformer`, setting a node to `None` will
    not remove the node.
    '''
    mr = fn(mr)
    if isinstance(mr, dict):
        for field, value in mr.items():
            if field != '_tag':
                if isinstance(value, list):
                    mr = dict(mr, **{field: [transform_mr(item, fn) for item in value]})
                else:
                    mr = dict(mr, **{field: transform_mr(value, fn)})
    return mr