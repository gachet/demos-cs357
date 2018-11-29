def dot_refs(objs):
    from objgraph import show_refs
    from io import StringIO
    sio = StringIO()
    show_refs(objs, output=sio)
    return sio.getvalue()
