function build_uri(uri, query, body) {
    uri = build_query_string(uri, query)
    return method == 'GET'?  build_query_string(uri, body)
    :      /* otherwise */   uri }