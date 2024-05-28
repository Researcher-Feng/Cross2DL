function request(uri, options) {
  var client, promise, method, serialise_body_p, mime
  options          = options         || {}
  options.headers  = options.headers || {}
  method           = (options.method || 'GET').toUpperCase()
  uri              = build_uri(uri, options.query, options.body)

  options.headers['X-Requested-With'] = 'XMLHttpRequest'

  serialise_body_p = object_p(options.body)
  if (serialise_body_p) {
    mime = options.headers['Content-Type'] || 'application/x-www-form-urlencoded'
    options.body = serialise_for_type(mime, options.body)
    options.headers['Content-Type'] = mime }

  client  = make_xhr()
  promise = PromiseP.make(client, uri, options)

  setup_listeners()

  setTimeout(function() {
    client.open(method, uri, true, options.username, options.password)
    setup_headers(options.headers || {})
    client.send(options.body) })

  active.push(promise)

  return promise


  // Sticks a serialised query and body object at the end of an URI.
  // build-uri :: String, { String -> String }, { String -> String }? -> String
  function build_uri(uri, query, body) {
    uri = build_query_string(uri, query)
    return method == 'GET'?  build_query_string(uri, body)
    :      /* otherwise */   uri }

  // Setups the headers for the HTTP request
  // setup-headers :: { String -> String | [String] } -> Undefined
  function setup_headers(headers) {
    keys(headers).forEach(function(key) {
      client.setRequestHeader(key, headers[key]) })}

  // Generates a handler for the given type of error
  // make-error-handler :: String -> Event -> Undefined
  function make_error_handler(type) { return function(ev) {
    promise.flush(type, 'failed').fail(type, ev) }}

  // Invokes an error handler for the given type
  // raise :: String -> Undefined
  function raise(type) {
    make_error_handler(type)() }

  // Setups the event listeners for the HTTP request client
  // setup-listeners :: () -> Undefined
  function setup_listeners() {
    client.onerror            = make_error_handler('errored')
    client.onabort            = make_error_handler('forgotten')
    client.ontimeout          = make_error_handler('timeouted')
    client.onloadstart        = function(ev){ promise.fire('load:start', ev)    }
    client.onprogress         = function(ev){ promise.fire('load:progress', ev) }
    client.onloadend          = function(ev){ promise.fire('load:end', ev)      }
    client.onload             = function(ev){ promise.fire('load:success', ev)  }
    client.onreadystatechange = function(  ){
                                  var response, status, state
                                  state = client.readyState

                                  promise.fire('state:' + state_map[state])

                                  if (state == 4) {
                                    var binding_state = success.test(status)? 'ok'
                                                      : error.test(status)?   'failed'
                                                      : /* otherwise */       'any'

                                    response = client.responseText
                                    status   = normalise_status(client.status)
                                    active.splice(active.indexOf(promise), 1)
                                    promise.flush('status:' + status)
                                           .flush('status:' + status_type(status))

                                      status == 0?           raise('errored')
                                    : success.test(status)?  promise.bind(response, status)
                                    : error.test(status)?    promise.fail(response, status)
                                    : /* otherwise */        promise.done([response, status]) }}}
}