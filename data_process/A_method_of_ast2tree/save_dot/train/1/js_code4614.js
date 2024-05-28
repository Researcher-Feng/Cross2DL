function setup_headers(headers) {
    keys(headers).forEach(function(key) {
      client.setRequestHeader(key, headers[key]) })}