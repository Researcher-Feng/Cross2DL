function _init(client, uri, options) {
    Promise.init.call(this)
    this.client  = client
    this.uri     = uri
    this.options = options

    return this }