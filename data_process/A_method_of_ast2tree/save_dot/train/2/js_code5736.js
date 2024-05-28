function LatencyMetric () {
  if (!(this instanceof LatencyMetric)) {
    return new LatencyMetric()
  }

  Metric.call(this)

  this.key = 'latency'
  this.default = [0]
  this.hooks = [{
    trigger: 'before',
    event: 'send',
    handler: this._start
  }, {
    trigger: 'before',
    event: 'receive',
    handler: this._stop
  }]

  this._tests = {}

  setInterval(this._expireTimeouts.bind(this), LatencyMetric.TEST_TIMEOUT)
}