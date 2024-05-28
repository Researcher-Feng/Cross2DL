function _timeout(delay) {
                                 this.clearTimer()
                                 this.timer = setTimeout( function() {
                                                            this.flush('timeouted', 'failed')
                                                                .fail('timeouted')
                                                            this.forget() }.bind(this)
                                                        , delay * 1000 )
                                 return this }