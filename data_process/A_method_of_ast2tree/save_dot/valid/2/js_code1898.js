function join (socket, multiaddr, pub, cb) {
    const log = socket.log = config.log.bind(config.log, '[' + socket.id + ']')

    if (getConfig().strictMultiaddr && !util.validateMa(multiaddr)) {
      joinsTotal.inc()
      joinsFailureTotal.inc()
      return cb('Invalid multiaddr')
    }

    if (getConfig().cryptoChallenge) {
      if (!pub.length) {
        joinsTotal.inc()
        joinsFailureTotal.inc()
        return cb('Crypto Challenge required but no Id provided')
      }

      if (!nonces[socket.id]) {
        nonces[socket.id] = {}
      }

      if (nonces[socket.id][multiaddr]) {
        log('response cryptoChallenge', multiaddr)

        nonces[socket.id][multiaddr].key.verify(
          Buffer.from(nonces[socket.id][multiaddr].nonce),
          Buffer.from(pub, 'hex'),
          (err, ok) => {
            if (err || !ok) {
              joinsTotal.inc()
              joinsFailureTotal.inc()
            }
            if (err) { return cb('Crypto error') } // the errors NEED to be a string otherwise JSON.stringify() turns them into {}
            if (!ok) { return cb('Signature Invalid') }

            joinFinalize(socket, multiaddr, cb)
          })
      } else {
        joinsTotal.inc()
        const addr = multiaddr.split('ipfs/').pop()

        log('do cryptoChallenge', multiaddr, addr)

        util.getIdAndValidate(pub, addr, (err, key) => {
          if (err) { joinsFailureTotal.inc(); return cb(err) }
          const nonce = uuid() + uuid()

          socket.once('disconnect', () => {
            delete nonces[socket.id]
          })

          nonces[socket.id][multiaddr] = { nonce: nonce, key: key }
          cb(null, nonce)
        })
      }
    } else {
      joinsTotal.inc()
      joinFinalize(socket, multiaddr, cb)
    }
  }