function connect() {
  
  const args = normalizeConnectArgs(arguments);
  
  const options = {
    
    host: 'localhost',
    port: 61613,
    timeout: 3000,
    connectHeaders: {},

    ...args[0]
  };
  
  const connectListener = args[1];
  
  let client = null;
  let socket = null;
  let timeout = null;
  let originalSocketDestroy = null;
  
  const cleanup = function() {
    
    if (timeout) {
      clearTimeout(timeout);
    }
    
    client.removeListener('error', onError);
    client.removeListener('connect', onConnected);
  };
  
  const onError = function(error) {
    
    cleanup();
    
    error.connectArgs = options;
    
    if (typeof connectListener === 'function') {
      connectListener(error);
    }
  };
  
  const onConnected = function() {
    
    if (originalSocketDestroy) {
      socket.destroy = originalSocketDestroy;
    }
    
    cleanup();
    
    client.emit('socket-connect');
    
    const connectOpts = Object.assign(
      {host: options.host}, 
      options.connectHeaders
    );
    
    client.connect(connectOpts, connectListener);
  };
  
  let transportConnect = net.connect;
  
  if ('connect' in options) {
    transportConnect = options.connect;
  }
  else{
    if ('ssl' in options) {
      if (typeof options.ssl === 'boolean') {
        if (options.ssl === true) {
          transportConnect = tls.connect;
        }
      }
      else{
        if (options.ssl !== void 0) {
          throw new Error('expected ssl property to have boolean value');
        }
      }
    }
  }
  
  socket = transportConnect(options, onConnected);

  if (options.timeout > 0) {

    timeout = setTimeout(function() {
      client.destroy(client.createTransportError('connect timed out'));
    }, options.timeout);

    originalSocketDestroy = socket.destroy;

    socket.destroy = function() {
      clearTimeout(timeout);
      socket.destroy = originalSocketDestroy;
      originalSocketDestroy.apply(socket, arguments);
    };
  }

  client = new Client(socket, options);
  
  client.on('error', onError);
  
  return client;
}