function SipFakeStack(config) {

    if (!config.server) {
        throw '(SipFakeStack) You need at least to specify a valid IPv4/6 target';
    }

    this.server = config.server || null;
    this.port = config.port || 5060;
    this.transport = config.transport || 'UDP';
//    this.lport = config.lport || utils.randomPort();
    this.lport = config.lport || null;
    this.srcHost = config.srcHost;
    this.timeout = config.timeout || 8000;
    this.wsPath = config.wsPath || null;
    this.domain = config.domain || null;
    this.onlyFirst = config.onlyFirst || true;

    if (net.isIPv6(config.server) && !config.srcHost) {
        this.srcHost = utils.randomIP6();
    } else if (!config.srcHost) {
        this.srcHost = utils.randomIP();
    }
}