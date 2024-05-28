function getDefinedNames() {
  return Object.keys(primitives).concat(Object.keys(registry).map(function (type) {
    return registry[type].type;
  }));
}