function formatSet(value, options) {
  return 'Set(' + registry.format(Array.from(value.values()), options) + ')';
}