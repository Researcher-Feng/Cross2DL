function defineType(type, validator) {
  var typeDef;
  var regKey;

  if (type instanceof Function) {
    validator = _customValidator(type);
    type = type.name;

    //console.log("Custom type", typeof type, type, validator);
  } else if (!(validator instanceof Function)) {
    throw TypeException('Validator must be a function for `{{type}}`', null, null, { type: type });
  }

  typeDef = parseTypeDef(type);
  regKey = typeDef.name.toLocaleLowerCase();

  if (primitives[regKey]) {
    throw TypeException('Cannot override primitive type `{{type}}`', null, null, { type: typeDef.name });
  } else if (registry[regKey] && (registry[regKey].validator !== validator)) {
    throw TypeException('Validator conflict for type `{{type}}` ', null, null, { type: typeDef.name });
  }

  registry[regKey] = {
    type: typeDef.name,
    validator: validator
  };

  return validator;
}