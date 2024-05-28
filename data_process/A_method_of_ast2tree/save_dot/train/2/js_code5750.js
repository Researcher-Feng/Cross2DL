function arrayValidation(typeDef, index, value, previous, attributeName, validator) {
  var indexInc;
  var i;
  var ilen;

  if (value === null || value === undefined || typeDef.indexes.length <= index) {
    //console.log("Validating", value, index, typeDef);
    return validator(value, previous, attributeName);
  } else if (typeDef.indexes.length > index) {
    //console.log("Checking array", value, index, typeDef);
    if (value instanceof Array) {
      if (value.length) {
        indexInc = Math.max(Math.floor(value.length / VALIDATE_MAX_ARR_INDEX), 1);

        for (i = 0, ilen = value.length; i < ilen; i += indexInc) {
          arrayValidation(typeDef, index + 1, value[i], previous instanceof Array ? previous[i] : undefined, attributeName, validator);
        }

        return value;
      } else if (previous instanceof Array && previous.length) {
        indexInc = Math.max(Math.floor(value.length / VALIDATE_MAX_ARR_INDEX), 1);

        for (i = 0, ilen = value.length; i < ilen; i += indexInc) {
          arrayValidation(typeDef, index + 1, null, previous[i], attributeName, validator)
        }

        return value;
      } else {
        return arrayValidation(typeDef, index + 1, undefined, undefined, attributeName, validator)
      }
    }
  }

  throw TypeException('Invalid array for `{{type}}`', null, [ attributeName ], { type: typeDef.name, indexes: typeDef.indexes });
}