import math
import re
from copy import copy
from collections import OrderedDict
import yaml
from train import OPTIMIZERS
from model.weights_init import WEIGHTS_INIT
from model.models import  MODELS
from model.fc import ACTIVATION
from model.fc import NORMALIZATION

class ConfigError(ValueError):
    pass


def _is_dict_like(entry):
    return hasattr(entry, '__iter__') and hasattr(entry, '__getitem__')


class BaseValidator:
    def validate(self, entry, field_uri=None):
        pass

    def raise_error(self, value, field_uri, reason=None):
        error_message = 'Invalid value "{value}" for {field_uri}'.format(value=value, field_uri=field_uri)
        if reason:
            error_message = '{error_message}: {reason}'.format(error_message=error_message, reason=reason)

        raise ConfigError(error_message.format(value, field_uri))


class ConfigValidator(BaseValidator):
    def __init__(self, config_uri, fields=None, **kwargs):
        super().__init__(**kwargs)
        self.fields = OrderedDict()
        self.field_uri = config_uri

        if fields:
            for name in fields.keys():
                self.fields[name] = fields[name]
        else:
            for name in dir(self):
                value = getattr(self, name)
                if not isinstance(value, BaseValidator):
                    continue

                field_copy = copy(value)
                field_copy.field_uri = "{}.{}".format(config_uri, name)
                self.fields[name] = field_copy

    def validate(self, entry, field_uri=None):
        super().validate(entry, field_uri)
        field_uri = field_uri or self.field_uri
        if not _is_dict_like(entry):
            raise ConfigError("{} is expected to be dict-like".format(field_uri))
        for key in entry:
            self.fields[key].validate(entry[key])

        required_fields = set(name for name, value in self.fields.items() if value.required())
        missing_arguments = required_fields.difference(entry)

        if missing_arguments:
            arguments = ', '.join(map(str, missing_arguments))
            self.raise_error(
                entry, field_uri, "Invalid config for {}: missing required fields: {}".format(field_uri, arguments)
            )

        for field in self.fields:
            if field not in entry and hasattr(self.fields[field], 'default'):
                entry[field] = self.fields[field].default

    @property
    def known_fields(self):
        return set(self.fields)

    def raise_error(self, value, field_uri, reason=None):
        raise ConfigError(reason)


class BaseField(BaseValidator):
    def __init__(self, optional=False, description=None, default=None, **kwargs):
        super().__init__(**kwargs)
        self.optional = optional
        self.description = description
        self.default = default

    def validate(self, entry, field_uri=None):
        super().validate(entry, field_uri)
        if self.required() and entry is None:
            raise ConfigError("{} is not allowed to be None".format(field_uri))

    @property
    def type(self):
        return str

    def required(self):
        return not self.optional and self.default is None

    def parameters(self):
        parameters_dict = {}
        for key, _ in self.__dict__.items():
            if not key.startswith('_') and hasattr(self, key) and not hasattr(BaseValidator(), key):
                if isinstance(self.__dict__[key], BaseField):
                    parameters_dict[key] = self.__dict__[key].parameters()
                else:
                    parameters_dict[key] = self.__dict__[key]
            parameters_dict['type'] = type(self.type()).__name__

        return parameters_dict


class StringField(BaseField):
    def __init__(self, choices=None, regex=None, case_sensitive=False, allow_own_choice=False, **kwargs):
        super().__init__(**kwargs)
        self.choices = choices if case_sensitive or not choices else list(map(str.lower, choices))
        self.allow_own_choice = allow_own_choice
        self.case_sensitive = case_sensitive
        self.set_regex(regex)

    def set_regex(self, regex):
        if regex is None:
            self._regex = regex
        self._regex = re.compile(regex, flags=re.IGNORECASE if not self.case_sensitive else 0) if regex else None

    def validate(self, entry, field_uri=None):
        super().validate(entry, field_uri)
        if entry is None:
            return

        source_entry = entry

        if not isinstance(entry, str):
            raise ConfigError("{} is expected to be str".format(source_entry))

        if not self.case_sensitive:
            entry = entry.lower()

        if self.choices and entry not in self.choices and not self.allow_own_choice:
            reason = "unsupported option, expected one of: {}".format(', '.join(map(str, self.choices)))
            self.raise_error(source_entry, field_uri, reason)

        if self._regex and not self._regex.match(entry):
            self.raise_error(source_entry, field_uri, reason=None)

    @property
    def type(self):
        return str


class NumberField(BaseField):
    def __init__(self, value_type=float, min_value=None, max_value=None, allow_inf=False, allow_nan=False, **kwargs):
        super().__init__(**kwargs)
        self._value_type = value_type
        self.min = min_value
        self.max = max_value
        self._allow_inf = allow_inf
        self._allow_nan = allow_nan

    def validate(self, entry, field_uri=None):
        super().validate(entry, field_uri)
        if entry is None:
            return

        if self.type != float and isinstance(entry, float):
            raise ConfigError("{} is expected to be int".format(field_uri))
        if not isinstance(entry, int) and not isinstance(entry, float):
            raise ConfigError("{} is expected to be number".format(field_uri))

        if self.min is not None and entry < self.min:
            reason = "value is less than minimal allowed - {}".format(self.min)
            self.raise_error(entry, field_uri, reason)
        if self.max is not None and entry > self.max:
            reason = "value is greater than maximal allowed - {}".format(self.max)
            self.raise_error(entry, field_uri, reason)

        if math.isinf(entry) and not self._allow_inf:
            self.raise_error(entry, field_uri, "value is infinity")
        if math.isnan(entry) and not self._allow_nan:
            self.raise_error(entry, field_uri, "value is NaN")

    @property
    def type(self):
        return self._value_type

class TrainConfig(ConfigValidator):
    batch_size = NumberField(value_type=int, optional=True, min_value=1, default=16)
    epochs = NumberField(value_type=int, min_value=1)
    optimizer = StringField(optional=True, default='Adamax', choices=OPTIMIZERS)
    initializer = StringField(optional=True, default='none', choices=WEIGHTS_INIT)
    seed = NumberField(value_type=int, optional=True, min_value=0, default=0)
    dropout = NumberField(value_type=float, optional=True, min_value=0.0, max_value=1.0, default=0.0)
    dropout_l = NumberField(value_type=float, optional=True, min_value=0.0, max_value=1.0, default=0.0)
    dropout_c = NumberField(value_type=float, optional=True, min_value=0.0, max_value=1.0, default=0.0)
    dropout_w = NumberField(value_type=float, optional=True, min_value=0.0, max_value=1.0, default=0.0)
    dropout_g = NumberField(value_type=float, optional=True, min_value=0.0, max_value=1.0, default=0.0)
    weights_decay = NumberField(value_type=float, optional=True, min_value=0.0, max_value=1.0, default=0.0)


    def required(self):
        return True

class ModelConfig(ConfigValidator):
    type = StringField(choices=MODELS)
    num_hidden = NumberField(value_type=int, default=1024, min_value=1)
    activation = StringField(choices=ACTIVATION)
    norm = StringField(choices=NORMALIZATION, optional=True, default='weights')
    train = TrainConfig('train')

def read_config(config_file):
    with open(config_file, 'r') as content:
        config_dict = yaml.safe_load(content)
        config_validator = ModelConfig('model')
        config_validator.validate(config_dict['model'])

    return config_dict['model']
