"""Resolve the generic roles in ``roles.py`` against one delivery's own names.

The delivery's table names, column names, types, fill rates and permitted-value
lists live in a private configuration file that is never committed. Committed
code asks for a role and gets back whatever that delivery calls it::

    cfg = SchemaConfig.load()
    f = cfg.fields(roles.ENCOUNTER)
    frame[f.person_id]

``config.example.json`` documents the structure with placeholder names.
"""

import json
import os
from types import SimpleNamespace

import roles

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(HERE, "config.local.json")

MISSING = """cannot read the private schema configuration

    {path}

That file maps the generic roles in roles.py onto the names this delivery uses.
It is deliberately not in the repository. Regenerate it from the data dictionary:

    python extract_schema.py --dictionary <dictionary.xlsx> --out {path}

config.example.json shows the structure, with placeholder names."""


class ConfigError(RuntimeError):
    """Raised for a missing, unreadable or incomplete configuration."""


def read_raw(path=None):
    """Parse the configuration file without checking it against ``roles``."""
    path = path or DEFAULT_PATH
    try:
        with open(path) as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise ConfigError(MISSING.format(path=path)) from None
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from None


def _real_fields(table):
    """Drop section-header pseudo-fields: they carry no type and no status."""
    return [f for f in table["fields"]
            if not (f.get("status") is None and f["type"].get("base") is None)]


class SchemaConfig:
    """Role-to-name lookup for one delivery, validated on load."""

    def __init__(self, raw, path=DEFAULT_PATH):
        self.path = path
        self.source = raw.get("source")
        self.layout = raw.get("dictionary_layout", {})
        self._value_sets = raw.get("value_sets", {})
        self._deid = raw.get("deidentification", {})
        self._by_name = {}
        self._order = []
        for table in raw.get("tables", []):
            self._by_name[table["name"]] = _real_fields(table)
            self._order.append(table["name"])
        mapping = raw.get("roles", {})
        self._tables = mapping.get("tables", {})
        self._fields = mapping.get("fields", {})
        self._value_set_roles = mapping.get("value_sets", {})
        self._members = mapping.get("value_members", {})
        self._validate()
        self._namespaces = {role: SimpleNamespace(**self._fields[role])
                            for role in self._tables}
        self.deid = SimpleNamespace(**self._deid)

    @classmethod
    def load(cls, path=None):
        path = path or DEFAULT_PATH
        return cls(read_raw(path), path)

    # ---- validation ------------------------------------------------------

    def _fail(self, message):
        raise ConfigError(f"{self.path}: {message}")

    def _validate(self):
        for role in roles.TABLES:
            name = self._tables.get(role)
            if name is None:
                self._fail(f"roles.tables has no entry for '{role}'")
            if name not in self._by_name:
                self._fail(f"roles.tables['{role}'] names '{name}', which has no table entry")
            declared = {f["name"] for f in self._by_name[name]}
            mapped = self._fields.get(role)
            if mapped is None:
                self._fail(f"roles.fields has no block for '{role}'")
            for field_role in roles.FIELDS[role]:
                field = mapped.get(field_role)
                if field is None:
                    self._fail(f"roles.fields['{role}'] has no entry for '{field_role}'")
                if field not in declared:
                    self._fail(f"roles.fields['{role}']['{field_role}'] names '{field}', "
                               f"which is not a field of '{name}'")
        for role in roles.VALUE_SETS:
            key = self._value_set_roles.get(role)
            if key is None:
                self._fail(f"roles.value_sets has no entry for '{role}'")
            if key not in self._value_sets:
                self._fail(f"roles.value_sets['{role}'] names '{key}', "
                           "which has no value list")
        for role, member_roles in roles.VALUE_MEMBERS.items():
            mapped = self._members.get(role, {})
            key = self._value_set_roles.get(role)
            # Categories documented in a field description have no value list to
            # check the members against; the rest must agree with theirs.
            documented = {v["value"] for v in self._value_sets[key]} \
                if key in self._value_sets else None
            for member_role in member_roles:
                value = mapped.get(member_role)
                if value is None:
                    self._fail(f"roles.value_members['{role}'] has no entry "
                               f"for '{member_role}'")
                if documented is not None and value not in documented:
                    self._fail(f"roles.value_members['{role}']['{member_role}'] names a "
                               f"value that is not in the '{role}' value list")
        for key in roles.DEIDENTIFICATION_KEYS:
            if key not in self._deid:
                self._fail(f"the deidentification block has no entry for '{key}'")
        if not self.null_marker:
            self._fail("dictionary_layout has no 'null_marker'")

    # ---- lookup ----------------------------------------------------------

    def table_roles(self):
        """Table roles in delivery order."""
        reverse = {name: role for role, name in self._tables.items()}
        return [reverse[name] for name in self._order if name in reverse]

    def table(self, table_role):
        """The name this delivery gives the table holding ``table_role``."""
        try:
            return self._tables[table_role]
        except KeyError:
            self._fail(f"unknown table role '{table_role}'")

    def fields(self, table_role):
        """Every field role of a table, as attributes: ``cfg.fields(X).person_id``."""
        self.table(table_role)
        return self._namespaces[table_role]

    def field(self, table_role, field_role):
        """The name this delivery gives one field."""
        self.table(table_role)
        try:
            return self._fields[table_role][field_role]
        except KeyError:
            self._fail(f"unknown field role '{table_role}.{field_role}'")

    def field_names(self, field_roles):
        """Names for a group of field roles, across every table that has one."""
        wanted = set(field_roles)
        return {name for mapped in self._fields.values()
                for role, name in mapped.items() if role in wanted}

    def field_specs(self, table_role):
        """Field descriptors in delivered column order: name, type, status, fill rate."""
        return self._by_name[self.table(table_role)]

    def fill_rate(self, table_role, field_role):
        """Documented percentage of rows in which the field is populated."""
        name = self.field(table_role, field_role)
        for spec in self.field_specs(table_role):
            if spec["name"] == name:
                return spec["fill_rate"] or 0.0
        self._fail(f"no specification for '{table_role}.{field_role}'")

    def value_set(self, value_set_role):
        """Permitted values for a categorical field, as documented."""
        try:
            return self._value_sets[self._value_set_roles[value_set_role]]
        except KeyError:
            self._fail(f"unknown value-set role '{value_set_role}'")

    def value_member(self, value_set_role, member_role):
        """The string this delivery uses for one member of a value set."""
        try:
            return self._members[value_set_role][member_role]
        except KeyError:
            self._fail(f"unknown value member '{value_set_role}.{member_role}'")

    def value_members(self, value_set_role, member_roles=None):
        """Member role -> delivered string, for a whole set or a chosen few."""
        wanted = roles.VALUE_MEMBERS.get(value_set_role, ()) if member_roles is None \
            else member_roles
        return {role: self.value_member(value_set_role, role) for role in wanted}

    @property
    def null_marker(self):
        """The token the dictionary writes instead of leaving a value blank."""
        return self.layout.get("null_marker")
