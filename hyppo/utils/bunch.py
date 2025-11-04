#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

"""Container object exposing keys as attributes."""

from collections.abc import Mapping
import copy


class Bunch(Mapping):
    """Container object exposing keys as attributes.

    Concept based on the sklearn.utils.Bunch.

    Bunch objects are sometimes used as an output for functions and methods.
    They extend dictionaries by enabling values to be accessed by key,
    `bunch["value_key"]`, or by an attribute, `bunch.value_key`.

    Examples
    --------
    >>> b = Bunch("data", {"a": 1, "b": 2})
    >>> b
    data({a, b})
    >>> b['b']
    2
    >>> b.b
    2
    >>> b.a = 3
    >>> b['a']
    3

    """

    def __init__(self, name, data):
        """Initialize Bunch with name and data dictionary."""
        if not isinstance(data, Mapping):
            raise TypeError("Data must be some kind of mapping")
        super().__setattr__("_name", str(name))
        super().__setattr__("_data", data)

    def __getitem__(self, k):
        """Get item by key."""
        return self._data[k]

    def __setitem__(self, k, v):
        """Set item by key (read-only, raises AttributeError)."""
        raise AttributeError(f"Bunch {self._name!r} is read-only")

    def __delitem__(self, k):
        """Delete item by key (read-only, raises AttributeError)."""
        raise AttributeError(f"Bunch {self._name!r} is read-only")

    def __getattr__(self, a):
        """Get attribute by name."""
        try:
            return self._data[a]
        except KeyError:
            raise AttributeError(a)

    def __setattr__(self, a, v):
        """Set attribute (read-only, raises AttributeError)."""
        raise AttributeError(f"Bunch {self._name!r} is read-only")

    def __copy__(self):
        """Create a shallow copy of the Bunch."""
        cls = type(self)
        return cls(str(self._name), data=self._data)

    def __deepcopy__(self, memo):
        """Create a deep copy of the Bunch."""
        # extract the class
        cls = type(self)

        # make the copy but without the data
        clone = cls(name=str(self._name), data={})

        # store in the memo that clone is copy of self
        # https://docs.python.org/3/library/copy.html
        memo[id(self)] = clone

        # now we copy the data
        super(cls, clone).__setattr__("_data", copy.deepcopy(self._data, memo))

        return clone

    def __iter__(self):
        """Return iterator over keys."""
        return iter(self._data)

    def __len__(self):
        """Return number of items."""
        return len(self._data)

    def __repr__(self):
        """Return string representation of the Bunch."""
        content = repr(set(self._data)) if self._data else "{}"
        return f"<{self._name} {content}>"

    def __dir__(self):
        """Return list of attributes and keys."""
        return super().__dir__() + list(self._data)

    def __setstate__(self, state):
        """Needed for multiprocessing environment."""
        self.__dict__.update(state)

    def get(self, key, default=None):
        """Get item from bunch."""
        return self._data.get(key, default)

    def to_dict(self):
        """
        Convert the Bunch object to a dictionary.

        This method performs a deep copy of the _data attribute, ensuring that
        the original data remains unchanged.

        Returns
        -------
        dict
            A deep copy of the _data attribute.

        Example
        -------
        >>> bunch = Bunch()
        >>> bunch._data = {'key1': 'value1', 'key2': 'value2'}
        >>> dict_data = bunch.to_dict()
        >>> print(dict_data)
        {'key1': 'value1', 'key2': 'value2'}
        """
        return copy.deepcopy(self._data)
