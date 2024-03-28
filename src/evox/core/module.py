import warnings
from functools import wraps
from typing import Any, Callable, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import dataclasses

from .state import State


def use_state(func: Callable):
    """Decorator for easy state management.

    This decorator will try to extract the sub-state belong to the module from current state
    and merge the result back to current state.

    Parameters
    ----------
    func
        The method to be wrapped with
    """

    err_msg = "Expect last return value must be State, got {}"

    @wraps(func)
    def wrapper(self, state: State, *args, **kwargs):
        assert isinstance(self, Stateful) and isinstance(state, State)
        # find the state that match the current module
        path, matched_state = state.find_path_to(self._node_id, self._module_name)

        return_value = func(self, matched_state, *args, **kwargs)

        # single return value, the value must be a State
        if not isinstance(return_value, tuple):
            assert isinstance(return_value, State), err_msg.format(type(return_value))
            aux, new_state = None, return_value
        else:
            # unpack the return value first
            assert isinstance(return_value[-1], State), err_msg.format(
                type(return_value[-1])
            )
            aux, new_state = return_value[:-1], return_value[-1]

        state = state.update_path(path, new_state)

        if aux is None:
            return state
        else:
            return (*aux, state)

    return wrapper


def jit_method(method: Callable):
    """Decorator for methods, wrapper the method with jax.jit, and set self as static argument.

    Parameters
    ----------
    method
        A python method

    Returns
    -------
    function
        A jit wrapped version of this method
    """
    return jax.jit(
        method,
        static_argnums=[
            0,
        ],
    )


def default_cond_fun(name: str):
    if name == "__call__":
        return True

    if name.startswith("_"):
        return False

    if name in ["init", "setup"]:
        return False

    return True


def jit_class(cls):
    """A helper function used to jit decorators to methods of a class

    Returns
    -------
    class
        a class with selected methods wrapped
    """
    for attr_name in dir(cls):
        func = getattr(cls, attr_name)
        if callable(func) and cond_fun(attr_name):
            if dataclasses.is_dataclass(cls):
                wrapped = wrapper(jax.jit(func))
            else:
                wrapped = wrapper(jit_method(func))
            setattr(cls, attr_name, wrapped)
    return cls


class Stateful:
    """Base class for all evox modules.

    This module allow easy managing of states.

    All the constants (e.g. hyperparameters) are initialized in the ``__init__``,
    and mutated states are initialized in the ``setup`` method.

    The ``init`` method will automatically call the ``setup`` of the current module
    and recursively call ``setup`` methods of all submodules.
    """

    def __init__(self) -> None:
        self._node_id = None

    def setup(self, key: jax.Array) -> State:
        """Setup mutable state here

        The state it self is immutable, but it act as a mutable state
        by returning new state each time.

        Parameters
        ----------
        key
            A PRNGKey.

        Returns
        -------
        State
            The state of this module.
        """
        return State()

    def _recursive_init(self, key, node_id, module_name) -> Tuple[State, int]:
        self._node_id = node_id
        self._module_name = module_name

        child_states = {}

        # Find all submodules and sort them according to their name.
        # Sorting is important because it makes sure that the node_id
        # is deterministic across different runs.
        submodules = []
        for attr_name in vars(self):
            attr = getattr(self, attr_name)
            if not attr_name.startswith("_") and isinstance(attr, Stateful):
                submodules.append((attr_name, attr))
        submodules.sort()

        for attr_name, attr in submodules:
            if key is None:
                subkey = None
            else:
                key, subkey = jax.random.split(key)
            submodule_state, node_id = attr._recursive_init(
                subkey, node_id + 1, attr_name
            )
            assert isinstance(
                submodule_state, State
            ), "setup method must return a State"
            child_states[attr_name] = submodule_state

        return (
            self.setup(key)
            ._set_state_id_mut(self._node_id)
            ._set_child_states_mut(child_states),
            node_id,
        )

    def init(self, key: jax.Array = None) -> State:
        """Initialize this module and all submodules

        This method should not be overwritten.

        Parameters
        ----------
        key
            A PRNGKey.

        Returns
        -------
        State
            The state of this module and all submodules combined.
        """
        state, _node_id = self._recursive_init(key, 0, None)
        return state
