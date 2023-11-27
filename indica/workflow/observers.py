#!/usr/bin/env python3

from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Protocol


class TypeObservable(Protocol):
    """
    Typing for Observable class
    """

    data: Any

    def add_observer(self, observer):
        pass

    def remove_observer(self, observer):
        pass

    def clear_observers(self):
        pass

    def notify_observers(self):
        pass


class TypeObserver(TypeObservable):
    """
    Typing for Observer class
    """

    data: Any

    def update(self, *args, **kwargs):
        pass


class Observable:
    """
    Class to hold some data, with information about Observer classes, and
    signal observers to update when stored data has changed
    """

    def __init__(self, initial_value: Optional[Any] = None):
        self.observers: List[TypeObserver] = []
        self._data = initial_value

    def add_observer(self, observer: TypeObserver):
        self.observers.append(observer)

    def remove_observer(self, observer: TypeObserver):
        self.observers.remove(observer)

    def clear_observers(self):
        self.observers = []

    def notify_observers(self):
        for observer in self.observers:
            observer.update()

    @property
    def data(self) -> Any:
        return self._data

    @data.setter
    def data(self, val: Any) -> None:
        self._data = val
        self.notify_observers()


class Observer(Observable):
    """
    Class to compute and store data depending on some source(s)
    Contains methods to be instructed by Observable when source data has
    changed to recompute dependent data
    Is itself Observable, enabling chains of dependent data
    """

    def __init__(
        self,
        operator: Callable,
        depends_on: List[TypeObservable],
        initial_value: Optional[Any] = None,
    ):
        super().__init__(initial_value=initial_value)
        for dependent in depends_on:
            dependent.add_observer(self)
        self._operator = operator

    def update(self, *args, **kwargs):
        self._data = None

    @property
    def data(self) -> Any:
        if self._data is None:
            self._data = self._operator()
            self.notify_observers()
        return self._data

    @data.setter
    def data(self, *args, **kwargs) -> None:
        raise UserWarning("Directly setting value of dependent data")
