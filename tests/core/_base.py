from abc import ABC, abstractmethod

class BaseTest(ABC):
	test_name: str
	test_area: str