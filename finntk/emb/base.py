from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple


class RefType(Enum):
    WORD_FORM = 1
    LEMMA = 2
    BOTH = 3


class LexicalVectorSpace(ABC):
    takes: RefType
    dim: int


class MultilingualVectorSpace(LexicalVectorSpace):

    @abstractmethod
    def get_vec(self, lang: str, ref):
        pass


class MonolingualVectorSpace(LexicalVectorSpace):

    @abstractmethod
    def get_vec(self, ref):
        pass


class MonoVectorSpaceAdapter(MonolingualVectorSpace):
    lang: str

    def __init__(self, inner: MultilingualVectorSpace, lang: str):
        self.inner = inner
        self.lang = lang
        self.takes = self.inner.takes
        self.dim = self.inner.dim

    def get_vec(self, ref):
        return self.inner.get_vec(self.lang, ref)


class BothVectorSpaceAdapter(MonolingualVectorSpace):
    takes = RefType.BOTH

    def __init__(self, inner: MonolingualVectorSpace):
        self.inner = inner
        self.dim = self.inner.dim

    def get_vec(self, ref: Tuple[str, str]):
        if self.inner.takes == RefType.WORD_FORM:
            return self.inner.get_vec(ref[0])
        elif self.inner.takes == RefType.LEMMA:
            return self.inner.get_vec(ref[1])
        else:
            return self.inner.get_vec(ref)
