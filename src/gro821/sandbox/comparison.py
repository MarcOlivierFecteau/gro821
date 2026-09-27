from dataclasses import dataclass


@dataclass
class TimeResults:
    first: float
    second: float
    factor: float

    def __iter__(self):
        return (self.first, self.second, self.factor).__iter__()


@dataclass
class ComparisonConfig:
    structure: str
    split_method: str | None = None  # used for quadtrees
    bb_type: str | None = None
    compressed: bool | None = None  # used for quadtrees

    def __post_init__(self) -> None:
        self.structure = self.structure.lower()
        if self.split_method:
            self.split_method = self.split_method.lower()
        if self.bb_type:
            self.bb_type = self.bb_type.lower()
