from .polystar import *
from .sudoku import *
from .tangram import * 
from .logicface import *
from .akari import * 

DATASET = dict(
    tangram=TangramDataset,
    sudoku=SudokuDataset,
    logicface=LogicFaceDataset,
    akari=AkariDataset,
    polystar=PolystarDataset
)