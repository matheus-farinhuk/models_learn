import os
import sys

# Defina o caminho do diretório do projeto para garantir que os módulos possam ser importados corretamente
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from .linear_regression import LinearRegression

__all__ = [
    "LinearRegression",
]
