__licence__ = 'MIT'
__author__ = 'kuyaki'
__credits__ = ['kuyaki']
__maintainer__ = 'kuyaki'
__date__ = '2021/03/23'

from enum import Enum

from tree_sitter import Tree

import os
from pathlib import Path
from typing import Optional

from tree_sitter import Language, Parser, Node

class Lang(Enum):
    JAVA = ".java"
    XML = ".xml"
    PYTHON = ".py"
    C = ".c"

project_path = Path(*Path(__file__).parts[:-4])
# print('project_path:', project_path)
Language.build_library(
    # Store the library in the `build` directory
    os.path.join(project_path, "EPVD/code/parserTool", "my-languages.so"),

    # Include one or more languages
    [
        os.path.join(project_path, "EPVD/code/parserTool", "tree-sitter-c"),
        os.path.join(project_path, "EPVD/code/parserTool", "tree-sitter-java")
    ]
)

def java():
    parser = Parser()
    parser.set_language(Language(os.path.join(
        project_path, "EPVD/code/parserTool", "my-languages.so"), "java"))
    return parser

def python():
    parser = Parser()
    parser.set_language(Language(os.path.join(
        project_path, "EPVD/code/parserTool", "my-languages.so"), "python"))
    return parser

def c():
    parser = Parser()
    parser.set_language(Language(os.path.join(
        project_path, "EPVD/code/parserTool", "my-languages.so"), "c"))
    return parser

def tree_sitter_ast(source_code: str, lang: Lang) -> Tree:
    """
    Parse the source code in a specified format into a Tree Sitter AST.
    :param source_code: string with the source code in it.
    :param lang: the source code Lang.
    :return: Tree Sitter AST.
    """
    if lang == Lang.JAVA:
        return java().parse(bytes(source_code, "utf8"))
    elif lang == Lang.PYTHON:
        return python().parse(bytes(source_code, "utf8"))
    elif lang == Lang.C:
        return c().parse(bytes(source_code, "utf8"))
    else:
        raise NotImplementedError()
