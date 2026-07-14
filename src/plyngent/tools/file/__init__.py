from .edit_replace import edit_replace as edit_replace
from .listdir import listdir as listdir
from .read import read_file as read_file
from .tree import tree as tree
from .write import write_file as write_file

FILE_TOOLS = [read_file, write_file, listdir, tree, edit_replace]
