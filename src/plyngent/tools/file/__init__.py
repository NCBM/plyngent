from plyngent.tools.temp_workspace import new_temporary_workspace as new_temporary_workspace

from .document import document_info as document_info
from .document import list_documents as list_documents
from .document import read_document as read_document
from .document import search_document as search_document
from .edit_lineno import edit_lineno as edit_lineno
from .edit_replace import edit_replace as edit_replace
from .fs_ops import copy_path as copy_path
from .fs_ops import delete_path as delete_path
from .fs_ops import move_path as move_path
from .get_truncated import get_truncated as get_truncated
from .glob_paths import glob_paths as glob_paths
from .listdir import listdir as listdir
from .read import read_file as read_file
from .regex_files import regex_files as regex_files
from .tree import tree as tree
from .view_image import view_image as view_image
from .write import write_file as write_file

FILE_TOOLS = [
    read_file,
    view_image,
    write_file,
    listdir,
    tree,
    glob_paths,
    regex_files,
    edit_replace,
    edit_lineno,
    copy_path,
    move_path,
    delete_path,
    new_temporary_workspace,
    get_truncated,
    read_document,
    search_document,
    document_info,
    list_documents,
]
