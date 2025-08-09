
from unidiff import PatchSet

def apply_unified_diff_to_string(code_str: str, diff_str: str) -> str:
    code_lines = code_str.splitlines(keepends=True)
    patch = PatchSet(diff_str.splitlines(keepends=True))

    for patched_file in patch:
        for hunk in patched_file:
            start = hunk.source_start - 1  # convert to 0-based index
            end = start + hunk.source_length
            new_lines = [line.value for line in hunk if not line.is_removed]
            code_lines[start:end] = new_lines

    return ''.join(code_lines)
