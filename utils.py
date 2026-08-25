from typing import Dict, Any

def format_cli_arg(arg: str) -> str:
    """Formats a string as a CLI argument name with appropriate hyphens.
    
    - Single-character names get a single hyphen ('-f')
    - Multi-character names get double hyphens ('--foo')
    - Existing hyphens are preserved ('-f' -> '-f', '--foo' -> '--foo')
    """
    # If the argument already starts with a hyphen, return it as-is
    if arg.startswith('-'):
        return arg
    
    # Single letter gets one hyphen, multi-letter gets two
    return f"-{arg}" if len(arg) == 1 else f"--{arg}"

def format_cli_args(args_dict: Dict[str, Any]) -> str:
    """Formats key-value argument pairs into standard CLI flag strings."""
    args_list = []
    for k, v in args_dict.items():
        flag = format_cli_arg(k)

        # Handle boolean flags / switches
        if isinstance(v, bool):
            if v:  # Only append the flag if True (e.g., '--verbose')
                args_list.append(flag)
        elif v is None:
            continue
        else:
            args_list.append(f'{flag} {v}')

    return " ".join(args_list)

def strip_hyphens(arg: str) -> str:
    """Strips leading hyphens from a CLI argument name string.
    
    Examples:
        '--verbose' -> 'verbose'
        '-v'        -> 'v'
        'foo'       -> 'foo'
        '--foo-bar' -> 'foo-bar'  (preserves internal hyphens)
    """
    return arg.lstrip('-')
