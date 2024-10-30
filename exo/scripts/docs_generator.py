import argparse
import re
import sys
from pathlib import Path
from typing import List

def generate_cli_options_doc(parser: argparse.ArgumentParser) -> str:
    """Generate markdown documentation for CLI options from argparse parser."""
    docs = ["## CLI Options\n\n"]
    
    actions: List[argparse.Action] = parser._actions
    
    for action in actions:
        # Skip help actions
        if isinstance(action, argparse._HelpAction):
            continue
            
        option_names = []
        if action.option_strings:
            option_names = action.option_strings
        elif action.dest in ['command', 'model_name']:  # Positional arguments
            option_names = [f"<{action.dest}>"]
        
        if not option_names:
            continue
            
        option_str = ', '.join(f"`{name}`" for name in option_names)
        
        # Get type hint
        type_str = ''
        if action.type:
            type_str = f" ({action.type.__name__})"
        elif isinstance(action, argparse._StoreTrueAction):
            type_str = " (flag)"
        elif isinstance(action, argparse._StoreAction) and action.choices:
            type_str = f" ({' | '.join(action.choices)})"
            
        # Get default value if exists
        default_str = ''
        if action.default is not None and action.default != argparse.SUPPRESS:
            if isinstance(action.default, str):
                default_str = f" (default: '{action.default}')"
            else:
                default_str = f" (default: {action.default})"
                
        # Combine help text
        help_text = action.help or ''
        
        docs.append(f"- {option_str}{type_str}{default_str}: {help_text}\n")
    
    return ''.join(docs) 

def update_readme_cli_options(parser: argparse.ArgumentParser, readme_path: Path) -> None:
    """Update README.md with current CLI options."""
    try:
        content = readme_path.read_text()
            
        # Generate new CLI options documentation
        cli_docs = generate_cli_options_doc(parser)
        
        # Replace existing CLI options section or append to end
        cli_section_pattern = r"## CLI Options\n\n(?:.*?\n)*?(?=##|$)"
        if re.search(cli_section_pattern, content, re.DOTALL):
            new_content = re.sub(cli_section_pattern, cli_docs, content, flags=re.DOTALL)
        else:
            # Remove trailing newline from content if it exists before adding cli_docs
            new_content = content.rstrip() + "\n" + cli_docs
            
        readme_path.write_text(new_content)
        print(f"Successfully updated CLI options in {readme_path}")
            
    except Exception as e:
        print(f"Error: Could not update README.md with CLI options: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Import main parser only when running as script
    try:
        from exo.main import parser
    except ImportError:
        print("Error: Could not import parser from exo.main. Make sure exo is installed.", file=sys.stderr)
        sys.exit(1)

    # Find repository root (where README.md is located)
    current_dir = Path(__file__).parent
    repo_root = current_dir
    while not (repo_root / "README.md").exists():
        repo_root = repo_root.parent
        if repo_root == repo_root.parent:  # Reached root directory
            print("Error: Could not find README.md in parent directories", file=sys.stderr)
            sys.exit(1)

    readme_path = repo_root / "README.md"
    update_readme_cli_options(parser, readme_path)