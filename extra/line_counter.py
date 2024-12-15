#!/usr/bin/env python3
import os
import sys
import json
import token
import tokenize
from datetime import datetime, timezone

TOKEN_WHITELIST = [token.OP, token.NAME, token.NUMBER, token.STRING]

def is_docstring(t):
    return t.type == token.STRING and t.string.startswith('"""') and t.line.strip().startswith('"""')

def gen_stats(base_path="."):
    table = []
    exo_path = os.path.join(base_path, "exo")
    if not os.path.exists(exo_path):
        print(f"Warning: {exo_path} directory not found")
        return table

    for path, _, files in os.walk(exo_path):
        for name in files:
            if not name.endswith(".py"):
                continue

            filepath = os.path.join(path, name)
            relfilepath = os.path.relpath(filepath, base_path).replace('\\', '/')

            try:
                with tokenize.open(filepath) as file_:
                    tokens = [t for t in tokenize.generate_tokens(file_.readline)
                            if t.type in TOKEN_WHITELIST and not is_docstring(t)]
                    token_count = len(tokens)
                    line_count = len(set([x for t in tokens
                                        for x in range(t.start[0], t.end[0]+1)]))
                    if line_count > 0:
                        table.append([relfilepath, line_count, token_count/line_count])
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue

    return table

def gen_diff(table_old, table_new):
    table = []
    files_new = set([x[0] for x in table_new])
    files_old = set([x[0] for x in table_old])

    added = files_new - files_old
    deleted = files_old - files_new
    unchanged = files_new & files_old

    for file in added:
        file_stat = [stats for stats in table_new if file in stats][0]
        table.append([file_stat[0], file_stat[1], file_stat[1], file_stat[2], file_stat[2]])

    for file in deleted:
        file_stat = [stats for stats in table_old if file in stats][0]
        table.append([file_stat[0], 0, -file_stat[1], 0, -file_stat[2]])

    for file in unchanged:
        file_stat_old = [stats for stats in table_old if file in stats][0]
        file_stat_new = [stats for stats in table_new if file in stats][0]
        if file_stat_new[1] != file_stat_old[1] or file_stat_new[2] != file_stat_old[2]:
            table.append([
                file_stat_new[0],
                file_stat_new[1],
                file_stat_new[1] - file_stat_old[1],
                file_stat_new[2],
                file_stat_new[2] - file_stat_old[2]
            ])

    return table

def create_json_report(table, is_diff=False):
    timestamp = datetime.now(timezone.utc).isoformat()
    commit_sha = os.environ.get('GITHUB_SHA', 'unknown')
    branch = os.environ.get('GITHUB_REF_NAME', 'unknown')
    pr_number = os.environ.get('GITHUB_EVENT_NUMBER', '')

    if is_diff:
        files = [{
            'name': row[0],
            'current_lines': row[1],
            'line_diff': row[2],
            'current_tokens_per_line': row[3],
            'tokens_per_line_diff': row[4]
        } for row in table]

        report = {
            'type': 'diff',
            'timestamp': timestamp,
            'commit_sha': commit_sha,
            'branch': branch,
            'pr_number': pr_number,
            'files': files,
            'total_line_changes': sum(row[2] for row in table),
            'total_files_changed': len(files)
        }
    else:
        files = [{
            'name': row[0],
            'lines': row[1],
            'tokens_per_line': row[2]
        } for row in table]

        report = {
            'type': 'snapshot',
            'timestamp': timestamp,
            'commit_sha': commit_sha,
            'branch': branch,
            'files': files,
            'total_lines': sum(row[1] for row in table),
            'total_files': len(files)
        }

    return report

def display_diff(diff):
    return "+" + str(diff) if diff > 0 else str(diff)

def format_table(rows, headers, floatfmt):
    if not rows:
        return ""

    # Add headers as first row
    all_rows = [headers] + rows

    # Calculate column widths
    col_widths = []
    for col in range(len(headers)):
        col_width = max(len(str(row[col])) for row in all_rows)
        col_widths.append(col_width)

    # Format rows
    output = []
    for row_idx, row in enumerate(all_rows):
        formatted_cols = []
        for col_idx, (value, width) in enumerate(zip(row, col_widths)):
            if isinstance(value, float):
                # Handle float formatting based on floatfmt
                fmt = floatfmt[col_idx]
                if fmt.startswith('+'):
                    value = f"{value:+.1f}"
                else:
                    value = f"{value:.1f}"
            elif isinstance(value, int) and col_idx > 0:  # Skip filename column
                # Handle integer formatting based on floatfmt
                fmt = floatfmt[col_idx]
                if fmt.startswith('+'):
                    value = f"{value:+d}"
                else:
                    value = f"{value:d}"
            formatted_cols.append(str(value).ljust(width))
        output.append("  ".join(formatted_cols))

        # Add separator line after headers
        if row_idx == 0:
            separator = []
            for width in col_widths:
                separator.append("-" * width)
            output.append("  ".join(separator))

    return "\n".join(output)

if __name__ == "__main__":
    if len(sys.argv) == 3:
        # Comparing two directories
        headers = ["File", "Lines", "Diff", "Tokens/Line", "Diff"]
        table = gen_diff(gen_stats(sys.argv[1]), gen_stats(sys.argv[2]))

        if table:
            # Print table output
            print("### Code Changes in 'exo' Directory")
            print("```")
            print(format_table(
                sorted(table, key=lambda x: abs(x[2]) if len(x) > 2 else 0, reverse=True),
                headers,
                (".1f", "d", "+d", ".1f", "+.1f")
            ))
            total_changes = sum(row[2] for row in table)
            print(f"\nTotal line changes: {display_diff(total_changes)}")
            print("```")

            # Generate JSON report
            report = create_json_report(table, is_diff=True)
            with open('line-count-diff.json', 'w') as f:
                json.dump(report, f, indent=2)
    else:
        # Single directory analysis
        headers = ["File", "Lines", "Tokens/Line"]
        table = gen_stats(sys.argv[1] if len(sys.argv) > 1 else ".")

        if table:
            # Print table output
            print("### Code Statistics for 'exo' Directory")
            print("```")
            print(format_table(
                sorted(table, key=lambda x: x[1], reverse=True),
                headers,
                (".1f", "d", ".1f")
            ))
            total_lines = sum(row[1] for row in table)
            print(f"\nTotal lines: {total_lines}")
            print("```")

            # Generate JSON report
            report = create_json_report(table, is_diff=False)
            with open('line-count-snapshot.json', 'w') as f:
                json.dump(report, f, indent=2)
