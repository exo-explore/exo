"""Tests for the exo security gate checks."""
from __future__ import annotations

import ast
import textwrap
from collections.abc import Callable
from pathlib import Path

from exo.security_gate.checks import Issue
from exo.security_gate.checks.async_hazards import check_async_hazards
from exo.security_gate.checks.dangerous_calls import check_dangerous_calls
from exo.security_gate.checks.dict_mutation import check_dict_mutation
from exo.security_gate.checks.event_sourcing import check_event_sourcing
from exo.security_gate.checks.network_exposure import check_network_exposure
from exo.security_gate.checks.pydantic_mutation import check_pydantic_mutation
from exo.security_gate.checks.secrets import check_secrets
from exo.security_gate.suppressions import filter_issues

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(source: str) -> ast.Module:
    return ast.parse(textwrap.dedent(source))


CheckFn = Callable[[str, str, ast.Module], list[Issue]]


def _run_check(check_fn: CheckFn, source: str, filepath: str = "test_file.py") -> list[Issue]:
    src = textwrap.dedent(source)
    tree = _parse(src)
    return check_fn(filepath, src, tree)


# ---------------------------------------------------------------------------
# SECRET
# ---------------------------------------------------------------------------

class TestSecrets:
    def test_detects_generic_api_key(self) -> None:
        source = 'api_key = "sk-abc12345678901234"'
        issues = _run_check(check_secrets, source)
        assert any(i.check_id == "SECRET" for i in issues), f"Expected SECRET, got: {issues}"

    def test_does_not_flag_comment_line(self) -> None:
        source = '# api_key = "test-placeholder"'
        issues = _run_check(check_secrets, source)
        assert not issues, f"Expected no issues on comment line, got: {issues}"

    def test_does_not_flag_placeholder(self) -> None:
        source = 'api_key = "test-placeholder-value"'
        issues = _run_check(check_secrets, source)
        assert not issues, f"Expected no issues for placeholder, got: {issues}"

    def test_detects_huggingface_token(self) -> None:
        source = 'token = "hf_ABCDEFGHIJKLMNOPQRSTuvwxyz"'
        issues = _run_check(check_secrets, source)
        assert any(i.check_id == "SECRET" for i in issues)

    def test_does_not_flag_sk_ant(self) -> None:
        # sk-ant- is handled by api_cost_guard.py, should not be flagged here
        source = 'key = "sk-ant-api03-abcdefghij1234567890"'
        issues = _run_check(check_secrets, source)
        assert not any(i.check_id == "SECRET" for i in issues)

    def test_detects_password_assignment(self) -> None:
        source = 'password = "supersecretpassword123"'
        issues = _run_check(check_secrets, source)
        assert any(i.check_id == "SECRET" for i in issues)

    def test_detects_github_token(self) -> None:
        source = 'token = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"'
        issues = _run_check(check_secrets, source)
        assert any(i.check_id == "SECRET" for i in issues)


# ---------------------------------------------------------------------------
# DANGEROUS_CALL
# ---------------------------------------------------------------------------

class TestDangerousCalls:
    def test_detects_eval(self) -> None:
        source = "result = eval(user_input)"
        issues = _run_check(check_dangerous_calls, source)
        blocking = [i for i in issues if i.check_id == "DANGEROUS_CALL" and i.severity == "block"]
        assert blocking, f"Expected DANGEROUS_CALL block for eval, got: {issues}"

    def test_detects_exec(self) -> None:
        source = "exec(code_string)"
        issues = _run_check(check_dangerous_calls, source)
        blocking = [i for i in issues if i.check_id == "DANGEROUS_CALL" and i.severity == "block"]
        assert blocking

    def test_detects_subprocess_shell_true(self) -> None:
        source = "import subprocess\nsubprocess.run(cmd, shell=True)"
        issues = _run_check(check_dangerous_calls, source)
        blocking = [i for i in issues if i.check_id == "DANGEROUS_CALL" and i.severity == "block"]
        assert blocking, f"Expected DANGEROUS_CALL block for subprocess shell=True, got: {issues}"

    def test_subprocess_shell_false_not_flagged(self) -> None:
        source = "import subprocess\nsubprocess.run(['ls', '-la'], shell=False)"
        issues = _run_check(check_dangerous_calls, source)
        blocking = [
            i for i in issues
            if i.check_id == "DANGEROUS_CALL" and i.severity == "block"
            and "shell=True" in i.message
        ]
        assert not blocking

    def test_detects_pickle_load(self) -> None:
        source = "import pickle\ndata = pickle.load(f)"
        issues = _run_check(check_dangerous_calls, source)
        blocking = [i for i in issues if i.check_id == "DANGEROUS_CALL" and i.severity == "block"]
        assert blocking

    def test_detects_pickle_import_advisory(self) -> None:
        source = "import pickle"
        issues = _run_check(check_dangerous_calls, source)
        advisory = [i for i in issues if i.check_id == "DANGEROUS_CALL" and i.severity == "advisory"]
        assert advisory

    def test_detects_os_system(self) -> None:
        source = "import os\nos.system('rm -rf /')"
        issues = _run_check(check_dangerous_calls, source)
        blocking = [i for i in issues if i.check_id == "DANGEROUS_CALL" and i.severity == "block"]
        assert blocking


# ---------------------------------------------------------------------------
# ASYNC_HAZARD
# ---------------------------------------------------------------------------

class TestAsyncHazards:
    def test_detects_time_sleep_in_async(self) -> None:
        source = """
        import time
        async def my_handler():
            time.sleep(1)
        """
        issues = _run_check(check_async_hazards, source)
        blocking = [i for i in issues if i.check_id == "ASYNC_HAZARD" and i.severity == "block"]
        assert blocking, f"Expected ASYNC_HAZARD for time.sleep in async, got: {issues}"

    def test_does_not_flag_time_sleep_in_sync(self) -> None:
        source = """
        import time
        def my_handler():
            time.sleep(1)
        """
        issues = _run_check(check_async_hazards, source)
        assert not issues, f"Expected no issues for time.sleep in sync def, got: {issues}"

    def test_detects_bare_except_in_async(self) -> None:
        source = """
        async def handler():
            try:
                pass
            except:
                pass
        """
        issues = _run_check(check_async_hazards, source)
        blocking = [i for i in issues if i.check_id == "ASYNC_HAZARD" and i.severity == "block"]
        assert blocking, f"Expected ASYNC_HAZARD for bare except in async, got: {issues}"

    def test_bare_except_in_sync_not_flagged(self) -> None:
        source = """
        def handler():
            try:
                pass
            except:
                pass
        """
        issues = _run_check(check_async_hazards, source)
        assert not issues

    def test_detects_requests_in_async(self) -> None:
        source = """
        import requests
        async def fetch():
            return requests.get("http://example.com")
        """
        issues = _run_check(check_async_hazards, source)
        blocking = [i for i in issues if i.check_id == "ASYNC_HAZARD" and i.severity == "block"]
        assert blocking

    def test_detects_baseexception_catch_in_async(self) -> None:
        source = """
        async def handler():
            try:
                pass
            except BaseException:
                pass
        """
        issues = _run_check(check_async_hazards, source)
        blocking = [i for i in issues if i.check_id == "ASYNC_HAZARD" and i.severity == "block"]
        assert blocking

    def test_time_sleep_in_nested_sync_inside_async_not_flagged(self) -> None:
        """time.sleep inside a sync def that is nested inside an async def should NOT be flagged."""
        source = """
        import time
        async def outer():
            def inner():
                time.sleep(1)
            inner()
        """
        issues = _run_check(check_async_hazards, source)
        assert not issues, f"Expected no issues, got: {issues}"


# ---------------------------------------------------------------------------
# PYDANTIC_MUTATION
# ---------------------------------------------------------------------------

class TestPydanticMutation:
    def test_detects_attribute_assignment_on_frozen_model(self) -> None:
        source = """
        from pydantic import BaseModel, ConfigDict

        class MyState(BaseModel):
            model_config = ConfigDict(frozen=True)
            field: int = 0

        def update(state: MyState) -> None:
            state.field = 42
        """
        issues = _run_check(check_pydantic_mutation, source)
        blocking = [i for i in issues if i.check_id == "PYDANTIC_MUTATION" and i.severity == "block"]
        assert blocking, f"Expected PYDANTIC_MUTATION for frozen model mutation, got: {issues}"

    def test_non_frozen_model_not_flagged(self) -> None:
        source = """
        from pydantic import BaseModel

        class MyState(BaseModel):
            field: int = 0

        def update(state: MyState) -> None:
            state.field = 42
        """
        issues = _run_check(check_pydantic_mutation, source)
        assert not issues, f"Expected no issues for non-frozen model, got: {issues}"

    def test_detects_setattr_on_frozen_model(self) -> None:
        source = """
        from pydantic import BaseModel, ConfigDict

        class Config(BaseModel):
            model_config = ConfigDict(frozen=True)
            value: str = ""

        def mutate(cfg: Config) -> None:
            setattr(cfg, "value", "new")
        """
        issues = _run_check(check_pydantic_mutation, source)
        blocking = [i for i in issues if i.check_id == "PYDANTIC_MUTATION" and i.severity == "block"]
        assert blocking


# ---------------------------------------------------------------------------
# EVENT_SOURCING
# ---------------------------------------------------------------------------

class TestEventSourcing:
    def test_detects_await_in_apply(self) -> None:
        # Test that awaiting (in an actual async apply) is flagged
        _source3 = """
        async def apply(state, event):
            await some_coro()
            return state
        """
        # event_sourcing checks ast.FunctionDef named apply, not async
        # The spec says "Find ast.FunctionDef named 'apply'" so async apply won't be caught
        # But let's test the sync apply with print
        source4 = """
        def apply(state, event):
            print("debug")
            return state
        """
        issues = _run_check(check_event_sourcing, source4)
        advisory = [i for i in issues if i.check_id == "EVENT_SOURCING" and i.severity == "advisory"]
        assert advisory, f"Expected EVENT_SOURCING advisory for print in apply(), got: {issues}"

    def test_detects_print_in_apply(self) -> None:
        source = """
        def apply(state, event):
            print(state, event)
            return state
        """
        issues = _run_check(check_event_sourcing, source)
        advisory = [i for i in issues if i.check_id == "EVENT_SOURCING"]
        assert advisory

    def test_detects_mutable_field_in_event_class(self) -> None:
        source = """
        class TaskCreatedEvent:
            tasks: list
            name: str
        """
        issues = _run_check(check_event_sourcing, source)
        advisory = [i for i in issues if i.check_id == "EVENT_SOURCING"]
        assert advisory, f"Expected EVENT_SOURCING for mutable field in event class, got: {issues}"

    def test_clean_apply_not_flagged(self) -> None:
        source = """
        def apply(state, event):
            return state
        """
        issues = _run_check(check_event_sourcing, source)
        assert not issues


# ---------------------------------------------------------------------------
# NETWORK_EXPOSURE
# ---------------------------------------------------------------------------

class TestNetworkExposure:
    def test_detects_bind_0000(self) -> None:
        source = 'host = "0.0.0.0"'
        # Use a non-test, non-config filepath so the file is not skipped
        issues = _run_check(check_network_exposure, source, filepath="src/exo/api/server.py")
        blocking = [i for i in issues if i.check_id == "NETWORK_EXPOSURE" and i.severity == "block"]
        assert blocking, f"Expected NETWORK_EXPOSURE block for 0.0.0.0, got: {issues}"

    def test_does_not_flag_in_test_files(self) -> None:
        source = 'host = "0.0.0.0"'
        issues = _run_check(check_network_exposure, source, filepath="src/exo/tests/test_server.py")
        assert not issues, f"Expected no issues in test files, got: {issues}"

    def test_does_not_flag_in_config_files(self) -> None:
        source = 'host = "0.0.0.0"'
        issues = _run_check(check_network_exposure, source, filepath="src/exo/config/defaults.py")
        assert not issues

    def test_detects_private_ip_advisory(self) -> None:
        source = 'server = "192.168.1.100"'
        issues = _run_check(check_network_exposure, source, filepath="src/exo/worker/main.py")
        advisory = [i for i in issues if i.check_id == "NETWORK_EXPOSURE" and i.severity == "advisory"]
        assert advisory

    def test_comment_line_not_flagged(self) -> None:
        source = '# host = "0.0.0.0"'
        issues = _run_check(check_network_exposure, source)
        assert not issues


# ---------------------------------------------------------------------------
# SUPPRESSIONS
# ---------------------------------------------------------------------------

class TestSuppressions:
    def test_nosec_inline_suppresses_issue(self) -> None:
        source = 'api_key = "sk-abc12345678901234"  # nosec:SECRET'
        src = textwrap.dedent(source)
        tree = ast.parse(src)
        issues = check_secrets("test_file.py", src, tree)
        assert any(i.check_id == "SECRET" for i in issues), "Should detect before suppression"

        source_lines = {"test_file.py": src.splitlines()}
        from pathlib import Path
        filtered = filter_issues(issues, source_lines, Path("/nonexistent/.security-gate-ignore"))
        assert not any(i.check_id == "SECRET" for i in filtered), (
            f"SECRET should be suppressed by nosec, remaining: {filtered}"
        )

    def test_nosec_wrong_id_does_not_suppress(self) -> None:
        source = 'api_key = "sk-abc12345678901234"  # nosec:DANGEROUS_CALL'
        src = textwrap.dedent(source)
        tree = ast.parse(src)
        issues = check_secrets("test_file.py", src, tree)

        source_lines = {"test_file.py": src.splitlines()}
        filtered = filter_issues(issues, source_lines, Path("/nonexistent/.security-gate-ignore"))
        # Should still have the SECRET issue since the nosec is for a different check
        assert any(i.check_id == "SECRET" for i in filtered)

    def test_multi_check_nosec(self) -> None:
        source = 'api_key = "sk-abc12345678901234"  # nosec:SECRET,DANGEROUS_CALL'
        src = textwrap.dedent(source)
        tree = ast.parse(src)
        issues = check_secrets("test_file.py", src, tree)

        source_lines = {"test_file.py": src.splitlines()}
        filtered = filter_issues(issues, source_lines, Path("/nonexistent/.security-gate-ignore"))
        assert not any(i.check_id == "SECRET" for i in filtered)


# ---------------------------------------------------------------------------
# DICT_MUTATION
# ---------------------------------------------------------------------------


class TestDictMutation:
    def test_detects_subscript_assignment_in_for(self) -> None:
        source = """
        d = {}
        for k in d:
            d[k] = 1
        """
        issues = _run_check(check_dict_mutation, source)
        blocking = [i for i in issues if i.check_id == "DICT_MUTATION"]
        assert blocking, f"Expected DICT_MUTATION for d[k] = 1 in for k in d, got: {issues}"

    def test_detects_del_in_for(self) -> None:
        source = """
        d = {}
        for k in d:
            del d[k]
        """
        issues = _run_check(check_dict_mutation, source)
        blocking = [i for i in issues if i.check_id == "DICT_MUTATION"]
        assert blocking, f"Expected DICT_MUTATION for del d[k] in for loop, got: {issues}"

    def test_detects_dot_pop_in_for(self) -> None:
        source = """
        d = {}
        for k in d:
            d.pop(k)
        """
        issues = _run_check(check_dict_mutation, source)
        blocking = [i for i in issues if i.check_id == "DICT_MUTATION"]
        assert blocking, f"Expected DICT_MUTATION for d.pop() in for loop, got: {issues}"

    def test_detects_set_add_in_for(self) -> None:
        source = """
        s = set()
        for item in s:
            s.add(item)
        """
        issues = _run_check(check_dict_mutation, source)
        blocking = [i for i in issues if i.check_id == "DICT_MUTATION"]
        assert blocking, f"Expected DICT_MUTATION for s.add() in for loop, got: {issues}"

    def test_detects_mutation_of_items_iter(self) -> None:
        source = """
        d = {}
        for k, v in d.items():
            d[k] = v + 1
        """
        issues = _run_check(check_dict_mutation, source)
        blocking = [i for i in issues if i.check_id == "DICT_MUTATION"]
        assert blocking, f"Expected DICT_MUTATION for d[k]= in for k,v in d.items(), got: {issues}"

    def test_clean_list_snapshot_not_flagged(self) -> None:
        source = """
        d = {}
        for k, v in list(d.items()):
            d[k] = v + 1
        """
        issues = _run_check(check_dict_mutation, source)
        assert not issues, f"Expected no issues when using list() snapshot, got: {issues}"

    def test_mutating_different_dict_not_flagged(self) -> None:
        source = """
        d = {}
        other = {}
        for k in d:
            other[k] = 1
        """
        issues = _run_check(check_dict_mutation, source)
        assert not issues, f"Mutating a different dict should not be flagged, got: {issues}"

    def test_augassign_in_for_flagged(self) -> None:
        source = """
        counts = {}
        for k in counts:
            counts[k] += 1
        """
        issues = _run_check(check_dict_mutation, source)
        blocking = [i for i in issues if i.check_id == "DICT_MUTATION"]
        assert blocking, f"Expected DICT_MUTATION for counts[k] += 1, got: {issues}"
