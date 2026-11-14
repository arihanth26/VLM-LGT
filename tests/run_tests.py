# run_tests.py
#
# Runs the test_* functions of the given test modules without pytest, which the ICE environment does
# not include. Exits non-zero if any test fails.
# Usage: python -m tests.run_tests tests.test_escalation [tests.test_other ...]

import importlib
import sys
import traceback


def main() -> int:
    failures = 0
    for name in sys.argv[1:]:
        module = importlib.import_module(name)
        for attribute in sorted(dir(module)):
            if attribute.startswith("test_") and callable(getattr(module, attribute)):
                try:
                    getattr(module, attribute)()
                    print(f"ok    {name}.{attribute}")
                except Exception:
                    failures += 1
                    print(f"FAIL  {name}.{attribute}")
                    traceback.print_exc()
    print(f"{failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
