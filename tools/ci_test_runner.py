if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Test suite runner.")
    parser.add_argument("--with-numba", action="store_true")

    args = parser.parse_args()

    import pytest
    import heyoka

    if args.with_numba:
        import numba

    # NOTE: run the bundled test suite against the *installed* package via pytest's
    # in-process entry point. In-process (rather than a pytest subprocess) is required
    # because the NumPy memory-handler variants below must be active in the same
    # process that runs the tests.
    pytest_args = ["--pyargs", "heyoka.test", "-v"]

    def run_test_suite():
        ret = pytest.main(pytest_args)
        if ret != 0:
            sys.exit(int(ret))

    run_test_suite()

    if hasattr(heyoka, "real"):
        heyoka.install_custom_numpy_mem_handler()
        run_test_suite()
        heyoka.remove_custom_numpy_mem_handler()
        run_test_suite()
