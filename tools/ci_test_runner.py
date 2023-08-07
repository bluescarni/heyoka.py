if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test suite runner.")
    parser.add_argument("--with-numba", action="store_true")

    args = parser.parse_args()

    import heyoka

    if args.with_numba:
        import numba

    heyoka.test.run_test_suite()

    if hasattr(heyoka, "real"):
        heyoka.install_custom_numpy_mem_handler()
        heyoka.test.run_test_suite()
        heyoka.remove_custom_numpy_mem_handler()
        heyoka.test.run_test_suite()
