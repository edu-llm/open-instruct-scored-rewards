"""Learning-science semantic atlas.

Deliberately empty of imports. `trace`, `analyze_representations` and
`checkpoint_delta` pull in torch, and importing the package should not: the CPU
test suite, `schema`, `build_corpus`, `labeling`, `agreement` and `ppi` all run
in an environment with no accelerator and, in `schema`'s case, with no
third-party packages at all. Import the module you want.
"""
