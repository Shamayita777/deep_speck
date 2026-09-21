# Target environment: the DOCUMENTED Gohr reproduction baseline
# (archive/notebooks/gohr_reproduction_tf2.ipynb, cells 8-10):
#   Python 3.12.13 / TensorFlow 2.20.0 / Keras 3.13.2
#
# HONESTY NOTE (Round 6, Issue 7): this Dockerfile has NOT been built
# or executed in the sandbox used to implement and test this codebase
# (no Docker daemon was available there). It is provided as the
# specification of the target reconciliation environment, not as a
# verified-working artifact. Before this image is used for any
# confirmatory production run:
#   1. build it;
#   2. run `pip freeze > requirements.lock.txt` inside it and commit
#      that file as the authoritative, fully-resolved pin set (numpy/
#      scipy versions below are intentionally left to pip's own
#      dependency resolution against tensorflow==2.20.0's declared
#      requirements, rather than guessed at here without verification -
#      fabricating an exact numpy/scipy pin for a TensorFlow release
#      this project has not built against would be an unverified
#      compatibility claim);
#   3. run the FULL test suite (`python -m pytest tests/ -v`) inside
#      the container and confirm all tests pass;
#   4. record the outcome of step 3 in provenance before treating this
#      as the production environment.
#
# Until steps 1-4 are done, requirements.txt (pinned to the environment
# actually verified during development) remains the only environment
# with a confirmed, executed test-pass record for this codebase.

FROM python:3.12.13-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    tensorflow-cpu==2.20.0 \
    keras==3.13.2 \
    pyyaml==6.0.* \
    pytest==8.*

COPY . /app

# numpy and scipy are intentionally NOT pinned to an exact version here -
# see the honesty note above. Run `pip freeze` after build and commit
# the result as requirements.lock.txt before using this image for any
# confirmatory production run.

CMD ["python3", "-m", "pytest", "tests/", "-v"]
