import os
from setuptools import setup, find_packages

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

README = open(os.path.join(SCRIPT_DIR, "README.md")).read()

setup_dict = dict(
    name="jrfl",
    version="0.1.0",
    description="Port of TRFL to JAX",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/cshesse/jrfl",
    author="Christopher Hesse",
    license="Apache License 2.0",
    packages=find_packages(),
    install_requires=["jax~=0.1.38"],
    extras_require={"dev": ["pytest", "tensorflow", "numpy", "tensorflow-probability"]},
)

if os.environ.get("USE_SCM_VERSION", "1") == "1":
    setup_dict["use_scm_version"] = {
        "root": "..",
        "relative_to": __file__,
        "local_scheme": "node-and-timestamp",
    }
    setup_dict["setup_requires"] = ["setuptools_scm"]

setup(**setup_dict)
