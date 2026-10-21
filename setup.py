from setuptools import setup, find_namespace_packages

setup(
    name="flash_attn_prologue",
    version="0.2.0",
    python_requires=">=3.10",
    packages=find_namespace_packages(include=["flash_attn_prologue", "flash_attn_prologue.*"]),
    install_requires=["torch>=2.8", "quack-kernels>=0.6.5,<0.7"],
    extras_require={"test": ["pytest>=8"]},
)
