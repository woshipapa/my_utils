from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent
readme_path = ROOT / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

install_requires = [
    "torch",
    "numpy",
]

extras_require = {
    "profiling": ["pandas", "matplotlib"],
    "tensordict": ["tensordict"],
    "etcd": ["etcd3"],
    "nvml": ["pynvml"],
    "nvtx": ["nvtx"],
    "system": ["psutil"],
    "megatron": ["megatron-core"],
}
extras_require["all"] = sorted({dep for deps in extras_require.values() for dep in deps})

setup(
    name="my_utils",
    version="0.1",
    packages=find_packages(),
    package_dir={"": "."},
    install_requires=install_requires,
    extras_require=extras_require,
    author="Your Name",
    author_email="your_email@example.com",
    description="Profiling and debugging utilities for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/my_utils",
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
