from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent
readme_path = ROOT / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

install_requires = [
    "numpy",
]

extras_require = {
    # Keep torch optional to avoid forcing a wheel/CUDA stack change
    # in environments that already have a tuned torch + cuDNN setup.
    "torch": ["torch"],
    "profiling": ["pandas", "matplotlib"],
    "tensordict": ["tensordict"],
    "etcd": ["etcd3"],
    "nvml": ["pynvml"],
    "nvtx": ["nvtx"],
    "system": ["psutil"],
    "megatron": ["megatron-core"],
}
extras_require["all"] = sorted(
    {
        dep
        for key, deps in extras_require.items()
        if key != "torch"
        for dep in deps
    }
)
extras_require["all_with_torch"] = sorted(
    {
        dep
        for deps in extras_require.values()
        for dep in deps
    }
)

setup(
    name="my_utils",
    version="0.1",
    packages=find_packages(),
    package_dir={"": "."},
    include_package_data=True,
    package_data={
        "my_utils": [
            "profiling/templates/*.sh",
            "profiling/templates/*.env",
            "profiling/templates/*.md",
        ]
    },
    entry_points={
        "console_scripts": [
            "myutils-profile=my_utils.profiling.cli:main",
            "nsys-sql-skill=my_utils.profiling.cli:entry_nsys_sql_skill",
            "nsys-export=my_utils.profiling.cli:entry_nsys_export",
            "nsys-analyze=my_utils.profiling.cli:entry_nsys_analyze",
            "nsys-diff=my_utils.profiling.cli:entry_nsys_diff",
            "nsys-timeline-html=my_utils.profiling.cli:entry_nsys_timeline_html",
            "nsys-iter-overlap=my_utils.profiling.cli:entry_nsys_iter_overlap",
            "nsys-iter-outliers=my_utils.profiling.cli:entry_nsys_iter_outliers",
        ]
    },
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
