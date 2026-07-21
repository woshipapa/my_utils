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
    version="0.1.0",
    packages=find_packages(),
    package_dir={"": "."},
    include_package_data=True,
    package_data={
        "my_utils": [
            "profiling/templates/*.sh",
            "profiling/templates/*.env",
            "profiling/templates/*.md",
            "profiling/templates/*.yaml",
            "profiling/templates/*.py",
            "profiling/ncu/*.md",
            "profiling/ncu/*.yaml",
            "profiling/ncu/*.sh",
            "profiling/nccl/*.md",
            "profiling/nccl/*.yaml",
            "profiling/nccl/*.sh",
        ]
    },
    entry_points={
        "console_scripts": [
            "myutils-profile=my_utils.profiling.cli:main",
            "nsys-panel=my_utils.profiling.cli:entry_nsys_panel",
            "nsys-sql-skill=my_utils.profiling.cli:entry_nsys_sql_skill",
            "nsys-export=my_utils.profiling.cli:entry_nsys_export",
            "nsys-analyze=my_utils.profiling.cli:entry_nsys_analyze",
            "nsys-diff=my_utils.profiling.cli:entry_nsys_diff",
            "nsys-module-kernel-compare=my_utils.profiling.cli:entry_nsys_module_kernel_compare",
            "nsys-timeline-html=my_utils.profiling.cli:entry_nsys_timeline_html",
            "nsys-iter-overlap=my_utils.profiling.cli:entry_nsys_iter_overlap",
            "nsys-iter-outliers=my_utils.profiling.cli:entry_nsys_iter_outliers",
            "ncu-csv-skill=my_utils.profiling.cli:entry_ncu_csv_skill",
            "ncu-csv-analyze=my_utils.profiling.cli:entry_ncu_csv_analyze",
            "ncu-report-skill=my_utils.profiling.cli:entry_ncu_report_skill",
            "ncu-report-analyze=my_utils.profiling.cli:entry_ncu_report_analyze",
            "nccl-inspector-skill=my_utils.profiling.cli:entry_nccl_inspector_skill",
            "nccl-inspector-analyze=my_utils.profiling.cli:entry_nccl_inspector_analyze",
        ]
    },
    install_requires=install_requires,
    extras_require=extras_require,
    author="papa",
    author_email="96102319+woshipapa@users.noreply.github.com",
    description="GPU profiling toolkit: nsys/ncu collection presets and an "
                "evidence-based kernel diagnosis engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/woshipapa/my_utils",
    python_requires=">=3.10",
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
