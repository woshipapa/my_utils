from setuptools import setup, find_packages

setup(
    name="my_utils",
    version="0.1",
    packages=find_packages(),  # 默认查找同级目录的包
    package_dir={"": "."},  # 确保从当前目录查找
    install_requires=[],  
    author="Your Name",
    author_email="your_email@example.com",
    description="A simple utility package",
    url="https://github.com/yourusername/my_utils",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
