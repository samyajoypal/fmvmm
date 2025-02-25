from setuptools import setup, find_packages

# Read long description from README.md if available
with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

project_url = {
    "Github": "https://github.com/samyajoypal/fmvmm",
}

setup(
    name="fmvmm",
    version="2.0.2",
    author="Samyajoy Pal",
    author_email="palsamyajoy@gmail.com",
    description="Flexible Multivariate Mixture Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/samyajoypal/fmvmm",
    project_urls=project_url,
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "conorm~=1.2.0",
        "matplotlib~=3.7",
        "numpy>=1.24, <1.26",
        "pandas>=2.0, <2.2",
        "scikit-learn>=1.6, <1.8",
        "scipy>=1.9, <1.11",
        "setuptools",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "my_command = fmvmm.command:main",
        ],
    },
    include_package_data=True,
)
