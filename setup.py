from setuptools import setup, find_packages

setup(
    name='fmvmm',
    version='1.0.4',
    author='Samyajoy Pal',
    author_email='palsamyajoy@gmail.com',
    description="flexible multivariate mixture model",
    long_description="mixture model with different distributions",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'math',
        'copy',
        'dirichlet',
        'sklearn'
    ],
    entry_points={
        'console_scripts': [
            'my_command = fmvmm.command:main',
        ],
    },
)
