from setuptools import setup, find_packages

setup(
    name='mixture_models',
    version='1.0',
    author='Samyajoy Pal',
    author_email='palsamyajoy@gmail.com',
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
            'my_command = mixture_models.command:main',
        ],
    },
)