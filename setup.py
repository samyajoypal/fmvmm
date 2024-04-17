from setuptools import setup, find_packages

project_url = {
  'Github': 'https://github.com/samyajoypal/fmvmm'
}

setup(
    name='fmvmm',
    version='1.0.13',
    author='Samyajoy Pal',
    author_email='palsamyajoy@gmail.com',
    description="flexible multivariate mixture model",
    long_description="mixture model with different distributions",
    project_urls = project_url,
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'dirichlet',
        'sklearn',
        'conorm'
    ],
    entry_points={
        'console_scripts': [
            'my_command = fmvmm.command:main',
        ],
    },
)
