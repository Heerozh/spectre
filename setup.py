from setuptools import setup

setup(
    name="spectre",
    author='Zhang Jianhao',
    author_email='heeroz@gmail.com',
    description='GPU-accelerated Parallel quantitative trading library',
    long_description=open('README.md', encoding='utf-8').read(),
    license='Apache 2.0',
    keywords='quantitative analysis backtesting parallel algorithmic trading',
    url='https://github.com/Heerozh/spectre',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',

    version="0.4",
    packages=['spectre', 'spectre.data', 'spectre.factors', 'spectre.parallel', 'spectre.trading',
              'spectre.plotting'],
)
