from setuptools import setup, find_packages

setup(
    name="financial_physics",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'arch',
        'scipy',
        'statsmodels',
        'scikit-learn',
        'yfinance'
    ]
) 