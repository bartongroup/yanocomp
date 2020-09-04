from setuptools import setup


setup(
    name='diffmod',
    version='0.1',
    description=(
        'GMM tests for analysis of differential RNA modifications'
    ),
    author='Matthew Parker',
    entry_points={
        'console_scripts': [
            'diffmod = diffmod.cli:cli',
        ]
    },
    packages=[
        'diffmod',
    ],
    install_requires=[
        'numpy',
        'scipy',
        'pandas>=1.1',
        'numexpr',
        'statsmodels',
        'pomegranate>=0.13',
        'scikit-learn',
        'h5py>=2.10.0',
        'click',
        'click_log',
    ],
)