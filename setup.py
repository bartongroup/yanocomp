from setuptools import setup


setup(
    name='yanocomp',
    version='0.1',
    description=(
        'GMM tests for analysis of differential RNA modifications'
    ),
    author='Matthew Parker',
    entry_points={
        'console_scripts': [
            'yanocomp = yanocomp.cli:cli',
        ]
    },
    packages=[
        'yanocomp',
    ],
    install_requires=[
        'numpy',
        'scipy',
        'pandas>=1.1',
        'statsmodels',
        'pomegranate>=0.13',
        'scikit-learn',
        'h5py>=2.10',
        'click',
        'click_log',
    ],
)
