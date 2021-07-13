from setuptools import setup


setup(
    name='yanocomp',
    version='0.2',
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
    include_package_data=True,
    package_data={
        'yanocomp': [
            'data/*.model',
        ]
    },
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
