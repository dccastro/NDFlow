from setuptools import find_packages, setup


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name='ndflow',
    version='0.1.0',
    description='Nonparametric density flows for MRI intensity normalisation',
    long_description=readme(),
    author='Daniel Coelho de Castro',
    author_email='dc315@imperial.ac.uk',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'ndflow-estimate=ndflow.tools.estimate:main',
            'ndflow-average=ndflow.tools.average:main',
            'ndflow-match=ndflow.tools.match:main',
            'ndflow-warp=ndflow.tools.warp:main'
        ]
    },
    install_requires=[
        'numpy',
        'scipy',
        'SimpleITK'
    ]
)
