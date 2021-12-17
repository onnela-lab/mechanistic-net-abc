import setuptools

with open("DESCRIPTION.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mechanistic_net_abc",
    version="0.1",
    author="Louis Raynal, Jukka-Pekka Onnela",
    author_email="llcraynal@hsph.harvard.edu",
    description="A package containing implementations of mechanistic models to understand network formation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/onnela-lab/mechanistic-net-abc",
    packages=setuptools.find_packages(include=['mechanistic_net_abc',
                                               'mechanistic_net_abc.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-3-Clause License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8.10',
    install_requires=[
            'joblib>=1.0.1',
            'matplotlib>=3.3.4',
            'networkx>=2.6.1',
            'numpy>=1.20.2',
            'pandas==1.3.2',
            'scipy>=1.6.2',
            'seaborn>=0.11.1',
            'sklearn'
    ],
    package_data={'': ['data/1-PA_one_noise/ref_table/*.csv',
                  'data/1-PA_one_noise/SMC_unselected/*.csv',
                  'data/2-PA_two_noises/ref_table/*.csv',
                  'data/2-PA_two_noises/SMC_unselected/*.csv',
                  'data/2-PA_two_noises/SMC_naive/*.csv',
                  'data/2-PA_two_noises/SMC_recursive/*.csv',
                  'data/2-PA_two_noises/rankings/*.p',
                  'data/3-PA_RA_TF_mixture/ref_table/*.csv',
                  'data/3-PA_RA_TF_mixture/SMC_unselected/*.csv',
                  'data/3-PA_RA_TF_mixture/SMC_naive/*.csv',
                  'data/3-PA_RA_TF_mixture/SMC_recursive/*.csv',
                  'data/3-PA_RA_TF_mixture/rankings/*.p',
                  'data/4-household_analysis/ref_table/*.csv',
                  'data/4-household_analysis/SMC_unselected/*.csv',
                  'data/4-household_analysis/SMC_naive/*.csv',
                  'data/4-household_analysis/SMC_recursive/*.csv',
                  'data/4-household_analysis/SMC_recursive/*.p',
                  'data/4-household_analysis/rankings/*.p',
                  'data/4-household_analysis/household_data/*.txt']}
)